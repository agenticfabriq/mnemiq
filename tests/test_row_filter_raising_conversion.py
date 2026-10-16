"""M159: a row filter whose conversion can raise on a row's value is refused before it runs.

A row filter runs over every row of its table, the ones it removes included. `CAST(region AS
INTEGER) > 0` over a hidden row holding 'north' failed every question on the table, and the error
quoted the hidden value. The owner's decision (2026-10-15): reject raising conversions, type-aware
where the snapshot knows the column's type, the refusal naming the engine's non-raising form --
TRY_CAST in DuckDB, DEFAULT NULL ON CONVERSION ERROR in Oracle. Two more spellings of the class are
measured here and refused the same way: DuckDB's JSON readers (text read as JSON, quoted on failure),
and a comparison that converts a text column implicitly (`region = 7`).

Not covered, and not convergent by listing spellings: an implicit conversion of an EXPRESSION over a
text column (`UPPER(region) > 0`, `coalesce(region, '0') = 7`, `region + 0 > 0`), one the snapshot
cannot type (a column with no recorded type), and a raising expression that is not a conversion
(`error()`, and whatever else an engine raises on data). #154's redaction is what bounds those: the
model reads the error class only.
"""

from __future__ import annotations

import logging
import os

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

TYPES = {("claim", "id"): "INTEGER", ("claim", "region"): "VARCHAR", ("claim", "amount"): "INTEGER",
         ("claim", "big"): "BIGINT", ("claim", "attrs"): "JSON",
         ("entitlement", "id"): "INTEGER", ("entitlement", "who"): "VARCHAR"}


def _snapshot(types=TYPES, oracle=False):
    if oracle:
        types = {k: {"VARCHAR": "VARCHAR2(30)", "INTEGER": "NUMBER", "BIGINT": "NUMBER",
                     "JSON": "VARCHAR2(4000)"}[v] for k, v in types.items()}
    return Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c, data_type=ty)
                             for (t, c), ty in types.items()])


def _decide(row_filter, engine="duckdb", snapshot=None):
    snapshot = snapshot or _snapshot(oracle=engine == "oracle")
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": row_filter})
    return decide("SELECT id FROM claim", visible_schema(snapshot, grants), dialect=engine,
                  target=engine, policy=build_access_policy(snapshot, grants))


# --- the premise, measured ------------------------------------------------------------------------

def test_a_raising_filter_reads_the_rows_it_removes_and_quotes_them_on_duckdb():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); "
                "INSERT INTO claim VALUES (1, '7'), (2, 'north')")
    for raising, safe in (("CAST(region AS INTEGER) > 0", "TRY_CAST(region AS INTEGER) > 0"),
                          ("region = 7", "TRY_CAST(region AS INTEGER) = 7"),
                          ("json_extract_string(region, '$.a') IS NULL",
                           "json_extract_string(TRY_CAST(region AS JSON), '$.a') IS NULL")):
        with pytest.raises(duckdb.Error, match="north"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
        assert con.execute(f"SELECT id FROM claim WHERE {safe}").fetchall()  # runs, NULL for 'north'


# Each JSON reader's arguments where it takes more than the text; the rest take the text alone.
_JSON_ARGS = {"json_extract": "region, '$.a'", "json_extract_string": "region, '$.a'",
              "json_extract_path": "region, '$.a'", "json_extract_path_text": "region, '$.a'",
              "json_value": "region, '$.a'", "json_exists": "region, '$.a'",
              "json_contains": "region, '1'", "json_merge_patch": "region, '{}'",
              "from_json": "region, '\"INTEGER\"'",
              "from_json_strict": "region, '\"INTEGER\"'", "json_transform": "region, '\"INTEGER\"'",
              "json_transform_strict": "region, '\"INTEGER\"'"}


def test_every_duckdb_json_reader_that_quotes_malformed_text_is_refused():
    """Re-measured from `duckdb_functions()` each run, so a reader DuckDB adds fails here."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); "
                "INSERT INTO claim VALUES (1, '{\"a\": 1}'), (2, 'north')")
    names = [r[0] for r in con.execute(
        "SELECT DISTINCT function_name FROM duckdb_functions() "
        "WHERE function_type IN ('scalar', 'macro') "  # `json` itself is a macro
        "AND (function_name ILIKE 'json%' OR function_name ILIKE 'from_json%') ORDER BY 1").fetchall()]
    quoting = []
    for name in names:
        call = f"{name}({_JSON_ARGS.get(name, 'region')})"
        try:
            con.execute(f"SELECT id FROM claim WHERE ({call}) IS NOT NULL").fetchall()
        except duckdb.Error as e:
            if "north" in str(e):
                quoting.append(call)
    assert len(quoting) >= 19, quoting  # control: 19 in 1.5.4 at these arguments
    admitted = [c for c in quoting if not isinstance(_decide(f"({c}) IS NOT NULL"), Refusal)]
    assert admitted == [], admitted


# --- refused --------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine,row_filter", [
    ("duckdb", "CAST(region AS INTEGER) > 0"),
    ("duckdb", "region::INTEGER > 0"),
    ("duckdb", "CAST(UPPER(region) AS INTEGER) > 0"),     # an expression's type is unknown
    ("duckdb", "CAST(region || '' AS INTEGER) > 0"),      # Codex's second #153 reproduction
    ("duckdb", "CAST(big AS INTEGER) > 0"),               # narrowing can overflow
    ("duckdb", "strptime(region, '%Y') > TIMESTAMP '2000-01-01'"),
    ("duckdb", "json_extract_string(region, '$.a') = '1'"),
    ("duckdb", "region->>'$.a' = '1'"),
    ("duckdb", "from_json(region, '\"INTEGER\"') > 0"),
    ("duckdb", "id IN (SELECT id FROM entitlement WHERE CAST(who AS INTEGER) > 0)"),
    ("duckdb", "json(region) IS NOT NULL"),
    ("duckdb", "json_value(region, '$.a') = '1'"),
    ("duckdb", "json_contains('[1]', region)"),         # its second argument is parsed too
    ("duckdb", "json_merge_patch('{}', region) IS NOT NULL"),
    ("duckdb", "region = 7"),                             # converted implicitly, row by row
    ("duckdb", "region = (7)"),
    ("duckdb", "region = TRUE"),
    ("oracle", "region = DATE '2000-01-01'"),             # an implicit TO_DATE
    ("duckdb", "region IN (1, 2)"),
    ("duckdb", "id = region"),
    ("duckdb", "region IN (SELECT id FROM entitlement)"),
    ("oracle", "region > 0"),                             # an implicit TO_NUMBER
    ("oracle", "CAST(region AS NUMBER) > 0"),
    ("oracle", "TO_NUMBER(region) > 0"),
    ("oracle", "TO_DATE(region, 'YYYY') > DATE '2000-01-01'"),
    ("oracle", "TO_BINARY_DOUBLE(region) > 0"),
    # sqlglot reads this as TO_DATE and drops the DEFAULT clause, so it would RUN as raising:
    ("oracle", "CAST(region AS DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYY') > DATE '2000-01-01'"),
])
def test_a_conversion_that_can_raise_is_refused_naming_the_engine_s_safe_form(engine, row_filter,
                                                                               caplog):
    with caplog.at_level(logging.WARNING):
        verdict = _decide(row_filter, engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    assert not verdict.repairable
    safe = {"duckdb": "TRY_CAST(value AS type)",
            "oracle": "DEFAULT NULL ON CONVERSION ERROR"}[engine]
    assert safe in verdict.message
    assert row_filter.split(" ")[0] not in verdict.message, "the policy's text is the operator's"
    assert "(M159)" in caplog.text, "the operator's log names the conversion"


def test_a_column_the_snapshot_does_not_type_is_unknown_and_refused():
    untyped = {k: None for k in TYPES}
    snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                        columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in untyped])
    assert isinstance(_decide("CAST(id AS BIGINT) > 0", snapshot=snapshot), Refusal)
    assert isinstance(_decide("CAST(id AS BIGINT) > 0"), Approved), "control: typed INTEGER"


# --- kept -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine,row_filter", [
    ("duckdb", "TRY_CAST(region AS INTEGER) > 0"),
    ("duckdb", "CAST(id AS BIGINT) > 0"),                 # widening
    ("duckdb", "CAST(id AS INTEGER) > 0"),                # the same type
    ("duckdb", "CAST(amount AS VARCHAR) = '10'"),         # every integer renders as text
    ("duckdb", "id = CAST('7' AS INTEGER)"),              # a constant cannot fail on a row
    ("duckdb", "json_extract_string(attrs, '$.a') = '1'"),  # typed JSON already
    ("duckdb", "json_extract_string(TRY_CAST(region AS JSON), '$.a') = '1'"),
    ("duckdb", "id IN (SELECT id FROM entitlement WHERE who = 'r')"),
    ("duckdb", "region = '7'"),
    ("duckdb", "id IN (1, 2)"),
    ("duckdb", "amount = id"),
    ("oracle", "CAST(region AS NUMBER DEFAULT NULL ON CONVERSION ERROR) > 0"),
    ("oracle", "TO_NUMBER(region DEFAULT NULL ON CONVERSION ERROR) > 0"),
    ("oracle", "CAST(region AS DATE DEFAULT NULL ON CONVERSION ERROR) > DATE '2000-01-01'"),
])
def test_a_conversion_that_cannot_raise_is_kept(engine, row_filter):
    assert isinstance(_decide(row_filter, engine), Approved), row_filter


def test_a_write_under_a_raising_filter_is_refused_too():
    snapshot = _snapshot()
    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}),
                      row_filters={"claim": "CAST(region AS INTEGER) > 0"})
    verdict = decide_write("DELETE FROM claim WHERE id = 1", visible_schema(snapshot, grants),
                           grants, policy=build_access_policy(snapshot, grants), dialect="duckdb",
                           target="duckdb", writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_through_the_agent_the_hidden_value_reaches_no_one(tmp_path):
    """The finding's own door: 3 model calls, and the repair prompt carried the hidden value."""
    from mnemiq.adapters.duckdb import DuckDBAdapter
    from mnemiq.agent.loop import Agent
    from mnemiq.agent.synthesize import FakeSynthesizer
    from mnemiq.cache.store import L1Cache, TwoTierCache
    from mnemiq.contract import DeferralReason, IdentityContext
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, amount INT, big BIGINT, attrs JSON);"
                "CREATE TABLE entitlement(id INT, who VARCHAR);"
                "INSERT INTO claim VALUES (1, '7', 10, 1, NULL), (2, 'secret_hidden_value', 20, 2, NULL)")
    con.close()
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "CAST(region AS INTEGER) > 0"})
    generator = FakeGenerator(['{"sql": "SELECT id FROM claim", "reason": "ok"}'] * 3)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"),
                  adapter=DuckDBAdapter.duckdb(path), cache=TwoTierCache(L1Cache()))
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    answer = agent.answer(packet, _snapshot(), grants,
                          IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    assert len(generator.calls) == 1, generator.calls
    assert answer.deferred and answer.reason_code is DeferralReason.UNGOVERNABLE, answer
    assert "secret_hidden_value" not in answer.answer


# --- Oracle, live: the advised forms run over a bad hidden row ------------------------------------

DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_the_advised_forms_run_where_the_raising_one_fails():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    # Every visible column, since the policy's derived table projects them all.
    data = ("(SELECT 1 AS id, '7' AS region, 10 AS amount, 1 AS big, NULL AS attrs FROM dual "
            "UNION ALL SELECT 2, 'north', 20, 2, NULL FROM dual) claim")
    try:
        for raising in ("CAST(region AS NUMBER) > 0", "region > 0"):
            with pytest.raises(oracledb.DatabaseError, match="ORA-01722"):
                cur.execute(f"SELECT id FROM {data} WHERE {raising}").fetchall()
            assert isinstance(_decide(raising, "oracle"), Refusal), raising
        for safe in ("CAST(region AS NUMBER DEFAULT NULL ON CONVERSION ERROR) > 0",
                     "TO_NUMBER(region DEFAULT NULL ON CONVERSION ERROR) > 0"):
            verdict = _decide(safe, "oracle")
            assert isinstance(verdict, Approved), (safe, verdict)
            sql = verdict.target_sql.replace("FROM claim", f"FROM {data}")
            assert cur.execute(sql).fetchall() == [(1,)], sql
    finally:
        con.close()

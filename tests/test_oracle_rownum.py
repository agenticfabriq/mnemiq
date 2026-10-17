"""M163: on Oracle, ROWNUM is refused with what to write instead, not as a column the table lacks.

`SELECT id FROM (SELECT id FROM claim ORDER BY amount DESC) WHERE ROWNUM <= 5` was refused
`unknown_column` with the generic sentence. The register's first answer was to admit ROWNUM as the
pseudo-column it is. The ui lane measured why not, on Oracle 23: admitted, the same pseudo-column
runs and answers wrong in the shapes models write most. `WHERE ROWNUM <= 5 ORDER BY amount DESC`
sorts five arbitrary rows, `ROWNUM > 5` is always empty, and a selected ROWNUM numbers rows in
storage order, not the query's order. The owner's decision (2026-10-16): keep it refused, and say
to write ORDER BY with NULLS LAST on the first sort key, then FETCH FIRST n ROWS ONLY: the bound
#156 keeps and the row cap reads, with NULLS LAST where M161's lint reads it, the first key, so it
does not cost a second repair (the ui lane's review).

ROWID is refused with its own reason: a physical row address is not data an answer reports, and
Oracle cannot read it through the view a row policy puts over a table (ORA-01446, measured below).
"""

from __future__ import annotations

import os

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id", "amount"}}
FILTERED = AccessPolicy(row_filters={"claim": "id <= 10"}, policy_schema={"claim": {"id", "amount"}})
ADVISED = "SELECT id FROM claim ORDER BY amount DESC NULLS LAST FETCH FIRST 5 ROWS ONLY"


def _decide(sql, engine="oracle", policy=None):
    return decide(sql, VISIBLE, dialect=engine, target=engine, policy=policy)


@pytest.mark.parametrize("sql", [
    "SELECT id FROM (SELECT id FROM claim ORDER BY amount DESC) WHERE ROWNUM <= 5",
    "SELECT id FROM claim WHERE ROWNUM <= 5 ORDER BY amount DESC",   # five arbitrary rows, sorted
    "SELECT id FROM claim WHERE ROWNUM > 5",                          # always empty
    "SELECT id, ROWNUM AS r FROM claim ORDER BY amount DESC",         # numbers in storage order
    "SELECT claim.rownum FROM claim",
])
@pytest.mark.parametrize("policy", [None, FILTERED])
def test_rownum_is_refused_with_the_bound_to_write_instead(sql, policy):
    verdict = _decide(sql, policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable
    assert "NULLS LAST on the first sort key, then FETCH FIRST n ROWS ONLY" in verdict.message


@pytest.mark.parametrize("policy", [None, FILTERED])
def test_the_advised_bound_is_approved_as_written(policy):
    verdict = _decide(ADVISED, policy=policy)
    assert isinstance(verdict, Approved), verdict
    assert verdict.target_sql.endswith("FETCH FIRST 5 ROWS ONLY") and verdict.row_cap is None
    # Why the advice says NULLS LAST, and on the first key: M161's lint reads the first key only,
    # so NULLS LAST missing, or written on a later key, costs a second repair.
    for second_repair in (ADVISED.replace(" NULLS LAST", ""),
                          "SELECT id FROM claim ORDER BY amount DESC, id NULLS LAST "
                          "FETCH FIRST 5 ROWS ONLY"):
        bare = _decide(second_repair, policy=policy)
        assert isinstance(bare, Refusal) and bare.code is RefusalCode.LOGIC_LINT, second_repair
    two_keys = _decide("SELECT id FROM claim ORDER BY amount DESC NULLS LAST, id "
                       "FETCH FIRST 5 ROWS ONLY", policy=policy)
    assert isinstance(two_keys, Approved), two_keys


def test_a_write_bounded_by_rownum_is_refused_too():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write

    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    verdict = decide_write("DELETE FROM claim WHERE ROWNUM <= 5", VISIBLE, grants,
                           policy=AccessPolicy(), dialect="oracle", target="oracle",
                           writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


def test_elsewhere_rownum_is_an_unknown_column_like_any_other():
    verdict = _decide("SELECT id FROM claim WHERE ROWNUM <= 5", "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN
    assert "FETCH FIRST" not in verdict.message


@pytest.mark.parametrize("sql", ["SELECT id, ROWID FROM claim", "SELECT c.ROWID FROM claim c"])
def test_rowid_is_refused_with_its_own_reason(sql):
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable and "physical address" in verdict.message


# Codex's review of #161: the column checks look nothing up for a select alias or for a column
# qualified by a CTE or subquery, and Oracle reads the pseudo-column in both.
EXEMPTED = [
    ("SELECT id AS rownum FROM claim WHERE rownum <= 5 ORDER BY amount DESC NULLS LAST",
     "FETCH FIRST n ROWS ONLY"),
    ("SELECT s.ROWNUM FROM (SELECT id FROM claim) s", "FETCH FIRST n ROWS ONLY"),
    ("WITH s AS (SELECT id FROM claim) SELECT s.ROWNUM FROM s", "FETCH FIRST n ROWS ONLY"),
    ("SELECT id AS rowid FROM claim WHERE rowid IS NOT NULL", "physical address"),
    ("SELECT s.ROWID FROM (SELECT id FROM claim) s", "physical address"),
    ("WITH s AS (SELECT id FROM claim) SELECT s.ROWID FROM s", "physical address"),
]


@pytest.mark.parametrize("sql, says", EXEMPTED)
@pytest.mark.parametrize("policy", [None, FILTERED])
def test_a_pseudo_column_is_refused_where_no_column_is_looked_up(sql, says, policy):
    verdict = _decide(sql, policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert says in verdict.message


def test_a_write_reading_rowid_through_a_subquery_is_refused():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write

    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    verdict = decide_write("DELETE FROM claim WHERE id IN (SELECT s.rowid FROM "
                           "(SELECT id FROM claim) s)", VISIBLE, grants, policy=AccessPolicy(),
                           dialect="oracle", target="oracle", writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert "physical address" in verdict.message


@pytest.mark.parametrize("sql", [
    'SELECT "ROWID" FROM claim',
    'SELECT s."ROWID" FROM (SELECT id FROM claim) s',
    'SELECT id AS "ROWID" FROM claim WHERE "ROWID" IS NOT NULL',
])
def test_quoted_in_upper_case_rowid_is_still_the_pseudo_column(sql):
    # Oracle reads `"ROWID"` as ROWID (measured below), so quoting it is no way past.
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert "physical address" in verdict.message


@pytest.mark.parametrize("name", ["ROWNUM", "rowid"])
def test_quoted_otherwise_it_is_a_column_name_like_any_other(name):
    # `"ROWNUM"` and `"rowid"` are ordinary names on Oracle, and a table may have either.
    quoted = f'SELECT "{name}" FROM claim'
    lacking = _decide(quoted)
    assert isinstance(lacking, Refusal) and lacking.code is RefusalCode.UNKNOWN_COLUMN, lacking
    assert f"{name!r}" in lacking.message and "FETCH FIRST" not in lacking.message
    assert "physical address" not in lacking.message
    having = decide(quoted, {"claim": {"id", name}}, dialect="oracle", target="oracle")
    assert isinstance(having, Approved), having


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_rownum_answers_wrong_and_the_advised_bound_right():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    # Amounts out of id order, so the first rows stored are not the top rows by amount.
    data = "(SELECT level AS id, MOD(level * 7, 20) AS amount FROM dual CONNECT BY level <= 20) claim"
    top_5 = [(17,), (14,), (11,), (8,), (5,)]

    def run(sql):
        return cur.execute(sql.replace("FROM claim", f"FROM {data}")).fetchall()

    try:
        # The premise: what admitting ROWNUM let run.
        assert run("SELECT id FROM claim WHERE ROWNUM <= 5 ORDER BY amount DESC") != top_5
        assert run("SELECT id FROM claim WHERE ROWNUM > 5") == []
        assert run(_decide(ADVISED).target_sql) == top_5
        caller_s = run(_decide(ADVISED, policy=FILTERED).target_sql)
        assert len(caller_s) == 5 and all(i <= 10 for (i,) in caller_s), caller_s
        fenced = _decide("SELECT id FROM claim", policy=FILTERED).target_sql
        with pytest.raises(oracledb.DatabaseError, match="ORA-01446"):
            run(fenced.replace("SELECT id FROM", "SELECT id, ROWID FROM", 1))
    finally:
        con.close()



@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_rowid_reads_through_a_subquery_and_quoted_too():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    # A stored table: ROWID is an address in one, and an inline view over dual has none.
    table, quoted_probe = "m163_rowid_probe", "m163_quoted_probe"

    def run(sql):
        return cur.execute(sql.replace("FROM claim", f"FROM {table}")).fetchall()

    try:
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        except oracledb.DatabaseError:
            pass
        cur.execute(f"CREATE TABLE {table} (id NUMBER, amount NUMBER)")
        cur.execute(f"INSERT INTO {table} SELECT level, MOD(level * 7, 20) FROM dual "
                    "CONNECT BY level <= 20")
        # What the two ROWID shapes approved before the fix returned: one address per row.
        for sql in ("SELECT s.ROWID FROM (SELECT id FROM claim) s",
                    "WITH s AS (SELECT id FROM claim) SELECT s.ROWID FROM s"):
            addresses = run(sql)
            assert len(addresses) == 20 and all(isinstance(a, str) and len(a) == 18
                                                for (a,) in addresses), addresses
        # The two ROWNUM shapes and the alias, Oracle rejected rather than answered.
        for sql, error in (("SELECT s.ROWNUM FROM (SELECT id FROM claim) s", "ORA-03050"),
                           ("SELECT id AS rownum FROM claim WHERE rownum <= 5", "ORA-00923")):
            with pytest.raises(oracledb.DatabaseError, match=error):
                run(sql)
        # Quoted: `"ROWID"` is still ROWID and no table may have it, the rest are ordinary names.
        assert sorted(run('SELECT "ROWID" FROM claim')) == sorted(run("SELECT ROWID FROM claim"))
        with pytest.raises(oracledb.DatabaseError, match="ORA-03050"):
            cur.execute(f'CREATE TABLE {quoted_probe} ("ROWID" NUMBER)')
        for name in ("rowid", "ROWNUM"):
            with pytest.raises(oracledb.DatabaseError, match="ORA-00904"):
                run(f'SELECT "{name}" FROM claim')
    finally:
        for probe in (table, quoted_probe):
            try:
                cur.execute(f"DROP TABLE {probe} PURGE")
            except oracledb.DatabaseError:
                pass  # the quoted probe exists only if Oracle accepted the column it must not
        con.close()

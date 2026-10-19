"""Two DuckDB spellings that read a column no node names (the ui lane's replays of #167).

Method-call syntax, `secret.lower()`, is `lower(secret)` to DuckDB and a Dot over an identifier
and a call to sqlglot, with no Column node; `COLUMNS(...)` inside an expression expands to
columns that appear in no node. Each returned a withheld value under deny, under mask and
hidden with the schema listing it. Both are refused by the shape check before any check runs.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

FULL = {"claim": {"id", "secret", "info"}}
COMPLETE = AccessPolicy(policy_schema=FULL)
HIDDEN = {"claim": {"id"}}  # the grant lists only `id`; the policy's schema lists `secret`


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INT, secret VARCHAR, info STRUCT(ssn VARCHAR)); "
                "INSERT INTO claim VALUES (1, 'CLAIM-SECRET', {'ssn': 'SSN-SECRET'})")
    return con


def _decide(sql, visible, policy=None, engine="duckdb"):
    return decide(sql, visible, dialect=engine, target=engine, policy=policy)


def _withheld(con, sql, value=None):
    """Hidden with the schema listing the column, denied, masked: refused, before any check.
    The premise: the engine runs the spelling, and returns the value where one is projected
    (a WHERE, ORDER BY or HAVING over it is an oracle, the rows following the hidden value)."""
    rows = str(con.execute(sql).fetchall()).upper()
    assert value is None or value in rows, (sql, rows)
    hidden = _decide(sql, HIDDEN, policy=COMPLETE)
    denied = _decide(sql, FULL, policy=AccessPolicy(denied={("claim", "secret")}, policy_schema=FULL))
    masked = _decide(sql, FULL, policy=AccessPolicy(masked={("claim", "secret")}, policy_schema=FULL))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, (sql, verdict)


@pytest.mark.parametrize("sql", [
    "SELECT secret.upper() AS s FROM claim",
    "SELECT secret.max() AS s FROM claim",
    "SELECT claim.secret.lower() AS s FROM claim",
    "SELECT c.secret.lower() AS s FROM claim c",
    "SELECT id FROM claim WHERE secret.starts_with('CLAIM')",
    "SELECT id, secret.lower() AS s FROM claim ORDER BY secret.lower()",
    "SELECT id FROM claim GROUP BY id HAVING secret.max() LIKE 'CLAIM%'",
    # The ui lane's edge list: a chain, the left side's spellings, a struct column, a derived alias
    "SELECT secret.lower().upper() AS s FROM claim",
    "SELECT SECRET.lower() AS s FROM claim",
    'SELECT "SECRET".lower() AS s FROM claim',
    "SELECT (secret).lower() AS s FROM claim",
    "SELECT info.struct_extract('ssn') AS s FROM claim",
    "SELECT d.secret.lower() AS s FROM (SELECT * FROM claim) d",
])
def test_a_method_call_on_a_column_is_refused_as_unnamed(con, sql):
    projected = sql.split(" FROM ")[0] != "SELECT id"
    value = "SSN-SECRET" if "info" in sql else "CLAIM-SECRET"
    _withheld(con, sql, value if projected else None)
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Refusal)  # granted or not: the spelling


def test_a_qualified_call_is_refused_on_duckdb_whatever_the_grant_says(con):
    # The decider cannot tell `main.lower(secret)` (a schema-qualified call, the column in the
    # arguments) from `secret.lower()` without resolving the left side against the whole schema,
    # and the grant lists only the visible columns: on DuckDB the shape goes, granted or not.
    sql = "SELECT main.lower(secret) AS s FROM claim"
    assert con.execute(sql).fetchall() == [("claim-secret",)]
    for visible in (HIDDEN, FULL):
        verdict = _decide(sql, visible, policy=COMPLETE)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, verdict
        assert "f(x)" in verdict.message
    # With no schema at all (a shape check alone, which the deciders never run), the same.
    from mnemiq.sql.guard import check_shape

    for bare in ("SELECT secret.lower() AS s FROM claim", "SELECT main.lower(secret) AS s FROM claim"):
        verdict = check_shape(bare, dialect="duckdb")
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, (bare, verdict)


def test_on_oracle_a_package_call_passes_and_an_alias_led_method_is_refused():
    # Oracle has no method syntax on a bare column: a method of an object-typed column is
    # reached only through a table alias. A bare identifier before the call is a package or
    # schema name, and the column in its arguments is what the checks read.
    package = "SELECT DBMS_LOB.SUBSTR(secret, 10, 1) AS s FROM claim"
    hidden = _decide(package, HIDDEN, policy=COMPLETE, engine="oracle")
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    granted = _decide(package, FULL, policy=COMPLETE, engine="oracle")
    assert not (isinstance(granted, Refusal) and granted.code is RefusalCode.UNMODELLED_CALL
                and "method syntax" in granted.message), granted
    for method in ("SELECT c.secret.lower() AS s FROM claim c", "SELECT claim.secret.lower() AS s FROM claim",
                   "SELECT (secret).lower() AS s FROM claim"):
        for visible in (HIDDEN, FULL):
            verdict = _decide(method, visible, policy=COMPLETE, engine="oracle")
            assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, (method, verdict)


def test_the_plain_call_is_read_as_the_column(con):
    plain = "SELECT lower(secret) AS s FROM claim"
    hidden = _decide(plain, HIDDEN, policy=COMPLETE)
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    granted = _decide(plain, FULL, policy=COMPLETE)
    assert isinstance(granted, Approved), granted
    assert con.execute(granted.target_sql).fetchall() == [("claim-secret",)]


@pytest.mark.parametrize("sql", [
    "SELECT d.secret FROM (SELECT upper(COLUMNS('e')) AS secret FROM claim) d",
    "SELECT d.secret FROM (SELECT max(COLUMNS('e')) AS secret FROM claim) d",
    "SELECT d.s FROM (SELECT struct_pack(COLUMNS('e')) AS s FROM claim) d",
    "SELECT d.x FROM (SELECT concat(*COLUMNS('e')) AS x FROM claim) d",
    "WITH w AS (SELECT upper(COLUMNS('e')) AS x FROM claim) SELECT w.x FROM w",
    "SELECT id FROM claim WHERE COLUMNS('ecr') LIKE 'CLAIM%'",
    "SELECT id FROM claim ORDER BY COLUMNS('ecr')",
    "SELECT id FROM claim WHERE claim.COLUMNS('ecr') LIKE 'CLAIM%'",
])
def test_columns_inside_an_expression_is_refused_as_unnamed(con, sql):
    try:
        con.execute(sql).fetchall()  # the engine accepts the spelling, or refuses it itself
    except duckdb.Error:
        pass
    for visible, policy in ((HIDDEN, COMPLETE),
                            (FULL, AccessPolicy(denied={("claim", "secret")}, policy_schema=FULL)),
                            (FULL, AccessPolicy(masked={("claim", "secret")}, policy_schema=FULL))):
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


def test_columns_as_a_bare_projection_of_a_derived_table_is_still_read(con):
    # The resolver reads a bare COLUMNS projection (#167); only the expression spellings go.
    sql = "SELECT d.id FROM (SELECT COLUMNS('id') FROM claim) d"
    assert con.execute(sql).fetchall() == [(1,)]
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)
    field = "SELECT (struct_pack(a := 1)).a AS a FROM claim"  # a Dot over a field, not a call
    assert con.execute(field).fetchall() == [(1,)]
    assert isinstance(_decide(field, FULL, policy=COMPLETE), Approved)

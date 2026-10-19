"""Two DuckDB spellings that read a column no node names (the ui lane's replays of #167).

Method-call syntax, `secret.lower()`, is `lower(secret)` to DuckDB and a Dot over an identifier
and a call to sqlglot, with no Column node; `COLUMNS(...)` inside an expression expands to
columns that appear in no node. Each returned a withheld value under deny, under mask and
hidden with the schema listing it. Both are refused by the shape check before any check runs.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
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


class _OkAdapter:
    def execute(self, sql):
        return []  # the write decider's EXPLAIN succeeds


SINK = {"sink": {"s"}}


def _decide_write(sql, visible, policy):
    grants = GrantSet(frozenset(visible), writable=frozenset(SINK))
    return decide_write(sql, visible, grants, adapter=_OkAdapter(), dialect="duckdb", target="duckdb",
                        writes_enabled=True, policy=policy)


@pytest.mark.parametrize("sql, code", [
    ("INSERT INTO sink SELECT secret.lower() FROM claim", RefusalCode.UNMODELLED_CALL),
    ("INSERT INTO sink SELECT upper(COLUMNS('ecr')) FROM claim", RefusalCode.SELECT_STAR),
    ("INSERT INTO sink SELECT secret FROM claim WHERE COLUMNS('ecr') LIKE 'CLAIM%'", RefusalCode.SELECT_STAR),
    ("UPDATE sink SET s = (SELECT secret.max() FROM claim) WHERE s IS NOT NULL", RefusalCode.UNMODELLED_CALL),
    ("DELETE FROM sink WHERE s IN (SELECT secret.lower() FROM claim)", RefusalCode.UNMODELLED_CALL),
])
def test_a_write_reading_through_either_spelling_is_refused(con, sql, code):
    # Codex's first review of #169: the write decider has a shape check of its own, which had
    # not run the guard, and an INSERT ... SELECT copied the denied value into a writable table.
    con.execute("CREATE TABLE sink (s VARCHAR); INSERT INTO sink VALUES ('x')")
    con.execute(sql)  # the engine accepts the spelling
    assert "CLAIM-SECRET" in str(con.execute("SELECT s FROM sink").fetchall()).upper() or sql.startswith("DELETE")
    hidden = _decide_write(sql, {**HIDDEN, **SINK}, AccessPolicy(policy_schema={**FULL, **SINK}))
    denied = _decide_write(sql, {**FULL, **SINK},
                           AccessPolicy(denied={("claim", "secret")}, policy_schema={**FULL, **SINK}))
    masked = _decide_write(sql, {**FULL, **SINK},
                           AccessPolicy(masked={("claim", "secret")}, policy_schema={**FULL, **SINK}))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is code, (sql, verdict)


@pytest.mark.parametrize("sql", [
    "INSERT INTO sink (s) SELECT COLUMNS('ecr') FROM claim",
    "INSERT INTO sink (s) SELECT COLUMNS('ecr') FROM claim UNION ALL SELECT 'x'",
    "INSERT INTO sink (s) SELECT * FROM (SELECT COLUMNS('ecr') FROM claim) d",
    "WITH w AS (SELECT COLUMNS('ecr') FROM claim) INSERT INTO sink (s) SELECT * FROM w",
    "INSERT INTO sink (s) SELECT d.secret FROM (SELECT COLUMNS('ecr') FROM claim) d",
    "INSERT INTO sink SELECT * FROM claim",
    "INSERT INTO sink SELECT c.* FROM claim c",
    "INSERT INTO sink SELECT * FROM (SELECT * FROM claim) d",
    "INSERT INTO sink (s) SELECT d.secret FROM (SELECT * FROM claim) d",
    "WITH w AS (SELECT * FROM claim) INSERT INTO sink (s) SELECT w.secret FROM w",
    "INSERT INTO sink (s) SELECT (SELECT * FROM (SELECT secret FROM claim) d)",
    "INSERT INTO sink (i, s) SELECT id, secret FROM claim RETURNING *",
    "DELETE FROM sink WHERE i = 1 RETURNING *",
    "UPDATE sink SET s = (SELECT * FROM (SELECT secret FROM claim) d) WHERE i = 1",
    "DELETE FROM sink WHERE s IN (SELECT * FROM (SELECT secret FROM claim) d)",
])
def test_a_write_through_a_star_or_a_bare_columns_is_refused(con, sql):
    # Codex's second review of #169 (a bare COLUMNS in an INSERT's source) and M195 (a star): a
    # write's INSERT source has no outermost star check, so each of these copied the withheld
    # value into a table the grant reads.
    con.execute("CREATE TABLE sink (i INT, s VARCHAR, j STRUCT(ssn VARCHAR)); INSERT INTO sink VALUES (1, 'x', NULL)")
    con.execute(sql)
    assert "CLAIM-SECRET" in str(con.execute("SELECT s FROM sink").fetchall()).upper() or sql.startswith("DELETE")
    if sql.endswith("RETURNING *"):
        assert "CLAIM-SECRET" in str(con.execute(sql).fetchall()).upper() or sql.startswith("DELETE")
    schema = {**FULL, "sink": {"i", "s", "j"}}
    hidden = _decide_write(sql, {**HIDDEN, "sink": {"i", "s", "j"}}, AccessPolicy(policy_schema=schema))
    denied = _decide_write(sql, schema, AccessPolicy(denied={("claim", "secret")}, policy_schema=schema))
    masked = _decide_write(sql, schema, AccessPolicy(masked={("claim", "secret")}, policy_schema=schema))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


def test_count_distinct_star_in_a_write_is_refused_as_a_star(con):
    # The review gate on the second round: its holder is a Distinct, not a Count, so it is a star.
    sql = "INSERT INTO sink (i) SELECT COUNT(DISTINCT *) FROM claim"
    schema = {**FULL, "sink": {"i", "s", "j"}}
    verdict = _decide_write(sql, schema, AccessPolicy(policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, verdict


def test_a_write_naming_its_columns_keeps_count_star_and_exists_star(con):
    con.execute("CREATE TABLE sink (i INT, s VARCHAR, j STRUCT(ssn VARCHAR)); INSERT INTO sink VALUES (1, 'x', NULL)")
    schema = {**FULL, "sink": {"i", "s", "j"}}
    for sql in ("INSERT INTO sink (i) SELECT COUNT(*) FROM claim",
                "DELETE FROM sink WHERE EXISTS (SELECT * FROM claim WHERE claim.id = sink.i)",
                "INSERT INTO sink (i) SELECT 1 WHERE EXISTS (SELECT COLUMNS('ecr') FROM claim)",
                "INSERT INTO sink (s) SELECT secret FROM claim"):
        con.execute(sql)
        verdict = _decide_write(sql, schema, AccessPolicy(policy_schema=schema))
        assert not isinstance(verdict, Refusal), (sql, verdict)


@pytest.mark.parametrize("sql", [
    "SELECT (SELECT * FROM one LIMIT 1) AS s FROM claim",
    "SELECT id FROM claim WHERE (SELECT * FROM one LIMIT 1) LIKE 'ONE%'",
    "SELECT id FROM claim WHERE secret IN (SELECT * FROM one)",
    "SELECT id FROM claim WHERE secret = ANY (SELECT * FROM one)",
    "SELECT (SELECT * FROM (SELECT secret FROM one) d LIMIT 1) AS s FROM claim",
    "SELECT ((SELECT * FROM one LIMIT 1)) AS s FROM claim",
])
def test_a_star_in_a_scalar_or_in_subquery_is_refused_as_unnamed(con, sql):
    # M196: a one-column table's star in a scalar subquery names no Column, and neither the
    # outermost star check nor the resolver reads it, so the value came back under every case.
    con.execute("CREATE TABLE one (secret VARCHAR); INSERT INTO one VALUES ('ONE-SECRET')")
    rows = str(con.execute(sql).fetchall()).upper()
    assert "ONE-SECRET" in rows or sql.split(" FROM ")[0] == "SELECT id", (sql, rows)
    schema = {**FULL, "one": {"secret"}}
    hidden = _decide(sql, {**FULL, "one": set()}, policy=AccessPolicy(policy_schema=schema))
    denied = _decide(sql, schema, policy=AccessPolicy(denied={("one", "secret")}, policy_schema=schema))
    masked = _decide(sql, schema, policy=AccessPolicy(masked={("one", "secret")}, policy_schema=schema))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


@pytest.mark.parametrize("sql", [
    "SELECT id FROM claim WHERE (SELECT COLUMNS('ecr') FROM one LIMIT 1) = 'ONE-SECRET'",
    "SELECT id FROM claim WHERE 'ONE-SECRET' IN (SELECT COLUMNS('ecr') FROM one)",
    "SELECT id FROM claim WHERE EXISTS (SELECT * FROM one INTERSECT SELECT 'ONE-SECRET')",
    "SELECT id FROM claim WHERE EXISTS (SELECT * FROM one EXCEPT SELECT 'NOT-IT')",
    "SELECT id FROM claim WHERE EXISTS (SELECT COLUMNS('ecr') FROM one INTERSECT SELECT 'ONE-SECRET')",
    "SELECT id FROM claim WHERE EXISTS ((SELECT * FROM one) INTERSECT (SELECT 'ONE-SECRET'))",
    "SELECT id FROM claim WHERE EXISTS (SELECT * FROM one INTERSECT SELECT 'ONE-SECRET' UNION ALL SELECT 'x' WHERE FALSE)",
])
def test_a_predicate_oracle_through_a_bare_columns_or_an_exists_intersect_is_refused(con, sql):
    # Codex's third review of #169: a bare COLUMNS in a scalar or IN subquery is a projection
    # under a Select, and an EXISTS over an INTERSECT or EXCEPT compares its projections, so
    # each returned a row for the withheld value and none for a wrong guess.
    con.execute("CREATE TABLE one (secret VARCHAR); INSERT INTO one VALUES ('ONE-SECRET')")
    assert con.execute(sql).fetchall() == [(1,)], sql
    wrong = sql.replace("ONE-SECRET", "WRONG").replace("NOT-IT", "ONE-SECRET")
    assert con.execute(wrong).fetchall() == [], wrong
    schema = {**FULL, "one": {"secret"}}
    hidden = _decide(sql, {**FULL, "one": set()}, policy=AccessPolicy(policy_schema=schema))
    denied = _decide(sql, schema, policy=AccessPolicy(denied={("one", "secret")}, policy_schema=schema))
    masked = _decide(sql, schema, policy=AccessPolicy(masked={("one", "secret")}, policy_schema=schema))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)
    write = "INSERT INTO sink (i) " + sql
    con.execute("CREATE TABLE sink (i INT, s VARCHAR, j STRUCT(ssn VARCHAR))")
    con.execute(write)
    assert con.execute("SELECT i FROM sink").fetchall() == [(1,)]
    schema_w = {**schema, "sink": {"i", "s", "j"}}
    verdict = _decide_write(write, schema_w, AccessPolicy(denied={("one", "secret")}, policy_schema=schema_w))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (write, verdict)


@pytest.mark.parametrize("sql", [
    "INSERT INTO sink (i, s) SELECT id, #2 FROM claim",
    "UPDATE sink SET s = (SELECT #2 FROM claim LIMIT 1) WHERE i = 1",
    "INSERT INTO sink (i, s) SELECT d.x, d.y FROM claim AS d(x, y, z)",
])
def test_a_column_reached_by_position_in_a_write_is_refused(con, sql):
    # M197 (the ui lane's replay of #169): the read side refuses `#2` and a renaming column
    # list on a table reference (M184, M185); the write decider's shape check never ran that
    # check (a column list over a derived STAR is refused by the star rule first).
    con.execute("CREATE TABLE sink (i INT, s VARCHAR, j STRUCT(ssn VARCHAR)); INSERT INTO sink VALUES (1, 'x', NULL)")
    con.execute(sql)
    assert "CLAIM-SECRET" in str(con.execute("SELECT s FROM sink").fetchall()).upper()
    schema = {**FULL, "sink": {"i", "s", "j"}}
    hidden = _decide_write(sql, {**HIDDEN, "sink": {"i", "s", "j"}}, AccessPolicy(policy_schema=schema))
    denied = _decide_write(sql, schema, AccessPolicy(denied={("claim", "secret")}, policy_schema=schema))
    masked = _decide_write(sql, schema, AccessPolicy(masked={("claim", "secret")}, policy_schema=schema))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


@pytest.mark.parametrize("sql", [
    "SELECT count(*) AS n FROM (SELECT DISTINCT * FROM dup) d",
    "SELECT count(*) AS n FROM (SELECT DISTINCT ON (1) * FROM dup) d",
    "SELECT count(*) AS n FROM (SELECT * FROM dup GROUP BY ALL) d",
    "SELECT count(*) AS n FROM (SELECT * FROM dup GROUP BY 1) d",
    "WITH w AS (SELECT DISTINCT * FROM dup) SELECT count(*) AS n FROM w",
    "SELECT d.id FROM (SELECT * FROM pair ORDER BY 2 LIMIT 1) d",
    "SELECT d.id FROM (SELECT * EXCLUDE (id), id FROM pair ORDER BY ALL LIMIT 1) d",
    "SELECT d.id FROM ((SELECT * FROM pair) ORDER BY 2 LIMIT 1) d",
    "SELECT d.id FROM (((SELECT * FROM pair)) ORDER BY 2 LIMIT 1) d",
    "SELECT d.id FROM (SELECT * FROM pair UNION ALL SELECT * FROM pair ORDER BY 2 LIMIT 1) d",
    "WITH w AS ((SELECT * FROM pair) ORDER BY 2 LIMIT 1) SELECT w.id FROM w",  # refused before the CTE branch
])
def test_a_derived_star_consumed_by_its_own_select_is_refused(con, sql):
    # M198 (the ui lane's replay of #169): the resolver reads a derived star only where the outer
    # query names a column, but a DISTINCT, a GROUP BY or an ORDER BY ALL or by position in the
    # derived Select reads every column itself: a count that says whether two withheld values
    # are equal, or the id of the smallest.
    con.execute("CREATE TABLE dup (secret VARCHAR); INSERT INTO dup VALUES ('S'), ('S'); "
                "CREATE TABLE pair (id INT, secret VARCHAR); INSERT INTO pair VALUES (1, 'Z'), (2, 'A')")
    first = con.execute(sql).fetchall()
    con.execute("INSERT INTO dup VALUES ('T'); UPDATE pair SET secret = 'A' WHERE id = 1; UPDATE pair SET secret = 'Z' WHERE id = 2")
    assert con.execute(sql).fetchall() != first, (sql, first)  # the answer follows the withheld values
    table = "dup" if "dup" in sql else "pair"
    schema = {**FULL, "dup": {"secret"}, "pair": {"id", "secret"}}
    hidden = _decide(sql, {**schema, table: schema[table] - {"secret"}}, policy=AccessPolicy(policy_schema=schema))
    denied = _decide(sql, schema, policy=AccessPolicy(denied={(table, "secret")}, policy_schema=schema))
    masked = _decide(sql, schema, policy=AccessPolicy(masked={(table, "secret")}, policy_schema=schema))
    for verdict in (hidden, denied, masked):
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


def test_a_derived_star_its_own_select_does_not_consume_is_still_resolved(con):
    con.execute("CREATE TABLE pair (id INT, secret VARCHAR); INSERT INTO pair VALUES (1, 'Z'), (2, 'A')")
    schema = {**FULL, "pair": {"id", "secret"}}
    for sql in ("SELECT d.id FROM (SELECT * FROM pair ORDER BY id LIMIT 1) d",
                "SELECT d.id FROM (SELECT * FROM pair LIMIT 1) d",
                "SELECT count(*) AS n FROM (SELECT * FROM pair) d",
                "SELECT d.id FROM (SELECT * FROM pair WHERE id IN (SELECT id FROM pair ORDER BY 1)) d"):
        con.execute(sql)
        assert isinstance(_decide(sql, schema, policy=AccessPolicy(policy_schema=schema)), Approved), sql
        denied = _decide(sql, schema, policy=AccessPolicy(denied={("pair", "secret")}, policy_schema=schema))
        assert not isinstance(denied, Refusal) or denied.code is not RefusalCode.SELECT_STAR, (sql, denied)


def test_a_star_the_resolver_or_the_star_check_reads_is_left_to_them(con):
    con.execute("CREATE TABLE one (secret VARCHAR); INSERT INTO one VALUES ('ONE-SECRET')")
    schema = {**FULL, "one": {"secret"}}
    granted = AccessPolicy(policy_schema=schema)
    for sql in ("SELECT id FROM claim WHERE EXISTS (SELECT * FROM one)",
                "SELECT id FROM claim WHERE EXISTS (SELECT * FROM one UNION ALL SELECT * FROM one)",
                "SELECT d.secret FROM (SELECT * FROM one) d",
                "WITH w AS (SELECT * FROM one) SELECT w.secret FROM w",
                "SELECT d.secret FROM ((SELECT * FROM one)) d",
                "SELECT id FROM claim, LATERAL (SELECT * FROM one) l",
                "SELECT COUNT(*) AS n FROM one"):
        con.execute(sql)
        assert isinstance(_decide(sql, schema, policy=granted), Approved), sql
    for sql in ("SELECT d.secret FROM (SELECT * FROM one) d", "WITH w AS (SELECT * FROM one) SELECT w.secret FROM w"):
        denied = _decide(sql, schema, policy=AccessPolicy(denied={("one", "secret")}, policy_schema=schema))
        assert isinstance(denied, Refusal) and denied.code is not RefusalCode.SELECT_STAR, (sql, denied)
    outermost = _decide("SELECT * FROM one", schema, policy=granted)
    assert isinstance(outermost, Refusal) and outermost.code is RefusalCode.SELECT_STAR, outermost


def test_columns_as_a_bare_projection_of_a_derived_table_is_still_read(con):
    # The resolver reads a bare COLUMNS projection (#167); only the expression spellings go.
    sql = "SELECT d.id FROM (SELECT COLUMNS('id') FROM claim) d"
    assert con.execute(sql).fetchall() == [(1,)]
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)
    field = "SELECT (struct_pack(a := 1)).a AS a FROM claim"  # a Dot over a field, not a call
    assert con.execute(field).fetchall() == [(1,)]
    assert isinstance(_decide(field, FULL, policy=COMPLETE), Approved)

"""A select alias was admitted as a bare name in every clause of its own SELECT, and Oracle reads
one in ORDER BY alone: `SELECT id AS mq_leak FROM claim WHERE mq_leak = 1` was approved there and
Oracle resolved the name to a no-argument function called without parentheses, an opaque call
the function guard never sees, returning every row; `AS ORA_ROWSCN ... WHERE ORA_ROWSCN IS NOT
NULL` filtered on the pseudo-column (M164, measured on Oracle 23). A select alias now binds only
in the clauses the executing engine reads it in: measured on DuckDB and SQLite here, on Oracle
for ORDER BY, as documented for Postgres and MySQL, ORDER BY alone for any other engine.
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id", "v"}}
SHAPES = {
    "where": "SELECT id AS x FROM claim WHERE x = 1",
    "group": "SELECT v AS x, count(*) AS n FROM claim GROUP BY x",
    "having": "SELECT v AS x, count(*) AS n FROM claim GROUP BY v HAVING x > 10",
    "order": "SELECT id AS x FROM claim ORDER BY x",
    "select": "SELECT id AS x, x + 1 AS y FROM claim",
    "qualify": "SELECT id AS x, row_number() OVER () AS r FROM claim QUALIFY x = 1",
    "join": "SELECT c.id AS x FROM claim c JOIN claim d ON x = d.id",
    "window": "SELECT id AS x, row_number() OVER (ORDER BY x) AS r FROM claim",
    "order in an expression": "SELECT id AS x FROM claim ORDER BY x + 0",
    "group in an expression": "SELECT v AS x, count(*) AS n FROM claim GROUP BY x + 0",
    "having in an expression": "SELECT v AS x, count(*) AS n FROM claim GROUP BY v HAVING x + 0 > 10",
    "qualify in an expression": "SELECT id AS x, row_number() OVER () AS r FROM claim QUALIFY x + 0 = 1",
}
# Where each engine reads the alias: DuckDB and SQLite measured in this file, Oracle's bare
# ORDER BY item measured in the register, Postgres and MySQL as documented, an output name
# inside an expression read as an input column on Postgres (`ORDER BY xmin + 0` would read the
# system column) and unmeasured on MySQL and Oracle, so not admitted; a predicate (WHERE,
# HAVING, QUALIFY, ON) always holds the name inside an expression and is read as the clause.
READS = {
    "duckdb": {"where", "group", "having", "order", "select", "qualify", "window",
               "order in an expression", "having in an expression", "qualify in an expression"},
    "sqlite": {"where", "group", "having", "order", "join", "order in an expression", "group in an expression",
               "having in an expression"},
    "postgres": {"group", "order"},
    "mysql": {"group", "having", "order", "having in an expression"},
    "oracle": {"order"},
}


def _decide(sql, dialect, policy=None):
    return decide(sql, VISIBLE, dialect=dialect, target=dialect, policy=policy)


@pytest.mark.parametrize("engine", sorted(READS))
@pytest.mark.parametrize("clause", sorted(SHAPES))
def test_the_alias_binds_where_the_engine_reads_it_and_nowhere_else(engine, clause):
    verdict = _decide(SHAPES[clause], engine)
    if clause in READS[engine]:
        assert isinstance(verdict, Approved), (engine, clause, verdict)
    else:
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (engine, clause, verdict)


@pytest.mark.parametrize("clause", sorted(SHAPES))
def test_duckdb_and_sqlite_read_the_alias_as_the_table_says(clause):
    # The table's two measured rows, re-measured: the engine runs the shape where the table
    # says it reads the alias and fails to bind it where it says not (QUALIFY is DuckDB's).
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT); INSERT INTO claim VALUES (1, 10), (2, 20)")
    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE claim (id INT, v INT)")
    lite.executemany("INSERT INTO claim VALUES (?, ?)", [(1, 10), (2, 20)])
    for engine, con in (("duckdb", duck), ("sqlite", lite)):
        if engine == "sqlite" and clause.startswith("qualify"):
            continue  # SQLite has no QUALIFY
        try:
            con.execute(SHAPES[clause]).fetchall()
            ran = True
        except Exception:  # noqa: BLE001 -- the engine's own binder error is the measurement
            ran = False
        assert ran == (clause in READS[engine]), (engine, clause)


def test_an_aggregates_own_order_by_and_a_limit_read_no_alias():
    # Codex's first review of #177: `SELECT 0 AS secret, list(id ORDER BY secret) FROM claim`
    # with `secret` hidden ordered by the hidden column, the aggregate's ORDER BY read as the
    # SELECT's, where DuckDB reads an alias. Measured: DuckDB rejects an alias in an aggregate's
    # ORDER BY with no source column of the name, and DuckDB and SQLite reject one in LIMIT;
    # the guard admits neither, so the hidden name is refused with or without the schema.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT, secret INT); INSERT INTO claim VALUES (1, 10, 2), (2, 20, 1)")
    assert duck.execute("SELECT 0 AS secret, list(id ORDER BY secret) FROM claim").fetchall() == [(0, [2, 1])]
    # WITHIN GROUP (ORDER BY ...) orders by the source column the same, and is no Func node
    # to sqlglot (the gate on the second commit)
    assert duck.execute("SELECT 0 AS secret, percentile_disc(0.5) WITHIN GROUP (ORDER BY secret) FROM claim").fetchall() == [(0, 1)]
    for sql in ("SELECT 1 AS x FROM claim LIMIT x", "SELECT 1 AS x FROM claim LIMIT 1 OFFSET x", "SELECT 0 AS x, list(id ORDER BY x) FROM claim",
                "SELECT 0 AS x, percentile_disc(0.5) WITHIN GROUP (ORDER BY x) FROM claim"):
        with pytest.raises(duckdb.Error):   # no source column of the name: the alias binds nowhere here
            duck.execute(sql).fetchall()
    full = AccessPolicy(policy_schema={"claim": {"id", "v", "secret"}})
    for sql in ("SELECT 0 AS secret, list(id ORDER BY secret) FROM claim",
                "SELECT 0 AS secret, percentile_disc(0.5) WITHIN GROUP (ORDER BY secret) AS p FROM claim",
                "SELECT 0 AS secret, mode() WITHIN GROUP (ORDER BY secret) AS p FROM claim"):
        for policy in (None, full):
            verdict = _decide(sql, "duckdb", policy=policy)
            assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, policy, verdict)
    # the guard itself refuses an alias in LIMIT or OFFSET (the shape check refuses an
    # unreadable LIMIT before it through `decide`)
    import sqlglot
    from mnemiq.sql.authz_guard import check_access
    for sql in ("SELECT 1 AS x FROM claim LIMIT x", "SELECT 1 AS x FROM claim LIMIT 1 OFFSET x"):
        verdict = check_access(sqlglot.parse_one(sql, read="duckdb"), VISIBLE, "duckdb")
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_source_column_named_like_the_alias_is_bound_under_the_complete_schema():
    # The same review's collision, `SELECT id AS secret FROM claim WHERE secret = 1` with
    # `secret` hidden: DuckDB and SQLite read the source column first in WHERE, the
    # projection list and QUALIFY (measured), and so does the resolver under the complete
    # schema, which every snapshot-built policy carries, refusing the hidden name; from the
    # grant alone the alias reading stands, the register's residual recorded at #167.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT, secret INT); INSERT INTO claim VALUES (1, 10, 2), (2, 20, 1)")
    assert duck.execute("SELECT id AS secret FROM claim WHERE secret = 1").fetchall() == [(2,)]
    full = AccessPolicy(policy_schema={"claim": {"id", "v", "secret"}})
    for sql in ("SELECT id AS secret FROM claim WHERE secret = 1", "SELECT id AS secret, secret + 0 AS s FROM claim",
                "SELECT id AS secret FROM claim QUALIFY row_number() OVER (ORDER BY secret) = 1"):
        verdict = _decide(sql, "duckdb", policy=full)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
        assert isinstance(_decide(sql, "duckdb"), Approved), sql   # the residual, from the grant alone


@pytest.mark.parametrize("sql", ["SELECT v AS x, count(*) AS n FROM claim GROUP BY ROLLUP(x)",
                                 "SELECT v AS x, count(*) AS n FROM claim GROUP BY CUBE(x)",
                                 "SELECT v AS x, count(*) AS n FROM claim GROUP BY GROUPING SETS ((x), ())",
                                 "SELECT v AS x, id AS y, count(*) AS n FROM claim GROUP BY GROUPING SETS ((x, y), (x))"])
def test_a_grouping_set_wrapper_keeps_a_bare_alias(sql):
    # The same review: `GROUP BY ROLLUP(x)` runs on DuckDB and the wrapper made the alias an
    # expression, which DuckDB's GROUP BY table does not read, a false refusal.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT); INSERT INTO claim VALUES (1, 10), (2, 20)")
    duck.execute(sql).fetchall()
    assert isinstance(_decide(sql, "duckdb"), Approved), sql
    assert isinstance(_decide(sql, "postgres"), Approved), sql


def test_an_aggregates_filter_predicate_reads_no_alias():
    # Codex's third review of #177: `count(*) FILTER (WHERE secret = 1)` beside an alias `secret`
    # counted the hidden column's rows, the predicate read as the SELECT's WHERE, where DuckDB
    # admits an alias; without the source column DuckDB rejects the query, so FILTER binds no
    # alias (measured). Refused with and without the schema.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT, secret INT); INSERT INTO claim VALUES (1, 10, 2), (2, 20, 1), (3, 30, 2)")
    assert duck.execute("SELECT 0 AS secret, count(*) FILTER (WHERE secret = 1) FROM claim").fetchall() == [(0, 1)]
    bare = duckdb.connect()
    bare.execute("CREATE TABLE claim (id INT, v INT); INSERT INTO claim VALUES (1, 10), (2, 20)")
    with pytest.raises(duckdb.Error):
        bare.execute("SELECT 0 AS x, count(*) FILTER (WHERE x = 1) FROM claim").fetchall()
    full = AccessPolicy(policy_schema={"claim": {"id", "v", "secret"}})
    for policy in (None, full):
        verdict = _decide("SELECT 0 AS secret, count(*) FILTER (WHERE secret = 1) FROM claim", "duckdb", policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (policy, verdict)
    verdict = _decide("SELECT 0 AS x, count(*) FILTER (WHERE x = 1) FROM claim", "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert isinstance(_decide("SELECT 0 AS x, count(*) FILTER (WHERE v = 10) FROM claim", "duckdb"), Approved)


def test_a_window_functions_filter_and_order_by_read_the_alias_on_duckdb():
    # Codex's fourth review of #177: `count(*) FILTER (WHERE secret = 1) OVER ()` binds the
    # alias on DuckDB (measured; the plain aggregate's FILTER binds none, the test above), and
    # so does an aggregate's own ORDER BY under OVER; Postgres binds the alias in neither.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT); INSERT INTO claim VALUES (1, 10), (2, 20), (1, 30)")
    for sql in ("SELECT v AS x, count(*) FILTER (WHERE x = 10) OVER () AS n FROM claim",
                "SELECT v AS x, string_agg(id, ',' ORDER BY x) OVER () AS s FROM claim"):
        duck.execute(sql).fetchall()
        assert isinstance(_decide(sql, "duckdb"), Approved), sql
        verdict = _decide(sql, "postgres")
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    plain = "SELECT v AS x, string_agg(id, ',' ORDER BY x) AS s FROM claim GROUP BY v"
    with pytest.raises(duckdb.Error):
        duck.execute(plain).fetchall()
    verdict = _decide(plain, "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    # WITHIN GROUP under OVER is not measured on DuckDB (no aggregate here takes that form under
    # a window), so it stays the aggregate's and admits no alias (the gate's round)
    verdict = _decide("SELECT v AS x, percentile_disc(0.5) WITHIN GROUP (ORDER BY x) OVER () AS p FROM claim", "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


def test_redundant_parentheses_around_a_bare_item_read_through():
    # The same review: `GROUP BY ((x))` reads the alias as `GROUP BY x` does, on DuckDB,
    # Postgres, MySQL and SQLite (measured), where the parentheses made the item an expression
    # and the table refused it; a list in GROUP BY is its items on DuckDB and Postgres, and in
    # ORDER BY a row constructor, an expression Postgres reads as input columns (measured).
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT); INSERT INTO claim VALUES (1, 10), (2, 20), (1, 30)")
    wrapped = ("SELECT v AS x FROM claim GROUP BY (x)", "SELECT v AS x FROM claim GROUP BY ((x))",
               "SELECT v AS x FROM claim GROUP BY (x, id)", "SELECT v AS x FROM claim GROUP BY ROLLUP(((x)))",
               "SELECT v AS x FROM claim GROUP BY CUBE((x), id)", "SELECT v AS x FROM claim GROUP BY GROUPING SETS (((x)))",
               "SELECT v AS x FROM claim ORDER BY ((x))", "SELECT DISTINCT ON (((x))) v AS x FROM claim ORDER BY x")
    for sql in wrapped:
        duck.execute(sql).fetchall()
        assert isinstance(_decide(sql, "duckdb"), Approved), sql
    for sql in wrapped:   # Postgres reads each too, DISTINCT ON since the ninth commit
        assert isinstance(_decide(sql, "postgres"), Approved), sql
    verdict = _decide("SELECT v AS x FROM claim ORDER BY (x, id)", "postgres")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    # a row constructor stays an expression under any parentheses: the inner list inherited the
    # outer parentheses' bare reading (Codex's fifth review), where Postgres reads the system
    # column inside the row
    for sql in ("SELECT v AS x FROM claim ORDER BY ((x, id))", "SELECT id AS ctid FROM claim ORDER BY ((ctid, id))"):
        verdict = _decide(sql, "postgres")
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    # Postgres reads a bare DISTINCT ON alias as it reads a bare ORDER BY one, measured on 16
    # (the same review): bare and in redundant parentheses admitted, inside an expression not
    assert isinstance(_decide("SELECT DISTINCT ON (x) id AS x FROM claim ORDER BY x", "postgres"), Approved)
    assert isinstance(_decide("SELECT DISTINCT ON (((x))) id AS x FROM claim ORDER BY x", "postgres"), Approved)
    verdict = _decide("SELECT DISTINCT ON (x + 0) id AS x FROM claim ORDER BY x", "postgres")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    # an engine not measured keeps the parentheses an expression, fail-closed (the gate's round)
    assert isinstance(_decide("SELECT v AS x FROM claim ORDER BY ((x))", "oracle"), Approved)   # measured on Oracle 23
    verdict = _decide("SELECT v AS x FROM claim ORDER BY ((x))", "snowflake")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert isinstance(_decide("SELECT v AS x FROM claim ORDER BY x", "snowflake"), Approved)
    with pytest.raises(duckdb.Error):
        duck.execute("SELECT v AS x FROM claim GROUP BY (x) + 0").fetchall()
    verdict = _decide("SELECT v AS x FROM claim GROUP BY (x) + 0", "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


def test_distinct_on_reads_a_bare_alias_and_a_source_column_inside_an_expression():
    # Codex's second review of #177: `SELECT DISTINCT ON (secret + 0) id AS secret FROM claim`
    # deduplicates on the hidden source column (measured: two rows of three), where the
    # classifier fell through to the projection list, whose DuckDB table admits an alias inside
    # an expression; a bare `DISTINCT ON (secret)` reads the alias (three rows). Its own clause
    # now: a bare alias admitted on DuckDB, an expression's name held to the source's columns.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, v INT, secret INT); INSERT INTO claim VALUES (1, 10, 2), (2, 20, 1), (3, 30, 2)")
    assert duck.execute("SELECT DISTINCT ON (secret + 0) id AS secret FROM claim ORDER BY id").fetchall() == [(1,), (2,)]
    assert duck.execute("SELECT DISTINCT ON (secret) id AS secret FROM claim ORDER BY id").fetchall() == [(1,), (2,), (3,)]
    full = AccessPolicy(policy_schema={"claim": {"id", "v", "secret"}})
    for policy in (None, full):
        for sql in ("SELECT DISTINCT ON (secret + 0) id AS secret FROM claim ORDER BY id",
                    "SELECT DISTINCT ON (abs(secret)) id AS secret FROM claim ORDER BY id"):
            verdict = _decide(sql, "duckdb", policy=policy)
            assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, policy, verdict)
        assert isinstance(_decide("SELECT DISTINCT ON (secret) id AS secret FROM claim ORDER BY id", "duckdb", policy=policy), Approved)
    # an alias of no source column's name, bare and inside an expression, as the table says
    assert isinstance(_decide("SELECT DISTINCT ON (x) id AS x FROM claim ORDER BY id", "duckdb"), Approved)
    verdict = _decide("SELECT DISTINCT ON (x + 0) id AS x FROM claim ORDER BY id", "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    # an aggregate's own DISTINCT is the same node to sqlglot and is its clause's expression
    # (the gate): SQLite reads the alias inside it in HAVING and ORDER BY, measured, and the
    # guard admits it there; DuckDB's table admits it in HAVING too, where the engine rejects
    # an alias inside an aggregate's argument, an error on no row's value
    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE claim (id INT, v INT)")
    lite.executemany("INSERT INTO claim VALUES (?, ?)", [(1, 10), (2, 20)])
    for sql in ("SELECT v AS x FROM claim GROUP BY v HAVING count(DISTINCT x) > 0",
                "SELECT v AS x FROM claim GROUP BY v ORDER BY count(DISTINCT x)"):
        assert lite.execute(sql).fetchall() == [(10,), (20,)], sql
        for engine in ("sqlite", "duckdb"):
            assert isinstance(_decide(sql, engine), Approved), (engine, sql)


def test_oracles_leak_shapes_are_refused():
    # The register's shapes: a no-argument function and a pseudo-column reached through the
    # alias in WHERE; the same alias in ORDER BY is Oracle's own reading and stays approved.
    for sql in ("SELECT id AS mq_leak FROM claim WHERE mq_leak = 1",
                "SELECT count(*) AS mq_leak FROM claim WHERE mq_leak > 0",
                "SELECT id AS ORA_ROWSCN FROM claim WHERE ORA_ROWSCN IS NOT NULL"):
        verdict = _decide(sql, "oracle")
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    assert isinstance(_decide("SELECT id AS mq_leak FROM claim ORDER BY mq_leak", "oracle"), Approved)


def test_an_engine_not_listed_reads_the_alias_as_a_bare_order_by_item_alone():
    assert isinstance(_decide("SELECT id AS x FROM claim ORDER BY x", "snowflake"), Approved)
    assert isinstance(_decide("SELECT id AS x FROM claim WHERE x = 1", "snowflake"), Refusal)
    assert isinstance(_decide("SELECT id AS x FROM claim ORDER BY x + 0", "snowflake"), Refusal)


def test_postgres_reads_an_output_name_inside_an_expression_as_an_input_column():
    # Documented, and the M164 class: `ORDER BY xmin + 0` would read the system column there.
    verdict = _decide("SELECT id AS xmin FROM claim ORDER BY xmin + 0", "postgres")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert isinstance(_decide("SELECT id AS xmin FROM claim ORDER BY xmin", "postgres"), Approved)


def test_a_set_operations_order_by_still_names_the_first_branchs_output():
    sql = "SELECT id AS x FROM claim UNION ALL SELECT id FROM claim ORDER BY x"
    for engine in sorted(READS):
        assert isinstance(_decide(sql, engine), Approved), engine

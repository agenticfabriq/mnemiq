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


def _decide(sql, dialect):
    return decide(sql, VISIBLE, dialect=dialect, target=dialect, policy=None)


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

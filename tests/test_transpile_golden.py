"""Transpile fidelity + guard false-positives, over the 99 TPC-DS benchmark queries.

A transpile bug does not crash: it returns a different query that runs fine and gives the
wrong answer. This is the guard against that.
"""

import duckdb
import pytest
import sqlglot
from sqlglot import exp

from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.guard import check_shape
from mnemiq.sql.verdict import Refusal


@pytest.fixture(scope="module")
def tpcds_queries() -> list[tuple[int, str]]:
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds;")
    return con.execute("SELECT query_nr, query FROM tpcds_queries() ORDER BY query_nr").fetchall()


def test_the_golden_set_is_the_whole_benchmark(tpcds_queries):
    assert len(tpcds_queries) == 99


def test_the_guard_accepts_every_real_analytic_query(tpcds_queries):
    """A guard that rejects legitimate SQL is broken, not safe."""
    rejected = []
    for nr, query in tpcds_queries:
        result = check_shape(query, dialect="duckdb")
        if isinstance(result, Refusal):
            rejected.append((nr, result.code, result.message))
    assert not rejected, f"the guard rejected legitimate analytics SQL: {rejected}"


def test_the_guard_always_imposes_a_limit(tpcds_queries):
    for nr, query in tpcds_queries:
        ast = check_shape(query, dialect="duckdb")
        assert ast.args.get("limit") is not None, f"q{nr} escaped without a row limit"


def test_the_authorization_guard_accepts_every_real_analytic_query(tpcds_queries):
    """The access guard must not reject valid SQL either -- only unauthorized SQL.

    Given the real TPC-DS schema as the visible schema, all 99 queries must pass. This is
    what catches false positives like rejecting a GROUP BY on a SELECT alias, which would
    make the engine unable to answer "count X by year".
    """
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf=0);")  # schema only, no rows
    visible: dict[str, set[str]] = {}
    for table, column in con.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main'"
    ).fetchall():
        visible.setdefault(table, set()).add(column)
    assert visible, "TPC-DS schema did not materialize"

    rejected = []
    for nr, query in tpcds_queries:
        ast = check_shape(query, dialect="duckdb")
        assert not isinstance(ast, Refusal), f"q{nr} failed the shape guard"
        refusal = check_access(ast, visible)
        if refusal is not None:
            rejected.append((nr, refusal.code, refusal.subject))
    assert not rejected, f"the access guard rejected authorized analytics SQL: {rejected}"


def test_every_query_transpiles_and_keeps_its_tables(tpcds_queries):
    """Fidelity: the postgres query must reference exactly the tables the duckdb one did."""
    for nr, query in tpcds_queries:
        ast = sqlglot.parse_one(query, read="duckdb")
        target = sqlglot.transpile(ast.sql(dialect="duckdb"), read="duckdb", write="postgres")[0]

        reparsed = sqlglot.parse_one(target, read="postgres")
        before = {t.name for t in ast.find_all(exp.Table)}
        after = {t.name for t in reparsed.find_all(exp.Table)}
        assert before == after, f"q{nr} lost or gained tables in transpile: {before ^ after}"

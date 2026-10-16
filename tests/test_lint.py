import sqlglot

from mnemiq.sql.lint import Violation, lint


def _ast(sql, dialect="duckdb"):
    return sqlglot.parse_one(sql, read=dialect)


# Where NULLs sort is the engine's (M161): Oracle puts them FIRST descending, SQLite ascending, and
# DuckDB last both ways. `test_lint_null_order.py` pins each engine's order and the decider's use.
def test_ordered_limit_without_null_guard_fires_where_nulls_sort_first():
    v = lint(_ast("SELECT name FROM t ORDER BY score DESC FETCH FIRST 1 ROWS ONLY", "oracle"))
    assert isinstance(v, Violation) and v.code == "null_before_ordered_limit"
    assert "NULLS LAST" in v.message and "IS NOT NULL" not in v.message


def test_ordered_limit_with_null_guard_does_not_fire():
    assert lint(_ast("SELECT name FROM t WHERE score IS NOT NULL ORDER BY score DESC "
                     "FETCH FIRST 1 ROWS ONLY", "oracle")) is None


def test_an_order_that_puts_nulls_last_does_not_fire():
    assert lint(_ast("SELECT name FROM t ORDER BY score LIMIT 1")) is None  # DuckDB: last
    assert lint(_ast("SELECT name FROM t ORDER BY score DESC LIMIT 1")) is None  # both ways
    assert lint(_ast("SELECT name FROM t ORDER BY score FETCH FIRST 1 ROWS ONLY", "oracle")) is None


def test_order_by_without_limit_does_not_fire():
    assert lint(_ast("SELECT name FROM t ORDER BY score DESC", "oracle")) is None


def test_plain_aggregate_does_not_fire():
    assert lint(_ast("SELECT count(*) AS n FROM t")) is None


def test_join_on_with_or_fires():
    v = lint(_ast("SELECT count(*) FROM a JOIN b ON a.id = b.id OR a.id = b.alt"))
    assert isinstance(v, Violation) and v.code == "join_or_fanout"


def test_plain_equality_join_does_not_fire():
    assert lint(_ast("SELECT count(*) FROM a JOIN b ON a.id = b.id")) is None

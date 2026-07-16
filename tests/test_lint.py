import sqlglot

from mnemiq.sql.lint import Violation, lint


def _ast(sql):
    return sqlglot.parse_one(sql, read="duckdb")


def test_ordered_limit_without_null_guard_fires():
    v = lint(_ast("SELECT name FROM t ORDER BY score LIMIT 1"))
    assert isinstance(v, Violation) and v.code == "null_before_ordered_limit"
    assert "IS NOT NULL" in v.message


def test_ordered_limit_with_null_guard_does_not_fire():
    assert lint(_ast("SELECT name FROM t WHERE score IS NOT NULL ORDER BY score LIMIT 1")) is None


def test_order_by_desc_does_not_fire():
    # DESC sorts NULLs last, so the LIMIT picks a real value -- not the trap
    assert lint(_ast("SELECT name FROM t ORDER BY score DESC LIMIT 1")) is None


def test_order_by_without_limit_does_not_fire():
    assert lint(_ast("SELECT name FROM t ORDER BY score")) is None


def test_plain_aggregate_does_not_fire():
    assert lint(_ast("SELECT count(*) AS n FROM t")) is None


def test_join_on_with_or_fires():
    v = lint(_ast("SELECT count(*) FROM a JOIN b ON a.id = b.id OR a.id = b.alt"))
    assert isinstance(v, Violation) and v.code == "join_or_fanout"


def test_plain_equality_join_does_not_fire():
    assert lint(_ast("SELECT count(*) FROM a JOIN b ON a.id = b.id")) is None

import sqlglot

from mnemiq.sql.values_check import check_values
from mnemiq.sql.verdict import Refusal, RefusalCode


class _FakeIndex:
    def __init__(self, data):  # data: {(table, column): set(values)}
        self._data = data

    def has(self, table, column):
        return (table, column) in self._data

    def contains(self, table, column, literal):
        return literal in self._data.get((table, column), set())

    def nearest(self, table, column, literal, k=8):
        return sorted(self._data.get((table, column), set()))[:k]


VISIBLE = {"gasstations": {"Country", "Segment"}}
IDX = _FakeIndex({("gasstations", "Country"): {"Czech Republic", "Slovakia"}})


def _ast(sql):
    return sqlglot.parse_one(sql, read="duckdb")


def test_missing_literal_is_flagged_with_the_nearest_values():
    v = check_values(_ast("SELECT Country FROM gasstations WHERE Country = 'CZE'"), VISIBLE, IDX)
    assert isinstance(v, Refusal) and v.code == RefusalCode.VALUE_GROUNDING
    assert "Czech Republic" in v.message and "gasstations.Country" in v.message


def test_a_present_literal_passes():
    sql = "SELECT Country FROM gasstations WHERE Country = 'Czech Republic'"
    assert check_values(_ast(sql), VISIBLE, IDX) is None


def test_an_unindexed_column_passes():
    assert check_values(_ast("SELECT * FROM gasstations WHERE Segment = 'X'"), VISIBLE, IDX) is None


def test_a_like_predicate_passes():
    sql = "SELECT * FROM gasstations WHERE Country LIKE 'CZ%'"
    assert check_values(_ast(sql), VISIBLE, IDX) is None


def test_a_numeric_literal_passes():
    assert check_values(_ast("SELECT * FROM gasstations WHERE Country = 5"), VISIBLE, IDX) is None


def test_an_in_list_flags_a_missing_member():
    sql = "SELECT * FROM gasstations WHERE Country IN ('CZE', 'Slovakia')"
    v = check_values(_ast(sql), VISIBLE, IDX)
    assert isinstance(v, Refusal) and v.code == RefusalCode.VALUE_GROUNDING


def test_an_ambiguous_unqualified_column_is_skipped():
    visible = {"a": {"Country"}, "b": {"Country"}}
    idx = _FakeIndex({("a", "Country"): {"X"}, ("b", "Country"): {"Y"}})
    sql = "SELECT * FROM a JOIN b ON a.id = b.id WHERE Country = 'Z'"
    assert check_values(_ast(sql), visible, idx) is None

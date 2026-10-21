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


# The BIRD world these shapes come from runs on SQLite, which folds a stored `Country` and an
# unquoted reference alike; the check reads names as the engine it is told resolves them (#178).
def test_missing_literal_is_flagged_with_the_nearest_values():
    v = check_values(_ast("SELECT Country FROM gasstations WHERE Country = 'CZE'"), VISIBLE, IDX, dialect="sqlite")
    assert isinstance(v, Refusal) and v.code == RefusalCode.VALUE_GROUNDING
    assert "Czech Republic" in v.message and "gasstations.Country" in v.message


def test_a_present_literal_passes():
    sql = "SELECT Country FROM gasstations WHERE Country = 'Czech Republic'"
    assert check_values(_ast(sql), VISIBLE, IDX, dialect="sqlite") is None


def test_an_unindexed_column_passes():
    assert check_values(_ast("SELECT * FROM gasstations WHERE Segment = 'X'"), VISIBLE, IDX, dialect="sqlite") is None


def test_a_like_predicate_passes():
    sql = "SELECT * FROM gasstations WHERE Country LIKE 'CZ%'"
    assert check_values(_ast(sql), VISIBLE, IDX, dialect="sqlite") is None


def test_a_numeric_literal_passes():
    assert check_values(_ast("SELECT * FROM gasstations WHERE Country = 5"), VISIBLE, IDX, dialect="sqlite") is None


def test_an_in_list_flags_a_missing_member():
    sql = "SELECT * FROM gasstations WHERE Country IN ('CZE', 'Slovakia')"
    v = check_values(_ast(sql), VISIBLE, IDX, dialect="sqlite")
    assert isinstance(v, Refusal) and v.code == RefusalCode.VALUE_GROUNDING


def test_an_ambiguous_unqualified_column_is_skipped():
    visible = {"a": {"Country"}, "b": {"Country"}}
    idx = _FakeIndex({("a", "Country"): {"X"}, ("b", "Country"): {"Y"}})
    sql = "SELECT * FROM a JOIN b ON a.id = b.id WHERE Country = 'Z'"
    assert check_values(_ast(sql), visible, idx) is None


def test_a_document_among_the_nearest_values_is_cut_in_the_refusal():
    # M139: the refusal goes back to the model as repair feedback, and quoted up to 8 of nova's
    # 2,975-character JSON values whole -- the overflow the card no longer causes.
    doc = '{"cells": [' + ", ".join('{"id": %d}' % i for i in range(300)) + "]}"
    idx = _FakeIndex({("t", "doc"): {doc, "short"}})
    v = check_values(_ast("SELECT doc FROM t WHERE doc = 'x'"), {"t": {"doc"}}, idx)
    assert isinstance(v, Refusal) and v.code == RefusalCode.VALUE_GROUNDING
    assert doc not in v.message and doc[:200] in v.message
    assert "never compare with it" in v.message and "short" in v.message

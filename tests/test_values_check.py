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


# --- repair_hints: a value absent from its column, named where it does appear (fix 3a) ------------------------------
#
# Under `hints`, a refused value is looked up in the other indexed columns the caller may read (`locatable`, built by
# `sql.locatable` from the caller's grants), and the refusal says where it is. Only `locatable`'s columns are ever
# named: the index is unfiltered (M5), so a column outside it could disclose a value the caller cannot read.

_HINT_VISIBLE = {"orders": {"id", "status", "region", "lane"}, "shipments": {"id", "state"}}
_HINT_INDEX = _FakeIndex({("orders", "status"): {"open", "closed"}, ("orders", "region"): {"west", "east"},
                          ("orders", "lane"): {"west"}, ("shipments", "state"): {"west", "pending"}})
_ALL = tuple(sorted((t, c) for t, cs in _HINT_VISIBLE.items() for c in cs))
_TODAY_WEST = "the value 'west' does not appear in orders.status; the real values include: closed, open. Use the intended one."


def _hinted(sql, locatable=_ALL, **kw):
    return check_values(_ast(sql), _HINT_VISIBLE, _HINT_INDEX, dialect="duckdb", hints=True, locatable=locatable, **kw)


def test_a_value_in_another_column_of_the_same_table_is_named_first():
    v = _hinted("SELECT id FROM orders WHERE status = 'west'")
    assert v.message == (_TODAY_WEST + " It does appear in orders.lane, orders.region and shipments.state: if that "
                         "column is the one meant, filter on it instead.")


def test_a_value_in_another_table_is_named():
    v = _hinted("SELECT id FROM orders WHERE status = 'pending'")
    assert v.message.endswith(" It does appear in shipments.state: if that column is the one meant, filter on it "
                              "instead.")


def test_only_the_columns_the_caller_may_locate_are_named():
    """A column the index holds the value in, left out of `locatable` (masked, row-filtered, denied or a view), is
    never named, while one beside it is."""
    v = _hinted("SELECT id FROM orders WHERE status = 'west'", locatable=(("orders", "region"), ("orders", "status")))
    assert v.message.endswith(" It does appear in orders.region: if that column is the one meant, filter on it "
                              "instead.")
    assert "orders.lane" not in v.message and "shipments.state" not in v.message


def test_at_most_three_columns_are_named():
    many = _FakeIndex({("orders", "status"): {"open"}, **{("wide", f"c{i}"): {"west"} for i in range(5)}})
    loc = tuple(("wide", f"c{i}") for i in range(5))
    v = check_values(_ast("SELECT id FROM orders WHERE status = 'west'"),
                     {"orders": {"id", "status"}, "wide": {f"c{i}" for i in range(5)}}, many, dialect="duckdb",
                     hints=True, locatable=loc)
    assert " It does appear in wide.c0, wide.c1 and wide.c2: " in v.message


def test_no_sentence_where_no_other_column_holds_it():
    v = _hinted("SELECT id FROM orders WHERE status = 'nowhere'")
    assert "It does appear" not in v.message


def test_a_row_filtered_refusal_is_unchanged():
    sql = "SELECT id FROM orders WHERE status = 'west'"
    assert _hinted(sql, row_filtered={"orders"}).message == check_values(
        _ast(sql), _HINT_VISIBLE, _HINT_INDEX, dialect="duckdb", row_filtered={"orders"}).message


def test_off_keeps_todays_message():
    v = check_values(_ast("SELECT id FROM orders WHERE status = 'west'"), _HINT_VISIBLE, _HINT_INDEX, dialect="duckdb",
                     locatable=_ALL)
    assert v.message == _TODAY_WEST

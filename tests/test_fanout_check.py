"""The fan-out check (register M109): refuse an aggregate whose rows a join has multiplied."""
from __future__ import annotations

from mnemiq.contract import Column, Snapshot
from mnemiq.sql.fanout_check import key_facts


def test_a_key_is_unique_when_every_non_null_value_is_distinct():
    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="t.a", object_id="t", name="a", row_count=10, distinct_count=10, null_count=0),
        Column(id="t.b", object_id="t", name="b", row_count=10, distinct_count=7, null_count=3),
        Column(id="t.c", object_id="t", name="c", row_count=10, distinct_count=6, null_count=3),
    ])
    assert key_facts(snap) == {("t", "a"): True, ("t", "b"): True, ("t", "c"): False}


def test_an_unmeasured_column_is_absent_not_guessed():
    """`distinct_count=None` is what profiling writes for a column it could not count (a LOB, a
    user-defined type). Reading that as either answer would make the guard guess."""
    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="t.a", object_id="t", name="a", row_count=10, distinct_count=None,
               null_count=None),
        Column(id="t.b", object_id="t", name="b"),
    ])
    assert key_facts(snap) == {}

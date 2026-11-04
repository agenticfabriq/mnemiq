"""A word the engine reads as a column is checked against the live table, not only the snapshot (M225).

A row filter or a statement holding bare a word the engine reads as a same-named column (M194's niladic
words, and TRUE and FALSE on SQLite and Oracle) was refused only where the snapshot's schema listed that
column (#202). A column the live table holds but the snapshot does not, from drift after introspection or
a column the snapshot left out, was approved, and the engine read it. So where such a word is present, the
decider asks the source which columns its tables hold now, and refuses on those too. Only then: an
ordinary statement asks nothing. The live columns only ever refuse; they never grant.
"""

from __future__ import annotations

import duckdb

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

# The snapshot knows only id; the live table has gained current_catalog since.
SNAPSHOT = {"kw": {"id"}, "plain": {"id"}}


class _Live:
    """A source whose tables hold what `live` says, and which proves any statement it is asked to."""
    dialect = "duckdb"

    def __init__(self, live):
        self.live, self.asked = live, 0

    def list_columns(self):
        self.asked += 1
        return [(table, column, "VARCHAR") for table, columns in self.live.items() for column in columns]

    def execute(self, sql):
        return []


def _engine():
    con = duckdb.connect()
    con.execute('CREATE TABLE kw (id INT, "current_catalog" VARCHAR)')
    con.execute("INSERT INTO kw VALUES (1, 'allow'), (2, 'deny')")
    return con


def test_the_premise_the_engine_reads_the_live_column():
    assert _engine().execute("SELECT id FROM kw WHERE current_catalog = 'allow'").fetchall() == [(1,)]


def test_a_filter_naming_a_column_only_the_live_table_holds_is_refused():
    source = _Live({"kw": {"id", "current_catalog"}, "plain": {"id"}})
    verdict = decide("SELECT id FROM kw", SNAPSHOT, adapter=source, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=SNAPSHOT, row_filters={"kw": "current_catalog = 'allow'"}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_statement_reading_a_column_only_the_live_table_holds_is_refused():
    source = _Live({"kw": {"id", "current_catalog"}, "plain": {"id"}})
    verdict = decide("SELECT id FROM kw WHERE current_catalog = 'allow'", SNAPSHOT, adapter=source,
                     dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=SNAPSHOT))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_a_write_s_filter_and_its_own_words_are_checked_against_the_live_table_too():
    source = _Live({"kw": {"id", "current_catalog"}, "plain": {"id"}})
    grants = GrantSet(frozenset(SNAPSHOT), writable=frozenset(SNAPSHOT), row_filters={"kw": "current_catalog = 'allow'"})
    verdict = decide_write("DELETE FROM kw WHERE id > 0", SNAPSHOT, grants, adapter=source, dialect="duckdb",
                           target="duckdb", writes_enabled=True,
                           policy=AccessPolicy(policy_schema=SNAPSHOT, row_filters={"kw": "current_catalog = 'allow'"}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_where_the_live_table_has_no_such_column_the_word_is_the_function():
    source = _Live({"kw": {"id", "d"}, "plain": {"id"}})
    verdict = decide("SELECT id FROM kw", {"kw": {"id", "d"}}, adapter=source, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema={"kw": {"id", "d"}},
                                         row_filters={"kw": "d > current_date - INTERVAL 30 DAY"}))
    assert isinstance(verdict, Approved), verdict


def test_an_ordinary_statement_does_not_ask_the_source():
    source = _Live({"kw": {"id", "current_catalog"}, "plain": {"id"}})
    verdict = decide("SELECT id FROM kw WHERE id = 1", SNAPSHOT, adapter=source, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=SNAPSHOT))
    assert isinstance(verdict, Approved), verdict
    assert source.asked == 0, "the live columns were asked for a statement holding no such word"


def test_the_live_columns_never_grant():
    # A live column the caller does not hold stays withheld: asking the source only ever refuses.
    source = _Live({"kw": {"id", "secret"}, "plain": {"id"}})
    verdict = decide("SELECT secret FROM kw", SNAPSHOT, adapter=source, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=SNAPSHOT))
    assert isinstance(verdict, Refusal), verdict


def test_with_no_source_the_snapshot_alone_decides():
    # No adapter, as in some callers: the check falls back to the snapshot, the behaviour before this.
    verdict = decide("SELECT id FROM kw", SNAPSHOT, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=SNAPSHOT, row_filters={"kw": "current_catalog = 'allow'"}))
    assert isinstance(verdict, Approved), verdict


def test_a_source_that_cannot_list_its_columns_refuses_a_statement_holding_such_a_word():
    # Fail-closed: where such a word is present and the source cannot say which columns it holds, nothing
    # can tell the column from the function.
    class _Silent(_Live):
        def list_columns(self):
            raise RuntimeError("the catalogue could not be read")
    verdict = decide("SELECT id FROM kw", SNAPSHOT, adapter=_Silent({}), dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=SNAPSHOT, row_filters={"kw": "current_catalog = 'allow'"}))
    assert isinstance(verdict, Refusal), verdict


def test_a_write_against_a_source_that_cannot_list_its_columns_is_refused():
    # The write path fails closed too: where a word reads as a column and the source cannot say which
    # columns it holds, nothing tells the column from the function.
    class _Silent(_Live):
        def list_columns(self):
            raise RuntimeError("the catalogue could not be read")
    grants = GrantSet(frozenset(SNAPSHOT), writable=frozenset(SNAPSHOT),
                      row_filters={"kw": "current_catalog = 'allow'"})
    verdict = decide_write("DELETE FROM kw WHERE id > 0", SNAPSHOT, grants, adapter=_Silent({}),
                           dialect="duckdb", target="duckdb", writes_enabled=True,
                           policy=AccessPolicy(policy_schema=SNAPSHOT, row_filters={"kw": "current_catalog = 'allow'"}))
    assert isinstance(verdict, Refusal), verdict


def test_a_generated_column_the_live_table_holds_is_refused_on_sqlite():
    # SQLite's list_columns reads PRAGMA table_xinfo, so a GENERATED column, which table_info omits but
    # the engine reads by name, is in the live inventory (Codex's review of this change, M225).
    import sqlite3
    from mnemiq.adapters.sqlite import SQLiteAdapter
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE kw (id INT, secret INT, "true" INT GENERATED ALWAYS AS (secret) VIRTUAL)')
    con.execute("INSERT INTO kw (id, secret) VALUES (1, 123), (2, 456)")
    adapter = SQLiteAdapter.__new__(SQLiteAdapter)
    adapter._con = con
    assert "true" in {c for _t, c, _ in adapter.list_columns()}  # the premise: xinfo lists it
    snapshot, schema = {"kw": {"id"}}, {"kw": {"id", "secret"}}
    read = decide("SELECT true FROM kw", snapshot, adapter=adapter, dialect="sqlite", target="sqlite",
                  policy=AccessPolicy(policy_schema=schema))
    assert isinstance(read, Refusal) and read.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, read
    filtered = decide("SELECT id FROM kw", snapshot, adapter=adapter, dialect="sqlite", target="sqlite",
                      policy=AccessPolicy(policy_schema=schema, row_filters={"kw": "true = 123"}))
    assert isinstance(filtered, Refusal) and filtered.code is RefusalCode.INVALID_ROW_FILTER, filtered

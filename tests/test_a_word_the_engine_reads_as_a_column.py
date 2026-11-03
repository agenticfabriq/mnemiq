"""A bare word the parse reads as a literal or a function, which the engine reads as a same-named
column, is refused where a table in scope has that column (M230, M225).

M194 refused DuckDB's seven niladic words in a caller's statement. Two gaps stayed beside it:

- SQLite and Oracle read a bare `TRUE` or `FALSE` as a column of the table read where it has one
  (measured: SQLite 3.50.4, and Oracle Free 23 live), and sqlglot reads the boolean literal, so no
  grant check saw the column: `SELECT true FROM kw` returned a withheld column's value. DuckDB and
  Postgres read the literal (measured), so the words are listed per engine (M230).
- A row filter runs as its operator wrote it (M222), and the rule never read one: a filter
  `current_catalog = 'allow'` on DuckDB filtered by a column the checks never saw as one (M225).
"""

from __future__ import annotations

import os
import sqlite3

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

LITERAL = ("true", "false")
NILADIC = ("current_date", "current_timestamp", "current_time", "localtime", "localtimestamp",
           "session_user", "current_catalog")


def _sqlite(columns=LITERAL):
    """kw holds a column named after each word, valued 'withheld'; plain holds none."""
    con = sqlite3.connect(":memory:")
    named = ", ".join(f'"{c}" TEXT' for c in columns)
    con.execute(f"CREATE TABLE kw (id INT, {named})")
    con.execute(f"INSERT INTO kw VALUES (1, {', '.join(repr('withheld') for _ in columns)})")
    con.execute("CREATE TABLE plain (id INT)")
    con.execute("INSERT INTO plain VALUES (1)")
    return con


SQLITE_SCHEMA = {"kw": {"id", *LITERAL}, "plain": {"id"}}
SQLITE_HIDDEN = {"kw": {"id"}, "plain": {"id"}}


@pytest.mark.parametrize("sql", [
    "SELECT true FROM kw",
    "SELECT false AS f FROM kw",
    "SELECT id FROM kw WHERE true = 'withheld'",
    "SELECT id FROM kw WHERE false LIKE 'w%'",
    "SELECT d.t FROM (SELECT true AS t FROM kw) AS d",
    "WITH c AS (SELECT id, false AS f FROM kw) SELECT f FROM c",
    "SELECT id FROM kw ORDER BY true",
])
def test_on_sqlite_a_bare_true_or_false_is_refused_where_the_table_read_has_the_column(sql):
    verdict = decide(sql, SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


@pytest.mark.parametrize("sql, rows", [
    ("SELECT id AS true FROM kw", [(1,)]),                       # an output name, not a read
    ("SELECT t.id FROM kw AS t", [(1,)]),
    ("WITH true AS (SELECT id FROM kw) SELECT id FROM true", [(1,)]),
])
def test_a_word_declared_as_a_name_is_no_read_of_the_column(sql, rows):
    # Matched by offset: an alias or a CTE's name is an identifier where it stands (Codex's second
    # review of this change: counted against column references alone, `AS true` was refused).
    verdict = decide(sql, SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Approved), verdict
    assert _sqlite().execute(verdict.target_sql).fetchall() == rows


def test_an_alias_beside_a_bare_read_still_refuses_the_read():
    verdict = decide("SELECT id AS true, true AS t FROM kw", SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_the_control_sqlite_reads_the_column_and_sqlglot_reads_the_literal():
    con = _sqlite()
    assert con.execute("SELECT true FROM kw").fetchall() == [("withheld",)]
    assert con.execute("SELECT true FROM plain").fetchall() == [(1,)]


def test_where_no_table_read_has_the_column_true_is_the_literal():
    verdict = decide("SELECT id FROM plain WHERE true", SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Approved), verdict
    assert _sqlite().execute(verdict.target_sql).fetchall() == [(1,)]


def test_the_advice_names_the_column_qualified_and_a_comparison_for_the_value():
    verdict = decide("SELECT true FROM kw", SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Refusal), verdict
    assert 'kw."true"' in verdict.message and "(1 = 1)" in verdict.message, verdict.message
    # Followed for the column, the quoted name is a column the grant check reads: withheld here,
    # and the held column's value where granted. Unquoted, `kw.true` is no column to the parse.
    qualified = decide('SELECT kw."true" FROM kw', SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                       policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(qualified, Refusal) and qualified.code is RefusalCode.UNKNOWN_COLUMN, qualified
    granted = decide('SELECT kw."true" FROM kw', SQLITE_SCHEMA, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(granted, Approved), granted
    assert _sqlite().execute(granted.target_sql).fetchall() == [("withheld",)]
    # Followed for the value, the comparison is approved and is the literal.
    value = decide("SELECT (1 = 1) AS t FROM kw", SQLITE_HIDDEN, dialect="sqlite", target="sqlite",
                   policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(value, Approved), value
    assert _sqlite().execute(value.target_sql).fetchall() == [(1,)]


def test_the_write_path_refuses_the_same_words_on_sqlite():
    grants = GrantSet(frozenset(SQLITE_SCHEMA), writable=frozenset({"plain"}))
    verdict = decide_write("UPDATE plain SET id = 2 WHERE id IN (SELECT id FROM kw WHERE true = 'withheld')",
                           SQLITE_HIDDEN, grants, dialect="sqlite", target="sqlite", writes_enabled=True,
                           policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


@pytest.mark.parametrize("sql", ["SELECT TRUE FROM KW", "SELECT ID FROM KW WHERE FALSE = 'withheld'"])
def test_on_oracle_a_bare_true_or_false_is_refused_where_the_table_read_has_the_column(sql):
    # Oracle Free 23 read both as the column, live (the register); decided here as Oracle stores it.
    verdict = decide(sql, {"KW": {"ID"}}, dialect="oracle", target="oracle",
                     policy=AccessPolicy(policy_schema={"KW": {"ID", "TRUE", "FALSE"}}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict



ORACLE_DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not ORACLE_DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_live_true_reads_the_column_and_the_rule_refuses_it():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=ORACLE_DSN)
    cur = con.cursor()
    table = "M230_KW"
    schema = {table: {"ID", "TRUE", "FALSE"}}
    try:
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        except oracledb.DatabaseError:
            pass
        cur.execute(f'CREATE TABLE {table} (ID NUMBER, "TRUE" VARCHAR2(20), "FALSE" VARCHAR2(20))')
        cur.execute(f"INSERT INTO {table} VALUES (1, 'withheld', 'withheld')")
        assert cur.execute(f"SELECT TRUE, FALSE FROM {table}").fetchall() == [("withheld", "withheld")]  # the premise
        for sql in (f"SELECT TRUE FROM {table}", f"SELECT ID FROM {table} WHERE FALSE = 'withheld'"):
            verdict = decide(sql, {table: {"ID"}}, dialect="oracle", target="oracle",
                             policy=AccessPolicy(policy_schema=schema))
            assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
        # The repair, followed where the column is granted, reads it.
        granted = decide(f'SELECT {table}."TRUE" FROM {table}', schema, dialect="oracle", target="oracle",
                         policy=AccessPolicy(policy_schema=schema))
        assert isinstance(granted, Approved), granted
        assert cur.execute(granted.target_sql).fetchall() == [("withheld",)]
        filtered = decide(f"SELECT ID FROM {table}", schema, dialect="oracle", target="oracle",
                          policy=AccessPolicy(policy_schema=schema, row_filters={table: "TRUE = 'withheld'"}))
        assert isinstance(filtered, Refusal) and filtered.code is RefusalCode.INVALID_ROW_FILTER, filtered
    finally:
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        finally:
            con.close()

@pytest.mark.parametrize("engine", ["duckdb", "postgres"])
def test_where_the_engine_reads_the_literal_the_words_are_not_refused(engine):
    # The words are listed per engine: DuckDB and Postgres read TRUE as the literal beside a column
    # named true (measured), so refusing there would refuse a correct statement.
    verdict = decide("SELECT id FROM kw WHERE true", SQLITE_HIDDEN, dialect=engine, target=engine,
                     policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Approved), verdict
    if engine == "duckdb":
        con = duckdb.connect()
        con.execute('CREATE TABLE kw (id INT, "true" VARCHAR, "false" VARCHAR)')
        con.execute("INSERT INTO kw VALUES (1, 'withheld', 'withheld')")
        assert con.execute("SELECT true FROM kw").fetchall() == [(True,)]  # the literal, measured
        assert con.execute(verdict.target_sql).fetchall() == [(1,)]


# --- Row filters (M225) -------------------------------------------------------------------------

DUCK_SCHEMA = {"kw": {"id", *NILADIC}, "ent": {"id", "current_date"}, "plain": {"id", "d"}}


def _duck():
    con = duckdb.connect()
    named = ", ".join(f'"{n}" VARCHAR' for n in NILADIC)
    con.execute(f"CREATE TABLE kw (id INT, {named})")
    con.execute(f"INSERT INTO kw VALUES (1, {', '.join(repr('allow') for _ in NILADIC)}), "
                f"(2, {', '.join(repr('deny') for _ in NILADIC)})")
    return con


def _filtered(table, filt, schema):
    return AccessPolicy(row_filters={table: filt}, policy_schema=schema)


@pytest.mark.parametrize("word", NILADIC)
def test_a_row_filter_naming_a_niladic_word_its_table_has_as_a_column_is_refused(word):
    filt = f"{word} = 'allow'"
    assert _duck().execute(f"SELECT id FROM kw WHERE {filt}").fetchall() == [(1,)]  # the engine reads the column
    verdict = decide("SELECT id FROM kw", DUCK_SCHEMA, dialect="duckdb", target="duckdb",
                     policy=_filtered("kw", filt, DUCK_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_the_filtered_table_s_column_counts_whether_the_caller_holds_it_or_not():
    # Withheld from the caller, listed by the policy: the engine binds the column all the same.
    withheld = decide("SELECT id FROM kw", {"kw": {"id"}}, dialect="duckdb", target="duckdb",
                      policy=_filtered("kw", "session_user = 'allow'", DUCK_SCHEMA))
    assert isinstance(withheld, Refusal) and withheld.code is RefusalCode.INVALID_ROW_FILTER, withheld
    # No policy schema at all: the caller's own columns are what is known of the table.
    unlisted = decide("SELECT id FROM kw", {"kw": {"id", "session_user"}}, dialect="duckdb", target="duckdb",
                      policy=AccessPolicy(row_filters={"kw": "session_user = 'allow'"}))
    assert isinstance(unlisted, Refusal) and unlisted.code is RefusalCode.INVALID_ROW_FILTER, unlisted


@pytest.mark.parametrize("filt", [
    "$$'$$ != 'x' AND current_catalog = 'allow'",      # a quote inside a dollar-quoted string
    "$t$'$t$ != 'x' AND localtime = 'allow'",
    "/* ' */ current_date = 'allow'",                  # inside a comment
    "-- '\ncurrent_time = 'allow'",
    "E'\\'' <> 'x' AND session_user = 'allow'",       # an escaped quote
    "'it''s' <> 'x' AND current_catalog = 'allow'",
])
def test_a_quote_inside_a_string_or_a_comment_does_not_hide_the_word(filt):
    # Read by the tokenizer, as the parse is: a regex over the text took the quote inside `$$'$$`
    # for a string's start and cut the bare word out (Codex's first review of this change).
    assert _duck().execute(f"SELECT id FROM kw WHERE {filt}").fetchall() == [(1,)]  # the engine reads the column
    verdict = decide("SELECT id FROM kw", DUCK_SCHEMA, dialect="duckdb", target="duckdb",
                     policy=_filtered("kw", filt, DUCK_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_row_filter_s_subquery_is_read_against_its_own_tables():
    filt = "id IN (SELECT id FROM ent WHERE current_date = 'allow')"
    verdict = decide("SELECT id FROM plain", DUCK_SCHEMA, dialect="duckdb", target="duckdb",
                     policy=_filtered("plain", filt, DUCK_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_row_filter_where_no_table_in_scope_has_the_column_reads_the_function():
    filt = "d > current_date - INTERVAL 30 DAY"
    verdict = decide("SELECT id FROM plain", DUCK_SCHEMA, dialect="duckdb", target="duckdb",
                     policy=_filtered("plain", filt, DUCK_SCHEMA))
    assert isinstance(verdict, Approved), verdict


def test_a_quoted_niladic_column_in_a_filter_is_a_column_and_passes():
    for filt in ('"current_catalog" = \'allow\'', '"localtime" = \'allow\''):
        verdict = decide("SELECT id FROM kw", DUCK_SCHEMA, dialect="duckdb", target="duckdb",
                         policy=_filtered("kw", filt, DUCK_SCHEMA))
        assert isinstance(verdict, Approved), (filt, verdict)
        assert _duck().execute(verdict.target_sql).fetchall() == [(1,)], filt


def test_a_row_filter_naming_true_on_sqlite_is_refused_where_its_table_has_the_column():
    verdict = decide("SELECT id FROM kw", SQLITE_SCHEMA, dialect="sqlite", target="sqlite",
                     policy=_filtered("kw", "true = 'withheld'", SQLITE_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    plain = decide("SELECT id FROM plain", SQLITE_SCHEMA, dialect="sqlite", target="sqlite",
                   policy=_filtered("plain", "true", SQLITE_SCHEMA))
    assert isinstance(plain, Approved), plain


def test_the_write_path_refuses_the_same_filter():
    grants = GrantSet(frozenset(DUCK_SCHEMA), writable=frozenset({"kw", "plain"}), row_filters={"kw": "localtime = 'allow'"})
    for sql in ("DELETE FROM kw WHERE id > 0", "INSERT INTO plain (id) SELECT id FROM kw"):
        verdict = decide_write(sql, DUCK_SCHEMA, grants, dialect="duckdb", target="duckdb", writes_enabled=True,
                               policy=_filtered("kw", "localtime = 'allow'", DUCK_SCHEMA))
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (sql, verdict)


def test_a_filter_on_a_table_without_such_columns_still_runs():
    # The control for the write path: the rule refuses on the columns, not on the shape.
    grants = GrantSet(frozenset(DUCK_SCHEMA), writable=frozenset({"plain"}), row_filters={"plain": "id > 0"})
    verdict = decide_write("DELETE FROM plain WHERE id > 0", DUCK_SCHEMA, grants, dialect="duckdb", target="duckdb",
                           writes_enabled=True, policy=_filtered("plain", "id > 0", DUCK_SCHEMA))
    assert isinstance(verdict, ApprovedWrite), verdict


def test_a_text_that_does_not_parse_counts_every_bare_word():
    # Fail-closed: with no parse there is no name to account for a word, so each bare one is a read.
    from mnemiq.sql.guard import words_read_as_columns
    assert words_read_as_columns("SELECT id AS true FROM", "sqlite", {"true"}) == ["true"]
    assert words_read_as_columns("SELECT id AS true FROM kw", "sqlite", {"true"}) == []  # the control


# --- Written in one dialect, run in another -----------------------------------------------------

CROSS = ("localtimestamp", "localtime", "session_user", "current_catalog")
CROSS_SCHEMA = {"kw": {"id", *CROSS}, "plain": {"id"}}


@pytest.mark.parametrize("word", CROSS)
@pytest.mark.parametrize("shape", ["SELECT {w} FROM kw", "SELECT id FROM kw WHERE {w} = 'withheld'"])
def test_written_as_duckdb_and_run_on_sqlite_a_word_sqlite_reads_as_a_column_is_refused(word, shape):
    # The checks read the DuckDB parse, where the word is a function; SQLite reads the printed word
    # as kw's column. The target's parse holds a column reference there, which the checks never read
    # (Codex's third review of this change: excused by it, all four were approved and answered).
    con = _sqlite(CROSS)
    assert con.execute(f"SELECT {word} FROM kw").fetchall() == [("withheld",)]  # the premise
    verdict = decide(shape.format(w=word), {"kw": {"id"}, "plain": {"id"}}, dialect="duckdb", target="sqlite",
                     policy=AccessPolicy(policy_schema=CROSS_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, (
        verdict, con.execute(verdict.target_sql).fetchall() if isinstance(verdict, Approved) else None)


def test_written_as_duckdb_the_write_path_refuses_the_same():
    grants = GrantSet(frozenset(CROSS_SCHEMA), writable=frozenset({"plain"}))
    verdict = decide_write("UPDATE plain SET id = 2 WHERE id IN (SELECT id FROM kw WHERE localtime = 'withheld')",
                           {"kw": {"id"}, "plain": {"id"}}, grants, dialect="duckdb", target="sqlite",
                           writes_enabled=True, policy=AccessPolicy(policy_schema=CROSS_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_written_as_duckdb_an_alias_is_still_a_name_on_sqlite():
    verdict = decide("SELECT id AS localtime FROM kw", {"kw": {"id"}, "plain": {"id"}}, dialect="duckdb",
                     target="sqlite", policy=AccessPolicy(policy_schema=CROSS_SCHEMA))
    assert isinstance(verdict, Approved), verdict
    assert _sqlite(CROSS).execute(verdict.target_sql).fetchall() == [(1,)]


def test_a_filter_word_the_parse_reads_as_a_column_is_checked_as_one_and_passes():
    # On SQLite, which has no such keyword, sqlglot reads `localtimestamp` as a column, so the
    # validators check it: a column the policy lists, which the filter may name.
    schema = {"kw": {"id", "localtimestamp"}}
    verdict = decide("SELECT id FROM kw", schema, dialect="sqlite", target="sqlite",
                     policy=AccessPolicy(policy_schema=schema, row_filters={"kw": "localtimestamp = 'withheld'"}))
    assert isinstance(verdict, Approved), verdict
    assert _sqlite(("localtimestamp",)).execute(verdict.target_sql).fetchall() == [(1,)]

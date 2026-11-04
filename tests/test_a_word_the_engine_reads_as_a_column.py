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
    schema = {table: {"ID", "TRUE", "FALSE", "SECRET"}}
    try:
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        except oracledb.DatabaseError:
            pass
        cur.execute(f'CREATE TABLE {table} (ID NUMBER, "TRUE" VARCHAR2(20), "FALSE" VARCHAR2(20), SECRET VARCHAR2(20))')
        cur.execute(f"INSERT INTO {table} VALUES (1, 'withheld', 'withheld', 'withheld')")
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
        # M232: the table aliased TRUE, and its withheld SECRET read through the alias.
        aliased = f"SELECT true.SECRET FROM {table} true"
        assert cur.execute(aliased).fetchall() == [("withheld",)]  # the premise
        verdict = decide(aliased, {table: {"ID"}}, dialect="oracle", target="oracle",
                         policy=AccessPolicy(policy_schema=schema))
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
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


def _joined(key_in_plain):
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE kw (id INT, "localtimestamp" TEXT)')
    con.execute("INSERT INTO kw VALUES (1, 'withheld')")
    con.execute('CREATE TABLE plain (id INT, "localtimestamp" TEXT)')
    con.execute(f"INSERT INTO plain VALUES (1, '{key_in_plain}')")
    return con


JOIN_SCHEMA = {"kw": {"id", "localtimestamp"}, "plain": {"id", "localtimestamp"}}


def test_written_as_duckdb_a_using_key_sqlite_reads_as_a_column_is_refused():
    # A USING key is a read of the column on both sides, not a declared name: written as DuckDB it was
    # a function to the checks, and SQLite's join answered on the withheld key (Codex's fourth review).
    sql = "SELECT kw.id FROM kw JOIN plain USING (localtimestamp)"
    assert _joined("withheld").execute(sql).fetchall() == [(1,)]  # the premise: the join reads the key
    assert _joined("other").execute(sql).fetchall() == []
    verdict = decide(sql, {"kw": {"id"}, "plain": {"id"}}, dialect="duckdb", target="sqlite",
                     policy=AccessPolicy(policy_schema=JOIN_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_a_using_key_the_checks_read_as_a_column_passes():
    # Written and run as SQLite, the key is a name to the checks too, and granted: the join stands.
    verdict = decide("SELECT kw.id FROM kw JOIN plain USING (localtimestamp)", JOIN_SCHEMA, dialect="sqlite",
                     target="sqlite", policy=AccessPolicy(policy_schema=JOIN_SCHEMA))
    assert isinstance(verdict, Approved), verdict
    assert _joined("withheld").execute(verdict.target_sql).fetchall() == [(1,)]


# --- A word the parse reads as a literal or a function, standing as a table's alias (M232) -------

ALIASED = {"kw": {"id", "secret"}, "plain": {"id"}, "ent": {"id", "secret"}}
ALIASED_HIDDEN = {"kw": {"id"}, "plain": {"id"}, "ent": {"id"}}
SHAPES = ["SELECT {w}.secret FROM kw AS {w}", "SELECT {w}.secret FROM kw {w}",
          "SELECT id FROM kw {w} WHERE {w}.secret = 'withheld'"]


def _with_secret(engine):
    con = duckdb.connect() if engine == "duckdb" else sqlite3.connect(":memory:")
    con.execute("CREATE TABLE kw (id INT, secret VARCHAR(20))")
    con.execute("INSERT INTO kw VALUES (1, 'withheld')")
    return con


@pytest.mark.parametrize("engine, word", [("duckdb", w) for w in NILADIC] + [("sqlite", w) for w in LITERAL])
@pytest.mark.parametrize("shape", SHAPES)
def test_a_table_aliased_by_a_word_the_parse_reads_otherwise_is_refused(engine, word, shape):
    # The engine reads `w.secret` as kw's column; sqlglot reads a field of the literal or the
    # function, so no check saw `secret`: approved, it returned the withheld value (M232).
    sql = shape.format(w=word)
    ran = _with_secret(engine).execute(sql).fetchall()
    assert ran and ("withheld",) in ran or ran == [(1,)], ran  # the premise: the engine reads the column
    verdict = decide(sql, ALIASED_HIDDEN, dialect=engine, target=engine, policy=AccessPolicy(policy_schema=ALIASED))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


@pytest.mark.parametrize("sql", ["SELECT TRUE.SECRET FROM KW TRUE", "SELECT ID FROM KW FALSE WHERE FALSE.SECRET = 'withheld'"])
def test_on_oracle_a_table_aliased_true_or_false_is_refused(sql):
    verdict = decide(sql, {"KW": {"ID"}}, dialect="oracle", target="oracle",
                     policy=AccessPolicy(policy_schema={"KW": {"ID", "SECRET"}}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_an_ordinary_alias_and_a_field_of_a_column_are_unchanged():
    granted = decide("SELECT c.secret FROM kw AS c", ALIASED, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=ALIASED))
    assert isinstance(granted, Approved), granted
    withheld = decide("SELECT c.secret FROM kw AS c", ALIASED_HIDDEN, dialect="duckdb", target="duckdb",
                      policy=AccessPolicy(policy_schema=ALIASED))
    assert isinstance(withheld, Refusal) and withheld.code is RefusalCode.UNKNOWN_COLUMN, withheld
    con = duckdb.connect()
    con.execute("CREATE TABLE st (id INT, s STRUCT(f VARCHAR))")
    con.execute("INSERT INTO st VALUES (1, {'f': 'x'})")
    field = decide("SELECT st.s.f FROM st", {"st": {"id", "s"}}, dialect="duckdb", target="duckdb")
    assert isinstance(field, Approved), field
    assert con.execute(field.target_sql).fetchall() == [("x",)]


def test_the_write_path_refuses_an_aliased_read_too():
    grants = GrantSet(frozenset(ALIASED), writable=frozenset({"plain"}))
    verdict = decide_write("DELETE FROM plain WHERE id IN (SELECT current_date.id FROM kw AS current_date "
                           "WHERE current_date.secret = 'withheld')", ALIASED_HIDDEN, grants, dialect="duckdb",
                           target="duckdb", writes_enabled=True, policy=AccessPolicy(policy_schema=ALIASED))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_a_row_filter_with_such_an_alias_is_refused():
    filt = "id IN (SELECT localtime.id FROM ent AS localtime WHERE localtime.secret = 'withheld')"
    verdict = decide("SELECT id FROM plain", ALIASED, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=ALIASED, row_filters={"plain": filt}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_qualifier_the_parse_reads_as_the_literal_is_refused_even_where_harmless():
    # `true.id` is a field of the literal TRUE to the parser, whatever `true` names: nothing can tell
    # which column it reads, so it is refused with a repair that costs one alias (M232).
    verdict = decide("WITH true AS (SELECT id FROM kw) SELECT true.id FROM true", SQLITE_HIDDEN, dialect="sqlite",
                     target="sqlite", policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
    assert "another alias" in verdict.message
    repaired = decide("WITH t AS (SELECT id FROM kw) SELECT t.id FROM t", SQLITE_HIDDEN, dialect="sqlite",
                      target="sqlite", policy=AccessPolicy(policy_schema=SQLITE_SCHEMA))
    assert isinstance(repaired, Approved), repaired


def test_a_qualifier_the_parse_reads_as_a_name_is_a_table_not_a_read():
    # On SQLite `localtimestamp` is a name to the parser too, so `localtimestamp.id` is kw's id under
    # that alias, which the grant check reads; the qualifier is no read of a column so named.
    verdict = decide("SELECT localtimestamp.id FROM kw AS localtimestamp", {"kw": {"id"}, "plain": {"id"}},
                     dialect="sqlite", target="sqlite", policy=AccessPolicy(policy_schema=CROSS_SCHEMA))
    assert isinstance(verdict, Approved), verdict
    assert _sqlite(CROSS).execute(verdict.target_sql).fetchall() == [(1,)]



@pytest.mark.parametrize("sql", [
    "SELECT (current_date).secret FROM kw AS current_date",      # a parenthesised base
    "SELECT current_date['secret'] FROM kw AS current_date",     # a bracket
    "SELECT current_date FROM kw AS current_date",               # the whole row, every column
    "SELECT id FROM kw AS current_date ORDER BY current_date.secret",
    "SELECT max(current_date.secret) FROM kw AS current_date",
    "WITH current_date AS (SELECT * FROM kw) SELECT current_date.secret FROM current_date",
])
def test_every_form_of_reference_through_such_a_name_is_refused(sql):
    # The name is the cause: refused for any reference through it (the review gate on M232's row
    # found the parenthesised, bracketed and whole-row forms past a check of field access alone).
    ran = _with_secret("duckdb").execute(sql).fetchall()  # the premise: DuckDB binds the name to kw
    assert ran and ("ORDER BY" in sql or "withheld" in str(ran)), ran  # ORDER BY uses the value unseen
    verdict = decide(sql, ALIASED_HIDDEN, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=ALIASED))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


@pytest.mark.parametrize("engine, sql", [
    ("postgres", 'SELECT id FROM "user"'),                  # a common table name, a word Postgres binds no table by
    ("sqlite", "SELECT localtimestamp.id FROM kw AS localtimestamp"),
    ("duckdb", "SELECT t.id FROM kw AS true_t AS t"),
])
def test_names_the_engine_binds_no_table_by_are_not_refused(engine, sql):
    verdict = decide(sql.replace("AS true_t AS t", "AS t"), {"kw": {"id"}, "user": {"id"}}, dialect=engine,
                     target=engine, policy=AccessPolicy(policy_schema={"kw": {"id", "secret"}, "user": {"id"}}))
    assert not (isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME), verdict


@pytest.mark.parametrize("word", ["bigint", "int", "date", "json", "list", "struct", "text", "uuid", "variant"])
def test_on_duckdb_a_table_aliased_by_a_type_name_and_read_by_bracket_is_refused(word):
    # sqlglot reads `bigint['secret']` as an array of the BIGINT type; DuckDB reads the `secret` field
    # of kw's row under that alias (M232, swept: 59 type names, this shape alone).
    sql = f"SELECT {word}['secret'] FROM kw AS {word}"
    assert _with_secret("duckdb").execute(sql).fetchall() == [("withheld",)]  # the premise
    verdict = decide(sql, ALIASED_HIDDEN, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=ALIASED))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


@pytest.mark.parametrize("sql", [
    "SELECT CAST(d.id AS DATE) AS x FROM kw AS date, kw AS d",   # a type in a cast's target names no table
    "SELECT date.id FROM kw AS date WHERE date.id > 0",          # the alias read as a column's qualifier
    "SELECT id FROM kw AS t WHERE CAST(t.id AS BIGINT) > 0",
])
def test_a_type_standing_as_a_type_or_a_qualifier_read_as_one_is_not_refused(sql):
    verdict = decide(sql, ALIASED, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=ALIASED))
    assert not (isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME), verdict


@pytest.mark.parametrize("sql", ['SELECT current_date.secret FROM "current_date"', 'SELECT current_date FROM "current_date"'])
def test_a_table_itself_named_by_such_a_word_is_refused_too(sql):
    # Not only an alias: a table stored as "current_date" is read through the bare word, its row whole.
    con = duckdb.connect()
    con.execute('CREATE TABLE "current_date" (id INT, secret VARCHAR)')
    con.execute("INSERT INTO \"current_date\" VALUES (1, 'withheld')")
    assert "withheld" in str(con.execute(sql).fetchall())  # the premise
    verdict = decide(sql, {"current_date": {"id"}}, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema={"current_date": {"id", "secret"}}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_a_table_named_by_such_a_word_and_holding_such_a_column_is_read_by_its_name():
    # The table's own name in FROM is a name, not a read of its column so named: approved, as DuckDB
    # reads `FROM localtime` as the table.
    con = duckdb.connect()
    con.execute('CREATE TABLE "localtime" (id INT, "localtime" VARCHAR)')
    con.execute("INSERT INTO \"localtime\" VALUES (1, 'x')")
    schema = {"localtime": {"id", "localtime"}}
    verdict = decide("SELECT id FROM localtime", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema))
    assert isinstance(verdict, Approved), verdict
    assert con.execute(verdict.target_sql).fetchall() == [(1,)]


def test_a_filter_s_subquery_reading_such_a_table_by_its_bare_name_passes():
    # A filter runs as written, so the table's name stays bare there: still a name, not a read of
    # the column so named that the table holds.
    schema = {"plain": {"id"}, "localtime": {"id", "localtime"}}
    verdict = decide("SELECT id FROM plain", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={"plain": "id IN (SELECT id FROM localtime)"}))
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("word", ["int8", "integer", "int4", "string", "bool", "numeric", "bigint", "current_date"])
def test_a_row_filter_reading_a_table_through_a_word_is_refused_by_its_own_spelling(word):
    # A filter runs as written, and sqlglot prints `int8` as BIGINT: checked on the reprint, the alias
    # was missed and the filter answered on kw's withheld secret (Codex's fifth review of this change).
    filt = f"id IN (SELECT id FROM kw AS {word} WHERE {word}['secret'] = 'withheld')"
    con = duckdb.connect()
    con.execute("CREATE TABLE kw (id INT, secret VARCHAR); INSERT INTO kw VALUES (1, 'withheld')")
    con.execute("CREATE TABLE plain (id INT); INSERT INTO plain VALUES (1)")
    assert con.execute(f"SELECT id FROM plain WHERE {filt}").fetchall() == [(1,)]  # the premise
    schema = {"kw": {"id"}, "plain": {"id"}}  # secret is in no schema the checks know
    verdict = decide("SELECT id FROM plain", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={"plain": filt}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_row_filter_with_an_ordinary_alias_or_a_type_in_a_cast_still_passes():
    schema = {"kw": {"id", "secret"}, "plain": {"id"}}
    for filt in ("id IN (SELECT t.id FROM kw AS t WHERE t.secret = 'x')",
                 "id IN (SELECT id FROM kw WHERE TRY_CAST(id AS int8) > 0)"):
        verdict = decide("SELECT id FROM plain", schema, dialect="duckdb", target="duckdb",
                         policy=AccessPolicy(policy_schema=schema, row_filters={"plain": filt}))
        assert isinstance(verdict, Approved), (filt, verdict)


def test_a_row_filter_calling_a_function_named_like_its_alias_passes():
    # A call names no table: `abs(abs.id)` under `FROM kw AS abs` reads the function and kw's id.
    schema = {"kw": {"id", "secret"}, "plain": {"id"}}
    filt = "id IN (SELECT abs.id FROM kw AS abs WHERE abs(abs.id) > 0)"
    verdict = decide("SELECT id FROM plain", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={"plain": filt}))
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("word, form", [("int8", "int8['secret'] = 'withheld'"), ("bigint", "bigint['secret'] = 'withheld'"),
                                        ("current_date", "current_date.secret = 'withheld'")])
def test_a_filter_reading_its_own_table_through_a_word_is_refused(word, form):
    # The filtered table is in the filter's scope though the filter declares nothing: on a table named
    # int8, `int8['secret']` read its withheld column (Codex's sixth review of this change).
    con = duckdb.connect()
    con.execute(f'CREATE TABLE "{word}" (id INT, secret VARCHAR)')
    con.execute(f"INSERT INTO \"{word}\" VALUES (1, 'withheld'), (2, 'other')")
    assert con.execute(f'SELECT id FROM "{word}" WHERE {form}').fetchall() == [(1,)]  # the premise
    schema = {word: {"id"}}  # secret is in no schema the checks know
    verdict = decide(f'SELECT id FROM "{word}"', schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={word: form}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (
        verdict, con.execute(verdict.target_sql).fetchall() if isinstance(verdict, Approved) else None)


def test_a_write_s_alias_cannot_rebind_a_word_in_the_target_s_filter():
    # A write's filter is conjoined to the statement, so the caller's aliases are in its scope: a target
    # aliased current_date would rebind the operator's `current_date` to the target's row.
    schema = {"kw": {"id", "d"}}
    grants = GrantSet(frozenset(schema), writable=frozenset(schema), row_filters={"kw": "d > current_date - INTERVAL 30 DAY"})
    verdict = decide_write("DELETE FROM kw AS current_date WHERE id > 0", schema, grants, dialect="duckdb",
                           target="duckdb", writes_enabled=True,
                           policy=AccessPolicy(policy_schema=schema, row_filters={"kw": "d > current_date - INTERVAL 30 DAY"}))
    assert isinstance(verdict, Refusal), verdict


def test_a_filter_s_correlated_reference_to_its_own_table_passes():
    # Bare, as a filter names its own table's columns (one qualified by the table's name is refused
    # before this change, and still is).
    schema = {"kw": {"id", "secret"}, "ent": {"id"}}
    verdict = decide("SELECT id FROM kw", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={"kw": "EXISTS (SELECT 1 FROM ent WHERE ent.id = id)"}))
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("table, sql, filt", [
    ("int8", 'SELECT id FROM "int8"', "TRY_CAST(id AS int8) > 0"),
    ("text", 'SELECT id FROM "text"', "TRY_CAST(id AS text) <> ''"),
    ("date", 'SELECT id FROM "date"', "d > DATE '2020-01-01'"),
    ("kw", "SELECT int8.id FROM kw AS int8", "TRY_CAST(id AS int8) > 0"),
])
def test_a_type_in_a_filter_spelled_like_a_table_in_scope_is_a_type(table, sql, filt):
    # A type position names a type, not the table the word also names (Codex's seventh review: such a
    # filter was refused, leaving a table named like the type it casts to unreadable).
    schema = {table: {"id", "d"}}
    verdict = decide(sql, schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={table: filt}))
    assert isinstance(verdict, Approved), verdict



@pytest.mark.parametrize("text, found", [
    ("id::int8 > 0", None),                        # after `::`
    ("CAST(id AS int8) > 0", None),                # after a cast's AS
    ("TRY_CAST(id AS int8) > 0", None),
    ("d > int8 '1'", None),                         # a typed literal
    ("int8['secret'] = 'x'", "int8"),             # the table, read by bracket
    ("(int8).secret = 'x'", None),                 # a column to the parse, which the filter's check reads
    ("f(id AS int8) > 0", None),                   # an alias, a declared name
])
def test_a_filter_s_type_positions_are_read_off_its_own_text(text, found):
    # The conversion rule refuses every CAST and `::` in a filter before this check runs, so these
    # positions are asked of the check itself, with the filtered table named int8 in scope.
    from mnemiq.sql.guard import a_table_named_in_text
    assert a_table_named_in_text(text, "duckdb", frozenset({"int8"})) == found


@pytest.mark.parametrize("table, filt", [
    ("int8", "TRY_CAST(ROW(id) AS STRUCT(a int8)).a > 0"),                # a nested type
    ("time", "TRY_CAST(d AS TIMESTAMP WITH TIME ZONE) IS NOT NULL"),     # a type of several words
    ("time", "d > TIMESTAMP WITH TIME ZONE '2000-01-01 00:00:00+00'"),    # a typed literal of several words
])
def test_a_nested_or_multiword_type_in_a_filter_is_a_type(table, filt):
    # The whole type is a type, not only its first word (Codex's eighth review of this change).
    con = duckdb.connect()
    con.execute(f'CREATE TABLE "{table}" (id INT, d TIMESTAMP)')
    con.execute(f"INSERT INTO \"{table}\" VALUES (1, TIMESTAMP '2020-01-01')")
    assert con.execute(f'SELECT id FROM "{table}" WHERE {filt}').fetchall() == [(1,)]  # runs on the engine
    schema = {table: {"id", "d"}}
    verdict = decide(f'SELECT id FROM "{table}"', schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema=schema, row_filters={table: filt}))
    assert isinstance(verdict, Approved), verdict
    assert con.execute(verdict.target_sql).fetchall() == [(1,)]


@pytest.mark.parametrize("text, found", [
    ("x::STRUCT(a int8) IS NOT NULL", None),
    ("x::int8[] IS NOT NULL", None),
    ("TRY_CAST(x AS DECIMAL(10, 2)) > int8['secret']", "int8"),   # the type spans its cast alone
    ("TRY_CAST(int8['secret'] AS VARCHAR) = 'x'", "int8"),        # before the AS is a value
    ("TRY_CAST(f(1 AS a, int8['secret']) AS VARCHAR) = 'x'", "int8"),  # an AS nested in the value is not the cast's
])
def test_a_type_span_ends_where_its_type_does(text, found):
    from mnemiq.sql.guard import a_table_named_in_text
    assert a_table_named_in_text(text, "duckdb", frozenset({"int8"})) == found

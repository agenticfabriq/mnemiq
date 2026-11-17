"""A date part written as a function's unit the way another SQL dialect writes it is refused as a unit, not as a
missing column, under `repair_hints`.

`TIMESTAMPDIFF(HOUR, s, e)` is MySQL's. Read as DuckDB, `HOUR` is a column, and the decider refused it as "No table in
this query has a column 'HOUR'": true of the tree it read, wrong about what the writer meant, and a writer told that
sends the same statement again, since it never thought HOUR was a column. With the setting on, the refusal names the
unit and gives the writer's own dialect's date arithmetic. Each sentence's claim is executed below where the suite can
reach that engine. No rewritten call is offered: sqlglot's transpile of these functions is wrong for several targets,
and DuckDB's `date_diff` counts boundaries where MySQL's `TIMESTAMPDIFF` counts whole units.
"""

from __future__ import annotations

import os
import sqlite3

import duckdb
import pytest

from mnemiq.sql.date_units import _unit_call
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, RefusalCode

VISIBLE = {"orders": {"id", "s", "e", "status", "region"}, "shifts": {"hour"}}
TODAY_HOUR = "No table in this query has a column 'HOUR'. Use only the columns listed on the schema cards."


def _decide(sql: str, visible=VISIBLE, dialect: str = "duckdb", **kwargs):
    return decide(sql, visible, dialect=dialect, target=dialect, **kwargs)


def test_timestampdiff_hour_is_named_as_a_unit_in_duckdb():
    v = _decide("SELECT TIMESTAMPDIFF(HOUR, s, e) AS h FROM orders", repair_hints=True)
    assert v.code == RefusalCode.UNKNOWN_COLUMN and v.subject == "HOUR"
    assert v.message == ("'HOUR' is not a column here: it is the unit argument of TIMESTAMPDIFF, written the way "
                         "another SQL dialect writes it. In duckdb, date_sub('hour', start, end) counts whole hours "
                         "elapsed and date_diff('hour', start, end) counts hour boundaries crossed, with the unit as "
                         "a quoted string.")


def test_the_function_is_named_as_written_whatever_its_case():
    v = _decide("SELECT timestampdiff(minute, s, e) FROM orders", repair_hints=True)
    assert v.message.startswith("'minute' is not a column here: it is the unit argument of TIMESTAMPDIFF,")
    assert "date_sub('minute', start, end) counts whole minutes elapsed" in v.message


def test_dateadd_gets_no_difference_sentence():
    v = _decide("SELECT DATEADD(day, 1, s) FROM orders", repair_hints=True)
    assert v.message == ("'day' is not a column here: it is the unit argument of DATEADD, written the way another "
                         "SQL dialect writes it. Write it with this database's own date functions.")


@pytest.mark.parametrize("dialect, sentence", [
    ("sqlite", "In SQLite, julianday(end) - julianday(start) is the days between two times, to multiply up to the "
               "unit wanted."),
    ("postgres", "In Postgres, EXTRACT(EPOCH FROM end - start) is the seconds between two timestamps, to divide down "
                 "to the unit wanted."),
    ("oracle", "In Oracle, subtracting two DATEs gives the days between them as a number, and subtracting two "
               "TIMESTAMPs gives an INTERVAL DAY TO SECOND."),
    ("tsql", "Write it with this database's own date functions."),
])
def test_each_dialect_gets_its_own_arithmetic(dialect, sentence):
    # Oracle's decider wants a table written as it is listed, and an Oracle catalogue lists names upper case.
    if dialect == "oracle":
        visible, sql = {"ORDERS": {"ID", "S", "E"}}, "SELECT TIMESTAMPDIFF(HOUR, S, E) FROM ORDERS"
    else:
        visible, sql = VISIBLE, "SELECT TIMESTAMPDIFF(HOUR, s, e) FROM orders"
    v = _decide(sql, visible, dialect=dialect, repair_hints=True)
    assert v.code == RefusalCode.UNKNOWN_COLUMN
    assert v.message.startswith("'HOUR' is not a column here: it is the unit argument of TIMESTAMPDIFF,"), v.message
    assert v.message.endswith(" " + sentence)


def test_off_keeps_todays_message():
    assert _decide("SELECT TIMESTAMPDIFF(HOUR, s, e) FROM orders").message == TODAY_HOUR


def test_a_real_column_named_hour_is_not_refused():
    assert isinstance(_decide("SELECT hour FROM shifts", repair_hints=True), Approved)


def test_a_qualified_missing_column_keeps_its_message():
    v = _decide("SELECT DATE_DIFF('day', o.hour, o.e) FROM orders o", repair_hints=True)
    assert v.message == "'orders' has no column 'hour'. Use only listed columns."


def test_a_missing_column_that_is_no_date_part_keeps_its_message():
    v = _decide("SELECT DATEADD(foo, 1, s) FROM orders", repair_hints=True)
    assert v.message == "No table in this query has a column 'foo'. Use only the columns listed on the schema cards."


@pytest.mark.parametrize("sql", [
    "SELECT SUM(hour) FROM orders",                   # an aggregate's argument
    "SELECT DATE_DIFF('day', hour, e) FROM orders",   # a timestamp operand, not the unit
    "SELECT COALESCE(hour, 0) FROM orders",           # an ordinary function's first argument
])
def test_a_date_part_name_used_as_data_keeps_its_message(sql):
    v = _decide(sql, repair_hints=True)
    assert v.message == "No table in this query has a column 'hour'. Use only the columns listed on the schema cards."


@pytest.mark.parametrize("sql", [
    "SELECT hour, 'TIMESTAMPDIFF(hour, s, e)' FROM orders",          # inside a string literal
    "SELECT hour FROM orders -- TIMESTAMPDIFF(hour, s, e)",          # inside a line comment
    "SELECT hour FROM orders /* TIMESTAMPDIFF(hour, s, e) */",       # inside a block comment
    "SELECT x.TIMESTAMPDIFF(hour, s, e) FROM orders",                # a qualified name, not the date function
    "SELECT MY_TIMESTAMPDIFF(hour, s, e) FROM orders",               # a name that only ends like one
])
def test_text_that_is_not_a_call_of_the_date_function_names_no_unit(sql):
    """Codex's review of this fix: a raw-text match fired inside literals and comments. Read as tokens, it cannot.
    Each is refused exactly as with the setting off (a qualified call by another check altogether)."""
    v = _decide(sql, repair_hints=True)
    assert v.message == _decide(sql).message and "is not a column here" not in v.message


def test_a_qualified_call_names_no_unit_even_where_a_refusal_would_reach_it():
    """The qualified call above is refused by another check before this one is asked, so the rule is tested here too."""
    assert _unit_call("SELECT x.TIMESTAMPDIFF(hour, s, e) FROM orders", "hour", "duckdb") is None
    assert _unit_call("SELECT TIMESTAMPDIFF(hour, s, e) FROM orders", "hour", "duckdb") == "TIMESTAMPDIFF"
    assert _unit_call("SELECT TIMESTAMPDIFF(hour, s, e", "hour", "no-such-dialect") is None


def test_a_quoted_unit_is_a_column_the_writer_meant():
    v = _decide('SELECT TIMESTAMPDIFF("HOUR", s, e) FROM orders', repair_hints=True)
    assert not v.message.startswith("'HOUR' is not a column here"), v.message


def test_another_refusal_is_left_as_it_is():
    v = _decide("SELECT * FROM orders", repair_hints=True)
    assert v.code == RefusalCode.SELECT_STAR
    assert v.message == _decide("SELECT * FROM orders").message


def test_the_duckdb_sentence_is_true():
    row = duckdb.sql("SELECT date_sub('hour', TIMESTAMP '2023-01-01 00:30:00', TIMESTAMP '2023-01-01 02:00:00'), "
                     "date_diff('hour', TIMESTAMP '2023-01-01 00:30:00', TIMESTAMP '2023-01-01 02:00:00')").fetchone()
    assert row == (1, 2)


def test_the_sqlite_sentence_is_true():
    (days,) = sqlite3.connect(":memory:").execute(
        "SELECT julianday('2023-01-01 02:00:00') - julianday('2023-01-01 00:30:00')").fetchone()
    assert days == 0.0625


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_the_postgres_sentence_is_true():
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(os.environ["MNEMIQ_PG_DSN"]) as con:
        (seconds,) = con.execute("SELECT EXTRACT(EPOCH FROM TIMESTAMP '2023-01-01 02:00:00' "
                                 "- TIMESTAMP '2023-01-01 00:30:00')").fetchone()
    assert float(seconds) == 5400.0


ORACLE_DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not ORACLE_DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_the_oracle_sentence_is_true():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=ORACLE_DSN)
    try:
        cur = con.cursor()
        (days,) = cur.execute("SELECT DATE '2023-01-03' - DATE '2023-01-01' FROM dual").fetchone()
        assert days == 2
        (interval,) = cur.execute("SELECT TIMESTAMP '2023-01-01 02:00:00' - TIMESTAMP '2023-01-01 00:30:00' "
                                  "FROM dual").fetchone()
        assert interval.total_seconds() == 5400.0  # an INTERVAL DAY TO SECOND arrives as a timedelta
    finally:
        con.close()


@pytest.mark.parametrize("sql", [
    "SELECT o.hour, TIMESTAMPDIFF(hour, o.s, o.e) FROM orders o",   # the refusal is about the qualified column
    "SELECT hour, TIMESTAMPDIFF(HOUR, s, e) FROM orders",           # the projection's `hour` is meant as a column
])
def test_a_statement_that_also_reads_the_word_as_a_column_keeps_its_message(sql):
    """The final review: naming the word a unit is right only when every place it is written is a unit position."""
    assert _decide(sql, repair_hints=True).message == _decide(sql).message

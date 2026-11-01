"""A bare date compared with a timestamp column is read as the whole day it names.

`ts <= '2024-03-09'` compares with midnight at the START of the 9th, so every row later that day
falls out, where the range a person means by "as of", "through" or "up to" the 9th ends before the
10th. A local 32B model wrote this bound on most runs of the questions that turn on it, and every
such answer was wrong; ending the range at the next day turned them right and turned no right
answer wrong. The decider reads every such comparison by day (`close_day_bounds`): `<=` becomes
`<` the next day and `>` becomes `>=` it, `=` and `<>` the day's range and its negation, so
complementary predicates stay complementary. The next day is written in the literal's own
spelling, so the statement stays valid in every dialect, and the caller, the judge and the source
all read the moved bound. Only a bare column bound directly to a base table of its own scope and
declared a naive timestamp (or an Oracle DATE, which holds a time of day) is read this way: on a
DATE the two readings are the same rows, and on text, a zone-aware timestamp or an expression
nothing moves.
"""
from __future__ import annotations

import json

import duckdb
import pytest
import sqlglot

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.generate.generator import FakeGenerator
from mnemiq.runtime import _build_mode_verifiers
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.server.serialize import answer_payload
from mnemiq.sql.day_bounds import close_day_bounds
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved

VISIBLE = {"claim": {"id", "ts", "filed", "tz", "note", "region", "raw"}}
TYPES = {"claim": {"id": "INTEGER", "ts": "TIMESTAMP", "filed": "DATE", "tz": "TIMESTAMPTZ",
                   "note": "VARCHAR", "region": "VARCHAR", "raw": "BLOB"}}


def _closed(sql: str, dialect: str = "duckdb", types=TYPES) -> tuple[str, int]:
    tree = sqlglot.parse_one(sql, read=dialect)
    moved = close_day_bounds(tree, VISIBLE, types, dialect)
    return tree.sql(dialect=dialect), moved


@pytest.mark.parametrize("asked, closed", [
    ("SELECT id FROM claim WHERE ts <= '2024-03-09'",
     "SELECT id FROM claim WHERE ts < '2024-03-10'"),
    ("SELECT id FROM claim WHERE ts <= CAST('2024-03-09' AS TIMESTAMP)",
     "SELECT id FROM claim WHERE ts < CAST('2024-03-10' AS TIMESTAMP)"),
    ("SELECT id FROM claim WHERE ts <= DATE '2024-03-09'",
     "SELECT id FROM claim WHERE ts < CAST('2024-03-10' AS DATE)"),
    ("SELECT id FROM claim WHERE ts <= TRY_CAST('2024-03-09' AS TIMESTAMP)",
     "SELECT id FROM claim WHERE ts < TRY_CAST('2024-03-10' AS TIMESTAMP)"),
    # The literal on the left is the same bound, written with the column first.
    ("SELECT id FROM claim WHERE '2024-03-09' >= ts",
     "SELECT id FROM claim WHERE ts < '2024-03-10'"),
    # After the day: from the next day's start, either way round.
    ("SELECT id FROM claim WHERE ts > '2024-03-09'",
     "SELECT id FROM claim WHERE ts >= '2024-03-10'"),
    ("SELECT id FROM claim WHERE '2024-03-09' < ts",
     "SELECT id FROM claim WHERE ts >= '2024-03-10'"),
    # On the day, and off it.
    ("SELECT id FROM claim WHERE ts = '2024-03-09'",
     "SELECT id FROM claim WHERE (ts >= '2024-03-09' AND ts < '2024-03-10')"),
    ("SELECT id FROM claim WHERE ts <> '2024-03-09'",
     "SELECT id FROM claim WHERE NOT (ts >= '2024-03-09' AND ts < '2024-03-10')"),
    # In a list, on any of its days; NULL-safe, NULL is never on the day.
    ("SELECT id FROM claim WHERE ts IN ('2024-03-09', '2024-03-10')",
     "SELECT id FROM claim WHERE ((ts >= '2024-03-09' AND ts < '2024-03-10') OR "
     "(ts >= '2024-03-10' AND ts < '2024-03-11'))"),
    ("SELECT id FROM claim WHERE ts NOT IN ('2024-03-09')",
     "SELECT id FROM claim WHERE NOT (ts >= '2024-03-09' AND ts < '2024-03-10')"),
    ("SELECT id FROM claim WHERE ts IS NOT DISTINCT FROM '2024-03-09'",
     "SELECT id FROM claim WHERE (NOT ts IS NULL AND ts >= '2024-03-09' AND ts < '2024-03-10')"),
    ("SELECT id FROM claim WHERE ts IS DISTINCT FROM '2024-03-09'",
     "SELECT id FROM claim WHERE NOT (NOT ts IS NULL AND ts >= '2024-03-09' AND ts < '2024-03-10')"),
    # A qualified column, through its alias.
    ("SELECT c.id FROM claim AS c WHERE c.ts <= '2024-03-09'",
     "SELECT c.id FROM claim AS c WHERE c.ts < '2024-03-10'"),
    # The month, the year and a leap day roll over as a calendar does.
    ("SELECT id FROM claim WHERE ts <= '2024-12-31'",
     "SELECT id FROM claim WHERE ts < '2025-01-01'"),
    ("SELECT id FROM claim WHERE ts <= '2024-02-28'",
     "SELECT id FROM claim WHERE ts < '2024-02-29'"),
])
def test_an_inclusive_end_on_a_bare_date_moves_to_the_next_days_start(asked, closed):
    assert _closed(asked) == (closed, 1)


def test_a_between_ending_on_a_bare_date_keeps_its_start_and_ends_before_the_next_day():
    assert _closed("SELECT id FROM claim WHERE ts BETWEEN '2024-03-01' AND '2024-03-09'") == (
        "SELECT id FROM claim WHERE (ts >= '2024-03-01' AND ts < '2024-03-10')", 1)
    # Negated, the whole range is negated: the rows outside the days the range names.
    assert _closed("SELECT id FROM claim WHERE NOT ts BETWEEN '2024-03-01' AND '2024-03-09'") == (
        "SELECT id FROM claim WHERE NOT (ts >= '2024-03-01' AND ts < '2024-03-10')", 1)


def test_every_such_bound_on_a_base_tables_column_moves_wherever_it_sits():
    sql, moved = _closed(
        "SELECT id, CASE WHEN ts <= '2024-03-05' THEN 'early' ELSE 'late' END AS half FROM claim "
        "WHERE ts <= '2024-03-09' AND EXISTS (SELECT 1 FROM claim AS c2 WHERE c2.ts <= '2024-03-01')")
    assert moved == 3
    assert ("ts < '2024-03-06'" in sql and "ts < '2024-03-10'" in sql and "c2.ts < '2024-03-02'" in sql
            and "<=" not in sql)
    # Inside a CTE's body the column is the base table's, and moves.
    assert _closed("WITH r AS (SELECT id FROM claim WHERE ts <= '2024-03-09') SELECT id FROM r") == (
        "WITH r AS (SELECT id FROM claim WHERE ts < '2024-03-10') SELECT id FROM r", 1)


@pytest.mark.parametrize("asked", [
    # A set operation's output is not the base column it shares a name with (Codex's review):
    # here it is text, which compares as text.
    "WITH x AS (SELECT note AS ts FROM claim UNION ALL SELECT note AS ts FROM claim) "
    "SELECT ts FROM x WHERE ts <= '2024-03-09'",
    # A derived table's output, the same.
    "SELECT ts FROM (SELECT note AS ts FROM claim) AS d WHERE ts <= '2024-03-09'",
    # A name the subquery's own table does not hold is the enclosing scope's (a CTE's output here).
    "WITH x AS (SELECT note AS stamp FROM claim) SELECT stamp FROM x "
    "WHERE EXISTS (SELECT 1 FROM claim AS u WHERE stamp <= '2024-03-09')",
    # The cost, accepted: a CTE's output read outside it is not followed, nor is an enclosing
    # scope's, even where it is the base table's timestamp column unchanged.
    "WITH r AS (SELECT ts FROM claim) SELECT ts FROM r WHERE ts <= '2024-03-09'",
    "WITH x AS (SELECT ts AS stamp FROM claim) SELECT stamp FROM x "
    "WHERE EXISTS (SELECT 1 FROM claim AS u WHERE stamp <= '2024-03-09')",
])
def test_a_column_not_bound_directly_to_a_base_table_is_never_moved(asked):
    assert _closed(asked)[1] == 0, asked


@pytest.mark.parametrize("asked", [
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND CAST(ts AS TIMESTAMP) <= '2024-03-09'",
    "SELECT id, CASE ts WHEN '2024-03-09' THEN 1 END AS d FROM claim WHERE ts <= '2024-03-09'",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts = ANY(['2024-03-09'])",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts IN ('2024-03-09', '2024-03-10 12:00:00')",
    "WITH r AS (SELECT id, ts FROM claim) SELECT c.id FROM claim AS c JOIN r ON r.id = c.id "
    "WHERE c.ts = '2024-03-09' AND r.ts <> '2024-03-09'",
    "WITH r AS (SELECT id, ts FROM claim) SELECT c.id FROM claim AS c JOIN r ON r.id = c.id "
    "WHERE c.ts <= '2024-03-09' AND r.ts BETWEEN '2024-03-01' AND '2024-03-09'",
    "WITH r AS (SELECT id, ts FROM claim) SELECT c.id FROM claim AS c JOIN r ON r.id = c.id "
    "WHERE c.ts <= '2024-03-09' AND r.ts IN ('2024-03-09')",
    # The same timestamp met by a day this cannot move (Codex's third review, and its kin): inside a
    # subquery, through a cast of the column, as a DATE column's value or today's date.
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND ts <> (SELECT CAST('2024-03-09' AS TIMESTAMP))",
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND ts NOT IN (SELECT CAST('2024-03-09' AS TIMESTAMP))",
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND CAST(ts AS TIMESTAMPTZ) <> '2024-03-09'",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts > filed",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts > CURRENT_DATE",
    "WITH r AS (SELECT '2024-03-09' AS d) SELECT id FROM claim, r WHERE ts <= '2024-03-09' AND ts > r.d",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND CAST(ts AS VARCHAR) > '2024-03-09'",
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND ts > CAST('2024-03-09' AS VARCHAR)",
    "SELECT id FROM claim WHERE ts = '2024-03-09' "
    "AND CAST(ts AS TIMESTAMPTZ) <> (SELECT CAST('2024-03-09' AS TIMESTAMPTZ))",
    # A comparison this does not read: a USING join's.
    "SELECT a.id FROM claim AS a JOIN claim AS b USING (id) WHERE a.ts <= '2024-03-09'",
    # A string an engine reads as a day in another form; text against a timestamp expression
    # (Codex's fourth review); a pattern match on the timestamp.
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND ts <> '2024-3-9'",
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND (ts + INTERVAL '0 seconds') <> note",
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND ts LIKE '2024-03-09%'",
    # A zone-aware copy of the timestamp, an epoch number (Codex's fifth review; its SQLite BLOB case
    # runs on SQLite below).
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts = tz AND tz > '2024-03-09'",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND epoch(ts) > 1709942400",
    # `IS` with a value, SQLite's NULL-safe equality, written for another engine.
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND ts IS '2024-03-09'",
    # The timestamp compared inside a function (Codex's tenth review), or carried into an expression.
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND NOT list_contains([ts], TIMESTAMP '2024-03-09')",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND NULLIF(ts, filed) IS NULL",
    "SELECT id, ts + INTERVAL 1 DAY AS later FROM claim WHERE ts <= '2024-03-09'",
    # An aggregate carrying the timestamp into a function that compares it (Codex's eleventh review).
    "SELECT NULLIF(MIN(ts), TIMESTAMP '2024-03-09') AS m FROM claim WHERE ts <= '2024-03-09'",
    "SELECT GREATEST(MAX(ts), TIMESTAMP '2024-03-09') AS m FROM claim WHERE ts <= '2024-03-09'",
    # Set operations that compare rows (Codex's thirteenth review).
    "SELECT COUNT(*) FROM (SELECT ts FROM claim WHERE ts <= '2024-03-09' EXCEPT SELECT TIMESTAMP '2024-03-09') AS q",
    "SELECT ts FROM claim WHERE ts <= '2024-03-09' INTERSECT SELECT TIMESTAMP '2024-03-09'",
    "SELECT ts FROM claim WHERE ts <= '2024-03-09' UNION SELECT TIMESTAMP '2024-03-09'",
    # A CTE's output, whose type is unknown, carried into a function that compares it.
    "WITH r AS (SELECT ts AS t FROM claim) SELECT list_contains([r.t], TIMESTAMP '2024-03-09') AS f "
    "FROM r, claim WHERE claim.ts <= '2024-03-09'",
    "SELECT id FROM claim WHERE ts = '2024-03-09' "
    "AND NOT list_contains([CAST(ts AS TIMESTAMPTZ)], TIMESTAMPTZ '2024-03-09 00:00:00+00')",
    # A day as the subject of an IN listing the timestamp (Codex's sixth review).
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND '2024-03-09' NOT IN (ts)",
    "SELECT id FROM claim WHERE ts = '2024-03-09' AND '2024-03-09' IN (ts, ts)",
    # The same, met only inside a BETWEEN: a DATE column as its low end, an expression as its subject.
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts BETWEEN filed AND '2024-03-20'",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND upper(note) BETWEEN 'a' AND 'b'",
    # The cost, accepted: any expression in any comparison -- a function, an aggregate, an IN over a
    # subquery -- leaves the statement as written.
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND upper(region) = 'WEST'",
    "SELECT region, COUNT(*) AS n FROM claim WHERE ts <= '2024-03-09' GROUP BY region HAVING COUNT(*) > 5",
    "SELECT id FROM claim WHERE ts <= '2024-03-09' AND id IN (SELECT id FROM claim WHERE region = 'west')",
])
def test_one_comparison_that_cannot_be_read_by_day_leaves_the_whole_statement_as_written(asked):
    """One statement, one reading: beside a day compared with something this cannot read by day (an
    expression of no known type, a simple CASE, ANY, a mixed list, a CTE's output), the statement's
    other comparisons are not moved either, so its predicates keep meaning together what they meant."""
    sql, moved = _closed(asked)
    assert moved == 0
    assert sql == sqlglot.parse_one(asked, read="duckdb").sql(dialect="duckdb")


def test_a_day_compared_with_whole_days_text_or_a_projection_does_not_stop_the_move():
    assert _closed("SELECT id, '2024-03-09' AS asof FROM claim WHERE ts <= '2024-03-09' "
                   "AND filed <= '2024-03-09' AND CAST(ts AS DATE) <= '2024-03-09' AND note <> '2024-03-09'") == (
        "SELECT id, '2024-03-09' AS asof FROM claim WHERE ts < '2024-03-10' AND filed <= '2024-03-09' "
        "AND CAST(ts AS DATE) <= '2024-03-09' AND note <> '2024-03-09'", 1)


@pytest.mark.parametrize("asked, closed", [
    # The timestamp passing through: projected, grouped, ordered, counted, its MAX, its day.
    ("SELECT ts, COUNT(DISTINCT ts) AS n, MAX(ts) AS last, CAST(ts AS DATE) AS d FROM claim "
     "WHERE ts <= '2024-03-09' GROUP BY ts ORDER BY ts",
     "SELECT ts, COUNT(DISTINCT ts) AS n, MAX(ts) AS last, CAST(ts AS DATE) AS d FROM claim "
     "WHERE ts < '2024-03-10' GROUP BY ts ORDER BY ts"),
    # UNION ALL compares nothing.
    ("SELECT ts FROM claim WHERE ts <= '2024-03-09' UNION ALL SELECT ts FROM claim WHERE ts >= '2024-03-20'",
     "SELECT ts FROM claim WHERE ts < '2024-03-10' UNION ALL SELECT ts FROM claim WHERE ts >= '2024-03-20'"),
    # A count is never a time, wherever it goes.
    ("SELECT NULLIF(COUNT(ts), 0) AS n FROM claim WHERE ts <= '2024-03-09'",
     "SELECT NULLIF(COUNT(ts), 0) AS n FROM claim WHERE ts < '2024-03-10'"),
    # Columns and literals only: a moment with its time of day, a number, other columns, a timestamp
    # join condition.
    ("SELECT id FROM claim WHERE ts <= '2024-03-09' AND ts > '2024-03-01 12:00:00' AND id > -1",
     "SELECT id FROM claim WHERE ts < '2024-03-10' AND ts > '2024-03-01 12:00:00' AND id > -1"),
    ("SELECT a.id FROM claim AS a JOIN claim AS b ON a.id = b.id AND a.ts >= b.ts WHERE a.ts <= '2024-03-09'",
     "SELECT a.id FROM claim AS a JOIN claim AS b ON a.id = b.id AND a.ts >= b.ts WHERE a.ts < '2024-03-10'"),
])
def test_comparisons_that_read_no_day_do_not_stop_the_move(asked, closed):
    assert _closed(asked) == (closed, 1)


def test_the_third_reviews_contradictions_stay_contradictions_on_the_engine():
    """Codex's third review: a day inside a subquery, and a cast of the same column to TIMESTAMPTZ,
    each kept its midnight while `=` moved, and a noon row satisfied a contradiction. A DATE
    column's value is the same class. Each statement is now left as written, and returns nothing."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP, filed DATE, tz TIMESTAMPTZ, note VARCHAR, "
                "region VARCHAR)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00', '2024-03-09', NULL, NULL, 'west')")
    for asked in ("SELECT count(*) FROM claim WHERE ts = '2024-03-09' AND ts <> (SELECT CAST('2024-03-09' AS TIMESTAMP))",
                  "SELECT count(*) FROM claim WHERE ts = '2024-03-09' AND ts NOT IN (SELECT CAST('2024-03-09' AS TIMESTAMP))",
                  "SELECT count(*) FROM claim WHERE ts = '2024-03-09' AND CAST(ts AS TIMESTAMPTZ) <> '2024-03-09'",
                  "SELECT count(*) FROM claim WHERE ts <= '2024-03-09' AND ts > filed",
                  # A day cast to TIMESTAMPTZ inside a subquery, against the same column's cast.
                  "SELECT count(*) FROM claim WHERE ts = '2024-03-09' "
                  "AND CAST(ts AS TIMESTAMPTZ) <> (SELECT CAST('2024-03-09' AS TIMESTAMPTZ))"):
        verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 0, (asked, verdict)
        assert con.execute(verdict.target_sql).fetchone() == con.execute(asked).fetchone() == (0,), asked


def test_the_fourth_reviews_contradictions_stay_contradictions_on_the_engine():
    """Codex's fourth review: SQLite's `IS NOT`, a NULL-safe comparison the walk did not visit, and
    a text column's day against a timestamp expression, each read the noon row at midnight while
    `=` moved. Nothing moves on SQLite now, and the text against an expression leaves the statement
    as written. Each still returns nothing."""
    import sqlite3

    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    lite.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00'), (2, NULL)")
    asked = "SELECT count(*) FROM claim WHERE ts = '2024-03-09' AND ts IS NOT '2024-03-09'"
    verdict = decide(asked, VISIBLE, dialect="sqlite", target="sqlite", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0, verdict  # never on SQLite
    assert lite.execute(verdict.target_sql).fetchone() == lite.execute(asked).fetchone() == (0,)

    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP, note VARCHAR)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00', '2024-03-09')")
    asked = "SELECT count(*) FROM claim WHERE ts = '2024-03-09' AND (ts + INTERVAL '0 seconds') <> note"
    verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0, verdict
    assert con.execute(verdict.target_sql).fetchone() == con.execute(asked).fetchone() == (0,)


def test_the_fifth_reviews_contradictions_stay_contradictions_on_the_engine():
    """Codex's fifth review: a zone-aware copy of the timestamp (`ts = tz AND tz > D`), an epoch number
    and, on SQLite, a BLOB holding the day each carried the same value to a midnight reading while
    `<=` moved. Every comparison must now be between bare columns and literals, and a timestamp met
    only by days, moments, numbers and timestamps: each statement is left as written and returns
    nothing."""
    import sqlite3

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP, filed DATE, tz TIMESTAMPTZ, note VARCHAR, "
                "region VARCHAR, raw BLOB)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00', NULL, '2024-03-09 12:00:00+00', NULL, NULL, NULL)")
    for asked in ("SELECT count(*) FROM claim WHERE ts <= '2024-03-09' AND ts = tz AND tz > '2024-03-09'",
                  "SELECT count(*) FROM claim WHERE ts <= '2024-03-09' AND epoch(ts) > 1709985600"):
        verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 0, (asked, verdict)
        assert con.execute(verdict.target_sql).fetchone() == con.execute(asked).fetchone() == (0,), asked
    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP, raw BLOB)")
    lite.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00', '2024-03-09')")
    asked = "SELECT count(*) FROM claim WHERE ts <= '2024-03-09' AND datetime(ts) > raw AND raw = '2024-03-09'"
    verdict = decide(asked, VISIBLE, dialect="sqlite", target="sqlite", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0, verdict
    assert lite.execute(verdict.target_sql).fetchone() == lite.execute(asked).fetchone() == (0,)


def test_a_day_as_an_in_subject_leaves_the_statement_as_written_on_both_engines():
    """Codex's sixth review: `'D' NOT IN (ts)` beside a moved `ts = 'D'` returned the noon row,
    where read by day both cannot hold. Left as written, it returns nothing on DuckDB and SQLite."""
    import sqlite3

    asked = "SELECT id FROM claim WHERE ts = '2024-03-09' AND '2024-03-09' NOT IN (ts)"
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00')")
    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    lite.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00')")
    for engine, dialect in ((con, "duckdb"), (lite, "sqlite")):
        verdict = decide(asked, VISIBLE, dialect=dialect, target=dialect, types=TYPES)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 0, (dialect, verdict)
        assert engine.execute(verdict.target_sql).fetchall() == engine.execute(asked).fetchall() == [], dialect


def test_on_sqlite_nothing_moves():
    """SQLite has no temporal type: `CAST('2024-03-09' AS TIMESTAMP)` is the integer 2024 there (Codex's
    seventh review), and a column declared TIMESTAMP may hold ISO text, a number or a BLOB. No
    reading of its bounds is proven, so nothing moves, a plain string included, and each statement
    returns what it returns as written."""
    import sqlite3

    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    lite.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00')")
    assert lite.execute("SELECT CAST('2024-03-09' AS TIMESTAMP)").fetchone() == (2024,)
    for asked in ("SELECT id FROM claim WHERE ts = '2024-03-09' AND ts <> CAST('2024-03-09' AS TIMESTAMP)",
                  "SELECT id FROM claim WHERE ts = CAST('2024-03-09' AS TIMESTAMP)",
                  "SELECT id FROM claim WHERE ts <= '2024-03-09'"):
        verdict = decide(asked, VISIBLE, dialect="sqlite", target="sqlite", types=TYPES)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 0, (asked, verdict)
        assert lite.execute(verdict.target_sql).fetchall() == lite.execute(asked).fetchall(), asked


def test_a_next_day_no_timestamp_can_hold_leaves_the_statement_as_written():
    """Codex's eighth review: DuckDB's TIMESTAMP_NS ends on 2262-04-11, and moved to the 12th a cast
    bound was NULL (TRY_CAST, no rows) or an error (CAST). A next day past that, or past the calendar
    (the '9999-12-31' sentinel), leaves the statement as written, and it returns what it returned."""
    types = {"claim": {**TYPES["claim"], "ts": "TIMESTAMP_NS"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP_NS)")
    con.execute("INSERT INTO claim VALUES (1, '2262-04-11 00:00:00'), (2, '2262-04-11 12:00:00')")
    for asked in ("SELECT id FROM claim WHERE ts <= TRY_CAST('2262-04-11' AS TIMESTAMP_NS) ORDER BY id",
                  "SELECT id FROM claim WHERE ts <= CAST('2262-04-11' AS TIMESTAMP_NS) ORDER BY id",
                  "SELECT id FROM claim WHERE ts IN ('2262-04-11') ORDER BY id",
                  "SELECT id FROM claim WHERE ts BETWEEN '2262-04-01' AND '2262-04-11' ORDER BY id"):
        verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=types)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 0, (asked, verdict)
        assert con.execute(verdict.target_sql).fetchall() == con.execute(asked).fetchall(), asked
    # The calendar's last day has no next one (on a TIMESTAMP column; TIMESTAMP_NS cannot hold it).
    sentinel = decide("SELECT id FROM claim WHERE ts <= '9999-12-31'", VISIBLE, dialect="duckdb",
                      target="duckdb", types=TYPES)
    assert sentinel.day_bounds == 0 and "<= '9999-12-31'" in sentinel.target_sql
    # Below the first day TIMESTAMP_NS holds, the bound is NULL as written: rewritten into a NULL-safe
    # range it turned true into unknown and dropped the row (Codex's ninth review).
    plain = duckdb.connect()
    plain.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    plain.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00')")
    for asked in ("SELECT id FROM claim WHERE ts IS DISTINCT FROM TRY_CAST('1600-01-01' AS TIMESTAMP_NS)",
                  "SELECT id FROM claim WHERE ts <> TRY_CAST('1677-09-21' AS TIMESTAMP_NS)"):
        verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 0, (asked, verdict)
        assert plain.execute(verdict.target_sql).fetchall() == plain.execute(asked).fetchall(), asked
    # The NULL-safe form as written returns the row (NULL is distinct from the noon timestamp).
    assert plain.execute("SELECT id FROM claim WHERE ts IS DISTINCT FROM TRY_CAST('1600-01-01' AS TIMESTAMP_NS)"
                         ).fetchall() == [(1,)]
    assert decide("SELECT id FROM claim WHERE ts <= '1677-09-22'", VISIBLE, dialect="duckdb",
                  target="duckdb", types=TYPES).day_bounds == 1
    # The day before the last still moves.
    assert decide("SELECT id FROM claim WHERE ts <= '2262-04-10'", VISIBLE, dialect="duckdb",
                  target="duckdb", types=types).day_bounds == 1


def test_the_tenth_reviews_membership_function_leaves_the_statement_as_written_on_the_engine():
    """Codex's tenth review: `NOT list_contains([ts], TIMESTAMP D)` kept its midnight while `ts = D`
    moved, and the noon row satisfied both. A timestamp inside a function now leaves the statement
    as written, and it returns nothing."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00')")
    asked = "SELECT id FROM claim WHERE ts = '2024-03-09' AND NOT list_contains([ts], TIMESTAMP '2024-03-09')"
    verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0, verdict
    assert con.execute(verdict.target_sql).fetchall() == con.execute(asked).fetchall() == []


def test_the_eleventh_reviews_aggregate_leaves_the_statement_as_written_on_the_engine():
    """Codex's eleventh review: MIN(ts) passed through, and NULLIF compared its result with the day's
    midnight unread, so moving the WHERE bound changed NULL to noon. MIN and MAX now pass through only
    projected bare or ordered; consumed by a function, the statement is left as written."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00')")
    asked = "SELECT NULLIF(MIN(ts), TIMESTAMP '2024-03-09') FROM claim WHERE ts <= '2024-03-09'"
    verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0, verdict
    assert con.execute(verdict.target_sql).fetchall() == con.execute(asked).fetchall() == [(None,)]
    # Projected bare, MIN still lets the bound move.
    assert decide("SELECT MIN(ts) AS m FROM claim WHERE ts <= '2024-03-09' ORDER BY MAX(ts)", VISIBLE,
                  dialect="duckdb", target="duckdb", types=TYPES).day_bounds == 1


def test_the_thirteenth_reviews_except_leaves_the_statement_as_written_on_the_engine():
    """Codex's thirteenth review: EXCEPT compared the noon row with the day's midnight while the WHERE
    bound moved, and the count went from 0 to 1. A set operation that compares rows now leaves the
    statement as written."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00:00')")
    asked = ("SELECT COUNT(*) FROM (SELECT ts FROM claim WHERE ts <= '2024-03-09' "
             "EXCEPT SELECT TIMESTAMP '2024-03-09') AS q")
    verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0, verdict
    assert con.execute(verdict.target_sql).fetchall() == con.execute(asked).fetchall() == [(0,)]


def test_contradictions_stay_contradictions_and_nulls_stay_null_on_the_engine():
    """Codex's counterexamples: `=` moved beside an unmoved IS DISTINCT FROM or NOT IN let a noon
    row satisfy `ts = D AND ts IS DISTINCT FROM D`. Read together by day, each still holds for no
    row, and the NULL-safe pair partitions rows with a NULL among them."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-08 23:00'), (2, '2024-03-09 00:00'), "
                "(3, '2024-03-09 12:00'), (4, '2024-03-10 00:00'), (5, NULL)")
    for contradiction in ("ts = '2024-03-09' AND ts IS DISTINCT FROM '2024-03-09'",
                          "ts = '2024-03-09' AND ts NOT IN ('2024-03-09')",
                          "ts IN ('2024-03-09') AND ts <> '2024-03-09'",
                          "ts IS NOT DISTINCT FROM '2024-03-09' AND ts > '2024-03-09'"):
        verdict = decide(f"SELECT count(*) FROM claim WHERE {contradiction}", VISIBLE, dialect="duckdb",
                         target="duckdb", types=TYPES)
        assert isinstance(verdict, Approved) and verdict.day_bounds == 2, (contradiction, verdict)
        assert con.execute(verdict.target_sql).fetchone()[0] == 0, (contradiction, verdict.target_sql)
    pair = decide("SELECT SUM(CASE WHEN ts IS NOT DISTINCT FROM '2024-03-09' THEN 1 ELSE 0 END), "
                  "SUM(CASE WHEN ts IS DISTINCT FROM '2024-03-09' THEN 1 ELSE 0 END) FROM claim",
                  VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
    assert con.execute(pair.target_sql).fetchone() == (2, 3)  # rows 2 and 3; rows 1, 4 and the NULL
    for null_kept in ("ts = '2024-03-09'", "ts <> '2024-03-09'", "ts NOT IN ('2024-03-09')"):
        verdict = decide(f"SELECT {null_kept} FROM claim WHERE id = 5", VISIBLE, dialect="duckdb",
                         target="duckdb", types=TYPES)
        assert con.execute(verdict.target_sql).fetchone() == (None,), null_kept


def test_the_unions_text_compares_as_written_on_the_engine():
    """Codex's counterexample: rewritten, `'2024-03-09a' < '2024-03-10'` returned two rows where
    the statement as written returns none."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP, note VARCHAR)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-09 12:00', '2024-03-09a')")
    asked = ("WITH x AS (SELECT note AS ts FROM claim UNION ALL SELECT note AS ts FROM claim) "
             "SELECT ts FROM x WHERE ts <= '2024-03-09'")
    verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
    assert isinstance(verdict, Approved) and verdict.day_bounds == 0
    assert con.execute(verdict.target_sql).fetchall() == con.execute(asked).fetchall() == []


def test_complementary_predicates_stay_complementary_on_the_engine():
    """Moving `<=` alone made a `<= D` bucket and a `> D` bucket both count a noon row on D
    (Codex's review). Read by day, each pair partitions the rows: every row in exactly one."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-08 23:00'), (2, '2024-03-09 00:00'), "
                "(3, '2024-03-09 12:00'), (4, '2024-03-10 00:00')")
    for low, high in (("ts <= '2024-03-09'", "ts > '2024-03-09'"),
                      ("ts = '2024-03-09'", "ts <> '2024-03-09'"),
                      ("ts <= '2024-03-09'", "NOT ts <= '2024-03-09'"),
                      ("ts BETWEEN '2024-03-09' AND '2024-03-09'",
                       "NOT ts BETWEEN '2024-03-09' AND '2024-03-09'")):
        asked = (f"SELECT SUM(CASE WHEN {low} THEN 1 ELSE 0 END) AS a, "
                 f"SUM(CASE WHEN {high} THEN 1 ELSE 0 END) AS b FROM claim")
        verdict = decide(asked, VISIBLE, dialect="duckdb", target="duckdb", types=TYPES)
        assert isinstance(verdict, Approved), verdict
        a, b = con.execute(verdict.target_sql).fetchone()
        assert a + b == 4, (low, high, a, b, verdict.target_sql)
    # And by day: the 9th is rows 2 and 3.
    day = decide("SELECT count(*) FROM claim WHERE ts = '2024-03-09'", VISIBLE, dialect="duckdb",
                 target="duckdb", types=TYPES)
    assert con.execute(day.target_sql).fetchone()[0] == 2


@pytest.mark.parametrize("asked", [
    # A DATE holds whole days, where `<=` the day and `<` the next are the same rows.
    "SELECT id FROM claim WHERE filed <= '2024-03-09'",
    # A zone-aware timestamp's midnight is the session's zone's.
    "SELECT id FROM claim WHERE tz <= '2024-03-09'",
    # Text compares as text.
    "SELECT id FROM claim WHERE note <= '2024-03-09'",
    # A column of no declared type, and one the statement does not resolve.
    "SELECT id FROM claim WHERE missing <= '2024-03-09'",
    # An expression is not the column.
    "SELECT id FROM claim WHERE CAST(ts AS DATE) <= '2024-03-09'",
    # A bound with its time of day says where it ends.
    "SELECT id FROM claim WHERE ts <= '2024-03-09 00:00:00'",
    "SELECT id FROM claim WHERE ts <= '2024-03-09 23:59:59'",
    # A string that names no day.
    "SELECT id FROM claim WHERE ts <= '2024-02-30'",
    # The exclusive end and the inclusive start already mean the day boundary they name.
    "SELECT id FROM claim WHERE ts < '2024-03-10'",
    "SELECT id FROM claim WHERE ts >= '2024-03-01'",
    "SELECT id FROM claim WHERE '2024-03-10' > ts",
    # A list holding anything but bare days.
    "SELECT id FROM claim WHERE ts IN ('2024-03-09', '2024-03-10 12:00:00')",
    # A BETWEEN whose end has a time of day.
    "SELECT id FROM claim WHERE ts BETWEEN '2024-03-01' AND '2024-03-09 12:00:00'",
    # Text, in the mirrored and the BETWEEN spellings too.
    "SELECT id FROM claim WHERE '2024-03-09' >= note",
    "SELECT id FROM claim WHERE note BETWEEN '2024-03-01' AND '2024-03-09'",
])
def test_nothing_moves_where_the_bound_is_not_a_bare_day_ending_a_range_on_a_naive_timestamp(asked):
    sql, moved = _closed(asked)
    assert moved == 0
    assert sql == sqlglot.parse_one(asked, read="duckdb").sql(dialect="duckdb")


def test_a_symmetric_between_is_left_as_written():
    asked = "SELECT id FROM claim WHERE ts BETWEEN SYMMETRIC '2024-03-09' AND '2024-03-01'"
    assert _closed(asked, dialect="postgres")[1] == 0


def test_without_declared_types_nothing_moves():
    assert _closed("SELECT id FROM claim WHERE ts <= '2024-03-09'", types=None)[1] == 0


def test_an_oracle_date_holds_a_time_of_day_and_its_bound_moves_in_every_spelling():
    types = {"claim": {"id": "NUMBER", "ts": "DATE", "filed": "DATE", "tz": "TIMESTAMP WITH TIME ZONE",
                       "note": "VARCHAR2(20)", "region": "VARCHAR2(20)"}}
    for asked, closed in [
        ("SELECT id FROM claim WHERE ts <= DATE '2024-03-09'",
         "SELECT id FROM claim WHERE ts < TO_DATE('2024-03-10', 'YYYY-MM-DD')"),
        ("SELECT id FROM claim WHERE ts <= TO_DATE('2024-03-09', 'YYYY-MM-DD')",
         "SELECT id FROM claim WHERE ts < TO_DATE('2024-03-10', 'YYYY-MM-DD')"),
        ("SELECT id FROM claim WHERE ts <= '2024-03-09'",
         "SELECT id FROM claim WHERE ts < '2024-03-10'"),
    ]:
        assert _closed(asked, dialect="oracle", types=types) == (closed, 1), asked
    # A TO_DATE with a time of day says where it ends; one whose format swaps day and month reads
    # '2024-03-09' as 3 September, so the string's next day is not the date's.
    for asked in ("SELECT id FROM claim WHERE ts <= TO_DATE('2024-03-09 10:00', 'YYYY-MM-DD HH24:MI')",
                  "SELECT id FROM claim WHERE ts <= TO_DATE('2024-03-09', 'YYYY-DD-MM')"):
        assert _closed(asked, dialect="oracle", types=types)[1] == 0, asked
    # Elsewhere a DATE is whole days.
    assert _closed("SELECT id FROM claim WHERE ts <= '2024-03-09'", dialect="duckdb", types=types)[1] == 0


def test_the_moved_bound_counts_the_day_whole_on_the_engine():
    """The premise, measured: on DuckDB `<=` the bare day drops the day's later rows, and the moved
    bound keeps them and nothing past the day."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-08 23:00'), (2, '2024-03-09 00:00'), "
                "(3, '2024-03-09 15:30'), (4, '2024-03-10 00:00')")
    asked = "SELECT id FROM claim WHERE ts <= '2024-03-09' ORDER BY id"
    closed, moved = _closed(asked)
    assert moved == 1
    assert [r[0] for r in con.execute(asked).fetchall()] == [1, 2]
    assert [r[0] for r in con.execute(closed).fetchall()] == [1, 2, 3]


def test_the_decider_approves_the_moved_bound_and_counts_it():
    verdict = decide("SELECT id FROM claim WHERE ts <= '2024-03-09'", VISIBLE, types=TYPES)
    assert isinstance(verdict, Approved), verdict
    assert verdict.day_bounds == 1
    # What runs, what the caller is shown and what the model calls read all carry the moved bound.
    for sql in (verdict.target_sql, verdict.asked_sql, verdict.plan_sql, verdict.model_sql()):
        assert "< '2024-03-10'" in sql and "<=" not in sql, sql
    untouched = decide("SELECT id FROM claim WHERE ts < '2024-03-10'", VISIBLE, types=TYPES)
    assert isinstance(untouched, Approved) and untouched.day_bounds == 0


def test_a_move_that_raises_leaves_the_statement_as_written_and_costs_no_decision(monkeypatch):
    """Half done and then raising, the move is discarded whole: the statement runs as written."""
    import mnemiq.sql.decide as decide_module

    def half_then_raise(tree, *a, **k):
        close_day_bounds(tree, *a, **k)
        raise RuntimeError("a shape the walk does not read")

    monkeypatch.setattr(decide_module, "close_day_bounds", half_then_raise)
    verdict = decide("SELECT id FROM claim WHERE ts <= '2024-03-09'", VISIBLE, types=TYPES)
    assert isinstance(verdict, Approved), verdict
    assert verdict.day_bounds is None
    assert "<= '2024-03-09'" in verdict.target_sql and "2024-03-10" not in verdict.target_sql


def test_a_row_policys_own_bound_is_never_moved():
    """The policy grants rows up to the 9th's midnight. Moving its bound would widen what the
    caller reads, so the move runs on the statement as asked, before the policy's predicate is
    added: the model's bound moves, the policy's stays as written."""
    from mnemiq.sql.policy import AccessPolicy

    policy = AccessPolicy(row_filters={"claim": "ts <= '2024-03-09'"})
    verdict = decide("SELECT id FROM claim WHERE ts <= '2024-03-20'", VISIBLE, dialect="duckdb",
                     target="duckdb", policy=policy, types=TYPES)
    assert isinstance(verdict, Approved), verdict
    assert verdict.day_bounds == 1
    assert "ts <= '2024-03-09'" in verdict.target_sql and "'2024-03-10'" not in verdict.target_sql
    assert "ts < '2024-03-21'" in verdict.target_sql
    assert "ts < '2024-03-21'" in verdict.asked_sql and "2024-03-09" not in verdict.asked_sql


# --- through the doors a caller uses -------------------------------------------------------------

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_CARD = "TABLE claim: id INTEGER, region VARCHAR (north|south|east|west), ts TIMESTAMP"
BOUND_SQL = "SELECT count(*) AS n FROM claim WHERE ts <= '2024-03-09'"


class _ApprovingJudgeClient:
    def __init__(self):
        self.calls: list[str] = []

    def complete(self, system, user, max_tokens=None, temperature=None, **_):
        self.calls.append(user)
        return '{"confidence": 0.95}'


@pytest.fixture
def source(tmp_path, monkeypatch):
    for var in ("MNEMIQ_VERIFY", "MNEMIQ_VERIFY_SANITY", "MNEMIQ_VERIFY_BASE_URL",
                "MNEMIQ_VERIFY_MODEL", "MNEMIQ_VERIFY_API_KEY", "MNEMIQ_VERIFY_GROUNDING",
                "MNEMIQ_VERIFY_THRESHOLD", "MNEMIQ_VERIFY_FAIL_CLOSED"):
        monkeypatch.delenv(var, raising=False)
    path = str(tmp_path / "syn.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim (id INTEGER, region VARCHAR, ts TIMESTAMP)")
    # Ten claims a day from the 1st to the 12th, each a quarter past an hour from 0 to 9. Through the
    # 9th is 90; `<=` the bare 9th stops at its midnight, before any of its ten, at 80.
    rows = [(d * 10 + h, ["north", "south", "east", "west"][h % 4], f"2024-03-{d:02d} {h:02d}:15:00")
            for d in range(1, 13) for h in range(10)]
    con.executemany("INSERT INTO claim VALUES (?, ?, ?)", rows)
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    assert adapter.execute("SELECT count(*) FROM claim WHERE ts <= '2024-03-09'")[0][0] == 80
    assert adapter.execute("SELECT count(*) FROM claim WHERE ts < '2024-03-10'")[0][0] == 90
    return adapter, enrich_structural(adapter, "syn")


def test_every_mode_answers_through_the_whole_day_and_says_it_moved_the_bound(source):
    adapter, snap = source
    judge = _ApprovingJudgeClient()
    verifiers, _ = _build_mode_verifiers(Settings(), client=judge)
    for name, mode in MODES.items():
        agent = build_agent(mode, generator=FakeGenerator([json.dumps({"sql": BOUND_SQL})] * 6),
                            synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                            cache=TwoTierCache(L1Cache()), corrector=None, values=None,
                            selector=None, verifier=verifiers[name])
        packet = ContextPacket(question="How many claims were filed as of 2024-03-09?",
                               cards=[RetrievedCard(object_id="claim", card=_CARD, score=1.0)],
                               grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")
        ans = agent.answer(packet, snap, _GRANTS, _IDENTITY)
        assert ans.deferred is False and ans.failed is False, (name, ans.answer)
        payload = answer_payload(ans)
        assert payload["day_bounds_closed"] == 1, name
        assert "< '2024-03-10'" in payload["sql"] and "<=" not in payload["sql"], (name, payload["sql"])
        assert ans.preview.rows == [[90]], (name, ans.preview)
    # The judge read the moved bound, never the one the model wrote.
    assert judge.calls and all("< '2024-03-10'" in c and "<= '2024-03-09'" not in c for c in judge.calls)

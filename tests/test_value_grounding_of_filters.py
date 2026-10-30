"""M146: what an empty result rests on, one string value at a time.

An empty result is answered only when every string value it filters on is grounded in the value
index. That is an allow-list: a value nothing could check declines it exactly as one the index
says is absent, because a misspelled value on a column the index does not cover is the wrong
"none" the no-rows rule existed to stop, and it is unknown rather than absent. So the outcomes
are kept apart -- grounded, absent, unchecked, and typed (a well-formed date or number, which has
no spelling to check) -- and only the reason a value is unchecked is told apart further, for the
sentence that names it. The column a value meets is placed by `check_values`'s own resolver, so a
CTE's column list renaming by position is read as the engine reads it.
"""

from __future__ import annotations

import pyarrow as pa
import sqlglot

from mnemiq.sql.values_check import ABSENT, GROUNDED, TYPED, UNCHECKED, ground_values
from mnemiq.verify.sanity import answered_note, sanity_check


class _Values:
    """The value index's read side, recording what it was asked."""

    def __init__(self, columns: dict[tuple[str, str], set[str]]):
        self.columns = columns
        self.asked: list[tuple[str, str, str]] = []

    def has(self, table, column):
        return (table, column) in self.columns

    def contains(self, table, column, literal):
        self.asked.append((table, column, literal))
        return literal in self.columns.get((table, column), set())


_VISIBLE = {"claim": {"id", "region", "status", "amount", "adjuster", "filed", "note", "ts", "tz",
                      "at", "tags"},
            "office": {"id", "region", "city"}}
_TYPES = {"claim": {"id": "INTEGER", "region": "VARCHAR", "status": "VARCHAR",
                    "amount": "DOUBLE", "adjuster": "VARCHAR", "filed": "DATE", "note": "VARCHAR",
                    "ts": "TIMESTAMP", "tz": "TIMESTAMP WITH TIME ZONE", "at": "TIME",
                    "tags": "VARCHAR[]"},
          "office": {"id": "INTEGER", "region": "VARCHAR", "city": "VARCHAR"}}


def _values() -> _Values:
    return _Values({("claim", "region"): {"north", "south", "east", "west"},
                    ("claim", "status"): {"open", "closed"},
                    ("office", "region"): {"north", "south"}})


def _ground(sql: str, dialect: str = "duckdb", values=None, masks=None, types=_TYPES,
            row_filtered=None):
    filtered = {} if row_filtered is None else {"row_filtered": row_filtered}
    return ground_values(sqlglot.parse_one(sql, read=dialect), _VISIBLE,
                         _values() if values is None else values, masks=masks, dialect=dialect,
                         types=types, **filtered)


def _read(sql: str, **kw) -> list[tuple]:
    return [(v.text, v.outcome, v.why, v.column) for v in _ground(sql, **kw).values]


# --- one literal class at a time ------------------------------------------------------------------

def test_a_value_its_indexed_column_holds_is_grounded():
    assert _read("SELECT id FROM claim WHERE status = 'open'") == [
        ("open", GROUNDED, "", "claim.status")]


def test_each_member_of_an_in_list_is_grounded_on_its_own():
    assert _read("SELECT id FROM claim WHERE region IN ('north', 'nrth')") == [
        ("north", GROUNDED, "", "claim.region"), ("nrth", ABSENT, "", "claim.region")]


def test_a_value_its_indexed_column_lacks_is_absent_not_unchecked():
    # `<>` is not refused before the run, so an absent value can reach an empty result there.
    assert _read("SELECT id FROM claim WHERE status <> 'opne'") == [
        ("opne", ABSENT, "", "claim.status")]


def test_a_value_on_an_unindexed_column_is_unchecked_not_absent():
    assert _read("SELECT id FROM claim WHERE adjuster = 'agnet_0042'") == [
        ("agnet_0042", UNCHECKED, "not_indexed", "claim.adjuster")]


def test_a_masked_column_is_unchecked_and_its_values_are_never_consulted():
    values = _values()
    read = _read("SELECT id FROM claim WHERE status = 'open'", values=values,
                 masks=lambda table, column: column == "status")
    assert read == [("open", UNCHECKED, "masked", "claim.status")]
    assert values.asked == []  # "does it exist" is what the mask withholds


def test_a_pattern_is_unchecked():
    assert _read("SELECT id FROM claim WHERE adjuster LIKE 'agn%'") == [
        ("agn%", UNCHECKED, "pattern", "claim.adjuster")]


def test_a_text_range_is_unchecked():
    assert _read("SELECT id FROM claim WHERE status BETWEEN 'a' AND 'm'") == [
        ("a", UNCHECKED, "range", "claim.status"), ("m", UNCHECKED, "range", "claim.status")]


def test_a_value_compared_with_an_expression_is_unchecked():
    assert _read("SELECT id FROM claim WHERE lower(status) = 'open'") == [
        ("open", UNCHECKED, "not_a_column", None)]


def test_a_value_compared_with_nothing_is_unchecked():
    assert _read("SELECT id FROM claim WHERE concat(note, 'x') IS NOT NULL") == [
        ("x", UNCHECKED, "not_compared", None)]


def test_a_column_the_resolver_cannot_place_leaves_its_value_unchecked():
    # `region` is on both sources: ambiguous, so no column is checked.
    assert _read("SELECT 1 FROM claim, office WHERE region = 'north'") == [
        ("north", UNCHECKED, "unresolved", None)]


def test_with_no_value_index_every_string_value_is_unchecked():
    read = [(v.text, v.outcome) for v in ground_values(
        sqlglot.parse_one("SELECT id FROM claim WHERE status = 'open'", read="duckdb"), _VISIBLE,
        None, dialect="duckdb", types=_TYPES).values]
    assert read == [("open", UNCHECKED)]


def test_numbers_need_no_grounding():
    g = _ground("SELECT id FROM claim WHERE amount > 1000000 AND id = 5")
    assert g.values == () and g.never_true is None


# --- dates: a well-formed date read as one is a bound, like a number ------------------------------

def test_a_typed_date_literal_needs_no_grounding():
    assert _read("SELECT id FROM claim WHERE filed > DATE '2030-01-01'") == [
        ("2030-01-01", TYPED, "", "claim.filed")]


def test_an_iso_date_against_a_date_column_needs_no_grounding():
    assert _read("SELECT id FROM claim WHERE filed BETWEEN '2031-01-01' AND '2031-12-31'") == [
        ("2031-01-01", TYPED, "", "claim.filed"), ("2031-12-31", TYPED, "", "claim.filed")]
    assert _read("SELECT id FROM claim WHERE filed = '2031-01-01'") == [
        ("2031-01-01", TYPED, "", "claim.filed")]


def test_a_malformed_date_is_unchecked_since_an_engine_comparing_no_row_never_reads_it():
    # Measured on DuckDB 1.5.4: `d = 'garbage'` and even `DATE 'garbage' < d` return no rows,
    # without an error, when no row reaches the comparison -- the empty case exactly.
    assert _read("SELECT id FROM claim WHERE filed > '2031/13/45'")[0][1] == UNCHECKED
    assert _read("SELECT id FROM claim WHERE filed > CAST('garbage' AS DATE)")[0][1] == UNCHECKED


def test_without_column_types_a_bare_date_string_is_a_text_range():
    assert _read("SELECT id FROM claim WHERE filed >= '2031-01-01'", types=None) == [
        ("2031-01-01", UNCHECKED, "range", "claim.filed")]


def test_on_sqlite_a_date_column_holds_text_so_a_date_string_is_a_text_range():
    assert _read("SELECT id FROM claim WHERE filed >= '2031-01-01'", dialect="sqlite") == [
        ("2031-01-01", UNCHECKED, "range", "claim.filed")]


def test_an_interval_needs_no_grounding():
    assert [o for _, o, _, _ in _read(
        "SELECT id FROM claim WHERE filed > current_date - INTERVAL '30 days'")] == [TYPED]


# --- a date-only string is exempt against a DATE column alone (fix rounds 1 and 2) ----------------
#
# Measured on DuckDB 1.5.4: `ts = '2024-03-03'` on a TIMESTAMP column returns no rows while 15 rows
# fall on that day (the string is midnight), and on a TIME column `t < '2024-03-03'` reads the string
# as 00:00, so it is never true. The cost, accepted: a true "none" filtered by a date range on a
# TIMESTAMP column (`ts >= '2031-01-01'`) is declined too, naming the date, since a date-only bound
# on a timestamp cannot be told from the midnight mistake. Fix round 2 found the same bypass one
# level deeper -- a DATE literal compared with an expression of timestamp type -- so a date is a
# bound only where its partner is a bare column proven DATE, and nowhere else.

def test_a_date_against_anything_but_a_bare_date_column_is_unchecked():
    for sql in ("SELECT id FROM claim WHERE CAST(ts AS TIMESTAMP) = DATE '2024-03-03'",
                "SELECT id FROM claim WHERE greatest(ts, ts) = DATE '2024-03-03'",
                "SELECT id FROM claim WHERE CAST(filed AS DATE) > DATE '2031-01-01'",
                "SELECT id FROM claim WHERE DATE '2031-01-01' < current_date",
                "SELECT id FROM claim WHERE CAST(filed AS DATE) > current_date - INTERVAL '30 days'"):
        assert [o for _, o, _, _ in _read(sql)] == [UNCHECKED], sql
    # A column whose declared type is unknown is not proven DATE.
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE filed > DATE '2031-01-01'",
                                      types=None)] == [UNCHECKED]


def test_a_full_timestamp_against_a_bare_timestamp_column_needs_no_grounding():
    """A bound of the column's own precision is as safe as a number (the controller's addition to
    fix round 2): a week written to the second on a TIMESTAMP column is a common genuine "none"."""
    for sql in ("SELECT id FROM claim WHERE ts >= '2031-01-01 00:00:00' AND ts < '2031-01-08 00:00:00'",
                "SELECT id FROM claim WHERE ts > TIMESTAMP '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE ts > CAST('2031-01-01 00:00:00' AS TIMESTAMP)",
                "SELECT id FROM claim WHERE tz > '2031-01-01T10:30:00+02:00'",
                "SELECT id FROM claim WHERE tz > TIMESTAMPTZ '2031-01-01 00:00:00+00'",
                "SELECT id FROM claim WHERE TIMESTAMP '2031-01-01 00:00:00' BETWEEN ts AND ts"):
        assert {o for _, o, _, _ in _read(sql)} == {TYPED}, sql


def test_a_full_timestamp_against_anything_but_a_bare_timestamp_column_is_unchecked():
    for sql in ("SELECT id FROM claim WHERE CAST(ts AS TIMESTAMP) > TIMESTAMP '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE CAST(ts AS TIMESTAMP) > '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE filed > TIMESTAMP '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE filed > '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE at > '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE ts > TIMESTAMP '2031-01-01'"):  # no time of day
        assert [o for _, o, _, _ in _read(sql)] == [UNCHECKED], sql
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE ts > '2031-01-01 00:00:00'",
                                      types=None)] == [UNCHECKED]
    # Named for what it is: a DATE column, not a zone mismatch.
    assert _read("SELECT id FROM claim WHERE filed > '2031-01-01 00:00:00'") == [
        ("2031-01-01 00:00:00", UNCHECKED, "not_on_a_timestamp_column", "claim.filed")]


def test_a_full_timestamp_is_a_bound_only_as_a_range_bound():
    """Fix round 3: `ts = '2024-03-03 00:00:00'` is a disguised day filter -- a midday row exists,
    the result is empty, and it passed. Equality, `<>` and IN are never bounds; a half-open range
    (a week written to the second) stays one."""
    for sql in ("SELECT id FROM claim WHERE ts = '2024-03-03 00:00:00'",
                "SELECT id FROM claim WHERE ts <> '2024-03-03 00:00:00'",
                "SELECT id FROM claim WHERE ts IN ('2024-03-03 00:00:00')",
                "SELECT id FROM claim WHERE ts = TIMESTAMP '2024-03-03 00:00:00'",
                "SELECT id FROM claim WHERE TIMESTAMP '2024-03-03 00:00:00' IN (ts)"):
        assert [o for _, o, _, _ in _read(sql)] == [UNCHECKED], sql
    assert {o for _, o, _, _ in _read(
        "SELECT id FROM claim WHERE ts >= '2031-01-01 00:00:00' AND ts < '2031-01-08 00:00:00'"
    )} == {TYPED}
    v = sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE ts = '2024-03-03 00:00:00'"))
    assert ("the timestamp '2024-03-03 00:00:00' compared for equality with claim.ts could not be "
            "checked as a bound") in v.reason


def test_a_timestamptz_column_takes_only_a_literal_with_an_explicit_offset():
    """A naive literal against TIMESTAMPTZ reads in the session's zone, measured on DuckDB 1.5.4:
    the same row and the same naive range give 0 rows under UTC and 1 under New York, where an
    offset range gives 0 under both. So the column's zone-awareness must match the literal's."""
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE z (tz TIMESTAMPTZ)")
    con.execute("INSERT INTO z VALUES ('2024-03-04 02:00:00+00')")
    naive = "SELECT count(*) FROM z WHERE tz >= '2024-03-03 00:00:00' AND tz < '2024-03-04 00:00:00'"
    aware = naive.replace(":00'", ":00+00'")
    counts = {}
    for zone in ("UTC", "America/New_York"):
        con.execute(f"SET TimeZone = '{zone}'")
        counts[zone] = (con.execute(naive).fetchone()[0], con.execute(aware).fetchone()[0])
    assert counts == {"UTC": (0, 0), "America/New_York": (1, 0)}, counts  # the premise
    assert {o for _, o, _, _ in _read(naive.replace("count(*) FROM z", "id FROM claim"))} == {
        UNCHECKED}
    assert {o for _, o, _, _ in _read(aware.replace("count(*) FROM z", "id FROM claim"))} == {TYPED}
    # The other way round: an offset literal against a naive TIMESTAMP column, and a cast whose
    # target's zone-awareness differs from its text's, are no bounds either.
    for sql in ("SELECT id FROM claim WHERE ts > '2031-01-01 00:00:00+00'",
                "SELECT id FROM claim WHERE tz > TIMESTAMP '2031-01-01 00:00:00'",
                "SELECT id FROM claim WHERE tz > TIMESTAMPTZ '2031-01-01 00:00:00'"):
        assert [o for _, o, _, _ in _read(sql)] == [UNCHECKED], sql


def test_a_date_literal_that_is_the_subject_of_between_reads_its_bounds():
    assert _read("SELECT id FROM claim WHERE DATE '2024-03-03' BETWEEN ts AND ts") == [
        ("2024-03-03", UNCHECKED, "date_on_time", "claim.ts")]
    assert _read("SELECT id FROM claim WHERE DATE '2024-03-03' BETWEEN filed AND filed") == [
        ("2024-03-03", TYPED, "", "claim.filed")]
    assert [o for _, o, _, _ in _read(
        "SELECT id FROM claim WHERE DATE '2024-03-03' BETWEEN filed AND ts")] == [UNCHECKED]

def test_a_date_only_string_against_a_timestamp_column_is_unchecked():
    for sql in ("SELECT id FROM claim WHERE ts = '2024-03-03'",
                "SELECT id FROM claim WHERE ts IN ('2024-03-03', '2024-03-04')",
                "SELECT id FROM claim WHERE ts BETWEEN '2024-03-03' AND '2024-03-03'",
                "SELECT id FROM claim WHERE tz = '2024-03-03'"):
        assert {o for _, o, _, _ in _read(sql)} == {UNCHECKED}, sql


def test_the_half_open_premise_measured_on_duckdb():
    """A date-only string against a TIMESTAMP column is midnight. As the inclusive start or the
    exclusive end of a range that is what the range means, and the day is counted whole; as an
    inclusive end it drops the day's hours."""
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t (ts TIMESTAMP)")
    con.execute("INSERT INTO t VALUES ('2024-03-03 10:00:00')")
    count = lambda where: con.execute(f"SELECT count(*) FROM t WHERE {where}").fetchone()[0]  # noqa: E731
    assert count("ts >= '2024-03-03' AND ts < '2024-03-04'") == 1
    assert count("'2024-03-03' <= ts AND '2024-03-04' > ts") == 1
    assert count("ts >= '2024-03-03' AND ts <= '2024-03-03'") == 0
    assert count("ts BETWEEN '2024-03-03' AND '2024-03-03'") == 0
    assert count("ts > '2024-03-03'") == 1  # "after the 3rd" that keeps the 3rd's hours


def test_a_date_only_string_bounding_a_timestamp_half_open_is_a_bound():
    """The inclusive start (`>=`) or the exclusive end (`<`) of a range on a naive TIMESTAMP column,
    written on either side: an empty result there is honest, so the date is a bound."""
    for sql in ("SELECT id FROM claim WHERE ts >= '2024-03-03' AND ts < '2024-03-04'",
                "SELECT id FROM claim WHERE '2024-03-03' <= ts AND '2024-03-04' > ts",
                "SELECT id FROM claim WHERE ts >= '2031-01-01'",
                "SELECT id FROM claim WHERE ts < '1990-01-01'"):
        assert {(o, w) for _, o, w, _ in _read(sql)} == {(TYPED, "")}, sql


def test_a_date_only_string_that_misreads_a_day_on_a_timestamp_stays_unchecked():
    """An inclusive end (`<=`, BETWEEN) drops the day's hours, an exclusive start (`>`) keeps them,
    and `=`, `<>` and IN compare with midnight alone."""
    for sql in ("SELECT id FROM claim WHERE ts <= '2024-03-03'",
                "SELECT id FROM claim WHERE ts > '2024-03-03'",
                "SELECT id FROM claim WHERE '2024-03-03' >= ts",
                "SELECT id FROM claim WHERE '2024-03-03' < ts",
                "SELECT id FROM claim WHERE ts BETWEEN '2024-03-03' AND '2024-03-04'",
                "SELECT id FROM claim WHERE ts = '2024-03-03'",
                "SELECT id FROM claim WHERE ts <> '2024-03-03'",
                "SELECT id FROM claim WHERE ts IN ('2024-03-03')"):
        assert {(o, w) for _, o, w, _ in _read(sql)} == {(UNCHECKED, "date_on_time")}, sql


def test_a_half_open_date_on_a_zone_aware_or_a_time_column_stays_unchecked():
    """On TIMESTAMPTZ a date-only string is midnight in the session's zone, so where its day starts
    is the session's, not the question's; a TIME column holds no day at all."""
    for sql in ("SELECT id FROM claim WHERE tz >= '2024-03-03'",
                "SELECT id FROM claim WHERE tz < '2024-03-04'",
                "SELECT id FROM claim WHERE at < '2024-03-04'"):
        assert {o for _, o, _, _ in _read(sql)} == {UNCHECKED}, sql


def test_a_half_open_bound_is_a_plain_date_string_against_a_bare_column_only():
    """A DATE literal, a cast column or an engine that compares a date string as text keeps the
    rule as it was."""
    for sql, dialect in (("SELECT id FROM claim WHERE ts >= DATE '2024-03-03'", "duckdb"),
                         ("SELECT id FROM claim WHERE CAST(ts AS TIMESTAMP) >= '2024-03-03'", "duckdb"),
                         ("SELECT id FROM claim WHERE ts >= '2024-03-03'", "sqlite")):
        assert {o for _, o, _, _ in _read(sql, dialect=dialect)} == {UNCHECKED}, (sql, dialect)


def test_a_date_only_string_against_a_time_column_is_unchecked():
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE at < '2024-03-03'")] == [UNCHECKED]


def test_a_string_with_a_time_part_against_a_date_column_is_unchecked():
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE filed = '2024-03-03 10:00'")] == [
        UNCHECKED]


def test_a_typed_date_literal_against_a_timestamp_column_is_unchecked():
    """`ts = DATE '2024-03-03'` is midnight too. (A TIMESTAMP literal with its time of day against a
    bare TIMESTAMP column is a bound since the controller's addition to fix round 2.)"""
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE ts = DATE '2024-03-03'")] == [
        UNCHECKED]


def test_a_date_column_and_a_typed_date_literal_stay_exempt():
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE filed = '2024-03-03'")] == [TYPED]
    assert [o for _, o, _, _ in _read("SELECT id FROM claim WHERE filed > DATE '2030-01-01'")] == [
        TYPED]


# --- a row-filtered table: never the unfiltered index (fix round 1) ------------------------------
#
# The index is built without the row filter (M5). Answering an empty result on its say-so turns a
# `<>` or NOT IN into a membership test on rows the caller cannot read: a value present only in
# hidden rows read GROUNDED and the empty result was answered, an absent guess read ABSENT and was
# declined with "does not appear". So nothing on such a table is grounded, and both read alike.

_HIDDEN, _ABSENT_GUESS = "south", "sooth"  # present only in rows the filter hides; nowhere


def _filtered(template: str, value: str, keyed: str = "claim"):
    values = _values()
    g = _ground(template.format(v=value), values=values, row_filtered={keyed})
    return g, values.asked


def test_on_a_row_filtered_table_a_hidden_value_and_an_absent_guess_read_alike():
    for template in ("SELECT id FROM claim WHERE region = '{v}'",
                     "SELECT id FROM claim WHERE region IN ('north', '{v}')",
                     "SELECT id FROM claim WHERE region <> 'north' AND region <> '{v}'",
                     "SELECT id FROM claim WHERE region NOT IN ('north', '{v}')"):
        hidden, asked_hidden = _filtered(template, _HIDDEN)
        guess, asked_guess = _filtered(template, _ABSENT_GUESS)
        read = [(v.outcome, v.why, v.column) for v in hidden.values]
        assert read == [(v.outcome, v.why, v.column) for v in guess.values], template
        assert {o for o, _, _ in read} == {UNCHECKED} and {w for _, w, _ in read} == {"row_filtered"}
        assert asked_hidden == asked_guess == [], "the unfiltered index was consulted"
        declined_hidden = sanity_check("q", _EMPTY, hidden)
        declined_guess = sanity_check("q", _EMPTY, guess)
        assert declined_hidden.defer and declined_guess.defer, template
        assert (declined_hidden.reason.replace(_HIDDEN, "?")
                == declined_guess.reason.replace(_ABSENT_GUESS, "?")), template
        assert "does not appear" not in declined_guess.reason


def test_the_row_filtered_set_is_folded_as_check_values_folds_it():
    g, asked = _filtered("SELECT id FROM claim WHERE region <> '{v}'", _HIDDEN, keyed="CLAIM")
    assert [(v.outcome, v.why) for v in g.values] == [(UNCHECKED, "row_filtered")] and asked == []


def test_a_date_on_a_row_filtered_table_is_unchecked_too():
    g, _ = _filtered("SELECT id FROM claim WHERE filed > DATE '{v}'", "2031-01-01")
    assert [(v.outcome, v.why) for v in g.values] == [(UNCHECKED, "row_filtered")]


def test_another_table_beside_a_row_filtered_one_is_still_grounded():
    g, _ = _filtered("SELECT o.id FROM office o JOIN claim c ON c.id = o.id "
                     "WHERE o.region = '{v}' AND c.amount > 9e9", "north")
    assert [(v.text, v.outcome) for v in g.values] == [("north", GROUNDED)]


# --- no output exemption: a value anywhere counts (fix round 2) ----------------------------------
#
# Fix round 1 exempted a projected value that provably removed no rows, and each review found it
# bypassed one level deeper: a set-returning call, an alias read in WHERE, then an alias read by a
# sibling item that WHERE, QUALIFY or a set-returning call reads. An exemption that has to model
# every way a projection can reach the rows does not converge, so there is none: every string value
# anywhere in the statement is grounded or typed, or the empty result is declined naming it.

def test_every_projected_value_counts_however_it_reaches_the_rows():
    for sql in ("SELECT unnest(list_filter(tags, x -> x = 'opne')) AS t FROM claim",
                "SELECT string_to_table(note, 'opne') AS piece FROM claim",
                "SELECT upper(status) = 'OPNE' AS flag FROM claim WHERE flag",
                "SELECT status = 'opne' AS x, x AS y FROM claim WHERE y",
                "SELECT id, adjuster = 'agnet_0042' AS f, f AS g FROM claim WHERE g",
                "SELECT list_filter(tags, x -> x = 'opne') AS l, unnest(l) AS t FROM claim",
                "SELECT adjuster = 'agnet_0042' AS x, unnest(CASE WHEN x THEN [1] ELSE [] END) AS u "
                "FROM claim",
                "SELECT CASE WHEN adjuster = 'agnet_0042' THEN 'x' END AS k FROM claim "
                "WHERE amount > 9e9 ORDER BY k"):
        v = sanity_check("q", _EMPTY, _ground(sql))
        assert v is not None and v.defer, sql


def test_a_constant_written_other_than_as_a_plain_string_is_never_grounded():
    """Fix round 3: sqlglot reads E'..' as ByteString, N'..' as National and $$..$$ as RawString,
    and a scan of plain literals saw none of them: `status = E'opne'` ran, came back empty, and
    passed with nothing to name. Any constant other than a numeric or plain string literal, a
    boolean or NULL -- a parameter or placeholder included -- is unchecked wherever it sits."""
    for sql, named in (("SELECT id FROM claim WHERE status = E'open'", "open"),
                       ("SELECT id FROM claim WHERE status = N'open'", "open"),
                       ("SELECT id FROM claim WHERE status = $$open$$", "open"),
                       ("SELECT N'open' AS s FROM claim WHERE amount > 9e9", "open"),
                       ("SELECT id FROM claim WHERE id IN (SELECT id FROM claim WHERE "
                        "status = E'open')", "open"),
                       ("SELECT id FROM claim WHERE status = ?", "?")):
        g = _ground(sql)
        assert [(v.outcome, v.why) for v in g.values] == [(UNCHECKED, "not_a_plain_string")], sql
        v = sanity_check("q", _EMPTY, g)
        assert v is not None and v.defer and named in v.reason, (sql, v and v.reason)
    pg = _ground("SELECT id FROM claim WHERE status = $1", dialect="postgres")
    assert [(v.outcome, v.why) for v in pg.values] == [(UNCHECKED, "not_a_plain_string")]


def test_a_json_path_is_one_unchecked_constant_whatever_its_leaves():
    """Fix round 4: a root, an integer subscript, a wildcard, a slice or a union carry no text, and
    the constant scan passed them as no value at all. The whole path is classified, never its
    parts: one unchecked value per path, named as written."""
    for path in ("$[9]", "$", "$[*]", "$[1:3]", "$[1,2]", "$.a.b"):
        g = _ground(f"SELECT id FROM claim WHERE (note->'{path}') IS NOT NULL")
        assert [(v.outcome, v.why) for v in g.values] == [(UNCHECKED, "json_path")], path
        v = sanity_check("q", _EMPTY, g)
        assert v is not None and v.defer and "a JSON path" in v.reason, (path, v and v.reason)


def test_a_grounded_projected_value_is_fine():
    assert sanity_check("q", _EMPTY, _ground(
        "SELECT CASE WHEN status = 'open' THEN 1 END AS o FROM claim WHERE amount > 9e9")) is None


# --- a predicate that can never hold -------------------------------------------------------------

def test_a_predicate_that_can_never_hold_is_recorded():
    assert _ground("SELECT id FROM claim WHERE 1 = 0").never_true == "1 = 0"
    assert _ground("SELECT id FROM claim WHERE FALSE").never_true == "FALSE"
    assert _ground("SELECT id FROM claim WHERE status = NULL").never_true == "status = NULL"
    assert _ground("SELECT id FROM claim LIMIT 0").never_true == "LIMIT 0"
    assert _ground("SELECT id FROM claim WHERE 1 = 1 AND amount > 5").never_true is None
    # No output exemption (fix round 2): a projection read by alias can be the filter.
    assert _ground("SELECT 1 = 0 AS flag FROM claim").never_true == "1 = 0"


# --- the projection is read like the rest; the scope's resolver places the column -----------------

def test_a_value_in_the_projection_is_checked_like_any_other():
    g = _ground("SELECT CASE WHEN status = 'open' THEN 'Open' END AS s FROM claim "
                "WHERE amount > 1000000000")
    assert [(v.text, v.outcome) for v in g.values] == [("open", GROUNDED), ("Open", UNCHECKED)]
    assert [v.text for v in g.blocking()] == ["Open"]


def test_a_subquerys_filter_inside_the_projection_counts_and_is_written_out():
    g = _ground("SELECT id, (SELECT max(amount) FROM claim c2 WHERE c2.status = 'opne') AS m "
                "FROM claim WHERE amount > 1000000000")
    assert [(v.text, v.outcome) for v in g.values] == [("opne", ABSENT)]
    assert g.filters == ("amount > 1000000000", "c2.status = 'opne'")


def test_a_cte_column_list_is_read_by_position_as_check_values_reads_it():
    """`x.region` is the CTE's FIRST projection, `claim.status`: a resolver matching by name would
    check 'open' against claim.region and call it absent."""
    assert _read("WITH x(region, status) AS (SELECT status, region FROM claim) "
                 "SELECT * FROM x WHERE x.region = 'open'") == [
        ("open", GROUNDED, "", "claim.status")]


def test_the_filters_are_rendered_as_asked():
    g = _ground("SELECT id FROM claim WHERE status = 'open' AND amount > 1000000")
    assert g.filters == ("status = 'open' AND amount > 1000000",)


def test_a_filter_inside_another_is_written_out_by_it_and_a_ctes_is_its_own():
    g = _ground("WITH big AS (SELECT id FROM claim WHERE amount > 9e9) SELECT id FROM big "
                "WHERE id IN (SELECT id FROM claim WHERE status = 'open')")
    assert g.filters == ("id IN (SELECT id FROM claim WHERE status = 'open')", "amount > 9e9")


# --- the decider records it, on the statement as asked -------------------------------------------

def test_the_decider_reads_the_statement_as_asked_never_the_policys_rewrite():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, Snapshot
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import build_access_policy

    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id=f"claim.{n}", object_id="claim", name=n) for n in ("id", "region", "status")])
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "region = 'north'"})
    verdict = decide("SELECT id FROM claim WHERE status = 'open'",
                     {"claim": {"id", "region", "status"}}, values=_values(),
                     policy=build_access_policy(snap, grants), target="duckdb")
    assert "north" in verdict.target_sql  # the control: the policy really applied
    g = verdict.value_grounding
    # The policy's own 'north' is not read, and the table it filters is never looked up.
    assert [(v.text, v.outcome, v.why) for v in g.values] == [("open", UNCHECKED, "row_filtered")]
    assert g.filters == ("status = 'open'",)


def test_a_walk_that_raises_costs_the_empty_answer_never_the_decision(monkeypatch):
    import mnemiq.sql.decide as decide_mod

    def _raises(*_a, **_k):
        raise RuntimeError("a shape the walk did not expect")

    monkeypatch.setattr(decide_mod, "ground_values", _raises)
    verdict = decide_mod.decide("SELECT id FROM claim WHERE status = 'open'",
                                {"claim": {"id", "status"}}, values=_values(), target="duckdb")
    assert verdict.plan_sql and verdict.value_grounding is None
    assert sanity_check("q", pa.table({"id": pa.array([], type=pa.int64())}),
                        verdict.value_grounding).defer


# --- the sanity rule on top of it ----------------------------------------------------------------

_EMPTY = pa.table({"id": pa.array([], type=pa.int64())})


def test_an_empty_result_whose_values_are_all_grounded_passes():
    assert sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE status = 'open'")) is None


def test_an_empty_result_on_numbers_alone_passes():
    assert sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE amount > 9e9")) is None


def test_an_empty_result_with_an_unchecked_value_defers_and_names_it():
    v = sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE adjuster = 'agnet_0042' "
                                          "AND status = 'open'"))
    assert v is not None and v.defer and v.layer == "sanity" and v.confidence == 0.0
    assert v.reason == ("No rows matched, and 'agnet_0042' could not be checked against "
                        "claim.adjuster's known values, so I can't tell a true \"none\" from a "
                        "misspelled value.")


def test_values_sharing_a_reason_and_a_column_are_named_together():
    v = sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE adjuster IN ('a1', 'a2') "
                                          "AND lower(status) = 'x'"))
    assert v.reason == ("No rows matched, and 'a1' and 'a2' could not be checked against "
                        "claim.adjuster's known values; 'x', compared with an expression rather "
                        "than a column, could not be checked, so I can't tell a true \"none\" from "
                        "a misspelled value.")


def test_an_empty_result_with_an_absent_value_defers_and_says_so():
    v = sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE status <> 'opne'"))
    assert v is not None and v.defer
    assert "'opne' does not appear among claim.status's known values" in v.reason


def test_an_empty_result_from_a_predicate_that_can_never_hold_defers():
    v = sanity_check("q", _EMPTY, _ground("SELECT id FROM claim WHERE 1 = 0"))
    assert v is not None and v.defer
    assert v.reason == "No rows matched, because the query's filter can never be true (1 = 0)."


def test_an_empty_result_and_a_null_both_read_the_projections_values():
    sql = "SELECT CASE WHEN status = 'open' THEN 'Open' END AS s FROM claim WHERE amount > 9e9"
    for table in (_EMPTY, pa.table({"s": [None]})):
        v = sanity_check("q", table, _ground(sql))
        assert v is not None and v.defer and "'Open'" in v.reason


def test_a_typed_value_is_named_for_what_it_is():
    """The task review's re-review of fix round 1: every cast was worded as a date, so a TIME
    literal read "the date '10:00:00'", and a TIMESTAMP literal against a DATE column read
    "compared with no column"."""
    reasons = {
        "SELECT id FROM claim WHERE at < TIME '10:00:00'":
            "'10:00:00', typed as something other than a date or a timestamp, could not be checked "
            "as a bound",
        "SELECT id FROM claim WHERE filed > TIMESTAMP '2031-01-01 00:00:00'":
            "the timestamp '2031-01-01 00:00:00', not compared with a column known to hold "
            "timestamps, could not be checked as a bound",
        "SELECT id FROM claim WHERE ts > TIMESTAMP '2031-01-01'":
            "'2031-01-01', not a well-formed timestamp with a time of day, could not be checked",
        "SELECT id FROM claim WHERE greatest(ts, ts) = DATE '2024-03-03'":
            "the date '2024-03-03', not compared with a column known to hold dates alone, could "
            "not be checked as a bound",
        "SELECT id FROM claim WHERE filed > DATE '2031-02-30'":
            "'2031-02-30', not a well-formed date, could not be checked",
        "SELECT id FROM claim WHERE ts = DATE '2024-03-03'":
            "the date '2024-03-03' against claim.ts, which holds times as well as dates, could not "
            "be checked as a bound",
    }
    for sql, phrase in reasons.items():
        v = sanity_check("q", _EMPTY, _ground(sql))
        assert v is not None and phrase in v.reason, (sql, v and v.reason)


def test_a_null_total_whose_values_are_grounded_passes_and_a_nan_does_not():
    g = _ground("SELECT sum(amount) FROM claim WHERE status = 'open' AND amount > 9e9")
    assert sanity_check("q", pa.table({"s": [None]}), g) is None
    assert sanity_check("q", pa.table({"s": [None], "t": [None]}), g) is None  # several aggregates
    assert sanity_check("q", pa.table({"s": [float("nan")]}), g).defer
    # Several rows holding nothing is no aggregate over nothing: still declined.
    assert sanity_check("q", pa.table({"s": [None, None]}), g).defer


def test_a_null_the_query_selects_as_written_is_no_aggregate_over_nothing():
    """Measured on the saved BIRD runs: `SELECT NULL, NULL` is a model saying "no answer" in SQL,
    and with no value to ground it passed as one NULL row."""
    row = pa.table({"a": [None], "b": [None]})
    v = sanity_check("q", row, _ground("SELECT NULL, NULL LIMIT 1000"))
    assert v is not None and v.defer
    assert v.reason == ("The query produced an empty value, because it selects NULL as written "
                        "(NULL), not a value from the data.")
    v = sanity_check("q", row, _ground("SELECT max(amount), CAST(NULL AS INT) AS x FROM claim"))
    assert v is not None and v.defer and "CAST(NULL AS INT) AS x" in v.reason


def test_with_no_grounding_recorded_the_rules_are_as_before():
    """`None` is "not evaluated" -- an `Approved` built outside the decider -- and declines."""
    assert sanity_check("q", _EMPTY).reason == (
        "The query returned no rows, so I can't answer this confidently.")
    assert sanity_check("q", pa.table({"s": [None]})).reason == (
        "The query produced an empty value, so I can't answer this confidently.")


def test_the_answered_note_names_the_filters():
    g = _ground("SELECT id FROM claim WHERE status = 'open' AND amount > 1000000")
    assert answered_note(_EMPTY, g) == "No rows matched: status = 'open' AND amount > 1000000."
    assert answered_note(_EMPTY, None) == "No rows matched."
    assert answered_note(pa.table({"s": [None]}), g) == (
        "The result holds no value: no rows matched (status = 'open' AND amount > 1000000), or "
        "every row that matched holds none.")
    assert answered_note(pa.table({"n": [3]}), g) == ""
    # The query's own NULL, answered only with the layer off, describes no "none".
    assert answered_note(pa.table({"s": [None]}), _ground("SELECT NULL AS s")) == ""


def test_the_answered_note_cuts_a_long_filter_and_says_so():
    conditions = " AND ".join(f"amount <> {i}" for i in range(200))
    note = answered_note(_EMPTY, _ground(f"SELECT id FROM claim WHERE {conditions}"))
    assert note.startswith("No rows matched: amount <> 0 AND amount <> 1 AND")
    assert note.endswith("... (cut at 300 characters; the SQL shown with this answer has them "
                         "whole).")
    assert len(note) < 420


# --- a value the index could not check, asked of the source -------------------------------------

def test_a_not_indexed_value_records_the_column_it_is_compared_with():
    g = _ground("SELECT id FROM claim WHERE adjuster IN ('agent_0042', 'agnet_0042')")
    assert [(v.text, v.why, v.placed) for v in g.values] == [
        ("agent_0042", "not_indexed", ("claim", "adjuster")),
        ("agnet_0042", "not_indexed", ("claim", "adjuster"))]


def test_only_a_not_indexed_value_carries_a_column_to_ask():
    """Masked, row-filtered, pattern, range, expression and date values are never asked: what a
    mask or a row filter withholds is whether a value exists, and the others are no equality."""
    for sql, kw in (("SELECT id FROM claim WHERE status = 'open'",
                     {"masks": lambda t, c: c == "status"}),
                    ("SELECT id FROM claim WHERE adjuster = 'agent_0042'", {"row_filtered": {"claim"}}),
                    ("SELECT id FROM claim WHERE adjuster LIKE 'agn%'", {}),
                    ("SELECT id FROM claim WHERE adjuster > 'a'", {}),
                    ("SELECT id FROM claim WHERE lower(adjuster) = 'x'", {}),
                    ("SELECT id FROM claim WHERE ts = '2024-03-03'", {})):
        assert [v.placed for v in _ground(sql, **kw).values] == [None], sql


def test_the_source_grounds_or_rules_out_a_value_and_a_silent_source_changes_nothing():
    from mnemiq.sql.values_check import confirm_by_source

    asked = []

    def holds(table, column, text):
        asked.append((table, column, text))
        return {"agent_0042": True, "agnet_0042": False}.get(text)

    g = _ground("SELECT id FROM claim WHERE adjuster IN ('agent_0042', 'agnet_0042', 'quiet') "
                "AND status = 'open' AND note LIKE 'x%'")
    read = [(v.text, v.outcome, v.why) for v in confirm_by_source(g, holds).values]
    assert read == [("agent_0042", GROUNDED, "probed"), ("agnet_0042", ABSENT, "probed"),
                    ("quiet", UNCHECKED, "not_indexed"), ("open", GROUNDED, ""),
                    ("x%", UNCHECKED, "pattern")]
    assert asked == [("claim", "adjuster", "agent_0042"), ("claim", "adjuster", "agnet_0042"),
                     ("claim", "adjuster", "quiet")]


def test_a_value_is_asked_once_and_at_most_the_limit_are_asked():
    from mnemiq.sql.values_check import confirm_by_source

    asked = []

    def holds(table, column, text):
        asked.append(text)
        return True

    many = ", ".join(f"'a{i}'" for i in range(12))
    g = _ground(f"SELECT id FROM claim WHERE adjuster IN ('a0', {many})")
    out = confirm_by_source(g, holds, limit=8)
    assert asked == [f"a{i}" for i in range(8)]
    assert [v.outcome for v in out.values] == [GROUNDED] * 9 + [UNCHECKED] * 4


def test_the_probe_quotes_the_stored_names_and_escapes_the_value():
    from mnemiq.sql.values_check import probe_sql

    assert probe_sql("claim", "adjuster", "O'Brien", "duckdb") == (
        "SELECT 1 FROM \"claim\" WHERE \"adjuster\" = 'O''Brien' LIMIT 1")
    assert probe_sql("CLAIM", "ADJUSTER", "x", "oracle") == (
        "SELECT 1 FROM \"CLAIM\" WHERE \"ADJUSTER\" = 'x' FETCH FIRST 1 ROWS ONLY")
    assert probe_sql("claim", "adjuster", "x", "mysql") == (
        "SELECT 1 FROM `claim` WHERE `adjuster` = 'x' LIMIT 1")
    # A dotted name stays one name: split, "a.b" would read table b in schema a, unchecked.
    assert probe_sql("a.b", "c", "x", "duckdb") == "SELECT 1 FROM \"a.b\" WHERE \"c\" = 'x' LIMIT 1"


def test_the_source_answers_found_missing_or_nothing(tmp_path):
    import duckdb

    from mnemiq.adapters.duckdb import DuckDBAdapter
    from mnemiq.sql.values_check import source_holds

    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim_raw (adjuster VARCHAR)")
    con.execute("INSERT INTO claim_raw VALUES ('agent_0042'), ('O''Brien')")
    con.close()
    holds = source_holds(DuckDBAdapter.duckdb(path), "duckdb", {"claim": "claim_raw"}, timeout_s=10)
    assert holds("claim", "adjuster", "agent_0042") is True
    assert holds("claim", "adjuster", "O'Brien") is True  # through its physical name, escaped
    assert holds("claim", "adjuster", "agnet_0042") is False
    assert holds("claim", "no_such_column", "x") is None  # a failed probe says nothing


def test_a_table_stored_with_a_dot_is_probed_as_itself_not_as_a_schema_and_a_table(tmp_path):
    """The probed object must be the checked one: a table stored as "a.b", with a table b in a schema
    a behind it holding the value, is asked as "a.b" and answers for itself."""
    import duckdb

    from mnemiq.adapters.duckdb import DuckDBAdapter
    from mnemiq.sql.values_check import source_holds

    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute('CREATE TABLE "a.b" (c VARCHAR)')
    con.execute("INSERT INTO \"a.b\" VALUES ('mine')")
    con.execute("CREATE SCHEMA a")
    con.execute("CREATE TABLE a.b (c VARCHAR)")
    con.execute("INSERT INTO a.b VALUES ('twin_only')")
    con.close()
    holds = source_holds(DuckDBAdapter.duckdb(path), "duckdb", {}, timeout_s=10)
    assert holds("a.b", "c", "mine") is True
    assert holds("a.b", "c", "twin_only") is False

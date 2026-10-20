"""M159: a row filter whose conversion can raise on a row's value is refused before it runs.

A row filter runs over every row of its table, the ones it removes included. `CAST(region AS
INTEGER) > 0` over a hidden row holding 'north' failed every question on the table, and the error
quoted the hidden value. The owner's decision (2026-10-15): reject raising conversions, type-aware
where the snapshot knows the column's type, the refusal naming the engine's non-raising form --
TRY_CAST in DuckDB, DEFAULT NULL ON CONVERSION ERROR in Oracle. Two more spellings of the class are
measured here and refused the same way: DuckDB's JSON readers (text read as JSON, quoted on failure),
and a comparison that converts a text column implicitly (`region = 7`).

Not covered, and not convergent by listing spellings: an implicit conversion of an EXPRESSION over a
text column (`UPPER(region) > 0`, `coalesce(region, '0') = 7`, `region + 0 > 0`), one the snapshot
cannot type (a column with no recorded type), and a raising expression that is not a conversion
(`error()`, and whatever else an engine raises on data). #154's redaction is what bounds those: the
model reads the error class only.
"""

from __future__ import annotations

import logging
import os

import duckdb
import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

TYPES = {("claim", "id"): "INTEGER", ("claim", "region"): "VARCHAR", ("claim", "amount"): "INTEGER",
         ("claim", "big"): "BIGINT", ("claim", "attrs"): "JSON", ("claim", "d"): "DATE",
         ("entitlement", "id"): "INTEGER", ("entitlement", "who"): "VARCHAR"}


def _snapshot(types=TYPES, oracle=False):
    if oracle:
        types = {k: {"VARCHAR": "VARCHAR2(30)", "INTEGER": "NUMBER", "BIGINT": "NUMBER",
                     "JSON": "VARCHAR2(4000)", "DATE": "DATE"}[v] for k, v in types.items()}
    return Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c, data_type=ty)
                             for (t, c), ty in types.items()])


def _decide(row_filter, engine="duckdb", snapshot=None):
    snapshot = snapshot or _snapshot(oracle=engine == "oracle")
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": row_filter})
    return decide("SELECT id FROM claim", visible_schema(snapshot, grants), dialect=engine,
                  target=engine, policy=build_access_policy(snapshot, grants))


# --- the premise, measured ------------------------------------------------------------------------

def test_a_raising_filter_reads_the_rows_it_removes_and_quotes_them_on_duckdb():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); "
                "INSERT INTO claim VALUES (1, '7'), (2, 'north')")
    for raising, safe in (("CAST(region AS INTEGER) > 0", "TRY_CAST(region AS INTEGER) > 0"),
                          ("region = 7", "TRY_CAST(region AS INTEGER) = 7"),
                          ("json_extract_string(region, '$.a') IS NULL",
                           "json_extract_string(TRY_CAST(region AS JSON), '$.a') IS NULL")):
        with pytest.raises(duckdb.Error, match="north"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
        assert con.execute(f"SELECT id FROM claim WHERE {safe}").fetchall()  # runs, NULL for 'north'


def test_codex_s_three_shapes_raise_on_the_hidden_row_and_their_advised_forms_run():
    # Codex's review of #160: each was approved, and each raised on the hidden row when run.
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON); "
                "INSERT INTO claim VALUES (1, '7', '{\"a\": 1}'), (2, 'north', '{\"a\": \"north\"}')")
    for raising, safe in (("(region) = 7", "TRY_CAST(region AS INTEGER) = 7"),
                          ("json_extract_string(TRY_CAST(region AS VARCHAR), '$.a') IS NULL",
                           "json_extract_string(TRY_CAST(region AS JSON), '$.a') IS NULL"),
                          ("json_transform_strict(attrs, '{\"a\":\"INTEGER\"}') IS NOT NULL",
                           "json_transform(attrs, '{\"a\":\"INTEGER\"}') IS NOT NULL")):
        with pytest.raises(duckdb.Error, match="north|Malformed JSON"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
        assert con.execute(f"SELECT id FROM claim WHERE {safe}").fetchall()
        assert isinstance(_decide(raising), Refusal), raising
        assert isinstance(_decide(safe), Approved), safe


RAISING_TYPED_FORMS = [
    # Codex's second review of #160: a typed value `_known_type` did not see, so the comparison
    # was approved, and DuckDB converted the text column to it row by row, quoting the row.
    "region = INTERVAL '1 day'",
    "region = (SELECT id FROM entitlement)",
    "region = (SELECT id AS r FROM entitlement)",
    "region = (SELECT e.id FROM entitlement e WHERE e.who = 'r')",
    "region = (SELECT 7)",
    "region = ['a']", "region = {'a': 1}", "region = (1, 2)", "region = ROW(1, 2)",
    "region = CASE WHEN id > 0 THEN 7 END",
    "region = CASE WHEN id > 0 THEN 7 ELSE 8.5 END",
    "region = CASE WHEN id > 0 THEN 7 ELSE 'b' END",  # a text literal takes the number's type
    "region = CASE WHEN id > 0 THEN attrs END",
    "region = COALESCE(id, 7)", "region = COALESCE(NULL, 7)", "region = IFNULL(id, 7)",
    "region = IF(id > 0, 7, 8)",
    "region = NULLIF(id, 7)",
    "NULLIF(region, 7) IS NULL",  # NULLIF compares as `=` does, inside the function
    "region = -id",
    "COALESCE(region, 'x') = 7",  # a text-typed value over the column, converted as it is
    "region = (SELECT max(id) FROM entitlement)",
    "region = (SELECT count(*) FROM entitlement)",
    # A constant on a branch is read only on the rows that take it (the review gate): the
    # hidden row takes this one, and 'x' is not JSON.
    "json_extract_string(CASE WHEN region = 'north' THEN 'x' ELSE attrs END, '$.a') IS NOT NULL",
    "json_extract_string(CASE WHEN region = 'north' THEN 'x' || '' ELSE attrs END, '$.a') IS NOT NULL",
    "json_extract_string(coalesce(attrs, 'x'), '$.a') IS NOT NULL",
    # A lone surrogate escape: Python's reader takes it, DuckDB's refuses it (the review gate).
    "json_extract_string(coalesce(attrs, '\"\\ud800\"'), '$.a') IS NOT NULL",
    # Codex's third review: a literal-only inner CASE is text, so the outer one stays text and
    # reads region; and a duplicate key must not hide the first value's bad escape.
    "CASE WHEN id > 0 THEN region ELSE CASE WHEN id = 0 THEN '0' ELSE '1' END END = 7",
    "COALESCE(region, CASE WHEN id = 0 THEN '0' END) = 7",
    "json_extract_string(coalesce(attrs, '{\"a\":\"\\ud800\",\"a\":1}'), '$.a') IS NOT NULL",
    # A literal the row chooses by a column is read on the rows that choose it (the review
    # gate): this one fails on the rows with id > 0, quoting the literal.
    "CASE WHEN id > 0 THEN 'x' ELSE CAST(1 AS VARCHAR) END = 7",
    # Codex's fourth review: a parenthesised NULL or literal is still one; and a JSON branch
    # makes the whole value JSON, every other branch converted to it on the rows that take it,
    # with no JSON reader around.
    "region = COALESCE((NULL), 7)",
    "region = CASE WHEN id = 1 THEN 7 ELSE (NULL) END",
    "region = CASE WHEN id = 1 THEN 7 ELSE ('b') END",
    "COALESCE(CASE WHEN id = 2 THEN 'x' END, attrs) = '{}'",
    "COALESCE(attrs, CASE WHEN id = 3 THEN 'x' END) IS NOT NULL",
    "COALESCE(attrs, 'x') IS NOT NULL",
    "COALESCE(attrs, region) IS NOT NULL",
    "CASE WHEN id = 2 THEN region ELSE attrs END IS NOT NULL",
    "IF(id = 2, 'x', attrs) IS NOT NULL",
    "region = COALESCE(attrs, '{}')",  # region converted to JSON
    # Codex's fifth review: a text literal beside a typed branch takes that type and is
    # converted on the rows that take it; 'north' is no number, date, boolean or interval.
    "CASE WHEN id = 1 THEN 7 ELSE 'north' END = 7",
    "CASE WHEN id = 1 THEN d ELSE 'north' END = DATE '2000-01-01'",
    "CASE WHEN id = 1 THEN TRUE ELSE 'x' END",
    "CASE WHEN id = 1 THEN INTERVAL '1 day' ELSE 'x' END = INTERVAL '1 day'",
    "IF(id = 1, 7, 'north') = 7",
    # The review gate: a literal Python's parsers take and DuckDB's casts refuse, an integer out
    # of range, Unicode digits, a basic ISO date, `nan` to a DECIMAL branch.
    "CASE WHEN id = 2 THEN 7 ELSE '99999999999' END = 7",
    "CASE WHEN id = 1 THEN big ELSE '99999999999999999999' END = 1",
    "CASE WHEN id = 1 THEN id ELSE '٣' END = 7",
    "CASE WHEN id = 1 THEN d ELSE '20000102' END = d",
    "CASE WHEN id = 1 THEN 1.5 ELSE 'nan' END = 1",
    # Codex's sixth review: INT32's bound, a DECIMAL's precision, a basic ISO date.
    "CASE WHEN id = 1 THEN id ELSE '2147483648' END = 1",
    "CASE WHEN id = 1 THEN CAST(1 AS DECIMAL(4,2)) ELSE '1000' END = 1",
    "CASE WHEN id = 1 THEN d ELSE '20200101' END = DATE '2020-01-01'",
    # The review gate: arithmetic typed as DuckDB types it (INTEGER, not DOUBLE), and a subquery
    # branch typed by the reader here rather than run on the validator's connection.
    "CASE WHEN id = 1 THEN id + 1 ELSE 'nan' END = 2",
    "CASE WHEN id = 1 THEN id + 1 ELSE '3000000000' END = 2",
    "CASE WHEN id = 1 THEN (SELECT 7 FROM entitlement LIMIT 1) ELSE 'north' END = 7",
    "CASE WHEN id = 1 THEN (SELECT id + 1 FROM entitlement LIMIT 1) ELSE 'nan' END = 7",
    # Codex's seventh review: a nested CASE of integers and a counted subquery were typed DOUBLE
    # by the reader here, where DuckDB types them INTEGER and BIGINT, and 'nan' converts to a
    # DOUBLE alone; a list and a struct likewise took no literal.
    "CASE WHEN id = 1 THEN CASE WHEN id > 0 THEN 1 ELSE 2 END ELSE 'nan' END IS NOT NULL",
    "CASE WHEN id = 1 THEN CASE WHEN id > 0 THEN 1.5 ELSE 2 END ELSE 'nan' END IS NOT NULL",
    "COALESCE(CASE WHEN id > 5 THEN 1 END, 'nan') IS NOT NULL",
    "CASE WHEN id = 1 THEN (SELECT count(*) FROM entitlement) ELSE 'nan' END IS NOT NULL",
    "CASE WHEN id = 1 THEN (SELECT sum(id) FROM entitlement) ELSE 'nan' END IS NOT NULL",
    "CASE WHEN id = 1 THEN [1, 2] ELSE 'nan' END IS NOT NULL",
    "CASE WHEN id = 1 THEN {'a': 1} ELSE 'nan' END IS NOT NULL",
    # Codex's eighth review: a JSON element or field makes the branches' unified type hold JSON
    # there, and a text element or field of another branch is parsed to it on the rows that
    # take the branch; a comparison of such containers converts on every row.
    "CASE WHEN id = 1 THEN [attrs] ELSE [region] END IS NOT NULL",
    "CASE WHEN id = 1 THEN {'a': attrs} ELSE {'a': region} END IS NOT NULL",
    "CASE WHEN id = 1 THEN [[attrs]] ELSE [[region]] END IS NOT NULL",
    "CASE WHEN id = 1 THEN [attrs] ELSE ['x'] END IS NOT NULL",
    "CASE WHEN id = 1 THEN [attrs] ELSE [lower(region)] END IS NOT NULL",
    "[attrs] = [region]",
    "{'a': attrs} = {'a': region}",
    # The review gate on 3070774: a MAP is a container too, keyed by what its keys evaluate to;
    # and TRY_CAST to a container gives a container of NULLs where an element does not convert,
    # so a text literal beside a container branch is refused.
    "CASE WHEN id = 1 THEN MAP {'a': attrs} ELSE MAP {'a': region} END IS NOT NULL",
    "MAP {'a': attrs} = MAP {'a': region}",
    "CASE WHEN id = 1 THEN MAP {attrs: 1} ELSE MAP {region: 1} END IS NOT NULL",
    "CASE WHEN id = 1 THEN MAP {id: 1} ELSE '{north=1}' END IS NOT NULL",
    "CASE WHEN id = 1 THEN MAP(['a'], [id]) ELSE '{a=north}' END IS NOT NULL",
    "CASE WHEN id = 1 THEN [id] ELSE '[north]' END IS NOT NULL",
    # Codex's ninth review: a list's elements and a map's values are unified as branches are.
    "[attrs, region][2] IS NOT NULL",
    "{'a': [attrs, region]} IS NOT NULL",
    "MAP(['a', 'b'], [attrs, region]) IS NOT NULL",
    "MAP {'a': attrs, 'b': region} IS NOT NULL",
    "MAP {attrs: 1, region: 2} IS NOT NULL",
    # The review gate on 3d52790: the list spelled as a call, and a map literal subscripted at
    # once, are the same shapes.
    "array_value(attrs, region)[2] IS NOT NULL",
    "list_value(attrs, region)[2] IS NOT NULL",
    "list_pack(attrs, region)[2] IS NOT NULL",
    "MAP {'a': attrs, 'b': region}['b'] IS NOT NULL",
    # The review gate on 2e09098: the call spelling on a branch or across a comparison is a
    # list, typed by DuckDB as one.
    "array_value(attrs) = array_value(region)",
    "CASE WHEN id = 1 THEN list_pack(attrs) ELSE list_pack(region) END IS NOT NULL",
    # Codex's tenth review: a value read out of a container keeps the container's element
    # type, and a list search unifies the element type with what it searches for.
    "COALESCE([attrs][1], region) IS NOT NULL",
    "COALESCE(({'a': attrs}).a, region) IS NOT NULL",
    "list_contains([attrs], region)",
    "array_position([attrs], region) > 0",
    "list_has_any([attrs], [region])",
    "array_contains([attrs], region)",
    "list_position([attrs], region) > 0",
    "list_has_all([attrs], [region])",
    "array_has_all([attrs], [region])",
    # The review gate on e12b82d: two more spellings, and a list the reader cannot type.
    "contains([attrs], region)",
    "[region] <@ [attrs]",
    "list_contains(list_distinct([attrs]), region)",
    # Codex's eleventh review: a map looked up by a key converts the key to the map's key type.
    "element_at(MAP(['{}'::JSON], [1]), region) = [1]",
    "map_extract_value(MAP(['{}'::JSON], [1]), region) = 1",
    "MAP(['{}'::JSON], [1])[region] = 1",
    "map_extract(MAP(['{}'::JSON], [1]), region) = [1]",
    # The review gate on cd17747: a subscripted map literal, and map_contains.
    "MAP {'{}'::JSON: 1}[region] = 1",
    "map_contains(MAP(['{}'::JSON], [1]), region)",
    # Codex's twelfth review: a container accessor's value is typed as the container's.
    "map_values(MAP(['a'], [attrs])) = map_values(MAP(['a'], [region]))",
    "map_keys(MAP([COALESCE(attrs, '{}'::JSON)], [1])) = map_keys(MAP([region], [1]))",
    "list_extract([attrs], 1) = region",
    "struct_extract({'a': attrs}, 'a') = region",
    "list_distinct([attrs]) = [region]",
    "COALESCE(list_extract([attrs], 1), region) IS NOT NULL",
    # Codex's thirteenth review: a row value built by a call, and the gate's array_reverse.
    "row(attrs) = row(region)",
    "struct_pack(a := attrs) = struct_pack(a := region)",
    "array_reverse([attrs]) = array_reverse([region])",
    # The review gate: an unnamed struct at any depth.
    "[row(attrs)] = [row(region)]",
    "row(row(attrs)) = row(row(region))",
    # Codex's fourteenth review: an ENUM field keeps its labels, which its cast target needs.
    "{'a': attrs, 'b': CAST('x' AS ENUM('x'))} = {'a': region, 'b': CAST('x' AS ENUM('x'))}",
    "[{'a': attrs, 'b': CAST('x' AS ENUM('x'))}] = [{'a': region, 'b': CAST('x' AS ENUM('x'))}]",
]
SAFE_TYPED_FORMS = [
    # JSON on every branch, or text on every branch, cannot fail on a row (the medium finding).
    "json_extract_string((attrs), '$.a') IS NOT NULL",
    # A native typed value beside a JSON branch is serialised, not parsed (Codex's fifth review).
    "COALESCE(attrs, 0) IS NOT NULL", "COALESCE(attrs, 1.5) IS NOT NULL", "COALESCE(attrs, []) IS NOT NULL",
    "COALESCE(attrs, TRUE) IS NOT NULL", "COALESCE(attrs, DATE '2000-01-01') IS NOT NULL",
    "COALESCE(attrs, {'a': 1}) IS NOT NULL", "COALESCE(attrs, id) IS NOT NULL", "COALESCE(attrs, d) IS NOT NULL",
    "COALESCE(attrs, (SELECT attrs FROM claim WHERE id = 1)) IS NOT NULL",
    "json_contains(COALESCE(attrs, []), '1')",
    # A text literal that converts to the typed branch's type, as DuckDB converts it.
    "CASE WHEN id = 1 THEN 7 ELSE '8' END = 7",
    "CASE WHEN id = 1 THEN d ELSE '2000-01-02' END = DATE '2000-01-01'",
    "CASE WHEN id = 1 THEN 1.5 ELSE '2.5' END = 1",
    "CASE WHEN id = 1 THEN 7 ELSE ' 8 ' END = 7",
    "CASE WHEN id = 1 THEN 7 ELSE '1e3' END = 7",  # DuckDB reads it as an integer, Python does not
    "json_extract_string(coalesce(attrs, '{}'::JSON), '$.a') IS NOT NULL",
    "json_extract_string(coalesce(attrs, '{}'), '$.a') IS NOT NULL",  # a literal that is JSON
    "json_extract_string(coalesce(attrs, 'NaN'), '$.a') IS NOT NULL",  # JSON to DuckDB, measured
    "json_extract_string(coalesce(attrs, '[1,]'), '$[0]') = '1'",  # DuckDB's grammar, not Python's
    "json_extract_string(coalesce(attrs, '-NaN'), '$.a') IS NOT NULL",
    "COALESCE(attrs, '{}') IS NOT NULL",  # JSON beside a JSON literal, no reader needed
    "json_extract_string(CASE WHEN id > 0 THEN attrs END, '$.a') IS NOT NULL",
    "json_extract_string(CASE WHEN id > 0 THEN '{\"a\": 1}'::JSON ELSE attrs END, '$.a') IS NOT NULL",
    "json_extract_string(IF(id > 0, attrs, NULL), '$.a') IS NOT NULL",
    "json_extract_string(coalesce(attrs, TRY_CAST(region AS JSON)), '$.a') IS NOT NULL",
    "region = CASE WHEN id > 0 THEN 'a' ELSE 'b' END",
    "region = COALESCE(region, 'x')",
    "region = (SELECT who FROM entitlement)",
    "region = (SELECT max(who) FROM entitlement)",
    "region = (SELECT '7')",
    "region = CAST(id AS VARCHAR)",
    # A nested branched value is typed as DuckDB types it (Codex's seventh review), a literal
    # beside a branch the validator cannot type is written as a cast, a list literal that is
    # one, and a text branch beside a call keeps a literal as text.
    "CASE WHEN id = 1 THEN CASE WHEN id > 0 THEN 1.5 ELSE 2 END ELSE '3' END = 1",
    "CASE WHEN id = 1 THEN (SELECT avg(id) FROM entitlement) ELSE CAST('7' AS DOUBLE) END = 7",
    "CASE WHEN id = 1 THEN lower(region) ELSE CAST('x' AS VARCHAR) END = 'x'",
    "CASE WHEN id = 1 THEN region WHEN id = 3 THEN lower(region) ELSE 'x' END = 'x'",
    # A container of JSON beside one of a native type, or of JSON already, is serialised or
    # taken as it is (Codex's eighth review).
    "CASE WHEN id = 1 THEN [attrs] ELSE [TRY_CAST(region AS JSON)] END IS NOT NULL",
    "CASE WHEN id = 1 THEN [attrs] ELSE [id] END IS NOT NULL",
    "CASE WHEN id = 1 THEN {'a': attrs} ELSE {'a': 1} END IS NOT NULL",
    "[attrs] = [TRY_CAST(region AS JSON)]",
    "CASE WHEN id = 1 THEN MAP {'a': attrs} ELSE MAP {'a': 1} END IS NOT NULL",
    "CASE WHEN id = 1 THEN MAP {id: 1} ELSE MAP {id: 2} END IS NOT NULL",
    "list_contains([attrs], TRY_CAST(region AS JSON))",
    "list_contains([region], 'x')",
    "contains(region, 'x')",
    "contains(lower(region), 'x')",
    "contains(attrs, 'x')",  # the string search over a JSON column (the review gate)
    "contains(region, lower('X'))",
    "list_contains([region], lower(region))",
    "element_at(MAP(['{}'::JSON], [1]), TRY_CAST(region AS JSON)) = [1]",
    "element_at(MAP([region], [1]), 'x') = [1]",
    "MAP([region], [1])['x'] = 1",
    "MAP {'a': 1}[region] = 1",
    "map_contains(MAP(['a'], [1]), region)",
    "map_values(MAP(['a'], [region])) = ['x']",
    "row(region) = row('x')",
    "row(row(region)) = row(row('x'))",  # a nested unnamed struct, read in DuckDB's grammar
    "{'a': region, 'b': CAST('x' AS ENUM('x'))} = {'a': 'y', 'b': CAST('x' AS ENUM('x'))}",
    "row(d, region) = row(d, 'x')",
    "list_extract([region], 1) = 'x'",
    # Typed now, so no longer refused as a list or a map the reader cannot type.
    "list_contains(list_distinct([region]), 'x')",
    "map_extract_value(map_from_entries([('a', 1)]), region) = 1",
    "[attrs][1] IS NOT NULL",
    "COALESCE([attrs][1], attrs) IS NOT NULL",
    "CASE WHEN id = 1 THEN {'a': attrs, 'b': 1.5} ELSE {'a': attrs, 'b': 10.25} END IS NOT NULL",
    "CASE WHEN id = 1 THEN {'a': attrs, 'b': region} ELSE {'b': region, 'a': attrs} END IS NOT NULL",
    "CASE WHEN id = 1 THEN [id] ELSE CAST('[2]' AS INTEGER[]) END IS NOT NULL",
    "[attrs, 1][1] IS NOT NULL",
    "[attrs, TRY_CAST(region AS JSON)][2] IS NOT NULL",
    "[region, 'x'] IS NOT NULL",
]


def test_codex_s_second_review_forms_raise_on_the_hidden_row_and_the_safe_ones_run():
    con = duckdb.connect()
    # The third row's NULL attrs is what makes a COALESCE read its other branch.
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON, d DATE, big BIGINT); "
                "INSERT INTO claim VALUES (1, '7', '{\"a\": 1}', DATE '2000-01-01', 1), "
                "(2, 'north', '{\"a\": \"north\"}', DATE '2000-01-02', 2), (3, 'south', NULL, NULL, 3); "
                "CREATE TABLE entitlement(id INT, who VARCHAR); INSERT INTO entitlement VALUES (7, 'r')")
    for raising in RAISING_TYPED_FORMS:
        # The list, struct and row forms quote the first row's value rather than the hidden one.
        # A conversion that fails on a row: quoting the row's value, or the literal the row chose.
        with pytest.raises(duckdb.Error, match="Conversion Error|Malformed JSON"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
    for safe in SAFE_TYPED_FORMS:
        con.execute(f"SELECT id FROM claim WHERE {safe}").fetchall()


@pytest.mark.parametrize("row_filter", RAISING_TYPED_FORMS)
def test_a_typed_value_the_checker_did_not_see_is_now_refused(row_filter):
    verdict = _decide(row_filter)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


@pytest.mark.parametrize("row_filter", SAFE_TYPED_FORMS)
def test_json_or_text_on_every_branch_is_kept(row_filter):
    assert isinstance(_decide(row_filter), Approved), row_filter


@pytest.mark.parametrize("row_filter", [
    # A text column beside a number is a bind fault on no row's value, the proof's to report, so
    # it is read as text and not refused here; a reader over a text column on one branch is.
    "region = CASE WHEN id > 0 THEN 7 ELSE region END",
    "region = COALESCE(region, 7)",
    "region = CASE WHEN id > 0 THEN 7 ELSE CASE WHEN id = 0 THEN '0' ELSE '1' END END",
])
def test_a_text_column_mixed_with_a_number_fails_at_bind_and_is_not_this_class(row_filter):
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); INSERT INTO claim VALUES (2, 'north')")
    with pytest.raises(duckdb.BinderException, match="Cannot mix"):
        con.execute(f"SELECT id FROM claim WHERE {row_filter}").fetchall()
    assert isinstance(_decide(row_filter), Approved), row_filter


@pytest.mark.parametrize("literal", [
    "[1,]", "-NaN", "NaN", '"\\ud800"', '{"a":"\\ud800","a":1}', "{}", "x", "", " {} ",
    '{"a":1} x', "'quoted'", "[1,,2]", "{a:1}", "01", "1.", ".5", "+1", "true", "nul",
])
def test_a_json_literal_is_read_as_duckdb_s_reader_reads_it(literal):
    """`_json_text` is DuckDB's own `json_valid`, so the two cannot drift; the differential is
    against the reader that fails on the row. Python's grammar refuses `[1,]` and `-NaN`, which
    DuckDB reads, and takes the lone surrogate, which DuckDB refuses."""
    from mnemiq.sql.rls import _json_text

    con = duckdb.connect()
    try:
        con.execute("SELECT json_extract_string($1::VARCHAR, '$')", [literal]).fetchone()
        reads = True
    except duckdb.Error:
        reads = False
    assert _json_text(literal) is reads, (literal, reads)


def test_the_validator_types_a_policy_without_running_it(monkeypatch):
    """What the validator hands DuckDB while deciding is read off the door itself (the review
    gate on #160: a time bound could not tell a run from a fast one): a trillion-row table
    function on a branch is the filter validator's refusal and never sent; a call reading a file
    is never sent; what is sent types literals and typed NULLs alone."""
    from mnemiq.sql import rls

    sent = []
    real = rls._duckdb

    def spy(sql, params=()):
        sent.append(sql)
        return real(sql, params)

    monkeypatch.setattr(rls, "_duckdb", spy)
    # A subquery's pure projection is typed with its column a typed NULL; its FROM is never sent.
    verdict = _decide("CASE WHEN id = 1 THEN (SELECT id + 1 FROM entitlement LIMIT 1) "
                      "ELSE 'nan' END = 7")
    assert isinstance(verdict, Refusal), verdict  # 'nan' converts to no integer
    # A table function on a branch is the filter validator's refusal before any typing.
    verdict = _decide("CASE WHEN id = 1 THEN (SELECT count(*) FROM range(1000000000000)) "
                      "ELSE 'x' END = 1")
    assert isinstance(verdict, Refusal), verdict
    # A subquery inside arithmetic is not a pure expression, so the literal's target type
    # cannot be told: refused, and the subquery's FROM is never sent.
    verdict = _decide("CASE WHEN id = 1 THEN (SELECT id FROM entitlement LIMIT 1) + 1 "
                      "ELSE 'x' END = 7")
    assert isinstance(verdict, Refusal), verdict
    # A call's type is unknown here, so a literal beside it is refused (Codex's seventh review:
    # the reader's guess is not taken instead); the call is never sent.
    verdict = _decide("CASE WHEN id = 1 THEN length(read_text('/nonexistent/never-read')) "
                      "ELSE 'x' END = 1")
    assert isinstance(verdict, Refusal), verdict
    assert sent, "DuckDB was asked for the literals' casts"
    for sql in sent:
        assert "entitlement" not in sql and "range(" not in sql and "read_text" not in sql, sql
        assert " FROM " not in sql, sql
        assert sql.startswith("SELECT typeof(") or sql.startswith("SELECT TRY_CAST("), sql


def test_a_literal_beside_a_branch_the_validator_cannot_type_is_refused_though_it_may_run():
    # Codex's seventh review of #160: the type reader guessed DOUBLE for an aggregate and a nested
    # CASE, and the guess let 'nan' through to an INTEGER. The validator now types a branch only
    # as DuckDB does, without running anything, so a call or an aggregate beside a text literal
    # leaves the literal's target unknown, and the filter is refused although these two run.
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); INSERT INTO claim VALUES (1, 'x'), (2, 'north'); "
                "CREATE TABLE entitlement(id INT, who VARCHAR); INSERT INTO entitlement VALUES (7, 'r')")
    for runs in ("CASE WHEN id = 1 THEN (SELECT avg(id) FROM entitlement) ELSE 'nan' END IS NOT NULL",
                 "CASE WHEN id = 1 THEN lower(region) ELSE 'x' END = 'x'",
                 # TRY_CAST to a container cannot tell a converting literal from one that does
                 # not (the review gate), so a literal beside a container branch is refused too.
                 "CASE WHEN id = 1 THEN [1, 2] ELSE '[3]' END IS NOT NULL",
                 # A list or a map a call the reader does not know returns cannot be typed,
                 # and may hold JSON (the review gate); a list-only search or a map-only lookup
                 # of one is refused.
                 "list_contains(list_filter([region], x -> x IS NOT NULL), 'x')",
                 "map_extract_value(map_from_entries(list_filter([('a', 1)], x -> true)), region) = 1"):
        con.execute(f"SELECT id FROM claim WHERE {runs}").fetchall()
        verdict = _decide(runs)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, runs


def test_a_union_whose_members_pair_text_with_json_is_refused_wherever_it_sits():
    """Codex's fifteenth review of #160 ran `CASE WHEN id = 1 THEN u1 ELSE u2 END` over a
    UNION(a JSON) and a UNION(a VARCHAR) and it ran. The sixteenth round's draft exempted a
    UNION on a branch, and the commit gate refuted it: the same pair inside a struct raises, and
    so does a list search. Measured (DuckDB 1.5.4): two UNION types unify by the operator and
    the operand order, COALESCE and a list keeping the first operand's type and CASE and IF the
    last's, so `u1 ELSE u2` is UNION(a VARCHAR) and runs, `s ELSE t` over STRUCT(x UNION(a
    VARCHAR)) and STRUCT(x UNION(a JSON)) is the JSON one and parses; the walk's COALESCE probe
    answers for COALESCE alone. A UNION beside JSON is refused wherever it sits, the operator
    that keeps the text member a false refusal, accepted and documented."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, u1 UNION(a JSON), u2 UNION(a VARCHAR), "
                "s STRUCT(x UNION(a VARCHAR)), t STRUCT(x UNION(a JSON))); "
                "INSERT INTO claim VALUES (1, union_value(a := '{}'::JSON), union_value(a := 'x'), "
                "{'x': union_value(a := 'x')}, {'x': union_value(a := '{}'::JSON)}), "
                "(2, union_value(a := '{}'::JSON), union_value(a := 'north'), "
                "{'x': union_value(a := 'north')}, {'x': union_value(a := '{}'::JSON)})")
    types = {("claim", "id"): "INTEGER", ("claim", "u1"): "UNION(a JSON)", ("claim", "u2"): "UNION(a VARCHAR)",
             ("claim", "s"): "STRUCT(x UNION(a VARCHAR))", ("claim", "t"): "STRUCT(x UNION(a JSON))"}
    branched = "CASE WHEN id = 1 THEN u1 ELSE u2 END IS NOT NULL"
    assert con.execute(f"SELECT id FROM claim WHERE {branched}").fetchall() == [(1,), (2,)]
    unified = con.execute("SELECT typeof(CASE WHEN id = 1 THEN u1 ELSE u2 END) FROM claim").fetchall()
    assert unified == [("UNION(a VARCHAR)",)] * 2, unified
    raising = ("CASE WHEN id = 1 THEN s ELSE t END IS NOT NULL", "list_contains([u1], u2)", "u1 = u2", "[u1] = [u2]")
    for row_filter in raising:
        with pytest.raises(duckdb.Error, match="Malformed JSON"):
            con.execute(f"SELECT id FROM claim WHERE {row_filter}").fetchall()
    for row_filter in (branched, *raising):
        verdict = _decide(row_filter, snapshot=_snapshot(types))
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (row_filter, verdict)


def test_a_doubled_quote_in_a_member_name_is_read_and_an_unreadable_declared_type_is_refused():
    """Codex's sixteenth review of #160: `_split_label` took the doubled quote in `UNION("a""b"
    JSON)` for the closing one, the declared type failed to parse, the failure read as an
    unknown value's type, and `u1 = u2` and `list_contains([u1], u2)` were approved where DuckDB
    parses the text member. The name is read as DuckDB spells it in DESCRIBE, rebuilt with its
    quote doubled, and a declared type the reader cannot parse is refused outright."""
    from mnemiq.sql.rls import _castable, _parse_type

    con = duckdb.connect()
    con.execute('CREATE TABLE claim(id INT, u1 UNION("a""b" JSON), u2 UNION("a""b" VARCHAR), '
                's STRUCT("a""b" JSON), t STRUCT("a""b" VARCHAR)); '
                'INSERT INTO claim VALUES (1, union_value("a""b" := \'{}\'::JSON), union_value("a""b" := \'north\'), '
                '{\'a"b\': \'{}\'::JSON}, {\'a"b\': \'north\'})')
    declared = {name: kind for name, kind, *_ in con.execute("DESCRIBE claim").fetchall()}
    assert declared["u1"] == 'UNION("a""b" JSON)', declared
    for kind in (declared["u1"], declared["s"]):
        parsed = _parse_type(kind)
        assert parsed[1][0][0] == 'a"b', parsed
        assert _castable(parsed) == kind, (kind, _castable(parsed))
    types = {("claim", "id"): "INTEGER", **{("claim", name): kind for name, kind in declared.items() if name != "id"}}
    # The CASE keeps the last branch's type (the sixteenth round), so the JSON struct goes last.
    for raising in ("u1 = u2", "list_contains([u1], u2)", "CASE WHEN id = 1 THEN t ELSE s END IS NOT NULL"):
        with pytest.raises(duckdb.Error, match="Malformed JSON"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
        verdict = _decide(raising, snapshot=_snapshot(types))
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (raising, verdict)
    # DuckDB quotes a keyword used as a type name: `typeof([])` is `"NULL"[]`, read and rebuilt
    # as spelled, so `COALESCE(attrs, [])` stays approved under the refusal below.
    assert con.execute("SELECT typeof([])").fetchall() == [('"NULL"[]',)]
    assert _parse_type('"NULL"[]') == ("array", ("leaf", '"NULL"', ""))
    assert _castable(_parse_type('"NULL"[]')) == '"NULL"[]'
    # An unnamed field whose type is a quoted keyword (the review gate on the seventeenth round
    # and Codex's seventeenth review): `row(NULL, id)` is STRUCT("NULL", INTEGER), a valid filter.
    assert con.execute("SELECT typeof(row(NULL, 1)), typeof(row(1, []))").fetchall() == [('STRUCT("NULL", INTEGER)', 'STRUCT(INTEGER, "NULL"[])')]
    assert _parse_type('STRUCT("NULL", INTEGER)') == ("struct", [(None, ("leaf", '"NULL"', "")), (None, ("leaf", "INTEGER", ""))])
    assert _parse_type('STRUCT(INTEGER, "NULL"[])') == ("struct", [(None, ("leaf", "INTEGER", "")), (None, ("array", ("leaf", '"NULL"', "")))])
    for valid in ("row(NULL, id) = row(NULL, 1)", "list_contains([row(NULL, id)], row(NULL, 1))",
                  "CASE WHEN id = 1 THEN row(id, []) ELSE row(1, [2]) END IS NOT NULL",
                  # Codex's eighteenth review: the list search's element typed by sqlglot, not DuckDB
                  "list_contains([row(id, [])], row(id, []))", "list_position([row(id, [])], row(1, [])) = 1"):
        assert con.execute(f"SELECT id FROM claim WHERE {valid}").fetchall() == [(1,)], valid
        assert isinstance(_decide(valid, snapshot=_snapshot(types)), Approved), valid
    # A declared type the reader cannot parse at all is refused, not read as an unknown value
    # (asked of the readers directly: a snapshot type sqlglot cannot parse never reaches them).
    from mnemiq.sql.rls import _UNREADABLE, _duckdb_type, _json_inside_a_container, _untypable
    import sqlglot

    u1, u2 = (sqlglot.parse_one(name, read="duckdb") for name in ("u1", "u2"))
    unreadable = lambda column: "UNION(a JSON" if column.name == "u1" else declared["u2"]  # noqa: E731
    assert _duckdb_type(u1, unreadable) is _UNREADABLE
    assert _json_inside_a_container([u1, u2], unreadable) is True
    assert _untypable(u1, unreadable, "duckdb") is True  # a list search over it is refused too
    assert _untypable(u2, unreadable, "duckdb") is False


@pytest.mark.parametrize("row_filter", [
    "list_contains([region], to_json(id))",
    "region = to_json(id)",
    "CASE WHEN id = 1 THEN region ELSE to_json(id) END IS NOT NULL",
    "[to_json(id)] = [region]",
    "CASE WHEN id = 1 THEN [region] ELSE [to_json(id)] END IS NOT NULL",
    "region IN (to_json(id))",
    "list_contains([region], json_object('a', id))",
    "list_contains([region], json_extract(attrs, '$.a'))",
    "[region] = [json_array(id)]",
    # DuckDB's `->` inside a call's arguments or beside an operator is its JSON extract, which
    # sqlglot reads as a lambda: refused as a statement the reader cannot see as DuckDB runs it.
    "list_contains([region], attrs -> '$.a')",
    "region = attrs -> '$.a'",
    "list_has_any([region], [attrs -> '$.a'])",
    # The review gate on the twentieth round: calls that take no lambda, returners the list had
    # missed, and an extract over a list of paths, which is JSON[] and not JSON.
    "list_contains([region], list_any_value(attrs -> ['$.a']))",
    "list_contains([region], array_to_json([id]))",
    "[region] = json_extract(attrs, ['$.a'])",
    "[[region]] = [json_extract(attrs, ['$.a'])]",
    "[region] = json_extract(attrs, ['$.a']::VARCHAR[])",  # a list of paths not spelled as a list
    # Codex's twenty-second review: the json extension's aggregate macros carry no return type
    # in the catalogue, and a text extract loses its type before an implicit conversion.
    "region = (SELECT json_group_array(id) FROM claim)",
    "list_contains([region], (SELECT json_group_structure(attrs) FROM claim))",
    "[region] = [(SELECT json_group_array(id) FROM claim)]",  # the subquery inside a list (the gate)
    # A scalar subquery inside a typed call goes with the call (the gate on the twenty-fourth round).
    "region = to_json((SELECT id FROM claim LIMIT 1))",
    "[region] = [to_json((SELECT id FROM claim LIMIT 1))]",
    "region = json_array((SELECT lower(region) FROM claim LIMIT 1))",
    "[region] = [json_array((SELECT lower(region) FROM claim LIMIT 1))]",
    # A scalar subquery projecting a bare column (the gate on the twenty-fourth round).
    "[region] = [(SELECT attrs FROM claim LIMIT 1)]",
    "[region] = [(SELECT c.attrs FROM claim c LIMIT 1)]",
    "list_contains([region], (SELECT attrs FROM claim LIMIT 1))",
    # Codex's twenty-third review: a set operation's subquery, and a correlated bare column.
    "COALESCE((SELECT attrs FROM claim UNION ALL SELECT attrs FROM claim LIMIT 1), region) IS NOT NULL",
    "[region] = [(SELECT attrs FROM claim UNION SELECT attrs FROM claim LIMIT 1)]",
    # Codex's twenty-fourth review: the branches converted to the unified type, a set query under
    # IN, a quantified comparison, and parenthesised branches.
    "(SELECT attrs FROM claim UNION ALL SELECT region FROM claim LIMIT 1) IS NOT NULL",
    "region IN (SELECT attrs FROM claim UNION ALL SELECT attrs FROM claim)",
    "region = ANY(SELECT attrs FROM claim)",
    "region = ALL(SELECT attrs FROM claim)",  # ALL and SOME wrap a bare SELECT in sqlglot (the gate)
    "region <> ALL(SELECT attrs FROM claim)",
    "region = SOME(SELECT attrs FROM claim)",
    "region = ((SELECT attrs FROM claim WHERE id = 2 LIMIT 1) UNION ALL (SELECT attrs FROM claim LIMIT 0))",
    # Codex's twenty-fifth review: a set operation wider than one column, read per position.
    "EXISTS (SELECT attrs, id FROM claim UNION ALL SELECT region, id FROM claim)",
    "EXISTS (SELECT attrs, id FROM claim INTERSECT SELECT region, id FROM claim)",
    "EXISTS (SELECT attrs, id FROM claim EXCEPT SELECT region, id FROM claim)",
    "EXISTS (VALUES (attrs), (region))",  # a VALUES list's rows unify as branches do (the gate)
])
def test_a_call_returning_json_beside_text_is_read_as_json(row_filter):
    """Codex's nineteenth review of #160: `to_json(id)` was of unknown type, and unknown beside
    text was approved, where DuckDB unifies the text to JSON and parses it on the row. A call
    DuckDB answers with JSON is typed by name, so nothing runs; `[to_json(id)]` is a typed NULL
    to the container probe."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON); "
                "INSERT INTO claim VALUES (1, 'north', NULL), (2, 'south', '{\"a\": 1}'::JSON)")
    with pytest.raises(duckdb.Error, match="Malformed JSON"):
        con.execute(f"SELECT id FROM claim WHERE {row_filter}").fetchall()
    verdict = _decide(row_filter)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (row_filter, verdict)


def test_a_correlated_column_in_a_sourceless_subquery_keeps_its_type():
    """Codex's twenty-third review of #160: `(SELECT region) = 7` was approved, the column
    unknown in a SELECT with no sources, where DuckDB converts the text to a number on the row."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON); INSERT INTO claim VALUES (1, 'north', NULL)")
    for raising in ("(SELECT region) = 7", "region = 7", "(SELECT (SELECT region)) = 7"):
        with pytest.raises(duckdb.Error, match="Could not convert"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
        verdict = _decide(raising)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (raising, verdict)
    # A VALUES list is a query too (Codex's twenty-fifth review): `region = (VALUES (1))` converts.
    with pytest.raises(duckdb.Error, match="Could not convert"):
        con.execute("SELECT id FROM claim WHERE region = (VALUES (1))").fetchall()
    verdict = _decide("region = (VALUES (1))")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    assert isinstance(_decide("region = (VALUES ('north'))"), Approved)
    running = "(SELECT region) = 'north'"
    assert con.execute(f"SELECT id FROM claim WHERE {running}").fetchall() == [(1,)]
    assert isinstance(_decide(running), Approved)
    # A SELECT with a FROM keeps its own sources (the review gate): a derived column is not the
    # outer base column of the same name, so the cast over the derived text is still refused and
    # the derived number beside a number is still approved.
    derived_text = "(SELECT CAST(id AS BIGINT) FROM (SELECT region AS id FROM claim) LIMIT 1) IS NOT NULL"
    with pytest.raises(duckdb.Error, match="Could not convert"):
        con.execute(f"SELECT id FROM claim WHERE {derived_text}").fetchall()
    assert isinstance(_decide(derived_text), Refusal)
    derived_number = "(SELECT region FROM (SELECT 7 AS region)) = 7"
    assert con.execute(f"SELECT id FROM claim WHERE {derived_number}").fetchall() == [(1,)]
    assert isinstance(_decide(derived_number), Approved)
    # A set operation's branches are unified as DuckDB unifies them, not read as the first's.
    # Each a text value beside a number, refused as `region = 7` is; the engine raises on the
    # rows it reaches (the reversed order and the sourceless branch happen to reach '1' and no
    # row here, and are refused on the type alone, as the bare comparison is).
    for mixed, raises in (("(SELECT id FROM claim UNION ALL SELECT region FROM claim ORDER BY 1 DESC LIMIT 1) = 7", True),
                          ("(SELECT region FROM claim UNION ALL SELECT id FROM claim ORDER BY 1 LIMIT 1) = 7", False),
                          # a correlated column inside a set operation's branch keeps its SELECT's
                          # tree (the gate on the twenty-seventh round: a wrapper built around the
                          # live node cut the branch off from the filter's SELECT)
                          ("EXISTS (SELECT 1 FROM claim UNION ALL SELECT 1 WHERE region = 7)", True),
                          ("EXISTS (SELECT 1 UNION ALL SELECT (SELECT region) = 7)", False)):
        if raises:
            with pytest.raises(duckdb.Error, match="Could not convert"):
                con.execute(f"SELECT id FROM claim WHERE {mixed}").fetchall()
        else:
            con.execute(f"SELECT id FROM claim WHERE {mixed}").fetchall()
        verdict = _decide(mixed)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (mixed, verdict)
    for running in ("region IN (SELECT region FROM claim)", "region = ANY(SELECT region FROM claim)"):
        assert con.execute(f"SELECT id FROM claim WHERE {running}").fetchall() == [(1,)]
        assert isinstance(_decide(running), Approved), running


def test_a_text_extract_is_typed_text_so_a_conversion_beside_it_is_read():
    """Codex's twenty-second review of #160: `(attrs ->> '$.a') = 7` was approved, the extract
    being of unknown type, and DuckDB converts the extracted text to a number on the row."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON); "
                "INSERT INTO claim VALUES (1, 'north', '{\"a\": \"hidden\"}'::JSON)")
    for raising in ("(attrs ->> '$.a') = 7", "json_extract_string(attrs, '$.a') = 7",
                    "json_extract_path_text(attrs, '$.a') = 1"):
        with pytest.raises(duckdb.Error, match="Could not convert"):
            con.execute(f"SELECT id FROM claim WHERE {raising}").fetchall()
        verdict = _decide(raising)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (raising, verdict)
    for running in ("(attrs ->> '$.a') = 'x'", "json_extract_string(attrs, ['$.a']) = ['x']"):
        assert con.execute(f"SELECT id FROM claim WHERE {running}").fetchall() == []
        assert isinstance(_decide(running), Approved), running


def test_a_call_returning_text_beside_text_is_not_refused_for_being_unknown():
    """The residual, by design: a call of unknown type beside text is not refused, since that
    would refuse `lower(region) = 'x'`; a source-defined function returning JSON stays unknown."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON); "
                "INSERT INTO claim VALUES (1, 'north', '{\"a\": 1}'::JSON)")
    for row_filter in ("list_contains([region], lower(region))", "[region] = [lower(region)]",
                       "lower(region) = 'north'", "list_contains([region], json_extract_string(attrs, '$.a'))",
                       "list_transform([region], x -> upper(x))[1] = 'NORTH'",  # a lambda DuckDB takes
                       "list_filter([region], x -> x = 'north')[1] = 'north'",
                       "array_apply([region], x -> upper(x))[1] = 'NORTH'",  # the catalogue's list, not a typed one
                       "list_sort([region])[1] = region",  # takes no lambda, reads the column
                       "json_value(attrs, '$.a') = region"):  # returns VARCHAR: the catalogue, not the json_ prefix
        assert con.execute(f"SELECT id FROM claim WHERE {row_filter}").fetchall() is not None
        assert isinstance(_decide(row_filter), Approved), row_filter
    # The typed lists stand in where the catalogue cannot be asked; on this engine they are it.
    from mnemiq.sql.rls import _JSON_MACROS, _JSON_RETURNERS_KNOWN, _LAMBDA_TAKERS_KNOWN, _json_returners, _lambda_takers

    assert _json_returners() == _JSON_RETURNERS_KNOWN | _JSON_MACROS  # the macros carry no catalogue type
    # Where the catalogue cannot be asked, the typed list stands in, the macros with it.
    import mnemiq.sql.rls as rls_module

    def no_catalogue(sql, params=()):
        raise RuntimeError("no catalogue")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(rls_module, "_JSON_RETURNERS", None)
        patched.setattr(rls_module, "_duckdb_rows", no_catalogue)
        assert _json_returners() == _JSON_RETURNERS_KNOWN | _JSON_MACROS
    assert rls_module._JSON_RETURNERS == _JSON_RETURNERS_KNOWN | _JSON_MACROS  # the live reader again
    assert _lambda_takers() == _LAMBDA_TAKERS_KNOWN
    # Residuals the gate named (pre-existing): a lambda DuckDB takes is not read for JSON
    # unification inside its body, `list_transform([region], x -> x = to_json(id))[1]`, where
    # `x` is the list's element and the reader does not type a lambda's parameter; and
    # `json_transform` or `from_json` beside text, whose type is the structure they are asked
    # for (typed JSON they would pass beside a JSON value where they return text).
    # The nineteenth round's tightening of the two-list search, both ways (the review gate): a
    # list literal holding a call of unknown type is refused, and the JSON one it closes raises.
    with pytest.raises(duckdb.Error, match="Malformed JSON"):
        con.execute("SELECT id FROM claim WHERE [json_extract(attrs, '$.a')] && [region]").fetchall()
    assert con.execute("SELECT id FROM claim WHERE list_has_any([lower(region)], ['x'])").fetchall() == []
    for two_lists in ("[json_extract(attrs, '$.a')] && [region]", "list_has_any([lower(region)], ['x'])"):
        verdict = _decide(two_lists)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (two_lists, verdict)


def test_duckdb_s_types_are_read_in_duckdb_s_own_grammar():
    """The container reading parses DuckDB's type strings itself (the review gate on #160:
    sqlglot's DataType reads a nested unnamed struct as a struct literal). An unnamed struct's
    fields are named by position for the cast, at any depth; an unnamed field of a multi-word
    type is a type, not a name; a type name may hold digits; the other text spellings are
    text."""
    from mnemiq.sql.rls import _TEXT, _castable, _duckdb_type, _parse_type, _text_becomes_json

    assert _castable(_parse_type("STRUCT(JSON)")) == 'STRUCT("f1" JSON)'
    assert _castable(_parse_type("STRUCT(STRUCT(JSON))[]")) == 'STRUCT("f1" STRUCT("f1" JSON))[]'
    assert _castable(_parse_type("STRUCT(TIMESTAMP WITH TIME ZONE, JSON)")) == (
        'STRUCT("f1" TIMESTAMP WITH TIME ZONE, "f2" JSON)')
    assert _castable(_parse_type('STRUCT("a b" INTEGER, c TIMESTAMP WITH TIME ZONE)')) == (
        'STRUCT("a b" INTEGER, "c" TIMESTAMP WITH TIME ZONE)')
    assert _parse_type("INT128") == ("leaf", "INT128", "")
    assert _parse_type("MAP(VARCHAR, DECIMAL(2,1))") == ("map", ("leaf", "VARCHAR", ""), ("leaf", "DECIMAL", "(2,1)"))
    assert _castable(_parse_type("STRUCT(a JSON, b ENUM('x', 'y'))")) == "STRUCT(\"a\" JSON, \"b\" ENUM('x', 'y'))"
    assert _text_becomes_json(_parse_type("STRUCT(VARCHAR)[]"), _parse_type("STRUCT(JSON)[]"))
    assert not _text_becomes_json(_parse_type("INTEGER[]"), _parse_type("JSON[]"))
    # A column the snapshot types in a text spelling other than VARCHAR (a cast's target is
    # rendered VARCHAR or TEXT by sqlglot, so only the snapshot's word reaches the names).
    column = sqlglot.parse_one("region", read="duckdb")
    for spelling in ("NVARCHAR", "NCHAR", "NAME", "VARCHAR"):
        assert _duckdb_type(column, lambda c, spelling=spelling: spelling) is _TEXT, spelling
    assert _duckdb_type(sqlglot.parse_one("CAST(id AS INT128)", read="duckdb"), lambda column: "INTEGER") == "INT128"


def test_a_container_mix_whose_unification_cannot_be_asked_is_refused(monkeypatch):
    """The unification is DuckDB's answer; where it cannot be had (a cast target the reader
    here rebuilt wrongly, which an ENUM without its labels was, Codex's fourteenth review of
    #160), the mix is refused rather than read as a bind fault on no row's value."""
    from mnemiq.sql import rls

    real = rls._duckdb

    def failing(sql, params=()):
        if "COALESCE(" in sql:
            raise RuntimeError("no unification today")
        return real(sql, params)

    monkeypatch.setattr(rls, "_duckdb", failing)
    verdict = _decide("[attrs] = [id]")  # serialised, and approved when the engine can be asked
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_the_branched_literal_advice_is_duckdb_s_alone(caplog):
    # The operator's log names the cast a text literal on a branch should be written as; on
    # Oracle the branched refusal is NVL's conversion, and the DuckDB advice does not apply
    # (the review gate on b725c67).
    import logging

    with caplog.at_level(logging.WARNING, logger="mnemiq.sql.rls"):
        duck = _decide("CASE WHEN id = 1 THEN 7 ELSE 'north' END = 7")
        assert isinstance(duck, Refusal), duck
        assert "CAST('...' AS type)" in caplog.text
        caplog.clear()
        oracle = _decide("NVL(id, region) IS NOT NULL", engine="oracle")  # the NVL is the node
        assert isinstance(oracle, Refusal), oracle
        assert "can raise on a row's value" in caplog.text
        assert "CAST('...' AS type)" not in caplog.text


def test_a_literal_on_a_branch_never_taken_is_still_refused_where_it_cannot_convert():
    # `COALESCE(NULL, 7, 'x')` never reads 'x' and runs on DuckDB; the checker does not evaluate
    # which branch a row takes, so a literal that cannot convert to the branches' type is refused.
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); INSERT INTO claim VALUES (2, 'north')")
    assert con.execute("SELECT id FROM claim WHERE COALESCE(NULL, 7, 'x') = 7").fetchall() == [(2,)]
    assert isinstance(_decide("COALESCE(NULL, 7, 'x') = 7"), Refusal)
    assert isinstance(_decide("COALESCE(NULL, 7, '8') = 7"), Approved)


def test_a_literal_the_row_chooses_by_a_column_is_refused_without_evaluating_it():
    """`'0'` and `'1'` convert, so this runs on DuckDB; the checker does not evaluate literals,
    so it is refused like the `'x'` form, and the operator writes the literals as numbers."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); INSERT INTO claim VALUES (2, 'north')")
    assert con.execute("SELECT id FROM claim WHERE CASE WHEN id > 0 THEN '0' ELSE '1' END = 7"
                       ).fetchall() == []
    assert isinstance(_decide("CASE WHEN id > 0 THEN '0' ELSE '1' END = 7"), Refusal)
    assert isinstance(_decide("CASE WHEN id > 0 THEN 0 ELSE 1 END = 7"), Approved)
    assert isinstance(_decide("CASE WHEN 1 = 1 THEN '0' ELSE '1' END = 7"), Approved), "no column"


def test_a_reader_over_a_branch_that_can_be_text_is_refused():
    for row_filter in ("json_extract_string(CASE WHEN id > 0 THEN attrs ELSE region END, '$.a') IS NOT NULL",
                       "json_extract_string(coalesce(region, '{}'), '$.a') IS NOT NULL"):
        assert isinstance(_decide(row_filter), Refusal), row_filter


# Each JSON reader's arguments where it takes more than the text; the rest take the text alone.
_JSON_ARGS = {"json_extract": "region, '$.a'", "json_extract_string": "region, '$.a'",
              "json_extract_path": "region, '$.a'", "json_extract_path_text": "region, '$.a'",
              "json_value": "region, '$.a'", "json_exists": "region, '$.a'",
              "json_contains": "region, '1'", "json_merge_patch": "region, '{}'",
              "from_json": "region, '\"INTEGER\"'",
              "from_json_strict": "region, '\"INTEGER\"'", "json_transform": "region, '\"INTEGER\"'",
              "json_transform_strict": "region, '\"INTEGER\"'"}


def test_every_duckdb_json_reader_that_quotes_malformed_text_is_refused():
    """Re-measured from `duckdb_functions()` each run, so a reader DuckDB adds fails here."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR); "
                "INSERT INTO claim VALUES (1, '{\"a\": 1}'), (2, 'north')")
    names = [r[0] for r in con.execute(
        "SELECT DISTINCT function_name FROM duckdb_functions() "
        "WHERE function_type IN ('scalar', 'macro') "  # `json` itself is a macro
        "AND (function_name ILIKE 'json%' OR function_name ILIKE 'from_json%') ORDER BY 1").fetchall()]
    quoting = []
    for name in names:
        call = f"{name}({_JSON_ARGS.get(name, 'region')})"
        try:
            con.execute(f"SELECT id FROM claim WHERE ({call}) IS NOT NULL").fetchall()
        except duckdb.Error as e:
            if "north" in str(e):
                quoting.append(call)
    assert len(quoting) >= 19, quoting  # control: 19 in 1.5.4 at these arguments
    admitted = [c for c in quoting if not isinstance(_decide(f"({c}) IS NOT NULL"), Refusal)]
    assert admitted == [], admitted


# --- refused --------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine,row_filter", [
    ("duckdb", "CAST(region AS INTEGER) > 0"),
    ("duckdb", "region::INTEGER > 0"),
    ("duckdb", "CAST(UPPER(region) AS INTEGER) > 0"),     # an expression's type is unknown
    ("duckdb", "CAST(region || '' AS INTEGER) > 0"),      # Codex's second #153 reproduction
    ("duckdb", "CAST(big AS INTEGER) > 0"),               # narrowing can overflow
    ("duckdb", "strptime(region, '%Y') > TIMESTAMP '2000-01-01'"),
    ("duckdb", "json_extract_string(region, '$.a') = '1'"),
    ("duckdb", "region->>'$.a' = '1'"),
    ("duckdb", "from_json(region, '\"INTEGER\"') > 0"),
    ("duckdb", "id IN (SELECT id FROM entitlement WHERE CAST(who AS INTEGER) > 0)"),
    ("duckdb", "json(region) IS NOT NULL"),
    ("duckdb", "json_value(region, '$.a') = '1'"),
    ("duckdb", "json_contains('[1]', region)"),         # its second argument is parsed too
    ("duckdb", "json_merge_patch('{}', region) IS NOT NULL"),
    ("duckdb", "region = 7"),                             # converted implicitly, row by row
    ("duckdb", "region = (7)"),
    ("duckdb", "(region) = 7"),                           # the column in parentheses (Codex, #160)
    ("duckdb", "7 = ((region))"),
    ("duckdb", "(region) IN (1, 2)"),
    # TRY_CAST to text is text again, which the reader parses (Codex, #160)
    ("duckdb", "json_extract_string(TRY_CAST(region AS VARCHAR), '$.a') = '1'"),
    # The strict transforms raise on a well-formed value that does not fit (Codex, #160)
    ("duckdb", "json_transform_strict(attrs, '{\"a\":\"INTEGER\"}') IS NOT NULL"),
    ("duckdb", "from_json_strict(attrs, '{\"a\":\"INTEGER\"}') IS NOT NULL"),
    ("duckdb", "json_transform_strict(TRY_CAST(region AS JSON), '{\"a\":\"INTEGER\"}') IS NOT NULL"),
    ("duckdb", "region = TRUE"),
    ("oracle", "region = DATE '2000-01-01'"),             # an implicit TO_DATE
    ("duckdb", "region IN (1, 2)"),
    ("duckdb", "id = region"),
    ("duckdb", "region IN (SELECT id FROM entitlement)"),
    ("oracle", "region > 0"),                             # an implicit TO_NUMBER
    ("oracle", "CAST(region AS NUMBER) > 0"),
    ("oracle", "TO_NUMBER(region) > 0"),
    ("oracle", "TO_DATE(region, 'YYYY') > DATE '2000-01-01'"),
    ("oracle", "TO_BINARY_DOUBLE(region) > 0"),
    # sqlglot reads this as TO_DATE and drops the DEFAULT clause, so it would RUN as raising:
    ("oracle", "CAST(region AS DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYY') > DATE '2000-01-01'"),
    # NVL takes its first argument's type, and the comparison converts that text row by row
    # (ORA-01722 on the hidden row, measured; the gate's advisory on #160).
    ("oracle", "NVL(region, id) = 7"),
    ("oracle", "region = NVL(id, 7)"),
    ("oracle", "region = (SELECT 7 FROM entitlement)"),
    # NVL and NVL2 convert their other argument to the leading one's type, row by row: ORA-01722
    # on a row whose id is NULL (measured); COALESCE and CASE refuse the mix at parse instead.
    ("oracle", "NVL(id, region) IS NOT NULL"),
    ("oracle", "region = NVL(id, region)"),
    ("oracle", "NVL2(id, id, region) IS NOT NULL"),
    ("oracle", "region = NVL2(id, id, 7)"),                  # typed by its second argument
])
def test_a_conversion_that_can_raise_is_refused_naming_the_engine_s_safe_form(engine, row_filter,
                                                                               caplog):
    with caplog.at_level(logging.WARNING):
        verdict = _decide(row_filter, engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    assert not verdict.repairable
    safe = {"duckdb": "TRY_CAST(value AS type)",
            "oracle": "DEFAULT NULL ON CONVERSION ERROR"}[engine]
    assert safe in verdict.message
    assert row_filter.split(" ")[0] not in verdict.message, "the policy's text is the operator's"
    assert "(M159)" in caplog.text, "the operator's log names the conversion"


def test_a_column_the_snapshot_does_not_type_is_unknown_and_refused():
    untyped = {k: None for k in TYPES}
    snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                        columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in untyped])
    assert isinstance(_decide("CAST(id AS BIGINT) > 0", snapshot=snapshot), Refusal)
    assert isinstance(_decide("CAST(id AS BIGINT) > 0"), Approved), "control: typed INTEGER"


# --- kept -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine,row_filter", [
    ("duckdb", "TRY_CAST(region AS INTEGER) > 0"),
    ("duckdb", "CAST(id AS BIGINT) > 0"),                 # widening
    ("duckdb", "CAST(id AS INTEGER) > 0"),                # the same type
    ("duckdb", "CAST(amount AS VARCHAR) = '10'"),         # every integer renders as text
    ("duckdb", "id = CAST('7' AS INTEGER)"),              # a constant cannot fail on a row
    ("duckdb", "json_extract_string(attrs, '$.a') = '1'"),  # typed JSON already
    ("duckdb", "json_extract_string(TRY_CAST(region AS JSON), '$.a') = '1'"),
    ("duckdb", "json_transform(attrs, '{\"a\":\"INTEGER\"}') IS NOT NULL"),  # NULL where strict raises
    ("duckdb", "json_transform_strict('{\"a\":1}', '{\"a\":\"INTEGER\"}') IS NOT NULL"),  # a constant
    ("duckdb", "id IN (SELECT id FROM entitlement WHERE who = 'r')"),
    ("duckdb", "region = '7'"),
    ("duckdb", "(region) = '7'"),                         # parentheses change no type
    ("duckdb", "(id) = 7"),
    ("duckdb", "id IN (1, 2)"),
    ("duckdb", "amount = id"),
    ("oracle", "CAST(region AS NUMBER DEFAULT NULL ON CONVERSION ERROR) > 0"),
    ("oracle", "TO_NUMBER(region DEFAULT NULL ON CONVERSION ERROR) > 0"),
    ("oracle", "CAST(region AS DATE DEFAULT NULL ON CONVERSION ERROR) > DATE '2000-01-01'"),
    ("oracle", "NVL(region, id) IS NOT NULL"),                # text leads: id renders as text
])
def test_a_conversion_that_cannot_raise_is_kept(engine, row_filter):
    assert isinstance(_decide(row_filter, engine), Approved), row_filter


def test_a_write_under_a_raising_filter_is_refused_too():
    snapshot = _snapshot()
    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}),
                      row_filters={"claim": "CAST(region AS INTEGER) > 0"})
    verdict = decide_write("DELETE FROM claim WHERE id = 1", visible_schema(snapshot, grants),
                           grants, policy=build_access_policy(snapshot, grants), dialect="duckdb",
                           target="duckdb", writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_through_the_agent_the_hidden_value_reaches_no_one(tmp_path):
    """The finding's own door: 3 model calls, and the repair prompt carried the hidden value."""
    from mnemiq.adapters.duckdb import DuckDBAdapter
    from mnemiq.agent.loop import Agent
    from mnemiq.agent.synthesize import FakeSynthesizer
    from mnemiq.cache.store import L1Cache, TwoTierCache
    from mnemiq.contract import DeferralReason, IdentityContext
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, amount INT, big BIGINT, attrs JSON);"
                "CREATE TABLE entitlement(id INT, who VARCHAR);"
                "INSERT INTO claim VALUES (1, '7', 10, 1, NULL), (2, 'secret_hidden_value', 20, 2, NULL)")
    con.close()
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "CAST(region AS INTEGER) > 0"})
    generator = FakeGenerator(['{"sql": "SELECT id FROM claim", "reason": "ok"}'] * 3)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"),
                  adapter=DuckDBAdapter.duckdb(path), cache=TwoTierCache(L1Cache()))
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    answer = agent.answer(packet, _snapshot(), grants,
                          IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    assert len(generator.calls) == 1, generator.calls
    assert answer.deferred and answer.reason_code is DeferralReason.UNGOVERNABLE, answer
    assert "secret_hidden_value" not in answer.answer


# --- Oracle, live: the advised forms run over a bad hidden row ------------------------------------

DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_the_advised_forms_run_where_the_raising_one_fails():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    # Every visible column, since the policy's derived table projects them all.
    data = ("(SELECT 1 AS id, '7' AS region, 10 AS amount, 1 AS big, NULL AS attrs, "
            "DATE '2000-01-01' AS d FROM dual "
            "UNION ALL SELECT 2, 'north', 20, 2, NULL, NULL FROM dual "
            "UNION ALL SELECT NULL, 'south', 30, 3, NULL, NULL FROM dual) claim")  # NULL id: NVL reads region
    try:
        # (as run here, as decided): the snapshot knows entitlement, the database knows dual.
        for raising, decided in (("CAST(region AS NUMBER) > 0", None), ("region > 0", None),
                                 ("NVL(region, id) = 7", None), ("region = NVL(id, 7)", None),
                                 ("NVL(id, region) IS NOT NULL", None), ("region = NVL(id, region)", None),
                                 ("NVL2(id, id, region) IS NOT NULL", None),
                                 ("region = NVL2(id, id, 7)", None),
                                 ("region = (SELECT 7 FROM dual)", "region = (SELECT 7 FROM entitlement)")):
            with pytest.raises(oracledb.DatabaseError, match="ORA-01722"):
                cur.execute(f"SELECT id FROM {data} WHERE {raising}").fetchall()
            verdict = _decide(decided or raising, "oracle")
            assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, (
                raising, verdict)
        for safe in ("CAST(region AS NUMBER DEFAULT NULL ON CONVERSION ERROR) > 0",
                     "TO_NUMBER(region DEFAULT NULL ON CONVERSION ERROR) > 0"):
            verdict = _decide(safe, "oracle")
            assert isinstance(verdict, Approved), (safe, verdict)
            sql = verdict.target_sql.replace("FROM claim", f"FROM {data}")
            assert cur.execute(sql).fetchall() == [(1,)], sql
    finally:
        con.close()


# --- an extract's path decides its shape (Codex's twenty-sixth review of #160) -------------------

@pytest.mark.parametrize("row_filter", [
    "json_extract_string(attrs, region) = 7",
    "attrs ->> region = 7",
    "json_extract_string(attrs, CAST(region AS VARCHAR)) = 7",
])
def test_a_text_column_as_the_path_makes_the_extract_one_text_value_converted_on_the_row(row_filter):
    # Typed as a list for any path but one string literal, the extract beside a number read as
    # a list beside a number and was approved, where DuckDB extracts one text value through a
    # VARCHAR path and converts it on the row (measured, the hidden value in the error).
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, attrs JSON); "
                "INSERT INTO claim VALUES (1, '$.a', '{\"a\": \"north\"}'), (2, '$.a', '{\"a\": \"7\"}')")
    with pytest.raises(duckdb.ConversionException, match="north"):
        con.execute(f"SELECT id FROM claim WHERE {row_filter}").fetchall()
    verdict = _decide(row_filter)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    assert "TRY_CAST(value AS type)" in verdict.message
    for safe in ("json_extract_string(attrs, region) = 'x'", "json_extract_string(attrs, region) IS NOT NULL"):
        assert con.execute(f"SELECT id FROM claim WHERE {safe}").fetchall() is not None
        assert isinstance(_decide(safe), Approved), safe


@pytest.mark.parametrize("row_filter", [
    "json_extract_string(attrs, lower(region)) = 'x'",
    "json_extract_string(attrs, [region]) = 'x'",
])
def test_a_path_whose_shape_the_reader_cannot_tell_is_refused_naming_the_path(row_filter):
    # A call, or a list of anything but string literals: scalar or list is the engine's call,
    # and either reading approves a conversion the other refuses, so neither is taken.
    verdict = _decide(row_filter)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    assert "one string literal, a list of string literals, or a text column" in verdict.message
    # Beside a number the comparison's own refusal comes first, the walk being top-down.
    beside_a_number = _decide(row_filter.replace("= 'x'", "= 7"))
    assert isinstance(beside_a_number, Refusal) and beside_a_number.code is RefusalCode.INVALID_ROW_FILTER
    # A list of string literals keeps the list reading: refused beside a list of text, as before.
    assert isinstance(_decide("[json_extract(attrs, ['$.a'])] = [region]"), Refusal)
    assert isinstance(_decide("list_contains([region], json_extract_string(attrs, ['$.a']))"), Approved)

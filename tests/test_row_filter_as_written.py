"""M222: a row filter runs as its operator wrote it, never as sqlglot prints it back.

A row filter is SQL in the source's own dialect, applied only to a statement decided as that engine
(M218). The deciders printed the governed statement from sqlglot's parse of the filter, and a parse
with no slot for an argument prints without it: `list_reverse_sort([id, NULL], 'NULLS FIRST')` is
read as ARRAY_REVERSE_SORT, which has no null order, so a filter keeping no row on DuckDB kept the
row once governed, and an INSERT copied it on (Codex's review of #189, executed on DuckDB). A
round trip of every DuckDB function of one to five arguments changed a literal in 17 of 730, some
losing one, and a list of those is not a fix. So each filter's own text is printed, in parentheses
and closed on a new line, and sqlglot's parse of it is what the validators read and nothing more.
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import as_written
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

SCHEMA = {"claim": {"id", "region"}, "scratch": {"id", "region"}}
ROWS = ("CREATE TABLE claim (id INT, region VARCHAR); CREATE TABLE scratch (id INT, region VARCHAR);"
        "INSERT INTO claim VALUES (1, 'west'), (2, 'north'), (3, 'east')")
# Keeps no row on DuckDB: the null sorts first, so element 1 is NULL. Printed back by sqlglot as
# ARRAY_REVERSE_SORT([id, NULL]), the null sorts last and element 1 is the id.
LOSSY = "list_reverse_sort([id, NULL], 'NULLS FIRST')[1] IS NOT NULL"


def _filtered(filt: str) -> AccessPolicy:
    return AccessPolicy(row_filters={"claim": filt}, policy_schema=SCHEMA)


def _duck():
    con = duckdb.connect()
    con.execute(ROWS)
    return con


def test_sqlglot_s_print_of_the_filter_keeps_a_row_the_text_does_not():
    # The defect's control: the text keeps nothing, sqlglot's print of it keeps every row.
    printed = sqlglot.transpile(LOSSY, read="duckdb", write="duckdb")[0]
    assert "NULLS FIRST" not in printed, printed
    con = _duck()
    assert con.execute(f"SELECT id FROM claim WHERE {LOSSY}").fetchall() == []
    assert con.execute(f"SELECT id FROM claim WHERE {printed} ORDER BY id").fetchall() == [(1,), (2,), (3,)]


def test_a_governed_read_runs_the_filter_as_written():
    verdict = decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target="duckdb", policy=_filtered(LOSSY))
    assert isinstance(verdict, Approved), verdict
    assert f"({LOSSY}\n)" in verdict.target_sql and f"({LOSSY}\n)" in verdict.plan_sql, verdict.target_sql
    assert _duck().execute(verdict.target_sql).fetchall() == []


@pytest.mark.parametrize("sql", [
    "INSERT INTO scratch (id, region) SELECT id, region FROM claim",  # the filter on a read
    "DELETE FROM claim WHERE id > 0",  # the filter conjoined to the target
    "UPDATE claim SET region = 'south' WHERE id > 0",
])
def test_a_governed_write_reaches_no_row_the_filter_keeps_out(sql):
    grants = GrantSet(frozenset(SCHEMA), writable=frozenset(SCHEMA), row_filters={"claim": LOSSY})
    verdict = decide_write(sql, SCHEMA, grants, dialect="duckdb", target="duckdb",
                           policy=_filtered(LOSSY), writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite), verdict
    assert f"({LOSSY}\n)" in verdict.target_sql, verdict.target_sql
    con = _duck()
    con.execute(verdict.target_sql)
    assert con.execute("SELECT count(*) FROM scratch").fetchone() == (0,)
    assert con.execute("SELECT id, region FROM claim ORDER BY id").fetchall() == [
        (1, "west"), (2, "north"), (3, "east")]


def test_a_filter_s_or_stays_inside_its_parentheses():
    # The statement's own condition and the filter's OR: unparenthesised, `id = 2` would bind to the
    # second disjunct alone and the west row would come back.
    policy = _filtered("region = 'west' OR region = 'east'")
    verdict = decide("SELECT id FROM claim WHERE id = 2", SCHEMA, dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(verdict, Approved), verdict
    assert _duck().execute(verdict.target_sql).fetchall() == []
    fenced = decide("SELECT id FROM claim", SCHEMA, dialect="oracle", target="oracle", policy=policy)
    # sqlglot's own parentheses round the OR, and the text's inside them.
    assert "((region = 'west' OR region = 'east'\n)) AND ROWNUM > 0" in fenced.target_sql, fenced.target_sql


@pytest.mark.parametrize("engine", ["duckdb", "sqlite"])
def test_a_line_comment_in_a_filter_ends_at_its_closing_line(engine):
    policy = _filtered("region = 'west' -- the west team")
    verdict = decide("SELECT id FROM claim WHERE id < 3 ORDER BY id", SCHEMA, dialect=engine, target=engine,
                     policy=policy)
    assert isinstance(verdict, Approved), verdict
    if engine == "duckdb":
        rows = _duck().execute(verdict.target_sql).fetchall()
    else:
        con = sqlite3.connect(":memory:")
        con.executescript(ROWS.replace("VARCHAR", "TEXT") + ";")
        rows = con.execute(verdict.target_sql).fetchall()
    assert rows == [(1,)], verdict.target_sql  # the statement's own WHERE and ORDER BY survive


def test_a_closing_semicolon_is_left_off():
    verdict = decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target="duckdb",
                     policy=_filtered("region = 'west';"))
    assert isinstance(verdict, Approved), verdict
    assert ";" not in verdict.target_sql and _duck().execute(verdict.target_sql).fetchall() == [(1,)]


def test_a_mask_alone_prints_as_before():
    policy = AccessPolicy(masked={("claim", "region")}, policy_schema=SCHEMA)
    verdict = decide("SELECT id, region FROM claim", SCHEMA, dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(verdict, Approved), verdict
    assert "\n" not in verdict.target_sql and "mnemiq_row_filter" not in verdict.target_sql


PH = "mnemiq_row_filter_0123456789abcdef"


@pytest.mark.parametrize("printed", [
    "SELECT id FROM t",  # the placeholder lost
    f"SELECT id FROM t WHERE {PH} AND {PH}",  # printed twice
    f"SELECT id FROM t WHERE {PH} <> 0",  # under a comparison, as T-SQL's printer puts a bare column
    f"SELECT id FROM t WHERE id = 1 OR {PH}",  # a disjunct, where its parentheses change nothing
    f"SELECT id FROM t WHERE NOT {PH}",
    f"SELECT {PH} FROM t",  # not a condition at all
    f"SELECT id FROM t WHERE {PH} AND x = '{PH}'",  # inside a literal too
])
def test_a_placeholder_is_replaced_only_where_it_stands_alone_in_a_where(printed):
    with pytest.raises(ValueError):
        as_written(printed, {PH: "region = 'west'"}, "duckdb")


@pytest.mark.parametrize("printed", [
    f"SELECT id FROM t WHERE {PH}",
    f"SELECT id FROM t WHERE x = 1 AND ({PH})",
    f"SELECT id FROM t WHERE ({PH}) AND ROWNUM > 0",
])
def test_a_placeholder_standing_alone_in_a_where_is_replaced(printed):
    out = as_written(printed, {PH: "region = 'west'"}, "duckdb")
    assert PH not in out and "(region = 'west'\n)" in out, out


def test_a_federated_filter_reads_its_tables_as_the_federation_names_them():
    # Federation names a source's table `catalog.table` and runs `catalog.schema.table`, which the
    # tree's tables are expanded to; the text is expanded the same way, at the place the parse found
    # each catalog. Without it the filter's own subquery named a table DuckDB could not find.
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS pg; CREATE SCHEMA pg.public;"
                "CREATE TABLE pg.public.person (id INT, last_name VARCHAR);"
                "CREATE TABLE pg.public.entitlement (person_id INT);"
                "INSERT INTO pg.public.person VALUES (1, 'a'), (2, 'b'); INSERT INTO pg.public.entitlement VALUES (2)")
    schema = {"pg.person": {"id", "last_name"}, "pg.entitlement": {"person_id"}}
    filt = "id IN (SELECT person_id FROM pg.entitlement) AND list_reverse_sort([id, NULL], 'NULLS LAST')[1] > 0"
    policy = AccessPolicy(row_filters={"pg.person": filt}, policy_schema=schema)
    verdict = decide("SELECT id FROM pg.person", {"pg.person": schema["pg.person"]}, dialect="duckdb",
                     target="duckdb", policy=policy, registry={"pg": "public"})
    assert isinstance(verdict, Approved), verdict
    assert "FROM pg.public.entitlement" in verdict.target_sql and "'NULLS LAST'" in verdict.target_sql
    assert con.execute(verdict.target_sql).fetchall() == [(2,)]


def _federated(filt: str):
    schema = {"pg.person": {"id", "last_name"}, "pg.entitlement": {"person_id"}}
    policy = AccessPolicy(row_filters={"pg.person": filt}, policy_schema=schema)
    return decide("SELECT id FROM pg.person", {"pg.person": schema["pg.person"]}, dialect="duckdb",
                  target="duckdb", policy=policy, registry={"pg": "public"})


def test_a_quoted_catalog_expands_as_the_statement_s_does():
    verdict = _federated('id IN (SELECT person_id FROM "pg".entitlement)')
    assert isinstance(verdict, Approved), verdict
    assert "FROM pg.public.entitlement)" in verdict.target_sql, verdict.target_sql


def test_a_text_that_does_not_read_back_as_the_expanded_tree_is_refused(monkeypatch):
    # The text is edited where the parse placed each catalog, then read back against the tree
    # `expand_tables` expanded. Made to disagree here, the filter is refused, never run half-edited.
    from mnemiq.sql import rls
    from mnemiq.sql.verdict import Refusal, RefusalCode

    real = rls.expand_tables
    monkeypatch.setattr(rls, "expand_tables", lambda tree, registry: real(tree, {"pg": "elsewhere"}))
    verdict = _federated("id IN (SELECT person_id FROM pg.entitlement)")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.POLICY_TARGET_UNSUPPORTED, verdict


@pytest.mark.parametrize(("dialect", "filt"), [
    # sqlglot reads `secret` as a date part, a Var no column check sees; DuckDB reads the column.
    # Printed from the parse it was the literal 'SECRET'; run as written it read claim.secret,
    # a column the policy does not hold (Codex's review of #193).
    ("duckdb", "date_diff(secret, DATE '2020-01-01', DATE '2021-01-01') = 1"),
    ("duckdb", "date_diff(year, DATE '2020-01-01', DATE '2021-01-01') = 1"),
    ("duckdb", "date_trunc(secret, DATE '2020-01-01') IS NOT NULL"),
    # The parse drops the name outright; only the reprint's names miss it.
    ("duckdb", "list_reverse_sort([id], secret) IS NOT NULL"),
    # A keyword the engine reads as a column: `comment` tokenizes as COMMENT, no name token at all
    # (Codex's second review of #193).
    ("duckdb", "date_diff(comment, DATE '2020-01-01', DATE '2021-01-01') = 1"),
    ("duckdb", "list_reverse_sort([id], comment) IS NOT NULL"),
    # The parse holds a date part that reprints bare; only the date-part check sees it.
    ("duckdb", "time_bucket(INTERVAL 1 DAY, d, secret) IS NOT NULL"),
    # A type's name, which a reprint respells and so cannot be counted, dropped by the parse while
    # DuckDB reads the column: `decode(blob, date)` (Codex's third review of #193). Executed, the
    # call reads the column; renamed to one the table lacks, DuckDB refuses it as not found.
    ("duckdb", "decode(d, date) IS NOT NULL"),
    ("duckdb", "decode(d, text) IS NOT NULL"),
    ("duckdb", "list_reverse_sort([id], date) IS NOT NULL"),
    # A cast to the same type elsewhere does not stand in for it.
    ("duckdb", "decode(d, timestamp) IS NOT NULL AND TRY_CAST(id AS TIMESTAMP) IS NOT NULL"),
    # A keyword that is neither a name nor a type to the parser, dropped where DuckDB reads the
    # column: `values` (Codex's fourth review of #193). Every keyword's place is probed, not a list.
    ("duckdb", "decode(d, values) IS NOT NULL"),
    # Oracle does not reserve BOTH (V$RESERVED_WORDS, Oracle Free 23), so the word sqlglot drops
    # from `TRIM(BOTH FROM region)` may be a column there: still refused, where DuckDB and Postgres,
    # which reserve it, pass.
    ("oracle", "TRIM(BOTH FROM region) = 'west'"),
])
def test_a_name_the_parse_does_not_read_as_a_column_or_a_table_refuses_the_filter(dialect, filt):
    schema = {"claim": {"id", "d"}}
    verdict = decide("SELECT id FROM claim", schema, dialect=dialect, target=dialect,
                     policy=AccessPolicy(row_filters={"claim": filt}, policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


@pytest.mark.parametrize(("dialect", "filt"), [
    ("duckdb", "EXTRACT(year FROM d) = 2020 AND d > DATE '2020-01-01' - INTERVAL 1 DAY"),
    ("duckdb", "date_diff('year', d, DATE '2021-01-01') = 1"),
    ("duckdb", "id IN (SELECT e.id FROM entitlement AS e WHERE e.who = 'r')"),
    ("duckdb", "TRY_CAST(id AS INTEGER) > 0 AND upper(region) = 'WEST'"),
    ("oracle", "d > SYSDATE - INTERVAL '1' DAY AND EXTRACT(YEAR FROM d) = 2020"),
    ("oracle", "TRUNC(d, 'DD') = d"),
    ("postgres", "date_part('day', d) = 1 AND d > CURRENT_DATE - INTERVAL '1 day'"),
    # Types, however the reprint respells them, and typed literals.
    ("duckdb", "TRY_CAST(id AS DECIMAL(10, 2)) > 0 AND d > DATE '2020-01-01' AND TRY_CAST(id AS VARCHAR) <> ''"),
    ("duckdb", "TRY_CAST(region AS CHARACTER VARYING) = 'a' AND TRY_CAST(id AS DOUBLE PRECISION) > 0"),
    ("oracle", "d > DATE '2020-01-01' AND d < TIMESTAMP '2030-01-01 00:00:00'"),
    # A keyword the reprint respells or leaves out where no name can stand: sqlglot prints ROW as
    # ROWS (Codex's sixth review of #193), and drops a default null order and a TRIM side.
    ("postgres", "id IN (SELECT e.id FROM entitlement AS e FETCH FIRST 1 ROW ONLY)"),
    ("oracle", "id IN (SELECT e.id FROM entitlement e FETCH FIRST 1 ROW ONLY)"),
    # Each place is excused, not each word once.
    ("oracle", "id IN (SELECT e.id FROM entitlement e FETCH FIRST 1 ROW ONLY) "
               "OR id IN (SELECT e.id FROM entitlement e WHERE e.who = 'r' FETCH FIRST 1 ROW ONLY)"),
    ("duckdb", "id IN (SELECT e.id FROM entitlement AS e ORDER BY e.id DESC NULLS LAST LIMIT 10)"),
    ("duckdb", "TRIM(BOTH ' ' FROM region) = 'west' AND TRIM(LEADING 'x' FROM region) = 'west'"),
    ("duckdb", "TRY_CAST(d AS TIMESTAMP WITH TIME ZONE) IS NOT NULL"),
    # A word the engine reserves, which it never reads as a name where it stands unquoted, though
    # a name could stand there to sqlglot: `TRIM(x FROM region)` trims x (Codex's seventh review).
    ("duckdb", "TRIM(BOTH FROM region) = 'west'"),
    ("postgres", "TRIM(BOTH FROM region) = 'west'"),
    ("duckdb", "id IN (SELECT ALL id FROM entitlement)"),
    ("postgres", "id IN (SELECT ALL id FROM entitlement)"),
    ("oracle", "id IN (SELECT ALL id FROM entitlement)"),
    # The same where the parse drops the word: NULL is the literal to DuckDB, never a column.
    ("duckdb", "list_reverse_sort([id], NULL) IS NOT NULL"),
    # A quoted call's name, which a parse holds as the function and no name: DuckDB runs it.
    ("duckdb", "\"upper\"(region) = 'WEST'"),
])
def test_names_the_parse_accounts_for_pass(dialect, filt):
    schema = {"claim": {"id", "region", "d"}, "entitlement": {"id", "who"}}
    verdict = decide("SELECT id FROM claim", {"claim": schema["claim"]}, dialect=dialect, target=dialect,
                     policy=AccessPolicy(row_filters={"claim": filt}, policy_schema=schema))
    assert isinstance(verdict, Approved), (dialect, filt, verdict)


@pytest.mark.parametrize("name", ["secret", "comment"])
@pytest.mark.parametrize("sql", ["DELETE FROM claim WHERE id > 0", "UPDATE claim SET d = NULL WHERE id > 0"])
def test_a_write_target_s_filter_is_held_to_the_same_names(sql, name):
    # The target's filter is conjoined, not wrapped, and checked on its own path.
    schema = {"claim": {"id", "d"}}
    filt = f"date_diff({name}, DATE '2020-01-01', DATE '2021-01-01') = 1"
    grants = GrantSet(frozenset(schema), writable=frozenset(schema), row_filters={"claim": filt})
    verdict = decide_write(sql, schema, grants, dialect="duckdb", target="duckdb", writes_enabled=True,
                           policy=AccessPolicy(row_filters={"claim": filt}, policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


@pytest.mark.parametrize("sql", ["DELETE FROM claim WHERE id > 0", "UPDATE claim SET d = NULL WHERE id > 0"])
def test_a_write_target_s_filter_dropping_a_keyword_s_place_is_refused(sql):
    schema = {"claim": {"id", "d"}}
    filt = "decode(d, values) IS NOT NULL"
    grants = GrantSet(frozenset(schema), writable=frozenset(schema), row_filters={"claim": filt})
    verdict = decide_write(sql, schema, grants, dialect="duckdb", target="duckdb", writes_enabled=True,
                           policy=AccessPolicy(row_filters={"claim": filt}, policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_type_s_name_read_as_a_date_part_is_refused_too():
    # `date` is a type's keyword, left out of the count since a reprint respells types; a date
    # part read from it is still refused.
    schema = {"claim": {"id", "d"}}
    filt = "date_diff(date, DATE '2020-01-01', DATE '2021-01-01') = 1"
    verdict = decide("SELECT id FROM claim", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(row_filters={"claim": filt}, policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict


def test_a_column_named_like_a_type_that_the_parse_reads_passes():
    schema = {"claim": {"id", "date"}}
    verdict = decide("SELECT id FROM claim", schema, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(row_filters={"claim": "date > DATE '2020-01-01'"}, policy_schema=schema))
    assert isinstance(verdict, Approved), verdict


def test_a_string_is_never_taken_for_a_name():
    # A word is read from the text where its token stands, quotes and all: a string's token holds
    # its contents, and `'last'` read as the word `last` is probed where sqlglot drops whatever
    # stands, so a constant would refuse the filter as a dropped name.
    from mnemiq.sql.rls import _unaccounted_names

    assert _unaccounted_names("list_reverse_sort(xs, 'last')[1] IS NOT NULL", "duckdb") == []


def test_every_cast_duckdb_accepts_passes_the_name_check():
    # Each of DuckDB's single-word type names, cast to wherever DuckDB accepts it: a type's name is
    # checked where the parse leaves it, and none of these leaves one out.
    from mnemiq.sql.rls import _unaccounted_names

    parser = sqlglot.Dialect.get_or_raise("duckdb").parser_class
    tokenizer = sqlglot.Dialect.get_or_raise("duckdb").tokenizer_class
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, region VARCHAR, d DATE)")
    accepted = []
    for name in sorted(k for k, v in tokenizer.KEYWORDS.items() if v in parser.TYPE_TOKENS):
        for filt in (f"CAST(id AS {name}) IS NOT NULL", f"TRY_CAST(region AS {name}) IS NOT NULL",
                     f"id::{name} IS NOT NULL"):
            try:
                con.execute(f"SELECT id FROM claim WHERE {filt}").fetchall()
            except duckdb.Error:
                continue
            accepted.append(filt)
    assert len(accepted) > 150, len(accepted)
    assert [(f, lost) for f in accepted if (lost := _unaccounted_names(f, "duckdb"))] == []


def test_a_reserved_word_after_a_dot_is_a_name():
    # Postgres and DuckDB read a reserved word after a dot as a column (`e.all`): only a word
    # standing apart is excused.
    from sqlglot.tokens import TokenType

    from mnemiq.sql import rls
    counted = frozenset(TokenType)  # every word counts, so only the excuse decides
    for dialect in ("duckdb", "postgres"):
        apart = rls._keyword_places("id IN (SELECT ALL id FROM entitlement)", dialect, counted, {"all"})
        dotted = rls._keyword_places("id IN (SELECT e.all FROM entitlement e)", dialect, counted, {"all"})
        assert apart["all"] == 1 and dotted["all"] == 0, (dialect, apart, dotted)


def test_every_word_taken_as_reserved_is_reserved_by_the_installed_duckdb():
    import duckdb

    from mnemiq.sql.reserved_words import RESERVED
    installed = {row[0] for row in duckdb.connect().execute(
        "SELECT keyword_name FROM duckdb_keywords() WHERE keyword_category = 'reserved'").fetchall()}
    assert RESERVED["duckdb"] <= installed, sorted(RESERVED["duckdb"] - installed)



def _reads_the_column(run, query: str) -> bool:
    """Whether `query`, an unquoted word selected beside a column of that name holding 7, reads it."""
    try:
        return run(query) == [(7,)]
    except Exception:  # a parse or bind error: the word is not read as a name there
        return False


def _beside(word: str, stored: str, tail: str = "") -> str:
    return f'SELECT {word} FROM (SELECT 7 AS "{stored}"{tail}) t'


def test_duckdb_reads_no_word_taken_as_reserved_as_a_column():
    import duckdb

    from mnemiq.sql.reserved_words import RESERVED
    con = duckdb.connect()
    run = lambda q: con.execute(q).fetchall()  # noqa: E731
    assert _reads_the_column(run, _beside("id_probe", "id_probe")), "the control"
    read = [w for w in sorted(RESERVED["duckdb"]) if _reads_the_column(run, _beside(w, w))]
    assert not read, read


@pytest.mark.integration
def test_oracle_reads_no_word_taken_as_reserved_as_a_column():
    import os

    oracledb = pytest.importorskip("oracledb")
    dsn = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
    if not dsn:
        pytest.skip("set MNEMIQ_ORACLE_TEST_DSN to run the Oracle check")
    from mnemiq.sql.reserved_words import RESERVED
    with oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                          password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=dsn) as con:
        cur = con.cursor()
        run = lambda q: cur.execute(q).fetchall()  # noqa: E731
        # The column spelled as Oracle stores the unquoted word, so a name would be read as it.
        assert _reads_the_column(run, _beside("id_probe", "ID_PROBE", " FROM dual")), "the control"
        read = [w for w in sorted(RESERVED["oracle"])
                if _reads_the_column(run, _beside(w, w.upper(), " FROM dual"))]
    assert not read, read


@pytest.mark.integration
def test_postgres_reads_no_word_taken_as_reserved_as_a_column():
    import os

    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("MNEMIQ_PG_DSN")
    if not dsn:
        pytest.skip("set MNEMIQ_PG_DSN to run the Postgres check")
    from mnemiq.sql.reserved_words import RESERVED
    with psycopg.connect(dsn, autocommit=True) as con:
        run = lambda q: con.execute(q).fetchall()  # noqa: E731
        assert _reads_the_column(run, _beside("id_probe", "id_probe")), "the control"
        read = [w for w in sorted(RESERVED["postgres"]) if _reads_the_column(run, _beside(w, w))]
    assert not read, read



@pytest.mark.parametrize("filt", [
    'list_reverse_sort([id], "all") IS NOT NULL AND id IN (SELECT id FROM entitlement UNION ALL SELECT id FROM entitlement)',
    'list_reverse_sort([id], "all") IS NOT NULL AND id IN (SELECT ALL id FROM entitlement)',
    "list_reverse_sort([id], \"both\") IS NOT NULL AND TRIM(BOTH FROM region) = 'x'",
])
def test_a_quoted_name_the_parse_drops_is_never_excused_as_a_keyword(filt):
    # The same word unquoted elsewhere stands where a keyword does, and counted together its place
    # excused the dropped name (Codex's eighth review of #193). A quoted name is a name wherever
    # it stands. Every table and column here is in the schema, so nothing else refuses it.
    schema = {"claim": {"id", "region", "d"}, "entitlement": {"id", "who"}}
    verdict = decide("SELECT id FROM claim", {"claim": schema["claim"]}, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(row_filters={"claim": filt}, policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.INVALID_ROW_FILTER, verdict
    control = decide("SELECT id FROM claim", {"claim": schema["claim"]}, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(row_filters={"claim": filt.split(" AND ", 1)[1]}, policy_schema=schema))
    assert isinstance(control, Approved), control  # the keyword's half alone passes


def test_a_quoted_name_is_held_only_by_a_quoted_name_in_the_parse():
    # An unquoted `d` the parse keeps does not stand in for the quoted `"d"` it dropped.
    from mnemiq.sql import rls
    filt = 'list_reverse_sort([id], "d") IS NOT NULL AND d > 0'
    assert rls._quoted_dropped(filt, "duckdb", rls._parsed(filt, "duckdb")) == {"d"}


def test_an_oracle_write_reading_a_filtered_table_holds_its_filter_as_oracle():
    # The write path held its filters as DuckDB, which cannot read Oracle's MINUS: the decider
    # raised on a filter the read path approves (Codex's ninth review of #193).
    schema = {"claim": {"id"}, "scratch": {"id"}, "entitlement": {"id"}}
    filt = "id IN (SELECT id FROM entitlement MINUS SELECT id FROM entitlement)"
    policy = AccessPolicy(row_filters={"claim": filt}, policy_schema=schema)
    visible = {"claim": {"id"}, "scratch": {"id"}}
    verdict = decide_write("INSERT INTO scratch (id) SELECT id FROM claim", visible,
                           GrantSet(frozenset(visible), writable=frozenset({"scratch"})),
                           policy=policy, dialect="oracle", writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite), verdict
    assert filt in verdict.target_sql, verdict.target_sql  # as written

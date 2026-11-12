"""The catalogue of a Postgres database behind Superset, read from `pg_catalog` through `execute_sql`: what each
statement may read, how it pages under Superset's response size, and what the adapter gives the callers that read it.

Most tests drive `PostgresCatalogue` over a function that answers each statement, or the adapter over a recording
client that answers each statement as `execute_sql` would. Where it matters what a statement selects, the statement the
catalogue sends is run, by sqlglot's executor, over a few rows shaped like Postgres's own catalogue.
"""

from __future__ import annotations

import json
import re

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.executor import env as executor_env
from sqlglot.executor import execute as run_sql

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.pg import PostgresAdapter
from mnemiq.catalog import ColumnInfo, TableInfo
from mnemiq.catalog import introspect as read_catalog
from mnemiq.enrichment.profiling import _UNAGGREGATABLE, UNSUPPORTED, _unaggregatable, profile_table
from mnemiq.sql.functions import inventory_from, opaque_columns
from mnemiq.superset import pg_catalog
from mnemiq.superset.adapter import JINJA_DELIMITERS, SupersetSqlAdapter
from mnemiq.superset.client import SupersetResponseTooLarge, SupersetUnavailable
from mnemiq.superset.payload import ResultCut, SupersetAccessRefused, SupersetQueryFailed
from mnemiq.superset.pg_catalog import CATALOGUE_SQL, PAGE_ROWS, SCHEMA_PLACEHOLDER, PostgresCatalogue
from tests.superset_fake import RecordingClient

# Copied from Superset's `superset/config.py` at tag 7.0.0rc2: `DISALLOWED_SQL_TABLES["postgresql"]` and
# `DISALLOWED_SQL_FUNCTIONS["postgresql"]`. `execute_sql` refuses a statement on a Postgres database that reads one of
# these tables or calls one of these functions.
SUPERSET_BLOCKED_TABLES = frozenset({
    "pg_stat_activity", "pg_roles", "pg_shadow", "pg_authid", "pg_settings", "pg_config", "pg_hba_file_rules",
    "pg_stat_ssl", "pg_stat_replication", "pg_stat_wal_receiver", "pg_user",
    "information_schema.tables", "information_schema.columns", "information_schema.schemata",
    "information_schema.views", "information_schema.routines", "information_schema.role_table_grants",
    "information_schema.role_column_grants", "information_schema.role_routine_grants",
    "information_schema.table_privileges", "information_schema.column_privileges",
    "information_schema.usage_privileges", "information_schema.key_column_usage",
    "information_schema.table_constraints", "information_schema.referential_constraints",
    "information_schema.view_table_usage", "information_schema.applicable_roles", "information_schema.enabled_roles",
})
SUPERSET_BLOCKED_FUNCTIONS = frozenset({
    "current_database", "current_schema", "current_user", "session_user", "current_setting", "version",
    "inet_client_addr", "inet_client_port", "inet_server_addr", "inet_server_port",
    "pg_read_file", "pg_ls_dir", "pg_read_binary_file",
    "lo_from_bytea", "lo_export", "lo_import", "lo_put", "lo_create", "lo_creat", "lowrite", "lo_get", "loread",
    "lo_truncate", "lo_truncate64", "lo_unlink",
    "database_to_xml", "database_to_xmlschema", "query_to_xml", "query_to_xmlschema", "table_to_xml",
    "table_to_xml_and_xmlschema", "query_to_xml_and_xmlschema", "table_to_xmlschema",
    "pg_sleep", "pg_terminate_backend",
})

METHODS = ("tables", "columns", "foreign_keys", "views", "user_functions", "builtin_functions",
           "reachable_user_functions", "virtual_columns")
# The output columns each statement selects, which are the keys of each row `execute_sql` answers with: what its method
# returns, then any value of its ordering key that the method does not return.
COLUMNS = {
    "tables": ("table_name",),
    "columns": ("table_name", "column_name", "data_type", "column_number"),
    "foreign_keys": ("from_table", "from_column", "to_table", "to_column", "constraint_name", "key_position"),
    "views": ("view_name", "definition"),
    "user_functions": ("function_name",),
    "builtin_functions": ("function_name",),
    "reachable_user_functions": ("function_name",),
    "virtual_columns": ("table_name", "column_name", "expression", "column_number"),
}
# How many of a row's values its method returns, where that is fewer than the row carries.
RETURNED = {"columns": 3, "foreign_keys": 5, "virtual_columns": 3}
# The columns that carry a number, which Postgres keeps as an integer.
NUMBERS = frozenset({"column_number", "key_position"})
# Each statement's ordering key, as where in a row its values are: the order its rows come in, which no two of them
# share. A table's name is unique in its schema, a column's number in its table, a position in its constraint and a
# constraint's name on its table; a function's lowered name is made unique by DISTINCT.
ORDER_KEYS = {"tables": (0,), "columns": (0, 3), "foreign_keys": (0, 5, 4), "views": (0,), "user_functions": (0,),
              "builtin_functions": (0,), "reachable_user_functions": (0,), "virtual_columns": (0, 3)}
# Each `SupersetSqlAdapter` method and the catalogue method it answers from.
ADAPTER_METHODS = {
    "introspect": "tables", "list_columns": "columns", "foreign_keys": "foreign_keys",
    "view_definitions": "views", "user_functions": "user_functions", "builtin_functions": "builtin_functions",
    "reachable_user_functions": "reachable_user_functions", "virtual_columns": "virtual_columns",
}
_LIMIT = re.compile(r" LIMIT (\d+)$")
# A name holding what a literal must be careful with: a quote, a backslash, each of Superset's template markers, and
# the schema's placeholder.
AWKWARD_NAME = "it's \\ {{x}} {%y%} {#z#} :schema"


@pytest.fixture(autouse=True)
def _postgres_functions_in_the_executor(monkeypatch):
    """sqlglot's executor runs a statement here as Postgres would, but knows neither `chr` nor `format_type`."""
    monkeypatch.setitem(executor_env.ENV, "CHR", chr)
    monkeypatch.setitem(executor_env.ENV, "FORMAT_TYPE", lambda type_oid, modifier: "integer")


def _blocked(sql: str, default_schema: str = "public") -> set[str]:
    """The entries of Superset's lists that `sql` reads or calls, matched as Superset matches them: a bare table entry
    by the table's name in any schema, a qualified one by schema and name, with an unqualified table read in
    `default_schema`; a function entry by the name it is called by, or by any name Postgres's parser gives the same
    function (`version()` parses as `CURRENT_VERSION`)."""
    tree = sqlglot.parse_one(sql, read="postgres")
    found: set[str] = set()
    for table in tree.find_all(exp.Table):
        name, schema = table.name.lower(), (table.db or default_schema).lower()
        found |= {entry for entry in SUPERSET_BLOCKED_TABLES if entry in (name, f"{schema}.{name}")}
    aliases = sqlglot.Dialect.get_or_raise("postgres").parser_class.FUNCTIONS
    called: set[str] = set()
    for function in tree.find_all(exp.Func):
        if isinstance(function, exp.Anonymous):
            called.add(function.name.upper())
            continue
        called.add(function.sql_name())
        called |= {alias for alias, build in aliases.items() if getattr(build, "__self__", None) is type(function)}
    return found | {name for name in SUPERSET_BLOCKED_FUNCTIONS if name.upper() in called}


def _page(sql: str, schema: str = "sales") -> tuple[str, int, tuple | None]:
    """A sent statement as the method that sent it, the rows it asks for, and the values of the ordering key it
    continues after: None on a first page. A later page is its method's statement with one more condition, that a
    row's ordering key comes after those values."""
    match = _LIMIT.search(sql)
    assert match, sql
    head, limit = sql[:match.start()], int(match.group(1))
    found = []
    for method in METHODS:
        first = PostgresCatalogue(lambda sent: [], schema).statement(method)
        select, order = first.rsplit(" ORDER BY ", 1)
        if head == first:
            found.append((method, None))
        elif head.startswith(f"{select} AND ") and head.endswith(f" ORDER BY {order}"):
            found.append((method, _continued_after(head[len(select) + len(" AND "):-len(f" ORDER BY {order}")], order)))
    assert len(found) == 1, sql
    ((method, after),) = found
    return method, limit, after


def _continued_after(condition: str, order: str) -> tuple:
    """The values `condition` keeps the rows after, checked to compare the expressions a page is ordered by: as a row,
    where they are more than one, which Postgres compares column by column, as its order does."""
    comparison = sqlglot.parse_one(condition, read="postgres")
    assert isinstance(comparison, exp.GT), condition
    ordered_by = sqlglot.parse_one(f"SELECT {order}", read="postgres").expressions
    key = exp.Tuple(expressions=ordered_by) if len(ordered_by) > 1 else ordered_by[0]
    assert comparison.this.sql("postgres") == key.sql("postgres"), condition
    return _values(comparison.expression)


def _values(node: exp.Expression) -> tuple:
    """The values `node` writes, one or a row of them, each as Postgres reads it."""
    values = node.expressions if isinstance(node, exp.Tuple) else [node]
    return tuple(run_sql(f"SELECT {value.sql('postgres')} AS v", dialect="postgres").rows[0][0] for value in values)


def _key(method: str, row: tuple) -> tuple:
    return tuple(row[i] for i in ORDER_KEYS[method])


def _after(method: str, rows: list[tuple], after: tuple | None) -> list[tuple]:
    """`rows`, given in the order of `method`'s ordering key, from the first after `after`."""
    keys = [_key(method, row) for row in rows]
    assert keys == sorted(keys), "the rows are not in the order of the ordering key"
    return [row for row in rows if after is None or _key(method, row) > after]


def _returned(method: str, rows: list[tuple]) -> list[tuple]:
    """`rows` as `method` returns them: without the ordering key's values it does not return."""
    return [row[:RETURNED.get(method, len(row))] for row in rows]


def _as_superset_writes_it(sql: str) -> str:
    """`sql` as Superset 7.0.0rc2 runs it: parsed and written back by sqlglot's Postgres generator, pretty, with a
    multi-argument DISTINCT kept as written (`_normalized_generator` and `SQLStatement.format` in
    `superset/sql/parse.py`)."""
    dialect = sqlglot.Dialect.get_or_raise("postgres")
    generator = type("Normalized", (dialect.generator_class,), {"MULTI_ARG_DISTINCT": True})
    return generator(dialect=dialect, pretty=True, comments=True).generate(sqlglot.parse_one(sql, read="postgres"),
                                                                           copy=True)


def _payload(names: tuple[str, ...], rows: list[tuple], types: tuple[str, ...] | None = None) -> dict:
    """A successful `execute_sql` answer carrying `rows` under `names`, as Superset sends one: each row an object keyed
    by column name, each column its name and the dtype Superset reports."""
    types = types or ("object",) * len(names)
    return {"success": True, "rows": [dict(zip(names, row, strict=True)) for row in rows],
            "columns": [{"name": name, "type": kind} for name, kind in zip(names, types, strict=True)],
            "row_count": len(rows), "affected_rows": None, "error": None, "error_type": None}


class Catalogue:
    """A Postgres catalogue behind a fake `execute_sql`: answers each catalogue statement with its rows, given in the
    order of its ordering key, a page at a time."""

    def __init__(self, schema: str = "sales", **rows: list[tuple]) -> None:
        self.schema = schema
        self.rows = rows

    def __call__(self, request: dict) -> dict:
        method, limit, after = _page(request["sql"], self.schema)
        return _payload(COLUMNS[method], _after(method, self.rows.get(method, []), after)[:limit])


class PgCatalog:
    """Postgres's own catalogue rows for schema `sales` (its tables, each table's columns, and its functions), over
    which sqlglot's executor runs each statement as Superset writes it back. `between_pages` is called once each page
    is answered, with this catalogue and the number of pages answered so far, to change it as the database might between
    two statements."""

    def __init__(self, tables: dict[str, tuple[str, ...]] | None = None, functions: tuple[str, ...] = (),
                 between_pages=None) -> None:
        self.tables = dict(tables or {})
        self.functions = list(functions)
        self.between_pages = between_pages
        self.sent: list[str] = []

    def __call__(self, sql: str) -> list[tuple]:
        self.sent.append(sql)
        rows = run_sql(_as_superset_writes_it(sql), dialect="postgres", tables=self._rows()).rows
        if self.between_pages:
            self.between_pages(self, len(self.sent))
        return rows

    def _rows(self) -> dict:
        classes = [{"oid": 1000 + i, "relname": name, "relnamespace": 2200, "relkind": "r"}
                   for i, name in enumerate(self.tables)]
        attributes = [{"attrelid": 1000 + i, "attnum": number, "attname": column, "atttypid": 23, "atttypmod": -1,
                       "attisdropped": False}
                      for i, columns in enumerate(self.tables.values()) for number, column in enumerate(columns, 1)]
        procs = [{"oid": 5000 + i, "proname": name, "pronamespace": 2200} for i, name in enumerate(self.functions)]
        return {"pg_catalog": {"pg_namespace": [{"oid": 11, "nspname": "pg_catalog"}, {"oid": 2200, "nspname": "sales"}],
                               "pg_class": classes, "pg_attribute": attributes, "pg_proc": procs}}


def _adapter(answer, schema: str = "sales") -> SupersetSqlAdapter:
    return SupersetSqlAdapter(RecordingClient(answer), database_id=7, schema=schema)


def _sent_by(method: str, schema: str = "sales") -> list[str]:
    sent: list[str] = []
    getattr(PostgresCatalogue(lambda sql: sent.append(sql) or [], schema), method)()
    return sent


def _awkward_row(method: str) -> tuple:
    """A row of `method`'s, every name in it `AWKWARD_NAME`."""
    return tuple(7 if column in NUMBERS else AWKWARD_NAME for column in COLUMNS[method])


def _sent_after(method: str, row: tuple, monkeypatch) -> str:
    """The statement `method` sends for the page after one whose last row is `row`."""
    monkeypatch.setattr(pg_catalog, "PAGE_ROWS", 1)
    sent: list[str] = []
    pages = iter([[row], []])
    getattr(PostgresCatalogue(lambda sql: sent.append(sql) or next(pages), "sales"), method)()
    assert len(sent) == 2
    return sent[1]


def test_no_statement_touches_what_superset_blocks():
    """Superset refuses a statement reading `information_schema` on Postgres, so the catalogue reads `pg_catalog`; and
    it refuses `current_schema()` and `current_setting()`, so the schema is written into each statement instead."""
    # The planted defects are flagged: a blocked table, qualified and bare, and a blocked function, under its own name
    # and under the one sqlglot gives it.
    assert _blocked("SELECT column_name FROM information_schema.columns") == {"information_schema.columns"}
    assert _blocked("SELECT table_name FROM tables", default_schema="information_schema") == {
        "information_schema.tables"}
    assert _blocked("SELECT rolname FROM pg_catalog.pg_roles") == {"pg_roles"}
    assert _blocked("SELECT current_setting('search_path')") == {"current_setting"}
    assert _blocked("SELECT version()") == {"version"}
    # The real set passes: each statement as written, and each as the methods send it, a later page's included.
    assert set(CATALOGUE_SQL) == set(METHODS)
    for method, statement in CATALOGUE_SQL.items():
        assert _blocked(statement.select) == set(), method
        for sent in _sent_by(method):
            assert _blocked(sent, default_schema="sales") == set(), method


@pytest.mark.parametrize("method", METHODS)
def test_a_later_page_touches_nothing_superset_blocks(method, monkeypatch):
    """A later page adds its continuation, which writes the last row's values: still nothing Superset refuses, and
    nothing it reads as a template marker, whatever the names hold."""
    sent = _sent_after(method, _awkward_row(method), monkeypatch)
    assert _blocked(sent, default_schema="sales") == set()
    assert not any(marker in sent for marker in JINJA_DELIMITERS)
    assert "\\" not in sent


def test_no_statement_names_an_ordinality_column():
    """Superset runs each statement as its SQL parser writes it back, and the parser Superset 7.0.0rc2 pins drops the
    ordinality column's name from `unnest(...) WITH ORDINALITY AS k(a, b, n)` in a join, so Postgres no longer knows
    `k.n` and the statement fails. sqlglot reads that name as the unnest's `offset`."""

    def names_ordinality(sql: str) -> bool:
        return any(node.args.get("offset") for node in sqlglot.parse_one(sql, read="postgres").find_all(exp.Unnest))

    assert names_ordinality("SELECT k.n FROM c CROSS JOIN LATERAL unnest(c.a, c.b) WITH ORDINALITY AS k(a, b, n)")
    assert not any(names_ordinality(statement.select) for statement in CATALOGUE_SQL.values())


@pytest.mark.parametrize("method", METHODS)
def test_each_method_sends_its_catalogue_statement(method):
    """What the guard above reads is what is sent: each method's statement, with the schema in place, in the order of
    its ordering key, and a first page."""
    statement = CATALOGUE_SQL[method]
    select = statement.select.replace(SCHEMA_PLACEHOLDER, "'sales'")
    assert _sent_by(method) == [f"{select} ORDER BY {', '.join(statement.key)} LIMIT {PAGE_ROWS}"]


@pytest.mark.parametrize("method", METHODS)
def test_each_ordering_value_is_read_from_the_column_it_orders_by(method):
    """A later page continues after the last row's values of the ordering key, so each value is read from the output
    column that selects the very expression the statement orders by."""
    statement = CATALOGUE_SQL[method]
    selected = sqlglot.parse_one(statement.select.replace(SCHEMA_PLACEHOLDER, "'sales'"), read="postgres").expressions
    assert [selected[i].alias_or_name for i in range(len(COLUMNS[method]))] == list(COLUMNS[method])
    assert statement.key_columns == ORDER_KEYS[method]
    for expression, column in zip(statement.key, statement.key_columns, strict=True):
        assert selected[column].unalias().sql("postgres") == sqlglot.parse_one(expression, read="postgres").sql(
            "postgres"), (method, expression)


@pytest.mark.parametrize("method", METHODS)
def test_the_continuation_survives_superset_writing_the_statement_back(method, monkeypatch):
    """Superset runs a statement as its SQL parser writes it back, so the condition a later page adds must come through
    that whole: one of the conditions the WHERE clause ANDs together, comparing the expressions the page is ordered by
    (as a row, where they are more than one) with the last row's values, a quote, a backslash and a brace included."""
    row = _awkward_row(method)
    sent = _sent_after(method, row, monkeypatch)
    for sql in (sent, sqlglot.parse_one(sent, read="postgres").sql("postgres"), _as_superset_writes_it(sent)):
        tree = sqlglot.parse_one(sql, read="postgres")
        ordered_by = [ordered.this for ordered in tree.args["order"].expressions]
        key = (exp.Tuple(expressions=ordered_by) if len(ordered_by) > 1 else ordered_by[0]).sql("postgres")
        where = tree.args["where"].this
        conditions = list(where.flatten()) if isinstance(where, exp.And) else [where]
        continuations = [condition for condition in conditions
                         if isinstance(condition, exp.GT) and condition.this.sql("postgres") == key]
        assert len(continuations) == 1, sql
        assert _values(continuations[0].expression) == _key(method, row), sql


def _where_conjuncts(sql: str) -> set[str]:
    """Each condition the WHERE clause of `sql` ANDs together, written back as Postgres SQL."""
    where = sqlglot.parse_one(sql, read="postgres").args["where"].this
    return {condition.sql("postgres") for condition in where.flatten()} if isinstance(where, exp.And) else {
        where.sql("postgres")}


def test_foreign_keys_stay_inside_the_configured_schema():
    """A key is read only where the table it references is in the configured schema too, so a key to another schema's
    table never attaches to a table of the same name here; and only as declared, never once more for each partition of
    a partitioned table. Checked on the statement sent, and on it as a SQL parser writes it back, as Superset runs it."""
    (sent,) = _sent_by("foreign_keys")
    for sql in (sent, sqlglot.parse_one(sent, read="postgres").sql("postgres")):
        conditions = _where_conjuncts(sql)
        assert "n.nspname = 'sales'" in conditions
        assert {"dst.relnamespace = src.relnamespace", "src.relnamespace = dst.relnamespace"} & conditions
        assert "con.conparentid = 0" in conditions
    # The check reads the WHERE clause: a statement without the two conditions fails it.
    assert not {"dst.relnamespace = src.relnamespace", "con.conparentid = 0"} & _where_conjuncts(
        "SELECT 1 FROM pg_catalog.pg_constraint AS con WHERE con.contype = 'f'")


def test_paging_stops_on_a_short_page():
    sizes = iter([500, 500, 3])
    sent: list[str] = []

    def rows(sql: str) -> list[tuple]:
        sent.append(sql)
        return [(f"fn_{len(sent)}_{i}",) for i in range(next(sizes))]

    assert PAGE_ROWS == 500
    assert len(PostgresCatalogue(rows, "sales").user_functions()) == 1003
    # Each page after the first continues after the last function the page before it gave.
    assert [_page(sql)[1:] for sql in sent] == [(500, None), (500, ("fn_1_499",)), (500, ("fn_2_499",))]


@pytest.mark.parametrize("method", ["tables", "columns", "user_functions"])
def test_a_row_dropped_between_pages_loses_no_other_row(method, monkeypatch):
    """Each page is a statement of its own, run afresh, so the catalogue can change between two pages. A row dropped
    once the first page is read would make a page asked for by offset start one row too far on, past a row still
    there, and the read would end without it. Each page continues after the last row read instead."""
    monkeypatch.setattr(pg_catalog, "PAGE_ROWS", 3)

    def drop_the_first(catalogue: PgCatalog, pages: int) -> None:
        if pages == 1:
            del catalogue.tables["t1"]
            catalogue.functions.remove("f1")

    source = PgCatalog(tables={f"t{i}": ("a",) for i in range(1, 5)}, functions=("f1", "f2", "f3", "f4"),
                       between_pages=drop_the_first)
    expected = {"tables": ["t1", "t2", "t3", "t4"],
                "columns": [(f"t{i}", "a", "integer") for i in range(1, 5)],
                "user_functions": ["f1", "f2", "f3", "f4"]}[method]
    assert getattr(PostgresCatalogue(source, "sales"), method)() == expected


@pytest.mark.parametrize("method", ["tables", "columns", "user_functions"])
def test_a_row_added_ahead_of_the_last_one_read_between_pages_repeats_none(method, monkeypatch):
    """A row added between pages, ahead in the order of the rows read so far, would make a page asked for by offset
    start one row back, on a row already read. Continuing after the last row read, the read neither repeats a row nor
    skips one that was there throughout; the row added behind it goes unread, as one added after the read would."""
    monkeypatch.setattr(pg_catalog, "PAGE_ROWS", 3)

    def add_one_ahead(catalogue: PgCatalog, pages: int) -> None:
        if pages == 1:
            catalogue.tables["a"] = ("a",)
            catalogue.functions.append("a")

    source = PgCatalog(tables={name: ("a",) for name in "bcde"}, functions=tuple("bcde"), between_pages=add_one_ahead)
    expected = {"tables": list("bcde"), "columns": [(name, "a", "integer") for name in "bcde"],
                "user_functions": list("bcde")}[method]
    assert getattr(PostgresCatalogue(source, "sales"), method)() == expected


def test_names_holding_quotes_backslashes_and_template_markers_continue_from_the_last_row(monkeypatch):
    """A name can hold a quote, which a literal doubles; a backslash, which escapes the next character in a literal where
    a server has `standard_conforming_strings` off; `{{`, `{%` or `{#`, which Superset reads as the start of a template;
    and `:schema`, where the schema goes in a statement. With a page of one row, each name is the one the next page
    continues after: read through the adapter, which refuses to send a statement holding a template marker, every name
    arrives once, in order."""
    monkeypatch.setattr(pg_catalog, "PAGE_ROWS", 1)
    names = sorted(["it's", "a\\b", "x{{y}}", "p{%q%}", "h{#i#}", "s:schema", "''", "plain", AWKWARD_NAME])
    source = PgCatalog(tables={name: ("a",) for name in names})
    adapter = _adapter(lambda request: _payload(COLUMNS["tables"], source(request["sql"])))
    assert adapter.introspect() == names
    sent = [call.request["sql"] for call in adapter.client.calls]
    assert [_page(sql)[2] for sql in sent] == [None, *((name,) for name in names)]


@pytest.mark.parametrize("value", [None, 3.0, True])
def test_a_last_row_without_a_usable_ordering_value_raises(value, monkeypatch):
    """The next page is asked for after the last row's values, so a value that is neither text nor an integer (a null,
    a float, a boolean) leaves no page to ask for. The read raises rather than answer with the rows so far as all of
    them."""
    monkeypatch.setattr(pg_catalog, "PAGE_ROWS", 2)
    pages = iter([[("t1", "a", "integer", 1), ("t1", "b", "integer", value)]])
    with pytest.raises(SupersetQueryFailed):
        PostgresCatalogue(lambda sql: next(pages), "sales").columns()


class CuttingSource:
    """Answers each page of `method`'s `rows`, and cuts a page `too_big` says is too big, as Superset cuts a response
    over the size its MCP service allows. Records each ask as (limit, the ordering key's values it continues after, the
    rows given, or None for a cut page)."""

    def __init__(self, method: str, rows: list[tuple], too_big) -> None:
        self.method = method
        self.rows = rows
        self.too_big = too_big
        self.asks: list[tuple[int, tuple | None, list[tuple] | None]] = []

    def __call__(self, sql: str) -> list[tuple]:
        method, limit, after = _page(sql)
        assert method == self.method
        assert limit >= 1, "a page of no rows asks for nothing"
        page = _after(method, self.rows, after)[:limit]
        if self.too_big(page):
            self.asks.append((limit, after, None))
            raise ResultCut("cut")
        self.asks.append((limit, after, page))
        return page


def _over_characters(budget: int):
    return lambda page: sum(len(str(value)) for row in page for value in row) > budget


COLUMN_ROWS = [(f"t{i // 40:02}", f"c{i:04}", "integer", i % 40 + 1) for i in range(1234)]
# Most views short, three of them kilobytes long, past the first page.
VIEW_ROWS = [(f"v{i:04}", f"SELECT {i} AS n" + (" " * 3000 if i in (700, 701, 1100) else "")) for i in range(1234)]


@pytest.mark.parametrize(("method", "rows", "too_big", "expected"), [
    ("columns", COLUMN_ROWS, lambda page: len(page) > 60, _returned("columns", COLUMN_ROWS)),
    ("views", VIEW_ROWS, _over_characters(4000), [(name, body, "postgres") for name, body in VIEW_ROWS]),
])
def test_a_cut_page_is_asked_again_smaller(method, rows, too_big, expected):
    """`execute_sql` carries each row twice, so a page of 500 catalogue rows can pass the size Superset allows a
    response, and a view's definition runs to kilobytes. A cut page is asked again from the same row at half the size,
    and every row arrives once, in order."""
    source = CuttingSource(method, rows, too_big)
    assert getattr(PostgresCatalogue(source, "sales"), method)() == expected
    assert source.asks[0][:2] == (PAGE_ROWS, None)
    assert any(given is None for _, _, given in source.asks), "no page was cut"
    received: list[tuple] = []
    for (limit, after, given), (next_limit, next_after, _) in zip(source.asks, source.asks[1:]):
        # Every page continues after the last row received, whatever the size of the page before it.
        assert after == (_key(method, received[-1]) if received else None)
        if given is None:
            assert (next_limit, next_after) == (limit // 2, after)
        else:
            # Only a page shorter than the size asked for it ends the read.
            assert len(given) == limit
            received += given
            assert next_after == _key(method, received[-1])
    last_limit, last_after, last_given = source.asks[-1]
    assert last_after == _key(method, received[-1]) and last_given is not None and len(last_given) < last_limit
    assert received + last_given == rows


def test_a_page_past_the_first_is_cut_too():
    """A cut can come on any page, and there the read halves from the size it had reached."""
    source = CuttingSource("views", VIEW_ROWS, _over_characters(4000))
    PostgresCatalogue(source, "sales").views()
    assert any(given is None and after is not None for _, after, given in source.asks)


def test_page_size_returns_after_a_cut_window():
    """One wide view makes its page too large. Once the rows of the page that was cut have been read at the smaller
    sizes, the read goes back to full pages, so one wide view early does not make every later page small; and every row
    still arrives once, in order."""
    views = [(f"v{i:04}", "SELECT 1" + (" " * 7900 if i == 520 else "")) for i in range(2345)]
    source = CuttingSource("views", views, _over_characters(8000))
    assert PostgresCatalogue(source, "sales").views() == [(name, body, "postgres") for name, body in views]
    # The page after the first 500 views was cut, so its window ran to the 1000th; every page asked after that view is
    # a full one.
    assert (PAGE_ROWS, ("v0499",), None) in source.asks
    later = [limit for limit, after, _ in source.asks if after is not None and after >= ("v0999",)]
    assert later and set(later) == {PAGE_ROWS}
    received: list[tuple] = []
    for _, after, given in source.asks:
        assert after == ((received[-1][0],) if received else None), "a row was skipped or repeated"
        received += given or []
    assert received == views


def test_one_row_still_cut_raises():
    """A row Superset cuts on its own cannot be read whole, so the read raises rather than answer without it."""
    views = [("v1_small", "SELECT 1"), ("v2_huge", "SELECT " + "1 + " * 5000 + "1"), ("v3_after", "SELECT 2")]
    source = CuttingSource("views", views, _over_characters(4000))
    with pytest.raises(ResultCut):
        PostgresCatalogue(source, "sales").views()
    assert [limit for limit, _, _ in source.asks] == [500, 250, 125, 62, 31, 15, 7, 3, 1, 1]
    assert source.asks[-2:] == [(1, None, [views[0]]), (1, ("v1_small",), None)]


class TooLargeToSend(Catalogue):
    """A catalogue whose `execute_sql` refuses a page whose rows pass `budget` characters as JSON, as Superset's MCP
    service refuses an answer too large for its response size that dropping rows does not bring under it. Counts the
    pages it refused."""

    def __init__(self, budget: int, **rows: list[tuple]) -> None:
        super().__init__(**rows)
        self.budget = budget
        self.refused = 0

    def __call__(self, request: dict) -> dict:
        payload = super().__call__(request)
        if len(json.dumps(payload["rows"])) > self.budget:
            self.refused += 1
            raise SupersetResponseTooLarge("Superset's answer to execute_sql was larger than its MCP service will "
                                           "send; ask for less", detail="Error: Response too large: 61,234 bytes")
        return payload


@pytest.mark.parametrize("method", ADAPTER_METHODS)
def test_a_page_too_large_to_send_is_asked_again_smaller(method):
    """A page Superset refuses as too large to send, rather than cut, is read as a cut page: asked again smaller, so the
    adapter still answers with every row, once and in order."""
    catalogue = ADAPTER_METHODS[method]
    rows = [tuple(i if column in NUMBERS else f"{column}_{i:04}" for column in COLUMNS[catalogue]) for i in range(1234)]
    source = TooLargeToSend(8000, **{catalogue: rows})
    expected = [(*row, "postgres") for row in rows] if catalogue == "views" else _returned(catalogue, rows)
    assert [tuple(row) if isinstance(row, tuple) else (row,) for row in getattr(_adapter(source), method)()] == expected
    assert source.refused > 0


def test_one_row_too_large_to_send_raises_as_cut():
    """A single row Superset will not send on its own cannot be read whole, so the read raises `ResultCut`, as it does
    for a row cut on its own, and never answers without that row."""
    views = [("v1_small", "SELECT 1"), ("v2_huge", "SELECT " + "1 + " * 5000 + "1"), ("v3_after", "SELECT 2")]
    source = TooLargeToSend(4000, views=views)
    with pytest.raises(ResultCut) as caught:
        _adapter(source).view_definitions()
    assert isinstance(caught.value.__cause__, SupersetResponseTooLarge)


@pytest.mark.parametrize(("failure", "raised"), [
    ({"success": False, "rows": None, "columns": None, "error": "You need access to the following tables: `x`",
      "error_type": "TABLE_SECURITY_ACCESS_ERROR"}, SupersetAccessRefused),
    ({"success": False, "rows": None, "columns": None, "error": "relation does not exist", "error_type": "failed"},
     SupersetQueryFailed),
    (SupersetUnavailable("Superset's MCP service could not be reached"), SupersetUnavailable),
])
def test_a_refused_catalogue_query_raises_never_empty(failure, raised):
    """A catalogue that could not be read raises, as `SourceAdapter` requires: an empty answer would say this source
    defines no functions. Nor is the failure remembered: the next read asks again."""
    full = _payload(COLUMNS["user_functions"], [(f"fn{i:03}",) for i in range(PAGE_ROWS)])
    short = _payload(COLUMNS["user_functions"], [("median",)])
    answers = iter([full, failure, short])
    adapter = _adapter(lambda request: next(answers))
    with pytest.raises(raised):
        adapter.user_functions()
    assert len(adapter.client.calls) == 2
    assert adapter.user_functions() == ["median"]
    assert len(adapter.client.calls) == 3


def test_catalogue_quotes_stored_names():
    """A name that needs quoting in SQL (a space, mixed case) comes back exactly as Postgres stores it, and reaches the
    snapshot's catalogue that way."""
    source = Catalogue(
        tables=[("Policy Lines",), ("claims",)],
        columns=[("Policy Lines", "Line Amount", "numeric(12,2)", 1), ("Policy Lines", "policyId", "integer", 2),
                 ("claims", "id", "integer", 1)],
        foreign_keys=[("Policy Lines", "policyId", "Policies", "Id", "Policy Lines_policyId_fkey", 1)],
        views=[("Open Lines", ' SELECT "Policy Lines"."Line Amount"\n   FROM "Policy Lines";')],
        virtual_columns=[("Policy Lines", "Doubled", '("Line Amount" * (2)::numeric)', 3)],
    )
    adapter = _adapter(source)
    assert adapter.introspect() == ["Policy Lines", "claims"]
    assert adapter.list_columns() == [("Policy Lines", "Line Amount", "numeric(12,2)"),
                                      ("Policy Lines", "policyId", "integer"), ("claims", "id", "integer")]
    assert adapter.foreign_keys() == [("Policy Lines", "policyId", "Policies", "Id", "Policy Lines_policyId_fkey")]
    assert adapter.view_definitions() == [
        ("Open Lines", ' SELECT "Policy Lines"."Line Amount"\n   FROM "Policy Lines";', "postgres")]
    assert adapter.virtual_columns() == [("Policy Lines", "Doubled", '("Line Amount" * (2)::numeric)')]
    assert read_catalog(adapter) == [
        TableInfo("Policy Lines", [ColumnInfo("Line Amount", "numeric(12,2)"), ColumnInfo("policyId", "integer")]),
        TableInfo("claims", [ColumnInfo("id", "integer")]),
    ]


@pytest.mark.parametrize("schema", ["Sales", "it's", "x:schema"])
def test_the_schema_is_a_literal_matched_as_stored(schema):
    """The schema goes into each statement as one string literal, its quotes doubled, and is compared as stored:
    Superset sets the search path to the schema quoted, so `Sales` and `sales` are two schemas there too."""
    for method in METHODS:
        if method in ("user_functions", "builtin_functions"):
            continue  # neither reads the configured schema
        (sent,) = _sent_by(method, schema)
        compared = [(eq.left, eq.right) for eq in sqlglot.parse_one(sent, read="postgres").find_all(exp.EQ)]
        assert any(isinstance(left, exp.Column) and left.name == "nspname" and isinstance(right, exp.Literal)
                   and right.is_string and right.this == schema for left, right in compared), method


def test_a_schema_holding_a_backslash_is_refused_before_anything_is_sent():
    """Where a server has `standard_conforming_strings` off, a backslash in a literal escapes the quote after it."""
    sent: list[str] = []
    with pytest.raises(ValueError):
        PostgresCatalogue(lambda sql: sent.append(sql) or [], "sales\\' OR true --").tables()
    assert sent == []


# Rows shaped like `pg_namespace` and `pg_proc`: the configured schema, another user schema, and the system schemas.
NAMESPACES = [{"oid": 11, "nspname": "pg_catalog"}, {"oid": 99, "nspname": "pg_toast"},
              {"oid": 98, "nspname": "pg_toast_temp_1"}, {"oid": 13, "nspname": "information_schema"},
              {"oid": 2200, "nspname": "sales"}, {"oid": 2300, "nspname": "audit"}]
PROCS = [{"oid": 1, "proname": "count", "pronamespace": 11}, {"oid": 2, "proname": "median", "pronamespace": 11},
         {"oid": 3, "proname": "_pg_expandarray", "pronamespace": 13},
         {"oid": 4, "proname": "toast_helper", "pronamespace": 99},
         {"oid": 5, "proname": "toast_temp_helper", "pronamespace": 98},
         {"oid": 6, "proname": "Median", "pronamespace": 2200}, {"oid": 7, "proname": "median", "pronamespace": 2200},
         {"oid": 8, "proname": "safe_helper", "pronamespace": 2200},
         {"oid": 9, "proname": "audit_log", "pronamespace": 2300}]


def test_user_functions_span_schemas_reachable_ones_do_not():
    """`user_functions` lists what any user schema defines, since a query may call `audit.audit_log(...)` by its
    qualified name; `reachable_user_functions` lists only what the configured schema defines, the one schema on the
    search path Superset sets. The statements the methods send, run over a few catalogue rows."""

    def rows(sql: str) -> list[tuple]:
        return run_sql(sql, dialect="postgres",
                       tables={"pg_catalog": {"pg_namespace": NAMESPACES, "pg_proc": PROCS}}).rows

    catalogue = PostgresCatalogue(rows, "sales")
    assert catalogue.user_functions() == ["audit_log", "median", "safe_helper"]
    assert catalogue.reachable_user_functions() == ["median", "safe_helper"]
    assert catalogue.builtin_functions() == ["count", "median"]


def test_catalogue_reads_bypass_superset_cache():
    """Superset answers a statement it has run before from its result cache unless the request says `force_refresh`
    (`_try_get_cached_result` in `superset/sql/execution/executor.py`), and a catalogue statement is the same text every
    time: a function created since the cached answer would read as a builtin. So every catalogue request, every page of
    it, has Superset run the statement afresh."""
    rows = {method: [tuple(f"{column}_1" for column in COLUMNS[method])] for method in METHODS}
    rows["builtin_functions"] = [(f"fn{i:04}",) for i in range(2 * PAGE_ROWS + 3)]
    source = Catalogue(**rows)
    adapter = _adapter(source)
    for method in ADAPTER_METHODS:
        getattr(adapter, method)()
    read = [_page(call.request["sql"])[0] for call in adapter.client.calls]
    assert sorted(set(read)) == sorted(METHODS) and read.count("builtin_functions") == 3
    for call in adapter.client.calls:
        assert call.request == {"database_id": 7, "sql": call.request["sql"], "schema": "sales", "timeout": 30,
                                "force_refresh": True}


def test_function_inventory_is_read_once_per_adapter():
    """`decide` reads the function inventory on every decision, and an adapter lives for one request, so each part of
    it is read once per adapter."""
    source = Catalogue(user_functions=[("median",)], builtin_functions=[("count",), ("median",)],
                       reachable_user_functions=[("median",)], virtual_columns=[])
    adapter = _adapter(source)
    assert adapter.user_functions() == ["median"]
    assert adapter.user_functions() == ["median"]
    assert len(adapter.client.calls) == 1
    for _ in range(2):
        inventory = inventory_from(adapter)
        assert inventory.names == frozenset({"median"})
        assert opaque_columns(adapter, inventory) == frozenset()
    read = sorted(_page(call.request["sql"])[0] for call in adapter.client.calls)
    assert read == ["builtin_functions", "reachable_user_functions", "user_functions", "virtual_columns"]
    assert len(_adapter(source).user_functions()) == 1  # a new adapter, for the next request, reads afresh


@pytest.mark.parametrize("method", ADAPTER_METHODS)
def test_each_catalogue_answer_is_read_once_and_handed_out_as_a_copy(method):
    source = Catalogue(**{catalogue: [tuple(f"{column}_1" for column in COLUMNS[catalogue])]
                          for catalogue in METHODS})
    adapter = _adapter(source)
    first = getattr(adapter, method)()
    first.append("changed by a caller")
    assert getattr(adapter, method)() == first[:-1]
    assert len(adapter.client.calls) == 1
    assert _page(adapter.client.calls[0].request["sql"])[0] == ADAPTER_METHODS[method]


def _counts(request: dict) -> dict:
    """What Superset answers a counts statement: one row of counts, its columns named as Superset's `dedup`
    (`superset/result_set.py`) renames Postgres's repeated `count`."""
    n = request["sql"].count("count(")
    names = ("count", *(f"count__{i}" for i in range(1, n)))
    return _payload(names, [(7,) * n], ("int64",) * n)


@pytest.mark.parametrize("data_type", ["json", "xml", "point", "line", "lseg", "box", "path", "polygon", "circle"])
def test_json_columns_are_left_out_of_counts(data_type):
    """Postgres has no equality operator for these types, so `count(DISTINCT doc)` on one fails, and with it the one
    counts statement for the whole table. Each is spelled as `format_type` reports it."""
    adapter = _adapter(_counts)
    stats = profile_table(adapter, TableInfo("docs", [ColumnInfo("id", "integer"), ColumnInfo("doc", data_type)]))
    sent = [call.request["sql"] for call in adapter.client.calls]
    assert sent == ['SELECT count(*), count(DISTINCT "id"), count("id") FROM "docs"']
    assert {s.column: s.measurement for s in stats}["doc"] == UNSUPPORTED


def test_a_type_postgres_can_compare_is_still_counted():
    adapter = _adapter(_counts)
    profile_table(adapter, TableInfo("docs", [ColumnInfo("id", "integer"), ColumnInfo("meta", "jsonb"),
                                              ColumnInfo("body", "text")]))
    assert 'count(DISTINCT "meta")' in adapter.client.calls[0].request["sql"]


class _DirectPostgres:
    """Stands in for a direct Postgres source: the dialect a direct `PostgresAdapter` declares, and no type list."""

    dialect = "postgres"

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, sql: str) -> list[tuple]:
        self.sql.append(sql)
        return [(7,) * sql.count("count(")]


def test_direct_postgres_profiling_is_unchanged():
    """The types are the Superset adapter's own, not Postgres's entry in the dialect table: the direct adapters say
    nothing of them, and profile a `json` column as they did."""
    assert "postgres" not in _UNAGGREGATABLE
    assert _unaggregatable(PostgresAdapter.__new__(PostgresAdapter)) == frozenset()
    assert _unaggregatable(DuckDBAdapter.__new__(DuckDBAdapter)) == frozenset()
    assert _unaggregatable(_DirectPostgres()) == frozenset()
    assert "CLOB" in _unaggregatable(type("Oracle", (), {"dialect": "oracle"})())
    direct = _DirectPostgres()
    profile_table(direct, TableInfo("docs", [ColumnInfo("id", "integer"), ColumnInfo("doc", "json")]))
    assert direct.sql[0] == ('SELECT count(*), count(DISTINCT "id"), count("id"), count(DISTINCT "doc"), '
                             'count("doc") FROM "docs"')

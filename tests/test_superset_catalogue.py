"""The catalogue of a Postgres database behind Superset, read from `pg_catalog` through `execute_sql`: what each
statement may read, how it pages under Superset's response size, and what the adapter gives the callers that read it.

Most tests drive `PostgresCatalogue` over a function that answers each statement, or the adapter over a recording
client that answers each statement as `execute_sql` would. Where it matters what a statement selects, the statement the
catalogue sends is run, by sqlglot's executor, over a few rows shaped like Postgres's own catalogue.
"""

from __future__ import annotations

import re

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.executor import execute as run_sql

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.pg import PostgresAdapter
from mnemiq.catalog import ColumnInfo, TableInfo
from mnemiq.catalog import introspect as read_catalog
from mnemiq.enrichment.profiling import _UNAGGREGATABLE, UNSUPPORTED, _unaggregatable, profile_table
from mnemiq.sql.functions import inventory_from, opaque_columns
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import SupersetUnavailable
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
# The output columns each statement selects, which are the keys of each row `execute_sql` answers with.
COLUMNS = {
    "tables": ("table_name",),
    "columns": ("table_name", "column_name", "data_type"),
    "foreign_keys": ("from_table", "from_column", "to_table", "to_column", "constraint_name"),
    "views": ("view_name", "definition"),
    "user_functions": ("function_name",),
    "builtin_functions": ("function_name",),
    "reachable_user_functions": ("function_name",),
    "virtual_columns": ("table_name", "column_name", "expression"),
}
# Each `SupersetSqlAdapter` method and the catalogue method it answers from.
ADAPTER_METHODS = {
    "introspect": "tables", "list_columns": "columns", "foreign_keys": "foreign_keys",
    "view_definitions": "views", "user_functions": "user_functions", "builtin_functions": "builtin_functions",
    "reachable_user_functions": "reachable_user_functions", "virtual_columns": "virtual_columns",
}
_PAGING = re.compile(r" LIMIT (\d+) OFFSET (\d+)$")


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


def _paging(sql: str) -> tuple[str, int, int]:
    """A sent statement as its catalogue statement, and the page it asks for: LIMIT and OFFSET."""
    match = _PAGING.search(sql)
    assert match, sql
    return sql[:match.start()], int(match.group(1)), int(match.group(2))


def _payload(names: tuple[str, ...], rows: list[tuple], types: tuple[str, ...] | None = None) -> dict:
    """A successful `execute_sql` answer carrying `rows` under `names`, as Superset sends one: each row an object keyed
    by column name, each column its name and the dtype Superset reports."""
    types = types or ("object",) * len(names)
    return {"success": True, "rows": [dict(zip(names, row, strict=True)) for row in rows],
            "columns": [{"name": name, "type": kind} for name, kind in zip(names, types, strict=True)],
            "row_count": len(rows), "affected_rows": None, "error": None, "error_type": None}


class Catalogue:
    """A Postgres catalogue behind a fake `execute_sql`: answers each catalogue statement with its rows, a page at a
    time. `statements` maps each statement, as sent before its page, to the method that sends it."""

    def __init__(self, schema: str = "sales", **rows: list[tuple]) -> None:
        self.rows = rows
        self.statements = {PostgresCatalogue(lambda sql: [], schema).statement(method): method for method in METHODS}

    def __call__(self, request: dict) -> dict:
        statement, limit, offset = _paging(request["sql"])
        method = self.statements[statement]
        return _payload(COLUMNS[method], self.rows.get(method, [])[offset:offset + limit])


def _adapter(answer, schema: str = "sales") -> SupersetSqlAdapter:
    return SupersetSqlAdapter(RecordingClient(answer), database_id=7, schema=schema)


def _sent_by(method: str, schema: str = "sales") -> list[str]:
    sent: list[str] = []
    getattr(PostgresCatalogue(lambda sql: sent.append(sql) or [], schema), method)()
    return sent


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
    # The real set passes: each statement as written, and each as the methods send it.
    assert set(CATALOGUE_SQL) == set(METHODS)
    for method, statement in CATALOGUE_SQL.items():
        assert _blocked(statement) == set(), method
        for sent in _sent_by(method):
            assert _blocked(sent, default_schema="sales") == set(), method


def test_no_statement_names_an_ordinality_column():
    """Superset runs each statement as its SQL parser writes it back, and the parser Superset 7.0.0rc2 pins drops the
    ordinality column's name from `unnest(...) WITH ORDINALITY AS k(a, b, n)` in a join, so Postgres no longer knows
    `k.n` and the statement fails. sqlglot reads that name as the unnest's `offset`."""

    def names_ordinality(sql: str) -> bool:
        return any(node.args.get("offset") for node in sqlglot.parse_one(sql, read="postgres").find_all(exp.Unnest))

    assert names_ordinality("SELECT k.n FROM c CROSS JOIN LATERAL unnest(c.a, c.b) WITH ORDINALITY AS k(a, b, n)")
    assert not any(names_ordinality(statement) for statement in CATALOGUE_SQL.values())


@pytest.mark.parametrize("method", METHODS)
def test_each_method_sends_its_catalogue_statement(method):
    """What the guard above reads is what is sent: each method's statement, with the schema in place, and a page."""
    statement = CATALOGUE_SQL[method].replace(SCHEMA_PLACEHOLDER, "'sales'")
    assert _sent_by(method) == [f"{statement} LIMIT {PAGE_ROWS} OFFSET 0"]


def test_paging_stops_on_a_short_page():
    sizes = iter([500, 500, 3])
    sent: list[str] = []

    def rows(sql: str) -> list[tuple]:
        sent.append(sql)
        return [(f"fn_{len(sent)}_{i}",) for i in range(next(sizes))]

    assert PAGE_ROWS == 500
    assert len(PostgresCatalogue(rows, "sales").user_functions()) == 1003
    assert [_paging(sql)[1:] for sql in sent] == [(500, 0), (500, 500), (500, 1000)]


class CuttingSource:
    """Answers each page of `rows`, and cuts a page `too_big` says is too big, as Superset cuts a response over the
    size its MCP service allows. Records each ask as (limit, offset, the rows given, or None for a cut page)."""

    def __init__(self, rows: list[tuple], too_big) -> None:
        self.rows = rows
        self.too_big = too_big
        self.asks: list[tuple[int, int, int | None]] = []

    def __call__(self, sql: str) -> list[tuple]:
        _, limit, offset = _paging(sql)
        assert limit >= 1, "a page of no rows asks for nothing"
        page = self.rows[offset:offset + limit]
        if self.too_big(page):
            self.asks.append((limit, offset, None))
            raise ResultCut("cut")
        self.asks.append((limit, offset, len(page)))
        return page


def _over_characters(budget: int):
    return lambda page: sum(len(str(value)) for row in page for value in row) > budget


COLUMN_ROWS = [(f"t{i // 40:02}", f"c{i:04}", "integer") for i in range(1234)]
# Most views short, three of them kilobytes long, past the first page.
VIEW_ROWS = [(f"v{i:04}", f"SELECT {i} AS n" + (" " * 3000 if i in (700, 701, 1100) else "")) for i in range(1234)]


@pytest.mark.parametrize(("method", "rows", "too_big", "expected"), [
    ("columns", COLUMN_ROWS, lambda page: len(page) > 60, COLUMN_ROWS),
    ("views", VIEW_ROWS, _over_characters(4000), [(name, body, "postgres") for name, body in VIEW_ROWS]),
])
def test_a_cut_page_is_asked_again_smaller(method, rows, too_big, expected):
    """`execute_sql` carries each row twice, so a page of 500 catalogue rows can pass the size Superset allows a
    response, and a view's definition runs to kilobytes. A cut page is asked again at the same offset at half the size,
    and every row arrives once, in order."""
    source = CuttingSource(rows, too_big)
    assert getattr(PostgresCatalogue(source, "sales"), method)() == expected
    assert source.asks[0][:2] == (PAGE_ROWS, 0)
    assert any(given is None for _, _, given in source.asks), "no page was cut"
    received = 0
    for (limit, offset, given), (next_limit, next_offset, _) in zip(source.asks, source.asks[1:]):
        assert offset == received
        if given is None:
            assert (next_limit, next_offset) == (limit // 2, offset)
        else:
            # Only a page shorter than the size asked for it ends the read.
            assert given == limit
            received += given
            assert next_offset == received
    last_limit, last_offset, last_given = source.asks[-1]
    assert last_offset == received and last_given is not None and last_given < last_limit
    assert received + last_given == len(rows)


def test_a_page_past_the_first_is_cut_too():
    """A cut can come at any offset, and there the read halves from the size it had reached."""
    source = CuttingSource(VIEW_ROWS, _over_characters(4000))
    PostgresCatalogue(source, "sales").views()
    assert any(given is None and offset > 0 for _, offset, given in source.asks)


def test_one_row_still_cut_raises():
    """A row Superset cuts on its own cannot be read whole, so the read raises rather than answer without it."""
    views = [("small", "SELECT 1"), ("huge", "SELECT " + "1 + " * 5000 + "1"), ("after", "SELECT 2")]
    source = CuttingSource(views, _over_characters(4000))
    with pytest.raises(ResultCut):
        PostgresCatalogue(source, "sales").views()
    assert [limit for limit, _, _ in source.asks] == [500, 250, 125, 62, 31, 15, 7, 3, 1, 1]
    assert source.asks[-2:] == [(1, 0, 1), (1, 1, None)]


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
        columns=[("Policy Lines", "Line Amount", "numeric(12,2)"), ("Policy Lines", "policyId", "integer"),
                 ("claims", "id", "integer")],
        foreign_keys=[("Policy Lines", "policyId", "Policies", "Id", "Policy Lines_policyId_fkey")],
        views=[("Open Lines", ' SELECT "Policy Lines"."Line Amount"\n   FROM "Policy Lines";')],
        virtual_columns=[("Policy Lines", "Doubled", '("Line Amount" * (2)::numeric)')],
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
    read = sorted(source.statements[_paging(call.request["sql"])[0]] for call in adapter.client.calls)
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
    assert source.statements[_paging(adapter.client.calls[0].request["sql"])[0]] == ADAPTER_METHODS[method]


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

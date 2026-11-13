"""The catalogue of a Postgres database behind Superset: its tables, columns, foreign keys, views, functions and virtual
columns, read from `pg_catalog` through statements Superset runs.

Superset refuses, on a Postgres database, a statement that reads `information_schema` or calls `current_schema()` or
`current_setting()` (`DISALLOWED_SQL_TABLES` and `DISALLOWED_SQL_FUNCTIONS` in `superset/config.py`). So every answer
here comes from `pg_catalog`, and the schema is written into each statement as a string literal and compared with the
schema's name as Postgres stores it, case and all, as Superset's own search path names it (`get_prequeries` in
`superset/db_engine_specs/postgres.py` quotes it).

Each statement is read a page at a time, because Superset's MCP service cuts a response past the size it allows, or
sends an error in its place, and `execute_sql` carries each row twice, so a few hundred catalogue rows can reach that
size. A page cut or refused as too large is asked for again at half the size, down to a single row; a row too large on
its own raises. A statement that fails raises too, whatever the page: a catalogue that could not be read is never
answered as an empty one.

Each page after the first continues from the last row read, by the values of the key the statement is ordered by,
rather than from an offset. Every page is a statement of its own, run afresh, so the catalogue can change between two
of them: a row dropped ahead of the last one read would make an offset skip a row still there, and a row added there
would make it repeat one. Continuing from the last row, neither happens.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from mnemiq.adapters.pg_keys import DECLARED_KEYS, SCHEMA_PLACEHOLDER, CatalogueStatement, quoted
from mnemiq.superset.payload import ResultCut, SupersetQueryFailed

# The rows asked for in a statement's first page.
PAGE_ROWS = 500

# The relations Postgres's own `information_schema.tables` lists: tables, partitioned tables, views and foreign tables.
_LISTED = "c.relkind IN ('r', 'p', 'v', 'f')"
_IN_SCHEMA = ("FROM pg_catalog.pg_class AS c JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
              f"WHERE n.nspname = {SCHEMA_PLACEHOLDER}")
_FUNCTIONS = ("SELECT DISTINCT lower(p.proname) AS function_name FROM pg_catalog.pg_proc AS p "
              "JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace")


# Each catalogue method's statement.
CATALOGUE_SQL: dict[str, CatalogueStatement] = {
    "tables": CatalogueStatement(f"SELECT c.relname AS table_name {_IN_SCHEMA} AND {_LISTED}", ("c.relname",), (0,)),
    # A column's number is unique in its table, and is the order its columns come in.
    "columns": CatalogueStatement(
        "SELECT c.relname AS table_name, a.attname AS column_name, "
        "format_type(a.atttypid, a.atttypmod) AS data_type, a.attnum AS column_number "
        "FROM pg_catalog.pg_attribute AS a JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
        "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
        f"WHERE n.nspname = {SCHEMA_PLACEHOLDER} AND {_LISTED} AND a.attnum > 0 AND NOT a.attisdropped",
        ("c.relname", "a.attnum"), (0, 3)),
    # The declared keys, read by the statement the direct Postgres adapter runs too (`mnemiq.adapters.pg_keys`).
    "foreign_keys": DECLARED_KEYS,
    "views": CatalogueStatement(
        f"SELECT c.relname AS view_name, pg_get_viewdef(c.oid, true) AS definition {_IN_SCHEMA} AND c.relkind = 'v'",
        ("c.relname",), (0,)),
    # Every schema a user can define a function in: a query may call one in another schema by its qualified name.
    # DISTINCT makes a function's lowered name unique.
    "user_functions": CatalogueStatement(
        f"{_FUNCTIONS} WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') "
        "AND left(n.nspname, 8) <> 'pg_toast'",
        ("lower(p.proname)",), (0,)),
    "builtin_functions": CatalogueStatement(
        f"{_FUNCTIONS} WHERE n.nspname = 'pg_catalog'", ("lower(p.proname)",), (0,)),
    # What an unqualified call can reach: Superset sets the search path to the configured schema alone, and Postgres
    # looks in `pg_catalog` before it, whose functions are the builtins.
    "reachable_user_functions": CatalogueStatement(
        f"{_FUNCTIONS} WHERE n.nspname = {SCHEMA_PLACEHOLDER}", ("lower(p.proname)",), (0,)),
    # A virtual column's expression runs as the column is read (Postgres 18 onward); a stored one's ran when the row
    # was written.
    "virtual_columns": CatalogueStatement(
        "SELECT c.relname AS table_name, a.attname AS column_name, pg_get_expr(d.adbin, d.adrelid) AS expression, "
        "a.attnum AS column_number "
        "FROM pg_catalog.pg_attribute AS a JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
        "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
        "JOIN pg_catalog.pg_attrdef AS d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
        f"WHERE n.nspname = {SCHEMA_PLACEHOLDER} AND a.attgenerated = 'v' AND a.attnum > 0 AND NOT a.attisdropped",
        ("c.relname", "a.attnum"), (0, 3)),
}

# The dialect of a view's definition: Postgres's own SQL, as `pg_get_viewdef` writes it.
VIEW_DIALECT = "postgres"

# The characters a value a page continues from is never written with as itself, each written as `chr(<code>)` instead:
# a backslash, which escapes the character after it in a literal where a server has `standard_conforming_strings` off
# (see `schema_literal` in `mnemiq.adapters.pg_keys`), and an opening brace, which starts each of the template markers
# Superset reads, `{{`, `{%` and `{#`.
_BY_CODE = re.compile(r"([\\{])")


class PostgresCatalogue:
    """The catalogue of `schema` in the Postgres database `rows` reads, in the shapes and order `SourceAdapter` gives
    them. `rows` runs one statement and returns its rows as the database has them now, never from a cache, raising
    `ResultCut` when the answer was too large to arrive whole."""

    def __init__(self, rows: Callable[[str], list[tuple]], schema: str) -> None:
        self._rows = rows
        self._schema = schema

    def statement(self, method: str, after: tuple | None = None) -> str:
        """`method`'s statement as it is sent, before its page size: the configured schema in place as a literal, then,
        for a page after the first, only the rows after `after`, the values of the ordering key in the last row read;
        and in the order of that key.

        The rows after are those whose key, compared as a row where it has more than one expression, is greater than
        `after`: Postgres compares a row column by column, each by the same comparison its ORDER BY sorts with, so the
        two agree on which rows come after. The schema goes in first, so a value holding the placeholder stays as it
        is."""
        statement = CATALOGUE_SQL[method]
        sql = statement.in_schema(self._schema)
        if after is not None:
            sql += f" AND {_row(statement.key)} > {_row(tuple(_key_value(value) for value in after))}"
        return f"{sql} ORDER BY {', '.join(statement.key)}"

    def tables(self) -> list[str]:
        return [table for (table,) in self._read("tables")]

    def columns(self) -> list[tuple[str, str, str]]:
        return [(table, column, data_type) for table, column, data_type, _ in self._read("columns")]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        return [(from_table, from_column, to_table, to_column, constraint)
                for from_table, from_column, to_table, to_column, constraint, _ in self._read("foreign_keys")]

    def views(self) -> list[tuple[str, str, str]]:
        return [(view, definition, VIEW_DIALECT) for view, definition in self._read("views")]

    def user_functions(self) -> list[str]:
        return [name for (name,) in self._read("user_functions")]

    def builtin_functions(self) -> list[str]:
        return [name for (name,) in self._read("builtin_functions")]

    def reachable_user_functions(self) -> list[str]:
        return [name for (name,) in self._read("reachable_user_functions")]

    def virtual_columns(self) -> list[tuple[str, str, str]]:
        return [(table, column, expression) for table, column, expression, _ in self._read("virtual_columns")]

    def _read(self, method: str) -> list[tuple]:
        """Every row of `method`'s statement, a page at a time, until a page holds fewer rows than it asked for.

        Each page after the first continues from the last row read, whatever its size, so a row is neither skipped nor
        repeated, even where the catalogue changed between pages. A cut page is asked for again from the same row with
        half as many rows, and the read goes on at that size until as many rows as the cut page asked for have been
        read; then it goes back to that page's size, through each level of halving in turn, so one wide row does not
        make every later page small. A single row that is still cut raises `ResultCut`."""
        key_columns = CATALOGUE_SQL[method].key_columns
        rows: list[tuple] = []
        size = PAGE_ROWS
        # For each page that was cut and not yet read through, innermost last: its size, and how many rows will have
        # been read once its rows have.
        windows: list[tuple[int, int]] = []
        while True:
            after = tuple(rows[-1][column] for column in key_columns) if rows else None
            try:
                page = self._rows(f"{self.statement(method, after)} LIMIT {size}")
            except ResultCut:
                if size == 1:
                    raise
                windows.append((size, len(rows) + size))
                size //= 2
                continue
            rows.extend(page)
            if len(page) < size:
                return rows
            while windows and len(rows) >= windows[-1][1]:
                size, _ = windows.pop()


def _row(items: tuple[str, ...]) -> str:
    """`items` as one SQL value: the item itself when there is one, else a row of them."""
    return items[0] if len(items) == 1 else f"({', '.join(items)})"


def _key_value(value) -> str:
    """One value of the ordering key in the last row read, as SQL: an integer as itself, and text as `_text` writes it.
    Any other value raises, since the next page could not be asked for after it, and a read that stopped there would
    answer with part of the catalogue as all of it."""
    if type(value) is int:  # `type`, since a bool is an int too
        return str(value)
    if isinstance(value, str):
        return _text(value)
    raise SupersetQueryFailed(f"a catalogue row's ordering value is a {type(value).__name__}, not text or an integer, "
                              "so the rows after it cannot be asked for")


def _text(text: str) -> str:
    """`text` as a Postgres string expression that holds neither a backslash nor an opening brace: each of those as
    `chr(<code>)`, each run of text between them as a literal, its quotes doubled, joined by `||` in parentheses. A
    name in the catalogue can hold either, which a literal cannot carry here (`_BY_CODE`)."""
    parts = [f"chr({ord(part)})" if _BY_CODE.fullmatch(part) else quoted(part)
             for part in _BY_CODE.split(text) if part]
    if len(parts) <= 1:
        return parts[0] if parts else "''"
    return f"({' || '.join(parts)})"

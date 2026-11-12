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
"""

from __future__ import annotations

from collections.abc import Callable

from mnemiq.superset.payload import ResultCut

# The rows asked for in a statement's first page.
PAGE_ROWS = 500

# Where the configured schema goes in a statement, as a string literal.
SCHEMA_PLACEHOLDER = ":schema"

# The relations Postgres's own `information_schema.tables` lists: tables, partitioned tables, views and foreign tables.
_LISTED = "c.relkind IN ('r', 'p', 'v', 'f')"
_IN_SCHEMA = ("FROM pg_catalog.pg_class AS c JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
              f"WHERE n.nspname = {SCHEMA_PLACEHOLDER}")
_FUNCTIONS = ("SELECT DISTINCT lower(p.proname) AS function_name FROM pg_catalog.pg_proc AS p "
              "JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace")

# Each catalogue method's statement, before its page. Each orders its rows by a key no two rows share, so that pages
# asked one after another neither repeat a row nor skip one.
CATALOGUE_SQL: dict[str, str] = {
    "tables": f"SELECT c.relname AS table_name {_IN_SCHEMA} AND {_LISTED} ORDER BY c.relname",
    "columns": (
        "SELECT c.relname AS table_name, a.attname AS column_name, "
        "format_type(a.atttypid, a.atttypmod) AS data_type "
        "FROM pg_catalog.pg_attribute AS a JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
        "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
        f"WHERE n.nspname = {SCHEMA_PLACEHOLDER} AND {_LISTED} AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY c.relname, a.attnum"),
    # One row for each column pair of each constraint, the pairs in the constraint's own order: `conkey` and `confkey`
    # read at each position `generate_subscripts` gives. Not `unnest(conkey, confkey) WITH ORDINALITY AS k(...)`:
    # Superset runs a statement as its SQL parser writes it back, and that drops the ordinality column's name from such
    # an alias in a join, so the statement would fail. Only a key whose referenced table is in the configured schema
    # too is read: a table here is named by its name alone, so a key to another schema's table would attach to one of
    # the same name here. And only the key as declared (`conparentid = 0`): under it Postgres keeps a row of its own
    # for each partition of a partitioned table the key is on or (from Postgres 12) references, each naming that
    # partition in place of the table. A declared key's name is unique on its table, which makes the order a key.
    "foreign_keys": (
        "SELECT src.relname AS from_table, src_col.attname AS from_column, dst.relname AS to_table, "
        "dst_col.attname AS to_column, con.conname AS constraint_name "
        "FROM pg_catalog.pg_constraint AS con "
        "JOIN pg_catalog.pg_class AS src ON src.oid = con.conrelid "
        "JOIN pg_catalog.pg_namespace AS n ON n.oid = src.relnamespace "
        "JOIN pg_catalog.pg_class AS dst ON dst.oid = con.confrelid "
        "CROSS JOIN LATERAL generate_subscripts(con.conkey, 1) AS k(key_position) "
        "JOIN pg_catalog.pg_attribute AS src_col ON src_col.attrelid = con.conrelid "
        "AND src_col.attnum = con.conkey[k.key_position] "
        "JOIN pg_catalog.pg_attribute AS dst_col ON dst_col.attrelid = con.confrelid "
        "AND dst_col.attnum = con.confkey[k.key_position] "
        f"WHERE con.contype = 'f' AND n.nspname = {SCHEMA_PLACEHOLDER} AND dst.relnamespace = src.relnamespace "
        "AND con.conparentid = 0 "
        "ORDER BY src.relname, k.key_position, con.conname"),
    "views": (
        f"SELECT c.relname AS view_name, pg_get_viewdef(c.oid, true) AS definition {_IN_SCHEMA} "
        "AND c.relkind = 'v' ORDER BY c.relname"),
    # Every schema a user can define a function in: a query may call one in another schema by its qualified name.
    "user_functions": (
        f"{_FUNCTIONS} WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') "
        "AND left(n.nspname, 8) <> 'pg_toast' ORDER BY function_name"),
    "builtin_functions": f"{_FUNCTIONS} WHERE n.nspname = 'pg_catalog' ORDER BY function_name",
    # What an unqualified call can reach: Superset sets the search path to the configured schema alone, and Postgres
    # looks in `pg_catalog` before it, whose functions are the builtins.
    "reachable_user_functions": f"{_FUNCTIONS} WHERE n.nspname = {SCHEMA_PLACEHOLDER} ORDER BY function_name",
    # A virtual column's expression runs as the column is read (Postgres 18 onward); a stored one's ran when the row
    # was written.
    "virtual_columns": (
        "SELECT c.relname AS table_name, a.attname AS column_name, pg_get_expr(d.adbin, d.adrelid) AS expression "
        "FROM pg_catalog.pg_attribute AS a JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
        "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
        "JOIN pg_catalog.pg_attrdef AS d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
        f"WHERE n.nspname = {SCHEMA_PLACEHOLDER} AND a.attgenerated = 'v' AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY c.relname, a.attnum"),
}

# The dialect of a view's definition: Postgres's own SQL, as `pg_get_viewdef` writes it.
VIEW_DIALECT = "postgres"


class PostgresCatalogue:
    """The catalogue of `schema` in the Postgres database `rows` reads, in the shapes and order `SourceAdapter` gives
    them. `rows` runs one statement and returns its rows as the database has them now, never from a cache, raising
    `ResultCut` when the answer was too large to arrive whole."""

    def __init__(self, rows: Callable[[str], list[tuple]], schema: str) -> None:
        self._rows = rows
        self._schema = schema

    def statement(self, method: str) -> str:
        """`method`'s statement as it is sent, before its page: the configured schema in place as a literal."""
        return CATALOGUE_SQL[method].replace(SCHEMA_PLACEHOLDER, _literal(self._schema))

    def tables(self) -> list[str]:
        return [table for (table,) in self._read("tables")]

    def columns(self) -> list[tuple[str, str, str]]:
        return [(table, column, data_type) for table, column, data_type in self._read("columns")]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        return [(from_table, from_column, to_table, to_column, constraint)
                for from_table, from_column, to_table, to_column, constraint in self._read("foreign_keys")]

    def views(self) -> list[tuple[str, str, str]]:
        return [(view, definition, VIEW_DIALECT) for view, definition in self._read("views")]

    def user_functions(self) -> list[str]:
        return [name for (name,) in self._read("user_functions")]

    def builtin_functions(self) -> list[str]:
        return [name for (name,) in self._read("builtin_functions")]

    def reachable_user_functions(self) -> list[str]:
        return [name for (name,) in self._read("reachable_user_functions")]

    def virtual_columns(self) -> list[tuple[str, str, str]]:
        return [(table, column, expression) for table, column, expression in self._read("virtual_columns")]

    def _read(self, method: str) -> list[tuple]:
        """Every row of `method`'s statement, a page at a time, until a page holds fewer rows than it asked for.

        A cut page is asked for again at the same offset with half as many rows, and the read goes on at that size
        until the rows the cut page asked for have been read; then it goes back to that page's size, through each
        level of halving in turn, so one wide row does not make every later page small. Each page starts at the rows
        read so far, whatever its size, so none is skipped or repeated. A single row that is still cut raises
        `ResultCut`."""
        statement = self.statement(method)
        rows: list[tuple] = []
        size = PAGE_ROWS
        # For each page that was cut and not yet read through, innermost last: its size, and how many rows will have
        # been read once its rows have.
        windows: list[tuple[int, int]] = []
        while True:
            try:
                page = self._rows(f"{statement} LIMIT {size} OFFSET {len(rows)}")
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


def _literal(text: str) -> str:
    """`text` as a Postgres string literal: quoted, each `'` in it doubled.

    A backslash is refused. Where a server has `standard_conforming_strings` off, a backslash in a literal escapes the
    character after it, so a doubled quote could end the literal; and the escape string that reads the same either way,
    `E'...'`, does not survive Superset, which runs a statement as its SQL parser writes it back, and that writes the
    escape string's doubled backslashes as single ones."""
    if "\\" in text:
        raise ValueError(f"a schema name holding a backslash cannot be written as a literal here: {text!r}")
    return "'" + text.replace("'", "''") + "'"

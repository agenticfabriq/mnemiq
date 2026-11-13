"""The one statement that reads a Postgres schema's declared foreign keys, and the type and rendering it shares with the
other statements that read a Postgres catalogue.

Both ways mnemiq reads a Postgres source run it: the direct adapter, whole, through DuckDB's `postgres_query`
(`DuckDBAdapter.foreign_keys`), and the Superset catalogue, a page at a time, through Superset's `execute_sql`
(`mnemiq.superset.pg_catalog`). It reads `pg_catalog`, not `information_schema`: Superset refuses a statement on a
Postgres database that reads `information_schema`, and `information_schema.constraint_column_usage` carries no position
in the key, so a composite key's columns could only be paired with every target column rather than each with its own.

Nothing here imports `mnemiq.superset`, so the direct path reads keys without it.
"""

from __future__ import annotations

from typing import NamedTuple

# Where the configured schema goes in a statement, as a string literal.
SCHEMA_PLACEHOLDER = ":schema"


class CatalogueStatement(NamedTuple):
    """A catalogue method's statement. `select` is the statement before its order and its page, with the configured
    schema as `SCHEMA_PLACEHOLDER` and a WHERE clause that only ANDs its conditions together, so one more can follow.
    `key` is the expressions its rows are ordered by, which no two of its rows share, so that a page can continue from
    the last row read; `key_columns` is where each of their values is in a row. A value of the key that the method does
    not return is selected after the columns it does."""

    select: str
    key: tuple[str, ...]
    key_columns: tuple[int, ...]

    def in_schema(self, schema: str) -> str:
        """`select` with `schema` in place of the placeholder, as `schema_literal` writes it."""
        return self.select.replace(SCHEMA_PLACEHOLDER, schema_literal(schema))

    def whole(self, schema: str) -> str:
        """The statement that reads every row at once: in `schema`, and in the order of its key."""
        return f"{self.in_schema(schema)} ORDER BY {', '.join(self.key)}"


# One row for each column pair of each constraint, the pairs in the constraint's own order: `conkey` and `confkey`
# read at each position `generate_subscripts` gives. Not `unnest(conkey, confkey) WITH ORDINALITY AS k(...)`:
# Superset runs a statement as its SQL parser writes it back, and that drops the ordinality column's name from such
# an alias in a join, so the statement would fail. Only a key whose referenced table is in the configured schema
# too is read: a table here is named by its name alone, so a key to another schema's table would attach to one of
# the same name here. And only the key as declared (`conparentid = 0`): under it Postgres keeps a row of its own
# for each partition of a partitioned table the key is on or (from Postgres 12) references, each naming that
# partition in place of the table. The order is a key: a table's name is unique in its schema, a declared key's
# name on its table, and a position in its key.
DECLARED_KEYS = CatalogueStatement(
    "SELECT src.relname AS from_table, src_col.attname AS from_column, dst.relname AS to_table, "
    "dst_col.attname AS to_column, con.conname AS constraint_name, k.key_position AS key_position "
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
    "AND con.conparentid = 0",
    ("src.relname", "k.key_position", "con.conname"), (0, 5, 4))


def quoted(text: str) -> str:
    """`text` as a Postgres string literal, each `'` in it doubled."""
    return "'" + text.replace("'", "''") + "'"


def schema_literal(text: str) -> str:
    """`text` as a Postgres string literal: quoted, each `'` in it doubled.

    A backslash is refused. Where a server has `standard_conforming_strings` off, a backslash in a literal escapes the
    character after it, so a doubled quote could end the literal; and the escape string that reads the same either way,
    `E'...'`, does not survive Superset, which runs a statement as its SQL parser writes it back, and that writes the
    escape string's doubled backslashes as single ones."""
    if "\\" in text:
        raise ValueError(f"a schema name holding a backslash cannot be written as a literal here: {text!r}")
    return quoted(text)

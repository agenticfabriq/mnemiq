from __future__ import annotations

import re

import duckdb

# information_schema.columns renders a DuckDB FLOAT[n] array type as the literal string
# "FLOAT[n]" -- this is the only place that width is recoverable once a table already exists,
# since neither build_index nor build_definition_index keep a separate record of it.
_WIDTH_RE = re.compile(r"\[(\d+)\]$")


def declared_width(con: duckdb.DuckDBPyConnection, table: str) -> int | None:
    """The `embedding` column's declared array width for `table`, or None if the table has
    never been built."""
    row = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = 'embedding'",
        [table],
    ).fetchone()
    if row is None:
        return None
    match = _WIDTH_RE.search(row[0])
    return int(match.group(1)) if match else None


def refuse_if_mismatched(con: duckdb.DuckDBPyConnection, table: str, dim: int) -> None:
    """Raise before any destructive statement runs if `table` already exists at an embedding
    width other than `dim`.

    CREATE TABLE IF NOT EXISTS no-ops against an existing table, so pointing an already-built
    store at a differently-sized embedder (an air-gapped deployment switching to a local model,
    say BGE-M3 at 1024 after a 1536-wide hosted build) leaves the OLD column type in place. The
    next INSERT then fails with DuckDB's own ConversionException -- but only after the
    per-source DELETE ahead of it has already run and autocommitted, emptying the index with no
    guidance. Checking here, before that DELETE, turns the crash into a refusal that names the
    remedy instead. There is no in-place column resize in DuckDB, so the only way forward is
    rebuilding the store from empty.
    """
    existing = declared_width(con, table)
    if existing is not None and existing != dim:
        raise RuntimeError(
            f"{table} is indexed at width {existing}, but the configured embedder produces "
            f"width {dim} vectors. DuckDB cannot resize an array column in place, so this store "
            f"must be rebuilt from empty (delete the store file, or DROP TABLE {table}) before "
            f"it can be used with this embedder."
        )

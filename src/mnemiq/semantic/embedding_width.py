from __future__ import annotations

import re

import duckdb

# DuckDB renders a FLOAT[n] array type as the literal string "FLOAT[n]" -- this is the only
# place that width is recoverable once a table already exists, since neither build_index nor
# build_definition_index keep a separate record of it.
_WIDTH_RE = re.compile(r"\[(\d+)\]$")


def declared_width(con: duckdb.DuckDBPyConnection, table: str) -> int | None:
    """The `embedding` column's declared array width for `table`, or None if the table has
    never been built.

    `DESCRIBE <table>`, not `SELECT ... FROM information_schema.columns WHERE table_name = ?`:
    that view spans every attached catalog and schema, and a bare `table_name` filter with no
    `table_schema`/`table_catalog` matches a same-named table anywhere among them, so
    `fetchone()` picked arbitrarily between this database's own table and an unrelated one
    attached alongside it. A different bug from the "destructive statement runs before the
    check" one this module's own `refuse_if_mismatched` already closed at every call site that
    has one (build_index, build_example_index, definition_index, and federated_build's own
    pre-DELETE checks) -- this one is IN the check itself, and slipped past every one of them
    because none of their tests attached a second catalog. `DESCRIBE`
    resolves the bare name through the same catalog/schema search path a real query against the
    table would use, so it cannot make that mistake, and it hands back the column type directly:
    no predicate to get wrong. A `SELECT ... LIMIT 0` probe was considered instead and rejected
    -- `LIMIT 0` returns a result set with the right shape but no rows, and there is no width to
    read off zero rows.
    """
    try:
        rows = con.execute(f"DESCRIBE {table}").fetchall()
    except duckdb.CatalogException:
        return None
    for column_name, column_type, *_rest in rows:
        if column_name == "embedding":
            match = _WIDTH_RE.search(column_type)
            return int(match.group(1)) if match else None
    return None


def refuse_if_mismatched(con: duckdb.DuckDBPyConnection, table: str, dim: int) -> None:
    """Raise before any statement that assumes `table`'s embedding width matches `dim`, if it
    already exists at a different one.

    Two call shapes rely on this, for two different DuckDB exceptions it heads off:

    - Before a destructive rebuild (build_index, build_example_index, definition_index,
      federated_build): CREATE TABLE IF NOT EXISTS no-ops against an existing table, so pointing
      an already-built store at a differently-sized embedder (an air-gapped deployment switching
      to a local model, say BGE-M3 at 1024 after a 1536-wide hosted build) leaves the OLD column
      type in place. The next INSERT then fails with DuckDB's own ConversionException -- but only
      after the per-source DELETE ahead of it has already run and autocommitted, emptying the
      index with no guidance.
    - Before a read query (retrieve): `array_cosine_similarity(embedding, ?::FLOAT[n])` against
      a column of a different width fails with DuckDB's own BinderException, which names neither
      cause nor remedy -- an operator who runs `ask` without ever rebuilding would see only that
      crash.

    Checking here, ahead of either one, turns the crash into a refusal that names the remedy
    instead. There is no in-place column resize in DuckDB, so the only way forward is rebuilding
    the store from empty.
    """
    existing = declared_width(con, table)
    if existing is not None and existing != dim:
        raise RuntimeError(
            f"{table} is indexed at width {existing}, but the configured embedder produces "
            f"width {dim} vectors. DuckDB cannot resize an array column in place, so this store "
            f"must be rebuilt from empty (delete the store file, or DROP TABLE {table}) before "
            f"it can be used with this embedder."
        )

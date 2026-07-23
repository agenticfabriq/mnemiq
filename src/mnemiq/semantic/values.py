from __future__ import annotations

import duckdb

from mnemiq.catalog import is_key_like, is_sensitive_name
from mnemiq.contract import Snapshot
from mnemiq.semantic.textmatch import similarity

# Personal-data levels the semantic pass assigns -- never indexed. Mirrors
# enrichment.semantic._SENSITIVE; the deterministic first line is is_sensitive_name.
_SENSITIVE_PII = {"pii", "phi"}

# Physical column names avoid `column`/`value`, which are reserved words in DuckDB.
_DDL = """
CREATE TABLE IF NOT EXISTS value_index (
  source_id   TEXT,
  object_id   TEXT,
  column_name TEXT,
  value_text  TEXT
)
"""

# String types across our adapters: DuckDB VARCHAR, Postgres 'character varying', SQLite TEXT.
# Substring matching stays robust to the exact spelling each source reports.
_STRING_TYPE_TOKENS = ("char", "text", "string", "clob")


def _is_string_type(data_type: str | None) -> bool:
    if not data_type:
        return False
    lowered = data_type.lower()
    return any(token in lowered for token in _STRING_TYPE_TOKENS)


def _qualifies(column, max_distinct: int) -> bool:
    return (
        column.distinct_count is not None
        and 0 < column.distinct_count <= max_distinct
        and _is_string_type(column.data_type)
        and not is_key_like(column.name)
        and not is_sensitive_name(column.name)
        and column.pii_level not in _SENSITIVE_PII
    )


def build_value_index(
    adapter, snapshot: Snapshot, con: duckdb.DuckDBPyConnection, max_distinct: int = 200
) -> int:
    """Harvest the full distinct-value set of each bounded, non-key, non-PII string column.

    One live index per source (delete-then-insert, like save_snapshot): a stale value set is
    worse than a missing one, because at check time it is indistinguishable from a fresh one.
    Only columns whose entire value set we can hold are indexed -- that is what makes a later
    'not found' a genuine signal rather than an artifact of sampling.
    """
    con.execute(_DDL)
    con.execute("DELETE FROM value_index WHERE source_id = ?", [snapshot.source_id])

    physical = {sb.object_id: sb.source_object for sb in snapshot.source_bindings}
    rows: list[list[str]] = []
    for column in snapshot.columns:
        if not _qualifies(column, max_distinct):
            continue
        table = physical.get(column.object_id, column.object_id)
        try:
            values = adapter.execute(
                f'SELECT DISTINCT "{column.name}" FROM "{table}" '
                f'WHERE "{column.name}" IS NOT NULL'
            )
        except Exception:
            continue  # fail-soft: one unreadable column never sinks enrichment
        for (value,) in values:
            rows.append([snapshot.source_id, column.object_id, column.name, str(value)])

    if rows:
        con.executemany(
            "INSERT INTO value_index (source_id, object_id, column_name, value_text) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


class ValueIndex:
    """Read side of the value index. Resilient to a store built before value grounding: the
    table is ensured on construction, so an un-indexed store simply answers 'not indexed'."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con
        con.execute(_DDL)

    def has(self, object_id: str, column: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM value_index WHERE object_id = ? AND column_name = ? LIMIT 1",
            [object_id, column],
        ).fetchone()
        return row is not None

    def contains(self, object_id: str, column: str, literal: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM value_index "
            "WHERE object_id = ? AND column_name = ? AND value_text = ? LIMIT 1",
            [object_id, column, literal],
        ).fetchone()
        return row is not None

    def _values(self, object_id: str, column: str) -> list[str]:
        return [
            r[0]
            for r in self._con.execute(
                "SELECT value_text FROM value_index WHERE object_id = ? AND column_name = ?",
                [object_id, column],
            ).fetchall()
        ]

    def nearest(self, object_id: str, column: str, literal: str, k: int = 8) -> list[str]:
        ranked = sorted(
            self._values(object_id, column),
            key=lambda v: similarity(literal, v),
            reverse=True,
        )
        return ranked[:k]  # returns all when the column has <= k values

from __future__ import annotations

import duckdb

from mnemiq.contract import Snapshot

_DDL = (
    "CREATE TABLE IF NOT EXISTS snapshot ("
    "  version TEXT, source_id TEXT, created_at TEXT, seq BIGINT, doc TEXT)"
)


def _ensure(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_DDL)


def save_snapshot(con: duckdb.DuckDBPyConnection, snapshot: Snapshot) -> None:
    _ensure(con)
    seq = con.execute("SELECT coalesce(max(seq), 0) + 1 FROM snapshot").fetchone()[0]
    con.execute(
        "DELETE FROM snapshot WHERE version = ? AND source_id = ?",
        [snapshot.version, snapshot.source_id],
    )
    con.execute(
        "INSERT INTO snapshot (version, source_id, created_at, seq, doc) VALUES (?, ?, ?, ?, ?)",
        [
            snapshot.version,
            snapshot.source_id,
            snapshot.created_at,
            seq,
            snapshot.model_dump_json(by_alias=True),
        ],
    )


def load_snapshot(con: duckdb.DuckDBPyConnection, version: str) -> Snapshot:
    _ensure(con)
    row = con.execute("SELECT doc FROM snapshot WHERE version = ?", [version]).fetchone()
    if row is None:
        raise KeyError(f"no snapshot with version {version}")
    return Snapshot.model_validate_json(row[0])


def has_snapshot(con: duckdb.DuckDBPyConnection, version: str) -> bool:
    _ensure(con)
    row = con.execute("SELECT 1 FROM snapshot WHERE version = ? LIMIT 1", [version]).fetchone()
    return row is not None


def current_version(con: duckdb.DuckDBPyConnection, source_id: str) -> str | None:
    _ensure(con)
    row = con.execute(
        "SELECT version FROM snapshot WHERE source_id = ? ORDER BY seq DESC LIMIT 1",
        [source_id],
    ).fetchone()
    return row[0] if row else None

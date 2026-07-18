from __future__ import annotations

import duckdb

from mnemiq.store.snapshot_store import current_version, has_snapshot

_CREATE = (
    "CREATE TABLE IF NOT EXISTS mnemiq_version "
    "(source_id TEXT PRIMARY KEY, version TEXT, updated_at TIMESTAMPTZ DEFAULT now())"
)
_UPSERT = (
    "INSERT INTO mnemiq_version (source_id, version) VALUES (%s, %s) "
    "ON CONFLICT (source_id) DO UPDATE SET version = EXCLUDED.version, updated_at = now()"
)


def publish_version(control_dsn: str, source_id: str, version: str) -> None:
    import psycopg

    with psycopg.connect(control_dsn, autocommit=True) as con:
        con.execute(_CREATE)
        con.execute(_UPSERT, (source_id, version))


def current_pointer(control_dsn: str, source_id: str) -> str | None:
    import psycopg

    with psycopg.connect(control_dsn, autocommit=True) as con:
        con.execute(_CREATE)
        row = con.execute(
            "SELECT version FROM mnemiq_version WHERE source_id = %s", (source_id,)
        ).fetchone()
        return row[0] if row else None


def resolve_version(
    con: duckdb.DuckDBPyConnection, control_dsn: str | None, source_id: str
) -> str | None:
    """Shared pointer if set and locally available, else the local current version."""
    if control_dsn:
        ptr = current_pointer(control_dsn, source_id)
        if ptr is not None and has_snapshot(con, ptr):
            return ptr
    return current_version(con, source_id)

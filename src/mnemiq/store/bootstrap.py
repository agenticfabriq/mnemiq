from __future__ import annotations

import duckdb

from mnemiq.duckdb_extensions import load_extension


def init_store(path: str, *, local_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the store with its `vss` and `fts` extensions. Every door in `mnemiq` and `scripts`
    passes `settings.local_only` -- a test holds them to it -- because the default installs by
    download, which MNEMIQ_LOCAL_ONLY forbids (M107)."""
    con = duckdb.connect(path)
    load_extension(con, "vss", local_only=local_only)
    load_extension(con, "fts", local_only=local_only)
    con.execute(
        "CREATE TABLE IF NOT EXISTS enrichment_version ("
        "  version TEXT PRIMARY KEY, source_id TEXT, created_at TIMESTAMP DEFAULT now())"
    )
    return con

from __future__ import annotations

import duckdb


def init_store(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("INSTALL fts; LOAD fts;")
    con.execute(
        "CREATE TABLE IF NOT EXISTS enrichment_version ("
        "  version TEXT PRIMARY KEY, source_id TEXT, created_at TIMESTAMP DEFAULT now())"
    )
    return con

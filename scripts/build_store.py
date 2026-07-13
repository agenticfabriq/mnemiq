"""Index the current snapshot of a source into the searchable semantic store."""

from __future__ import annotations

import os
import sys

import duckdb

from mnemiq.config import Settings
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot


def main() -> int:
    settings = Settings.from_env()
    source_id = os.getenv("MNEMIQ_SOURCE_ID", "acme")
    store_path = os.getenv("MNEMIQ_STORE_PATH", "mnemiq.duckdb")

    con: duckdb.DuckDBPyConnection = init_store(store_path)
    version = current_version(con, source_id)
    if version is None:
        print(f"no snapshot for source {source_id!r} in {store_path}", file=sys.stderr)
        print("run scripts/enrich_acme.py first", file=sys.stderr)
        return 1

    snapshot = load_snapshot(con, version)
    print(f"indexing snapshot {version} ({len(snapshot.source_bindings)} tables)...")
    indexed = build_index(con, snapshot, LLMEmbedder(settings))
    print(f"indexed {indexed} cards -> {store_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

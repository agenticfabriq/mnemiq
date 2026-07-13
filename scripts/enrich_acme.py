"""Enrich the ACME source end to end (structural + semantic) and persist the snapshot."""

from __future__ import annotations

import os
import sys

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.config import Settings
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.llm.client import LLMClient
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot


def main() -> int:
    settings = Settings.from_env()
    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN", file=sys.stderr)
        return 1

    adapter = DuckDBPostgresAdapter(settings.pg_dsn)

    print("structural pass...")
    snapshot = enrich_structural(adapter, "acme")
    print(
        f"  {len(snapshot.source_bindings)} tables, {len(snapshot.columns)} columns, "
        f"{len(snapshot.relationships)} relationships"
    )

    print(f"semantic pass ({len(snapshot.source_bindings)} LLM calls)...")
    snapshot = enrich_semantic(snapshot, LLMEnricher(LLMClient(settings)))

    described = sum(1 for c in snapshot.columns if c.description)
    codes = [cv for c in snapshot.columns for cv in c.coded_values]
    explained = sum(1 for cv in codes if cv.meaning)
    failed = [j.id for j in snapshot.jobs if j.status == "failed"]

    print(f"  described {described}/{len(snapshot.columns)} columns")
    print(f"  explained {explained}/{len(codes)} codes")
    if failed:
        print(f"  failed: {', '.join(failed)}")

    store_path = os.getenv("MNEMIQ_STORE_PATH", "mnemiq.duckdb")
    save_snapshot(init_store(store_path), snapshot)
    print(f"snapshot {snapshot.version} -> {store_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

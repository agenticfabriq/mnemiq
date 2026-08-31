"""Ask a question. Get SQL the engine is willing to run -- or an honest deferral."""

from __future__ import annotations

import os
import sys

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.authz.grants import DenyAll, FileAuthzProvider, GrantSet
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.generate.generator import LLMGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot


class _AllowAll:
    """Only for a local demo run with no policy file. Never a default in a real deployment."""

    def __init__(self, objects):
        self._grants = GrantSet(frozenset(objects))

    def grants_for(self, identity):
        return self._grants


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: ask.py "your question"', file=sys.stderr)
        return 2

    question = sys.argv[1]
    settings = Settings.from_env()
    source_id = os.getenv("MNEMIQ_SOURCE_ID", "acme")
    store_path = os.getenv("MNEMIQ_STORE_PATH", "mnemiq.duckdb")

    con = init_store(store_path)
    version = current_version(con, source_id)
    if version is None:
        print(
            "no snapshot: run scripts/enrich_acme.py then scripts/build_store.py",
            file=sys.stderr,
        )
        return 1
    snapshot = load_snapshot(con, version)

    if settings.authz_path:
        authz = FileAuthzProvider(settings.authz_path)
    elif os.getenv("MNEMIQ_ALLOW_ALL") == "1":
        indexed = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
        authz = _AllowAll(indexed)
        print("WARNING: MNEMIQ_ALLOW_ALL=1 -- every table is visible", file=sys.stderr)
    else:
        authz = DenyAll()  # no policy is not permission

    identity = IdentityContext(
        tenant_id="local",
        principal_id=os.getenv("MNEMIQ_PRINCIPAL", "local"),
        roles=[r for r in os.getenv("MNEMIQ_ROLES", "").split(",") if r],
    )
    embedder = LLMEmbedder(settings)

    packet = retrieve(con, question, identity, authz, embedder, k=5)
    print(f"retrieved: {[c.object_id for c in packet.cards]}")

    adapter = DuckDBPostgresAdapter(settings.pg_dsn) if settings.pg_dsn else None
    outcome = plan_query(
        packet,
        snapshot,
        authz.grants_for(identity),
        LLMGenerator(LLMClient(settings),
                     declare_assumed_terms=settings.guard_undefined_terms),
        adapter=adapter,
        target="duckdb",
        guard_undefined_terms=settings.guard_undefined_terms,
    )

    if isinstance(outcome, Deferred):
        print(f"\nDEFERRED: {outcome.reason}")
        return 0

    print(f"\ntables: {outcome.tables}")
    print(f"columns: {outcome.columns}")
    print(f"\nSQL:\n{outcome.target_sql}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

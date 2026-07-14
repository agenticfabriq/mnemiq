"""Ask a question. Get an answer, the SQL behind it, and the trace."""

from __future__ import annotations

import os
import sys

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.authz.grants import DenyAll, FileAuthzProvider, GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.generate.generator import LLMGenerator
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
        print('usage: answer.py "your question"', file=sys.stderr)
        return 2

    question = sys.argv[1]
    settings = Settings.from_env()
    source_id = os.getenv("MNEMIQ_SOURCE_ID", "acme")
    store_path = os.getenv("MNEMIQ_STORE_PATH", "mnemiq.duckdb")

    con = init_store(store_path)
    version = current_version(con, source_id)
    if version is None:
        print("no snapshot: run enrich_acme.py then build_store.py", file=sys.stderr)
        return 1
    snapshot = load_snapshot(con, version)

    if settings.authz_path:
        authz = FileAuthzProvider(settings.authz_path)
    elif os.getenv("MNEMIQ_ALLOW_ALL") == "1":
        indexed = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
        authz = _AllowAll(indexed)
        print("WARNING: MNEMIQ_ALLOW_ALL=1 -- every table is visible", file=sys.stderr)
    else:
        authz = DenyAll()

    identity = IdentityContext(
        tenant_id="local",
        principal_id=os.getenv("MNEMIQ_PRINCIPAL", "local"),
        roles=[r for r in os.getenv("MNEMIQ_ROLES", "").split(",") if r],
    )

    packet = retrieve(con, question, identity, authz, LLMEmbedder(settings), k=5)
    print(f"retrieved: {[c.object_id for c in packet.cards]}\n")

    agent = Agent(
        generator=LLMGenerator(LLMClient(settings)),
        synthesizer=LLMSynthesizer(LLMClient(settings)),
        adapter=DuckDBPostgresAdapter(settings.pg_dsn),
        cache=TwoTierCache(L1Cache()),
    )
    result = agent.answer(packet, snapshot, authz.grants_for(identity), identity)

    if result.deferred:
        print(f"DEFERRED: {result.answer}")
        return 0

    print(f"ANSWER: {result.answer}\n")
    print(f"SQL:\n{result.trace.target_sql}\n")
    print(f"tables: {result.trace.tables_used}")
    print(f"timing: {result.trace.timing}")
    print(f"cached: {result.cached}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

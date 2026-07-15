from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent, AgentAnswer
from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.authz.grants import AuthzProvider, DenyAll, FileAuthzProvider
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext, Snapshot
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import Embedder, LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot


class SnapshotMissing(RuntimeError):
    """No enriched snapshot in the store -- the source has not been indexed yet."""


@dataclass
class Runtime:
    """The engine assembled once for product use (CLI, MCP). Distinct from eval's in-memory
    build_engine: this loads a *persisted* snapshot and connects a real source read-only."""

    con: Any
    snapshot: Snapshot | None
    adapter: Any
    agent: Agent | None
    embedder: Embedder | None
    authz: AuthzProvider
    settings: Settings | None

    def ask(self, question: str, identity: IdentityContext) -> AgentAnswer:
        packet = retrieve(self.con, question, identity, self.authz, self.embedder, k=6)
        grants = self.authz.grants_for(identity)
        return self.agent.answer(packet, self.snapshot, grants, identity)

    def schema(self, identity: IdentityContext) -> list[dict]:
        """The tables/cards this identity may see -- access-scoped, so metadata never leaks."""
        grants = self.authz.grants_for(identity)
        rows = self.con.execute("SELECT object_id, card FROM semantic_object").fetchall()
        allowed = grants.objects
        return [{"object_id": oid, "card": card} for oid, card in rows if oid in allowed]


def _authz(settings: Settings) -> AuthzProvider:
    return FileAuthzProvider(settings.authz_path) if settings.authz_path else DenyAll()


def build_runtime(settings: Settings) -> Runtime:
    con = init_store(settings.store_path)
    version = current_version(con, settings.source_id)
    if version is None:
        raise SnapshotMissing(
            f"no snapshot for source {settings.source_id!r} in {settings.store_path!r} -- "
            "run `mnemiq enrich` then `mnemiq build` first"
        )
    snapshot = load_snapshot(con, version)
    if not settings.pg_dsn:
        raise SnapshotMissing("no MNEMIQ_PG_DSN -- the engine needs a source to query")

    client = LLMClient(settings)
    agent = Agent(
        generator=LLMGenerator(client),
        synthesizer=LLMSynthesizer(client),
        adapter=DuckDBPostgresAdapter(settings.pg_dsn),
        cache=TwoTierCache(L1Cache()),
        budget=Budget(wall_clock_s=120.0),
    )
    return Runtime(
        con=con,
        snapshot=snapshot,
        adapter=agent.adapter,
        agent=agent,
        embedder=LLMEmbedder(settings),
        authz=_authz(settings),
        settings=settings,
    )

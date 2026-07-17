from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.agent.loop import Agent, AgentAnswer
from mnemiq.agent.modes import DEFAULT_MODE, MODES, build_agent
from mnemiq.agent.route import Router, StaticRouter, UnknownMode
from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.authz.grants import AuthzProvider, DenyAll, FileAuthzProvider
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext, Snapshot
from mnemiq.execute.select import LLMSelector
from mnemiq.generate.correct import LLMCorrector
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import Embedder, LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.values import ValueIndex
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
    agents: dict[str, Agent] | None = None  # one per mode; None = single-agent (tests)
    router: Router = field(default_factory=StaticRouter)

    def ask(
        self, question: str, identity: IdentityContext, mode: str | None = None
    ) -> AgentAnswer:
        # Route first: an unknown mode fails before any retrieval or LLM work.
        name = self.router.route(question, mode)
        agent = (self.agents or {}).get(name, self.agent)
        packet = retrieve(self.con, question, identity, self.authz, self.embedder, k=6)
        grants = self.authz.grants_for(identity)
        answer = agent.answer(packet, self.snapshot, grants, identity)
        answer.mode = name
        return answer

    def schema(self, identity: IdentityContext) -> list[dict]:
        """The tables/cards this identity may see -- access-scoped, so metadata never leaks."""
        grants = self.authz.grants_for(identity)
        rows = self.con.execute("SELECT object_id, card FROM semantic_object").fetchall()
        allowed = grants.objects
        return [{"object_id": oid, "card": card} for oid, card in rows if oid in allowed]


def _authz(settings: Settings) -> AuthzProvider:
    return FileAuthzProvider(settings.authz_path) if settings.authz_path else DenyAll()


def build_runtime(settings: Settings) -> Runtime:
    default_mode = settings.default_mode or DEFAULT_MODE
    if default_mode not in MODES:
        raise UnknownMode(
            f"MNEMIQ_MODE={default_mode!r} names no mode; valid modes: {sorted(MODES)}"
        )
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

    # Shared components, built once; each mode is a thin Agent over the same instances.
    client = LLMClient(settings)
    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    # Generate in the dialect the source executes (duckdb here); keeps generation, parsing,
    # and execution on one dialect so no cross-dialect transpile gap can bite.
    generator = LLMGenerator(client, dialect=adapter.dialect)
    synthesizer = LLMSynthesizer(client)
    cache = TwoTierCache(L1Cache())
    corrector = LLMCorrector(client)
    values = ValueIndex(con)
    selector = LLMSelector(client)
    agents = {
        name: build_agent(
            mode,
            generator=generator,
            synthesizer=synthesizer,
            adapter=adapter,
            cache=cache,
            corrector=corrector,
            values=values,
            selector=selector,
        )
        for name, mode in MODES.items()
    }
    return Runtime(
        con=con,
        snapshot=snapshot,
        adapter=adapter,
        agent=agents[default_mode],
        embedder=LLMEmbedder(settings),
        authz=_authz(settings),
        settings=settings,
        agents=agents,
        router=StaticRouter(default=default_mode),
    )

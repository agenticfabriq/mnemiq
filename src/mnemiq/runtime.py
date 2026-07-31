from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.agent.loop import Agent, AgentAnswer
from mnemiq.agent.modes import DEFAULT_MODE, MODES, build_agent
from mnemiq.agent.route import Router, StaticRouter, UnknownMode
from mnemiq.assembly import build_components
from mnemiq.authz.grants import AuthzProvider, DenyAll, FileAuthzProvider
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext, Snapshot
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import Embedder, LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.ontology_index import OntologyIndex
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy, build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import ApprovedWrite
from mnemiq.store.bootstrap import init_store
from mnemiq.store.control import resolve_version
from mnemiq.store.snapshot_store import load_snapshot
from mnemiq.verify.judge import SemanticJudge
from mnemiq.verify.verifier import Verifier


class SnapshotMissing(RuntimeError):
    """No enriched snapshot in the store -- the source has not been indexed yet."""


@dataclass
class WriteResult:
    approved: bool
    target: str | None = None
    rows_affected: int | None = None
    refusal: str | None = None
    plan_sql: str | None = None
    target_sql: str | None = None


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
    loaded_versions: dict[str, str] = field(default_factory=dict)
    sink: Any = None  # observability sink; None = NullSink (no record written)
    ontology: Any = None  # OntologyIndex reader; None = no question-time code resolution

    def reload_if_stale(self) -> None:
        """Hot-swap the in-memory snapshot when the shared version pointer has advanced to a
        version this replica already has locally. No-op without a control DSN."""
        if not self.settings or not self.settings.control_dsn:
            return
        snapshot, versions = load_current_snapshot(self.settings, self.con)
        if versions != self.loaded_versions:
            self.snapshot = snapshot
            self.loaded_versions = versions

    def ask(
        self, question: str, identity: IdentityContext, mode: str | None = None
    ) -> AgentAnswer:
        self.reload_if_stale()
        started = time.perf_counter()
        # Route first: an unknown mode fails before any retrieval or LLM work.
        name = self.router.route(question, mode)
        agent = (self.agents or {}).get(name, self.agent)
        packet = retrieve(
            self.con, question, identity, self.authz, self.embedder,
            k=self.settings.retrieval_k if self.settings else 12,
            table_facts=self.snapshot.table_facts if self.snapshot else (),
            # Definitions ride with the snapshot: the glossary channel had no runtime producer
            # until the ontology digest, so this stayed unfed from Plan 08 until SP1.
            definitions=self.snapshot.definitions if self.snapshot else (),
            columns=self.snapshot.columns if self.snapshot else (),
            ontology_index=self.ontology,
        )
        grants = self.authz.grants_for(identity)
        answer = agent.answer(packet, self.snapshot, grants, identity)
        answer.mode = name
        if self.sink is not None:  # observability loop: one fail-soft record per answer
            from mnemiq.observability.metrics import AnswerRecord

            self.sink.record(
                self.settings.source_id if self.settings else "unknown",
                AnswerRecord(deferred=answer.deferred, cached=answer.cached,
                             total_ms=(time.perf_counter() - started) * 1000, mode=name,
                             failed=answer.failed, reason_code=answer.reason_code),
            )
        return answer

    def schema(self, identity: IdentityContext) -> list[dict]:
        """The tables/cards this identity may see -- access-scoped, so metadata never leaks."""
        grants = self.authz.grants_for(identity)
        rows = self.con.execute("SELECT object_id, card FROM semantic_object").fetchall()
        allowed = grants.objects
        return [{"object_id": oid, "card": card} for oid, card in rows if oid in allowed]

    def write(self, sql: str, identity: IdentityContext) -> WriteResult:
        """Governed write: the caller supplies SQL; the decider screens it, then it executes
        only if a write grant exists AND the source is attached read-write. Two locks."""
        grants = self.authz.grants_for(identity)
        visible = visible_schema(self.snapshot, grants) if self.snapshot else {}
        policy = build_access_policy(self.snapshot, grants) if self.snapshot else AccessPolicy()
        dialect = getattr(self.adapter, "dialect", "duckdb")
        verdict = decide_write(sql, visible, grants, adapter=self.adapter, dialect=dialect,
                               policy=policy)
        if not isinstance(verdict, ApprovedWrite):
            return WriteResult(approved=False, refusal=verdict.message, target=verdict.subject)
        try:
            result = self.adapter.execute(verdict.target_sql)
        except Exception as exc:  # the read-only attach backstop rejects the write here
            return WriteResult(approved=False, refusal=f"the source rejected the write: {exc}",
                               target=verdict.target, plan_sql=verdict.plan_sql,
                               target_sql=verdict.target_sql)
        rows = (result[0][0] if result and len(result[0]) == 1
                and isinstance(result[0][0], int) else None)
        return WriteResult(approved=True, target=verdict.target, rows_affected=rows,
                           plan_sql=verdict.plan_sql, target_sql=verdict.target_sql)


def _authz(settings: Settings) -> AuthzProvider:
    return FileAuthzProvider(settings.authz_path) if settings.authz_path else DenyAll()


def _resolve_verify_level(mode_verify: str, override: str | None) -> str:
    """Apply the MNEMIQ_VERIFY override to a mode's default verify level.
    "0" forces off (byte-for-byte escape hatch); "1" forces full; unset keeps the mode default."""
    if override == "0":
        return "off"
    if override == "1":
        return "full"
    return mode_verify


def _build_verifier(level: str, *, threshold: float, grounding: bool, judge):
    """None when off; else a Verifier with sanity always on and the judge only at 'full'."""
    if level == "off":
        return None
    return Verifier(threshold=threshold, sanity=True, grounding=grounding,
                    judge=judge if level == "full" else None)


def _build_mode_verifiers(settings: Settings, *, client):
    """Return ({mode_name: Verifier|None}, judge_build_count). The judge is built at most once
    (self-judge on the local model, unless MNEMIQ_VERIFY_* points elsewhere) and shared across
    every mode that needs it."""
    override = settings.verify_override
    levels = {name: _resolve_verify_level(m.verify, override) for name, m in MODES.items()}
    judge = None
    judge_calls = 0
    if any(lvl == "full" for lvl in levels.values()):
        base, key = settings.verify_endpoint()
        if settings.verify_base_url or settings.verify_model or settings.verify_api_key:
            judge_client = LLMClient(settings.model_copy(update={
                "llm_base_url": base, "llm_api_key": key,
                "llm_model": settings.verify_model or settings.llm_model}))
        else:
            judge_client = client  # self-judge: reuse the generation client
        judge = SemanticJudge(judge_client)
        judge_calls = 1
    verifiers = {
        name: _build_verifier(levels[name], threshold=settings.verify_threshold,
                              grounding=settings.verify_grounding, judge=judge)
        for name in MODES
    }
    return verifiers, judge_calls


def load_current_snapshot(settings: Settings, con) -> tuple[Snapshot, dict[str, str]]:
    """Resolve the current version per source (shared pointer when a control DSN is set, else
    local) and load the snapshot -- merged across sources when federated."""
    specs = settings.source_specs()
    versions: dict[str, str] = {}
    if len(specs) > 1:
        from mnemiq.semantic.federation import merge_snapshots

        pairs = []
        for spec in specs:
            v = resolve_version(con, settings.control_dsn, spec.id)
            if v is None:
                raise SnapshotMissing(
                    f"no snapshot for source {spec.id!r} -- run `mnemiq enrich` then "
                    "`mnemiq build` first"
                )
            versions[spec.id] = v
            pairs.append((spec, load_snapshot(con, v)))
        return merge_snapshots(pairs), versions
    sid = settings.source_id
    v = resolve_version(con, settings.control_dsn, sid)
    if v is None:
        raise SnapshotMissing(
            f"no snapshot for source {settings.source_id!r} in {settings.store_path!r} -- "
            "run `mnemiq enrich` then `mnemiq build` first"
        )
    versions[sid] = v
    return load_snapshot(con, v), versions


def build_runtime(settings: Settings) -> Runtime:
    default_mode = settings.default_mode or DEFAULT_MODE
    if default_mode not in MODES:
        raise UnknownMode(
            f"MNEMIQ_MODE={default_mode!r} names no mode; valid modes: {sorted(MODES)}"
        )
    con = init_store(settings.store_path)
    # Current version per source (shared pointer when a control DSN is set, else local) + snapshot.
    snapshot, loaded_versions = load_current_snapshot(settings, con)
    specs = settings.source_specs()
    if len(specs) > 1:
        # Federated: ATTACH all sources into one DuckDB; the snapshot is the qualified union,
        # carrying the catalog->schema registry the decider expands with.
        from mnemiq.adapters.federated import FederatedAdapter

        adapter = FederatedAdapter(specs, read_only=not settings.write_enabled)
    else:
        # Single-source fast path -- unchanged from v0.1.
        if not settings.pg_dsn:
            raise SnapshotMissing("no MNEMIQ_PG_DSN -- the engine needs a source to query")
        # Read-only attach unless writes are explicitly enabled -- the backstop under db_write.
        adapter = DuckDBPostgresAdapter(settings.pg_dsn, read_only=not settings.write_enabled)

    # Shared components, built once; each mode is a thin Agent over the same instances.
    kit = build_components(settings, adapter, con)
    # Live L2 (cross-replica) + observability sink when a control Postgres is configured;
    # L1-only + no-op sink otherwise (today, byte-for-byte).
    if settings.control_dsn:
        from mnemiq.cache.postgres import PostgresCache
        from mnemiq.observability.metrics import PostgresSink

        cache = TwoTierCache(L1Cache(), PostgresCache(settings.control_dsn))
        sink = PostgresSink(settings.control_dsn)
    else:
        from mnemiq.observability.metrics import NullSink

        cache = TwoTierCache(L1Cache())
        sink = NullSink()
    # Built at enrich time and persisted, exactly like the value index -- so ask-time needs the
    # store, not the records file. An unindexed store simply resolves nothing.
    ontology = OntologyIndex(con)
    # Per-mode result verifier: sanity (free) in every mode, judge in `deep`; the judge is
    # built once and shared. MNEMIQ_VERIFY=0 forces off (byte-for-byte), =1 forces full.
    verifiers, _ = _build_mode_verifiers(settings, client=kit.client)
    agents = {
        name: build_agent(
            mode,
            generator=kit.generator,
            synthesizer=kit.synthesizer,
            adapter=adapter,
            cache=cache,
            corrector=kit.corrector,
            values=kit.values,
            selector=kit.selector,
            verifier=verifiers[name],
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
        loaded_versions=loaded_versions,
        sink=sink,
        ontology=ontology,
    )

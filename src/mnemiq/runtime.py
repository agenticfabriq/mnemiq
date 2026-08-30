from __future__ import annotations

import logging

import time
from dataclasses import dataclass, field
from typing import Any

from mnemiq.agent.history import scope_history
from mnemiq.agent.loop import Agent, AgentAnswer
from mnemiq.agent.modes import DEFAULT_MODE, MODES, build_agent
from mnemiq.agent.route import Router, StaticRouter, UnknownMode
from mnemiq.assembly import build_components
from mnemiq.authz.grants import AuthzProvider, DenyAll, FileAuthzProvider
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import DEFAULT_RETRIEVAL_K, Settings
from mnemiq.contract import HistoryTurn, IdentityContext, Snapshot
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import Embedder, LLMEmbedder
from mnemiq.progress import Emit, Stage, step
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.ontology_index import OntologyIndex
from mnemiq.sql.decide_write import decide_write
from mnemiq.semantic.cards import build_cards
from mnemiq.semantic.starters import compose_starters
from mnemiq.sql.views import inventory_for
from mnemiq.sql.policy import AccessPolicy, build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import ApprovedWrite
from mnemiq.store.bootstrap import init_store
from mnemiq.store.control import resolve_version
from mnemiq.store.snapshot_store import load_snapshot
from mnemiq.verify.judge import SemanticJudge
from mnemiq.verify.verifier import Verifier


logger = logging.getLogger(__name__)


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
    # A SECOND seam beside `sink`, not a widening of it. The metrics sink is a counter that wants
    # six scalars fast; this one batches, crosses a network to another product and may retry.
    # None = emit nothing, and the engine is byte-for-byte unchanged.
    trace_sink: Any = None
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
        self,
        question: str,
        identity: IdentityContext,
        mode: str | None = None,
        emit: Emit | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> AgentAnswer:
        self.reload_if_stale()
        started = time.perf_counter()
        # Route first: an unknown mode fails before any retrieval or LLM work.
        name = self.router.route(question, mode)
        agent = (self.agents or {}).get(name, self.agent)
        with step(emit, Stage.RETRIEVE, mode=name):
            packet = retrieve(
                self.con, question, identity, self.authz, self.embedder,
                # The default lives on the Settings field, not here. It was written out as a
                # literal 12 and silently kept the old value when the measured default moved
                # to 24 -- a second copy of a number is a second thing to forget.
                k=self.settings.retrieval_k if self.settings else DEFAULT_RETRIEVAL_K,
                table_facts=self.snapshot.table_facts if self.snapshot else (),
                # Definitions ride with the snapshot: the glossary channel had no runtime
                # producer until the ontology digest, so this stayed unfed from Plan 08 until SP1.
                definitions=self.snapshot.definitions if self.snapshot else (),
                # Certified metrics and dimensions were appended to the snapshot by
                # `apply_certified` and passed to nothing -- five of the fs corpus's twenty-eight
                # records reaching the model not at all, and the five carrying the most explicit
                # meaning.
                metrics=self.snapshot.metrics if self.snapshot else (),
                dimensions=self.snapshot.dimensions if self.snapshot else (),
                columns=self.snapshot.columns if self.snapshot else (),
                ontology_index=self.ontology,
                # M4: lets retrieve re-render each card against this identity's column policy.
                snapshot=self.snapshot,
            )
        grants = self.authz.grants_for(identity)
        # Scoped against THIS identity's boundary, never the one that produced the turn.
        packet.history = scope_history(history, grants.fingerprint)
        answer = agent.answer(packet, self.snapshot, grants, identity, emit=emit)
        answer.mode = name
        answer.grant_fingerprint = grants.fingerprint
        if self.trace_sink is not None:
            # One governed trace per answer. Fail-soft by the same contract as the metrics sink --
            # a Verity outage cannot stop mnemiq answering -- and the payload is built from the
            # tier table inside the emitter, never from this event, so a field nobody classified
            # cannot ship.
            from mnemiq.observability.trace_sink import AnswerEvent

            event = AnswerEvent(
                source_id=self.settings.source_id if self.settings else "unknown",
                question=packet.question,
                identity=identity,
                answer=answer,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                packet=packet,
            )
            # What the SNAPSHOT holds; the emitter intersects it with what the PACKET selected, so
            # the trace names what the answer USED rather than what was available to it.
            event.certified_refs = self.snapshot.certified_refs if self.snapshot else []
            try:
                self.trace_sink.record_answer(event)
            except Exception as exc:  # never into the caller's answer
                logger.warning("trace emit failed; answer unaffected: %s", exc)

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
        """The tables and cards this identity may see, scoped by TABLE and by COLUMN."""
        return self._cards(self.authz.grants_for(identity))

    def scope(self, identity: IdentityContext) -> dict:
        """Everything an opening screen needs: the cards in scope, and the questions to offer.

        Both from **one** grant resolution. Two would let the panel and the starters answer to
        different versions of the policy file within a single request -- the divergence M13
        describes, which is cheap to avoid here and awkward to detect later.
        """
        grants = self.authz.grants_for(identity)
        starters: list[str] = []
        if self.snapshot is not None:
            starters = compose_starters(
                self.snapshot, grants, build_access_policy(self.snapshot, grants)
            )
        return {"tables": self._cards(grants), "starters": starters}

    def _cards(self, grants) -> list[dict]:
        """M4: this filtered on `oid in grants.objects` alone -- table level -- while promising
        that metadata never leaks. The stored card is rendered once at enrich time from the whole
        snapshot, so a granted table's card listed every column including denied ones, with their
        pii_level and their harvested coded values. The card is re-rendered per identity now; the
        stored one stays identity-independent because it is what gets embedded and indexed.
        """
        rows = self.con.execute("SELECT object_id, card FROM semantic_object").fetchall()
        allowed = grants.objects
        visible = [(oid, card) for oid, card in rows if oid in allowed]
        if self.snapshot is None:
            return [{"object_id": oid, "card": card} for oid, card in visible]
        scoped = {
            card.object_id: card.text
            for card in build_cards(self.snapshot, policy=build_access_policy(self.snapshot, grants))
        }
        return [
            {"object_id": oid, "card": scoped.get(oid, card)} for oid, card in visible
        ]

    def write(self, sql: str, identity: IdentityContext) -> WriteResult:
        """Governed write: the caller supplies SQL; the decider screens it, then it executes
        only if a write grant exists AND the source is attached read-write. Two locks."""
        grants = self.authz.grants_for(identity)
        visible = visible_schema(self.snapshot, grants) if self.snapshot else {}
        policy = build_access_policy(self.snapshot, grants) if self.snapshot else AccessPolicy()
        dialect = getattr(self.adapter, "dialect", "duckdb")
        # The same map the read path builds in `plan_query`. Without it the write decider's view
        # floor is a parameter nobody passes -- the whole control, satisfied in tests and absent
        # in production.
        views = inventory_for(self.snapshot)
        verdict = decide_write(sql, visible, grants, adapter=self.adapter, dialect=dialect,
                               policy=policy, views=views,
                               # M3: the deployment switch reaches the decider, so a disabled
                               # deployment refuses in our vocabulary instead of letting the
                               # read-only attachment raise and calling that a refusal.
                               writes_enabled=bool(
                                   self.settings and self.settings.write_enabled))
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


def _warn_policy_advisories(authz: AuthzProvider, snapshot: Snapshot | None) -> None:
    """Say, at boot, the two ways a policy silently grants less than its author meant:

    a row filter that fails to reach every table hanging off its tenancy axis, and a
    `pii_clearance`/`pii_mask` value that names no PII level and so clears nothing (M40).
    Both faithfully apply what the policy said, so nothing downstream complains. Advisory:
    it reports, it never refuses. The operator's policy is the operator's.
    """
    from mnemiq.authz.coverage import warn_unfiltered_dependents, warn_unknown_pii_levels
    from mnemiq.contract import IdentityContext

    roles = getattr(authz, "policy_roles", None)
    if roles is None:
        return  # a provider that cannot enumerate roles is not a provider with no holes
    for role in roles():
        grants = authz.grants_for(
            IdentityContext(tenant_id="boot", principal_id="boot", roles=[role])
        )
        # A clearance value outside the PII vocabulary is inert whatever the snapshot holds,
        # so it is warned about even when no snapshot is available to check filter coverage.
        warn_unknown_pii_levels(grants, role=role)
        if snapshot is not None:
            warn_unfiltered_dependents(grants, snapshot.relationships, role=role)


def _acknowledged(settings: Settings) -> frozenset[str]:
    """`MNEMIQ_ACK_ADVISORIES` parsed into `{advisory}:{verdict}` keys, lowercased."""
    raw = (settings.ack_advisories or "").strip()
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def _warn_source_enforcement(adapter, acknowledged: frozenset[str] = frozenset()) -> None:
    """Say, at boot, whether the SOURCE is actually enforcing row security -- when it can tell.

    `OracleAdapter.assert_enforcing` was written, tested against a live instance across four
    verdicts, and **called by nothing outside its own tests**. That is the shape **M26** is about:
    a seam wired at one end, with a green suite, reporting nothing in production. Found by an
    adversarial review of the lane that built it, not by the lane.

    **It warns; it does not refuse**, and that is a decision rather than caution. Under the
    2026-08-29 direction the ENGINE still enforces RLS/CLS -- delegation to the database is v2 --
    so a `bypassing` connection today is one where mnemiq's own filters are still in force and
    refusing to boot would take a working deployment down over a control that is not yet load
    bearing. **When delegation lands, this must become fail-closed**, which is exactly what
    **M57** is open for; the value of wiring it now is that the verdict is visible from the first
    Oracle deployment rather than from the one after the trust boundary moves.

    It reports on two questions, both optional and both skipped by an adapter that cannot answer:
    whether the SOURCE is enforcing row security, and whether `read_only` rests on this adapter's
    statement gate alone. The second exists because that gate provably cannot see a write reached
    through an `AUTONOMOUS_TRANSACTION` function, and a view can hide the call so the statement
    text names nothing -- so the reachable question is whether the connection could write at all.

    Any adapter that grows either method is picked up here without further wiring, which is the
    property whose absence produced the finding.
    """
    # The level tracks whether the operator can DO anything, and BOTH of this function's earlier
    # answers to that were wrong in opposite directions.
    #
    # It first warned on every verdict, so a correctly configured read-only deployment printed a
    # warning at every boot -- noise, and a review said so. I then demoted `assert_read_only`'s
    # `unverifiable` to INFO on the premise that "nothing further exists to do". **That premise is
    # false, and demoting it suppressed a real gap**: default logging is WARNING, so the operator
    # of exactly the deployment we recommend was told NOTHING, while a principal holding SELECT on
    # one view can still cause a write. Measured both ways.
    #
    # There IS an action, and it is the one the 2026-08-29 direction already names: stop treating
    # `read_only=True` as a guarantee and enforce read-only in the DATABASE. So the verdict warns,
    # and the message says that rather than describing a state. The cost is one line per process
    # start, which is the right price for "this safety property does not hold"; the earlier noise
    # complaint is answered by making the warning true and actionable, not by silencing it.
    advisories = (("assert_enforcing", "source enforcement", {"attached"}),
                  ("assert_read_only", "read-only basis", {"writable"}))
    # A typo'd acknowledgement silently does nothing and looks exactly like no acknowledgement --
    # the operator keeps getting the warning and has no way to tell which. Two different causes
    # share one observable, which is the collapse this codebase keeps closing, here in the
    # mechanism added to answer a review. The two halves are reported differently because only one
    # is definitely a mistake: an unknown ADVISORY name cannot be right, while an acknowledged
    # verdict that simply did not occur this boot is the normal case for a deployment whose state
    # improved, and warning about it would recreate the noise this setting exists to remove.
    known = {label.replace(" ", "-").lower() for _, label, _ in advisories}
    for entry in sorted(acknowledged):
        if entry.split(":", 1)[0] not in known:
            logger.warning("MNEMIQ_ACK_ADVISORIES: %r names no advisory; known: %s. "
                           "It acknowledges nothing", entry, ", ".join(sorted(known)))
    matched: set[str] = set()
    assessed: set[str] = set()

    for name, label, quiet in advisories:
        assess = getattr(adapter, name, None)
        if assess is None:
            continue  # this adapter cannot answer; only OracleAdapter implements either today
        try:
            verdict, detail = assess()
        except Exception as exc:  # a source that will not answer must not stop the engine booting
            logger.warning("could not assess %s: %s", label, exc)
            continue
        # Recorded only on SUCCESS. Set before the call, an advisory that RAISED counted as
        # assessed, so an unmatched ack for it was diagnosed "the verdict changed" while the real
        # cause -- the exception -- was logged two lines above it.
        assessed.add(label.replace(" ", "-").lower())
        # An operator who has assessed a gap and accepted it can acknowledge THAT VERDICT, by
        # `<advisory>:<verdict>` -- not the advisory as a whole. Acknowledging the advisory would
        # silence a WORSE verdict arriving later: `read-only basis` moving from `unverifiable`
        # (the unclosable ceiling) to `gate_only` (a principal that now holds write privilege) is
        # exactly the change worth paging on, and a coarser switch would hide the regression along
        # with the accepted state.
        key = f"{label.replace(' ', '-')}:{verdict}".lower()
        if verdict in quiet:
            logger.info("%s: %s -- %s", label, verdict, detail)
        elif key in acknowledged:
            matched.add(key)
            logger.info("%s: %s (acknowledged via MNEMIQ_ACK_ADVISORIES) -- %s",
                        label, verdict, detail)
        else:
            logger.warning("%s: %s -- %s", label, verdict, detail)

    # Three reasons an acknowledgement goes unused, and the message names the right one. The third
    # is the one a review caught: only `OracleAdapter` implements either method, so an ack carried
    # from an Oracle deployment to a Postgres one is neither stale nor misspelled -- the check
    # never ran. Telling that operator "the verdict changed" sends them to look at a verdict that
    # was never produced.
    for entry in sorted(acknowledged - matched):
        advisory = entry.split(":", 1)[0]
        if advisory not in known:
            continue  # already warned about above, as an advisory that does not exist
        if advisory not in assessed:
            logger.info("MNEMIQ_ACK_ADVISORIES: %r did not apply -- this source produced no "
                        "verdict for %r (not implemented here, or the assessment failed above), "
                        "so nothing was acknowledged", entry, advisory)
        else:
            logger.info("MNEMIQ_ACK_ADVISORIES: %r did not apply this boot -- either the verdict "
                        "changed, or the verdict half is misspelled", entry)


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
    # specs[0].id, not settings.source_id. Every multi-source branch above already keys on
    # spec.id; the single-source ones keyed on settings.source_id, which defaults to "acme" and is
    # never reconciled with a manifest's id. `enrich` writes the snapshot under the spec's id, so
    # a one-entry manifest naming anything else stored a snapshot that build, refresh and ask then
    # looked for under "acme" and never found -- "run `mnemiq enrich` first", forever, after a
    # successful enrich. Without a manifest the two are the same value, so this is a no-op there.
    sid = specs[0].id
    v = resolve_version(con, settings.control_dsn, sid)
    if v is None:
        raise SnapshotMissing(
            f"no snapshot for source {sid!r} in {settings.store_path!r} -- "
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
        # Single source: dispatch on the SPEC, not on pg_dsn. This branch used to read
        # `settings.pg_dsn` directly, which meant a one-entry manifest was resolved into a spec
        # and then thrown away -- two sources were honoured and one was not. `adapter_for` is
        # shared with the enrich and refresh commands so the three doors cannot drift apart.
        # Read-only attach unless writes are explicitly enabled -- the backstop under db_write.
        from mnemiq.adapters.resolve import SourceUnconfigured, UnknownSourceKind, adapter_for

        try:
            adapter = adapter_for(specs[0], settings, read_only=not settings.write_enabled)
        except (SourceUnconfigured, UnknownSourceKind) as exc:
            # Preserves the established contract: a source the engine cannot connect to raises
            # SnapshotMissing, now with a message that names what is actually missing. The
            # unknown-kind case is translated too because `mnemiq ask` and `mnemiq write` catch
            # SnapshotMissing and print it -- a typo in a manifest should be a sentence, not a
            # traceback, on every door.
            raise SnapshotMissing(str(exc)) from exc

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
    # M26: the emitter shipped with NO CALLER. `Runtime.trace_sink` existed, the sink existed, its
    # tests passed, and nothing here ever constructed one -- so every answer in production emitted
    # nothing while a full suite stayed green. That is the seventh instance in two days of a seam
    # wired at one end, committed in the slice that was about that species. It is also why this
    # branch exists rather than a docstring saying the emitter is "available".
    trace_sink = None
    if settings.verity_traces_url:
        from mnemiq.observability.trace_sink import VerityTraceSink

        trace_sink = VerityTraceSink(settings)
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
    _boot_authz = _authz(settings)
    _warn_policy_advisories(_boot_authz, snapshot)
    _warn_source_enforcement(adapter, _acknowledged(settings))
    return Runtime(
        con=con,
        snapshot=snapshot,
        adapter=adapter,
        agent=agents[default_mode],
        embedder=LLMEmbedder(settings),
        authz=_boot_authz,
        settings=settings,
        agents=agents,
        router=StaticRouter(default=default_mode),
        loaded_versions=loaded_versions,
        sink=sink,
        trace_sink=trace_sink,
        ontology=ontology,
    )

from __future__ import annotations

from dataclasses import dataclass

from mnemiq.agent.budget import Budget
from mnemiq.agent.synthesize import Synthesizer
from mnemiq.agent.trace import build_trace
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.keys import cache_key
from mnemiq.cache.store import Cache, from_ipc, to_ipc
from mnemiq.contract import DeferralReason, IdentityContext, Snapshot, Trace
from mnemiq.progress import Emit, Stage, step
from mnemiq.execute.render import render_result
from mnemiq.execute.resultset import cluster
from mnemiq.execute.runner import ExecutionError, run
from mnemiq.execute.select import ClusterView, auto_accepted, majority_index
from mnemiq.generate.generator import Generator, StrategyGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved

# Cycled across candidates in multi-candidate mode: engineered disagreement, so the
# selector has something real to select over (uniform resampling measured 56% unanimous).
STRATEGIES = ("direct", "decompose", "skeleton")


@dataclass(frozen=True)
class ResultPreview:
    """A bounded slice of the executed result, for display -- never re-executed."""

    columns: list[str]
    rows: list[list[object]]
    row_count: int  # true count, not the capped length
    truncated: bool


def result_preview(table, cap: int) -> ResultPreview:
    cols = [str(c) for c in table.column_names]
    raw = table.slice(0, cap).to_pylist()
    return ResultPreview(columns=cols, rows=[[r[c] for c in cols] for r in raw],
                         row_count=table.num_rows, truncated=table.num_rows > cap)


@dataclass
class AgentAnswer:
    answer: str
    trace: Trace | None = None
    deferred: bool = False
    # A deferral is a decision we made; a failure is something that happened to us. Collapsing
    # them let a source outage raise the deferral rate -- the one number the product claim rests
    # on -- and the eval harness graded it "safe: gave up on an answerable question" (M6).
    failed: bool = False
    reason_code: DeferralReason | None = None
    cached: bool = False
    agreement: float | None = None
    judge_engaged: bool | None = None  # multi-candidate only: did the judge get consulted?
    judge_override: bool | None = None  # ...and did it pick against the majority?
    candidates_executed: int | None = None  # multi-candidate only: how many of N ran
    mode: str | None = None  # resolved mode name, stamped by the Runtime (the Agent IS a mode)
    preview: ResultPreview | None = None  # None on every deferral path -- never fabricated


def _shape(row_count: int, column_count: int) -> str:
    return "scalar" if row_count == 1 and column_count == 1 else "table"


class Agent:
    """Budgeted, self-terminating, hand-rolled.

    It ends in an answer, a deferral, or a forced answer -- there is no fourth branch where
    it spins.
    """

    def __init__(
        self,
        generator: Generator,
        synthesizer: Synthesizer,
        adapter,
        cache: Cache,
        budget: Budget | None = None,
        timeout_s: float = 30.0,
        candidates: int = 1,
        corrector=None,
        values=None,
        selector=None,
        min_agreement: float | None = None,
        verifier=None,
        preview_rows: int = 100,
    ) -> None:
        self.generator = generator
        self.synthesizer = synthesizer
        self.adapter = adapter
        self.cache = cache
        self.budget = budget or Budget()
        self.timeout_s = timeout_s
        self.candidates = candidates
        self.corrector = corrector
        self.values = values
        self.selector = selector
        self.min_agreement = min_agreement
        self.verifier = verifier
        self.preview_rows = preview_rows
        # The source's SQL dialect: the model writes it, `decide` parses it, the source runs
        # it. Product adapter is duckdb (transpile is a no-op); a SQLite source is SQLite
        # end-to-end -- no cross-dialect transpile gap (SQLGlot can't map DuckDB YEAR()/
        # EXTRACT to SQLite strftime). The generator is constructed with the same dialect.
        self.dialect = getattr(adapter, "dialect", "duckdb")

    def answer(
        self,
        packet: ContextPacket,
        snapshot: Snapshot,
        grants: GrantSet,
        identity: IdentityContext,
        emit: Emit | None = None,
    ) -> AgentAnswer:
        deadline = self.budget.started()
        if self.candidates <= 1:
            return self._answer_single(packet, snapshot, grants, identity, deadline, emit)
        return self._answer_consistent(packet, snapshot, grants, identity, deadline, emit)

    def _answer_single(
        self,
        packet: ContextPacket,
        snapshot: Snapshot,
        grants: GrantSet,
        identity: IdentityContext,
        deadline,
        emit: Emit | None = None,
    ) -> AgentAnswer:
        feedback: str | None = None
        failure: str | None = None

        for _attempt in range(self.budget.max_attempts):
            # Two nested repair loops, answering two different judges. plan_query repairs what the
            # *decider* rejects (a star, a bad column) -- never disable that by capping it to
            # one attempt. This outer loop repairs what the *database* rejects, which is a
            # thing no amount of static analysis could have known.
            with step(emit, Stage.PLAN, attempt=_attempt + 1, of=self.budget.max_attempts):
                outcome = plan_query(
                    packet,
                    snapshot,
                    grants,
                    self.generator,
                    adapter=self.adapter,
                    dialect=self.dialect,
                    target=self.dialect,
                    feedback=feedback,
                    corrector=self.corrector,
                    values=self.values,
                )
            if isinstance(outcome, Deferred):
                return AgentAnswer(answer=outcome.reason, deferred=True,
                                   reason_code=outcome.code)

            approved: Approved = outcome
            key = cache_key(approved.plan_sql, grants.fingerprint, packet.enrichment_version)

            hit = self.cache.get(key)
            if hit is not None:
                table = from_ipc(hit)
                with step(emit, Stage.VERIFY, cached=True):
                    blocked = self._verified(packet, approved, table)
                if blocked is not None:
                    return blocked
                return self._synthesize(
                    packet, approved, identity, table, deadline, cached=True, forced=False,
                    emit=emit,
                )

            try:
                with step(emit, Stage.EXECUTE):
                    result = run(self.adapter, approved.target_sql, timeout_s=self.timeout_s)
            except ExecutionError as exc:
                # The database is the one authority that cannot be wrong about itself: its
                # complaint is the cheapest accuracy lever we have. Feed it back and retry.
                failure = str(exc)
                feedback = failure
                continue

            self.cache.put(key, to_ipc(result.table))
            with step(emit, Stage.VERIFY):
                blocked = self._verified(packet, approved, result.table)
            if blocked is not None:
                return blocked
            return self._synthesize(
                packet,
                approved,
                identity,
                result.table,
                deadline,
                cached=False,
                forced=deadline.expired,
                execute_ms=result.elapsed_ms,
                emit=emit,
            )

        # NOT a deferral. We did not decline to answer -- the source refused to serve us, and
        # counting that as abstention is what let an outage look like the engine working (M6).
        return AgentAnswer(
            answer=(
                "Could not answer this question: the database rejected every attempt. "
                f"Last error: {failure}"
            ),
            failed=True,
            reason_code=DeferralReason.EXECUTION_FAILED,
        )

    def _execute(self, approved: Approved, grants: GrantSet, packet: ContextPacket):
        """Cache-or-run one plan; None on execution error (this candidate just drops out)."""
        key = cache_key(approved.plan_sql, grants.fingerprint, packet.enrichment_version)
        hit = self.cache.get(key)
        if hit is not None:
            return from_ipc(hit)
        try:
            result = run(self.adapter, approved.target_sql, timeout_s=self.timeout_s)
        except ExecutionError:
            return None
        self.cache.put(key, to_ipc(result.table))
        return result.table

    def _answer_consistent(
        self,
        packet: ContextPacket,
        snapshot: Snapshot,
        grants: GrantSet,
        identity: IdentityContext,
        deadline,
        emit: Emit | None = None,
    ) -> AgentAnswer:
        # Generate N candidates across engineered strategies and let execution vote. Each
        # is decided independently; a refusal/deferral just drops that candidate.
        executed: list[tuple[Approved, object]] = []
        for i in range(self.candidates):
            with step(emit, Stage.CANDIDATE, index=i + 1, of=self.candidates):
                outcome = plan_query(
                    packet,
                    snapshot,
                    grants,
                    StrategyGenerator(self.generator, STRATEGIES[i % len(STRATEGIES)]),
                    adapter=self.adapter,
                    dialect=self.dialect,
                    target=self.dialect,
                    max_attempts=1,
                    corrector=self.corrector,
                    values=self.values,
                )
                if not isinstance(outcome, Approved):
                    continue
                table = self._execute(outcome, grants, packet)
            if table is not None:
                executed.append((outcome, table))

        if not executed:
            # no candidate ran -> fall back to the single repairing path (today's floor)
            return self._answer_single(packet, snapshot, grants, identity, deadline, emit)

        groups = cluster([table for _, table in executed])
        views = [
            ClusterView(
                sql=executed[group[0]][0].plan_sql,
                preview=render_result(executed[group[0]][1], max_rows=5),
                size=len(group),
            )
            for group in groups
        ]
        majority = majority_index(views)

        if self.min_agreement is not None:
            # Selective answering: fragmentation is a confidence verdict on the QUESTION,
            # judged before any pick. A reduced set never passes -- unanimity among
            # survivors is survival bias, not agreement (Plan 12: 4/4, 3/3 -> 11%, 0%).
            largest = views[majority].size
            if len(executed) < self.candidates or largest / len(executed) < self.min_agreement:
                return AgentAnswer(
                    answer=(
                        "The candidates disagreed too much to answer confidently "
                        f"(best agreement {largest} of {self.candidates})."
                    ),
                    deferred=True,
                    reason_code=DeferralReason.DISAGREEMENT,
                    agreement=largest / len(executed),
                    candidates_executed=len(executed),
                )

        judge_engaged = judge_override = None
        if self.selector is not None:
            judge_engaged = not auto_accepted(views)
            chosen = self.selector.select(packet.question, views) if judge_engaged else majority
            judge_override = chosen != majority
        else:
            chosen = majority  # no selector wired: Plan 12's vote, byte-for-byte

        winner = groups[chosen]
        approved, table = executed[winner[0]]
        agreement = len(winner) / len(executed)

        with step(emit, Stage.VERIFY):
            blocked = self._verified(packet, approved, table)
        if blocked is not None:
            return blocked

        base = self._synthesize(
            packet, approved, identity, table, deadline, cached=False,
            forced=deadline.expired, emit=emit,
        )
        caution = "" if agreement >= 0.5 else " -- treat with caution"
        note = f" (confidence: {len(winner)}/{len(executed)} candidates agreed{caution}.)"
        return AgentAnswer(
            answer=base.answer + note,
            trace=base.trace,
            deferred=False,
            cached=False,
            agreement=agreement,
            judge_engaged=judge_engaged,
            judge_override=judge_override,
            candidates_executed=len(executed),
            preview=base.preview,
        )

    def _verified(self, packet: ContextPacket, approved: Approved, table) -> AgentAnswer | None:
        """None if the answer clears the verifier (or none is wired), else a deferral. This is the
        correctness gate: the decider guards validity, the verifier guards likely-correctness, and
        both end in the same defer-don't-guess path."""
        if self.verifier is None:
            return None
        verdict = self.verifier.verify(packet, approved, table)
        if verdict.defer:
            return AgentAnswer(answer=verdict.reason, deferred=True,
                               reason_code=DeferralReason.VERIFICATION)
        return None

    def _synthesize(
        self,
        packet: ContextPacket,
        approved: Approved,
        identity: IdentityContext,
        table,
        deadline,
        cached: bool,
        forced: bool,
        execute_ms: float = 0.0,
        emit: Emit | None = None,
    ) -> AgentAnswer:
        with step(emit, Stage.SYNTHESIZE, forced=forced):
            answer = self.synthesizer.answer(
                packet.question, approved.plan_sql, render_result(table), forced=forced
            )
        trace = build_trace(
            question=packet.question,
            approved=approved,
            identity=identity,
            enrichment_version=packet.enrichment_version,
            timing={"execute_ms": execute_ms, "total_ms": deadline.elapsed_ms},
            result_shape=_shape(table.num_rows, table.num_columns),
        )
        return AgentAnswer(answer=answer, trace=trace, deferred=False, cached=cached,
                           preview=result_preview(table, self.preview_rows))

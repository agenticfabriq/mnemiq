from __future__ import annotations

from dataclasses import dataclass

from mnemiq.agent.budget import Budget
from mnemiq.agent.synthesize import Synthesizer
from mnemiq.agent.trace import build_trace
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.keys import cache_key
from mnemiq.cache.store import Cache, from_ipc, to_ipc
from mnemiq.contract import IdentityContext, Snapshot, Trace
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


@dataclass
class AgentAnswer:
    answer: str
    trace: Trace | None = None
    deferred: bool = False
    cached: bool = False
    agreement: float | None = None
    judge_engaged: bool | None = None  # multi-candidate only: did the judge get consulted?
    judge_override: bool | None = None  # ...and did it pick against the majority?
    candidates_executed: int | None = None  # multi-candidate only: how many of N ran
    mode: str | None = None  # resolved mode name, stamped by the Runtime (the Agent IS a mode)


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
        # Transpile the plan (written in duckdb) to whatever the source actually runs.
        # Default duckdb: the product adapter executes via DuckDB, so it is a no-op there;
        # a SQLite source gets SQLite instead of failing on un-transpiled DuckDB.
        self.target_dialect = getattr(adapter, "dialect", "duckdb")

    def answer(
        self,
        packet: ContextPacket,
        snapshot: Snapshot,
        grants: GrantSet,
        identity: IdentityContext,
    ) -> AgentAnswer:
        deadline = self.budget.started()
        if self.candidates <= 1:
            return self._answer_single(packet, snapshot, grants, identity, deadline)
        return self._answer_consistent(packet, snapshot, grants, identity, deadline)

    def _answer_single(
        self,
        packet: ContextPacket,
        snapshot: Snapshot,
        grants: GrantSet,
        identity: IdentityContext,
        deadline,
    ) -> AgentAnswer:
        feedback: str | None = None
        failure: str | None = None

        for _attempt in range(self.budget.max_attempts):
            # Two nested repair loops, for two different oracles. plan_query repairs what the
            # *decider* rejects (a star, a bad column) -- never disable that by capping it to
            # one attempt. This outer loop repairs what the *database* rejects, which is a
            # thing no amount of static analysis could have known.
            outcome = plan_query(
                packet,
                snapshot,
                grants,
                self.generator,
                adapter=self.adapter,
                target=self.target_dialect,
                feedback=feedback,
                corrector=self.corrector,
                values=self.values,
            )
            if isinstance(outcome, Deferred):
                return AgentAnswer(answer=outcome.reason, deferred=True)

            approved: Approved = outcome
            key = cache_key(approved.plan_sql, grants.fingerprint, packet.enrichment_version)

            hit = self.cache.get(key)
            if hit is not None:
                return self._synthesize(
                    packet, approved, identity, from_ipc(hit), deadline, cached=True, forced=False
                )

            try:
                result = run(self.adapter, approved.target_sql, timeout_s=self.timeout_s)
            except ExecutionError as exc:
                # The database is the one authority that cannot be wrong about itself: its
                # complaint is the cheapest accuracy lever we have. Feed it back and retry.
                failure = str(exc)
                feedback = failure
                continue

            self.cache.put(key, to_ipc(result.table))
            return self._synthesize(
                packet,
                approved,
                identity,
                result.table,
                deadline,
                cached=False,
                forced=deadline.expired,
                execute_ms=result.elapsed_ms,
            )

        return AgentAnswer(
            answer=(
                "Could not answer this question: the database rejected every attempt. "
                f"Last error: {failure}"
            ),
            deferred=True,
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
    ) -> AgentAnswer:
        # Generate N candidates across engineered strategies and let execution vote. Each
        # is decided independently; a refusal/deferral just drops that candidate.
        executed: list[tuple[Approved, object]] = []
        for i in range(self.candidates):
            outcome = plan_query(
                packet,
                snapshot,
                grants,
                StrategyGenerator(self.generator, STRATEGIES[i % len(STRATEGIES)]),
                adapter=self.adapter,
                target=self.target_dialect,
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
            return self._answer_single(packet, snapshot, grants, identity, deadline)

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

        base = self._synthesize(
            packet, approved, identity, table, deadline, cached=False, forced=deadline.expired
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
        )

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
    ) -> AgentAnswer:
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
        return AgentAnswer(answer=answer, trace=trace, deferred=False, cached=cached)

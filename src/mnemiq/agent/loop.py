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
from mnemiq.generate.generator import Generator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved


@dataclass
class AgentAnswer:
    answer: str
    trace: Trace | None = None
    deferred: bool = False
    cached: bool = False
    agreement: float | None = None


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
    ) -> None:
        self.generator = generator
        self.synthesizer = synthesizer
        self.adapter = adapter
        self.cache = cache
        self.budget = budget or Budget()
        self.timeout_s = timeout_s
        self.candidates = candidates

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
            # Two nested repair loops, answering two different judges. plan_query repairs what the
            # *decider* rejects (a star, a bad column) -- never disable that by capping it to
            # one attempt. This outer loop repairs what the *database* rejects, which is a
            # thing no amount of static analysis could have known.
            outcome = plan_query(
                packet,
                snapshot,
                grants,
                self.generator,
                adapter=self.adapter,
                target="duckdb",
                feedback=feedback,
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
        # Generate N single-shot candidates and let the deterministic executor vote. Each
        # is decided independently; a refusal/deferral just drops that candidate.
        executed: list[tuple[Approved, object]] = []
        for _ in range(self.candidates):
            outcome = plan_query(
                packet,
                snapshot,
                grants,
                self.generator,
                adapter=self.adapter,
                target="duckdb",
                max_attempts=1,
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
        winner = max(groups, key=len)  # ties -> first (earliest-appearance) cluster
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

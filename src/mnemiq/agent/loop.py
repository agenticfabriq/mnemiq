from __future__ import annotations

import logging
from dataclasses import dataclass

from mnemiq.agent.budget import Budget
from mnemiq.agent.synthesize import Synthesizer
from mnemiq.agent.trace import build_trace
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.keys import cache_key
from mnemiq.cache.store import Cache, from_ipc, to_ipc
from mnemiq.contract.seams import disclosure_sentence
from mnemiq.contract import DeferralReason, IdentityContext, Snapshot, Trace
from mnemiq.llm.client import ModelUnavailable
from mnemiq.progress import Emit, Stage, step
from mnemiq.execute.render import render_result
from mnemiq.execute.resultset import cluster
from mnemiq.execute.runner import ExecutionError, run
from mnemiq.execute.select import (
    ClusterView,
    SelectorRead,
    auto_accepted,
    fallback_reason,
    majority_index,
    trust,
)
from mnemiq.generate.generator import Generator, StrategyGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved

logger = logging.getLogger(__name__)

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


def _narrowed_of(executed):
    """The access decision behind a multi-candidate deferral.

    Every candidate was governed by the same grants against the same objects, so the narrowing is
    a property of the request rather than of whichever candidate won -- and none of them won here,
    which is why this path exists. Taking the first EXECUTED candidate reports the decision that
    actually reached the source; `None` when nothing executed, because then nothing was decided
    against a real statement.
    """
    for approved, _table in executed:
        found = getattr(approved, "narrowed", None)
        if found is not None:
            return found
    return None


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
    # ...and did it ANSWER. `judge_engaged` says only that the clusters disagreed enough to ask,
    # and `LLMSelector` fails closed to the majority on an outage, an unreadable reply or a pick
    # outside the clusters -- so without this, a dead selector and a judge that studied the
    # clusters and agreed with the majority are the same two booleans on the wire and in the audit
    # record (M11). `None` = no judgement was attempted, which is not the same as one that held.
    # Non-None exactly when `judge_engaged` is True, and a selector speaking only the int protocol
    # is taken at its word rather than assumed broken.
    judge_fell_back: bool | None = None
    # ...and which way it broke, when it did. An outage, a model that cannot emit the format and
    # a pick naming a cluster that does not exist want three different responses, and the audit
    # record is where that is asked after the fact by someone who cannot re-run the request.
    #
    # `None` whenever there is no fallback to explain -- INCLUDING a successful judgement, which
    # is not a fallback and whose "ok" would make an operator's `IS NOT NULL` count every judged
    # answer as a failure. Otherwise a member of `FALLBACK_REASONS` **or `UNRECOGNISED_REASON`**,
    # which is deliberately not in that frozenset: enumerating the vocabulary from the set alone
    # misses the one value that means a selector this build has not been taught. `fallback_reason`
    # reads it off the pick `trust` has already narrowed, so nothing downstream carries a
    # third party's free text into the audit record's always tier. Deliberately NOT on the wire: a client acts
    # on whether the answer was judged, not on how the judge broke.
    judge_fallback_reason: str | None = None
    # Multi-candidate only: how many of N produced a TABLE. Not how many were attempted --
    # a candidate that deferred, or that ran and hit an ExecutionError, is dropped by _execute and
    # never counted. The looser "how many of N ran" left that ambiguous at the definition site.
    candidates_executed: int | None = None
    # What the mode actually spent. `instant` and `thinking` differ only in the corrector and
    # the retry ceiling, and both are invisible on a question that succeeds first time -- so the
    # control read as inert when it was working exactly as designed (M33). `attempts` counts the
    # OUTER loop, which repairs what the database rejected; `corrected` is the inner surgical
    # pass, which repairs what the decider rejected. Two different judges, two different numbers.
    attempts: int | None = None
    # What the access decision narrowed, carried on the ANSWER and not only via `trace`.
    #
    # The trace is built after execution -- it needs timing and result shape -- so a deferral or a
    # provider failure that happens AFTER the decision returns an answer with no trace at all. The
    # audit sink read that as "governance was never evaluated" and recorded it on queries that were
    # governed and had already run against the source. A false audit fact is worse than a missing
    # one, so the fact rides here from the moment the decider produces it.
    narrowed: list | None = None
    corrected: bool | None = None
    mode: str | None = None  # resolved mode name, stamped by the Runtime (the Agent IS a mode)
    preview: ResultPreview | None = None  # None on every deferral path -- never fabricated
    # The authorization boundary this answer was computed under. A client echoes it back
    # with the turn; the engine replays a turn only to the same boundary (history.py).
    grant_fingerprint: str = ""
    # The verifier's read, kept on the PASS path as well as the defer path. Without the
    # passes there is no way to ask the only question that matters about a judge -- does it
    # score wrong answers below right ones -- so a threshold can only be tuned by running the
    # whole suite again. Measured once with these dropped: caught 4, killed 4, missed 37.
    verify_confidence: float | None = None
    # EVERY value here is mapped on the wire, by `verified_state` in `server/serialize.py` --
    # a new layer added below is not inert there, it ships as `unknown`. `judge_unavailable`
    # means the judge was configured and could not be reached, so the score is the fail-open
    # constant and not a judgement. `None` is not "no verifier": a deferral or an execution
    # failure also leaves it unset, because the verifier never saw a table.
    verify_layer: str | None = None  # sanity | grounding | judge | judge_unavailable | pass


def _stamp(answer: AgentAnswer, verdict) -> AgentAnswer:
    """Carry the verifier's read onto an answer that PASSED it.

    A confidence recorded only when the verifier refuses is a sample of one tail. Ranking is
    the only property of a judge that matters -- does it score wrong answers below right ones
    -- and that is unanswerable without the scores it gave to the answers it let through.
    """
    if verdict is not None:
        answer.verify_confidence = verdict.confidence
        answer.verify_layer = verdict.layer
    return answer


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
        guard_undefined_terms: bool = False,
    ) -> None:
        self.generator = generator
        self.guard_undefined_terms = guard_undefined_terms
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
        try:
            if self.candidates <= 1:
                return self._answer_single(packet, snapshot, grants, identity, deadline, emit)
            return self._answer_consistent(packet, snapshot, grants, identity, deadline, emit)
        except ModelUnavailable as exc:
            # Symmetry with the source-outage path: something happened TO us, so it is a
            # stated failure the caller can act on, never a deferral and never a traceback
            # out of the transport. `failed` is what keeps it out of the deferral rate (M6).
            #
            # The provider's own words ride along. This message used to assert an outage
            # and drop the cause, so a misconfigured model name, a context-length refusal
            # and a dead endpoint were indistinguishable in a results file -- three very
            # different problems wearing one sentence. Diagnosing a 30-case failure took
            # twenty minutes for want of the string that was already in hand.
            detail = str(exc).strip()
            return AgentAnswer(
                answer=(
                    "Could not answer this question: the model provider did not respond. "
                    "This is an outage, not a judgement about your data -- try again."
                    + (f" The provider said: {detail[:300]}" if detail else "")
                ),
                failed=True,
                reason_code=DeferralReason.MODEL_UNAVAILABLE,
            )

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
        failure: ExecutionError | None = None

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
                    guard_undefined_terms=self.guard_undefined_terms,
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
                    blocked, verdict = self._verified(packet, approved, table)
                if blocked is not None:
                    return blocked
                return _stamp(self._synthesize(
                    packet, approved, identity, table, deadline, cached=True, forced=False,
                    emit=emit, attempts=_attempt + 1,
                ), verdict)

            try:
                with step(emit, Stage.EXECUTE):
                    result = run(self.adapter, approved.target_sql, timeout_s=self.timeout_s)
            except ExecutionError as exc:
                # The database is the one authority that cannot be wrong about itself: its
                # complaint is the cheapest accuracy lever we have. Feed it back and retry.
                # Every attempt, not just the last: `failure` is overwritten each time round,
                # so an error that a later attempt repaired would otherwise be recorded nowhere.
                logger.warning("execution attempt %d failed; the source said: %s",
                               _attempt + 1, exc.repair_text)
                failure = exc
                feedback = exc.repair_text
                continue

            self.cache.put(key, to_ipc(result.table))
            with step(emit, Stage.VERIFY):
                blocked, verdict = self._verified(packet, approved, result.table)
            if blocked is not None:
                return blocked
            return _stamp(self._synthesize(
                packet,
                approved,
                identity,
                result.table,
                deadline,
                cached=False,
                forced=deadline.expired,
                execute_ms=result.elapsed_ms,
                emit=emit,
                attempts=_attempt + 1,
            ), verdict)

        # NOT a deferral. We did not decline to answer -- the source refused to serve us, and
        # counting that as abstention is what let an outage look like the engine working (M6).
        #
        # Our half of the failure goes to the caller; the source's half does not. A rejection
        # in the database's own words is made of the caller's schema -- it names the table and
        # the column and often quotes the statement -- so an identity denied a table would
        # learn the table exists by being told why it could not read it -- the side-channel
        # disclosure SECURITY.md names, and it arrives through the ANSWER, which means
        # `/v1/ask` carried it as readily as the stream's error frame.
        #
        # Not by dropping the text, though: `message` is the sentence this engine wrote, and
        # it is often the useful one -- a timeout says to ask something cheaper. Only
        # `source_detail` is withheld, and each attempt logs its own as it happens, so the
        # log holds the whole retry rather than just whichever error came last.
        logger.warning(
            "every execution attempt failed; last source error: %s",
            failure.repair_text if failure is not None else "(no attempt was made)",
        )
        return AgentAnswer(
            answer=(
                "Could not answer this question: the database rejected every attempt. "
                + (failure.message if failure is not None else "No attempt was made.")
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
        # Generate N candidates across engineered strategies and let execution vote. Each is
        # decided independently; a refusal or deferral just drops that candidate -- EXCEPT an
        # undefined-term deferral, which is a finding about the question rather than about the
        # candidate, and is carried out below rather than outvoted.
        executed: list[tuple[Approved, object]] = []
        undefined: Deferred | None = None
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
                    guard_undefined_terms=self.guard_undefined_terms,
                )
                if not isinstance(outcome, Approved):
                    # M35: an UNDEFINED_TERM deferral is a finding about the QUESTION, not a
                    # failed attempt by this candidate, so it must not be dropped and outvoted.
                    #
                    # The asymmetry is the point. A candidate declaring "lifetime value" is
                    # evidence the question names an undefined term; the others NOT declaring it
                    # is not evidence against -- they simply did not say. Letting three silent
                    # candidates outvote two that spoke would restore the exact failure the guard
                    # exists for, and deep mode is where it would land, because deep mode is what
                    # a caller reaches for on the hard questions.
                    if getattr(outcome, "code", None) == DeferralReason.UNDEFINED_TERM:
                        undefined = outcome
                    continue
                table = self._execute(outcome, grants, packet)
            if table is not None:
                executed.append((outcome, table))

        if undefined is not None:
            # Before the vote and before the fallback: one candidate that could not ground a term
            # settles the question, however many produced runnable SQL from a guessed meaning.
            # `candidates_executed` like the DISAGREEMENT deferral below, and it means what that
            # field means -- how many produced a table, so on the M35 shape it reports 3 of 5 and
            # not 5. Without it the workbench drops the candidate chip and the harness records
            # nothing for a deep-mode turn that spent the full budget.
            return AgentAnswer(answer=undefined.reason, deferred=True,
                               reason_code=undefined.code,
                               candidates_executed=len(executed),
                               narrowed=_narrowed_of(executed))

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
                    narrowed=_narrowed_of(executed),
                )

        judge_engaged = judge_override = judge_fell_back = judge_fallback_reason = None
        if self.selector is not None:
            judge_engaged = not auto_accepted(views)
            if not judge_engaged:
                chosen = majority
            else:
                # `read` when the selector offers it, because the fact must come back WITH the
                # pick -- the fallback returns the majority index, which is also what a judgement
                # agreeing with the majority returns. A selector speaking only `select` (any
                # stub, `FakeSelector`, `MajoritySelector`) is taken at its word, the same rule
                # the verifier applies to a judge without `read`: absence of the richer protocol
                # is not evidence of a failure.
                reader = getattr(self.selector, "read", None)
                if reader is not None:
                    got = trust(reader(packet.question, views), views)
                else:
                    # Taken at its word about whether it JUDGED -- absence of `read` is not
                    # evidence of a failure -- and not taken at its word that the word is an
                    # index. Same narrowing, because it is the same untrusted object.
                    got = trust(SelectorRead(self.selector.select(packet.question, views),
                                             fell_back=False), views)
                chosen, judge_fell_back = got.choice, got.fell_back
                judge_fallback_reason = fallback_reason(got)
            judge_override = chosen != majority
        else:
            chosen = majority  # no selector wired: Plan 12's vote, byte-for-byte

        winner = groups[chosen]
        approved, table = executed[winner[0]]
        agreement = len(winner) / len(executed)

        with step(emit, Stage.VERIFY):
            blocked, verdict = self._verified(packet, approved, table)
        if blocked is not None:
            return blocked

        base = _stamp(self._synthesize(
            packet, approved, identity, table, deadline, cached=False,
            forced=deadline.expired, emit=emit,
        ), verdict)
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
            judge_fell_back=judge_fell_back,
            judge_fallback_reason=judge_fallback_reason,
            candidates_executed=len(executed),
            preview=base.preview,
            # Carried from `base`: this branch rebuilds the answer to append the agreement
            # note, and a field added to AgentAnswer is silently dropped here unless copied.
            verify_confidence=base.verify_confidence,
            verify_layer=base.verify_layer,
        )

    def _verified(self, packet: ContextPacket, approved: Approved, table):
        """The verifier's read, or None when none is wired. Returns (deferral, verdict).

        The correctness gate: the decider guards validity, the verifier guards
        likely-correctness, and both end in the same defer-don't-guess path. The verdict comes
        back even when it passes, because a confidence recorded only on refusals cannot answer
        whether the judge ranks wrong answers below right ones.
        """
        if self.verifier is None:
            return None, None
        verdict = self.verifier.verify(packet, approved, table)
        if verdict.defer:
            return AgentAnswer(answer=verdict.reason, deferred=True,
                               reason_code=DeferralReason.VERIFICATION,
                               verify_confidence=verdict.confidence,
                               verify_layer=verdict.layer,
                               # The query RAN before the verifier saw it, so the decision is a
                               # fact about this attempt whatever the verifier then decided.
                               narrowed=getattr(approved, "narrowed", None)), verdict
        return None, verdict

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
        attempts: int | None = None,
    ) -> AgentAnswer:
        with step(emit, Stage.SYNTHESIZE, forced=forced):
            answer = self.synthesizer.answer(
                packet.question, approved.plan_sql, render_result(table), forced=forced,
                row_count=table.num_rows,
            )
        trace = build_trace(
            question=packet.question,
            approved=approved,
            identity=identity,
            enrichment_version=packet.enrichment_version,
            timing={"execute_ms": execute_ms, "total_ms": deadline.elapsed_ms},
            result_shape=_shape(table.num_rows, table.num_columns),
        )
        # Appended HERE, once, rather than rendered by each surface. The four surfaces were fixed
        # one at a time before -- the comment in `cli.py` records the commit messages saying "the
        # two surfaces", then three -- and a disclosure that four places must remember to print is
        # a disclosure that will be missing from one of them. Every surface prints `answer`.
        #
        # After the synthesizer and never through it: a governed fact must not pass a stochastic
        # step that can soften it, drop it, or attach it to the wrong object.
        disclosure = disclosure_sentence(trace.narrowed)
        if disclosure:
            answer = f"{answer}\n\n{disclosure}"
        return AgentAnswer(answer=answer, trace=trace, deferred=False, cached=cached,
                           narrowed=getattr(approved, "narrowed", None),
                           preview=result_preview(table, self.preview_rows),
                           attempts=attempts, corrected=approved.corrected)

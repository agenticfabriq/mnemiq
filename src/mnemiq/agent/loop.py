from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from mnemiq.agent.budget import Budget
from mnemiq.agent.synthesize import Synthesizer
from mnemiq.agent.trace import build_trace
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.keys import cache_key
from mnemiq.cache.store import Cache, from_ipc, to_ipc
from mnemiq.config import DEFAULT_MAX_QUESTION_CHARS
from mnemiq.contract.seams import (
    append_notes,
    disclosure_sentence,
    lineage_disclosure_sentence,
)
from mnemiq.contract import DeferralReason, IdentityContext, Snapshot, Trace
from mnemiq.llm.client import ContextTooLong, ModelUnavailable, PromptCut, seed_offset
from mnemiq.progress import Emit, Stage, result_ready, step
from mnemiq.redact import policy_safe_detail
from mnemiq.execute.render import render_result
from mnemiq.execute.resultset import cluster
from mnemiq.execute.runner import (
    DEFAULT_TIMEOUT_S,
    ExecutionError,
    QueryCancelledError,
    SourceUnavailableError,
    run,
)
from mnemiq.execute.select import (
    ClusterView,
    SelectorRead,
    auto_accepted,
    fallback_reason,
    majority_index,
    trust,
)
from mnemiq.generate.generator import Generator, StrategyGenerator, temperature_floor
from mnemiq.generate.plan_query import Deferred, Feedback, plan_query
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.sql.values_check import confirm_by_source, source_holds
from mnemiq.verify.sanity import answered_note, grounding_decides, sanity_check

logger = logging.getLogger(__name__)

# The most the probes of filter values may take in one turn, together (`Agent._confirmed`): an
# equality on a column the source has no index for can scan its table, and a probe not asked or
# not answered only keeps its value unchecked, the empty result declined.
PROBE_BUDGET_S = 10.0

# Cycled across candidates in multi-candidate mode: engineered disagreement, so the
# selector has something real to select over (uniform resampling measured 56% unanimous).
STRATEGIES = ("direct", "decompose", "skeleton")

# The least temperature sent to a candidate whose strategy's prompt an earlier candidate already
# sent, which only a mode set to more than len(STRATEGIES) candidates has (deep's fourth and fifth
# of five): at a writer temperature of 0 it would return that candidate's SQL again, and the
# agreement gate would count the copy. Every other candidate, every mode with one candidate, and
# the fallback to the single path are sent the writer's temperature as set. 0.7 is the server
# default that `writer_temperature`'s description names for Qwen2.5. Engine code, not a setting or
# a mode field, so no profile version hashes it, as none hashes `seed_offset`.
REPEAT_TEMPERATURE_FLOOR: float = 0.7


@dataclass(frozen=True)
class ResultPreview:
    """A bounded slice of the executed result, for display -- never re-executed."""

    columns: list[str]
    rows: list[list[object]]
    row_count: int  # rows the query returned; a floor, not the total, when `capped`
    truncated: bool  # the preview shows fewer rows than the query returned
    capped: bool = False  # the query stopped at the guard's row limit and may have more (M118)


def result_preview(table, cap: int, capped: bool = False) -> ResultPreview:
    cols = [str(c) for c in table.column_names]
    raw = table.slice(0, cap).to_pylist()
    return ResultPreview(columns=cols, rows=[[r[c] for c in cols] for r in raw],
                         row_count=table.num_rows, truncated=table.num_rows > cap, capped=capped)


def cap_sentence(row_cap: int) -> str:
    """Said after the model, like the other disclosures: a cut result must not pass a step that
    could present 1,000 rows as the whole answer."""
    return (f"These results stop at the {row_cap:,}-row limit, so there may be more: counts and "
            "totals over them are not the whole.")


def _cut(approved, table) -> int | None:
    """The guard's limit, when the result filled it -- the one case where rows may be missing."""
    cap = getattr(approved, "row_cap", None)
    return cap if cap is not None and table.num_rows >= cap else None


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
    # On an answer the model provider failed, the writer's model or the judge's, which way it failed
    # (`ModelUnavailable.kind`): a server that stopped answering (`connect`, `status_503`) reads apart from one
    # question's refusal (`status_400`, `context_too_long`) without parsing the provider's words. None on every other
    # answer, a judge reply that could not be read included.
    model_error: str | None = None
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
    # The rows were verified and stand, and the prose call that should have put them into a sentence failed. Not
    # `failed`: the answer is the rows. Flagged so a prose model that is down never reads as prose switched off.
    prose_failed: bool = False
    # What the mode actually spent. `instant` and `thinking` differ only in the corrector and
    # the retry ceiling, and both are invisible on a question that succeeds first time -- so the
    # control read as inert when it was working exactly as designed (M33). `attempts` counts the
    # OUTER loop, which repairs what the database rejected; `corrected` is the inner surgical
    # pass, which repairs what the decider rejected. Two different judges, two different numbers.
    # An empty result tried again (`empty_retries`) is one more round of the outer loop.
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
    # How many ranges the decider moved to end before the day after a bare date (`Approved.day_bounds`).
    day_bounds_closed: int | None = None
    mode: str | None = None  # resolved mode name, stamped by the Runtime (the Agent IS a mode)
    # What produced this answer, stamped by the Runtime. The version pins the settings and the prompts, and the commit the
    # rest of the engine: card rendering and the decider are in neither alone.
    profile: str | None = None
    profile_version: str | None = None
    model: str | None = None
    engine_commit: str | None = None
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
    # means the judge was configured and did not answer, so `verify_confidence` beside it is
    # None: there was no judgement, and no float says that. `None` on the LAYER is not "no
    # verifier" either -- a deferral or an execution failure also leaves it unset, because the
    # verifier never saw a table.
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
        timeout_s: float = DEFAULT_TIMEOUT_S,  # the server-side bound's value too (M152)
        candidates: int = 1,
        corrector=None,
        values=None,
        selector=None,
        min_agreement: float | None = None,
        verifier=None,
        preview_rows: int = 100,
        guard_undefined_terms: bool = False,
        guard_fanout: bool | None = False,
        max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS,
        plan_attempts: int = 3,
        prose: bool = True,
        empty_retries: int = 0,
        retry_answered_empties: bool = False,
    ) -> None:
        self.generator = generator
        self.plan_attempts = plan_attempts
        self.prose = prose
        self.empty_retries = empty_retries
        self.retry_answered_empties = retry_answered_empties
        # Defaults to the shipped limit, not to none: a construction site that forgets it still declines what no
        # stage would read alike (M147).
        self.max_question_chars = max_question_chars
        self.guard_undefined_terms = guard_undefined_terms
        self.guard_fanout = guard_fanout
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

    def decline_if_too_long(self, question: str) -> AgentAnswer | None:
        """The decline for a question too long to read whole, or None. Every stage reads the question whole, so
        a longer one is declined, never cut (M147). Both doors call this before retrieval, which embeds the
        question; `answer` calls it again for a caller that skipped it."""
        if len(question) <= self.max_question_chars:
            return None
        return AgentAnswer(
            answer=(f"Your question is {len(question):,} characters long; the limit is "
                    f"{self.max_question_chars:,}. Shorten it, or ask an operator to raise "
                    "MNEMIQ_MAX_QUESTION_CHARS. It was not cut: a cut question is answered as a "
                    "different question."),
            deferred=True,
            reason_code=DeferralReason.QUESTION_TOO_LONG,
        )

    def answer(
        self,
        packet: ContextPacket,
        snapshot: Snapshot,
        grants: GrantSet,
        identity: IdentityContext,
        emit: Emit | None = None,
    ) -> AgentAnswer:
        if (declined := self.decline_if_too_long(packet.question)) is not None:
            return declined
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
            if isinstance(exc, PromptCut):
                # Not an outage, and "try again" would send the same prompt to the same window.
                return AgentAnswer(
                    answer=("Could not answer this question: the model server cut the prompt to fit "
                            "its context window, so the model never saw all of it. This is a "
                            f"configuration problem, not a judgement about your data. {detail}"),
                    failed=True,
                    reason_code=DeferralReason.MODEL_UNAVAILABLE,
                    model_error=exc.kind,
                )
            if isinstance(exc, ContextTooLong):
                # Refused rather than cut: the same configuration problem, told the same way.
                return AgentAnswer(
                    answer=("Could not answer this question: the model server refused the prompt "
                            "as longer than its context window. This is a configuration problem, "
                            "not a judgement about your data -- raise the server's window (vLLM: "
                            "--max-model-len) up to the model's own limit; past it, send less: "
                            "fewer tables a question (MNEMIQ_RETRIEVAL_K) when many fill the "
                            "prompt, or fewer in the source manifest's `tables` list. Where the "
                            "server reports no window, set MNEMIQ_LLM_CONTEXT_WINDOW so mnemiq can "
                            f"trim column descriptions to fit. {detail[:300]}"),
                    failed=True,
                    reason_code=DeferralReason.MODEL_UNAVAILABLE,
                    model_error=exc.kind,
                )
            return AgentAnswer(
                answer=(
                    "Could not answer this question: the model provider did not respond. "
                    "This is an outage, not a judgement about your data -- try again."
                    + (f" The provider said: {detail[:300]}" if detail else "")
                ),
                failed=True,
                reason_code=DeferralReason.MODEL_UNAVAILABLE,
                model_error=exc.kind,
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
        # `Feedback` carries its own provenance: whether the SOURCE wrote any of it. `plan_query`
        # needs that because the model is shown the feedback and its own words reach the caller,
        # so the two together form a path back out for whatever the source said.
        feedback: Feedback | None = None
        held: dict = {"spent_s": 0.0}  # what the source said about a filter value, asked once a turn
        failure: ExecutionError | None = None
        # An empty result tried again (`empty_retries`): each retry adds one try to the budget, so
        # `max_attempts` still counts what the database rejects. `standing` is the result before
        # the retry, kept to answer with if the retry produces nothing to answer with.
        empties_left = self.empty_retries
        standing: tuple | None = None
        # Each statement's result for the rest of the turn, so a retry that writes it again reads it
        # here rather than running it twice when the shared cache has evicted it (Codex's review).
        ran: dict[str, object] = {}
        tries, _attempt = self.budget.max_attempts, 0

        while _attempt < tries:
            _attempt += 1
            # Two nested repair loops, answering two different judges. plan_query repairs what the
            # *decider* rejects (a star, a bad column) -- never disable that by capping it to
            # one attempt. This outer loop repairs what the *database* rejects, which is a
            # thing no amount of static analysis could have known.
            with step(emit, Stage.PLAN, attempt=_attempt, of=tries):
                outcome = plan_query(
                    packet,
                    snapshot,
                    grants,
                    self.generator,
                    adapter=self.adapter,
                    dialect=self.dialect,
                    target=self.dialect,
                    max_attempts=self.plan_attempts,
                    feedback=feedback,
                    corrector=self.corrector,
                    values=self.values,
                    guard_undefined_terms=self.guard_undefined_terms,
                    guard_fanout=self.guard_fanout,
                )
            if isinstance(outcome, Deferred):
                # A source that could not be asked happened TO us, so it is a stated failure and
                # stays out of the deferral rate, like a model or a judge that did not answer (M6).
                outage = outcome.code is DeferralReason.SOURCE_UNAVAILABLE
                if standing is not None and outcome.code is DeferralReason.INVALID_QUERY:
                    # The retry wrote nothing the decider would run, so what it retried stands. Any
                    # other deferral says something about the question or what this identity may
                    # read -- an undefined term (M35, as deep's candidates read one), a table not
                    # granted, a writer that says the data cannot answer -- and stands over it
                    # (Codex's review).
                    return self._settled(packet, identity, deadline, emit, _attempt, *standing)
                return AgentAnswer(answer=outcome.reason, deferred=not outage, failed=outage,
                                   reason_code=outcome.code)

            approved: Approved = outcome
            key = cache_key(approved.plan_sql, grants.fingerprint, packet.enrichment_version)

            hit = ran.get(key)
            if hit is None:
                stored = self.cache.get(key)
                hit = None if stored is None else from_ipc(stored)
            if hit is not None:
                table, cached, execute_ms = hit, True, 0.0
            else:
                try:
                    with step(emit, Stage.EXECUTE):
                        result = run(self.adapter, approved.target_sql, timeout_s=self.timeout_s)
                except SourceUnavailableError as exc:
                    # Not the statement's failure, so not fed back: the model would rewrite a correct
                    # query because the network failed (M188). A failure, not a deferral (M6), as the
                    # proof seam reports the same fault before the run. Not run again either: a cancel
                    # or a statement timeout worded in another language, and an administrator's kill,
                    # read as outages too, and a second run would repeat what someone stopped (Codex's
                    # review of #179).
                    logger.warning("the source did not answer while a query ran: %s", exc.full_text)
                    return AgentAnswer(answer=exc.message, failed=True,
                                       reason_code=DeferralReason.SOURCE_UNAVAILABLE)
                except QueryCancelledError as exc:
                    # Someone stopped the statement, so the turn ends here: repaired, a model writing
                    # the same SQL again would run it a second time (Codex's second review of #179).
                    logger.warning("the source cancelled a query while it ran: %s", exc.full_text)
                    return AgentAnswer(answer=exc.message, failed=True,
                                       reason_code=DeferralReason.EXECUTION_FAILED)
                except ExecutionError as exc:
                    # The database is the one authority that cannot be wrong about itself: its
                    # complaint is the cheapest accuracy lever we have. Feed it back and retry.
                    # Every attempt, not just the last: `failure` is overwritten each time round,
                    # so an error that a later attempt repaired would otherwise be recorded nowhere.
                    logger.warning("execution attempt %d failed; the source said: %s",
                                   _attempt, exc.full_text)
                    if standing is not None:
                        # A retry of an empty result the database rejected: what it retried stands.
                        return self._settled(packet, identity, deadline, emit, _attempt, *standing)
                    failure = exc
                    # The source ran the policy's rewrite, so its words can quote the policy and the
                    # rows it removes: under a policy the model reads the error's class and nothing
                    # else (`policy_safe_detail`). The log above keeps the whole text.
                    detail = policy_safe_detail(exc.source_detail, asked=approved.asked_sql,
                                                executed=approved.target_sql, dialect=self.dialect)
                    shown = ExecutionError(exc.message, source_detail=detail)
                    feedback = Feedback(shown.repair_text, from_source=detail is not None)
                    continue
                self.cache.put(key, to_ipc(result.table))
                table, cached, execute_ms = result.table, False, result.elapsed_ms
            ran[key] = table

            with step(emit, Stage.VERIFY, **({"cached": True} if cached else {})):
                approved = self._confirmed(approved, table, snapshot, held)
                retry = self._empty_retry(packet, approved, table) if empties_left else None
                if retry is None:
                    blocked, verdict = self._verified(packet, approved, table)
            if retry is not None and not deadline.expired:
                # Engine-written: the sentence a caller would be shown, of values the writer wrote.
                empties_left, tries = empties_left - 1, tries + 1
                standing = (approved, table, cached, execute_ms)
                feedback = Feedback(retry, from_source=False)
                continue
            if retry is not None:  # no time left to try again: read as it would be with no retry
                return self._settled(packet, identity, deadline, emit, _attempt, approved, table,
                                     cached, execute_ms)
            if blocked is not None:
                return blocked
            return _stamp(self._synthesize(
                packet, approved, identity, table, deadline, cached=cached,
                forced=False if cached else deadline.expired, execute_ms=execute_ms, emit=emit,
                attempts=_attempt,
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
            failure.full_text if failure is not None else "(no attempt was made)",
        )
        return AgentAnswer(
            answer=(
                "Could not answer this question: the database rejected every attempt. "
                + (failure.message if failure is not None else "No attempt was made.")
            ),
            failed=True,
            reason_code=DeferralReason.EXECUTION_FAILED,
        )

    def _empty_retry(self, packet: ContextPacket, approved: Approved, table) -> str | None:
        """What the writer is told when this result is an empty one this mode tries again, else
        None. Empty is what the sanity layer decides on the grounding (`grounding_decides`): no
        rows, or one NULL row from aggregates over nothing. One the layer would decline is tried
        again, the writer given its sentence; one it would answer, every value checked, only under
        `retry_answered_empties`. Read from the grounding whether or not the layer runs, so a
        profile's retry means the same in every mode."""
        if not grounding_decides(table):
            return None
        ran = ("The query ran and returned no rows" if table.num_rows == 0
               else "The query ran and returned only an empty value")
        again = "If that is the true answer, write the same query again; otherwise "
        declined = sanity_check(packet.question, table, getattr(approved, "value_grounding", None))
        if declined is not None:
            return (f"{ran}. {declined.reason} {again}correct the values the filters use, the "
                    "columns they are on, or the joins.")
        if self.retry_answered_empties:
            return (f"{ran}, though every value it filters on was found in its column. {again}"
                    "check the joins and which column each filter is on.")
        return None

    def _settled(self, packet: ContextPacket, identity: IdentityContext, deadline, emit,
                 attempts: int, approved: Approved, table, cached: bool,
                 execute_ms: float) -> AgentAnswer:
        """The answer for a result kept while an empty retry was tried, once the retry produced
        nothing to answer with: read and answered or declined as it would have been with no retry."""
        with step(emit, Stage.VERIFY, **({"cached": True} if cached else {})):
            blocked, verdict = self._verified(packet, approved, table)
        if blocked is not None:
            return blocked
        return _stamp(self._synthesize(
            packet, approved, identity, table, deadline, cached=cached,
            forced=False if cached else deadline.expired, execute_ms=execute_ms, emit=emit,
            attempts=attempts,
        ), verdict)

    def _execute(self, approved: Approved, grants: GrantSet, packet: ContextPacket):
        """Cache-or-run one plan; None on execution error (this candidate just drops out). A
        source that did not answer, or a statement someone cancelled, raises its own error
        (`SourceUnavailableError`, `QueryCancelledError`), which ends the turn rather than this
        candidate alone (M188)."""
        key = cache_key(approved.plan_sql, grants.fingerprint, packet.enrichment_version)
        hit = self.cache.get(key)
        if hit is not None:
            return from_ipc(hit)
        try:
            result = run(self.adapter, approved.target_sql, timeout_s=self.timeout_s)
        except (SourceUnavailableError, QueryCancelledError):
            raise
        except ExecutionError as exc:
            # This candidate drops out of the vote. Logged for the same reason the single-answer
            # path logs: when three of five candidates are rejected by the source and two
            # succeed, the answer reports `candidates_executed=2` and the three rejections are
            # otherwise recorded nowhere -- a source half-refusing looks like a narrower vote.
            logger.warning("a candidate's execution failed; the source said: %s", exc.full_text)
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
        held: dict = {"spent_s": 0.0}  # what the source said about a filter value, asked once a turn
        undefined: Deferred | None = None
        for i in range(self.candidates):
            # A candidate that repeats an earlier one's strategy -- from the fourth on, in any mode
            # set to more candidates than STRATEGIES holds, as deep's five are -- must not also
            # repeat its sample: seed_offset sends it its own seed (M124), and
            # temperature_floor at least REPEAT_TEMPERATURE_FLOOR, since at temperature 0 a seed
            # changes nothing. Every other candidate is sent the writer's temperature as set.
            floor = REPEAT_TEMPERATURE_FLOOR if i >= len(STRATEGIES) else None
            with (step(emit, Stage.CANDIDATE, index=i + 1, of=self.candidates), seed_offset(i),
                  temperature_floor(floor)):
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
                    guard_fanout=self.guard_fanout,
                )
                if getattr(outcome, "code", None) is DeferralReason.SOURCE_UNAVAILABLE:
                    # The source did not answer this candidate's proof, so it would not answer the
                    # next one's either: stop here, before another model call and before the
                    # single path's fallback, which would ask the same source again (Codex's
                    # review of #159). A failure, not a deferral (M6), and the turn fails even
                    # when earlier candidates ran: a reduced set is never an answer here -- under
                    # `min_agreement` it is DISAGREEMENT by rule, which would count the outage as
                    # a deferral after all.
                    return AgentAnswer(answer=outcome.reason, failed=True,
                                       reason_code=outcome.code,
                                       candidates_executed=len(executed),
                                       narrowed=_narrowed_of(executed))
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
                try:
                    table = self._execute(outcome, grants, packet)
                except SourceUnavailableError as exc:
                    # The run-time half of the stop above (M188): the source did not answer this
                    # candidate's run, so it would not answer the next candidate's, and dropping it
                    # would ask every later candidate and then the single path.
                    logger.warning("the source did not answer while a query ran: %s", exc.full_text)
                    return AgentAnswer(answer=exc.message, failed=True,
                                       reason_code=DeferralReason.SOURCE_UNAVAILABLE,
                                       candidates_executed=len(executed),
                                       narrowed=_narrowed_of(executed))
                except QueryCancelledError as exc:
                    # A later candidate, or the single path's fallback, could write the stopped
                    # statement again: the turn ends on the cancel, as the single path does.
                    logger.warning("the source cancelled a query while it ran: %s", exc.full_text)
                    return AgentAnswer(answer=exc.message, failed=True,
                                       reason_code=DeferralReason.EXECUTION_FAILED,
                                       candidates_executed=len(executed),
                                       narrowed=_narrowed_of(executed))
            if table is not None:
                executed.append((self._confirmed(outcome, table, snapshot, held), table))

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
                # As asked, not the policy's rewrite: the selector is a model call (M156).
                sql=executed[group[0]][0].model_sql(),
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
        approved, table = self._representative(packet, [executed[i] for i in winner])
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
            day_bounds_closed=base.day_bounds_closed,
            prose_failed=base.prose_failed,
        )

    def _representative(self, packet: ContextPacket, members: list):
        """The cluster member whose plan stands for it. The results agree, so for most clusters any
        member does and the first is kept, as before. An empty cluster is the exception (M146):
        empty results with the same columns cluster together, so one candidate may rest on a
        misspelled value and another on grounded ones, and which came first decided the answer.
        The first member the sanity layer would pass stands for it; where none would, the first,
        which the verifier then declines. Without a sanity layer, the first."""
        if self.verifier is None or not getattr(self.verifier, "sanity", False):
            return members[0]
        for approved, table in members:
            if sanity_check(packet.question, table, getattr(approved, "value_grounding", None)) is None:
                return approved, table
        return members[0]

    def _confirmed(self, approved: Approved, table, snapshot: Snapshot, held: dict) -> Approved:
        """`approved` with each filter value the value index could not check asked of the source,
        where the sanity layer decides on the grounding: an empty result, or one NULL row from
        aggregates over nothing. Found is grounded and missing is absent; unanswered stays unchecked
        (`confirm_by_source`). Only a not_indexed value is ever asked, never one on a masked column
        or a row-filtered table, whose existence those withhold. `held` keeps the answers for the
        turn, so the candidates of a vote ask the source once between them, and the time spent,
        so the turn's probes stop at `PROBE_BUDGET_S`."""
        grounding = getattr(approved, "value_grounding", None)
        if (grounding is None or not getattr(self.verifier, "sanity", False)
                or not any(v.placed for v in grounding.values) or not grounding_decides(table)):
            return approved
        physical = {b.object_id: b.source_object for b in snapshot.source_bindings}
        def holds(table_name: str, column: str, text: str) -> bool | None:
            key = (table_name, column, text)
            if key not in held:
                left = PROBE_BUDGET_S - held["spent_s"]
                if left <= 0:
                    return None  # not kept: a later turn may ask it
                started = time.monotonic()
                held[key] = source_holds(self.adapter, self.dialect, physical,
                                         min(self.timeout_s, left))(*key)
                held["spent_s"] += time.monotonic() - started
                # The column and the verdict, never the value: a log is no place for the data.
                logger.info("asked the source whether %s.%s holds a filter value: %s", table_name,
                            column, {True: "found", False: "missing", None: "no answer"}[held[key]])
            return held[key]
        return replace(approved, value_grounding=confirm_by_source(grounding, holds))

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
            # `failed`, not `deferred`, when the check did not HAPPEN. Same call as the
            # `ModelUnavailable` branch above and for the same reason: something happened TO us,
            # so it is a stated failure and must stay out of the deferral rate (M6). A judge
            # outage stopping the answer and the judge scoring it low both arrive here, and
            # grading them alike would put an infrastructure blip in the column that measures
            # whether the engine abstains well.
            return AgentAnswer(answer=verdict.reason,
                               deferred=not verdict.failed, failed=verdict.failed,
                               reason_code=verdict.code,
                               # The judge's model failing, as the writer's does in `answer`.
                               model_error=verdict.model_error if verdict.failed else None,
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
        cut = _cut(approved, table)
        trace = build_trace(
            question=packet.question,
            approved=approved,
            identity=identity,
            enrichment_version=packet.enrichment_version,
            # `total_ms` is set after the prose; `rows_ms` is when the verified rows were ready to show.
            timing={"execute_ms": execute_ms, "rows_ms": deadline.elapsed_ms},
            result_shape=_shape(table.num_rows, table.num_columns),
            definitions=packet.definitions,
        )
        # Appended HERE, once, rather than rendered by each surface. The four surfaces were fixed
        # one at a time before -- the comment in `cli.py` records the commit messages saying "the
        # two surfaces", then three -- and a disclosure that four places must remember to print is
        # a disclosure that will be missing from one of them. Every surface prints `answer`.
        #
        # After the synthesizer and never through it: a governed fact must not pass a stochastic
        # step that can soften it, drop it, or attach it to the wrong object.
        # Both disclosures appended in one place, so a third cannot be added and forgotten.
        # The lineage one is silent on a source-level `unknown`: that is a standing property of
        # the database, said once at boot, not on every answer it will ever give.
        notes = [note for note in (
            # First: what the answer is a "none" of (M146). Whatever the verifier did with it, an
            # empty result that is answered says so in the engine's words, not only the model's.
            answered_note(table, getattr(approved, "value_grounding", None)),
            disclosure_sentence(trace.narrowed),
            lineage_disclosure_sentence(trace.lineage_completeness,
                                        list(trace.lineage_unresolved),
                                        list(trace.lineage_reasons)),
            cap_sentence(cut) if cut is not None else "",
        ) if note]
        preview = result_preview(table, self.preview_rows, capped=cut is not None)
        # Verified, so nothing after this point withdraws the rows: the prose, the trace's total and the notes follow,
        # and none of them declines. Sent with the notes, so a reader of the table alone still reads what was withheld
        # or cut, and with the preview and trace the final answer is built from, so the two cannot differ.
        result_ready(emit, trace=trace, preview=preview, notes=notes)
        rows_line = f"Returned {table.num_rows} row{'' if table.num_rows == 1 else 's'}."
        prose_failed = False
        if self.prose:
            try:
                with step(emit, Stage.SYNTHESIZE, forced=forced):
                    # As asked, not the policy's rewrite (M156): the synthesizer is a model call, and the
                    # policy's predicate is not the caller's to read in the prose either.
                    answer = self.synthesizer.answer(
                        packet.question, approved.model_sql(), render_result(table, cut_at=cut), forced=forced,
                        row_count=table.num_rows,
                    )
            except ModelUnavailable as exc:
                # The rows stand: they were verified, and a listener may already be showing them. The failure is stated
                # with the provider's own words, as `answer` states one before the rows, and flagged, so a prose model
                # that is down never reads as prose switched off.
                prose_failed = True
                answer = f"{rows_line} The sentence for these rows could not be written: {str(exc).strip()}"
        else:
            # The rows are the answer, in the preview. A fact the engine can state itself, not a model's reading of them.
            answer = rows_line
        trace.timing["total_ms"] = deadline.elapsed_ms
        answer = append_notes(answer, *notes)
        return AgentAnswer(answer=answer, trace=trace, deferred=False, cached=cached,
                           narrowed=getattr(approved, "narrowed", None), preview=preview,
                           attempts=attempts, corrected=approved.corrected,
                           day_bounds_closed=getattr(approved, "day_bounds", None), prose_failed=prose_failed)

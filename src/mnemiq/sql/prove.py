"""Prove an approved plan against the source before running it, in the way that source understands.

Both deciders do this, and for the same reason: a snapshot can be stale, and only the source knows
whether the objects a plan names are really there. Both did it by issuing `EXPLAIN <sql>` and
treating any exception as `EXPLAIN_FAILED`.

**That is Postgres and DuckDB syntax, and it is not universal.** Measured against a live Oracle
23ai instance: `EXPLAIN SELECT id FROM t` raises ORA-02000, "missing PLAN keyword". So every
otherwise-valid Oracle plan was refused before execution, on a step whose entire purpose is to
distinguish a valid plan from an invalid one. Nothing caught it because the adapter's own tests
exercise the adapter and the decider's tests use adapters that speak `EXPLAIN` -- the failure lived
exactly in the gap between two well-tested things, which is where **M26** lived too.

The seam is optional rather than a required method on every adapter. Adapters here are structurally
typed -- there is no ABC and no Protocol for them -- so requiring `validate` would mean editing four
adapters plus every test double that stands in for one, and a test double that silently lacks the
method would then fail in a way that looks like a source error. An adapter that knows a better proof
offers one; the rest keep the behaviour they already had, unchanged.

A proof has three outcomes, not two (Codex's review of #159). The source reads the statement and
accepts it; reads it and refuses it, which is the model's to repair or, where only the policy's
rewrite is refused, the operator's (M158); or it cannot be asked at all -- a connection fault, a
timeout, an interrupt. The third was read as the second: a dropped connection on the rewrite's
second proof blamed the policy, and one on the proof of the statement as asked sent a statement no
source had read back to the model. The adapter tells the third apart (`source_unavailable`, the
second optional method here), since the taxonomy is the driver's: DuckDB types its connection
faults `OperationalError` and its interrupt apart from both, python-oracledb types network faults
`OperationalError`, flags a dead session recoverable and names its call timeout DPY-4024, and
sqlite3 types a missing table and a locked file alike, so that adapter says nothing. Without the
method only Python's own connection, timeout and interrupt errors count, and any other exception
keeps its old reading, a refusal, which the repair loop handles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)


class Proof(StrEnum):
    ACCEPTED = "accepted"  # the source read the statement and would run it
    REJECTED = "rejected"  # the source read the statement and refused it
    UNAVAILABLE = "unavailable"  # the source could not be asked: a connection fault, a timeout


@dataclass(frozen=True)
class ProofOutcome:
    proof: Proof
    detail: str | None = None  # the source's words: the operator's log and the repair loop read them


def _unavailable(adapter: Any, exc: Exception) -> bool:
    """Whether `exc` says the source could not be asked, rather than that it answered no.

    The adapter's word where it has one, whole: its driver's faults are its to know. Without one,
    Python's own connection, timeout and interrupt errors, so an exception nobody classified keeps
    its old reading, a refusal, and the repair loop handles it as before.
    """
    told = getattr(adapter, "source_unavailable", None)
    if told is None:
        return isinstance(exc, (ConnectionError, TimeoutError, InterruptedError))
    try:
        return bool(told(exc))
    except Exception:  # noqa: BLE001 -- a classifier that fails says nothing, and the old reading stands
        logger.warning("The adapter could not classify a proof fault; it is read as a refusal.",
                       exc_info=True)
        return False


def prove_outcome(adapter: Any, target_sql: str) -> ProofOutcome:
    """Ask the source whether it would run the statement, in the form the source understands."""
    validate = getattr(adapter, "validate", None)
    try:
        if validate is not None:
            validate(target_sql)
        else:
            # `EXPLAIN` plans without executing on Postgres, DuckDB and SQLite.
            adapter.execute(f"EXPLAIN {target_sql}")
    except Exception as exc:
        # Logged here rather than where the loop gives up, because a refusal on an attempt that
        # a later one repairs is exactly the one an operator wants to read, and by then it has
        # been overwritten. This is the only record of it: `source_detail` is withheld from the
        # caller, so without this line an EXPLAIN failure would be visible nowhere at all.
        if _unavailable(adapter, exc):
            logger.warning("The source could not be asked to prove a plan: %s", exc)
            return ProofOutcome(Proof.UNAVAILABLE, str(exc))
        logger.warning("EXPLAIN refused a plan; the source said: %s", exc)
        return ProofOutcome(Proof.REJECTED, str(exc))
    return ProofOutcome(Proof.ACCEPTED)


def _refusal(outcome: ProofOutcome) -> Refusal:
    """The refusal a proof's outcome is, for a proof that did not accept.

    The source's words ride in `source_detail`, never in `message`. For a refusal that is
    deliberate and load-bearing: it is what a correction loop reads to repair the SQL, so
    replacing it with a generic sentence would cost the repair its only evidence. For an
    unanswered proof they are the operator's: a driver's fault names a host or a pool.
    """
    if outcome.proof is Proof.UNAVAILABLE:
        return Refusal(
            code=RefusalCode.SOURCE_UNAVAILABLE,
            message=("The source could not be asked whether this query is valid: it did not "
                     "answer. That is not about the query. Try again, and tell the operator if it "
                     "persists; the server log has the source's error."),
            source_detail=outcome.detail,
        )
    return Refusal(
        code=RefusalCode.EXPLAIN_FAILED,
        message="The source rejected this query.",
        source_detail=outcome.detail,
    )


def prove(adapter: Any, target_sql: str) -> Refusal | None:
    """`None` when the source accepts the statement, a `Refusal` when it does not, or could not
    be asked (SOURCE_UNAVAILABLE)."""
    outcome = prove_outcome(adapter, target_sql)
    return None if outcome.proof is Proof.ACCEPTED else _refusal(outcome)


def prove_governed(adapter: Any, target_sql: str, asked_sql: str) -> Refusal | None:
    """`prove` the statement as it will run, and where the source refuses only what the access
    policy added, say so instead (M158); where the source could not be asked, say that.

    A row filter gone stale against its source -- a column renamed under it -- fails the proof of
    every statement that reads its table. EXPLAIN_FAILED is repairable, so each attempt went back
    to the model to repair a statement that was never wrong, and the caller was told the model had
    failed. When the policy's rewrite is refused and the statement as asked is accepted, the fault
    is in what the engine added: the policy, or the snapshot it was checked against. The cost is
    one or two more proofs, and only for a refused statement that a policy rewrote.

    A proof the source did not answer is nobody's statement to repair, at whichever step it comes
    (Codex's review of #159): read as a refusal, one on the first proof sent a sound statement to
    the model, one on the proof of the statement as asked did the same with nothing read, and one
    on the rewrite's second proof blamed the policy. Each is SOURCE_UNAVAILABLE now.

    The rewrite is proved once more before the policy is blamed, still: a fault the adapter does
    not know is read as a refusal, so a rewrite refused once and accepted now was not the
    policy's.

    The proof of the statement as asked tells the caller one thing: that it is valid. A working
    policy would have told them the same. A proof reads the statement and the schema, never rows:
    EXPLAIN plans without running, and Oracle's `validate` parses.
    """
    rewrite = prove_outcome(adapter, target_sql)
    if rewrite.proof is Proof.ACCEPTED:
        return None
    if rewrite.proof is Proof.UNAVAILABLE or asked_sql == target_sql:
        return _refusal(rewrite)  # unanswered, or refused with nothing narrowed to compare
    logger.info("Proving the statement as asked, without the access policy's rewrite.")
    asked = prove_outcome(adapter, asked_sql)
    if asked.proof is Proof.REJECTED:
        return _refusal(rewrite)  # the statement itself is refused: the model's to repair, as before
    if asked.proof is Proof.UNAVAILABLE:
        return _refusal(asked)  # whose the first refusal was cannot be told now
    again = prove_outcome(adapter, target_sql)
    if again.proof is Proof.ACCEPTED:
        logger.info("The access policy's rewrite was accepted when proved again; its first "
                    "refusal was not the policy's.")
        return None
    if again.proof is Proof.UNAVAILABLE:
        return _refusal(again)  # not the policy's either
    # `prove_outcome` logged the source's words for the rewrite; this says whose they were.
    logger.warning("The source accepts the statement as asked and refuses the access policy's "
                   "rewrite of it: a row filter, or the snapshot it was checked against, no "
                   "longer matches the source.")
    return Refusal(
        code=RefusalCode.POLICY_REWRITE_REJECTED,
        message=("The source accepts this query as written but rejects it once your access policy "
                 "is applied, so it cannot be run for you. A row filter, or the schema this "
                 "deployment checked the policy against, no longer matches the source. This is a "
                 "configuration fault for the operator, not a limit on your access; the server log "
                 "has the source's error."),
    )

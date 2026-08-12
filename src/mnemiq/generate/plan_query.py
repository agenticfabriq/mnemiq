from __future__ import annotations

from dataclasses import dataclass, replace

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import DeferralReason, Snapshot
from mnemiq.generate.generator import Generator
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode


@dataclass
class Deferred:
    reason: str
    # What the caller should do next. `reason` is prose for a human; this is for a machine
    # (register M6).
    code: DeferralReason = DeferralReason.UNANSWERABLE


Outcome = Approved | Deferred

# Refusals the corrector can fix with one surgical edit: both are silently-wrong SQL that
# runs fine and answers wrong. A guard (unauthorized table) is never in this set.
CORRECTABLE = frozenset({RefusalCode.LOGIC_LINT, RefusalCode.VALUE_GROUNDING})


def plan_query(
    packet: ContextPacket,
    snapshot: Snapshot,
    grants: GrantSet,
    generator: Generator,
    adapter=None,
    max_attempts: int = 3,
    dialect: str = "duckdb",
    target: str = "postgres",
    feedback: str | None = None,
    corrector=None,
    values=None,
) -> Outcome:
    """Propose, decide, repair -- and defer rather than guess.

    The model proposes; `decide` has the final say. A refusal it can act on comes back as
    feedback. An unauthorized reference does not: retrying it would be inviting the model to
    find another route to data this identity may not have.

    `feedback` seeds the loop -- the agent uses it to hand back what the *database* said, a
    thing no amount of static analysis could have known.
    """
    visible = visible_schema(snapshot, grants)
    policy = build_access_policy(snapshot, grants)
    registry = getattr(snapshot, "registry", {})  # {} for a plain single-source Snapshot
    if not packet.cards or not visible:
        if not grants.available:
            # The policy could not be READ. Denying everything is correct; saying "your access"
            # is not -- this is an outage, and it needs an operator, not a rephrase (M2).
            return Deferred(
                reason=("The authorization policy could not be read, so no access could be "
                        "resolved. This is a configuration fault, not a limit on your account."),
                code=DeferralReason.POLICY_UNAVAILABLE,
            )
        return Deferred(reason="No tables are available to answer this question with your access.",
                        code=DeferralReason.NO_TABLES)

    last: Refusal | None = None

    for _attempt in range(max_attempts):
        proposal = generator.propose(packet, feedback)
        if proposal.sql is None:
            return Deferred(
                reason=proposal.reason or "The model could not answer from these tables.",
                code=DeferralReason.UNANSWERABLE,
            )

        verdict = decide(
            proposal.sql, visible, adapter=adapter, dialect=dialect, target=target,
            values=values, policy=policy, registry=registry
        )

        corrected = False
        if (
            isinstance(verdict, Refusal)
            and verdict.code in CORRECTABLE
            and corrector is not None
        ):
            # one surgical pass: fix only the flagged problem, then re-decide (which re-runs
            # shape/access/lint/values/EXPLAIN, so a bad edit cannot slip through)
            verdict = decide(
                corrector.correct(proposal.sql, verdict.message),
                visible,
                adapter=adapter,
                dialect=dialect,
                target=target,
                values=values,
                policy=policy,
                registry=registry,
            )
            # Only when the repair is what carried it: a correction that still refuses is not
            # a corrected plan, it is a failed one, and reporting it would overstate the work.
            corrected = isinstance(verdict, Approved)

        if isinstance(verdict, Approved):
            return replace(verdict, corrected=corrected)

        if verdict.code == RefusalCode.UNAUTHORIZED_TABLE:
            # Do not retry. A guard that can be retried is a puzzle, not a guard.
            return Deferred(
                reason=(
                    f"Answering this would require access to {verdict.subject!r}, "
                    "which you do not have."
                ),
                code=DeferralReason.AUTHORIZATION,
            )

        last = verdict
        feedback = verdict.message

    reason = last.message if last else "The query could not be made valid."
    return Deferred(
        reason=f"Could not produce a valid query after {max_attempts} attempts. {reason}",
        code=DeferralReason.INVALID_QUERY,
    )

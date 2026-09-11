from __future__ import annotations

import logging

from dataclasses import dataclass, replace

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import DeferralReason, Snapshot
from mnemiq.generate.undefined_terms import ungrounded_terms
from mnemiq.generate.generator import Generator
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.decide import decide
from mnemiq.sql.views import inventory_for
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class Feedback:
    """Text fed back to the model, and whether the SOURCE wrote any of it.

    One value rather than two parameters, and `from_source` has no default, because the unsafe
    setting is the one a caller reaches by forgetting. A bare string plus an optional bool lets
    a future caller seed the feedback, omit the flag, and silently reopen the exit this closes --
    with every existing test still green, because they pin the caller that remembers.
    """

    text: str
    from_source: bool


# What a caller is told when the decider refuses in a way no rewrite can fix: the deferral code,
# and a sentence to use INSTEAD of the refusal's own when that one would echo source words back.
# `UNGOVERNABLE` is the default and says the true thing -- the engine cannot establish that this
# query is governed -- for the source and policy failures that make up the rest, whose subjects
# are the engine's own (a view name from the snapshot, a function name from the catalogue).
#
# INVALID_ROW_FILTER is deliberately NOT mapped to `policy_unavailable`, which was the first
# answer and reads well until an operator sees the card: that code's title is "The access policy
# could not be read", and here it was read and one predicate in it is invalid. UNGOVERNABLE is
# what this codebase already calls that, and the contract enum lists this case under it.
#
# UNAUTHORIZED_TABLE is absent because it keeps its own branch, the one that logs and suppresses
# its subject: its sentence is worded for the caller's next move rather than the model's.
_UNREPAIRABLE: dict[RefusalCode, tuple[DeferralReason, str | None]] = {
    # Grants, exactly like the table case, and it used to spend three attempts inviting the model
    # to find another route to a column this identity may not read -- one it FINDS answers a
    # subtly different question than the one that was asked.
    #
    # And the subject is MODEL-AUTHORED, read off `exp.Column` in the model's own SQL, so a model
    # handed the source's words can name a "column" spelled out of them. Same provenance as the
    # table branch's subject and the same suppression, which is the third exit of that kind.
    RefusalCode.UNAUTHORIZED_COLUMN: (
        DeferralReason.AUTHORIZATION,
        "Answering this would require access to a column you do not have.",
    ),
}


def plan_query(
    packet: ContextPacket,
    snapshot: Snapshot,
    grants: GrantSet,
    generator: Generator,
    adapter=None,
    max_attempts: int = 3,
    dialect: str = "duckdb",
    target: str = "postgres",
    feedback: Feedback | None = None,
    corrector=None,
    values=None,
    guard_undefined_terms: bool = False,
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
    # A view's body is what makes a filter on its base tables reach anything (M27). It comes
    # from the snapshot rather than from the source at ask time: it is versioned content, and
    # a body that drifted from the one the policy was reasoned about is a governance change.
    views = inventory_for(snapshot)
    if not views.available and policy.row_filters:
        # Knowable before the first `generator.propose`, and unfixable by rephrasing: the refusal
        # fires ahead of the AST walk, so all `max_attempts` iterations would propose, be refused
        # identically, and feed the same message back to the model. Same shape as the
        # `grants.available` fast-path below, and the same reason -- an outage needs an operator,
        # not a retry.
        return Deferred(
            reason=("This source could not report its views, so the engine cannot confirm that "
                    "a query does not read around a row filter. This is a configuration or "
                    "connectivity fault, not a limit on your access."),
            code=DeferralReason.POLICY_UNAVAILABLE,
        )
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
    # Whether the feedback in hand carries words the SOURCE wrote rather than words this engine
    # wrote. The model is shown those words on purpose -- a rejection is the repair's whole input
    # -- but `proposal.reason` is model-authored free text that this function forwards to the
    # caller verbatim, so a model that quotes its feedback carries the source's words back out.
    # Measured: a generator answering "the database said: <feedback>" returns a DSN through
    # `AgentAnswer.answer`.
    #
    # This tracks the PROVENANCE of the string, not a guess about its content. There is no
    # pattern to match and so no false-positive rate to measure, and it stays correct when a
    # source starts phrasing its errors differently or a DSN turns up in a shape nobody listed.
    carries_source_words = feedback.from_source if feedback else False

    for _attempt in range(max_attempts):
        proposal = generator.propose(packet, feedback.text if feedback else None)

        # M35, and OFF BY DEFAULT -- withdrawn on its own pre-registered criterion.
        #
        # The rule: a term the model had to ASSUME a meaning for, with nothing certified behind
        # it, is a question the data cannot answer however plausible the SQL looks. Checked before
        # the SQL is decided, because the SQL is what makes it look answerable -- it parses, it
        # runs, it returns rows -- and the guard the engine already had is "no such column", which
        # cannot see a term whose derivation is spelled from columns that all exist.
        #
        # It works on the item it was built for: `lifetime value` defers with a reason naming the
        # term, where the engine used to answer with an invented derivation 4 times in 6.
        #
        # It is off because of what it costs. Beacon's answerable band, one pass over 24 items at
        # `30632b9` (run 06a95802-d794-7a2b-8000-ead4e3573b73): **12 deferrals**, against a prior
        # of 0 in 144 observations. Strict accuracy 66.7% -> 45.8%. The threshold named in advance
        # was 2-3%, with "pull it rather than tune it" written down before the number was seen.
        #
        # Two causes, and only the smaller one is fixable here. Some refusals are containment --
        # `total revenue` declared against a certified `revenue` -- which grounding could discharge
        # if a phrase containing a defined term counted as that term. It must not: "revenue per
        # customer" is a DERIVATION over a certified term and is the exact shape M35 is about, so
        # discharging containment silences the guard on its own finding.
        #
        # The rest is the declaration itself. The model declared `data` ("which currencies appear
        # in the data"), `processed`, `take in`, `fourth quarter of 2025` and `merchants` as terms
        # requiring a certified definition. No glossary defines those, and none ever will. The
        # 2026-08-13 review killed the DETERMINISTIC version of this on the sentence "two
        # consecutive words absent from a schema vocabulary is the normal condition of a sentence,
        # not a signal" -- and the same failure arrived through the model-declared version, which
        # was built specifically to route around it. Even discounting every containment case, the
        # residual is ~29%, an order of magnitude past the line.
        #
        # Kept, not deleted, because the mechanism is proven on ltv and the failure is in the
        # DECLARATION's scope rather than in the engine keeping the decision. Reviving it means a
        # narrower thing for the model to declare, and a new measurement -- not a prompt tweak.
        missing = (
            ungrounded_terms(proposal.assumed_terms, packet.definitions)
            if guard_undefined_terms
            else []
        )
        if missing:
            # `missing` is model-authored too -- the terms come from `proposal.assumed_terms`,
            # parsed straight out of the reply -- so this exit needs the same check as the
            # stated-reason one below, and it runs FIRST. A model that echoes its feedback into
            # the term list rather than into `reason` leaves through here.
            named = ", ".join(repr(t) for t in missing)
            # Which term is the one fact this deferral exists to carry, so when it cannot be
            # said to the caller it still has to be said somewhere.
            logger.warning("undefined terms refused: %s", named)
            return Deferred(
                reason=(
                    (f"No certified definition for {named}. " if not carries_source_words
                     else "A term in this question has no certified definition. ")
                    + "The data does not say how to compute it, so any answer would be a guess "
                    "at your business rule rather than a reading of your data."
                ),
                code=DeferralReason.UNDEFINED_TERM,
            )

        if proposal.sql is None:
            # The model's own words, EXCEPT when it was shown the source's. Then they are the
            # one thing it might be repeating, and this is the path they would leave by.
            return Deferred(
                reason=(
                    "The model could not answer from these tables."
                    if carries_source_words
                    else (proposal.reason or "The model could not answer from these tables.")
                ),
                code=DeferralReason.UNANSWERABLE,
            )

        verdict = decide(
            proposal.sql, visible, adapter=adapter, dialect=dialect, target=target,
            values=values, policy=policy, registry=registry, views=views
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
                corrector.correct(proposal.sql, verdict.repair_text),
                visible,
                adapter=adapter,
                dialect=dialect,
                target=target,
                values=values,
                policy=policy,
                registry=registry,
                views=views,
            )
            # Only when the repair is what carried it: a correction that still refuses is not
            # a corrected plan, it is a failed one, and reporting it would overstate the work.
            corrected = isinstance(verdict, Approved)

        if isinstance(verdict, Approved):
            return replace(verdict, corrected=corrected)

        if verdict.code == RefusalCode.UNAUTHORIZED_TABLE:
            # Do not retry. A guard that can be retried is a puzzle, not a guard.
            #
            # `subject` is model-authored: `authz_guard` reads it off the table name in the
            # model's own SQL, so a model handed the source's words can name a "table" spelled
            # out of them. Third exit of the same kind, and the naming is what makes this one
            # worth keeping when it is safe -- a caller told WHICH table it lacks can ask for it.
            logger.warning("unauthorized table refused: %r", verdict.subject)
            return Deferred(
                reason=(
                    f"Answering this would require access to {verdict.subject!r}, "
                    "which you do not have."
                    if not carries_source_words
                    else "Answering this would require access you do not have."
                ),
                code=DeferralReason.AUTHORIZATION,
            )

        if not verdict.repairable:
            # The generalisation of the branch above, and it took until M98 to notice the branch
            # WAS one. `REPAIRABLE` existed with ten codes and `Refusal.repairable` read it, and
            # nothing else did: every other refusal was retried whatever the classification said,
            # so an unrepairable one cost three model calls to reach the verdict the first one
            # already had, and arrived as INVALID_QUERY -- "could not produce a valid query after
            # 3 attempts", which claims an attempt that could not have worked.
            #
            # The message goes back whole, because these refusals are the ones that say what an
            # OPERATOR must change, and burying that under a retry count is what made the useful
            # sentence look like a footnote to a failure.
            # The subject too, and that is not decoration: when it is withheld from the caller
            # below, the deployment's own log is the only place the denied name is written. Same
            # rule the table branch follows -- what cannot be said to the caller still has to be
            # said somewhere -- and the server log is inside the boundary that holds the DSN.
            logger.warning("unrepairable refusal, not retried: %s (%r)",
                           verdict.code.value, verdict.subject)
            code, without_subject = _UNREPAIRABLE.get(
                verdict.code, (DeferralReason.UNGOVERNABLE, None))
            return Deferred(
                reason=(without_subject if without_subject and carries_source_words
                        else verdict.message),
                code=code,
            )

        last = verdict
        carries_source_words = verdict.source_detail is not None
        feedback = Feedback(verdict.repair_text, from_source=carries_source_words)

    reason = last.message if last else "The query could not be made valid."
    return Deferred(
        reason=f"Could not produce a valid query after {max_attempts} attempts. {reason}",
        code=DeferralReason.INVALID_QUERY,
    )

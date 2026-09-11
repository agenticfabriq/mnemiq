from __future__ import annotations

from mnemiq.contract.seams import DeferralReason
from mnemiq.execute.render import render_result
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.grounding import grounding_check
from mnemiq.verify.sanity import sanity_check
from mnemiq.verify.verdict import VerifyVerdict


def _cards_text(packet: ContextPacket) -> str:
    return "\n".join(c.card for c in packet.cards)


# Written out rather than composed from a clause, because composing them produced "The verifier
# it could not be reached" -- a sentence no test read, since the only parametrised case checked a
# substring of the other branch. These reach the CALLER: `loop` sends `reason` back as the answer
# text and the workbench renders it.
_UNAVAILABLE_REASON = {
    ("error", True): "I could not check this answer: the verifier could not be reached. "
                     "I am not giving you a result I cannot stand behind.",
    ("error", False): "The verifier could not be reached; this answer was not checked.",
    ("unparsed", True): "I could not check this answer: the verifier answered with a reply I "
                        "could not read. I am not giving you a result I cannot stand behind.",
    ("unparsed", False): "The verifier answered with a reply I could not read; this answer was "
                         "not checked.",
}


def _unavailable_reason(why: str, stopping: bool) -> str:
    """Say which failure it was, because the two send an operator to different places.

    `JudgeRead` splits them and this used to flatten them back: `error` is the endpoint not
    answering, and `unparsed` is the endpoint answering with a reply holding no confidence it
    could read -- a model or a token budget, not connectivity. Blaming the network for the
    second is the shape `test_llm_reasoning_budget` exists for, where a starved reasoning model
    silently disabled the verifier for every answer.

    An unrecognised `why` falls back to the connectivity wording rather than raising: a reason
    string this has not been taught is a reporting gap, and taking down an answer path over it
    would be worse than one imprecise sentence.
    """
    return _UNAVAILABLE_REASON.get((why, stopping), _UNAVAILABLE_REASON[("error", stopping)])


class Verifier:
    """Cascade: deterministic sanity -> grounding -> optional LLM judge. An enabled deterministic
    layer that fires is a hard defer and short-circuits. The judge produces a graded confidence;
    defer when it falls below `threshold`. `judge=None` -> deterministic-only.

    Grounding is OFF by default: measured on saved runs it caught 28 wrong but lost 16 correct
    (a real EX cost), where sanity caught 31 wrong for only 2 lost. Grounding is a dial position
    (safety-first) and a soft signal the judge subsumes -- not a free default.

    `fail_closed` decides what an UNREACHABLE judge means, and it defaults to STOPPING the
    answer -- as a failure, not a deferral, since nothing was judged (see `VerifyVerdict`). The
    judge is only ever wired where a mode asks for it -- `deep` alone, out of the box -- so this
    changes nothing for a deployment that did not ask to be checked, and for one that did, the
    old behaviour was to hand back an answer stamped *this was not checked* and hand it back
    anyway. An operator trading assurance for availability during a provider outage sets
    `MNEMIQ_VERIFY_FAIL_CLOSED=0`; that is a decision worth making on purpose rather than a
    default nobody chose."""

    def __init__(self, *, threshold: float = 0.5, sanity: bool = True,
                 grounding: bool = False, judge=None, fail_closed: bool = True) -> None:
        self.threshold = threshold
        self.sanity = sanity
        self.grounding = grounding
        self.judge = judge
        self.fail_closed = fail_closed

    def verify(self, packet: ContextPacket, approved: Approved, table) -> VerifyVerdict:
        if self.sanity:
            v = sanity_check(packet.question, table)
            if v is not None:
                return v
        if self.grounding:
            v = grounding_check(packet, approved)
            if v is not None:
                return v
        if self.judge is not None:
            # `read` when the judge offers it, because the fact must come back WITH the score.
            # The counter-delta this replaced was concurrency-unsafe: the counters are cumulative
            # and the judge is shared across every mode and every request, so a delta measured
            # around one call includes any other thread's failure. MEASURED: a request judged 0.95
            # was reported `judge_unavailable` because a concurrent one failed mid-call.
            #
            # A judge without `read` -- FakeJudge, anyone's stub -- is taken at its word rather
            # than assumed broken, the same way the counter version treated a judge with no
            # counters.
            reader = getattr(self.judge, "read", None)
            if reader is not None:
                got = reader(packet.question, _cards_text(packet), approved.plan_sql,
                             render_result(table, max_rows=5))
                c, fell_open, why = got.score, got.fell_open, got.reason
            else:
                c, fell_open, why = self.judge.score(
                    packet.question, _cards_text(packet), approved.plan_sql,
                    render_result(table, max_rows=5)
                ), False, "ok"
            if fell_open:
                # No score, because there was no judgement. The fail-open constant used to travel
                # in this field and it reads as a confident pass -- M89's shape one field over,
                # where the record said `judge_unavailable` and the number beside it said 1.0.
                return VerifyVerdict(None, self.fail_closed, _unavailable_reason(why,
                                     self.fail_closed), "judge_unavailable",
                                     failed=self.fail_closed,
                                     code=DeferralReason.VERIFIER_UNAVAILABLE)
            return VerifyVerdict(c, c < self.threshold, "The result may not correctly answer the question.", "judge")
        return VerifyVerdict.passed()

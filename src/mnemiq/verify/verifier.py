from __future__ import annotations

from mnemiq.execute.render import render_result
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.grounding import grounding_check
from mnemiq.verify.sanity import sanity_check
from mnemiq.verify.verdict import VerifyVerdict


def _cards_text(packet: ContextPacket) -> str:
    return "\n".join(c.card for c in packet.cards)


class Verifier:
    """Cascade: deterministic sanity -> grounding -> optional LLM judge. An enabled deterministic
    layer that fires is a hard defer and short-circuits. The judge produces a graded confidence;
    defer when it falls below `threshold`. `judge=None` -> deterministic-only.

    Grounding is OFF by default: measured on saved runs it caught 28 wrong but lost 16 correct
    (a real EX cost), where sanity caught 31 wrong for only 2 lost. Grounding is a dial position
    (safety-first) and a soft signal the judge subsumes -- not a free default."""

    def __init__(self, *, threshold: float = 0.5, sanity: bool = True,
                 grounding: bool = False, judge=None) -> None:
        self.threshold = threshold
        self.sanity = sanity
        self.grounding = grounding
        self.judge = judge

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
            # The counter delta, for the reason `_RetryingJudge` documents at length: `score` swallows
            # the exception and returns the fail-open constant, so the RETURN VALUE cannot say whether
            # a judgement happened. `fallbacks` counts errors plus unreadable replies; a judge without
            # the counter (FakeJudge, a stub) is taken at its word rather than assumed broken.
            before = getattr(self.judge, "fallbacks", None)
            c = self.judge.score(
                packet.question, _cards_text(packet), approved.plan_sql, render_result(table, max_rows=5)
            )
            if before is not None and getattr(self.judge, "fallbacks", before) != before:
                # Answer anyway -- that is the product's decision and it is unchanged -- but stop
                # calling it verified. This is the shape M89 was: the verifier switched off against
                # every reasoning model and every answer still read as confidently checked.
                return VerifyVerdict(c, False,
                                     "The verifier could not be reached; this answer was not checked.",
                                     "judge_unavailable")
            return VerifyVerdict(c, c < self.threshold, "The result may not correctly answer the question.", "judge")
        return VerifyVerdict.passed()

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
            c = self.judge.score(
                packet.question, _cards_text(packet), approved.plan_sql, render_result(table, max_rows=5)
            )
            return VerifyVerdict(c, c < self.threshold, "The result may not correctly answer the question.", "judge")
        return VerifyVerdict.passed()

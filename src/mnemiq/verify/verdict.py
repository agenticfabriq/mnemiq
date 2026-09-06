from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifyVerdict:
    """The verifier's read on one executed answer. `defer=True` routes to the engine's existing
    deferral path; `confidence` is the graded judge score (deterministic layers use 0.0)."""

    confidence: float
    defer: bool
    reason: str
    layer: str  # "sanity" | "grounding" | "judge" | "judge_unavailable" | "pass"
    #
    # `judge_unavailable` is the one value that is NOT a judgement. `SemanticJudge.score` fails
    # open -- an unreachable judge scores 1.0, which is exactly what approval scores -- so without
    # this the two are indistinguishable downstream and every answer reads as verified. `defer`
    # stays False on it, deliberately: a dead judge must not stop an answer. What changes is that
    # the record stops claiming the answer was checked.
    #
    # `confidence` is still the fail-open 1.0 in that case, because the field is a float and a
    # consumer reading only the number cannot be helped by this type. `layer` is the discriminator;
    # anything computing on confidence must filter on it first.

    @classmethod
    def passed(cls) -> "VerifyVerdict":
        return cls(confidence=1.0, defer=False, reason="", layer="pass")

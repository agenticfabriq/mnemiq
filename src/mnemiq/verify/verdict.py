from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifyVerdict:
    """The verifier's read on one executed answer. `defer=True` routes to the engine's existing
    deferral path; `confidence` is the graded judge score (deterministic layers use 0.0)."""

    confidence: float | None
    defer: bool
    reason: str
    layer: str  # "sanity" | "grounding" | "judge" | "judge_unavailable" | "pass"
    #
    # `judge_unavailable` is the one value that is NOT a judgement. `SemanticJudge.score` fails
    # open -- an unreachable judge scores 1.0, which is exactly what approval scores -- so without
    # this the two are indistinguishable downstream and every answer reads as verified.
    #
    # `confidence` is **None** there, not the fail-open constant, and that is the second half of
    # the same fix. Carrying 1.0 beside `layer="judge_unavailable"` left the number lying to
    # anyone who read it without the discriminator, and the argument for keeping it -- that a
    # float field cannot help such a reader -- was answered by widening the field. Every consumer
    # already types it `float | None`. No float is honest here: 1.0 reads as a confident pass and
    # 0.0 as a confident failure, and what happened was neither.
    #
    # Whether `defer` is True on that layer is the DEPLOYMENT's call, not this type's. It was
    # unconditionally False -- a dead judge must not stop an answer -- which meant the mode a
    # caller picks for assurance returned answers stamped "not checked" and returned them anyway.
    # `Verifier(fail_closed=...)` decides now; see the issue #2 discussion recorded there.

    @classmethod
    def passed(cls) -> "VerifyVerdict":
        """Nothing objected. 1.0 is a real reading here -- every enabled layer ran and cleared it
        -- which is exactly what it is not on `judge_unavailable`."""
        return cls(confidence=1.0, defer=False, reason="", layer="pass")

from __future__ import annotations

from dataclasses import dataclass

from mnemiq.contract.seams import DeferralReason


@dataclass
class VerifyVerdict:
    """The verifier's read on one executed answer.

    `defer=True` means do not return this answer. `confidence` is the graded judge score, None
    where no judgement happened, and 0.0 from the deterministic layers, which grade nothing.

    `failed` says the check did not HAPPEN, and it is not a synonym for `defer`. Both stop the
    answer; only one is a statement about the data. An outage that lands in the deferral rate is
    M6, which this codebase has already paid for twice -- `EXECUTION_FAILED` and
    `MODEL_UNAVAILABLE` are both marked *NOT a deferral* at the enum for it -- and a fail-closed
    verifier is the third way to arrive.
    """

    confidence: float | None
    defer: bool
    reason: str
    layer: str  # "sanity" | "grounding" | "judge" | "judge_unavailable" | "pass"
    failed: bool = False
    # What the caller is told this was. Carried rather than derived from `layer`, so the engine
    # does not have to keep a second copy of this mapping in a branch.
    code: DeferralReason = DeferralReason.VERIFICATION
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
    # `Verifier(fail_closed=...)` decides now, and when it stops the answer it stops it as a
    # FAILURE, not a deferral: see `failed` above.

    @classmethod
    def passed(cls) -> "VerifyVerdict":
        """Nothing objected. 1.0 is a real reading here -- every enabled layer ran and cleared it
        -- which is exactly what it is not on `judge_unavailable`."""
        return cls(confidence=1.0, defer=False, reason="", layer="pass")

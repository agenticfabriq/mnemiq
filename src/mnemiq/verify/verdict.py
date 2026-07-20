from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifyVerdict:
    """The verifier's read on one executed answer. `defer=True` routes to the engine's existing
    deferral path; `confidence` is the graded judge score (deterministic layers use 0.0)."""

    confidence: float
    defer: bool
    reason: str
    layer: str  # "sanity" | "grounding" | "judge" | "pass"

    @classmethod
    def passed(cls) -> "VerifyVerdict":
        return cls(confidence=1.0, defer=False, reason="", layer="pass")

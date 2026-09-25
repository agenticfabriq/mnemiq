from __future__ import annotations

from dataclasses import dataclass

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent

# Modes are USER-INTENT bundles (pay ~6x when the answer matters), not an accuracy
# router: routing by difficulty measured no signal (plan 18 calibration -- deep's gain
# concentrates where candidates can converge, not on "hard" questions).


@dataclass(frozen=True)
class Mode:
    """A named bundle of Agent knobs. An Agent configured with these IS the mode."""

    name: str
    candidates: int
    corrector: bool  # wire the SQL corrector?
    judge: bool  # wire the selector-judge? (meaningful only when candidates > 1)
    min_agreement: float | None
    budget_s: float
    max_attempts: int
    verify: str = "sanity"  # result-verifier level: "off" | "sanity" | "full" (sanity+judge)


# Verify levels: sanity (deterministic empty/null net) in every mode; the LLM judge only
# in `deep`. MNEMIQ_VERIFY=0 forces off (byte-for-byte); =1 forces full everywhere.
MODES: dict[str, Mode] = {
    "instant": Mode("instant", candidates=1, corrector=False, judge=False,
                    min_agreement=None, budget_s=30.0, max_attempts=1, verify="sanity"),
    # `full`, not `sanity`, and it is the DEFAULT mode -- so a fresh install is judged.
    # The trade is measured, not assumed (technical report §7, 437 gradable mini-dev
    # cases): sanity alone catches 31 of 190 wrong for 2 of 232 correct lost, and the
    # judge at the 0.5 cutoff catches 94 for 22 lost -- the wrong-answer rate falls
    # 39.0% -> 19.7% and exact match falls 47.6% -> 43.1%. Roughly four and a half
    # points of EX buys halving the wrong-rate, and being wrong is the expensive
    # failure for this product. Sanity stays in front of the judge rather than being
    # replaced by it: `Verifier.verify` returns on the first layer that produces a
    # verdict, so the cases sanity defers never reach the judge and 17% of answers
    # skip the LLM call. `instant` keeps `sanity` -- a mode whose contract is
    # "cheapest, no retries" cannot also buy an LLM call per answer.
    # `MNEMIQ_VERIFY=0` forces every mode off, byte for byte, for a deployment that
    # wants the old behaviour.
    "thinking": Mode("thinking", candidates=1, corrector=True, judge=False,
                     min_agreement=None, budget_s=120.0, max_attempts=3, verify="full"),
    # 0.6 keeps ~85% coverage at +3.8 precision (plan 17 offline table); the gate's
    # full-set rule turns a dropped candidate into an honest deferral.
    "deep": Mode("deep", candidates=5, corrector=True, judge=True,
                 min_agreement=0.6, budget_s=300.0, max_attempts=3, verify="full"),
}
DEFAULT_MODE = "thinking"


def build_agent(
    mode: Mode,
    *,
    generator,
    synthesizer,
    adapter,
    cache,
    corrector,
    values,
    selector,
    verifier=None,
    guard_undefined_terms: bool = False,
    guard_fanout: bool | None = False,
) -> Agent:
    """One thin Agent per mode over SHARED components (safe: they hold no per-call state)."""
    return Agent(
        generator=generator,
        synthesizer=synthesizer,
        adapter=adapter,
        cache=cache,
        budget=Budget(wall_clock_s=mode.budget_s, max_attempts=mode.max_attempts),
        candidates=mode.candidates,
        corrector=corrector if mode.corrector else None,
        values=values,
        selector=selector if mode.judge else None,
        min_agreement=mode.min_agreement,
        verifier=verifier,
        guard_undefined_terms=guard_undefined_terms,
        guard_fanout=guard_fanout,
    )

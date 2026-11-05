from __future__ import annotations

from dataclasses import dataclass, replace

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.config import DEFAULT_MAX_QUESTION_CHARS, Settings

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
    card_style: str | None = None  # the schema form its prompt carries; None: `Settings.card_style`
    plan_attempts: int = 3  # writer calls spent on what the decider refuses, on the single path; deep's candidates get 1
    prose: bool = True  # a model call turns the rows into sentences; off, the rows are the answer


# Verify levels: sanity (deterministic empty/null net) in every mode; the LLM judge in
# `thinking` and `deep`. MNEMIQ_VERIFY=0 forces off (byte-for-byte); =1 forces full everywhere.
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
    # verdict, so the cases sanity defers never reach the judge -- on that replay 33 of
    # 437 executed answers (7.6%) skip the LLM call. (Quoted here as 17% until M146: that
    # counted 83 deferrals, 50 of them records that never executed and end before the
    # verifier in the product.) Since M146 an empty result whose filter values are all
    # grounded goes on to the judge, so fewer skip it. `instant` keeps `sanity` -- a mode
    # whose contract is "cheapest, no retries" cannot also buy an LLM call per answer.
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


def modes_for(settings: Settings) -> dict[str, Mode]:
    """The modes these settings run: the table above, then `Settings.mode_settings`, setting by setting. The best prompt
    and the checks worth paying for depend on the model and the dataset, so these are an operator's to benchmark, and a
    profile's sections arrive here as its `mode_settings` (`mnemiq.profiles`).

    Only what was set is applied, so a null reaches the knob that takes one. Refuses a profile or mode that does not
    exist, and a knob only several candidates read on a mode with one: each would otherwise score as "no effect"."""
    from mnemiq.agent.route import UnknownMode  # route imports this module

    modes = dict(MODES)
    for name, override in (settings.mode_settings or {}).items():
        if name not in MODES:
            raise UnknownMode(f"MNEMIQ_MODE_SETTINGS names no mode {name!r}; valid modes: {sorted(MODES)}")
        modes[name] = replace(modes[name], **{field: getattr(override, field) for field in override.model_fields_set})
    for name, mode in modes.items():
        unread = [knob for knob, value in (("judge", mode.judge), ("min_agreement", mode.min_agreement)) if value]
        if mode.candidates <= 1 and unread:
            raise ValueError(f"{name} has one candidate, which {' and '.join(unread)} never read; "
                             "set candidates above 1 or leave them out")
    return modes


def card_styles(settings: Settings) -> list[str]:
    """Every schema form a mode sends. No form is always the largest, so the window is checked for each."""
    return sorted({mode.card_style or settings.card_style for mode in modes_for(settings).values()})


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
    max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS,
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
        plan_attempts=mode.plan_attempts,
        prose=mode.prose,
        guard_undefined_terms=guard_undefined_terms,
        guard_fanout=guard_fanout,
        max_question_chars=max_question_chars,
    )

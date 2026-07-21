import dataclasses

import pytest

from mnemiq.agent.loop import Agent
from mnemiq.agent.modes import DEFAULT_MODE, MODES, build_agent


def test_registry_has_exactly_the_three_modes():
    assert sorted(MODES) == ["deep", "instant", "thinking"]
    assert DEFAULT_MODE == "thinking"
    assert all(name == m.name for name, m in MODES.items())


def test_thinking_is_todays_product_bundle():
    # These are build_runtime's pre-plan-18 knob values; `thinking` now ALSO carries the
    # sanity verify net (verifier-productionization) -- a deliberate, documented safety add.
    m = MODES["thinking"]
    assert (m.candidates, m.corrector, m.judge, m.min_agreement) == (1, True, False, None)
    assert (m.budget_s, m.max_attempts) == (120.0, 3)


def test_instant_strips_the_repair_machinery():
    m = MODES["instant"]
    assert (m.candidates, m.corrector, m.judge, m.min_agreement) == (1, False, False, None)
    assert (m.budget_s, m.max_attempts) == (30.0, 1)


def test_deep_is_the_parked_multi_candidate_machinery():
    m = MODES["deep"]
    assert (m.candidates, m.corrector, m.judge, m.min_agreement) == (5, True, True, 0.6)
    assert (m.budget_s, m.max_attempts) == (300.0, 3)


def test_mode_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MODES["instant"].candidates = 9


def test_build_agent_wires_knobs_per_mode():
    gen, syn, ad, cache, corr, vals, sel = (object() for _ in range(7))

    deep = build_agent(MODES["deep"], generator=gen, synthesizer=syn, adapter=ad,
                       cache=cache, corrector=corr, values=vals, selector=sel)
    assert isinstance(deep, Agent)
    assert deep.candidates == 5 and deep.corrector is corr and deep.selector is sel
    assert deep.min_agreement == 0.6
    assert (deep.budget.wall_clock_s, deep.budget.max_attempts) == (300.0, 3)

    instant = build_agent(MODES["instant"], generator=gen, synthesizer=syn, adapter=ad,
                          cache=cache, corrector=corr, values=vals, selector=sel)
    assert instant.corrector is None and instant.selector is None
    assert instant.budget.max_attempts == 1
    assert instant.values is vals  # value grounding stays wired in EVERY mode


def test_modes_carry_verify_levels():
    # sanity (free deterministic net) everywhere; the LLM judge only in deep.
    assert MODES["instant"].verify == "sanity"
    assert MODES["thinking"].verify == "sanity"
    assert MODES["deep"].verify == "full"

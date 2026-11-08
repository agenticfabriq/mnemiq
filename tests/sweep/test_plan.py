"""`plan`: what a sweep will run and cost, and the gain it could show, said before anything runs."""

from __future__ import annotations

import pytest

from mnemiq.sweep.config import load_config
from mnemiq.sweep.plan import LOAD_WAIT_S, estimate, render

from .conftest import KEY_ENV, UNREACHABLE, live_postgres_env, tiny_dataset, write_config


def test_the_plan_says_when_a_set_is_too_small_to_search(tmp_path):
    cfg = load_config(write_config(tmp_path, dataset=tiny_dataset(tmp_path, cases=15, answerable=11), arms_in_screen=10))
    assert "check mode" in render(cfg)


def test_the_plan_shows_the_gain_the_room_and_the_copies(tmp_path):
    cfg = load_config(write_config(tmp_path, dataset=tiny_dataset(tmp_path, cases=26, answerable=17), arms_in_screen=10))
    text = render(cfg)
    assert "9 more right" in text and "at most 12" in text and "store copy" in text


def test_the_plan_flags_a_live_source(tmp_path):
    cfg = load_config(write_config(tmp_path, deployment=live_postgres_env(tmp_path)))
    assert "source not pinned" in render(cfg)


def test_the_cost_counts_each_round(tmp_path):
    # 26 cases, 17 answerable: 5 + 2 held out, 19 to tune on. The screen is the start and 2 arms; the start runs twice
    # either way, since that is how a sweep learns whether runs repeat.
    cfg = load_config(write_config(tmp_path, dataset=tiny_dataset(tmp_path, cases=26, answerable=17), arms_in_screen=2,
                                   search={"seconds_per_question": 36.0}, machine={"parallel": 2}))
    est = estimate(cfg)
    screen_one, screen_two = (2 + 2) * 19, 3 * 2 * 19
    timing = 2 * 19  # the start and one finalist, once each, alone
    confirm_one = 2 * 7  # the assembled profile and the start, on the held-out part
    assert est.case_runs == {1: screen_one + timing + confirm_one, 2: screen_two + timing + 2 * confirm_one}
    # The timing round first waits, at most LOAD_WAIT_S, for the machine's load to fall.
    assert est.hours == {1: pytest.approx(((screen_one + confirm_one) / 2 + timing) * 36 / 3600 + LOAD_WAIT_S / 3600),
                         2: pytest.approx(((screen_two + 2 * confirm_one) / 2 + timing) * 36 / 3600
                                          + LOAD_WAIT_S / 3600)}
    assert "waiting up to 5 minutes" in render(cfg)


def test_dollars_wait_for_measured_tokens(tmp_path):
    cfg = load_config(write_config(
        tmp_path, dataset=tiny_dataset(tmp_path, cases=26, answerable=17), arms_in_screen=2,
        models={"m": {"llm_model": "stub-model", "llm_base_url": UNREACHABLE, "api_key_env": KEY_ENV,
                      "price_in_per_mtok": 2.0, "price_out_per_mtok": 8.0}}))
    assert estimate(cfg).dollars is None and "unknown until the first questions run" in render(cfg)
    est = estimate(cfg, tokens={"m": (1000.0, 100.0)})
    per_case = (1000 * 2.0 + 100 * 8.0) / 1_000_000
    assert est.dollars == {runs: pytest.approx(count * per_case) for runs, count in est.case_runs.items()}


def test_a_value_the_engine_refuses_is_never_kept_for_the_grid(tmp_path):
    space = {"writer_rules": ["none"], "instant": {"min_agreement": [0.6]}}  # instant has one candidate
    est = estimate(load_config(write_config(tmp_path, space=space)))
    assert [arm.label for arm, _ in est.refused] == ["instant.min_agreement=0.6"]
    assert est.arms["grid"] == 0 and est.compared == {("m", "instant"): 1}
    # with no screen, each combination is put to the engine on its own
    grid = estimate(load_config(write_config(tmp_path, space=space, search={"strategy": "grid"})))
    assert [arm.label for arm in grid.rounds[("m", "instant")]["grid"]] == ["start", "writer_rules=none"]
    assert len(grid.refused) == 2


def test_when_the_engine_refuses_the_start_nothing_runs(tmp_path, monkeypatch):
    cfg = load_config(write_config(tmp_path, arms_in_screen=2))
    monkeypatch.delenv(KEY_ENV)
    est = estimate(cfg)
    assert est.case_runs == {1: 0, 2: 0} and [arm.label for arm, _ in est.refused] == ["start"]
    assert "nothing runs" in render(cfg) and KEY_ENV in render(cfg)


def test_the_plan_says_why_nothing_is_compared_with_the_start(tmp_path, monkeypatch):
    def why(cfg) -> str:
        (line,) = [line for line in render(cfg).splitlines() if "nothing is compared" in line]
        return line

    assert why(load_config(write_config(tmp_path))).endswith("the space has no value to try")
    one_refused = load_config(write_config(tmp_path, space={"instant": {"min_agreement": [0.6]}}))
    assert "the engine refuses every arm" in why(one_refused)
    two_modes = load_config(write_config(tmp_path, modes=["instant", "thinking"],
                                         space={"instant": {"min_agreement": [0.6]}}))
    assert [line.strip() for line in render(two_modes).splitlines() if "nothing is compared" in line] == [
        "m, instant: nothing is compared with the start: the engine refuses every arm (see skipped)",
        "m, thinking: nothing is compared with the start: the space has no value to try"]
    cfg = load_config(write_config(tmp_path, arms_in_screen=2))
    monkeypatch.delenv(KEY_ENV)
    assert "the engine refuses the start" in why(cfg) and "no value to try" not in render(cfg)


def test_check_mode_has_no_timing_and_no_confirm_round(tmp_path):
    # A start profile that changes a setting gives the defaults arm changes, which a search would time and confirm.
    start = tmp_path / "start.toml"
    start.write_text('default = "tuned"\n[profiles.tuned]\nretrieval_k = 12\n')
    cfg = load_config(write_config(tmp_path, start={"profile_file": str(start)}, search={"strategy": "check"}))
    est = estimate(cfg)
    assert est.arms == {"check": 2}
    assert est.case_runs == {1: (2 + 1) * 8, 2: 2 * 2 * 8}  # every question; the start twice in the first round
    rounds = [line.strip() for line in render(cfg).splitlines()]  # the store's path holds this test's name
    assert not [line for line in rounds if line.startswith(("timing:", "confirm,"))]


def _priced_model(**judge) -> dict:
    return {"llm_model": "stub-model", "llm_base_url": UNREACHABLE, "api_key_env": KEY_ENV,
            "price_in_per_mtok": 2.0, "price_out_per_mtok": 8.0, **judge}


def test_a_judge_on_another_endpoint_leaves_dollars_unknown_where_a_mode_calls_it(tmp_path):
    # Measured tokens cannot be split between the writer and the judge, so the judge's would be priced as the writer's.
    tokens = {"m": (1000.0, 100.0)}
    elsewhere = _priced_model(verify_base_url="http://127.0.0.1:9/judge/v1", verify_api_key_env=KEY_ENV)
    for sections in ({"models": {"m": elsewhere}, "modes": ["thinking"]},
                     {"models": {"m": _priced_model()}, "judge": {"verify_model": "a-judge-model"}, "modes": ["deep"]},
                     {"models": {"m": elsewhere}, "modes": ["instant"], "space": {"instant": {"verify": ["full"]}}}):
        cfg = load_config(write_config(tmp_path, search={"budget_dollars": 5.0}, **sections))
        assert estimate(cfg, tokens=tokens).dollars is None, sections
        lines = render(cfg, tokens=tokens).splitlines()
        assert any("dollars: unknown" in line and "the judge's price is not known" in line for line in lines)
        budget = next(line for line in lines if line.strip().startswith("budget:"))
        assert "the judge's price is not known" in budget and "fits" not in budget


def test_a_mode_that_never_calls_the_judge_is_priced(tmp_path):
    tokens = {"m": (1000.0, 100.0)}
    elsewhere = _priced_model(verify_base_url="http://127.0.0.1:9/judge/v1", verify_api_key_env=KEY_ENV)
    est = estimate(load_config(write_config(tmp_path, models={"m": elsewhere}, modes=["instant"])), tokens=tokens)
    per_case = (1000 * 2.0 + 100 * 8.0) / 1_000_000
    assert est.dollars == {runs: pytest.approx(count * per_case) for runs, count in est.case_runs.items()}

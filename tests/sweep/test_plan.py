"""`plan`: what a sweep will run and cost, and the gain it could show, said before anything runs."""

from __future__ import annotations

import pytest

from mnemiq.sweep.config import load_config
from mnemiq.sweep.plan import estimate, render

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
    assert est.hours == {1: pytest.approx(((screen_one + confirm_one) / 2 + timing) * 36 / 3600),
                         2: pytest.approx(((screen_two + 2 * confirm_one) / 2 + timing) * 36 / 3600)}


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

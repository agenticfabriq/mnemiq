"""The rounds of a sweep: each arm run in a process of its own, the start run twice to decide whether runs repeat, a
search stopped when it has no room, the budget checked before each arm, and a stopped sweep resumed where it stopped."""

from __future__ import annotations

import json
import multiprocessing
import re
from dataclasses import replace
from pathlib import Path

import pytest

from mnemiq.config import DEFAULT_RETRIEVAL_K
from mnemiq.sweep.arms import check_arms, grid_arms, screen_arms
from mnemiq.sweep.config import SweepConfigError, load_config, load_questions
from mnemiq.sweep.report import report
from mnemiq.sweep.run import arm_folder, room, run

from .conftest import (
    CASES, alter, one_arm, out, read_rows, rewrite, start_of, started_sweep, sweep_json, tiny_dataset, tiny_deployment,
    tuning, write_config, write_rows,
)


def roomy(tmp_path: Path, stub, *, known: bool = False, **sections):
    """A loaded config over 10 questions about the tiny tables, 9 of them with an answer, 7 of those to tune on: enough
    for an arm to win against a start that misses them all, with up to 3 arms compared with it. The stub declines a
    question it does not know, so the start misses every one; with `known` it learns them, each with a wrong answer."""
    dataset = tiny_dataset(tmp_path, cases=10, answerable=9, stub=stub if known else None)
    return load_config(write_config(tmp_path, model=stub, dataset=dataset, **sections))


def all_cases(cfg) -> set[str]:
    return {case.id for case in load_questions(cfg)}


def _owed(cfg, arm) -> set[tuple[str, str]]:
    cases = all_cases(cfg) if cfg.search.strategy == "check" else {case.id for case in tuning(cfg)}
    return {(case, mode) for case in cases for mode in arm.modes}


def run_complete(cfg, arm, repeat: int) -> bool:
    """Run `repeat` of `arm` holds one row for each case and mode it owes, and is not invalid."""
    keys = [(row["id"], row["mode"]) for row in read_rows(cfg, arm, repeat)]
    return (not (arm_folder(cfg, arm, repeat) / "invalid.json").exists() and len(keys) == len(set(keys))
            and set(keys) == _owed(cfg, arm))


def arm_complete(cfg, arm) -> bool:
    """Every run `arm` owes is complete: as many as the sweep decided."""
    return all(run_complete(cfg, arm, repeat) for repeat in range(1, sweep_json(cfg)["repeats"] + 1))


def arm_complete_or_absent(cfg, arm) -> bool:
    """Each run of `arm` that started is complete: none was stopped part way."""
    runs = arm_folder(cfg, arm, 1).parent
    return all(run_complete(cfg, arm, int(folder.name.removeprefix("run-"))) for folder in runs.glob("run-*"))


def every_first_round_arm_has_run(cfg, times: int) -> bool:
    arms = check_arms(cfg) if cfg.search.strategy == "check" else screen_arms(cfg)
    return all(run_complete(cfg, arm, repeat) for arm in arms for repeat in range(1, times + 1))


def _arm_dirs(cfg, round: str) -> list[Path]:
    return [folder for model in cfg.models for folder in (out(cfg) / model / round).glob("*/*") if folder.is_dir()]


def grid_folders(cfg) -> list[Path]:
    return _arm_dirs(cfg, "grid")


def arm_ids(cfg, round: str) -> set[str]:
    return {folder.name for folder in _arm_dirs(cfg, round)}


def rows_for(cfg, profile_id: str) -> set[str]:
    """The cases the first check run of the arm `profile_id` answered."""
    return {row["id"] for folder in _arm_dirs(cfg, "check") if folder.name == profile_id
            for row in _rows(folder / "run-1" / "rows.jsonl")}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def smoke_rows(cfg) -> list[dict]:
    return [row for path in (out(cfg) / "smoke").rglob("rows.jsonl") for row in _rows(path)]


def round_rows(cfg, round: str) -> list[dict]:
    return [row for model in cfg.models for path in (out(cfg) / model / round).rglob("rows.jsonl")
            for row in _rows(path)]


def test_an_arm_resumes_and_drops_a_torn_line(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    write_rows(cfg, arm, first=3, torn_last=True)
    run(cfg, until="screen")
    ids = [r["id"] for r in read_rows(cfg, arm)]
    assert len(ids) == len(set(ids)) == len(tuning(cfg))


@pytest.mark.parametrize("change", ["store", "authz", "env_setting", "start_profile"])
def test_a_changed_input_is_refused_before_the_next_arm(tmp_path, stub_model, change):
    stub = stub_model(gold=True)
    cfg = started_sweep(tmp_path, stub)
    alter(cfg, change)
    asked = stub.calls
    with pytest.raises(SweepConfigError, match="another config"):
        run(load_config(cfg.path))
    # Refused before any arm: nothing asked, and no arm marked invalid by its own check of the pinned files.
    assert stub.calls == asked and not list(out(cfg).rglob("invalid.json"))


def test_a_larger_budget_resumes(tmp_path, stub_model):
    cfg = roomy(tmp_path, stub_model(gold=True, seconds=0.2), arms_in_screen=1, search={"budget_hours": 0.0002})
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "budget"
    runs, spent = sorted(out(cfg).rglob("run-*")), sweep_json(cfg)["seconds"]
    run(cfg)   # the time the first run took counts against the same budget
    assert sweep_json(cfg)["stopped"] == "budget" and sorted(out(cfg).rglob("run-*")) == runs
    assert sweep_json(cfg)["seconds"] >= spent > 0
    run(load_config(rewrite(cfg, search={"budget_hours": 1})))
    assert "stopped" not in sweep_json(cfg) and all(arm_complete(cfg, a) for a in screen_arms(cfg))


def test_the_split_is_written_once_and_read_back(tmp_path, stub_model, monkeypatch):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    first = (out(cfg) / "split.json").read_text()
    # Were the split drawn again, every case would be owed, and the start asked the held-out ones too.
    monkeypatch.setattr("mnemiq.sweep.run.split", lambda cfg, cases: {"tuning": list(cases), "held_out": []})
    run(cfg)
    held_out = set(json.loads(first)["held_out"])
    assert (out(cfg) / "split.json").read_text() == first and held_out
    assert not held_out & {r["id"] for r in read_rows(cfg, start_of(cfg, "instant"))}


@pytest.mark.parametrize("lost", ["split.json", "sweep.json"])
def test_rows_without_the_sweeps_own_files_are_refused(tmp_path, stub_model, lost):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    (out(cfg) / lost).unlink()
    with pytest.raises(SweepConfigError, match=re.escape(lost)):
        run(cfg)


def test_the_budget_stops_before_an_arm_not_during_one(tmp_path, stub_model):
    cfg = roomy(tmp_path, stub_model(gold=True, seconds=0.2), arms_in_screen=1, search={"budget_hours": 0.0002})
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "budget" and all(arm_complete_or_absent(cfg, a) for a in screen_arms(cfg))
    assert run_complete(cfg, start_of(cfg, "instant"), 1)   # the first arm ran: there was no pace to judge it by


def test_the_estimate_stops_the_first_arm_before_any_pace_is_measured(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg = load_config(write_config(tmp_path, model=stub, search={"budget_hours": 0.1, "seconds_per_question": 1000.0}))
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "budget" and not arm_ids(cfg, "screen") and stub.calls == 0


def test_the_dollars_of_the_arms_running_count_before_another_starts(tmp_path, stub_model):
    priced = {"m": {"llm_model": "stub-model", "llm_base_url": "", "api_key_env": "SWEEP_TEST_KEY",
                    "price_in_per_mtok": 2.0, "price_out_per_mtok": 8.0}}
    stub = stub_model(gold=True)
    priced["m"]["llm_base_url"] = stub.url
    cfg = roomy(tmp_path, stub, arms_in_screen=3, models=priced, machine={"parallel": 2},
                search={"budget_hours": 0.0002})
    run(cfg)   # two arms at once, with no pace yet; then the hours are used
    rows = [row for path in out(cfg).rglob("rows.jsonl") for row in _rows(path)]
    spent = sum(row["prompt_tokens"] * 2.0 + row["completion_tokens"] * 8.0 for row in rows) / 1e6
    arm = spent / len(rows) * len(tuning(cfg))
    before = sorted(out(cfg).rglob("run-*"))
    # Room for one and a half arms more: the first starts, and the second would overspend beside it.
    run(load_config(rewrite(cfg, search={"budget_hours": 1, "budget_dollars": spent + 1.5 * arm})))
    assert len(before) == 2 and len(sorted(out(cfg).rglob("run-*"))) == 3 and sweep_json(cfg)["stopped"] == "budget"


def test_another_engine_build_is_refused(tmp_path, stub_model, monkeypatch):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    monkeypatch.setattr("mnemiq.sweep.run.engine_build", lambda *a, **k: {"commit": None, "source_sha": "f" * 64})
    with pytest.raises(SweepConfigError, match="engine"):
        run(cfg)


@pytest.mark.parametrize("strategy", ["screen-then-grid", "check"])
def test_a_model_that_does_not_repeat_runs_twice(tmp_path, stub_model, strategy):
    stub = stub_model(varies=True, gold_when=lambda request: False)   # wrong everywhere, spelled anew each time
    screen = {"arms_in_screen": 1} if strategy != "check" else {}
    cfg = roomy(tmp_path, stub, known=True, search={"strategy": strategy}, **screen)
    run(cfg, until="screen" if strategy != "check" else None)
    assert sweep_json(cfg)["repeats"] == 2 and every_first_round_arm_has_run(cfg, 2)


def test_no_room_stops_the_search(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(start_misses=2), arms_in_screen=4))
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "no room" and not grid_folders(cfg)
    # No other arm of the model and mode started, and the room the start left is recorded beside the gain it needed:
    # with 5 answerable cases to tune on, none can win.
    first_two = [case["id"] for case in CASES if case.get("answerable", True)][:2]
    missed = set(first_two) & {case.id for case in tuning(cfg)}
    assert arm_ids(cfg, "screen") == {"start"} and room(cfg, "m", "instant") == len(missed)
    assert sweep_json(cfg)["room"] == {"m": {"instant": {"misses": len(missed), "needed": None, "compared": 4,
                                                         "stopped": "no room"}}}


def test_check_mode_runs_two_arms_on_every_case(tmp_path, stub_model):
    profile = tmp_path / "start-profiles.toml"
    profile.write_text('default = "start"\n[profiles.start]\nretrieval_k = 12\n')
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True), search={"strategy": "check"},
                                   start={"profile_file": str(profile), "profile": "start"}))
    run(cfg)
    assert arm_ids(cfg, "check") == {"start", "defaults"}
    assert all(rows_for(cfg, a) == all_cases(cfg) for a in ("start", "defaults"))
    # The defaults undo what the start set, yet check mode searches nothing, so it measures no room. The start repeated
    # itself exactly, so every other arm runs once.
    (defaults,) = [arm for arm in check_arms(cfg) if arm.profile_id == "defaults"]
    assert defaults.changes == (("retrieval_k", DEFAULT_RETRIEVAL_K),) and "room" not in sweep_json(cfg)
    assert sweep_json(cfg)["repeats"] == 1 and not arm_folder(cfg, defaults, 2).exists()


def test_a_count_of_runs_the_config_fixes_is_run_by_every_arm(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True), search={"strategy": "check"},
                                   sweep={"repeats": 2}))
    run(cfg)   # the start repeats itself exactly, which decides nothing here
    assert sweep_json(cfg)["repeats"] == 2 and every_first_round_arm_has_run(cfg, 2)
    assert not any(arm_folder(cfg, arm, 3).exists() for arm in check_arms(cfg))


def test_an_invalid_run_is_reported_and_never_run_again(tmp_path, stub_model, capsys):
    stub = stub_model(gold=True, fail_after=3, status=500)
    cfg, start = one_arm(tmp_path, stub)
    with pytest.raises(SystemExit):
        run(cfg, until="screen")
    # As run_arm records a pinned file that moved: final, whatever the stop left in retries.json for that run.
    (arm_folder(cfg, start, 1) / "invalid.json").write_text(json.dumps({"changed": ["the store"]}))
    rows = read_rows(cfg, start)
    stub.recover()
    run(cfg, until="screen")
    # Its cases in retries.json were no work left, so the start's two runs were done and the repeats decided.
    assert read_rows(cfg, start) == rows and "repeats" in sweep_json(cfg)
    assert json.loads((out(cfg) / "retries.json").read_text()) and "start, run 1: not valid" in capsys.readouterr().out


def test_a_smoke_is_never_read(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True)))
    run(cfg, smoke=2)
    assert smoke_rows(cfg) and not round_rows(cfg, "screen")   # the report reads only the rounds' folders
    assert len(smoke_rows(cfg)) == 2 and report(cfg, "screen", repeats=1)[1] == []
    assert not (out(cfg) / "sweep.json").exists() and not (out(cfg) / "split.json").exists()


def test_a_dead_endpoint_stops_the_round_and_resume_finishes_it(tmp_path, stub_model):
    model = stub_model(gold=True, fail_after=5, status=500)
    cfg = roomy(tmp_path, model, arms_in_screen=1)
    with pytest.raises(SystemExit, match=re.escape(model.url)) as stopped:
        run(cfg, until="screen")
    assert stopped.value.code not in (0, None)
    assert arm_ids(cfg, "screen") == {"start"} and sweep_json(cfg)["stopped"] == "endpoint"   # no other arm started
    model.recover()
    run(cfg, until="screen")
    assert all(arm_complete(cfg, a) for a in screen_arms(cfg)) and "stopped" not in sweep_json(cfg)


def test_a_dead_endpoint_is_named_without_its_credentials(tmp_path, stub_model):
    stub = stub_model(gold=True)
    # The embedder is asked first, and nothing listens on the discard port.
    d = tiny_deployment(tmp_path, stub, env_extra={"MNEMIQ_EMBED_BASE_URL": "http://sweep:hunter2@127.0.0.1:9/v1"})
    cfg = load_config(write_config(tmp_path, deployment=d, model=stub))
    with pytest.raises(SystemExit) as stopped:
        run(cfg, until="screen")
    assert "the embedder at http://127.0.0.1:9/v1" in str(stopped.value) and "hunter2" not in str(stopped.value)


def test_a_worker_that_dies_leaves_its_rows_and_the_next_run_resumes_it(tmp_path, stub_model):
    def kill_the_worker(calls: int) -> None:
        if calls == 4:
            for worker in multiprocessing.active_children():
                worker.kill()

    cfg, start = one_arm(tmp_path, stub_model(gold=True, on_call=kill_the_worker))
    with pytest.raises(SystemExit, match="stopped without finishing"):
        run(cfg, until="screen")
    written = read_rows(cfg, start)
    assert not arm_folder(cfg, start, 2).exists()   # no other run started after it
    run(cfg, until="screen")
    ids = [r["id"] for r in read_rows(cfg, start)]
    assert written and len(ids) == len(set(ids)) == len(tuning(cfg))


@pytest.mark.parametrize("varies, repeats", [(False, 1), (True, 2)], ids=["repeats", "does-not-repeat"])
def test_the_grid_combines_what_the_screen_kept_as_often_as_decided(tmp_path, stub_model, varies, repeats):
    # Either setting alone puts something in the writer's request that the stub answers right for.
    stub = stub_model(varies=varies, gold_when=lambda request: "response_format" in request or "seed" in request)
    space = {"guided_sql": [True], "llm_seed": [7]}
    cfg = roomy(tmp_path, stub, known=True, space=space, machine={"parallel": 2})
    run(cfg, until="grid")
    (combined,) = [arm for arm in grid_arms(cfg, {("m", "instant"): space}) if arm.round == "grid"]
    assert sweep_json(cfg)["repeats"] == repeats and arm_ids(cfg, "grid") == {combined.profile_id}
    assert all(run_complete(cfg, combined, n) for n in range(1, repeats + 1))
    assert not arm_folder(cfg, combined, repeats + 1).exists()


def test_a_value_is_kept_only_when_every_run_the_sweep_decided_on_keeps_it(tmp_path, stub_model):
    asked: dict[str, int] = {}

    def gold_when(request: dict) -> bool:
        if "seed" in request:
            return True   # llm_seed=7 answers right in every run
        if "response_format" not in request:
            return False   # the start answers wrong, and spells it anew each run, so every arm runs twice
        prompt = json.dumps(request["messages"])
        asked[prompt] = asked.get(prompt, 0) + 1
        return asked[prompt] == 1   # guided_sql=true answers right in its first run only

    space = {"guided_sql": [True], "llm_seed": [7]}
    cfg = roomy(tmp_path, stub_model(varies=True, gold_when=gold_when), known=True, space=space)
    run(cfg, until="grid")
    # Read at one run, both values were kept, and the grid would have combined them.
    assert sweep_json(cfg)["repeats"] == 2 and not grid_folders(cfg)


def test_a_round_the_strategy_does_not_run_cannot_be_the_last(tmp_path):
    cfg = load_config(write_config(tmp_path, search={"strategy": "check"}))
    with pytest.raises(ValueError, match="check"):
        run(cfg, until="screen")


def test_a_sweep_with_no_screen_runs_its_start_in_the_grid(tmp_path, stub_model):
    cfg = roomy(tmp_path, stub_model(gold=True), arms_in_screen=1, search={"strategy": "grid"})
    run(cfg)
    start = replace(start_of(cfg, "instant"), round="grid")
    assert arm_ids(cfg, "grid") == {"start", screen_arms(cfg)[1].profile_id} and run_complete(cfg, start, 2)
    assert not arm_ids(cfg, "screen") and [r["arm"] for r in report(cfg, "grid", repeats=1)[1]][0] == "start"

"""The rounds of a sweep: each arm run in a process of its own, the start run twice to decide whether runs repeat, a
search stopped when it has no room, the budget checked before each arm, and a stopped sweep resumed where it stopped."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

import mnemiq
from mnemiq.config import DEFAULT_RETRIEVAL_K
from mnemiq.sweep.arms import check_arms, grid_arms, screen_arms
from mnemiq.sweep.config import SweepConfigError, load_config, load_questions
from mnemiq.sweep.report import report
from mnemiq.sweep.run import _begin, _parts, _rows_so_far, arm_folder, room, run
from mnemiq.sweep.split import split_digest

from .conftest import (
    CASES, KEY_ENV, alter, one_arm, out, read_rows, rewrite, start_of, started_sweep, sweep_json, tiny_dataset,
    tiny_deployment, tuning, write_config, write_rows,
)


def roomy(tmp_path: Path, stub, *, known: bool = False, **sections):
    """A loaded config over 10 questions about the tiny tables, 9 of them with an answer, 7 of those to tune on: enough
    for an arm to win against a start that misses them all, with up to 3 arms compared with it. The stub declines a
    question it does not know, so the start misses every one; with `known` it learns them, each with a wrong answer."""
    dataset = tiny_dataset(tmp_path, cases=10, answerable=9, stub=stub if known else None)
    return load_config(write_config(tmp_path, model=stub, dataset=dataset, **sections))


def priced(*stubs, price: float = 1e6) -> dict:
    """A `[models]` section, a model for each stub (`m`, then `n`), each priced at `price` dollars a million tokens each
    way. At the default a token costs a dollar, so any question costs more than a dollar, whatever runs it and however
    fast."""
    return {name: {"llm_model": "stub-model", "llm_base_url": stub.url, "api_key_env": KEY_ENV,
                   "price_in_per_mtok": price, "price_out_per_mtok": price} for name, stub in zip("mn", stubs)}


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


def _arm_dirs(cfg, round: str, model: str | None = None) -> list[Path]:
    models = cfg.models if model is None else [model]
    return [folder for name in models for folder in (out(cfg) / name / round).glob("*/*") if folder.is_dir()]


def grid_folders(cfg) -> list[Path]:
    return _arm_dirs(cfg, "grid")


def arm_ids(cfg, round: str, model: str | None = None) -> set[str]:
    return {folder.name for folder in _arm_dirs(cfg, round, model)}


def rows_for(cfg, profile_id: str) -> set[str]:
    """The cases the first check run of the arm `profile_id` answered."""
    return {row["id"] for folder in _arm_dirs(cfg, "check") if folder.name == profile_id
            for row in _rows_so_far(folder / "run-1" / "rows.jsonl")}


def smoke_rows(cfg) -> list[dict]:
    return [row for path in (out(cfg) / "smoke").rglob("rows.jsonl") for row in _rows_so_far(path)]


def round_rows(cfg, round: str) -> list[dict]:
    return [row for model in cfg.models for path in (out(cfg) / model / round).rglob("rows.jsonl")
            for row in _rows_so_far(path)]


def all_rows(cfg) -> list[dict]:
    return [row for path in out(cfg).rglob("rows.jsonl") for row in _rows_so_far(path)]


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
    stub = stub_model(gold=True)
    cfg = roomy(tmp_path, stub, arms_in_screen=1, models=priced(stub), search={"budget_dollars": 1.0})
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "budget"
    runs = sorted(out(cfg).rglob("run-*"))
    run(cfg)   # what the first run spent counts against the same budget
    assert sweep_json(cfg)["stopped"] == "budget" and sorted(out(cfg).rglob("run-*")) == runs
    run(load_config(rewrite(cfg, search={"budget_dollars": 1e15})))
    assert "stopped" not in sweep_json(cfg) and all(arm_complete(cfg, a) for a in screen_arms(cfg))


def test_the_time_already_spent_counts_against_the_hours(tmp_path, stub_model):
    stub = stub_model(gold=True, fail_after=3, status=500)
    cfg, _ = one_arm(tmp_path, stub)
    with pytest.raises(SystemExit):
        run(cfg, until="screen")   # stopped part way through the start's first run, with its cases left to ask
    spent = sweep_json(cfg)["seconds"]
    stub.recover()
    asked = stub.calls
    run(load_config(rewrite(cfg, search={"budget_hours": spent / 2 / 3600})))
    assert sweep_json(cfg)["stopped"] == "budget" and stub.calls == asked and sweep_json(cfg)["seconds"] >= spent > 0


def test_the_split_is_written_once_and_read_back(tmp_path, stub_model, monkeypatch):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    first = (out(cfg) / "split.json").read_text()
    # Were the split drawn again, every case would be owed, and the start asked the held-out ones too.
    monkeypatch.setattr("mnemiq.sweep.run.split", lambda cfg, cases: {"tuning": list(cases), "held_out": []})
    run(cfg)
    held_out = set(json.loads(first)["held_out"])
    assert (out(cfg) / "split.json").read_text() == first and held_out
    assert not held_out & {r["id"] for r in read_rows(cfg, start_of(cfg, "instant"))}


def test_a_new_sweep_draws_its_split_over_one_left_by_another(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True)))
    out(cfg).mkdir(parents=True)
    (out(cfg) / "split.json").write_text(json.dumps({"tuning": ["book-count"], "held_out": []}))
    _begin(cfg)
    assert json.loads((out(cfg) / "split.json").read_text())["tuning"] == [case.id for case in tuning(cfg)]
    assert len(tuning(cfg)) > 1


@pytest.mark.parametrize("change, refused", [("another", "not the split this sweep drew"), ("twice", "twice"),
                                            ("both", "in both parts"), ("missing", "does not name")])
def test_a_split_that_is_not_the_one_drawn_is_refused_by_run_and_report(tmp_path, stub_model, change, refused):
    stub = stub_model(gold=True)
    cfg, start = one_arm(tmp_path, stub)
    write_rows(cfg, start, first=3)   # a run stopped part way
    path = out(cfg) / "split.json"
    ids = json.loads(path.read_text())
    tuning_ids, held = ids["tuning"], ids["held_out"]
    if change == "another":   # every case once, in a part of its own: only the record can tell it from the one drawn
        ids = {"tuning": tuning_ids[1:] + held, "held_out": tuning_ids[:1]}
    elif change == "twice":
        ids["tuning"] = tuning_ids + tuning_ids[:1]
    elif change == "both":
        ids["held_out"] = held + tuning_ids[:1]
    else:
        ids["tuning"] = tuning_ids[1:]
    path.write_text(json.dumps(ids))
    # A run stopped by its endpoint, which the next run would clear, were it to write the record before refusing.
    record = sweep_json(cfg) | {"stopped": "endpoint"}
    if change != "another":   # the record agrees with the file, so only the file's own checks can refuse it
        record["split_digest"] = split_digest(ids)
    (out(cfg) / "sweep.json").write_text(json.dumps(record))
    asked, record = stub.calls, (out(cfg) / "sweep.json").read_text()
    with pytest.raises(SweepConfigError, match=refused):
        run(cfg, until="screen")
    with pytest.raises(SweepConfigError, match=refused):
        report(cfg, "screen", repeats=1)
    assert stub.calls == asked and (out(cfg) / "sweep.json").read_text() == record   # refused before it wrote a thing


def test_a_record_with_no_split_digest_says_so_rather_than_blame_the_split(tmp_path, stub_model):
    cfg, start = one_arm(tmp_path, stub_model(gold=True))
    write_rows(cfg, start, first=3)
    record = sweep_json(cfg)
    del record["split_digest"]   # a record written before sweeps kept one
    (out(cfg) / "sweep.json").write_text(json.dumps(record))
    with pytest.raises(SweepConfigError, match="records no digest"):
        report(cfg, "screen", repeats=1)


def test_run_and_report_owe_the_cases_of_the_one_saved_split(tmp_path, stub_model, monkeypatch):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    saved = json.loads((out(cfg) / "split.json").read_text())
    # Were either to draw the split again, it would owe every case.
    for module in ("run", "report"):
        monkeypatch.setattr(f"mnemiq.sweep.{module}.split", lambda cfg, cases: {"tuning": list(cases), "held_out": []})
    (start,) = [record for record in report(cfg, "screen", repeats=1)[1] if record["arm"] == "start"]
    assert [case.id for case in _parts(cfg)["tuning"]] == saved["tuning"]
    assert start["valid"] and start["runs"][0]["cases"] == len(saved["tuning"])


@pytest.mark.parametrize("lost", ["split.json", "sweep.json"])
def test_rows_without_the_sweeps_own_files_are_refused(tmp_path, stub_model, lost):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    (out(cfg) / lost).unlink()
    with pytest.raises(SweepConfigError, match=re.escape(lost)):
        run(cfg)


def test_the_budget_stops_before_an_arm_not_during_one(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg = roomy(tmp_path, stub, arms_in_screen=1, models=priced(stub), search={"budget_dollars": 1.0})
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "budget" and all(arm_complete_or_absent(cfg, a) for a in screen_arms(cfg))
    assert run_complete(cfg, start_of(cfg, "instant"), 1)   # the first arm ran: there was no pace to judge it by


def test_the_estimate_stops_the_first_arm_before_any_pace_is_measured(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg = load_config(write_config(tmp_path, model=stub, search={"budget_hours": 0.1, "seconds_per_question": 1000.0}))
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "budget" and not arm_ids(cfg, "screen") and stub.calls == 0


def test_the_dollars_of_the_arms_running_count_before_another_starts(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg = roomy(tmp_path, stub, arms_in_screen=3, models=priced(stub), machine={"parallel": 2},
                search={"budget_dollars": 1.0})
    run(cfg)   # two arms start at once, with no pace yet, and spend the dollar many times over
    before = sorted(out(cfg).rglob("run-*"))
    rows = all_rows(cfg)
    spent = sum(row["prompt_tokens"] + row["completion_tokens"] for row in rows)   # a dollar a token
    arm = spent / len(rows) * len(tuning(cfg))
    # Room for one and a half arms more: the first starts, and the second would overspend beside it.
    run(load_config(rewrite(cfg, search={"budget_dollars": spent + 1.5 * arm})))
    assert len(before) == 2 and len(sorted(out(cfg).rglob("run-*"))) == 3 and sweep_json(cfg)["stopped"] == "budget"


def test_another_engine_build_is_refused(tmp_path, stub_model, monkeypatch):
    cfg = started_sweep(tmp_path, stub_model(gold=True))
    monkeypatch.setattr("mnemiq.sweep.run.engine_build", lambda *a, **k: {"commit": None, "source_sha": "f" * 64})
    with pytest.raises(SweepConfigError, match="engine"):
        run(cfg)


def test_a_second_run_on_the_same_out_folder_is_refused_at_once(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg = load_config(write_config(tmp_path, model=stub))
    out(cfg).mkdir(parents=True)
    with open(out(cfg) / ".sweep.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)   # as another run of the sweep holds it
        with pytest.raises(SweepConfigError, match="in use by another run"):
            run(cfg)
        with pytest.raises(SweepConfigError, match="in use by another run"):
            run(cfg, smoke=1)
    assert stub.calls == 0 and not (out(cfg) / "sweep.json").exists()


@pytest.mark.parametrize("strategy", ["screen-then-grid", "check"])
def test_a_model_that_does_not_repeat_runs_twice(tmp_path, stub_model, strategy):
    stub = stub_model(varies=True, gold_when=lambda request: False)   # wrong everywhere, spelled anew each time
    screen = {"arms_in_screen": 1} if strategy != "check" else {}
    cfg = roomy(tmp_path, stub, known=True, search={"strategy": strategy}, **screen)
    run(cfg, until="screen" if strategy != "check" else None)
    assert sweep_json(cfg)["repeats"] == 2 and every_first_round_arm_has_run(cfg, 2)


def test_every_arm_runs_once_before_any_runs_twice_with_the_starts_second_run_among_them(tmp_path, stub_model,
                                                                                          capsys):
    stub = stub_model(varies=True, gold_when=lambda request: False)
    cfg = roomy(tmp_path, stub, known=True, arms_in_screen=2)
    run(cfg, until="screen")
    ran = re.findall(r"^m, instant, screen: (\S+), run (\d): complete$", capsys.readouterr().out, re.MULTILINE)
    one, two = (f"retrieval_k={DEFAULT_RETRIEVAL_K + 4 * i}" for i in (1, 2))
    assert ran == [("start", "1"), (one, "1"), ("start", "2"), (two, "1"), (one, "2"), (two, "2")]


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


def test_a_model_with_no_room_stops_while_another_searches_on(tmp_path, stub_model):
    right, wrong = stub_model(gold=True), stub_model(gold_when=lambda request: False)
    cfg = roomy(tmp_path, right, known=True, arms_in_screen=1, models=priced(right, wrong))
    run(cfg, until="screen")
    (arm,) = [arm.profile_id for arm in screen_arms(cfg) if arm.model == "n" and arm.changes]
    assert arm_ids(cfg, "screen", "m") == {"start"} and arm_ids(cfg, "screen", "n") == {"start", arm}
    assert sweep_json(cfg)["room"]["m"]["instant"]["stopped"] == "no room"
    assert "stopped" not in sweep_json(cfg)["room"]["n"]["instant"] and "stopped" not in sweep_json(cfg)


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
    run(cfg, smoke=1)   # a folder a smoke made, which the next smoke replaces whole
    assert len(smoke_rows(cfg)) == 1


@pytest.mark.parametrize("found, refused", [("folder", "no mark of a smoke"), ("link", "symbolic link"),
                                            ("file", "not a folder")])
def test_a_smoke_leaves_a_smoke_folder_it_did_not_make_as_it_is(tmp_path, stub_model, found, refused):
    stub = stub_model(gold=True)
    cfg = load_config(write_config(tmp_path, model=stub))
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    (theirs / "notes.txt").write_text("not a smoke's")
    smoke = out(cfg) / "smoke"
    out(cfg).mkdir(parents=True)
    if found == "folder":
        theirs.rename(smoke)
        theirs = smoke
    elif found == "link":
        smoke.symlink_to(theirs, target_is_directory=True)
    else:
        smoke.write_text("not a smoke's")
    with pytest.raises(SweepConfigError, match=refused):
        run(cfg, smoke=1)
    assert (theirs / "notes.txt").read_text() == "not a smoke's" and stub.calls == 0
    assert smoke.is_symlink() if found == "link" else smoke.exists()


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


def test_a_resume_while_the_endpoint_is_still_down_loses_no_case(tmp_path, stub_model):
    model = stub_model(gold=True, fail_after=5, status=500)
    cfg = roomy(tmp_path, model, arms_in_screen=1)
    with pytest.raises(SystemExit):
        run(cfg, until="screen")
    with pytest.raises(SystemExit, match="An endpoint stopped answering"):
        run(cfg, until="screen")   # resumed too soon: the cases it held are held still, none kept as an error
    assert sweep_json(cfg)["stopped"] == "endpoint" and all(row["error_kind"] != "endpoint" for row in all_rows(cfg))
    assert not run_complete(cfg, start_of(cfg, "instant"), 1)
    model.recover()
    run(cfg, until="screen")
    assert all(arm_complete(cfg, a) for a in screen_arms(cfg))
    assert all(row["error_kind"] is None for row in all_rows(cfg))
    assert all(record["valid"] for record in report(cfg, "screen", controls=False)[1])


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
    assert sweep_json(cfg)["stopped"] == "worker" and not arm_folder(cfg, start, 2).exists()   # no other run started
    run(cfg, until="screen")
    ids = [r["id"] for r in read_rows(cfg, start)]
    assert written and len(ids) == len(set(ids)) == len(tuning(cfg)) and "stopped" not in sweep_json(cfg)


def test_a_workers_exception_is_named_in_the_exit(tmp_path, stub_model):
    cfg, start = one_arm(tmp_path, stub_model(gold=True))
    # The arm cannot write its profile file over a folder, and raises in its own process.
    (arm_folder(cfg, start, 1) / "profile.toml").mkdir(parents=True)
    with pytest.raises(SystemExit, match=r"stopped without finishing: IsADirectoryError: .*profile\.toml"):
        run(cfg, until="screen")
    assert sweep_json(cfg)["stopped"] == "worker"


# Read by each worker process at its start, from the folder a test puts first on its PYTHONPATH: the arm raises an error
# whose words, chained cause and note quote a source's password and the model's key, as a failed ATTACH can.
_RAISING_WORKER = """
import os

if os.environ.get("SWEEP_TEST_WORKER_RAISES"):
    import mnemiq.sweep.run as run

    def run_arm(*args, **kwargs):
        password, key = os.environ["SWEEP_TEST_PASSWORD"], os.environ["SWEEP_TEST_KEY"]
        try:
            raise ConnectionError(f"could not connect to postgresql://sweep:{password}@127.0.0.1:9/library ({key})")
        except ConnectionError as cause:
            error = RuntimeError(f"ATTACH failed: the source said {password} and the key {key} were refused, "
                                 "as was postgresql://reader:never-configured@10.0.0.9/db with password=also-unknown")
            error.add_note(f"a note that comes last: {password}")
            raise error from cause

    run.run_arm = run_arm
"""


def test_a_workers_error_never_shows_a_secret(tmp_path, stub_model, monkeypatch, capfd):
    password, key = "pg-pass-for-this-test", "model-key-for-this-test"
    stub = stub_model(gold=True)
    d = tiny_deployment(tmp_path, stub, env_extra={"MNEMIQ_PG_DSN": f"postgresql://sweep:{password}@127.0.0.1:9/lib"})
    cfg = load_config(write_config(tmp_path, deployment=d, model=stub))
    hook = tmp_path / "worker-hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_RAISING_WORKER)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(hook), str(Path(mnemiq.__file__).parents[1])]))
    monkeypatch.setenv("SWEEP_TEST_WORKER_RAISES", "1")
    monkeypatch.setenv("SWEEP_TEST_PASSWORD", password)
    monkeypatch.setenv(KEY_ENV, key)
    with pytest.raises(SystemExit) as stopped:
        run(cfg, until="screen")
    said, child = str(stopped.value), capfd.readouterr().err
    assert "stopped without finishing: RuntimeError: ATTACH failed" in said and "a note" not in said
    # The known secrets by their values, and the ones no setting holds by where they stand in a URL or a field.
    secrets = (password, key, "never-configured", "also-unknown")
    assert not [secret for secret in secrets if secret in said or secret in child]
    assert sweep_json(cfg)["stopped"] == "worker"


@pytest.mark.parametrize("varies, repeats", [(False, 1), (True, 2)], ids=["repeats", "does-not-repeat"])
def test_the_grid_combines_what_the_screen_kept_as_often_as_decided(tmp_path, stub_model, monkeypatch, varies,
                                                                    repeats):
    def graded(cfg, control):
        raise AssertionError("a control was graded to decide the grid")

    monkeypatch.setattr("mnemiq.sweep.report.grade_control", graded)
    # Either setting alone puts something in the writer's request that the stub answers right for.
    stub = stub_model(varies=varies, gold_when=lambda request: "response_format" in request or "seed" in request)
    space = {"guided_sql": [True], "llm_seed": [7]}
    answers = tmp_path / "other-system.json"
    answers.write_text(json.dumps([{"id": "after-1900", "declined": True}]))
    cfg = roomy(tmp_path, stub, known=True, space=space, machine={"parallel": 2},
                control=[{"name": "another-system", "answers": str(answers)}])
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

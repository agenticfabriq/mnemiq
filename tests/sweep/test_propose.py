"""The proposal: the arms that beat the start beyond noise, the finalists timed alone, one profile assembled from each
mode's pick and confirmed on the held-out part, check mode's verdict, and the files a person promotes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemiq.profiles import _read as read_profile_file
from mnemiq.sweep.config import SweepConfigError, load_config
from mnemiq.sweep.propose import candidates, pick, propose, server_running
from mnemiq.sweep.report import report

from .conftest import (
    KEY_ENV, STUB_KEY, UNREACHABLE, checked_sweep, confirm_arms_run, confirm_modes, evidence, graded_sweep,
    json_lines, out, play_after_search, proposal_file, rewrite, sweep_json, tuning,
)

# Settings a profile can change, each tried at one value other than the engine's default.
OTHERS = ["prompt_layout=question_last", "assertive_sql=true", "answer_markdown=true", "card_style=ddl", "llm_seed=7"]


def test_with_no_candidate_the_start_is_proposed(tmp_path):
    result = propose(graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=3, other=1)}))
    assert result["m"]["chosen"] == "start" and "beyond noise" in result["m"]["why"]


def test_holm_counts_the_screen_and_the_grid(tmp_path):
    # only=8 gives p 0.0078: it passes Holm over the 6 screen arms (0.0083) and fails over all 10 (0.005)
    arms = {"writer_rules=none": dict(only=8, other=0), **{label: dict(only=1, other=1) for label in OTHERS}}
    grid = {f"writer_rules=none + {label}": dict(only=2, other=1) for label in OTHERS[:4]}
    assert propose(graded_sweep(tmp_path, arms=arms, grid=grid))["m"]["chosen"] == "start"
    # The same screen with no grid beside it: the arm passes, so the grid's arms are what held it back.
    screen_only = graded_sweep(tmp_path / "screen-only", arms=arms)
    assert propose(screen_only)["m"]["chosen"] == "writer_rules=none"


def test_a_candidate_that_harms_on_held_out_gives_the_start_and_no_second_try(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0), "retrieval_k=36": dict(only=8, other=0)},
                       held_out={"assembled": dict(more_wrong=1)})
    assert candidates(cfg, report(cfg, "screen", controls=False)[1])[("m", "instant")] == [
        "writer_rules=none", "retrieval_k=36"]   # a second candidate was there to try
    assert propose(cfg)["m"]["chosen"] == "start" and confirm_arms_run(cfg) == 1
    assert evidence(cfg)["failed_mode"] == "instant" and evidence(cfg)["confirmed"] is False


def test_a_tie_goes_to_the_faster_arm_on_a_local_server(tmp_path):
    # Label order would pick retrieval_k=36: the time decides.
    cfg = graded_sweep(tmp_path, arms={"retrieval_k=36": dict(only=9, other=0, ms=4000),
                                       "writer_rules=none": dict(only=9, other=0, ms=2500)})
    assert propose(cfg)["m"]["picks"]["instant"] == "writer_rules=none"
    assert pick(cfg, "m", "instant").label == "writer_rules=none"


def test_a_tie_on_a_hosted_endpoint_goes_to_fewer_tokens(tmp_path):
    # The faster arm, and label order, would both pick retrieval_k=36: on a hosted endpoint the tokens decide.
    cfg = graded_sweep(tmp_path, hosted=True, arms={"retrieval_k=36": dict(only=9, other=0, ms=1000, prompt=900),
                                                    "writer_rules=none": dict(only=9, other=0, ms=3000, prompt=600)})
    assert propose(cfg)["m"]["picks"]["instant"] == "writer_rules=none"
    assert evidence(cfg)["timing"]["instant"]["indicative"] is True


def test_a_timing_reading_with_other_requests_on_the_server_is_not_taken(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, server_running=2)
    propose(cfg)
    assert evidence(cfg)["timing"]["instant"]["taken"] is False
    idle = graded_sweep(tmp_path / "idle", arms={"writer_rules=none": dict(only=9, other=0)})
    propose(idle)
    assert evidence(idle)["timing"]["instant"]["taken"] is True


def test_a_timing_reading_while_the_machine_is_loaded_is_not_taken(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, load=8.0)   # on 8 cores
    propose(cfg)
    assert evidence(cfg)["timing"]["instant"]["taken"] is False


def test_the_proposed_file_is_the_one_that_was_confirmed(tmp_path, stub_model):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, real_profiles=stub_model)
    result = propose(cfg)
    assert evidence(cfg)["version"] == evidence(cfg)["confirmed_arm_version"]
    # The version the runtime stamped on the confirm round's rows, read here from the rows themselves.
    (assembled,) = (out(cfg) / "m" / "confirm" / "instant").glob("assembled-*")
    assert {row["profile_version"] for row in json_lines(assembled / "run-1" / "rows.jsonl")} == {evidence(cfg)["version"]}
    assert result["m"]["chosen"] == "writer_rules=none" and evidence(cfg)["confirmed"] is True
    written = read_profile_file(str(proposal_file(cfg)))
    assert written.default == "m" and written.profiles["m"].writer_rules == "none"
    # Each timed case was checked first, in the worker that ran it.
    timed = min(20, len(tuning(cfg)))
    assert [reading["checks"]["cases"] for reading in evidence(cfg)["timing"]["instant"]["readings"]] == [timed, timed]


def test_two_modes_are_confirmed_as_one_profile(tmp_path):
    cfg = graded_sweep(tmp_path, modes=["instant", "thinking"],
                       picks={"instant": "writer_reasoning=before_sql", "thinking": "start"})
    propose(cfg)
    assert evidence(cfg)["conflicts"] == ["writer_reasoning"] and confirm_modes(cfg) == ["instant", "thinking"]


def test_the_confirm_round_runs_in_every_mode_when_the_first_had_no_room(tmp_path, stub_model):
    # The assembled arm and the start each run in both modes, their folder named for the first, which stopped searching.
    cfg = graded_sweep(tmp_path, modes=["instant", "thinking"], picks={"instant": "start", "thinking": "writer_rules=none"},
                       no_room=("instant",), real_profiles=stub_model)
    result = propose(cfg)
    assert confirm_modes(cfg) == ["instant", "thinking"] and evidence(cfg)["confirm"]["modes"] == ["instant", "thinking"]
    assert result["m"]["picks"] == {"instant": "start", "thinking": "writer_rules=none"}
    assert set(evidence(cfg)["numbers"]) == {"instant", "thinking"}
    assert all(each["held_out"][arm][0]["cases"] == 3 for each in evidence(cfg)["numbers"].values()
               for arm in ("start", "arm"))


def test_check_mode_proposes_nothing_when_an_arm_did_not_answer(tmp_path):
    cfg = checked_sweep(tmp_path, start=dict(errors=26), defaults=dict(right=10, wrong=3))
    assert propose(cfg)["m"]["chosen"] is None and "not valid" in propose(cfg)["m"]["why"]
    assert not proposal_file(cfg).exists()


def test_check_mode_proposes_the_start_only_when_it_does_no_harm(tmp_path):
    assert propose(checked_sweep(tmp_path / "a", start=dict(right=11, wrong=3),
                                 defaults=dict(right=10, wrong=3)))["m"]["chosen"] == "start"
    assert propose(checked_sweep(tmp_path / "b", start=dict(right=12, wrong=4),
                                 defaults=dict(right=10, wrong=3)))["m"]["chosen"] == "defaults"


def test_a_person_may_choose_and_it_is_recorded(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0), "retrieval_k=36": dict(only=2, other=0)})
    propose(cfg, chosen="retrieval_k=36")
    assert evidence(cfg)["chosen_by"] == "person" and evidence(cfg)["rule_pick"] == "writer_rules=none"
    assert evidence(cfg)["confirmed"] is False
    assert read_profile_file(str(proposal_file(cfg))).profiles["m"].retrieval_k == 36


# --- beyond the rule ----------------------------------------------------------------------------------------------


def test_a_person_may_choose_only_an_arm_that_ran_and_is_valid(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    with pytest.raises(SweepConfigError, match="retrieval_k=36"):
        propose(cfg, chosen="retrieval_k=36")


def test_the_timing_round_runs_alone_on_the_first_20_cases_to_tune_on(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, cases=(40, 36), play=False)
    assert len(tuning(cfg)) > 20
    rounds = play_after_search(cfg)
    timed = [(parallel, job) for parallel, jobs in rounds.drained for job in jobs if job.arm.round == "timing"]
    assert [job.arm.label for _, job in timed] == ["start", "writer_rules=none"]
    assert all(parallel == 1 and [case.id for case in job.cases] == [case.id for case in tuning(cfg)[:20]]
               for parallel, job in timed)


def test_a_proposal_whose_version_is_not_the_one_stamped_is_refused(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    (assembled,) = (out(cfg) / "m" / "confirm" / "instant").glob("assembled-*")
    rows = assembled / "run-1" / "rows.jsonl"
    rows.write_text("".join(json.dumps(row | {"profile_version": "0123456789ab"}) + "\n" for row in json_lines(rows)))
    with pytest.raises(SweepConfigError, match="version"):
        propose(cfg)
    assert not proposal_file(cfg).exists()


def test_a_sweep_with_no_room_anywhere_proposes_its_start(tmp_path):
    cfg = graded_sweep(tmp_path, arms={})
    state = sweep_json(cfg)
    del state["repeats"]   # the start's second run never happened, so the repeats were never decided
    state |= {"stopped": "no room", "room": {"m": {"instant": {"misses": 2, "needed": None, "compared": 4,
                                                                "stopped": "no room"}}}}
    (out(cfg) / "sweep.json").write_text(json.dumps(state))
    cfg = load_config(rewrite(cfg, sweep={"repeats": "auto"}))
    (out(cfg) / "sweep.json").write_text(json.dumps(state | {"fingerprint": _fingerprint(cfg)}))
    result = propose(cfg)
    assert result["m"]["chosen"] == "start" and "no room" in result["m"]["why"]
    assert [record["label"] for record in report(cfg, "screen", controls=False)[1]] == ["start"]


def test_a_proposal_is_refused_under_another_config(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    with pytest.raises(SweepConfigError, match="another config"):
        propose(load_config(rewrite(cfg, propose={"alpha": 0.1})))


def test_a_proposal_is_refused_before_the_sweep_has_finished(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, play=False)
    with pytest.raises(SweepConfigError, match="not finished"):
        propose(cfg)


def test_the_proposal_files_hold_no_key(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, propose={"profile": "tuned"})
    result = propose(cfg)
    written = [Path(result["m"]["profiles_file"]), Path(result["m"]["evidence_file"])]
    assert [path.name for path in written] == ["profiles.toml", "evidence.json"]
    assert all(STUB_KEY not in path.read_text() for path in written)
    assert f'api_key_env = "{KEY_ENV}"' in written[0].read_text()
    assert read_profile_file(str(written[0])).default == "tuned"


def test_a_proposal_over_two_modes_holds_each_modes_settings(tmp_path):
    cfg = checked_sweep(tmp_path, start=dict(right=11, wrong=3), defaults=dict(right=10, wrong=3),
                        modes=["instant", "thinking"])
    assert propose(cfg)["m"]["chosen"] == "start"
    written = read_profile_file(str(proposal_file(cfg))).profiles["m"]
    assert written.instant is not None and written.thinking is not None
    assert evidence(cfg)["measured"] == "mode by mode" and set(evidence(cfg)["measured_versions"]) == {"instant",
                                                                                                       "thinking"}


def test_a_server_publishes_its_running_requests_or_counts_as_hosted(stub_model):
    busy = stub_model()
    busy.running = 2
    assert server_running(busy.url) == 2
    assert server_running(stub_model().url) == 0
    assert server_running(UNREACHABLE) is None
    # A server with no metrics page at its root answers 404 there.
    assert server_running(busy.url.replace("/v1", "/elsewhere/v1")) is None


def _fingerprint(cfg) -> str:
    from mnemiq.sweep.config import fingerprint

    return fingerprint(cfg)

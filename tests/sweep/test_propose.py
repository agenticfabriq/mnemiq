"""The proposal: the arms that beat the start beyond noise, the finalists timed alone, one profile assembled from each
mode's pick and confirmed on the held-out part, check mode's verdict, and the files a person promotes."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from mnemiq.agent.modes import MODES
from mnemiq.profiles import _read as read_profile_file
from mnemiq.sweep import propose as proposal
from mnemiq.sweep.arms import arm_of, engine_refusal
from mnemiq.sweep.config import SweepConfigError, load_config
from mnemiq.sweep.propose import candidates, pick, propose, quiet_machine, server_running
from mnemiq.sweep.report import report
from mnemiq.sweep.run import _begin, _parts, arm_folder, run

from .conftest import (
    KEY_ENV, STUB_KEY, UNREACHABLE, checked_sweep, confirm_arms_run, confirm_modes, evidence, graded_rows,
    graded_sweep, json_lines, out, play_after_search, proposal_file, read_rows, rewrite, stamped_version, start_of,
    start_refused_in_instant, sweep_json, tiny_dataset, tiny_deployment, tuning, write_config, write_run,
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
    assert propose(cfg)["m"]["measured"] == "whole" and proposal_file(cfg).exists()   # an earlier proposal
    (assembled,) = (out(cfg) / "m" / "confirm" / "instant").glob("assembled-*")
    rows = assembled / "run-1" / "rows.jsonl"
    rows.write_text("".join(json.dumps(row | {"profile_version": "0123456789ab"}) + "\n" for row in json_lines(rows)))
    with pytest.raises(SweepConfigError, match="version"):
        propose(cfg)
    # Neither file is left to look current.
    assert not proposal_file(cfg).exists() and not (out(cfg) / "proposal" / "m" / "evidence.json").exists()


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


def test_an_arm_that_ran_beside_a_start_with_no_room_is_no_candidate(tmp_path):
    # The record says the search stopped for want of room. The arm that was running beside the start's first run
    # finished, and its numbers alone would pass the rule.
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, no_room=("instant",))
    result = propose(cfg)
    assert result["m"]["chosen"] == "start" and "no room" in result["m"]["why"]
    (arm,) = evidence(cfg)["candidates"]["instant"]["arms"]
    assert arm["label"] == "writer_rules=none" and arm["holm_passed"] == [True]   # it would have passed the rule
    assert arm["candidate"] is False and "want of room" in arm["why_not"][0]
    assert candidates(cfg, report(cfg, "screen", controls=False)[1]) == {("m", "instant"): []}
    # Nothing of that mode was timed or confirmed, and the evidence says so.
    assert evidence(cfg)["timing"] == {} and evidence(cfg)["confirm"] is None and evidence(cfg)["numbers"] is None
    assert not (out(cfg) / "m" / "timing").exists() and confirm_arms_run(cfg) == 0


def test_a_mode_that_ran_out_of_room_with_arms_running_beside_its_start_proposes_the_start(tmp_path, stub_model,
                                                                                          monkeypatch):
    monkeypatch.setattr(proposal, "LOAD_WAIT_S", 0.0)
    stub = stub_model(gold_when=lambda request: "response_format" in request)   # guided SQL is right, else wrong
    dataset = tiny_dataset(tmp_path, cases=10, answerable=9, stub=stub)
    # Four arms leave no room for a win on 7 answerable questions, and four at a time start beside the start's run.
    cfg = load_config(write_config(tmp_path, model=stub, dataset=dataset, machine={"parallel": 4},
                                   space={"guided_sql": [True], "retrieval_k": [16, 28, 36]}))
    run(cfg)
    assert sweep_json(cfg)["room"]["m"]["instant"]["stopped"] == "no room"
    assert propose(cfg)["m"]["chosen"] == "start"
    ran = evidence(cfg)["candidates"]["instant"]["arms"]
    assert "guided_sql=true" in [arm["label"] for arm in ran]   # an arm that gets every question right ran beside it
    assert not any(arm["candidate"] for arm in ran) and all("want of room" in arm["why_not"][0] for arm in ran)
    assert not (out(cfg) / "m" / "timing").exists() and confirm_arms_run(cfg) == 0
    assert evidence(cfg)["confirm"] is None and evidence(cfg)["timing"] == {}


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


# --- a start the engine refuses, the machine's load, the build, and what a person may choose ------------------------


def refused_in_instant(tmp_path, stub, *, only: int):
    """A two-mode sweep whose start sets min_agreement on instant's one candidate, which the engine refuses there and
    accepts in thinking, loaded. Thinking's screen is written as rows: a start that gets nothing right, and
    writer_rules=none `only` more right."""
    start = tmp_path / "start-profiles.toml"
    start.write_text('default = "tuned"\n[profiles.tuned]\n[profiles.tuned.instant]\nmin_agreement = 0.6\n')
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub), model=stub,
                                   dataset=tiny_dataset(tmp_path, cases=14, answerable=12, stub=stub),
                                   space={"writer_rules": ["none"]}, modes=["instant", "thinking"],
                                   sweep={"repeats": 1}, start={"profile_file": str(start)}))
    _begin(cfg)
    tuned = _parts(cfg)["tuning"]
    for changes, spec in (({}, None), ({"writer_rules": "none"}, {"only": only, "other": 0})):
        arm = arm_of("m", "thinking", changes, round="screen")
        write_run(cfg, arm, graded_rows(arm, tuned, right=0, version=stamped_version(cfg, arm), spec=spec))
    return cfg


def test_a_mode_whose_start_the_engine_refuses_is_left_out_of_the_profile_confirmed(tmp_path, stub_model):
    cfg = refused_in_instant(tmp_path, stub_model, only=9)
    run(cfg)   # ends normally: the confirm round runs where the start runs
    result = propose(cfg)
    assert result["m"]["chosen"] == "writer_rules=none" and evidence(cfg)["confirm"]["modes"] == ["thinking"]
    # What the first round's plan recorded, which the proposal reads.
    assert sweep_json(cfg)["starts"]["m"]["thinking"] is None and "min_agreement" in sweep_json(cfg)["starts"]["m"]["instant"]
    refused = evidence(cfg)["refused_modes"]["instant"]
    assert "min_agreement" in refused["why"] and refused["unmeasured_profile_wide"] == {"writer_rules": "none"}
    # Instant keeps the engine's per-mode settings, yet runs under the profile's writer_rules, never measured there.
    assert "writer_rules=none" in result["m"]["why"] and "never measured" in result["m"]["why"]
    written = read_profile_file(str(proposal_file(cfg))).profiles["m"]
    assert written.instant is None and written.thinking is not None and written.writer_rules == "none"


def test_with_no_candidate_the_start_is_proposed_where_the_engine_runs_it(tmp_path, stub_model):
    cfg = refused_in_instant(tmp_path, stub_model, only=0)
    run(cfg)
    result = propose(cfg)
    assert result["m"]["chosen"] == "start" and "the engine refuses the start" in result["m"]["why"]
    assert "not valid" not in result["m"]["why"]
    written = read_profile_file(str(proposal_file(cfg))).profiles["m"]
    assert written.instant is None and written.thinking is not None


def test_when_the_engine_refuses_every_start_the_run_ends_and_propose_says_why(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True), arms_in_screen=2,
                                   start=start_refused_in_instant(tmp_path)))   # instant is the sweep's one mode
    run(cfg)
    assert sweep_json(cfg)["finished"] is True and "repeats" not in sweep_json(cfg)
    result = propose(cfg)
    assert result["m"]["chosen"] is None and "refuses the start in every mode" in result["m"]["why"]
    assert "min_agreement" in result["m"]["why"] and not proposal_file(cfg).exists()
    # The report and the pick say that no start ran, not that the first round has yet to decide.
    with pytest.raises(SweepConfigError, match="no start ran"):
        report(cfg, "screen", controls=False)
    with pytest.raises(SweepConfigError, match="no start ran"):
        pick(cfg, "m", "instant")


def test_a_real_sweep_with_no_room_proposes_its_start(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(start_misses=2), arms_in_screen=4))
    run(cfg)
    assert sweep_json(cfg)["stopped"] == "no room" and "repeats" not in sweep_json(cfg)
    result = propose(cfg)
    assert result["m"]["chosen"] == "start" and "no room" in result["m"]["why"]
    assert evidence(cfg)["rule"]["repeats_decided"] is False


def test_the_wait_for_a_quiet_machine_ends_when_the_load_falls(capsys):
    loads, now = iter([9.0, 9.0, 3.0]), [0.0]
    quiet = quiet_machine(load=lambda: (next(loads), 8), clock=lambda: now[0],
                          sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert quiet is True and now[0] == 2 * proposal.LOAD_WAIT_EVERY_S and "waiting" in capsys.readouterr().out


def test_the_wait_for_a_quiet_machine_is_bounded(capsys):
    now = [0.0]
    quiet = quiet_machine(load=lambda: (9.0, 8), clock=lambda: now[0],
                          sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert quiet is False and proposal.LOAD_WAIT_S <= now[0] < proposal.LOAD_WAIT_S + proposal.LOAD_WAIT_EVERY_S
    assert "timing anyway" in capsys.readouterr().out


def test_the_timing_round_waits_for_the_machine_to_quieten_first(tmp_path, capsys):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, play=False)
    reads = iter([9.0, 9.0])   # the load the grid left, then an idle machine
    play_after_search(cfg, load=lambda: next(reads, 0.5), wait_s=60.0)
    propose(cfg)
    assert evidence(cfg)["timing"]["instant"]["taken"] is True and "waiting" in capsys.readouterr().out


def test_a_proposal_under_another_engine_build_is_packaged_with_a_warning(tmp_path, monkeypatch):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    ran_on = sweep_json(cfg)["engine_build"]
    monkeypatch.setattr(proposal, "engine_build", lambda *a, **k: {"commit": None, "source_sha": "f" * 64})
    result = propose(cfg)
    assert result["m"]["chosen"] == "writer_rules=none" and "engine build" in " ".join(result["m"]["warnings"])
    assert evidence(cfg)["sweep"]["engine_build"] == ran_on and evidence(cfg)["packaged_by"]["source_sha"] == "f" * 64


def test_a_person_may_choose_the_assembled_arm_after_its_confirm_failed(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)},
                       held_out={"assembled": dict(more_wrong=1)})
    propose(cfg, chosen="writer_rules=none")
    assert evidence(cfg)["rule_pick"] == "start" and evidence(cfg)["confirmed"] is False
    (source,) = evidence(cfg)["profile_from"]
    assert source["round"] == "confirm" and source["arm"].startswith("assembled-")


def test_the_evidence_says_when_the_assembled_arm_is_the_start(tmp_path):
    cfg = graded_sweep(tmp_path, modes=["instant", "thinking"],
                       picks={"instant": "writer_reasoning=before_sql", "thinking": "start"})
    result = propose(cfg)
    assert evidence(cfg)["confirm"]["same_as_start"] is True and "start's settings" in result["m"]["why"]


def test_a_metrics_page_with_an_endless_count_counts_as_none(stub_model):
    endless = stub_model()
    endless.running = float("inf")
    assert server_running(endless.url) is None


def test_a_report_reads_no_repeats_from_another_configs_sweep(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    cfg = load_config(rewrite(cfg, sweep={"repeats": "auto"}))   # its sweep.json holds the earlier config's record
    with pytest.raises(SweepConfigError, match="another config"):
        report(cfg, "screen", controls=False)


def _proposal_files(cfg, model: str = "m") -> dict[str, bytes]:
    folder = out(cfg) / "proposal" / model
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def test_the_note_on_refused_modes_reads_right_for_several(tmp_path):
    note = proposal._refused_note({"instant": "one reason", "deep": "another"}, {"writer_rules": "none",
                                                                                "assertive_sql": True})
    assert "instant and deep" in note and "those modes" in note and "each keeps the engine's own per-mode" in note
    assert "(assertive_sql=true, writer_rules=none) were never measured there" in note
    one = proposal._refused_note({"instant": "a reason"}, {})
    assert "that mode" in one and "it keeps" in one and "change nothing from the start" in one


def test_a_typo_in_a_persons_choice_leaves_the_earlier_proposal_as_it_was(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0), "retrieval_k=36": dict(only=2, other=0)})
    propose(cfg, chosen="retrieval_k=36")
    before = _proposal_files(cfg)
    with pytest.raises(SweepConfigError, match="nnoe"):
        propose(cfg, chosen="writer_rules=nnoe")
    assert _proposal_files(cfg) == before


def test_an_edited_config_leaves_the_earlier_proposal_as_it_was(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    propose(cfg)
    before = _proposal_files(cfg)
    with pytest.raises(SweepConfigError, match="another config"):
        propose(load_config(rewrite(cfg, propose={"alpha": 0.1})))
    assert _proposal_files(cfg) == before


def test_an_unset_key_variable_at_propose_time_is_named_and_changes_nothing(tmp_path, monkeypatch):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    propose(cfg)
    before = _proposal_files(cfg)
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(SweepConfigError, match=KEY_ENV) as refused:
        propose(cfg)
    assert "refuses the start" not in str(refused.value) and _proposal_files(cfg) == before


def test_what_ran_is_read_from_the_record_not_checked_again(tmp_path):
    cfg = graded_sweep(tmp_path, modes=["instant", "thinking"], arms={"writer_rules=none": dict(only=9, other=0)},
                       play=False)
    # Today's engine runs the start in both modes, and rows of both are on disk. The record, written when the first
    # round was planned, says the engine refused thinking's start then: what ran is what the record says.
    assert all(engine_refusal(cfg, start_of(cfg, mode)) is None for mode in cfg.modes)
    state = sweep_json(cfg)
    state["starts"]["m"]["thinking"] = "refused by the build the first round ran on"
    (out(cfg) / "sweep.json").write_text(json.dumps(state))
    play_after_search(cfg)
    result = propose(cfg)["m"]
    assert result["picks"] == {"instant": "writer_rules=none"} and confirm_modes(cfg) == ["instant"]
    assert evidence(cfg)["refused_modes"]["thinking"]["why"] == "refused by the build the first round ran on"
    assert read_profile_file(str(proposal_file(cfg))).profiles["m"].thinking is None


def test_a_search_whose_start_is_not_valid_proposes_nothing(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=3, other=1)}, play=False)
    start = arm_of("m", "instant", {}, round="screen")
    rows = read_rows(cfg, start)
    rows[0] |= {"outcome": "error", "error_kind": "endpoint", "model_error": "status_503"}   # one case not measured
    write_run(cfg, start, rows)
    play_after_search(cfg)
    result = propose(cfg)["m"]
    assert result["chosen"] is None and "nothing can be compared with it" in result["why"]
    assert "run 1 did not measure 1 cases" in result["why"]
    assert evidence(cfg)["start_valid"] is False and not proposal_file(cfg).exists()


def test_a_start_out_of_room_under_two_fixed_runs_is_read_at_the_one_run_it_owed(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model(start_misses=2), arms_in_screen=4,
                                   sweep={"repeats": 2}))
    run(cfg)
    # The search stopped after the start's first run, so its second never ran, on purpose.
    assert sweep_json(cfg)["room"]["m"]["instant"]["stopped"] == "no room"
    assert not arm_folder(cfg, start_of(cfg, "instant"), 2).exists()
    result = propose(cfg)["m"]
    assert result["chosen"] == "start" and "no room" in result["why"] and "not measured" not in result["why"]
    assert evidence(cfg)["start_valid"] is True and proposal_file(cfg).exists()
    assert [each["run"] for each in evidence(cfg)["candidates"]["instant"]["start"]] == [1]
    # The report reads it the same way: one run, valid, and no column for a run it never owed.
    table, records = report(cfg, "screen", controls=False)
    start = next(record for record in records if record["arm"] == "start")
    assert start["valid"] is True and [each["run"] for each in start["runs"]] == [1] and "Run 2" not in table


def test_a_model_out_of_room_beside_one_that_decided_two_runs_proposes_its_start(tmp_path, stub_model, monkeypatch):
    monkeypatch.setattr(proposal, "LOAD_WAIT_S", 0.0)
    right, wrong = stub_model(gold=True), stub_model(varies=True, gold_when=lambda request: False)
    models = {name: {"llm_model": "stub-model", "llm_base_url": stub.url, "api_key_env": KEY_ENV}
              for name, stub in zip("mn", (right, wrong))}
    cfg = load_config(write_config(tmp_path, model=right, models=models, arms_in_screen=1,
                                   dataset=tiny_dataset(tmp_path, cases=10, answerable=9, stub=right)))
    run(cfg)
    # m's start gets every question right, so m has no room; n's start does not repeat itself, so every arm owes two
    # runs, and m's start, its search stopped, ran one.
    assert sweep_json(cfg)["repeats"] == 2 and sweep_json(cfg)["room"]["m"]["instant"]["stopped"] == "no room"
    result = propose(cfg)["m"]
    assert result["chosen"] == "start" and "not measured" not in result["why"]
    assert evidence(cfg, "m")["start_valid"] is True and proposal_file(cfg, "m").exists()


def test_a_failed_confirm_over_a_start_not_valid_says_which_start(tmp_path):
    cfg = graded_sweep(tmp_path, modes=["instant", "thinking"], arms={"writer_rules=none": dict(only=9, other=0)},
                       held_out={"assembled": dict(more_wrong=1)}, play=False)
    start = arm_of("m", "thinking", {}, round="screen")
    rows = read_rows(cfg, start)
    rows[0] |= {"outcome": "error", "error_kind": "endpoint", "model_error": "status_503"}   # one case not measured
    write_run(cfg, start, rows)
    play_after_search(cfg, held_out={"assembled": dict(more_wrong=1)})
    result = propose(cfg)["m"]
    assert result["chosen"] is None and evidence(cfg)["failed_mode"] is not None
    assert "the confirm round's start" in result["why"] and "questions to tune on" in result["why"]
    assert "in thinking, run 1 did not measure 1 cases" in result["why"]
    assert "the start it falls back to is not valid" not in result["why"]


@pytest.mark.parametrize("fault", ["errors", "invalid"])
def test_a_persons_choice_is_refused_where_the_start_it_would_hold_is_not_valid(tmp_path, fault):
    cfg = graded_sweep(tmp_path, modes=["instant", "thinking"], arms={"writer_rules=none": dict(only=9, other=0)},
                       play=False)
    # An arm of instant's own setting, valid where it ran: chosen, the profile holds the start's section elsewhere.
    mine = arm_of("m", "instant", {"instant.prose": False}, round="screen")
    write_run(cfg, mine, graded_rows(mine, _parts(cfg)["tuning"], right=0, version=stamped_version(cfg, mine),
                                     spec=dict(only=2, other=0)))
    start = arm_of("m", "thinking", {}, round="screen")
    if fault == "errors":
        rows = read_rows(cfg, start)
        rows[0] |= {"outcome": "error", "error_kind": "endpoint", "model_error": "status_503"}   # over max_errors 0
        write_run(cfg, start, rows)
    else:   # a file it pins moved
        (arm_folder(cfg, start, 1) / "invalid.json").write_text(json.dumps({"changed": ["the store"]}))
    play_after_search(cfg)
    propose(cfg)
    before = _proposal_files(cfg)
    with pytest.raises(SweepConfigError, match="thinking") as refused:
        propose(cfg, chosen="instant.prose=false")
    assert "not valid" in str(refused.value) and _proposal_files(cfg) == before


def test_the_evidence_names_the_modes_the_sweep_did_not_search(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    result = propose(cfg)["m"]
    unsearched = [mode for mode in MODES if mode != "instant"]
    assert sorted(evidence(cfg)["unsearched_modes"]) == sorted(unsearched)
    assert all(entry["unmeasured_profile_wide"] == {"writer_rules": "none"}
               for entry in evidence(cfg)["unsearched_modes"].values())
    said = result["why"].split("did not search", 1)[1]
    assert " and ".join(unsearched) in result["why"] and "writer_rules=none" in said and "never measured" in said


def test_a_mode_never_searched_names_the_section_of_the_start_the_profile_drops(tmp_path):
    start = tmp_path / "start-profiles.toml"
    start.write_text('default = "start"\n[profiles.start]\nretrieval_k = 12\n'
                     '[profiles.start.thinking]\nmax_attempts = 2\n')
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)}, start={"profile_file": str(start)})
    result = propose(cfg)["m"]
    unsearched = evidence(cfg)["unsearched_modes"]
    assert unsearched["thinking"]["dropped_start_section"] == {"max_attempts": 2}
    assert unsearched["deep"]["dropped_start_section"] == {}
    # Promoting the profile moves thinking from the start's section to the engine's settings, and the reason says so.
    assert read_profile_file(str(proposal_file(cfg))).profiles["m"].thinking is None
    said = result["why"].split("did not search", 1)[1]
    assert "start's thinking section (max_attempts=2)" in said and "engine's own" in said


def test_a_proposal_waits_for_no_run_and_no_other_proposal_of_the_same_sweep(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    propose(cfg)
    before = _proposal_files(cfg)
    with open(out(cfg) / ".sweep.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)   # as a run, or another proposal, of the sweep holds it
        with pytest.raises(SweepConfigError, match="in use"):
            propose(cfg)
    assert _proposal_files(cfg) == before
    assert propose(cfg)["m"]["chosen"] == "writer_rules=none"   # once it is free


def test_a_model_that_ran_nothing_does_not_block_a_persons_choice_for_another(tmp_path, stub_model):
    stub = stub_model(gold_when=lambda request: False)   # wrong everywhere, so m's start leaves room to search
    models = {name: {"llm_model": "stub-model", "llm_base_url": stub.url, "api_key_env": KEY_ENV} for name in "mn"}
    # n's own start, in instant, the sweep's one mode, is one the engine refuses, so nothing of n runs.
    cfg = load_config(write_config(tmp_path, model=stub, models=models, space={"writer_rules": ["none"]},
                                   dataset=tiny_dataset(tmp_path, cases=10, answerable=9, stub=stub),
                                   sweep={"repeats": 1}, start={"n": start_refused_in_instant(tmp_path)}))
    run(cfg)
    result = propose(cfg, chosen="writer_rules=none")
    assert result["m"]["chosen"] == "writer_rules=none" and result["m"]["chosen_by"] == "person"
    assert result["n"]["chosen"] is None and "refuses the start in every mode" in result["n"]["why"]
    assert proposal_file(cfg, "m").exists() and not proposal_file(cfg, "n").exists()


def test_a_write_interrupted_part_way_leaves_no_proposal_files(tmp_path, monkeypatch):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    propose(cfg)

    def interrupted(path, value):
        raise KeyboardInterrupt   # as Ctrl-C lands after the profile is written and before its evidence

    monkeypatch.setattr(proposal, "_write_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        propose(cfg)
    assert not (out(cfg) / "proposal" / "m").exists() or not list((out(cfg) / "proposal" / "m").iterdir())


def test_a_mode_once_run_stays_recorded_when_a_later_plan_refuses_it(tmp_path, stub_model, monkeypatch):
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True), sweep={"repeats": 1}))
    run(cfg)
    assert sweep_json(cfg)["starts"] == {"m": {"instant": None}}
    # The next run's plan finds the engine refusing the start: a stray MNEMIQ_PROFILE in the shell refuses every
    # profile, and changes nothing the sweep's record checks.
    monkeypatch.setenv("MNEMIQ_PROFILE", "elsewhere")
    assert engine_refusal(cfg, start_of(cfg, "instant")) is not None
    run(cfg)
    assert sweep_json(cfg)["starts"] == {"m": {"instant": None}}


def test_a_record_from_before_starts_were_kept_points_to_a_new_sweep(tmp_path):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    propose(cfg)
    before = _proposal_files(cfg)
    state = sweep_json(cfg)
    del state["starts"]   # as a build that did not keep which modes ran wrote it
    (out(cfg) / "sweep.json").write_text(json.dumps(state))
    with pytest.raises(SweepConfigError, match="start a new sweep in another out folder") as refused:
        propose(cfg)
    assert "Run the sweep again" not in str(refused.value) and _proposal_files(cfg) == before


@pytest.mark.parametrize("damage", ["missing", "unparseable"])
def test_an_arm_file_that_cannot_be_read_withdraws_the_earlier_proposal(tmp_path, damage):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    propose(cfg)
    (assembled,) = (out(cfg) / "m" / "confirm" / "instant").glob("assembled-*")
    profile = assembled / "run-1" / "profile.toml"
    if damage == "missing":
        profile.unlink()
    else:
        profile.write_text("default = \n[profiles.\n")
    with pytest.raises(SweepConfigError, match="profile"):
        propose(cfg)
    assert not proposal_file(cfg).exists() and not (out(cfg) / "proposal" / "m" / "evidence.json").exists()

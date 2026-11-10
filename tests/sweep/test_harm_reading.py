"""Check mode's harm rule read worst against worst (`[propose] harm = "worst"`): each arm's highest wrong and made-up
counts over its runs compared, where the default reading (`"per_run"`) compares run k with run k. The option is check
mode's alone, and its absence reads, fingerprints and is written exactly as before it existed, in the config, the
proposal's evidence and a starter's evidence."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnemiq import profiles
from mnemiq.cli import main
from mnemiq.sweep import config as sweep_config
from mnemiq.sweep.arms import check_arms
from mnemiq.sweep.config import SweepConfig, SweepConfigError, fingerprint, fingerprint_body, load_config
from mnemiq.sweep.propose import _check, propose
from mnemiq.sweep.run import _begin, _parts, engine_build
from mnemiq.sweep.starters import RuleRecord, StarterEvidence, publish_starter, read_evidence

from .conftest import (
    _START_MARK, CONFIG_BUILD, KEY_ENV, STUB_WINDOW, UNREACHABLE, _start_profile, checked_rows, evidence,
    evidence_file, finished_check, idle_deployment, one_model, out, play_after_search, read_json, rewrite,
    stamped_version, start_that_differs, starter_folder, tiny_dataset, tiny_deployment, tiny_files, valid_evidence,
    write, write_config, write_run,
)
from .stub_model import _WRITER, _messages, _proposal

REPO = Path(__file__).resolve().parents[2]
# What `propose` wrote on main, before `[propose] harm`, for `PER_RUN_FIXTURE`'s check: the tmp path, the engine build
# and the fingerprint written as placeholders, every other byte as it was.
PER_RUN_EVIDENCE = REPO / "tests" / "fixtures" / "sweep" / "per-run-check-evidence.json"
PER_RUN_FIXTURE = {"start": [dict(right=11, wrong=3, made_up=1), dict(right=11, wrong=4, made_up=1)],
                   "defaults": [dict(right=10, wrong=5, made_up=1), dict(right=10, wrong=3, made_up=0)]}
# The config part of `_pinned_config`'s fingerprint body, and the sha256 of the whole body with each deployment's
# settings held fixed, both as main wrote them before `[propose] harm`, the tmp path written as "<tmp>".
CHECK_CONFIG_DUMP = (
    '{"dataset": {"deployments": null, "env": "<tmp>/deployment/deployment.env", "held_out": 0.3, "identity": '
    '{"principal": "sweep", "roles": ["sweep-test"], "tenant": "local"}, "questions": '
    '"<tmp>/deployment/questions.json", "split_seed": 7}, "judge": null, "machine": {"parallel": 1}, "models": {"m": '
    '{"api_key_env": "SWEEP_TEST_KEY", "llm_base_url": "http://127.0.0.1:9/v1", "llm_context_window": null, '
    '"llm_model": "stub-model", "price_in_per_mtok": null, "price_out_per_mtok": null, "verify_api_key_env": null, '
    '"verify_base_url": null, "verify_model": null}}, "modes": ["instant"], "propose": {"alpha": 0.05, '
    '"max_more_made_up": 0, "max_more_wrong": 0, "profile": null}, "search": {"screen_alpha": 0.25, "strategy": '
    '"check"}, "space": {}, "start": {"per_model": {}, "profile": "start", "profile_file": '
    '"<tmp>/start-profiles.toml"}, "sweep": {"max_errors": 0, "name": "tiny", "out": "<tmp>/sweep-out", "repeats": 2}}'
)
CHECK_BODY_SHA256 = "8a1adce84b942e2a7fdd5d1095ba6f855f2437fc556cbfcab017691c6702b15e"
_TALLIED = ("right", "strict", "wrong", "made_up", "declined", "failed", "errors", "ungraded", "unjudged")


# --- helpers ----------------------------------------------------------------------------------------------------------


def checked(**propose) -> SweepConfig:
    """A check-mode config for the parts that read no file, with `propose` as its `[propose]`."""
    return SweepConfig.model_validate({
        "sweep": {"name": "pure", "out": "sweep-out", "repeats": 2},
        "dataset": {"questions": "questions.json", "env": "deployment.env", "split_seed": 7,
                    "identity": {"roles": ["sweep-test"]}},
        "models": {"m": {"llm_model": "stub-model", "llm_base_url": UNREACHABLE, "api_key_env": KEY_ENV}},
        "modes": ["instant", "thinking"], "search": {"strategy": "check"}, "propose": propose,
    })


def tally(run: int, wrong: int, made_up: int = 0, *, right: int = 10, errors: int = 0) -> dict:
    """One run of an arm in one mode as `report` tallies it: 26 questions, each counted once, every one not right,
    wrong, made up or an error declined."""
    counts = dict.fromkeys(_TALLIED, 0) | {"right": right, "strict": right, "wrong": wrong, "made_up": made_up,
                                           "errors": errors}
    counts["declined"] = 26 - right - wrong - made_up - errors
    return {"run": run, "cases": 26, **counts, "valid": errors == 0}


def records(spec: dict) -> list[dict]:
    """Check mode's records for model `m`, as `report` gives them: each mode to its start's runs and its defaults' runs,
    each run a `(wrong, made_up)` pair or a dict for `tally`."""
    found = []
    for mode, (start, defaults) in spec.items():
        for arm, runs in (("start", start), ("defaults", defaults)):
            tallied = [tally(number, **run) if isinstance(run, dict) else tally(number, *run)
                       for number, run in enumerate(runs, start=1)]
            found.append({"model": "m", "mode": mode, "arm": arm, "valid": all(run["valid"] for run in tallied),
                          "runs": tallied})
    return found


def verdict(spec: dict, **propose) -> dict:
    """Check mode's verdict for model `m` on `spec` (`records`), in each of its modes, under `[propose]` `propose`."""
    return _check(checked(**propose), "m", records(spec), list(spec), {})


def checked_runs(tmp_path: Path, *, start: list[dict], defaults: list[dict], modes=("instant", "thinking"),
                 propose: dict | None = None):
    """A check-mode sweep of model `m`, already run, written as rows, over 26 synthetic questions, 17 of them with an
    answer, loaded: one run of each arm for each dict of `start` and `defaults` (`checked_rows`), in every mode of
    `modes`, under `[propose]` `propose`."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    sections = {"propose": propose} if propose is not None else {}
    cfg = load_config(write_config(tmp_path, deployment=idle_deployment(tmp_path), modes=list(modes),
                                   dataset=tiny_dataset(tmp_path, cases=26, answerable=17),
                                   search={"strategy": "check"}, sweep={"repeats": len(start)},
                                   start=start_that_differs(tmp_path), **sections))
    state = _begin(cfg) | {"starts": {"m": {mode: None for mode in modes}}}   # as the first round's plan records it
    (out(cfg) / "sweep.json").write_text(json.dumps(state))
    for arm in check_arms(cfg):
        for repeat, spec in enumerate(start if arm.profile_id == "start" else defaults, start=1):
            write_run(cfg, arm, checked_rows(arm, _parts(cfg)["tuning"], spec, stamped_version(cfg, arm)), repeat)
    play_after_search(cfg)
    return cfg


def _placeheld(text: str, tmp_path: Path, cfg) -> str:
    """The proposal's evidence with what moves from run to run written as `PER_RUN_EVIDENCE` writes it."""
    build = engine_build()
    text = text.replace(str(tmp_path), "<tmp>").replace(build["source_sha"], "<source_sha>")
    if build["commit"]:
        text = text.replace(build["commit"], "<commit>")
    return text.replace(json.loads((out(cfg) / "sweep.json").read_text())["fingerprint"], "<fingerprint>")


def _pinned_config(tmp_path: Path):
    """A check-mode config with no `harm` key, over the tiny deployment's files, its source and store given fixed bytes:
    a fingerprint hashes them and reads nothing else of them."""
    d = tiny_files(tmp_path)
    d.source.write_bytes(b"a fixed source\n")
    d.store.write_bytes(b"a fixed store\n")
    return load_config(write_config(tmp_path, deployment=d, search={"strategy": "check"}, sweep={"repeats": 2},
                                    start=start_that_differs(tmp_path)))


# --- the config -------------------------------------------------------------------------------------------------------


def test_the_harm_reading_defaults_to_per_run():
    assert checked().propose.harm == "per_run"


def test_worst_loads_in_check_mode(tmp_path):
    cfg = load_config(write_config(tmp_path, search={"strategy": "check"}, start=start_that_differs(tmp_path),
                                   sweep={"repeats": 2}, propose={"harm": "worst"}))
    assert cfg.propose.harm == "worst"


@pytest.mark.parametrize("strategy", ["screen-then-grid", "grid"])
def test_worst_outside_check_mode_is_refused_naming_the_key(tmp_path, strategy):
    with pytest.raises(SweepConfigError, match=r'\[propose\] harm = "worst"'):
        load_config(write_config(tmp_path, search={"strategy": strategy}, propose={"harm": "worst"}))


@pytest.mark.parametrize("repeats", [None, "auto", 1])
def test_worst_with_fewer_than_two_fixed_runs_is_refused_naming_both_keys(tmp_path, repeats):
    # "auto" runs every arm once when the start writes the same SQL twice, as a start at temperature 0 does
    with pytest.raises(SweepConfigError, match=r'\[propose\] harm = "worst".*\[sweep\] repeats'):
        load_config(write_config(tmp_path, search={"strategy": "check"}, start=start_that_differs(tmp_path),
                                 sweep={} if repeats is None else {"repeats": repeats}, propose={"harm": "worst"}))


def test_an_unknown_harm_reading_is_refused(tmp_path):
    with pytest.raises(SweepConfigError, match=r"propose\.harm: .*'per_run' or 'worst'"):
        load_config(write_config(tmp_path, search={"strategy": "check"}, start=start_that_differs(tmp_path),
                                 propose={"harm": "median"}))


def test_a_check_config_without_harm_fingerprints_as_before(tmp_path, monkeypatch):
    cfg = _pinned_config(tmp_path)
    body = fingerprint_body(cfg)
    assert json.dumps(body["config"], sort_keys=True).replace(str(tmp_path), "<tmp>") == CHECK_CONFIG_DUMP
    # The whole body, each deployment's settings held fixed: a setting the engine gains moves every fingerprint alike,
    # and is no part of what this pins.
    monkeypatch.setattr(sweep_config, "_answering_settings", lambda settings: {"held": "fixed"})
    text = json.dumps(fingerprint_body(cfg), sort_keys=True, separators=(",", ":"))
    assert fingerprint(cfg) == hashlib.sha256(text.encode()).hexdigest()
    assert hashlib.sha256(text.replace(str(tmp_path), "<tmp>").encode()).hexdigest() == CHECK_BODY_SHA256


def test_the_worst_reading_is_part_of_the_fingerprint(tmp_path):
    per_run = _pinned_config(tmp_path)
    worst = load_config(rewrite(per_run, propose={"harm": "worst"}))
    assert worst.propose.harm == "worst" and fingerprint(worst) != fingerprint(per_run)


def test_a_finished_per_run_check_is_never_read_under_worst(tmp_path):
    cfg = checked_runs(tmp_path, start=[dict(right=11, wrong=3)] * 2, defaults=[dict(right=10, wrong=3)] * 2,
                       modes=("instant",))
    with pytest.raises(SweepConfigError, match="this sweep was run under another config"):
        propose(load_config(rewrite(cfg, propose={"harm": "worst"})))


# --- the verdict under "worst" ----------------------------------------------------------------------------------------


def test_a_start_whose_worst_equals_the_defaults_worst_is_proposed():
    found = verdict({"instant": ([(3, 0), (5, 0)], [(5, 0), (4, 0)])}, harm="worst")
    assert found["rule"] == "start"


def test_a_start_whose_worst_is_one_above_the_defaults_worst_is_refused():
    found = verdict({"instant": ([(3, 0), (6, 0)], [(5, 0), (4, 0)])}, harm="worst")
    assert found["rule"] == "defaults"


def test_made_up_answers_are_read_worst_against_worst_too():
    assert verdict({"instant": ([(3, 2), (3, 1)], [(3, 1), (3, 2)])}, harm="worst")["rule"] == "start"
    assert verdict({"instant": ([(3, 3), (3, 1)], [(3, 1), (3, 2)])}, harm="worst")["rule"] == "defaults"


def test_a_start_worse_in_one_run_whose_worst_is_not_above_the_defaults_is_proposed_under_worst_alone():
    # run 2: 4 wrong against 2, which the per-run reading refuses; worst against worst, 4 is not above 5
    spec = {"instant": ([(4, 0), (4, 0)], [(5, 0), (2, 0)])}
    assert verdict(spec, harm="worst")["rule"] == "start"
    assert verdict(spec)["rule"] == "defaults"


def test_max_more_wrong_allows_that_many_more_at_the_worst():
    assert verdict({"instant": ([(6, 0), (6, 0)], [(5, 0), (4, 0)])}, harm="worst", max_more_wrong=1)["rule"] == "start"
    assert verdict({"instant": ([(7, 0), (6, 0)], [(5, 0), (4, 0)])}, harm="worst",
                   max_more_wrong=1)["rule"] == "defaults"


def test_made_up_answers_are_held_to_their_own_bound():
    # one more made-up answer at the worst, with room for one more wrong one only
    found = verdict({"instant": ([(3, 2), (3, 2)], [(3, 1), (3, 1)])}, harm="worst", max_more_wrong=1)
    assert found["rule"] == "defaults"
    assert verdict({"instant": ([(3, 2), (3, 2)], [(3, 1), (3, 1)])}, harm="worst",
                   max_more_made_up=1)["rule"] == "start"


def test_one_mode_over_at_the_worst_refuses_the_start():
    found = verdict({"instant": ([(3, 0), (3, 0)], [(3, 0), (4, 0)]), "thinking": ([(2, 0), (5, 0)], [(4, 0), (4, 0)])},
                    harm="worst")
    assert found["rule"] == "defaults"


def test_the_why_names_the_reading_and_each_modes_highest_counts():
    spec = {"instant": ([(3, 1), (4, 0)], [(5, 1), (2, 0)]), "thinking": ([(2, 0), (6, 0)], [(4, 0), (5, 0)])}
    refused = verdict(spec, harm="worst")
    assert refused["rule"] == "defaults"
    assert refused["why"] == (
        'The defaults are proposed: read worst against worst ([propose] harm = "worst"), the start\'s highest wrong or '
        "made-up count over its runs exceeds the defaults' highest by more than [propose] allows in thinking "
        "(max_more_wrong 0, max_more_made_up 0). The highest counts, the start's against the defaults': in instant, "
        "4 wrong against 5 and 1 made-up against 1; in thinking, 6 wrong against 5 and 0 made-up against 0.")
    proposed = verdict(spec, harm="worst", max_more_wrong=1)
    assert proposed["rule"] == "start"
    assert proposed["why"] == (
        'The start is proposed: read worst against worst ([propose] harm = "worst"), in every mode the start\'s '
        "highest wrong and made-up counts over its runs exceed the defaults' highest by no more than [propose] allows. "
        "The highest counts, the start's against the defaults': in instant, 4 wrong against 5 and 1 made-up against "
        "1; in thinking, 6 wrong against 5 and 0 made-up against 0. Both arms' right answers are in the evidence.")


def test_worst_records_each_arms_highest_counts_beside_the_per_run_comparison():
    found = verdict({"instant": ([(3, 1), (4, 0)], [(5, 1), (2, 2)])}, harm="worst")
    assert found["check"]["instant"]["highest"] == {"start": {"wrong": 4, "made_up": 1},
                                                    "defaults": {"wrong": 5, "made_up": 2}}
    assert found["check"]["instant"]["start_against_defaults"] == [
        {"run": 1, "more_wrong": -2, "more_made_up": 0}, {"run": 2, "more_wrong": 2, "more_made_up": -2}]


def test_worst_still_proposes_nothing_unless_both_arms_are_valid():
    found = verdict({"instant": ([dict(wrong=3, errors=2), dict(wrong=3)], [(5, 0), (5, 0)])}, harm="worst")
    assert found["rule"] is None and "Nothing is proposed: the start is not valid in instant" in found["why"]


# --- the verdict under "per_run", as main gave it -------------------------------------------------------------------

# Each record set, with the verdict, the reason and the per-run comparison main's `_check` gave it under the default.
PER_RUN_VERDICTS = [
    ({"instant": ([(3, 1), (4, 1)], [(5, 1), (3, 0)]), "thinking": ([(2, 0), (2, 0)], [(2, 0), (2, 0)])},
     "defaults",
     "The defaults are proposed: in run 2 of instant the start gives 1 more wrong and 1 more made-up answers than the "
     "defaults, beyond what [propose] allows (max_more_wrong 0, max_more_made_up 0).",
     {"instant": [{"run": 1, "more_wrong": -2, "more_made_up": 0}, {"run": 2, "more_wrong": 1, "more_made_up": 1}],
      "thinking": [{"run": 1, "more_wrong": 0, "more_made_up": 0}, {"run": 2, "more_wrong": 0, "more_made_up": 0}]}),
    ({"instant": ([(3, 2), (3, 1)], [(3, 1), (3, 2)]), "thinking": ([(2, 0), (2, 0)], [(2, 0), (2, 0)])},
     "defaults",
     "The defaults are proposed: in run 1 of instant the start gives 0 more wrong and 1 more made-up answers than the "
     "defaults, beyond what [propose] allows (max_more_wrong 0, max_more_made_up 0).",
     {"instant": [{"run": 1, "more_wrong": 0, "more_made_up": 1}, {"run": 2, "more_wrong": 0, "more_made_up": -1}],
      "thinking": [{"run": 1, "more_wrong": 0, "more_made_up": 0}, {"run": 2, "more_wrong": 0, "more_made_up": 0}]}),
    ({"instant": ([(3, 1), (3, 1)], [(5, 1), (3, 1)]), "thinking": ([(2, 0), (2, 0)], [(2, 0), (2, 0)])},
     "start",
     "The start is proposed: in every mode and run its wrong and made-up answers exceed the defaults' by no more than "
     "[propose] allows. Both arms' right answers are in the evidence.",
     {"instant": [{"run": 1, "more_wrong": -2, "more_made_up": 0}, {"run": 2, "more_wrong": 0, "more_made_up": 0}],
      "thinking": [{"run": 1, "more_wrong": 0, "more_made_up": 0}, {"run": 2, "more_wrong": 0, "more_made_up": 0}]}),
    ({"instant": ([dict(wrong=3, errors=2), dict(wrong=3)], [(5, 1), (3, 1)]),
      "thinking": ([(2, 0), (2, 0)], [(2, 0), (2, 0)])},
     None,
     "Nothing is proposed: the start is not valid in instant (run 1 did not measure 2 cases, more than max_errors 0). "
     "Check mode proposes only between two arms valid in every run.",
     {"instant": [{"run": 1, "more_wrong": -2, "more_made_up": -1}, {"run": 2, "more_wrong": 0, "more_made_up": -1}],
      "thinking": [{"run": 1, "more_wrong": 0, "more_made_up": 0}, {"run": 2, "more_wrong": 0, "more_made_up": 0}]}),
]


@pytest.mark.parametrize("spec, rule, why, apart", PER_RUN_VERDICTS)
def test_per_run_gives_mains_verdict_reason_and_numbers(spec, rule, why, apart):
    given = records(spec)
    found = _check(checked(), "m", given, list(spec), {})
    runs = {(record["mode"], record["arm"]): record["runs"] for record in given}
    assert (found["rule"], found["why"]) == (rule, why)
    assert found["check"] == {mode: {"start": runs[(mode, "start")], "defaults": runs[(mode, "defaults")],
                                     "start_against_defaults": apart[mode]} for mode in spec}


def test_per_run_spelled_out_reads_as_the_default():
    spec = PER_RUN_VERDICTS[0][0]
    assert verdict(spec, harm="per_run") == verdict(spec)


# --- the proposal's evidence ------------------------------------------------------------------------------------------


def test_a_worst_checks_evidence_records_the_reading_and_each_arms_highest_counts(tmp_path):
    cfg = checked_runs(tmp_path, start=[dict(right=11, wrong=4), dict(right=11, wrong=4, made_up=1)],
                       defaults=[dict(right=10, wrong=5, made_up=1), dict(right=10, wrong=2)], modes=("instant",),
                       propose={"harm": "worst"})
    result = propose(cfg)
    held = evidence(cfg)
    assert result["m"]["chosen"] == "start" and held["rule"]["harm"] == "worst"
    assert held["check"]["instant"]["highest"] == {"start": {"wrong": 4, "made_up": 1},
                                                   "defaults": {"wrong": 5, "made_up": 1}}
    assert [each["more_wrong"] for each in held["check"]["instant"]["start_against_defaults"]] == [-1, 2]


def test_a_per_run_checks_evidence_is_what_main_wrote(tmp_path):
    cfg = checked_runs(tmp_path, **PER_RUN_FIXTURE)
    propose(cfg)
    written = (out(cfg) / "proposal" / "m" / "evidence.json").read_text()
    assert _placeheld(written, tmp_path, cfg) == PER_RUN_EVIDENCE.read_text()


# --- starters -----------------------------------------------------------------------------------------------------


def test_the_starter_schema_reads_the_reading_and_its_absence(tmp_path, monkeypatch):
    shipped = valid_evidence(tmp_path, monkeypatch)
    assert "harm" not in shipped["rule"]                       # evidence written before the reading existed
    assert StarterEvidence.model_validate(shipped).rule.harm == "per_run"
    worst = shipped | {"rule": shipped["rule"] | {"harm": "worst"}}
    assert StarterEvidence.model_validate(worst).rule.harm == "worst"
    with pytest.raises(ValidationError, match="harm"):
        StarterEvidence.model_validate(shipped | {"rule": shipped["rule"] | {"harm": "median"}})


def test_starter_evidence_writes_the_reading_only_when_it_is_worst(tmp_path, monkeypatch):
    shipped = valid_evidence(tmp_path, monkeypatch)
    per_run = json.loads(StarterEvidence.model_validate(shipped).model_dump_json())
    assert per_run["rule"] == shipped["rule"]                  # written as before: no harm key
    worst = StarterEvidence.model_validate(shipped | {"rule": shipped["rule"] | {"harm": "worst"}})
    assert json.loads(worst.model_dump_json())["rule"] == shipped["rule"] | {"harm": "worst"}


def test_a_per_run_check_publishes_a_starter_whose_rule_is_written_as_before(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    held = read_json(publish_starter(cfg, "m", "the tiny set")[1])
    assert set(held["rule"]) == {"strategy", "alpha", "max_more_wrong", "max_more_made_up", "max_errors", "repeats"}


def test_a_worst_check_publishes_a_starter_that_records_worst(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, propose={"harm": "worst"}, repeats=2)
    _, evidence_path = publish_starter(cfg, "m", "the tiny set")
    assert read_json(evidence_path)["rule"]["harm"] == "worst" and read_json(evidence_path)["safe_by"] == "check"
    assert read_evidence("model-2.5-awq").rule.harm == "worst"


def test_a_worst_check_that_proposed_the_defaults_is_never_published(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, propose={"harm": "worst"}, repeats=2,
                         defaults_better=True)
    assert evidence(cfg)["chosen"] == "defaults" and evidence(cfg)["rule"]["harm"] == "worst"
    with pytest.raises(SweepConfigError, match="the engine's defaults: there is no starter"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


# --- the whole path -------------------------------------------------------------------------------------------------


def _wrong_once_each(stub, case_id: str):
    """A rule for the scripted model: under the defaults' settings it answers `case_id` wrong the first time it is asked
    and right after, and under the start's settings (its cards, `_start_profile`) right the first time and wrong after.
    In instant mode a case is asked once a run, so each arm gives one wrong answer, in a run of its own."""
    asked: dict[str, int] = {}
    lock = threading.Lock()

    def rule(request: dict) -> str | None:
        system, user = _messages(request)
        case = stub._script.find(user) if _WRITER in system else None
        if case is None or case.id != case_id:
            return None
        arm = "start" if _START_MARK in f"{system}\n{user}" else "defaults"
        with lock:
            asked[arm] = count = asked.get(arm, 0) + 1
        return _proposal(case.wrong_sql) if (count == 1) == (arm == "defaults") else None

    return rule


def test_a_worst_check_from_config_to_starter(tmp_path, monkeypatch, stub_model):
    starter_folder(tmp_path, monkeypatch, {})
    model = stub_model(_wrong_once_each(stub_model, "book-count"), gold=True)
    cfg_path = write_config(tmp_path, deployment=tiny_deployment(tmp_path, model), model=model,
                            search={"strategy": "check"}, sweep={"repeats": 2}, propose={"harm": "worst"},
                            start=_start_profile(tmp_path, {"retrieval_k": 12}, ["instant"]),
                            models=one_model(model, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW))
    for action in ("run", "propose"):
        assert main(["sweep", action, "--config", str(cfg_path)]) == 0
    cfg = load_config(cfg_path)
    held = evidence(cfg)
    # One wrong answer each, in run 2 of the start and run 1 of the defaults: the per-run reading would refuse run 2.
    assert [each["more_wrong"] for each in held["check"]["instant"]["start_against_defaults"]] == [-1, 1]
    assert held["check"]["instant"]["highest"] == {"start": {"wrong": 1, "made_up": 0},
                                                   "defaults": {"wrong": 1, "made_up": 0}}
    assert held["chosen"] == "start" and held["rule"]["harm"] == "worst"
    assert main(["sweep", "starter", "--config", str(cfg_path), "--model", "m", "--dataset", "the tiny set"]) == 0
    published = read_evidence("model-2.5-awq")
    assert published.rule.harm == "worst" and published.safe_by == "check" and published.search is None


# --- publishing reads the reading the config fixed, not one the evidence was edited to ------------------------------


def _reading_edited(cfg, harm: str | None) -> None:
    """The proposal's evidence for model `m` with its rule's `harm` set to `harm`, or taken out when None, and nothing
    else changed: as a hand edit of `evidence.json` leaves it."""
    held = read_json(evidence_file(cfg))
    held["rule"].pop("harm", None)
    held["rule"] |= {"harm": harm} if harm is not None else {}
    write(evidence_file(cfg), json.dumps(held, indent=2))


def test_a_worst_proposal_whose_evidence_lost_its_reading_is_never_published(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, propose={"harm": "worst"}, repeats=2)
    _reading_edited(cfg, None)
    with pytest.raises(SweepConfigError, match=r"records harm 'per_run'.*'worst'"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_per_run_proposal_whose_evidence_claims_worst_is_never_published(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    _reading_edited(cfg, "worst")
    with pytest.raises(SweepConfigError, match=r"records harm 'worst'.*'per_run'"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


# --- the starter's rule record ------------------------------------------------------------------------------------

_RULE = {"strategy": "check", "alpha": 0.05, "max_more_wrong": 0, "max_more_made_up": 0, "max_errors": 0, "repeats": 2}


@pytest.mark.parametrize("strategy", ["screen-then-grid", "grid"])
def test_the_starter_schema_refuses_worst_outside_check_mode(tmp_path, monkeypatch, strategy):
    shipped = valid_evidence(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="check mode's reading alone"):
        StarterEvidence.model_validate(shipped | {"rule": shipped["rule"] | {"strategy": strategy, "harm": "worst"}})


def test_a_rule_dump_that_already_leaves_harm_out_is_written():
    for harm in ("per_run", "worst"):
        assert RuleRecord(**_RULE, harm=harm).model_dump(exclude={"harm"}) == _RULE
    assert RuleRecord(**_RULE).model_dump(exclude_defaults=True) == _RULE

"""`mnemiq sweep starter`: a starter and its evidence written from finished sweeps whose rule found it safe over the
engine's defaults, refused before anything is written when it is not safe or its build is not whole, and moved into
place only once it resolves to the version that was measured on every deployment.

Each test runs with the starters folder patched to `tmp_path / "starters"`, never the folder the engine ships."""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnemiq import profiles
from mnemiq.cli import main
from mnemiq.profiles import ENDPOINT_SETTINGS, ProfileFileError, _starter_entry, starter_names
from mnemiq.sweep import starters
from mnemiq.sweep.config import SweepConfigError, load_config
from mnemiq.sweep.propose import _agrees, _versions, propose
from mnemiq.sweep.run import OUT_LOCK, _sole
from mnemiq.sweep.starters import (
    DatasetRecord, StarterEvidence, _whole_build, fewer_right, publish_starter, read_evidence,
)

from .conftest import (
    CONFIG_BUILD, KEY_ENV, STUB_WINDOW, deployment_file_naming, drafted_versions, drops_the_instant_section, evidence,
    evidence_file, finished_check, finished_search, graded_sweep, missing_from, one_model, out, plant, proposal_file,
    read_json, starter_folder, two_deployments, write, write_config,
)

REPO = Path(__file__).resolve().parents[2]


def test_a_safe_check_publishes_a_starter_that_resolves_to_what_was_measured(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)                 # verdict "start"
    toml_path, evidence_path = publish_starter(cfg, "m", "the tiny set")
    assert toml_path.name == "model-2.5-awq.toml"                                # build_name("fixture/model-2.5-awq")
    held = tomllib.loads(toml_path.read_text())["profiles"]["model-2.5-awq"]
    assert not set(held) & {*ENDPOINT_SETTINGS, "api_key_env", "verify_api_key_env", "roles"}
    assert StarterEvidence.model_validate_json(evidence_path.read_text()).safe_by == "check"
    dep = write(tmp_path / "dep.toml", deployment_file_naming("model-2.5-awq", cfg))
    assert _agrees(_versions(cfg, dep, {}), drafted_versions(cfg))


def test_two_deployments_and_two_modes_each_resolve_to_their_measured_version(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, dataset=two_deployments(tmp_path, stub_model()),
                         modes=["instant", "thinking"])
    held = read_json(publish_starter(cfg, "m", "the tiny set")[1])
    # numbered in the order of the sorted names, which are never published
    assert set(held["versions"]) == {"deployment-1", "deployment-2"} and len(set(held["versions"].values())) == 2
    assert set(held["measured_versions"]) == {"instant", "thinking"}
    dep = write(tmp_path / "dep.toml", deployment_file_naming("model-2.5-awq", cfg))
    assert _versions(cfg, dep, {}) == {"lib-a": held["versions"]["deployment-1"],
                                       "lib-b": held["versions"]["deployment-2"]}


@pytest.mark.parametrize("make, match", [
    (lambda t, m, s: finished_check(t, m, s, defaults_better=True), "the engine's defaults: there is no starter"),
    (lambda t, m, s: finished_check(t, m, s, person_chose="start"), "chosen by a person"),
    (lambda t, m, s: finished_check(t, m, s, propose={"max_more_wrong": 1}), "max_more_wrong"),
    (lambda t, m, s: finished_check(t, m, s, start_declines=1), "fewer right answers than the engine's defaults"),
    (lambda t, m, s: finished_search(t, m, s, start=True), "did not start from the engine's defaults"),
    (lambda t, m, s: finished_search(t, m, s, confirmed=False), "not confirmed"),
    (lambda t, m, s: finished_check(t, m, s, build={"model": "fixture/model-2.5-awq", "quantization": "awq"}), "gpu"),
    (lambda t, m, s: finished_check(t, m, s, reports_build=False), "not read from the server"),
    (lambda t, m, s: finished_check(t, m, s, llm_context_window=8192), "context_window"),
    (lambda t, m, s: finished_check(t, m, s, unfinished=True), "has not finished"),
    (lambda t, m, s: finished_check(t, m, s, config_edited_after=True), "another config"),
])
def test_what_is_not_safe_or_not_whole_is_never_published(tmp_path, monkeypatch, stub_model, make, match):
    with pytest.raises(SweepConfigError, match=match):
        publish_starter(make(tmp_path, monkeypatch, stub_model), "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


@pytest.mark.parametrize("start, defaults, refused", [
    ([10], [10], False),                # level: published
    ([10], [11], True),                 # one run each: any shortfall
    ([10, 10], [12, 9], False),         # lowest against lowest: 10 is not below 9
    ([10, 10], [12, 11], True),         # 10 is below 11
    ([11, 9], [10, 10], True),          # the starter's lowest counts, not its best
])
def test_the_right_answers_refusal_compares_lowest_with_lowest(start, defaults, refused):
    assert bool(fewer_right({"instant": start}, {"instant": defaults})) is refused


def test_a_shortfall_in_one_mode_is_named_with_both_counts():
    assert fewer_right({"instant": [12], "thinking": [9, 10]}, {"instant": [12], "thinking": [11, 10]}) == [
        "in thinking its lowest right count over 2 runs is 9, and the defaults' lowest over 2 runs is 10"]


def test_a_shortfall_in_thinking_alone_is_refused_naming_thinking(tmp_path, monkeypatch, stub_model):
    # two measured modes, level in instant: a right-answers check that read only the first measured mode would publish
    # this
    cfg = finished_check(tmp_path, monkeypatch, stub_model, modes=("instant", "thinking"),
                         start_declines={"thinking": 1})
    with pytest.raises(SweepConfigError, match="fewer right answers than the engine's defaults") as refused:
        publish_starter(cfg, "m", "the tiny set")
    [line] = str(refused.value).split("defaults: ", 1)[1].split("; ")            # instant is not named
    a, b = map(int, re.fullmatch(r"in thinking its lowest right count over \d+ runs? is (\d+), and the defaults' "
                                 r"lowest over \d+ runs? is (\d+)", line).groups())
    assert a == b - 1                                                          # both counts, one declined
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_set_where_every_question_has_an_answer_publishes_saying_made_up_was_not_measured(
        tmp_path, monkeypatch, stub_model):
    # as on a set where every question has an answer: made-up answers cannot show, and that alone refuses nothing
    cfg = finished_check(tmp_path, monkeypatch, stub_model, answerable_only=True)
    held = read_json(publish_starter(cfg, "m", "the tiny set")[1])
    assert held["made_up_measured"] is False and held["safe_by"] == "check"


def test_publishing_on_another_engine_build_is_refused(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_build", lambda *a, **k: {"commit": "abc1234", "source_sha": "e" * 64})
    with pytest.raises(SweepConfigError, match="publish from a checkout at the sweep's commit"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_publishing_needs_a_checkout(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_commit", lambda *a, **k: None)
    with pytest.raises(SweepConfigError, match="run it from a checkout"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))
    assert _shipped(profiles.starters_folder()) == []      # not even the publisher's lock, in an installed package


def test_an_unset_key_is_refused_first(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.delenv(KEY_ENV)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_commit", lambda *a, **k: None)   # a later check would also fail
    with pytest.raises(SweepConfigError, match=KEY_ENV):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_name_that_is_not_the_builds_is_refused(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    with pytest.raises(SweepConfigError, match="is not the build's name 'model-2.5-awq'"):
        publish_starter(cfg, "m", "the tiny set", name="something-else")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))
    assert not list(profiles.starters_folder().glob("something-else*"))


def test_a_servers_local_weights_path_is_never_recorded(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, root="/srv/models/qwen-awq")
    text = publish_starter(cfg, "m", "the tiny set")[1].read_text()
    assert "/srv/models" not in text and "qwen-awq" not in text and '"model": "fixture/model-2.5-awq"' in text


def test_a_dotted_name_is_written_as_a_quoted_key(tmp_path, monkeypatch, stub_model):
    toml_path = publish_starter(finished_check(tmp_path, monkeypatch, stub_model), "m", "the tiny set")[0]
    assert '[profiles."model-2.5-awq"]' in toml_path.read_text()
    assert tomllib.loads(toml_path.read_text())["default"] == "model-2.5-awq"


def test_a_search_from_the_defaults_confirmed_is_published(tmp_path, monkeypatch, stub_model):
    cfg = finished_search(tmp_path, monkeypatch, stub_model)    # the stub answers gold only with guided_sql
    assert read_json(publish_starter(cfg, "m", "the tiny set")[1])["safe_by"] == "search_from_defaults"


def test_the_search_must_be_the_one_the_check_started_from(tmp_path, monkeypatch, stub_model):
    check = finished_check(tmp_path, monkeypatch, stub_model)                  # started from a profile file of its own
    other = finished_search(tmp_path / "x", monkeypatch, stub_model)           # its proposal is another file
    with pytest.raises(SweepConfigError, match="is not the file the check started from"):
        publish_starter(check, "m", "the tiny set", search=other)
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_check_started_from_the_searchs_proposal_publishes_with_both_in_its_evidence(tmp_path, monkeypatch,
                                                                                         stub_model):
    # The whole route: a search, check mode started from its proposal, then the command with --search.
    search = finished_search(tmp_path / "search", monkeypatch, stub_model)
    check = finished_check(tmp_path / "check", monkeypatch, stub_model, start_file=proposal_file(search))
    assert main(["sweep", "starter", "--config", str(check.path), "--model", "m", "--dataset", "the tiny set",
                 "--search", str(search.path)]) == 0
    held = read_json(profiles.starters_folder() / "model-2.5-awq.evidence.json")
    assert held["safe_by"] == "check" and held["search"] is not None
    assert held["search"]["confirmed"] is True and held["search"]["picks"] == {"instant": "guided_sql=true"}
    assert set(held["check"]) == set(held["modes_measured"]) == {"instant"}
    dep = write(tmp_path / "dep.toml", deployment_file_naming("model-2.5-awq", check))
    assert _agrees(_versions(check, dep, {}), drafted_versions(check))


def test_the_evidence_holds_aggregate_counts_only(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    plant(evidence_file(cfg), case_id="case-0001", question="How many widgets", path=str(tmp_path),
          url="http://192.0.2.9/v1")
    text = publish_starter(cfg, "m", "the tiny set")[1].read_text()
    assert not any(s in text for s in ("case-0001", "How many widgets", str(tmp_path), "192.0.2.9"))


def test_a_mode_left_unmeasured_and_what_is_held_unset_are_named(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, start_sets={"writer_temperature": 0.0})
    held = read_json(publish_starter(cfg, "m", "the tiny set")[1])
    assert held["modes_measured"] == ["instant"]
    assert held["unmeasured_modes"] == {"thinking": ["writer_temperature"], "deep": ["writer_temperature"]}
    # the tiny cases hold some with no answer
    assert "llm_seed" in held["held_unset"] and held["made_up_measured"] is True


def test_a_starter_that_resolves_otherwise_is_never_moved_into_place(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.setattr("mnemiq.sweep.starters.starter_text", drops_the_instant_section)
    with pytest.raises(SweepConfigError, match="does not resolve to the profile that was measured"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_republish_that_fails_leaves_the_shipped_starter(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    first = [path.read_bytes() for path in publish_starter(cfg, "m", "the tiny set")]
    monkeypatch.setattr("mnemiq.sweep.starters.starter_text", drops_the_instant_section)
    with pytest.raises(SweepConfigError):
        publish_starter(cfg, "m", "the tiny set")
    folder = profiles.starters_folder()
    assert [(folder / n).read_bytes() for n in ("model-2.5-awq.toml", "model-2.5-awq.evidence.json")] == first
    assert not list(folder.glob("*.publishing*"))


def test_the_doc_names_every_evidence_field_and_every_shipped_starter():
    text = (REPO / "docs/starters.md").read_text()
    assert all(f"`{field}`" in text for field in StarterEvidence.model_fields)
    assert missing_from(text, starter_names()) == [] and missing_from(text, ["not-a-starter"]) == ["not-a-starter"]


# --- what a person types, and what a second sweep brings -------------------------------------------------------------

# A dataset label that could hold a path, that is not printable ASCII, that is empty, or that runs past the cap.
UNFIT_LABELS = ["../questions", "C:\\sets\\mine", "the tiny set\n", "the tiny s\u00e9t", "x" * 81, "", "   "]


def _nothing_ran(tmp_path: Path):
    """A config whose sweep never ran: its out folder made, nothing in it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = load_config(write_config(tmp_path))
    out(cfg).mkdir(parents=True)
    return cfg


@pytest.mark.parametrize("label", UNFIT_LABELS)
def test_a_dataset_label_that_could_hold_a_path_or_is_unbounded_is_refused_first(tmp_path, monkeypatch, label):
    starter_folder(tmp_path, monkeypatch, {})
    with pytest.raises(SweepConfigError, match="--dataset"):
        publish_starter(_nothing_ran(tmp_path), "m", label)    # refused before any sweep is read
    assert _shipped(profiles.starters_folder()) == []


@pytest.mark.parametrize("label", UNFIT_LABELS)
def test_the_evidence_schema_refuses_such_a_dataset_label(label):
    with pytest.raises(ValidationError):
        DatasetRecord(name=label, questions=8, answerable=6)


def test_a_printable_dataset_label_is_one_the_schema_takes():
    assert DatasetRecord(name="the tiny set (v2): 8 questions", questions=8, answerable=6).name.startswith("the tiny")


def test_a_search_whose_pick_lost_every_change_is_the_defaults_and_is_never_published(tmp_path, monkeypatch):
    starter_folder(tmp_path, monkeypatch, {})
    # Each mode's pick sets one profile-wide setting its own way, so the profile assembled from them keeps the start's
    # value: the confirm round confirms the engine's defaults.
    cfg = graded_sweep(tmp_path, modes=("instant", "thinking"),
                       picks={"instant": "retrieval_k=36", "thinking": "retrieval_k=12"})
    propose(cfg)
    assert evidence(cfg)["confirm"]["same_as_start"] is True and evidence(cfg)["confirmed"] is True
    with pytest.raises(SweepConfigError, match="the search's pick is the engine's defaults: there is no starter"):
        publish_starter(cfg, "m", "the tiny set")
    assert _shipped(profiles.starters_folder()) == []


@pytest.mark.parametrize("held", ["check", "search"])
def test_publishing_holds_each_sweeps_out_folder_lock_as_propose_does(tmp_path, monkeypatch, held):
    starter_folder(tmp_path, monkeypatch, {})
    sweeps = {"check": _nothing_ran(tmp_path / "check"), "search": _nothing_ran(tmp_path / "search")}
    with _sole(out(sweeps[held]) / OUT_LOCK, RuntimeError("held by the test")):
        with pytest.raises(SweepConfigError, match=re.escape(f"{out(sweeps[held])} is in use by a run or a proposal")):
            publish_starter(sweeps["check"], "m", "the tiny set", search=sweeps["search"])
    assert _shipped(profiles.starters_folder()) == []


def test_a_search_that_is_itself_a_check_is_refused(tmp_path, monkeypatch, stub_model):
    first = finished_check(tmp_path / "first", monkeypatch, stub_model)
    second = finished_check(tmp_path / "second", monkeypatch, stub_model, start_file=proposal_file(first))
    with pytest.raises(SweepConfigError, match="--search names a sweep in check mode"):
        publish_starter(second, "m", "the tiny set", search=first)
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_search_given_beside_a_sweep_that_is_not_a_check_is_refused(tmp_path, monkeypatch, stub_model):
    cfg = finished_search(tmp_path, monkeypatch, stub_model)
    with pytest.raises(SweepConfigError, match="--search is given, so the sweep published must be a check"):
        publish_starter(cfg, "m", "the tiny set", search=cfg)
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def test_a_shortfall_in_the_second_run_alone_is_refused(tmp_path, monkeypatch, stub_model):
    # level with the defaults in its first run and one short in its second: a right-answers check that read only the
    # first run would publish this
    cfg = finished_check(tmp_path, monkeypatch, stub_model, repeats=2, start_declines=1, declining_from_run=2)
    runs = evidence(cfg)["check"]["instant"]
    start, defaults = ([run["right"] for run in runs[arm]] for arm in ("start", "defaults"))
    assert start[0] == defaults[0] and start[1] == defaults[1] - 1
    with pytest.raises(SweepConfigError, match="in instant its lowest right count over 2 runs is"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


def _servers(**read) -> dict:
    """A sweep's record holding what it read of model `m`'s server: the stub's build, with `read` changed."""
    return {"servers": {"m": {"server": "vllm", "server_version": "0.9.0.1", "context_window": STUB_WINDOW,
                              "root_name": "model-2.5-awq"} | read}}


def _stating(tmp_path: Path, build: dict | None, **fields):
    """A config whose model states `build` (none when None), loaded; nothing runs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    return load_config(write_config(tmp_path, models=one_model(build=build, **fields)))


@pytest.mark.parametrize("search_build, search_read, match", [
    (CONFIG_BUILD, {"server_version": None}, "the model build is incomplete: server_version not read"),
    (CONFIG_BUILD, {"server_version": "0.10.0"}, "the two sweeps read different servers: server_version '0.9.0.1' and "
                                                 "'0.10.0'"),
    (CONFIG_BUILD | {"quantization": "gptq"}, {}, r"the two sweeps state different builds in \[models.m.build\]: "
                                                  "quantization 'awq' and 'gptq'"),
])
def test_two_sweeps_that_read_or_state_the_build_otherwise_are_refused(tmp_path, search_build, search_read, match):
    check = _stating(tmp_path / "check", CONFIG_BUILD, llm_context_window=STUB_WINDOW)
    search = _stating(tmp_path / "search", search_build)
    with pytest.raises(SweepConfigError, match=match):
        _whole_build("m", [(check, _servers()), (search, _servers(**search_read))])


def test_two_sweeps_on_one_build_pass_and_a_build_the_search_does_not_state_is_not_compared(tmp_path):
    check = _stating(tmp_path / "check", CONFIG_BUILD, llm_context_window=STUB_WINDOW)
    for number, build in enumerate((CONFIG_BUILD, None)):
        search = _stating(tmp_path / f"search-{number}", build)
        # the weights folder is never compared
        _whole_build("m", [(check, _servers()), (search, _servers(root_name="3f2a9c"))])


def test_evidence_the_schema_refuses_is_refused_before_anything_is_written(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    # a value of another type than the schema's, as a later proposal might write
    plant(evidence_file(cfg), rule=read_json(evidence_file(cfg))["rule"] | {"alpha": "0.05"})
    with pytest.raises(SweepConfigError, match="evidence does not have the form the engine reads: alpha"):
        publish_starter(cfg, "m", "the tiny set")
    assert not list(profiles.starters_folder().glob("model-2.5-awq*"))


# --- the starter bound to its evidence, staged outside the folder, and moved in under a lock ------------------------


def _shipped(folder: Path) -> list[str]:
    """Every entry of the starters folder, hidden ones included."""
    return sorted(path.name for path in folder.iterdir())


def _second_move_fails(monkeypatch) -> list[str]:
    """The machine stops between publishing's two moves into the starters folder: the second raises. Returns the name
    each move into the folder was for, in order."""
    real, moves = os.replace, []

    def replace(source, target, *args, **kwargs):
        if Path(target).parent == profiles.starters_folder():
            moves.append(Path(target).name)
            if len(moves) == 2:
                raise OSError("the machine stopped between the two moves")
        return real(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    return moves


def test_the_evidence_binds_the_starter_by_the_sha256_of_its_file(tmp_path, monkeypatch, stub_model):
    toml_path, evidence_path = publish_starter(finished_check(tmp_path, monkeypatch, stub_model), "m", "the tiny set")
    assert read_json(evidence_path)["starter_sha256"] == hashlib.sha256(toml_path.read_bytes()).hexdigest()
    assert read_evidence("model-2.5-awq").starter == "model-2.5-awq"


def test_a_republish_stopped_between_its_two_moves_leaves_a_pair_the_engine_refuses(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    publish_starter(cfg, "m", "the tiny set")
    moves = _second_move_fails(monkeypatch)
    with pytest.raises(OSError, match="the machine stopped"):
        publish_starter(cfg, "m", "the tiny set, again")     # another label: another starter file and evidence
    assert moves == ["model-2.5-awq.evidence.json", "model-2.5-awq.toml"]      # the evidence first
    with pytest.raises(ProfileFileError, match="starter 'model-2.5-awq' and its evidence do not match"):
        _starter_entry("model-2.5-awq")
    with pytest.raises(SweepConfigError, match="ships without valid evidence"):
        read_evidence("model-2.5-awq")
    assert not list(profiles.starters_folder().glob(".*.tmp"))


def test_a_new_starter_stopped_between_its_two_moves_is_never_listed(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    _second_move_fails(monkeypatch)
    with pytest.raises(OSError, match="the machine stopped"):
        publish_starter(cfg, "m", "the tiny set")
    # only its evidence was moved in, which no reader takes for a starter
    assert "model-2.5-awq" not in starter_names()
    with pytest.raises(ProfileFileError, match="'model-2.5-awq' is not one this engine ships"):
        _starter_entry("model-2.5-awq")
    assert not list(profiles.starters_folder().glob(".*.tmp"))


def test_a_failure_while_the_starter_is_checked_leaves_nothing_new_in_the_folder(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    folder = profiles.starters_folder()
    stale = write(folder / f".model-2.5-awq.toml.{'c' * 32}.tmp", "left by a publish killed mid-move")
    before = _shipped(folder)
    monkeypatch.setattr("mnemiq.sweep.starters.starter_text", drops_the_instant_section)
    with pytest.raises(SweepConfigError, match="does not resolve to the profile that was measured"):
        publish_starter(cfg, "m", "the tiny set")
    assert _shipped(folder) == before == [stale.name]      # no lock taken, and nothing cleared, for a staged refusal


def test_no_draft_is_ever_listed_as_a_starter(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    listed, checking, moving = [], starters._resolves_as_measured, os.replace

    def checked(*args, **kwargs):
        listed.append(starter_names())                       # while the drafted starter is checked
        return checking(*args, **kwargs)

    def replace(source, target, *args, **kwargs):
        if Path(target).parent == profiles.starters_folder():
            listed.append(starter_names())                   # just before each move into the folder
        return moving(source, target, *args, **kwargs)

    monkeypatch.setattr("mnemiq.sweep.starters._resolves_as_measured", checked)
    monkeypatch.setattr(os, "replace", replace)
    publish_starter(cfg, "m", "the tiny set")
    assert len(listed) == 3 and listed == [[], [], []]
    assert starter_names() == ["model-2.5-awq"]


def test_a_held_publish_lock_refuses_a_second_publisher(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    folder = profiles.starters_folder()
    with _sole(folder / ".publish.lock", RuntimeError("held by another publisher")):
        with pytest.raises(SweepConfigError, match=re.escape(f"{folder} is in use by another publisher")):
            publish_starter(cfg, "m", "the tiny set")
    assert _shipped(folder) == [".publish.lock"]           # the other publisher's lock alone


def test_a_hidden_copy_a_killed_publish_left_is_cleared(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    folder = profiles.starters_folder()
    # what a publish killed between its copy and its move leaves: `.<file>.<32 hex>.tmp`, for either file
    stale = [write(folder / f".model-2.5-awq.toml.{'a' * 32}.tmp", "left by a publish killed mid-move"),
             write(folder / f".model-2.5-awq.evidence.json.{'b' * 32}.tmp", "left by a publish killed mid-move")]
    publish_starter(cfg, "m", "the tiny set")
    assert not any(path.exists() for path in stale)
    assert _shipped(folder) == [".publish.lock", "model-2.5-awq.evidence.json", "model-2.5-awq.toml"]


def test_the_hidden_copies_publishing_makes_are_the_ones_it_clears(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    moved, moving = [], os.replace

    def replace(source, target, *args, **kwargs):
        if Path(target).parent == profiles.starters_folder():
            moved.append(Path(source).name)
        return moving(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    publish_starter(cfg, "m", "the tiny set")
    assert len(moved) == 2 and all(starters._HIDDEN_COPY.fullmatch(name) for name in moved)


def test_a_hidden_file_publishing_never_makes_survives_a_publish(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    folder = profiles.starters_folder()
    kept = [write(folder / ".notes.tmp", "a person's own"), write(folder / ".x.tmp", "another tool's"),
            write(folder / f".model-2.5-awq.toml.{'A' * 32}.tmp", "not a copy publishing names: its hex is upper case")]
    publish_starter(cfg, "m", "the tiny set")
    assert all(path.exists() for path in kept)


def test_deployments_are_numbered_in_the_evidence_and_never_named(tmp_path, monkeypatch, stub_model):
    names = ("customer_acme_prod", "warehouse_eu")
    cfg = finished_check(tmp_path, monkeypatch, stub_model, dataset=two_deployments(tmp_path, stub_model(), names),
                         modes=["instant", "thinking"])
    shipped = [path.read_bytes() for path in publish_starter(cfg, "m", "the tiny set")]
    assert not any(name.encode() in data for name in names for data in shipped)
    held = read_json(profiles.starters_folder() / "model-2.5-awq.evidence.json")
    numbered = {"customer_acme_prod": "deployment-1", "warehouse_eu": "deployment-2"}      # one mapping, sorted
    proposed = evidence(cfg)                                                       # the sweep's own, by name
    assert held["versions"] == {numbered[name]: version for name, version in proposed["version"].items()}
    assert held["measured_versions"] == {
        mode: {numbered[name]: version for name, version in each.items()}
        for mode, each in proposed["measured_versions"].items()}


# --- what a server reports of itself, and what a person states of the build, as published ---------------------------


@pytest.mark.parametrize("reports, field", [
    ({"owned_by": "customer_acme_prod", "version": "/srv/customer_acme/private/build"}, "server"),
    ({"owned_by": "customer_acme_prod"}, "server"),
    ({"version": "/srv/customer_acme/private/build"}, "server_version"),
])
def test_a_server_that_reports_itself_as_no_known_release_is_never_published(tmp_path, monkeypatch, stub_model,
                                                                             reports, field):
    # the sweep ran and proposed on it: only what may be published is restricted
    cfg = finished_check(tmp_path, monkeypatch, stub_model, server=reports)
    folder = profiles.starters_folder()
    with pytest.raises(SweepConfigError, match=rf"^{field} "):
        publish_starter(cfg, "m", "the tiny set")
    assert _shipped(folder) == []
    assert not any(b"customer_acme" in path.read_bytes() for path in folder.rglob("*") if path.is_file())


@pytest.mark.parametrize("version, published", [("0.9.0+rocm6.2", True), ("0.9.0+acme", False)])
def test_a_server_version_publishes_only_as_a_release(tmp_path, monkeypatch, stub_model, version, published):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, server={"version": version})
    if published:
        assert read_json(publish_starter(cfg, "m", "the tiny set")[1])["model_build"]["server_version"] == version
    else:
        with pytest.raises(SweepConfigError, match="^server_version '0.9.0\\+acme' is not a release version"):
            publish_starter(cfg, "m", "the tiny set")
        assert _shipped(profiles.starters_folder()) == []


@pytest.mark.parametrize("field, value", [("gpu", "lab/rtx-4090"), ("quantization", "awq\\int4"),
                                          ("quantization", "awq\u0001"), ("gpu", "x" * 65)])
def test_a_stated_build_that_could_carry_a_path_is_never_published(tmp_path, monkeypatch, stub_model, field, value):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, build=CONFIG_BUILD | {field: value})
    with pytest.raises(SweepConfigError, match=rf"^\[models\.m\.build\] {field} "):
        publish_starter(cfg, "m", "the tiny set")
    assert _shipped(profiles.starters_folder()) == []

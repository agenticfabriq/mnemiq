"""`mnemiq sweep starter`: a starter and its evidence written from finished sweeps whose rule found it safe over the
engine's defaults, refused before anything is written when it is not safe or its build is not whole, and moved into
place only once it resolves to the version that was measured on every deployment.

Each test runs with the starters folder patched to `tmp_path / "starters"`, never the folder the engine ships."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from mnemiq import profiles
from mnemiq.cli import main
from mnemiq.profiles import ENDPOINT_SETTINGS, starter_names
from mnemiq.sweep.config import SweepConfigError
from mnemiq.sweep.propose import _agrees, _versions
from mnemiq.sweep.starters import StarterEvidence, fewer_right, publish_starter

from .conftest import (
    KEY_ENV, deployment_file_naming, drafted_versions, drops_the_instant_section, evidence_file, finished_check,
    finished_search, missing_from, plant, proposal_file, read_json, two_deployments, write,
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
    assert set(held["versions"]) == {"lib-a", "lib-b"} and len(set(held["versions"].values())) == 2
    assert set(held["measured_versions"]) == {"instant", "thinking"}
    dep = write(tmp_path / "dep.toml", deployment_file_naming("model-2.5-awq", cfg))
    assert _versions(cfg, dep, {}) == {"lib-a": held["versions"]["lib-a"], "lib-b": held["versions"]["lib-b"]}


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
    # as on BIRD: no case without an answer, so made-up answers cannot show, and that alone refuses nothing
    cfg = finished_check(tmp_path, monkeypatch, stub_model, answerable_only=True)
    held = read_json(publish_starter(cfg, "m", "the tiny set")[1])
    assert held["made_up_measured"] is False and held["safe_by"] == "check"


def test_publishing_on_another_engine_build_is_refused(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_build", lambda *a, **k: {"commit": "abc1234", "source_sha": "e" * 64})
    with pytest.raises(SweepConfigError, match="publish from a checkout at the sweep's commit"):
        publish_starter(cfg, "m", "the tiny set")


def test_publishing_needs_a_checkout(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_commit", lambda *a, **k: None)
    with pytest.raises(SweepConfigError, match="run it from a checkout"):
        publish_starter(cfg, "m", "the tiny set")


def test_an_unset_key_is_refused_first(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    monkeypatch.delenv(KEY_ENV)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_commit", lambda *a, **k: None)   # a later check would also fail
    with pytest.raises(SweepConfigError, match=KEY_ENV):
        publish_starter(cfg, "m", "the tiny set")


def test_a_name_that_is_not_the_builds_is_refused(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model)
    with pytest.raises(SweepConfigError, match="is not the build's name 'model-2.5-awq'"):
        publish_starter(cfg, "m", "the tiny set", name="something-else")


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
    plant(evidence_file(cfg), case_id="bird-1471", question="How many schools", path=str(tmp_path), url="http://192.0.2.9/v1")
    text = publish_starter(cfg, "m", "the tiny set")[1].read_text()
    assert not any(s in text for s in ("bird-1471", "How many schools", str(tmp_path), "192.0.2.9"))


def test_a_mode_left_unmeasured_and_what_is_held_unset_are_named(tmp_path, monkeypatch, stub_model):
    cfg = finished_check(tmp_path, monkeypatch, stub_model, start_sets={"writer_temperature": 0.0})
    held = read_json(publish_starter(cfg, "m", "the tiny set")[1])
    assert held["modes_measured"] == ["instant"]
    assert held["unmeasured_modes"] == {"thinking": ["writer_temperature"], "deep": ["writer_temperature"]}
    assert "llm_seed" in held["held_unset"] and held["made_up_measured"] is True   # the tiny cases hold some with no answer


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

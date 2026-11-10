"""A starter's evidence and the model server's build: what the sweep reads from the server, records and compares with
the build the starter was swept on, and the schema that holds a starter's evidence to the fields it names."""

from __future__ import annotations

import hashlib
import json
from typing import get_args

import pytest
from pydantic import ValidationError

from mnemiq.agent.modes import MODES
from mnemiq.profiles import starter_names
from mnemiq.sweep.config import SweepConfigError, load_config
from mnemiq.sweep.plan import render
from mnemiq.sweep.propose import propose
from mnemiq.sweep.run import run
from mnemiq.sweep.starters import (
    Mode, StarterEvidence, _publishable_build, build_mismatches, merged_readings, read_evidence, server_readings,
)

from .conftest import (
    COMMIT, CONFIG_BUILD, NAMED_KEY, PATH_SHAPED_KEY, SHA12, STUB_WINDOW, altered, evidence, one_model, started_sweep,
    starter_with_evidence, sweep_json, valid_evidence, write, write_config,
)

# A date in ASCII form but no calendar's, and a real date written in Arabic-Indic digits.
NO_SUCH_DAY = "2026-99-99"
DAY_IN_OTHER_DIGITS = "٢٠٢٦-١١-٠٧"


def test_the_server_reports_the_served_entry_and_never_a_path(stub_model):
    assert server_readings(stub_model().url, "stub-model", None) == {
        "server": "vllm", "server_version": "0.9.0.1", "context_window": STUB_WINDOW, "root_name": "model-2.5-awq"}
    assert server_readings(stub_model(root="/srv/models/qwen-awq").url, "stub-model", None)["root_name"] == "qwen-awq"
    assert server_readings(stub_model().url, "not-served", None)["context_window"] is None     # no entry: not read
    assert set(server_readings(stub_model(reports_build=False).url, "stub-model", None).values()) == {None}


def test_an_unread_field_is_never_a_difference_and_a_later_read_fills_it(stub_model):
    recorded = {"server": "vllm", "server_version": None, "context_window": 4096, "root_name": "model-2.5-awq"}
    now = {"server": None, "server_version": "0.9.0.1", "context_window": 4096, "root_name": None}
    assert merged_readings(recorded, now) == recorded | {"server_version": "0.9.0.1"}
    with pytest.raises(SweepConfigError, match="changed since the sweep started: context_window"):
        merged_readings(recorded, now | {"context_window": 8192})


def test_a_weights_folder_that_moved_is_recorded_never_refused():
    recorded = {"server": "vllm", "server_version": "0.9.0.1", "context_window": 4096, "root_name": "model-2.5-awq"}
    assert merged_readings(recorded, recorded | {"root_name": "3f2a9c"}) == recorded | {"root_name": "3f2a9c"}


def test_matching_builds_warn_nothing(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model()
    cfg = load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"},
                                   models=one_model(model, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW)))
    assert build_mismatches(cfg, "m", server_readings(model.url, "stub-model", None)) == []


def test_a_differing_server_version_is_named_with_both_values(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model(version="0.10.0")
    cfg = load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"},
                                   models=one_model(model, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW)))
    assert build_mismatches(cfg, "m", server_readings(model.url, "stub-model", None)) == [
        "starter 'fixture-2.5-build', model 'm': server_version differs: swept on '0.9.0.1', this sweep has '0.10.0'"]


def test_case_and_spacing_are_not_a_difference(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model()
    cfg = load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"},
                                   models=one_model(model, build=CONFIG_BUILD | {"gpu": "geforce  rtx 4090"}, llm_context_window=STUB_WINDOW)))
    assert build_mismatches(cfg, "m", server_readings(model.url, "stub-model", None)) == []


def test_what_is_neither_stated_nor_read_is_named_as_not_checked(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch)
    [line] = build_mismatches(load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"})), "m", None)
    named = set(line.split("report them: ", 1)[1].split(", "))      # the list itself, not the line's opening
    assert {"model", "quantization", "gpu", "server", "server_version", "llm_context_window"} <= named


def test_another_engine_build_is_named(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    monkeypatch.setattr("mnemiq.sweep.starters.engine_build", lambda *a, **k: {"commit": "abc1234", "source_sha": "e" * 64})
    model = stub_model()
    cfg = load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"},
                                   models=one_model(model, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW)))
    assert build_mismatches(cfg, "m", server_readings(model.url, "stub-model", None)) == [
        f"starter 'fixture-2.5-build': the engine differs: swept on {SHA12} ({COMMIT}), running eeeeeeeeeeee (abc1234)"]


def test_plan_shows_the_build_the_starter_and_its_warnings(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch)
    text = render(load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"}, models=one_model(build=CONFIG_BUILD))))
    assert "m build: gpu GeForce RTX 4090, quantization awq" in text
    assert "m starts from starter 'fixture-2.5-build'" in text and "read when the sweep runs" in text


def test_run_records_the_servers_build_and_says_the_warnings(tmp_path, monkeypatch, stub_model, capsys):
    starter_with_evidence(tmp_path, monkeypatch)
    cfg = load_config(write_config(tmp_path, model=stub_model(gold=True), start={"starter": "fixture-2.5-build"}))
    run(cfg, until="screen")      # a run, not a smoke: a smoke writes nothing outside its own folder
    assert sweep_json(cfg)["servers"]["m"]["server_version"] == "0.9.0.1"
    assert capsys.readouterr().out.count("not checked") == 1


def test_a_resume_on_another_server_is_refused(tmp_path, monkeypatch, stub_model):
    model = stub_model(gold=True)
    cfg = started_sweep(tmp_path, model)
    model.version = "0.10.0"
    with pytest.raises(SweepConfigError, match="changed since the sweep started: server_version"):
        run(cfg)


def test_a_resume_whose_weights_folder_moved_is_said_never_refused(tmp_path, monkeypatch, stub_model, capsys):
    model = stub_model(gold=True)
    cfg = started_sweep(tmp_path, model)
    model.root = "/srv/snapshots/3f2a9c"                          # a cache snapshot's folder is a hash
    run(cfg, until="screen")
    assert sweep_json(cfg)["servers"]["m"]["root_name"] == "3f2a9c"
    assert ("the server's weights folder for 'm' changed since the sweep started: was 'model-2.5-awq', now '3f2a9c'"
            in capsys.readouterr().out)


def test_a_resume_that_cannot_read_the_server_is_not_refused(tmp_path, monkeypatch, stub_model):
    model = stub_model(gold=True)
    cfg = started_sweep(tmp_path, model)
    model.reports_build = False                                   # as a timeout or a 429 on the build requests reads
    run(cfg)
    assert sweep_json(cfg)["servers"]["m"]["server_version"] == "0.9.0.1"


def test_a_weights_folder_with_another_name_is_said_never_refused(tmp_path, monkeypatch, stub_model, capsys):
    model = stub_model(gold=True, root="/srv/snapshots/3f2a9c")
    cfg = load_config(write_config(tmp_path, model=model, models=one_model(model, build=CONFIG_BUILD)))
    run(cfg, smoke=1)
    assert "weights folder for 'm' is '3f2a9c'" in capsys.readouterr().out


def test_the_proposal_records_the_starter_and_its_mismatches(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model(gold=True)
    cfg = load_config(write_config(tmp_path, model=model, start={"starter": "fixture-2.5-build"}, search={"strategy": "check"}))
    run(cfg)
    propose(cfg)
    assert evidence(cfg)["start_starter"] == {"name": "fixture-2.5-build",
                                              "mismatches": build_mismatches(cfg, "m", sweep_json(cfg)["servers"]["m"])}


def test_the_proposal_warns_of_each_mismatch_and_a_weights_folder_of_another_name(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model(gold=True, version="0.10.0", root="/srv/snapshots/3f2a9c")
    cfg = load_config(write_config(tmp_path, model=model, models=one_model(model, build=CONFIG_BUILD),
                                   start={"starter": "fixture-2.5-build"}, search={"strategy": "check"}))
    run(cfg)
    propose(cfg)
    warnings = evidence(cfg)["warnings"]
    mismatches = build_mismatches(cfg, "m", sweep_json(cfg)["servers"]["m"])
    assert any("server_version differs" in line for line in mismatches) and set(mismatches) <= set(warnings)
    assert ("the server's weights folder for 'm' is '3f2a9c', not fixture/model-2.5-awq's name: check "
            "[models.m.build] model") in warnings


def test_the_proposal_compares_the_starter_with_the_engine_the_arms_ran_on(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model(gold=True)
    cfg = load_config(write_config(tmp_path, model=model, start={"starter": "fixture-2.5-build"},
                                   search={"strategy": "check"}))
    run(cfg)
    # The engine packaging the proposal is another than the one the arms ran on, and the starter was swept on.
    monkeypatch.setattr("mnemiq.sweep.starters.engine_build", lambda *a, **k: {"commit": "abc1234",
                                                                               "source_sha": "e" * 64})
    propose(cfg)
    record = sweep_json(cfg)
    shown = evidence(cfg)["start_starter"]["mismatches"]
    assert shown == build_mismatches(cfg, "m", record["servers"]["m"], engine=record["engine_build"])
    assert not any("the engine differs" in line for line in shown)
    assert build_mismatches(cfg, "m", record["servers"]["m"])[-1].endswith("running eeeeeeeeeeee (abc1234)")


def test_a_given_engine_is_compared_in_place_of_the_running_one(tmp_path, monkeypatch, stub_model):
    starter_with_evidence(tmp_path, monkeypatch)
    model = stub_model()
    cfg = load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"},
                                   models=one_model(model, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW)))
    readings = server_readings(model.url, "stub-model", None)
    assert build_mismatches(cfg, "m", readings, engine={"commit": None, "source_sha": "d" * 64}) == [
        f"starter 'fixture-2.5-build': the engine differs: swept on {SHA12} ({COMMIT}), running dddddddddddd (no commit)"]


def test_a_starter_without_valid_evidence_is_refused_at_load(tmp_path, monkeypatch):
    folder = starter_with_evidence(tmp_path, monkeypatch)
    # bound to the starter's file, so the engine's own loader takes it, and nothing else the schema needs
    sha = hashlib.sha256((folder / "fixture-2.5-build.toml").read_bytes()).hexdigest()
    write(folder / "fixture-2.5-build.evidence.json",
          json.dumps({"starter": "fixture-2.5-build", "starter_sha256": sha}))
    with pytest.raises(SweepConfigError, match="ships without valid evidence"):
        load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"}))


@pytest.mark.parametrize("plant_at", [("dataset", "extra"), ("rule", "extra"), ("model_build", "extra"),
                                      ("check", "instant", 0, "start", "extra"), ("unmeasured_modes", "fast"),
                                      ("rule", "max_more_wrong"), ("published", None),
                                      ("versions", PATH_SHAPED_KEY), ("measured_versions", "instant", PATH_SHAPED_KEY),
                                      ("versions", NAMED_KEY), ("measured_versions", "instant", NAMED_KEY),
                                      ("model_build", "server", None, "customer_acme_prod"),
                                      ("model_build", "server_version", None, "/srv/customer_acme/private/build"),
                                      ("model_build", "gpu", None, "lab/rtx-4090"),
                                      ("model_build", "quantization", None, "awq\u0001"),
                                      ("model_build", "model", None, "/srv/models/qwen-awq"),
                                      ("published", None, NO_SUCH_DAY), ("published", None, DAY_IN_OTHER_DIGITS)])
def test_the_evidence_schema_refuses_what_it_does_not_name_at_any_depth(tmp_path, monkeypatch, plant_at):
    good = valid_evidence(tmp_path, monkeypatch)                     # starter_with_evidence's evidence, as a dict
    StarterEvidence.model_validate(good)                             # positive control: the unaltered one passes
    with pytest.raises(ValidationError) as refused:
        StarterEvidence.model_validate(altered(good, plant_at))
    # altered: an unknown key at that path; "fast" as a mode key; max_more_wrong 1; published "07/11/2026", a day no
    # calendar has, or one in other digits; a path-shaped database key; a model given as a path
    where = plant_at[:plant_at.index(None)] if None in plant_at else plant_at
    assert refused.value.errors()[0]["loc"][:len(where)] == where    # refused where it was planted, not elsewhere


def test_evidence_that_names_another_starter_file_is_refused(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch, starter_sha256="0" * 64)
    with pytest.raises(SweepConfigError, match=r"ships without valid evidence \(its starter_sha256 is not the sha256 "
                                               r"of fixture-2\.5-build\.toml\)"):
        read_evidence("fixture-2.5-build")


# Releases as PEP 440 names their parts, with a local part only for the accelerator a build is for, and what is not.
RELEASES = ["0.9.0.1", "0.6.3.post1", "0.9.0rc2", "0.9.0+cu124", "0.9.0+rocm6.2"]
NOT_RELEASES = ["0.9.0+acme", "0.9.0+customer_acme_prod", "/srv/customer_acme/private/build", "0.9.0 rc2", ""]


@pytest.mark.parametrize("version", RELEASES)
def test_a_release_of_a_known_server_is_one_the_schema_takes(tmp_path, monkeypatch, version):
    good = valid_evidence(tmp_path, monkeypatch)
    good["model_build"] |= {"server": "vllm", "server_version": version}
    assert StarterEvidence.model_validate(good).model_build.server_version == version


@pytest.mark.parametrize("version", NOT_RELEASES)
def test_a_version_that_is_not_a_release_is_refused_by_the_schema(tmp_path, monkeypatch, version):
    good = valid_evidence(tmp_path, monkeypatch)
    with pytest.raises(ValidationError):
        StarterEvidence.model_validate(good | {"model_build": good["model_build"] | {"server_version": version}})


@pytest.mark.parametrize("version, published", [*((v, True) for v in RELEASES), *((v, False) for v in NOT_RELEASES)])
def test_publishing_takes_a_release_and_refuses_anything_else(version, published):
    build = {"server": "vllm", "server_version": version, "gpu": "GeForce RTX 4090", "quantization": "awq"}
    if published:
        _publishable_build("m", build)
    else:
        with pytest.raises(SweepConfigError, match="^server_version "):
            _publishable_build("m", build)


def test_the_mode_alias_is_the_engines_modes():
    assert set(get_args(Mode)) == set(MODES)


def test_every_shipped_starter_has_valid_evidence():
    # vacuous until the engine ships its first starter
    for name in starter_names():
        assert read_evidence(name).starter == name

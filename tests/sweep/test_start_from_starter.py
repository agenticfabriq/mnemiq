"""A sweep starts from a starter the engine ships, named in `[start]`, or from a profile whose chain reaches one: read
whole, pinned, and fingerprinted, as is the build each model states."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mnemiq.config import DEFAULT_RETRIEVAL_K
from mnemiq.sweep.arms import start_settings
from mnemiq.sweep.config import (
    ModelBuild, SweepConfigError, fingerprint, fingerprint_body, load_config, pinned_files, start_starter,
)

from .conftest import (
    CONFIG_BUILD, ONE_MODEL, ROOT_AND_CHILD, bind, edit, idle_deployment, one_model, start_that_differs, start_via,
    starter_with_evidence, write, write_config,
)

# The config part of `write_config`'s fingerprint body, as the engine wrote it before starters and build tables: a
# config that names neither must fingerprint as it did.
GOLDEN_CONFIG_DUMP = (
    '{"dataset": {"deployments": null, "env": "<tmp>/deployment/deployment.env", "held_out": 0.3, "identity": '
    '{"principal": "sweep", "roles": ["sweep-test"], "tenant": "local"}, "questions": '
    '"<tmp>/deployment/questions.json", "split_seed": 7}, "judge": null, "machine": {"parallel": 1}, "models": {"m": '
    '{"api_key_env": "SWEEP_TEST_KEY", "llm_base_url": "http://127.0.0.1:9/v1", "llm_context_window": null, '
    '"llm_model": "stub-model", "price_in_per_mtok": null, "price_out_per_mtok": null, "verify_api_key_env": null, '
    '"verify_base_url": null, "verify_model": null}}, "modes": ["instant"], "propose": {"alpha": 0.05, '
    '"max_more_made_up": 0, "max_more_wrong": 0, "profile": null}, "search": {"screen_alpha": 0.25, "strategy": '
    '"screen-then-grid"}, "space": {}, "start": {"per_model": {}, "profile": null, "profile_file": null}, "sweep": '
    '{"max_errors": 0, "name": "tiny", "out": "<tmp>/sweep-out", "repeats": "auto"}}'
)
# The `[start]` part of the same dump for a config whose model has a start table of its own, naming a profile.
GOLDEN_START_DUMP = (
    '{"per_model": {"m": {"profile": "b", "profile_file": null}}, "profile": null, "profile_file": "<tmp>/shared.toml"}'
)


def test_a_sweep_starts_from_a_starter_by_name(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch)
    cfg = load_config(write_config(tmp_path, start={"starter": "fixture-2.5-build"}))
    start = start_settings(cfg, "m")
    assert (start["retrieval_k"], start["instant.card_style"]) == (36, "bare") and start_starter(cfg, "m") == "fixture-2.5-build"


def test_a_starter_is_never_matched_from_the_model_name(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch)
    cfg = load_config(write_config(tmp_path, models=one_model(llm_model="fixture-2.5-build")))
    assert start_settings(cfg, "m")["retrieval_k"] == DEFAULT_RETRIEVAL_K and start_starter(cfg, "m") is None


@pytest.mark.parametrize("start, match", [
    ({"starter": "fixture-2.5-build", "profile_file": "p.toml"}, "a start is a starter or a profile file, not both"),
    ({"starter": "fixture-2.5-build", "m": {"profile": "x"}}, r"\[start.m\] names a profile, and \[start\] names a starter"),
    ({"starter": "nope"}, "it ships: fixture-2.5-build"),
])
def test_a_start_that_reads_two_ways_or_none_is_refused(tmp_path, monkeypatch, start, match):
    starter_with_evidence(tmp_path, monkeypatch)
    with pytest.raises(SweepConfigError, match=match):
        load_config(write_config(tmp_path, start=start))


def test_a_models_own_starter_replaces_the_start(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch)
    cfg = load_config(write_config(tmp_path, start={**start_that_differs(tmp_path), "m": {"starter": "fixture-2.5-build"}}))
    assert start_starter(cfg, "m") == "fixture-2.5-build" and start_settings(cfg, "m")["retrieval_k"] == 36


def test_a_start_profile_that_extends_is_read_through_its_chain(tmp_path, monkeypatch):
    starter_with_evidence(tmp_path, monkeypatch)
    path = write(tmp_path / "mine.toml", ROOT_AND_CHILD)       # root names the starter; child extends root
    cfg = load_config(write_config(tmp_path, start={"profile_file": str(path), "profile": "child"}))
    start = start_settings(cfg, "m")
    assert (start["retrieval_k"], start["writer_temperature"], start["instant.card_style"]) == (12, 0.2, "bare")
    assert start_starter(cfg, "m") == "fixture-2.5-build"


@pytest.mark.parametrize("via", ["starter", "profile_file"])
@pytest.mark.parametrize("changed, old, new", [("fixture-2.5-build.toml", "retrieval_k = 36", "retrieval_k = 40"),
                                               ("fixture-2.5-build.evidence.json", '"awq"', '"gptq"')])
def test_the_starter_file_is_pinned(tmp_path, monkeypatch, via, changed, old, new):
    folder = starter_with_evidence(tmp_path, monkeypatch)
    cfg = load_config(write_config(tmp_path, deployment=idle_deployment(tmp_path), start=start_via(tmp_path, via)))
    before, pins = fingerprint(cfg), pinned_files(cfg)
    edit(folder / changed, old, new)
    bind(folder, "fixture-2.5-build")   # the starter and its evidence stay a pair, as publishing writes them
    assert fingerprint(load_config(cfg.path)) != before
    assert pinned_files(load_config(cfg.path))[str(folder / changed)] != pins[str(folder / changed)]


def test_a_config_without_the_new_keys_fingerprints_as_before(tmp_path):
    cfg = load_config(write_config(tmp_path, deployment=idle_deployment(tmp_path)))
    dumped = json.dumps(fingerprint_body(cfg)["config"], sort_keys=True).replace(str(tmp_path), "<tmp>")
    assert dumped == GOLDEN_CONFIG_DUMP     # recorded by running the same function on main before this change


def test_a_models_own_start_table_without_a_starter_fingerprints_as_before(tmp_path):
    shared = write(tmp_path / "shared.toml", 'default = "a"\n[profiles.a]\nretrieval_k = 12\n[profiles.b]\nretrieval_k = 16\n')
    cfg = load_config(write_config(tmp_path, deployment=idle_deployment(tmp_path),
                                   start={"profile_file": str(shared), "m": {"profile": "b"}}))
    dumped = json.dumps(fingerprint_body(cfg)["config"]["start"], sort_keys=True).replace(str(tmp_path), "<tmp>")
    assert dumped == GOLDEN_START_DUMP      # recorded by running the same dump on main before this change


def test_the_build_table_is_fingerprinted(tmp_path):
    d = idle_deployment(tmp_path)     # a fingerprint pins the store, so it needs one built
    a = fingerprint(load_config(write_config(tmp_path, deployment=d, models=one_model(build=CONFIG_BUILD))))
    b = fingerprint(load_config(write_config(tmp_path, deployment=d, models=one_model(build=CONFIG_BUILD | {"gpu": "A100"}))))
    assert a != b


def test_no_model_may_be_named_starter(tmp_path):
    with pytest.raises(SweepConfigError,
                       match=r"models\.starter: \[start\] reads starter as its own key, so no model can take the name"):
        load_config(write_config(tmp_path, models={"starter": ONE_MODEL}))


@pytest.mark.parametrize("model", ["/srv/models/qwen-awq", "qwen-awq", "~/qwen-awq", "./qwen-awq", "a/b/c"])
def test_a_build_read_on_its_own_refuses_what_is_not_a_repository_id(model):
    ModelBuild.model_validate({"model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"})    # positive control
    with pytest.raises(ValidationError):
        ModelBuild.model_validate({"model": model})


@pytest.mark.parametrize("model", ["/srv/models/qwen-awq", "models/qwen-awq", "~/qwen-awq", "./qwen-awq"])
def test_a_local_path_is_never_a_model_id(tmp_path, model):
    (tmp_path / "models" / "qwen-awq").mkdir(parents=True)     # the relative one exists beside the config
    with pytest.raises(SweepConfigError, match=r"\[models.m.build\] model"):
        load_config(write_config(tmp_path, models=one_model(build=CONFIG_BUILD | {"model": model})))

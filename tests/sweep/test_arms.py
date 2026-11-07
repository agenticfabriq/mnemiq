"""The arms: what each round runs, the profile file each arm loads, and the engine's own word on whether it runs."""

from __future__ import annotations

import re

import pytest

from mnemiq.sweep.arms import assembled_arm, check_arms, engine_refusal, grid_arms, profile_text, screen_arms
from mnemiq.sweep.config import load_config

from .conftest import KEY_ENV, STUB_KEY, UNREACHABLE, arm_with, start_of, tiny_deployment, tiny_files, write_config


def test_every_arm_id_is_a_name_the_engine_accepts(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model),
                                   space={"writer_rules": ["full", "none"], "retrieval_k": [12, 24, 36]}))
    arms = screen_arms(cfg) + grid_arms(cfg, {("m", "instant"): {"writer_rules": ["none"], "retrieval_k": [36]}})
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", a.profile_id) for a in arms)
    assert all(engine_refusal(cfg, a) is None for a in arms)


def test_the_screen_is_the_start_and_one_arm_per_other_value(tmp_path):
    cfg = load_config(write_config(tmp_path, space={"writer_rules": ["full", "none"], "retrieval_k": [12, 24, 36]}))
    assert [a.label for a in screen_arms(cfg)] == ["start", "retrieval_k=12", "retrieval_k=36", "writer_rules=none"]


def test_the_grid_reuses_the_screens_one_change_arms(tmp_path):
    cfg = load_config(write_config(tmp_path))
    kept = {("m", "instant"): {"writer_rules": ["none"], "retrieval_k": [36]}}
    grid = grid_arms(cfg, kept)
    assert {a.label for a in grid if a.round == "grid"} == {"retrieval_k=36 + writer_rules=none"}


def test_a_combination_the_engine_refuses_is_named_not_run(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model),
                                   modes=["instant"], space={"instant": {"min_agreement": [0.6]}}))
    arm = next(a for a in screen_arms(cfg) if a.profile_id != "start")
    assert "one candidate" in engine_refusal(cfg, arm)


def test_an_env_files_setting_never_reaches_the_baseline(tmp_path, stub_model):
    dep = tiny_deployment(tmp_path, stub_model, env_extra={"MNEMIQ_RETRIEVAL_K": "5"})
    cfg = load_config(write_config(tmp_path, deployment=dep))
    assert 'retrieval_k = 24' in profile_text(cfg, check_arms(cfg)[1])   # the defaults arm: the engine's field default


def test_modes_that_disagree_keep_the_start(tmp_path):
    cfg = load_config(write_config(tmp_path, modes=["instant", "thinking"]))
    picks = {"instant": arm_with(cfg, writer_reasoning="before_sql"), "thinking": start_of(cfg, "thinking")}
    arm, conflicts = assembled_arm(cfg, "m", picks)
    assert conflicts == ["writer_reasoning"] and ("writer_reasoning", "before_sql") not in arm.changes


def test_an_unset_key_variable_is_named(tmp_path, monkeypatch):
    cfg = load_config(write_config(tmp_path))
    monkeypatch.delenv(KEY_ENV)
    refusal = engine_refusal(cfg, start_of(cfg, "instant"))
    assert KEY_ENV in refusal and STUB_KEY not in refusal


def test_a_deployment_that_embeds_on_its_chat_endpoint_keeps_its_embedder(tmp_path):
    # The arm's baseline clears the chat endpoint, a profile setting; the embedder, which falls back to it, must not
    # move with it, since the store's index was built with that embedder.
    d = tiny_files(tmp_path)
    d.env.write_text(f"MNEMIQ_SOURCES_PATH='{d.sources}'\nMNEMIQ_STORE_PATH='{d.store}'\n"
                     f"MNEMIQ_AUTHZ_PATH='{d.authz}'\nMNEMIQ_LLM_BASE_URL='{UNREACHABLE}'\n"
                     f"MNEMIQ_LLM_API_KEY='{STUB_KEY}'\n")
    cfg = load_config(write_config(tmp_path, deployment=d))
    assert engine_refusal(cfg, start_of(cfg, "instant")) is None


def test_the_start_profile_is_the_start_and_the_defaults_arm_undoes_it(tmp_path):
    start = tmp_path / "start.toml"
    start.write_text('default = "tuned"\n[profiles.tuned]\nretrieval_k = 12\nllm_model = "another-build"\n'
                     '[profiles.tuned.instant]\nplan_attempts = 2\n')
    cfg = load_config(write_config(tmp_path, start={"profile_file": str(start)},
                                   space={"retrieval_k": [12, 36]}))
    assert [a.label for a in screen_arms(cfg)] == ["start", "retrieval_k=36"]
    text = profile_text(cfg, start_of(cfg, "instant"))
    # the start's own settings, with the model's endpoint from the sweep's config rather than the start file
    assert "retrieval_k = 12" in text and "plan_attempts = 2" in text and 'llm_model = "stub-model"' in text
    defaults = check_arms(cfg)[1]
    assert defaults.changes == (("instant.plan_attempts", 3), ("retrieval_k", 24))
    assert engine_refusal(cfg, defaults) is None


def test_an_env_files_setting_never_reaches_the_engines_check(tmp_path):
    # A temperature can be set in an env file but not cleared by a profile, since TOML has no null; under this model it
    # is a refusal. The arm resets it before its profile applies, so the engine judges the arm and not the env file.
    d = tiny_files(tmp_path)
    with open(d.env, "a") as fh:
        fh.write("MNEMIQ_WRITER_TEMPERATURE='0'\n")
    cfg = load_config(write_config(tmp_path, deployment=d, models={
        "m": {"llm_model": "gpt-5-mini", "llm_base_url": UNREACHABLE, "api_key_env": KEY_ENV}}))
    assert engine_refusal(cfg, start_of(cfg, "instant")) is None


@pytest.mark.parametrize("url", ["http://embed:SECRET@embedder.example/v1?api_key=SECRET",
                                 "http://embed:abc/SECRET@embedder.example/v1"])
def test_a_refusal_never_quotes_a_credential_from_a_deployment_url(tmp_path, url):
    # Whatever an arm's refusal names, the sweep shows a deployment's URLs without their credentials. Here the refusal
    # is the local-only check's, for an embedder on a DNS name.
    d = tiny_files(tmp_path)
    embedder = f"MNEMIQ_EMBED_BASE_URL='{url}'"
    env = d.env.read_text().replace(f"MNEMIQ_EMBED_BASE_URL='{UNREACHABLE}'", embedder)
    d.env.write_text(env + "MNEMIQ_LOCAL_ONLY='true'\n")
    cfg = load_config(write_config(tmp_path, deployment=d))
    refusal = engine_refusal(cfg, start_of(cfg, "instant"))
    assert "embed_base_url='http://embedder.example/v1'" in refusal and "SECRET" not in refusal


def test_no_profile_text_carries_a_credential_from_a_deployment_url(tmp_path):
    d = tiny_files(tmp_path)
    with open(d.env, "a") as fh:
        fh.write("MNEMIQ_LLM_BASE_URL='http://chat:SECRET@127.0.0.1:9/v1'\n"
                 "MNEMIQ_VERIFY_BASE_URL='http://judge:SECRET@127.0.0.1:9/v1?api_key=SECRET'\n")
    start = tmp_path / "start.toml"
    start.write_text('default = "s"\n[profiles.s]\nllm_base_url = "http://start:SECRET@127.0.0.1:9/v1"\n')
    cfg = load_config(write_config(tmp_path, deployment=d, modes=["instant", "thinking"],
                                   start={"profile_file": str(start)}, space={"writer_rules": ["none"]}))
    arms = screen_arms(cfg) + check_arms(cfg)
    arms.append(assembled_arm(cfg, "m", {"instant": arm_with(cfg, writer_rules="none"),
                                         "thinking": start_of(cfg, "thinking")})[0])
    assert not [arm.label for arm in arms if "SECRET" in profile_text(cfg, arm)]

"""A sweep's config file: what it may vary is checked against the engine's own fields, and its fingerprint moves with
what changes an answer and with nothing else."""

from __future__ import annotations

import pytest

from mnemiq.config import Settings
from mnemiq.sweep.config import SweepConfigError, deployment_settings, fingerprint, load_config

from .conftest import STUB_KEY, alter, broken_questions, tiny_deployment, tiny_files, write_config


def test_a_value_of_the_wrong_type_is_refused(tmp_path):
    with pytest.raises(SweepConfigError, match="guided_sql"):
        load_config(write_config(tmp_path, space={"guided_sql": ["yes"]}))


def test_a_value_outside_the_fields_limits_is_refused(tmp_path):
    with pytest.raises(SweepConfigError, match="retrieval_k"):
        load_config(write_config(tmp_path, space={"retrieval_k": [0, 24]}))


def test_a_mode_value_is_checked_strictly(tmp_path):
    with pytest.raises(SweepConfigError, match="max_attempts"):
        load_config(write_config(tmp_path, space={"thinking": {"max_attempts": [True]}}))


def test_one_value_twice_is_refused(tmp_path):
    with pytest.raises(SweepConfigError, match="twice"):
        load_config(write_config(tmp_path, space={"writer_temperature": [1, 1.0]}))


def test_a_deployment_setting_cannot_vary(tmp_path):
    with pytest.raises(SweepConfigError, match="store_path.*deployment"):
        load_config(write_config(tmp_path, space={"store_path": ["a", "b"]}))


@pytest.mark.parametrize("name", ["llm_model", "llm_base_url", "llm_context_window", "verify_model", "verify_base_url"])
def test_an_endpoint_fact_belongs_to_models(tmp_path, name):
    # each is in PROFILE_SETTINGS, so only a check that runs first gives this message
    with pytest.raises(SweepConfigError, match=rf"{name}.*\[models\]"):
        load_config(write_config(tmp_path, space={name: ["a", "b"]}))


@pytest.mark.parametrize("bad", ["duplicate_id", "answerable_without_gold", "unknown_db_id"])
def test_a_question_file_that_cannot_be_graded_is_refused(tmp_path, bad):
    with pytest.raises(SweepConfigError):
        load_config(write_config(tmp_path, questions=broken_questions(tmp_path, bad)))


def test_max_errors_must_be_a_count(tmp_path):
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(SweepConfigError, match="max_errors"):
            load_config(write_config(tmp_path, sweep={"max_errors": bad}))


@pytest.mark.parametrize("change", ["store", "authz", "env_setting", "start_profile", "questions"])
def test_what_changes_an_answer_changes_the_fingerprint(tmp_path, stub_model, change):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model)))
    before = fingerprint(cfg)
    alter(cfg, change)
    assert fingerprint(load_config(cfg.path)) != before


@pytest.mark.parametrize("change", ["budget", "control"])
def test_what_does_not_change_an_answer_leaves_it(tmp_path, stub_model, change):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model)))
    before = fingerprint(cfg)
    alter(cfg, change)
    assert fingerprint(load_config(cfg.path)) == before


def test_no_key_enters_the_fingerprint(tmp_path, stub_model, monkeypatch):
    monkeypatch.setenv("SWEEP_TEST_KEY", "sk-test-one")
    a = fingerprint(load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model))))
    monkeypatch.setenv("SWEEP_TEST_KEY", "sk-test-two")
    assert fingerprint(load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model)))) == a


def test_the_deployment_is_read_from_its_env_file_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMIQ_RETRIEVAL_K", "3")
    monkeypatch.setenv("MNEMIQ_AUTHZ_PATH", str(tmp_path / "another-policy.json"))
    assert Settings().retrieval_k == 3, "the positive control: the engine's own reading takes the variable"
    settings = deployment_settings(load_config(write_config(tmp_path)))
    assert type(settings) is Settings
    assert settings.retrieval_k == Settings.model_fields["retrieval_k"].default
    # The suite sets MNEMIQ_STORE_PATH for every test, so the store is a second variable ignored.
    assert (settings.authz_path, settings.store_path) == (str(tiny_files(tmp_path).authz), str(tiny_files(tmp_path).store))


def test_a_key_in_the_env_file_stays_out_of_the_fingerprint(tmp_path, stub_model):
    deployment = tiny_deployment(tmp_path, stub_model)
    cfg = load_config(write_config(tmp_path, deployment=deployment))
    before = fingerprint(cfg)
    deployment.env.write_text(deployment.env.read_text().replace(f"'{STUB_KEY}'", "'another-key'"))
    assert deployment_settings(cfg).llm_api_key == "another-key"
    assert fingerprint(load_config(cfg.path)) == before


@pytest.mark.parametrize("dsn", ["postgresql://sweep:{password}@db.example/{database}",
                                 "host=db.example user=sweep password={password} dbname={database}"])
def test_a_dsn_keeps_its_database_and_loses_its_password_in_the_fingerprint(tmp_path, stub_model, dsn):
    deployment = tiny_deployment(tmp_path, stub_model)
    env = deployment.env.read_text()

    def fingerprint_with(password: str, database: str) -> str:
        deployment.env.write_text(env + f"MNEMIQ_CONTROL_DSN='{dsn.format(password=password, database=database)}'\n")
        return fingerprint(load_config(write_config(tmp_path, deployment=deployment)))

    assert fingerprint_with("first-pass", "library") == fingerprint_with("second-pass", "library")
    assert fingerprint_with("first-pass", "library") != fingerprint_with("first-pass", "archive")


def test_the_start_profile_files_content_is_pinned(tmp_path, stub_model):
    profile = tmp_path / "starts.toml"
    profile.write_text('default = "start"\n[profiles.start]\nretrieval_k = 12\n')
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model),
                                   start={"profile_file": str(profile), "profile": "start"}))
    before = fingerprint(cfg)
    alter(cfg, "start_profile")
    assert fingerprint(load_config(cfg.path)) != before

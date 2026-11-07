"""One arm of a sweep: run on the product path over its own copies, graded by `run_case`, one row per case and mode, and
stopped when its endpoint stops answering, never by one question's error."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from mnemiq.sweep.config import load_config
from mnemiq.sweep.run import arm_folder, arm_settings, engine_build, identity_refusal, run_arm

from .conftest import (
    cancel_gold_once, copy_package, one_arm, out_files, read_rows, start_of, tiny_deployment, touch_store, tuning,
    two_deployments, write_config,
)


def test_an_arm_runs_the_product_path_and_stamps_one_version(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "complete"
    rows = read_rows(cfg, arm)
    assert len(rows) == len(tuning(cfg)) and len({r["profile_version"] for r in rows}) == 1
    assert all(r["prompt_tokens"] > 0 and r["engine_build"]["source_sha"] for r in rows)
    # Each row counts its own calls, and every answered row has its time to rows, which comes before the prose.
    assert sum(r["calls"] for r in rows) == stub.calls
    assert all((r["rows_ms"] is not None) == (r["outcome"] == "correct") for r in rows)
    assert all(r["rows_ms"] is None or 0 < r["rows_ms"] <= r["ms"] for r in rows)


def test_an_answer_to_a_question_with_no_answer_is_made_up(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model())   # answers every question, wrongly
    run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1))
    answerable = {case.id: case.answerable for case in tuning(cfg)}
    rows = read_rows(cfg, arm)
    assert {r["outcome"] for r in rows} == {"wrong"} and all(r["made_up"] == (not answerable[r["id"]]) for r in rows)


def test_a_profile_setting_the_env_sets_is_reset_even_when_its_default_is_unset(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(), env_extra={"MNEMIQ_WRITER_TEMPERATURE": "0.7",
                                                          "MNEMIQ_CARD_STYLE": "bare"})
    s = arm_settings(cfg, arm, arm_folder(cfg, arm, 1), None)
    assert s.writer_temperature is None and s.card_style == "cards"   # each field's own default


def test_sql_the_source_rejects_never_stops_the_arm(tmp_path, stub_model):
    # The decider passes it, and the source refuses it as it runs: every question an error, and none an outage.
    cfg, arm = one_arm(tmp_path, stub_model(sql="SELECT CAST(title AS INTEGER) AS n FROM books"))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "complete"
    kinds = [r["error_kind"] for r in read_rows(cfg, arm) if r["outcome"] == "error"]
    assert len(kinds) >= 3 and set(kinds) <= {"source", "answer"}


@pytest.mark.parametrize("status", [400, "context_too_long"])   # one per question, so 3 in a row is no outage
def test_a_per_question_model_error_never_stops_the_arm(tmp_path, stub_model, status):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, fail_first=3, status=status))   # three in a row, through Runtime.ask
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "complete"


def test_an_arm_clears_shared_state(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(), env_extra={"MNEMIQ_CONTROL_DSN": "postgresql://x/y",
                                                          "MNEMIQ_VERITY_TRACES_URL": "http://127.0.0.1:9/t"})
    s = arm_settings(cfg, arm, arm_folder(cfg, arm, 1), None)
    assert s.control_dsn is None and s.verity_traces_url is None and str(arm_folder(cfg, arm, 1)) in s.answer_log_path


def test_an_identity_with_a_row_filter_is_refused(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model, row_filter=True)))
    assert "row filter" in identity_refusal(cfg)


def test_a_case_whose_gold_reads_an_unseen_table_is_refused(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model, withhold="loans")))
    assert "loans" in identity_refusal(cfg)


def test_an_identity_that_sees_every_gold_table_unfiltered_is_not_refused(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model)))
    assert identity_refusal(cfg) is None


@pytest.mark.parametrize("status", [500, 429, 401, 404, "connect", 403, 503])
def test_an_endpoint_that_stops_answering_stops_the_arm(tmp_path, stub_model, status):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, fail_after=3, status=status))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "endpoint"
    assert all(r["outcome"] != "error" for r in read_rows(cfg, arm))


def test_a_case_asked_again_after_an_endpoint_stop_keeps_its_error(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, fail_after=3, status=500))
    folder = arm_folder(cfg, arm, 1)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "endpoint"
    retries = json.loads((Path(cfg.sweep.out) / "retries.json").read_text())
    retried = {entry["id"] for entries in retries.values() for entry in entries}
    run_arm(cfg, arm, tuning(cfg), folder)   # the endpoint is still down
    errors = {r["id"] for r in read_rows(cfg, arm) if r["outcome"] == "error"}
    assert len(retried) == 3 and retried <= errors


def test_a_statement_cancelled_under_load_is_an_error_not_wrong(tmp_path, stub_model, monkeypatch):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    cancel_gold_once(monkeypatch)                    # the source raises its timeout on one gold query
    run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1))
    assert [r["outcome"] for r in read_rows(cfg, arm)].count("error") == 1


def test_a_store_changed_during_an_arm_makes_it_invalid(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, on_call=lambda n: n == 2 and touch_store(cfg, b"x")))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "invalid"


def test_the_source_hash_changes_with_the_source(tmp_path):
    pkg = copy_package(tmp_path)
    before = engine_build(package=pkg)["source_sha"]
    (pkg / "config.py").write_text((pkg / "config.py").read_text() + "\n# edit\n")
    assert engine_build(package=pkg)["source_sha"] != before


def test_no_key_reaches_a_file(tmp_path, stub_model, monkeypatch):
    monkeypatch.setenv("SWEEP_TEST_KEY", "sk-test-not-a-real-key")
    cfg, arm = one_arm(tmp_path, stub_model(gold=True), api_key_env="SWEEP_TEST_KEY")
    run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1))
    assert not any("sk-test-not-a-real-key" in p.read_text(errors="ignore") for p in out_files(cfg))


def test_each_deployment_answers_its_own_cases_on_its_own_copy(tmp_path, stub_model):
    stub = stub_model(gold=True)
    dataset = two_deployments(tmp_path, stub)
    cfg = load_config(write_config(tmp_path, model=stub, dataset=dataset))
    arm = start_of(cfg, "instant")
    folder = arm_folder(cfg, arm, 1)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "complete"
    db_of = {case["id"]: case["db_id"] for case in json.loads(Path(dataset["questions"]).read_text())}
    versions = {(db_of[r["id"]], r["profile_version"]) for r in read_rows(cfg, arm)}
    # Each deployment's cases carry its own version, and the two differ: no case was asked of the other's runtime.
    assert len(read_rows(cfg, arm)) == len(tuning(cfg))
    assert len(versions) == 2 and len({version for _, version in versions}) == 2
    stores = {arm_settings(cfg, arm, folder, db_id).store_path for db_id in ("lib-a", "lib-b")}
    assert len(stores) == 2 and all(Path(store).is_file() for store in stores)


def test_an_arm_can_be_handed_to_a_process_of_its_own(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    args = (cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1))
    function, copied = pickle.loads(pickle.dumps((run_arm, args)))
    assert function is run_arm and copied == args and copied[0].path == cfg.path

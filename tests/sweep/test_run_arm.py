"""One arm of a sweep: run on the product path over its own copies, graded by `run_case`, one row per case and mode, and
stopped when its endpoint stops answering, never by one question's error."""

from __future__ import annotations

import fcntl
import json
import pickle
from pathlib import Path

import pytest

from mnemiq.sweep.config import deployment_settings, load_config
from mnemiq.sweep.run import ArmInUse, arm_folder, arm_settings, engine_build, identity_refusal, run_arm

from .conftest import (
    cancel_gold_once, copy_package, judged_by, one_arm, out_files, read_rows, source_down, start_of, tiny_deployment,
    touch_store, tuning, two_deployments, write_config,
)
from .stub_model import _carries, _messages


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
    # The decider passes it, and the source refuses it on every repair: each question the arm's own error, none an
    # outage.
    cfg, arm = one_arm(tmp_path, stub_model(sql="SELECT CAST(title AS INTEGER) AS n FROM books"))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "complete"
    kinds = [r["error_kind"] for r in read_rows(cfg, arm) if r["outcome"] == "error"]
    assert len(kinds) >= 3 and set(kinds) == {"answer"}


def test_a_source_that_stops_answering_is_the_sources_error_and_never_stops_the_arm(tmp_path, stub_model, monkeypatch):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    source_down(monkeypatch)
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "complete"
    kinds = [r["error_kind"] for r in read_rows(cfg, arm) if r["outcome"] == "error"]
    assert len(kinds) >= 3 and set(kinds) == {"source"}


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


def test_an_identity_with_a_mask_is_refused(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model, mask=True)))
    assert "mask" in identity_refusal(cfg)


def test_a_case_whose_gold_reads_an_unseen_table_is_refused(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model, withhold="loans")))
    refusal = identity_refusal(cfg)
    assert "loans" in refusal
    assert "loans-by-genre" in refusal and "open-loans" in refusal and "book-count" not in refusal   # by case id


def test_an_access_policy_that_cannot_be_read_is_named_as_such(tmp_path, stub_model):
    d = tiny_deployment(tmp_path, stub_model)
    d.authz.write_text("{ not json")
    refusal = identity_refusal(load_config(write_config(tmp_path, deployment=d)))
    assert "cannot be read" in refusal and "book-count" not in refusal


def test_an_identity_that_sees_every_gold_table_unfiltered_is_not_refused(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model)))
    assert identity_refusal(cfg) is None


@pytest.mark.parametrize("status", [500, 429, 401, 404, "connect", 403, 503])
def test_an_endpoint_that_stops_answering_stops_the_arm(tmp_path, stub_model, status):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, fail_after=3, status=status))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "endpoint"
    assert all(r["outcome"] != "error" for r in read_rows(cfg, arm))


@pytest.mark.parametrize("status", [503, "connect"])
def test_an_embedder_that_stops_answering_stops_the_arm(tmp_path, stub_model, status):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, embed_status=status))   # retrieval embeds the question first
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "endpoint"
    assert read_rows(cfg, arm) == []


@pytest.mark.parametrize("judge, ends", [(dict(fail_after=0, status=503), "endpoint"),
                                         (dict(rule=lambda request: "no score in this reply"), "complete")])
def test_a_judge_on_its_own_endpoint_stops_the_arm_only_when_it_stops_answering(tmp_path, stub_model, judge, ends):
    judging = stub_model(**judge)
    cfg, arm = one_arm(tmp_path, stub_model(gold=True), modes=["thinking"], judge=judged_by(judging))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == ends and judging.calls >= 3


def test_a_fail_open_judge_that_is_down_is_on_every_answered_row(tmp_path, stub_model):
    judging = stub_model(fail_after=0, status=503)
    cfg, arm = one_arm(tmp_path, stub_model(gold=True), env_extra={"MNEMIQ_VERIFY_FAIL_CLOSED": "0"},
                       modes=["thinking"], judge=judged_by(judging))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "complete"
    assert {r["verify_layer"] for r in read_rows(cfg, arm) if r["outcome"] == "correct"} == {"judge_unavailable"}


def held_for_retry(cfg) -> list[tuple[str, str]]:
    """The cases and modes `retries.json` holds for the endpoint to be asked again, each as listed."""
    path = Path(cfg.sweep.out) / "retries.json"
    entries = json.loads(path.read_text()) if path.exists() else {}
    return [(entry["id"], entry["mode"]) for listed in entries.values() for entry in listed]


def test_a_resume_while_the_endpoint_is_still_down_writes_no_error_and_stops_again(tmp_path, stub_model):
    stub = stub_model(gold=True, fail_after=3, status=500)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "endpoint"
    held = held_for_retry(cfg)
    assert len(held) == 3
    assert run_arm(cfg, arm, tuning(cfg), folder) == "endpoint"   # the endpoint is still down
    # Nothing it failed was written, the held cases are held still, and none is listed twice.
    assert all(r["outcome"] != "error" for r in read_rows(cfg, arm))
    assert set(held) <= set(held_for_retry(cfg)) and len(held_for_retry(cfg)) == len(set(held_for_retry(cfg)))
    assert not {case for case, _ in held} & {r["id"] for r in read_rows(cfg, arm)}
    stub.recover()
    assert run_arm(cfg, arm, tuning(cfg), folder) == "complete"
    rows = read_rows(cfg, arm)
    assert sorted(r["id"] for r in rows) == sorted(case.id for case in tuning(cfg))
    assert all(r["error_kind"] is None for r in rows)


def test_a_held_case_that_fails_again_once_the_endpoint_answers_others_keeps_its_error(tmp_path, stub_model):
    stub = stub_model(gold=True, fail_after=3, status=500)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "endpoint"
    held = {case for case, _ in held_for_retry(cfg)}
    questions = [case.question for case in tuning(cfg) if case.id in held]
    # The endpoint answers again, except that each held question fails as a server error would: its own failure.
    stub.recover()
    stub.rule = lambda request: 500 if any(_carries(_messages(request)[1], q) for q in questions) else None
    assert run_arm(cfg, arm, tuning(cfg), folder) == "complete"
    errors = {r["id"] for r in read_rows(cfg, arm) if r["error_kind"] == "endpoint"}
    assert len(read_rows(cfg, arm)) == len(tuning(cfg)) and errors == held


def test_a_statement_cancelled_under_load_is_an_error_not_wrong(tmp_path, stub_model, monkeypatch):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    cancel_gold_once(monkeypatch)                    # the source raises its timeout on one gold query
    run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1))
    assert [r["outcome"] for r in read_rows(cfg, arm)].count("error") == 1
    # The answer's own statement ran and showed rows; grading it could not, so the source failed, not the arm.
    errors = [r for r in read_rows(cfg, arm) if r["outcome"] == "error"]
    assert [(r["error_kind"], r["rows_ms"] is not None) for r in errors] == [("source", True)]


def test_a_store_changed_during_an_arm_makes_it_invalid(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True, on_call=lambda n: n == 2 and touch_store(cfg, b"x")))
    assert run_arm(cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1)) == "invalid"


def test_a_pin_that_moves_between_a_stop_and_its_resume_makes_the_arm_invalid(tmp_path, stub_model):
    stub = stub_model(gold=True, fail_after=3, status=500)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "endpoint"
    touch_store(cfg, b"x")
    stub.recover()
    assert run_arm(cfg, arm, tuning(cfg), folder) == "invalid"


def test_an_invalid_arm_outranks_an_endpoint_stop_and_stays_invalid(tmp_path, stub_model):
    stub = stub_model(gold=True, fail_after=3, status=500, on_call=lambda n: n == 2 and touch_store(cfg, b"x"))
    cfg, arm = one_arm(tmp_path, stub)
    store = Path(deployment_settings(cfg, None).store_path)
    original = store.read_bytes()
    folder = arm_folder(cfg, arm, 1)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "invalid"
    store.write_bytes(original)   # put back, so only the arm's own record says the store moved
    stub.recover()
    asked = stub.calls
    assert run_arm(cfg, arm, tuning(cfg), folder) == "invalid" and stub.calls == asked   # nothing asked again


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


def test_an_arm_another_process_is_running_is_refused_and_left_as_it_is(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    folder.mkdir(parents=True)
    with open(folder / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)   # as a worker a stopped sweep left running holds it
        with pytest.raises(ArmInUse, match="another process"):
            run_arm(cfg, arm, tuning(cfg), folder)
    assert stub.calls == 0 and not (folder / "rows.jsonl").exists() and not (folder / "pins.json").exists()


def test_an_arm_can_be_handed_to_a_process_of_its_own(tmp_path, stub_model):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    args = (cfg, arm, tuning(cfg), arm_folder(cfg, arm, 1))
    function, copied = pickle.loads(pickle.dumps((run_arm, args)))
    assert function is run_arm and copied == args and copied[0].path == cfg.path


def test_a_resume_that_stops_while_writing_the_profile_file_leaves_the_last_one_whole(tmp_path, stub_model,
                                                                                      monkeypatch):
    cfg, arm = one_arm(tmp_path, stub_model(gold=True))
    folder = arm_folder(cfg, arm, 1)
    run_arm(cfg, arm, tuning(cfg)[:1], folder)
    whole = (folder / "profile.toml").read_text()
    write_text = Path.write_text

    def stopped_halfway(self, text, *args, **kwargs):
        if "profile.toml" in self.name:
            write_text(self, text[:len(text) // 2], *args, **kwargs)
            raise OSError("the process stopped here")
        return write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", stopped_halfway)
    with pytest.raises(OSError, match="stopped here"):
        run_arm(cfg, arm, tuning(cfg), folder)
    assert (folder / "profile.toml").read_text() == whole

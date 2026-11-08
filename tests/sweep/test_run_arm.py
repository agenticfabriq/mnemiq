"""One arm of a sweep: run on the product path over its own copies, graded by `run_case`, one row per case and mode, and
stopped when its endpoint stops answering, never by one question's error."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import pickle
from pathlib import Path

import pytest

from mnemiq.agent.loop import Agent
from mnemiq.config import DEFAULT_RETRIEVAL_K
from mnemiq.runtime import Runtime
from mnemiq.sweep.config import SweepConfigError, deployment_settings, load_config
from mnemiq.sweep.run import (
    ArmInUse, _Job, _Rounds, _work, arm_folder, arm_settings, engine_build, identity_refusal, run_arm,
)

from .conftest import (
    arm_with, cancel_gold_once, copy_package, judged_by, one_arm, out_files, read_rows, source_down, start_of,
    tiny_deployment, touch_store, tuning, two_deployments, write_config,
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


def own_failures(stub, cfg, case_ids) -> None:
    """From now on `stub` fails each of `case_ids`' questions with a 500 of its own, and answers every other request."""
    questions = [case.question for case in tuning(cfg) if case.id in set(case_ids)]
    stub.recover()
    stub.rule = lambda request: 500 if any(_carries(_messages(request)[1], q) for q in questions) else None


def dies_before(stub, case_ids):
    """A `before` callback for `run_arm`: just before any of `case_ids` is asked, `stub` stops answering anything."""
    def before(case_id: str, mode: str) -> None:
        if case_id in set(case_ids):
            stub.rule = lambda request: 500
    return before


def test_held_questions_left_alone_finish_once_a_probe_shows_the_endpoint_answers(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    last = [case.id for case in tuning(cfg)[-3:]]
    assert run_arm(cfg, arm, tuning(cfg), folder, before=dies_before(stub, last[:1])) == "endpoint"
    held = [case for case, _ in held_for_retry(cfg)]
    # Only the held questions are left, and each now fails on its own while the endpoint answers everything else.
    assert held == last and len(read_rows(cfg, arm)) + len(held) == len(tuning(cfg))
    own_failures(stub, cfg, held)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "complete"
    assert sorted(r["id"] for r in read_rows(cfg, arm) if r["error_kind"] == "endpoint") == sorted(held)
    assert held_for_retry(cfg) == []   # each entry left retries.json as its error became final


def stopped_on_three_before_the_last(stub, cfg, arm, folder) -> list[str]:
    """Run `arm` until the endpoint stops answering just before its fourth question from the end, which holds three
    questions and leaves the last one never asked; then the endpoint answers again. The held questions, in order."""
    assert run_arm(cfg, arm, tuning(cfg), folder, before=dies_before(stub, [tuning(cfg)[-4].id])) == "endpoint"
    held = [case for case, _ in held_for_retry(cfg)]
    assert held == [case.id for case in tuning(cfg)[-4:-1]]
    stub.rule = None
    return held


def test_an_endpoint_that_dies_after_answering_one_question_keeps_the_held_questions_held(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    held = stopped_on_three_before_the_last(stub, cfg, arm, folder)
    # The one question never asked is answered, then the endpoint stops answering before the held ones are asked.
    assert run_arm(cfg, arm, tuning(cfg), folder, before=dies_before(stub, held)) == "endpoint"
    rows = read_rows(cfg, arm)
    assert len(rows) == len(tuning(cfg)) - len(held) and all(r["error_kind"] is None for r in rows)
    assert sorted(case for case, _ in held_for_retry(cfg)) == sorted(held)


def test_a_streak_of_questions_failing_on_their_own_settles_on_the_next_resume(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    last = [case.id for case in tuning(cfg)[-3:]]
    own_failures(stub, cfg, last)   # three in a row, at the end: they look like an outage
    assert run_arm(cfg, arm, tuning(cfg), folder) == "endpoint"
    assert sorted(case for case, _ in held_for_retry(cfg)) == sorted(last)
    assert run_arm(cfg, arm, tuning(cfg), folder) == "complete"
    assert sorted(r["id"] for r in read_rows(cfg, arm) if r["error_kind"] == "endpoint") == sorted(last)
    assert held_for_retry(cfg) == []


def test_retries_lose_each_entry_as_it_settles(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    first, second, third = stopped_on_three_before_the_last(stub, cfg, arm, folder)
    # The first held question fails on its own and settles; the endpoint dies before the second is asked.
    own_failures(stub, cfg, [first, second, third])
    assert run_arm(cfg, arm, tuning(cfg), folder, before=dies_before(stub, [second])) == "endpoint"
    assert [case for case, _ in held_for_retry(cfg)] == [second, third]
    assert [r["id"] for r in read_rows(cfg, arm) if r["error_kind"] == "endpoint"] == [first]


def test_an_endpoint_that_dies_on_the_arms_last_questions_leaves_them_held(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    last = [case.id for case in tuning(cfg)[-2:]]   # two: too few for a streak
    assert run_arm(cfg, arm, tuning(cfg), folder, before=dies_before(stub, last[:1])) == "endpoint"
    assert all(r["error_kind"] is None for r in read_rows(cfg, arm))
    assert sorted(case for case, _ in held_for_retry(cfg)) == sorted(last)


def test_with_nothing_answered_to_replay_a_held_question_stays_held(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    own_failures(stub, cfg, [case.id for case in tuning(cfg)])   # no question of this arm, its start, is answered
    # Two streaks of three, then the last question beside the first held one: every question held, none written.
    assert [run_arm(cfg, arm, tuning(cfg), folder) for _ in range(4)] == ["endpoint"] * 4
    assert read_rows(cfg, arm) == [] and len(held_for_retry(cfg)) == len(tuning(cfg)) == 7


def declined_unread(monkeypatch, question: str) -> None:
    """From now on the engine declines `question` as too long to read whole, before any model call, wherever it is
    asked."""
    original = Agent.decline_if_too_long

    def decline(self, asked: str):
        return original(self, "x" * (self.max_question_chars + 1) if asked == question else asked)

    monkeypatch.setattr(Agent, "decline_if_too_long", decline)


def test_a_probe_replays_only_a_question_that_called_the_model(tmp_path, stub_model, monkeypatch):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    cases = tuning(cfg)
    declined_unread(monkeypatch, cases[-4].question)   # the last question answered before the endpoint dies
    last = [case.id for case in cases[-3:]]
    assert run_arm(cfg, arm, cases, folder, before=dies_before(stub, last[:1])) == "endpoint"
    assert read_rows(cfg, arm)[-1]["calls"] == 0 and [case for case, _ in held_for_retry(cfg)] == last
    own_failures(stub, cfg, last)   # the endpoint answers everything else now
    assert run_arm(cfg, arm, cases, folder) == "complete"
    assert sorted(r["id"] for r in read_rows(cfg, arm) if r["error_kind"] == "endpoint") == sorted(last)


def test_a_probe_falls_back_to_a_question_the_start_answered_with_the_model(tmp_path, stub_model, monkeypatch):
    stub = stub_model(gold=True)
    cfg, start = one_arm(tmp_path, stub)
    cases = tuning(cfg)
    declined_unread(monkeypatch, cases[0].question)   # the start's first row: answered with no model call
    assert run_arm(cfg, start, cases, arm_folder(cfg, start, 1)) == "complete"
    arm = arm_with(cfg, retrieval_k=DEFAULT_RETRIEVAL_K + 4)
    folder = arm_folder(cfg, arm, 1)
    mine = cases[2:5]   # not the start's second question, the first it answered by calling the model
    ids = [case.id for case in mine]
    # The arm asks three questions and the endpoint dies before the first, so no question of the arm's own is
    # answered: a probe can replay only one the start answered.
    assert run_arm(cfg, arm, mine, folder, before=dies_before(stub, ids[:1])) == "endpoint"
    assert read_rows(cfg, arm) == [] and [case for case, _ in held_for_retry(cfg)] == ids
    own_failures(stub, cfg, ids)
    assert run_arm(cfg, arm, mine, folder) == "complete"
    assert sorted(r["id"] for r in read_rows(cfg, arm) if r["error_kind"] == "endpoint") == sorted(ids)


def test_a_probe_that_raises_leaves_the_held_questions_held_and_says_why(tmp_path, stub_model, monkeypatch, capsys):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    folder = arm_folder(cfg, arm, 1)
    last = [case.id for case in tuning(cfg)[-3:]]
    assert run_arm(cfg, arm, tuning(cfg), folder, before=dies_before(stub, last[:1])) == "endpoint"
    own_failures(stub, cfg, last)
    ask = Runtime.ask

    def broken(self, question, identity, mode=None, emit=None, history=None, profile=None):
        if emit is None:   # the probe passes none; every graded question does
            raise RuntimeError("the probe broke reading postgresql://sweep:SECRETPW@db.example/library")
        return ask(self, question, identity, mode=mode, emit=emit, history=history, profile=profile)

    monkeypatch.setattr(Runtime, "ask", broken)
    # As a worker runs it: what it sends back is an endpoint stop with the probe's failure, not a worker that died.
    receiver, sender = multiprocessing.Pipe(duplex=False)
    _work(cfg, arm, tuning(cfg), folder, sender, [])
    outcome = receiver.recv()
    assert outcome[0] == "endpoint" and "probe" in outcome[1] and "RuntimeError" in outcome[1]
    assert "SECRETPW" not in outcome[1]
    assert all(r["error_kind"] is None for r in read_rows(cfg, arm)) and [c for c, _ in held_for_retry(cfg)] == last
    rounds = object.__new__(_Rounds)   # what the sweep does with it: an endpoint stop, said with its reason
    rounds.meter, rounds.state, rounds.halt, rounds.secrets, rounds.dead, rounds.died = None, None, None, [], [], []
    rounds._ended(_Job(arm, 1, tuple(tuning(cfg)), folder), outcome, 1)
    assert rounds.dead == ["m"] and not rounds.died and outcome[1] in capsys.readouterr().out


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


def test_an_arm_refuses_to_run_under_another_engine_build_than_its_sweeps(tmp_path, stub_model):
    stub = stub_model(gold=True)
    cfg, arm = one_arm(tmp_path, stub)
    out = Path(cfg.sweep.out)
    out.mkdir(parents=True)
    # As after an edit or a pull during the sweep: the worker imports the engine from disk, and it is another build.
    (out / "sweep.json").write_text(json.dumps({"engine_build": {"commit": None, "source_sha": "f" * 64}}))
    folder = arm_folder(cfg, arm, 1)
    asked = stub.calls
    with pytest.raises(SweepConfigError, match="engine build") as refused:
        run_arm(cfg, arm, tuning(cfg), folder)
    assert "ffffffffffff" in str(refused.value) and engine_build()["source_sha"][:12] in str(refused.value)
    assert stub.calls == asked and not (folder / "rows.jsonl").exists() and not (folder / "pins.json").exists()


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

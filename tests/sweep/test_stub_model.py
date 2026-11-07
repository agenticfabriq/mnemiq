"""The scripted model answers the engine's real calls, so a sweep test can drive `Runtime.ask` and grade with `run_case`
over the tiny deployment, with nothing scripted but the model."""

from __future__ import annotations

import httpx
import pytest

from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.eval.harness import Outcome, run_case
from mnemiq.llm.client import ContextTooLong, LLMClient, ModelUnavailable
from mnemiq.runtime import build_runtime
from mnemiq.sweep.config import deployment_settings, load_config, load_questions

from .conftest import CASES, ROLE, tiny_deployment, write_config

WHO = IdentityContext(tenant_id="local", principal_id="sweep", roles=[ROLE])
ANSWERABLE = [case["id"] for case in CASES if case.get("answerable", True)]
UNANSWERABLE = [case["id"] for case in CASES if not case.get("answerable", True)]


def _graded(tmp_path, stub, mode="instant", **settings) -> tuple[dict[str, Outcome], dict[str, str]]:
    """Each tiny case asked through the product path on `stub`: its outcome, and the SQL the answer ran."""
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub)))
    runtime = build_runtime(deployment_settings(cfg).model_copy(update={"llm_base_url": stub.url, **settings}))
    results = [run_case(case, lambda question: runtime.ask(question, WHO, mode=mode), runtime.adapter)
               for case in load_questions(cfg)]
    return {r.case_id: r.outcome for r in results}, {r.case_id: r.sql for r in results}


@pytest.mark.parametrize("mode", ["instant", "thinking", "deep"])
def test_a_gold_model_is_graded_right_in_every_mode(tmp_path, stub_model, mode):
    outcomes, _ = _graded(tmp_path, stub_model(gold=True), mode)
    assert outcomes == {**dict.fromkeys(ANSWERABLE, Outcome.CORRECT),
                        **dict.fromkeys(UNANSWERABLE, Outcome.DEFERRED_CORRECTLY)}


def test_a_model_that_is_not_gold_is_wrong_and_makes_answers_up(tmp_path, stub_model):
    outcomes, _ = _graded(tmp_path, stub_model)
    assert outcomes == dict.fromkeys(ANSWERABLE + UNANSWERABLE, Outcome.WRONG)


def test_start_misses_are_the_first_answerable_cases(tmp_path, stub_model):
    outcomes, _ = _graded(tmp_path, stub_model(gold=True, start_misses=2))
    assert [outcomes[case] for case in ANSWERABLE] == [Outcome.WRONG] * 2 + [Outcome.CORRECT] * (len(ANSWERABLE) - 2)


@pytest.mark.parametrize("guided_sql, right", [(True, Outcome.CORRECT), (False, Outcome.WRONG)])
def test_a_rule_can_answer_one_setting_differently(tmp_path, stub_model, guided_sql, right):
    stub = stub_model(gold_when=lambda request: "response_format" in request)
    outcomes, _ = _graded(tmp_path, stub, guided_sql=guided_sql)
    assert {outcomes[case] for case in ANSWERABLE} == {right}


def test_repeated_asks_vary_the_spelling_and_keep_the_rows(tmp_path, stub_model):
    stub = stub_model(gold=True, varies=True)
    first_outcomes, first = _graded(tmp_path, stub)
    second_outcomes, second = _graded(tmp_path, stub)
    assert {first_outcomes[case] for case in ANSWERABLE} == {second_outcomes[case] for case in ANSWERABLE} == {
        Outcome.CORRECT}
    assert all(first[case] != second[case] for case in ANSWERABLE)


def _client(stub) -> LLMClient:
    return LLMClient(Settings(llm_base_url=stub.url, llm_api_key="stub-key", llm_model="stub-model"))


@pytest.mark.parametrize("status, raised", [(503, ModelUnavailable), (401, ModelUnavailable),
                                            ("context_too_long", ContextTooLong)])
def test_a_failure_status_reaches_the_client_once(stub_model, status, raised):
    stub = stub_model(fail_first=1, status=status)
    with pytest.raises(raised) as failed:
        _client(stub).complete("system", "a user message")
    assert raised is ContextTooLong or not isinstance(failed.value, ContextTooLong)
    assert stub.calls == 1
    assert _client(stub).complete("system", "a user message") == "{}"


def test_a_closed_connection_is_retried_by_the_client_and_then_fails(stub_model):
    stub = stub_model(fail_after=0, status="connect")
    with pytest.raises(ModelUnavailable):
        _client(stub).complete("system", "a user message")
    assert stub.calls == 3


def test_recover_ends_a_failure(stub_model):
    seen: list[int] = []
    stub = stub_model(fail_after=1, status=500, on_call=seen.append)
    client = _client(stub)
    client.complete("system", "a user message")
    with pytest.raises(ModelUnavailable):
        client.complete("system", "a user message")
    stub.recover()
    client.complete("system", "a user message")
    assert seen == [1, 2, 3]


def test_vectors_are_the_texts_own_and_metrics_report_what_is_set(stub_model):
    other = stub_model()
    vectors = [httpx.post(f"{stub.url}/embeddings", json={"model": "e", "input": ["a card", "another"]}).json()
               for stub in (stub_model, other)]
    assert vectors[0]["data"] == vectors[1]["data"]
    assert vectors[0]["data"][0]["embedding"] != vectors[0]["data"][1]["embedding"]
    other.running = 3
    assert "vllm:num_requests_running" in httpx.get(other.url.removesuffix("/v1") + "/metrics").text
    assert httpx.get(other.url.removesuffix("/v1") + "/metrics").text.rstrip().endswith(" 3.0")

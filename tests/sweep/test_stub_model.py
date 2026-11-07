"""The scripted model answers the engine's real calls, so a sweep test can drive `Runtime.ask` and grade with `run_case`
over the tiny deployment, with nothing scripted but the model."""

from __future__ import annotations

import json

import httpx
import pytest

from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.eval.harness import CaseResult, Outcome, run_case
from mnemiq.llm.client import ContextTooLong, LLMClient, ModelUnavailable
from mnemiq.runtime import build_runtime
from mnemiq.sweep.config import deployment_settings, load_config, load_questions

from .conftest import CASES, ROLE, tiny_deployment, write_config

pytest_plugins = ["pytester"]

WHO = IdentityContext(tenant_id="local", principal_id="sweep", roles=[ROLE])
ANSWERABLE = [case["id"] for case in CASES if case.get("answerable", True)]
UNANSWERABLE = [case["id"] for case in CASES if not case.get("answerable", True)]


def _graded(tmp_path, stub, mode="instant", **settings) -> dict[str, CaseResult]:
    """Each tiny case asked through the product path on `stub`, graded by `run_case`."""
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub)))
    runtime = build_runtime(deployment_settings(cfg).model_copy(update={"llm_base_url": stub.url, **settings}))
    return {case.id: run_case(case, lambda question: runtime.ask(question, WHO, mode=mode), runtime.adapter)
            for case in load_questions(cfg)}


def _rows(rows: list[dict]) -> list[str]:
    """Rows in no particular order, as `run_case` compares them: a GROUP BY may return them in any."""
    return sorted(json.dumps(list(row.values()), default=str) for row in rows)


def _outcomes(results: dict[str, CaseResult]) -> dict[str, Outcome]:
    return {case: result.outcome for case, result in results.items()}


@pytest.mark.parametrize("mode", ["instant", "thinking", "deep"])
def test_a_gold_model_is_graded_right_in_every_mode(tmp_path, stub_model, mode):
    assert _outcomes(_graded(tmp_path, stub_model(gold=True), mode)) == {
        **dict.fromkeys(ANSWERABLE, Outcome.CORRECT), **dict.fromkeys(UNANSWERABLE, Outcome.DEFERRED_CORRECTLY)}


def test_a_model_that_is_not_gold_is_wrong_and_makes_answers_up(tmp_path, stub_model):
    assert _outcomes(_graded(tmp_path, stub_model)) == dict.fromkeys(ANSWERABLE + UNANSWERABLE, Outcome.WRONG)


@pytest.mark.parametrize("options", [{"start_misses": 2}, {"start_misses": 2, "gold": True}],
                         ids=["alone", "with-gold"])
def test_start_misses_are_exactly_the_first_answerable_cases(tmp_path, stub_model, options):
    outcomes = _outcomes(_graded(tmp_path, stub_model(**options)))
    assert [outcomes[case] for case in ANSWERABLE] == [Outcome.WRONG] * 2 + [Outcome.CORRECT] * (len(ANSWERABLE) - 2)
    assert {outcomes[case] for case in UNANSWERABLE} == {Outcome.DEFERRED_CORRECTLY}


@pytest.mark.parametrize("guided_sql, right", [(True, Outcome.CORRECT), (False, Outcome.WRONG)])
def test_a_rule_can_answer_one_setting_differently(tmp_path, stub_model, guided_sql, right):
    stub = stub_model(gold_when=lambda request: "response_format" in request)
    outcomes = _outcomes(_graded(tmp_path, stub, guided_sql=guided_sql))
    assert {outcomes[case] for case in ANSWERABLE} == {right}


@pytest.mark.parametrize("options, asked, outcome", [
    ({"varies": True}, ANSWERABLE, Outcome.CORRECT),
    ({"varies": True, "gold": True}, ANSWERABLE, Outcome.CORRECT),
    ({"varies": True, "gold_when": lambda request: False}, ANSWERABLE + UNANSWERABLE, Outcome.WRONG),
], ids=["alone", "with-gold", "wrong"])
def test_repeated_asks_vary_the_sql_and_keep_the_rows(tmp_path, stub_model, options, asked, outcome):
    stub = stub_model(**options)
    first, second = _graded(tmp_path, stub), _graded(tmp_path, stub)
    for case in asked:
        assert first[case].sql and second[case].sql and first[case].sql != second[case].sql, case
        assert first[case].outcome == second[case].outcome == outcome, case
        if case in ANSWERABLE:  # `run_case` runs no answer to a question that has none
            assert _rows(first[case].engine_rows) == _rows(second[case].engine_rows), case


def test_every_spelling_returns_its_answers_rows(tmp_path):
    """The tiny cases themselves: each variant returns its answer's rows, and each wrong answer other rows than gold."""
    from mnemiq.adapters.resolve import adapter_for

    settings = deployment_settings(load_config(write_config(tmp_path)))
    adapter = adapter_for(settings.source_specs()[0], settings)

    def rows(sql: str) -> list[str]:
        return _rows(adapter.execute_arrow(sql, timeout_s=30).to_pylist())

    for case in CASES:
        wrong = rows(case["wrong_sql"])
        assert wrong and all(rows(variant) == wrong for variant in case["wrong_variants"]), case["id"]
        if case.get("answerable", True):
            gold = rows(case["gold_sql"])
            assert gold != wrong and all(rows(variant) == gold for variant in case["variants"]), case["id"]


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


_RAISING = """
import httpx

from mnemiq.generate.prompts import system_prompt


def test_asks_through_a_predicate(stub_model):
    stub = stub_model(gold_when=lambda request: {predicate})
    stub.learn([{{"id": "q", "question": "How many?", "gold_sql": "SELECT 1"}}])
    httpx.post(f"{{stub.url}}/chat/completions", json={{"model": "m", "messages": [
        {{"role": "system", "content": system_prompt()}}, {{"role": "user", "content": "QUESTION: How many?"}}]}})
"""


@pytest.mark.parametrize("predicate, errors", [("1 / 0", 1), ("True", 0)], ids=["raises", "control"])
def test_a_predicate_that_raises_fails_the_test_that_used_it(pytester, predicate, errors):
    """Run as its own session, so the failure is the real fixture's, at teardown, and the test body itself passes."""
    pytester.makeconftest("from tests.sweep.conftest import stub_model  # noqa: F401 -- the fixture under test\n")
    pytester.makepyfile(_RAISING.format(predicate=predicate))
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, errors=errors)
    if errors:
        result.stdout.fnmatch_lines(["*raised 1 time(s); the first: ZeroDivisionError*"])

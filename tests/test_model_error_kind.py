"""A failed answer says what kind of model error failed it, as the client read it from the provider's error: no answer
from the server, an HTTP status, a prompt too long for the window, or a prompt the server cut. Never from the words."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from mnemiq.contract import DeferralReason
from mnemiq.runtime import build_runtime
from tests.scripted_deployment import IDENTITY, SQL, deployment, retrieve_the_one_table
from tests.sweep.stub_model import StubModel


def _stub(raised) -> StubModel:
    if raised == "prompt_cut":
        # The first reply reads ten tokens of a prompt far longer, and the check that would confirm the cut fails with a
        # 400: the cut is still the kind, not the check's status.
        return StubModel(sql=SQL, prompt_tokens=10, fail_after=1, status=400)
    if raised == "other":
        return StubModel(sql=SQL)
    # Every chat request fails. A too-long prompt arrives as a 400, so its kind is read before the status.
    return StubModel(sql=SQL, fail_after=0, status=raised)


@pytest.fixture
def scripted_llm_error(tmp_path, monkeypatch):
    """`scripted_llm_error(raised)`: a deployment whose model fails as `raised` says, with `ask(question)`."""
    stubs: list[StubModel] = []

    def deployment_failing(raised):
        retrieve_the_one_table(monkeypatch)
        stub = _stub(raised)
        stubs.append(stub)
        if raised == "other":
            # The provider's error is neither a connection's nor a status's: no reply over HTTP raises one, so the
            # provider's own call raises it.
            def create(self, *args, **kwargs):
                raise openai.APIError("the provider sent no usable reply",
                                      httpx.Request("POST", f"{stub.url}/chat/completions"), body=None)

            monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", create)
        runtime = build_runtime(deployment(tmp_path, stub.url))
        return SimpleNamespace(ask=lambda question: runtime.ask(question, IDENTITY))

    yield deployment_failing
    for stub in stubs:
        stub.stop()
        assert not stub.all_errors()


@pytest.mark.parametrize("raised, kind", [("connect", "connect"), (503, "status_503"), (429, "status_429"),
                                          (400, "status_400"), ("context_too_long", "context_too_long"),
                                          ("prompt_cut", "prompt_cut"), ("other", "other")])
def test_a_failed_answer_says_what_kind_of_model_error(scripted_llm_error, raised, kind):
    assert scripted_llm_error(raised).ask("q").model_error == kind


def test_an_answer_the_model_gave_has_no_model_error(tmp_path, monkeypatch):
    retrieve_the_one_table(monkeypatch)
    with StubModel(sql=SQL) as stub:
        answer = build_runtime(deployment(tmp_path, stub.url)).ask("q", IDENTITY)
    assert not answer.failed and answer.model_error is None


@pytest.mark.parametrize("judge, kind", [(dict(fail_after=0, status=503), "status_503"),
                                         (dict(rule=lambda request: "no score in this reply"), None)])
def test_a_judge_on_its_own_endpoint_says_what_kind_of_model_error(tmp_path, monkeypatch, judge, kind):
    retrieve_the_one_table(monkeypatch)
    with StubModel(sql=SQL) as writer, StubModel(**judge) as judging:
        settings = deployment(tmp_path, writer.url, verify_base_url=judging.url, verify_model="judge-model")
        answer = build_runtime(settings).ask("q", IDENTITY, mode="thinking")
    assert judging.calls >= 1 and answer.failed and answer.reason_code == DeferralReason.VERIFIER_UNAVAILABLE
    assert answer.model_error == kind   # a reply that could not be read is not the endpoint failing

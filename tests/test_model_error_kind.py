"""A failed answer says what kind of model error failed it, as the client read it from the provider's error: no answer
from the server, an HTTP status, a prompt too long for the window, or a prompt the server cut. Never from the words."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mnemiq.llm.client import LLMClient, ModelUnavailable
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
            # Raised with no kind, as any raise site outside the client does.
            def complete(self, *args, **kwargs):
                raise ModelUnavailable("the model did not reply")

            monkeypatch.setattr(LLMClient, "complete", complete)
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

"""What a profile spent: `Runtime.usage` sums the calls and the tokens of one profile's model clients, its writer's and
its judge's, as the server counted them, and never another profile's."""

from __future__ import annotations

import pytest

from mnemiq.runtime import build_runtime
from tests.scripted_deployment import IDENTITY, SQL, deployment, retrieve_the_one_table
from tests.sweep.stub_model import StubModel

PROFILES = """
default = "a"

[profiles.a]
llm_model = "model-a"

[profiles.b]
llm_model = "model-b"
roles = ["*"]
"""


@pytest.fixture
def scripted_llm():
    """The scripted model, answering the writer with SQL over the source's one table."""
    with StubModel(sql=SQL) as stub:
        yield stub
    assert not stub.all_errors()


@pytest.fixture
def profiles_runtime(tmp_path, scripted_llm, monkeypatch):
    """A runtime with two profiles, `a` (the default) and `b`, each with a model client of its own."""
    retrieve_the_one_table(monkeypatch)
    return build_runtime(deployment(tmp_path, scripted_llm.url, profiles=PROFILES))


def test_usage_counts_one_profiles_calls_and_not_anothers(profiles_runtime, scripted_llm):
    profiles_runtime.ask("q", IDENTITY, profile="a")
    assert profiles_runtime.usage("a").calls >= 1 and profiles_runtime.usage("b").calls == 0


@pytest.mark.parametrize("judge", ["on a client of its own", "on the writer's client"])
def test_usage_counts_the_judge_once_on_whichever_client_it_calls(tmp_path, scripted_llm, monkeypatch, judge):
    retrieve_the_one_table(monkeypatch)
    own = {"verify_model": "judge-model"} if judge == "on a client of its own" else {}
    runtime = build_runtime(deployment(tmp_path, scripted_llm.url, **own))
    answer = runtime.ask("q", IDENTITY, mode="thinking")
    assert answer.verify_confidence == scripted_llm.confidence  # the judge was asked, and answered
    usage = runtime.usage()
    assert usage.calls == scripted_llm.calls and usage.prompt_tokens > 0 and usage.completion_tokens > 0

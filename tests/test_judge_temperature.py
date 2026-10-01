"""M120: the judge reads at temperature 0, so the same SQL and rows get the same verdict.

mnemiq sent no temperature at all, so a local server sampled every call at the model's own
default -- 0.7 for Qwen2.5 -- and the default mode's judge accepted and declined the identical
query and result on two tries. Measured: five identical requests as mnemiq sent them gave five
different answers; at temperature 0, one.
"""

from types import SimpleNamespace

import pytest

from mnemiq.config import Settings
from mnemiq.llm.client import LLMClient, accepts_temperature
from mnemiq.verify.judge import SemanticJudge


class _Capturing:
    sent: dict = {}

    def __init__(self, *_, **__):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        type(self).sent = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"confidence": 0.9}'))],
            usage=None)


def _client(monkeypatch, model: str) -> LLMClient:
    import mnemiq.llm.client as module

    monkeypatch.setattr(module, "OpenAI", _Capturing)
    _Capturing.sent = {}
    return LLMClient(Settings(llm_base_url="http://x", llm_api_key="k", llm_model=model,
                              pg_dsn=None, acme_data_dir=None))


def test_the_judge_built_the_way_the_product_builds_it_reads_at_temperature_0(monkeypatch):
    """The join: the real judge over the real client, only the transport faked."""
    SemanticJudge(_client(monkeypatch, "qwen2.5-coder-32b")).score("q", "schema", "SELECT 1", "rows")
    assert _Capturing.sent["temperature"] == 0


def test_a_caller_that_asks_for_nothing_still_sends_nothing(monkeypatch):
    # Generation keeps the server's default until a measurement says otherwise.
    _client(monkeypatch, "qwen2.5-coder-32b").complete("s", "u")
    assert "temperature" not in _Capturing.sent


@pytest.mark.parametrize("model", ["openai.gpt-5.5", "gpt-5-mini", "o3", "o4-mini", "openai.o1",
                                   "openai/o3", "azure/gpt-5"])
def test_a_reasoning_model_is_never_sent_one(monkeypatch, model):
    # They reject any temperature but their default: asked for 0, the provider fails the request.
    SemanticJudge(_client(monkeypatch, model)).score("q", "schema", "SELECT 1", "rows")
    assert "temperature" not in _Capturing.sent


@pytest.mark.parametrize("model", ["qwen2.5-coder-32b", "llama3.1:8b", "openai.gpt-4o", "o3x-local"])
def test_every_other_model_takes_it(model):
    assert accepts_temperature(model)

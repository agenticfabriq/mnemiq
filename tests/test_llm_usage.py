from types import SimpleNamespace

from mnemiq.config import Settings
from mnemiq.llm.client import LLMClient


class _FakeOpenAI:
    def __init__(self, *_, **__):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        )


def _settings() -> Settings:
    return Settings(
        llm_base_url="http://x", llm_api_key="k", llm_model="m", pg_dsn=None, acme_data_dir=None
    )


def test_usage_accumulates_across_calls(monkeypatch):
    import mnemiq.llm.client as module

    monkeypatch.setattr(module, "OpenAI", _FakeOpenAI)
    client = LLMClient(_settings())
    assert client.calls == 0 and client.total_tokens == 0

    client.complete("s", "u")
    client.complete("s", "u")

    assert client.calls == 2
    assert client.prompt_tokens == 20
    assert client.completion_tokens == 6
    assert client.total_tokens == 26


def test_a_response_without_usage_does_not_crash(monkeypatch):
    import mnemiq.llm.client as module

    class _NoUsage(_FakeOpenAI):
        def _create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))], usage=None
            )

    monkeypatch.setattr(module, "OpenAI", _NoUsage)
    client = LLMClient(_settings())
    client.complete("s", "u")
    assert client.calls == 1 and client.total_tokens == 0  # counted the call, no tokens to add

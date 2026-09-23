"""LLMClient construction. `complete()`'s behaviour (token params, reasoning budget, retries)
has no dedicated suite yet -- these tests cover only what fix round 1 item 6b touched: whether
the underlying OpenAI client is built to follow redirects."""

from mnemiq.config import Settings


def _captured_openai_kwargs(monkeypatch, settings) -> dict:
    from mnemiq.llm import client as mod

    captured: dict = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mod, "OpenAI", fake_openai)
    mod.LLMClient(settings)
    return captured


def test_llm_client_disables_redirects_when_local_only(monkeypatch):
    # Same gap as LLMEmbedder (see test_embeddings.py): the OpenAI SDK's own _DefaultHttpxClient
    # sets follow_redirects=True, so a compliant local endpoint that later 302s to a public host
    # would be followed there silently -- boot's assert_local_only checks the CONFIGURED
    # base_url once, not where a live response's Location header points on every later call.
    settings = Settings(llm_base_url="http://127.0.0.1:8000/v1", llm_api_key="k", local_only=True)
    kwargs = _captured_openai_kwargs(monkeypatch, settings)
    http_client = kwargs.get("http_client")
    assert http_client is not None
    assert http_client.follow_redirects is False


def test_llm_client_keeps_default_redirects_when_not_local_only(monkeypatch):
    # local_only defaults False: an existing hosted deployment must construct the client
    # byte-identically to before this fix, i.e. no http_client override at all.
    settings = Settings(llm_base_url="http://127.0.0.1:8000/v1", llm_api_key="k")
    kwargs = _captured_openai_kwargs(monkeypatch, settings)
    assert kwargs.get("http_client") is None

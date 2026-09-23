import os

import pytest

from mnemiq.llm.embeddings import EMBED_DIM, FakeEmbedder


def test_fake_embedder_is_deterministic_and_shaped():
    a = FakeEmbedder().embed(["claim", "policy"])
    b = FakeEmbedder().embed(["claim", "policy"])
    assert a == b
    assert len(a) == 2
    assert all(len(v) == EMBED_DIM for v in a)
    assert a[0] != a[1]


def test_fake_embedder_vectors_are_normalized():
    (v,) = FakeEmbedder().embed(["claim"])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6  # unit length: cosine == dot product


def test_fake_embedder_handles_empty_input():
    assert FakeEmbedder().embed([]) == []


def test_the_protocol_carries_the_dimension():
    # The index DDL is built from this, so an embedder that cannot state its width silently
    # writes into a column of the wrong one.
    from mnemiq.llm.embeddings import FakeEmbedder

    assert FakeEmbedder().dim == EMBED_DIM
    assert FakeEmbedder(dim=768).dim == 768
    assert len(FakeEmbedder(dim=768).embed(["x"])[0]) == 768


@pytest.mark.live_llm
@pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured")
def test_live_embedder_puts_related_text_closer():
    from mnemiq.config import Settings
    from mnemiq.llm.embeddings import LLMEmbedder

    vectors = LLMEmbedder(Settings.from_env()).embed(
        ["an insurance claim for fire damage", "a fire loss claim", "the price of tea"]
    )
    assert all(len(v) == EMBED_DIM for v in vectors)

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])


class _Recorder:
    """Stands in for the OpenAI client, recording what each call was asked to embed."""

    def __init__(self, fail_while_longer_than: int | None = None, error: str | None = None,
                 dim: int = EMBED_DIM):
        self.sent: list[list[str]] = []
        self._fail_over = fail_while_longer_than
        self._error = error or "maximum input length is 8192 tokens"
        self._dim = dim
        self.embeddings = self

    def create(self, model, input):  # noqa: A002 - the provider's own parameter name
        self.sent.append(list(input))
        if self._fail_over is not None and max((len(t) for t in input), default=0) > self._fail_over:
            raise RuntimeError(self._error)
        return type("R", (), {"data": [
            type("D", (), {"index": i, "embedding": [0.0] * self._dim})() for i in range(len(input))
        ]})()


def _embedder(monkeypatch, client, **kwargs):
    from mnemiq.config import Settings
    from mnemiq.llm import embeddings as mod

    monkeypatch.setattr(mod, "OpenAI", lambda **_: client)
    return mod.LLMEmbedder(
        Settings(llm_base_url="http://x", llm_api_key="k"), **kwargs
    )


def test_llm_embedder_probes_the_endpoint_once_and_caches_the_width(monkeypatch):
    # The endpoint is the only authority on its own width -- assuming EMBED_DIM would silently
    # write into the wrong column for a local model that isn't 1536 wide.
    client = _Recorder(dim=768)
    emb = _embedder(monkeypatch, client)
    assert emb.dim == 768
    assert emb.dim == 768  # cached: a second access must not probe again
    assert client.sent == [["dimension probe"]]


def _captured_openai_kwargs(monkeypatch, mod, settings) -> dict:
    """Builds an embedder/client with `OpenAI` replaced by a kwarg-recording fake, and returns
    what it was constructed with. `_embedder`'s `lambda **_: client` (above) throws the kwargs
    away, which is exactly what hides whether `http_client` was ever passed."""
    captured: dict = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return _Recorder()

    monkeypatch.setattr(mod, "OpenAI", fake_openai)
    mod.LLMEmbedder(settings)
    return captured


def test_llm_embedder_disables_redirects_when_local_only(monkeypatch):
    # The OpenAI SDK's own _DefaultHttpxClient sets follow_redirects=True. assert_local_only only
    # ever validated the CONFIGURED base_url; a compliant local endpoint that later answers with
    # a 302 to a public host would have every subsequent request followed there and exfiltrate,
    # with boot having already passed. Refusing that needs the transport itself to stop
    # following redirects, not another check on a URL that was never wrong.
    from mnemiq.config import Settings
    from mnemiq.llm import embeddings as mod

    settings = Settings(llm_base_url="http://127.0.0.1:8000/v1", llm_api_key="k", local_only=True)
    kwargs = _captured_openai_kwargs(monkeypatch, mod, settings)
    http_client = kwargs.get("http_client")
    assert http_client is not None
    assert http_client.follow_redirects is False


def test_llm_embedder_keeps_default_redirects_when_not_local_only(monkeypatch):
    # local_only defaults False: an existing hosted deployment must construct the client
    # byte-identically to before this fix, i.e. no http_client override at all.
    from mnemiq.config import Settings
    from mnemiq.llm import embeddings as mod

    settings = Settings(llm_base_url="http://127.0.0.1:8000/v1", llm_api_key="k")
    kwargs = _captured_openai_kwargs(monkeypatch, mod, settings)
    assert kwargs.get("http_client") is None


def test_an_over_long_card_is_trimmed_rather_than_failing_its_whole_batch(monkeypatch):
    # The provider rejects the BATCH, not the offending input, so one 110-column fact table
    # used to take down the index build for an entire source.
    client = _Recorder()
    emb = _embedder(monkeypatch, client, max_chars=100)
    emb.embed(["a" * 5000, "short"])
    assert [len(t) for t in client.sent[0]] == [100, 5]


def test_a_provider_that_still_says_too_long_is_believed_over_the_character_guess(monkeypatch):
    # No fixed chars-per-token ratio is safe -- a card of sampled UUIDs approaches 1 -- so the
    # budget is only a first guess and the batch is retried at half until it fits.
    client = _Recorder(fail_while_longer_than=1000)
    emb = _embedder(monkeypatch, client, max_chars=8192)
    emb.embed(["z" * 9000])
    assert [len(t[0]) for t in client.sent] == [8192, 4096, 2048, 1024, 512]


def test_halving_stops_at_the_floor_instead_of_indexing_noise(monkeypatch):
    client = _Recorder(fail_while_longer_than=1)  # never satisfied
    emb = _embedder(monkeypatch, client, max_chars=2048)
    with pytest.raises(RuntimeError, match="maximum input length"):
        emb.embed(["y" * 9000])
    assert [len(t[0]) for t in client.sent] == [2048, 1024, 512]  # not 256, not forever


def test_an_unrelated_provider_error_is_not_retried(monkeypatch):
    client = _Recorder(fail_while_longer_than=1, error="invalid api key")
    emb = _embedder(monkeypatch, client, max_chars=8192)
    with pytest.raises(RuntimeError, match="invalid api key"):
        emb.embed(["anything"])
    assert len(client.sent) == 1


def test_results_are_ordered_by_the_index_the_api_reports(monkeypatch):
    client = _Recorder()
    emb = _embedder(monkeypatch, client, max_chars=100)
    assert len(emb.embed(["a", "b", "c"])) == 3

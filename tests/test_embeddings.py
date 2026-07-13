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

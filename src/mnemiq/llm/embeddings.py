from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol

from openai import OpenAI

from mnemiq.config import Settings

EMBED_DIM = 1536


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMEmbedder:
    def __init__(self, settings: Settings, batch_size: int = 64) -> None:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise RuntimeError("LLM base_url/api_key not configured (set MNEMIQ_LLM_* env)")
        self._model = settings.embed_model
        self._batch_size = batch_size
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            # the API may return results out of order; index is authoritative
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return vectors


class FakeEmbedder:
    """Deterministic vectors, no network.

    Unit tests assert plumbing and access control, never semantic quality -- a hash cannot
    make "fire claim" close to "fire loss". Semantic quality is the live test's job.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        raw = b""
        counter = 0
        while len(raw) < self._dim * 4:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        floats = struct.unpack(f"{self._dim}f", raw[: self._dim * 4])
        norm = math.sqrt(sum(f * f for f in floats)) or 1.0
        return [f / norm for f in floats]

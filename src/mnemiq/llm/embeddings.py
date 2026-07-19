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
        base_url, api_key = settings.embed_endpoint()
        if not base_url or not api_key:
            raise RuntimeError("embed base_url/api_key not configured (set MNEMIQ_LLM_* or "
                               "MNEMIQ_EMBED_* env)")
        self._model = settings.embed_model
        self._batch_size = batch_size
        self._client = OpenAI(base_url=base_url, api_key=api_key)

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
        # Read the digest as unsigned ints, not floats: a random 32-bit pattern is a valid
        # float32 only by luck -- most are NaN or Inf, and NaN poisons both the norm and
        # equality (nan != nan).
        ints = struct.unpack(f"{self._dim}I", raw[: self._dim * 4])
        floats = [(i / 0xFFFFFFFF) * 2.0 - 1.0 for i in ints]  # -> [-1, 1]
        norm = math.sqrt(sum(f * f for f in floats)) or 1.0
        return [f / norm for f in floats]

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol

from openai import OpenAI

from mnemiq.config import Settings

EMBED_DIM = 1536

# Embedding endpoints reject an input longer than their context, and the whole batch fails
# with it -- one 110-column fact table takes down the index build for the entire source.
# A card that long is also past the point where more text helps retrieval: the table name
# and its first columns carry the signal, and the tail is column 90 of a shadow table.
# So the head is kept and the tail is dropped, per input, rather than letting the provider
# refuse the batch. Budget is in characters because tokenising here would mean shipping a
# tokeniser for every provider -- but no fixed chars-per-token ratio is safe: prose runs ~4,
# identifier-dense DDL ~2, and a card carrying sampled UUIDs or hex keys approaches 1. So the
# character budget is only a first guess, and a provider that still calls the input too long
# is believed over the guess: the batch is retried at half the budget until it fits.
EMBED_MAX_TOKENS = 8192
_CHARS_PER_TOKEN = 2.0
EMBED_MAX_CHARS = int(EMBED_MAX_TOKENS * _CHARS_PER_TOKEN)
_MIN_EMBED_CHARS = 512  # below this a card has lost its identity; fail rather than embed noise
_TOO_LONG = "maximum input length"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMEmbedder:
    def __init__(self, settings: Settings, batch_size: int = 64,
                 max_chars: int = EMBED_MAX_CHARS) -> None:
        base_url, api_key = settings.embed_endpoint()
        if not base_url or not api_key:
            raise RuntimeError("embed base_url/api_key not configured (set MNEMIQ_LLM_* or "
                               "MNEMIQ_EMBED_* env)")
        self._model = settings.embed_model
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _fit(self, text: str) -> str:
        """Trim one input to the endpoint's context, keeping the head."""
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars]

    def _embed_batch(self, batch: list[str], budget: int) -> list[list[float]]:
        """One batch, shrinking the per-input budget until the provider accepts it."""
        while True:
            trimmed = [t[:budget] for t in batch]
            try:
                resp = self._client.embeddings.create(model=self._model, input=trimmed)
            except Exception as exc:
                if _TOO_LONG not in str(exc) or budget <= _MIN_EMBED_CHARS:
                    raise
                budget //= 2
                continue
            # the API may return results out of order; index is authoritative
            return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        fitted = [self._fit(t) for t in texts]
        for start in range(0, len(fitted), self._batch_size):
            batch = fitted[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch, self._max_chars))
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

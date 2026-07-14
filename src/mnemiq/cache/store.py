from __future__ import annotations

from typing import Protocol

import pyarrow as pa
from cachetools import LRUCache


def to_ipc(table: pa.Table) -> bytes:
    """Arrow IPC: the hot-path cache format. Round-trips a table exactly."""
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


def from_ipc(raw: bytes) -> pa.Table:
    return pa.ipc.open_stream(raw).read_all()


class Cache(Protocol):
    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, value: bytes) -> None: ...


class L1Cache:
    """In-process LRU. The only backend v0.1 ships -- a single node needs nothing shared."""

    def __init__(self, maxsize: int = 128) -> None:
        self._cache: LRUCache = LRUCache(maxsize=maxsize)

    def get(self, key: str) -> bytes | None:
        return self._cache.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._cache[key] = value


class NullCache:
    """The L2 default: the seam exists, no backend is wired (spec 6.8)."""

    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, value: bytes) -> None:
        return None


class TwoTierCache:
    """L1 in front of a pluggable L2. In v1 the L2 becomes a config swap, not a redesign."""

    def __init__(self, l1: Cache, l2: Cache | None = None) -> None:
        self._l1 = l1
        self._l2 = l2 or NullCache()

    def get(self, key: str) -> bytes | None:
        hit = self._l1.get(key)
        if hit is not None:
            return hit

        hit = self._l2.get(key)
        if hit is not None:
            self._l1.put(key, hit)  # promote: the next read stays in-process
        return hit

    def put(self, key: str, value: bytes) -> None:
        self._l1.put(key, value)
        self._l2.put(key, value)

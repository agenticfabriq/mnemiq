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


_DEFAULT_MAX_BYTES = 256 * 1024 * 1024  # 256 MiB -- a bounded, predictable ceiling


class L1Cache:
    """In-process LRU, budgeted by bytes.

    The cache stores IPC-serialised Arrow tables, so ``len(value)`` is the exact byte cost.
    The old entry-count budget was unbounded in memory: 128 large result
    sets could exhaust the process.  A byte ceiling keeps the footprint predictable regardless
    of result-set size.
    """

    def __init__(self, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        """`max_bytes`, not `maxsize`, and keyword-only.

        The budget changed from entries to bytes and the parameter kept its old name for one
        release. Nothing broke -- no caller passes it -- but `L1Cache(maxsize=128)` written by
        someone carrying the old meaning would have built a 128-BYTE cache: not an error, a
        cache that silently never hits, presenting as a mystery miss rate rather than a
        failure. Keyword-only so a positional `L1Cache(128)` cannot mean it either.
        """
        self._cache: LRUCache = LRUCache(maxsize=max_bytes, getsizeof=len)

    def get(self, key: str) -> bytes | None:
        return self._cache.get(key)

    def put(self, key: str, value: bytes) -> None:
        try:
            self._cache[key] = value
        except ValueError:
            pass  # value exceeds the byte budget -- drop it; a miss is fine, a crash is not



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

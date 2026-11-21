"""Every statement enrich sends to its source carries a time limit.

On a production database one exact-profiling statement ran for hours, and nothing bounded it: `execute` took no time
limit. `EnrichDeadline` is the adapter enrich reads through. Each `execute` carries `timeout_s`; a statement the limit
cuts raises `QueryTimedOut` as before, and is first recorded by the tables it reads, so enrich can say what it cut. The
caller decides what the cut costs: the table's profile, or one grounding or value-index item (the plan's Task 3).
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

from mnemiq.adapters.base import QueryTimedOut
from mnemiq.contract import Job

logger = logging.getLogger(__name__)


def _takes_a_deadline(adapter: Any) -> bool:
    execute = getattr(adapter, "execute", None)
    if execute is None:  # nothing to bound: an adapter that runs no statement
        return True
    try:
        return "timeout_s" in inspect.signature(execute).parameters
    except (TypeError, ValueError):
        return False


def _tables(sql: str, dialect: str | None) -> tuple[str, ...]:
    """The tables a cut statement reads, by the engine's own reading; empty when it does not parse."""
    import sqlglot

    from mnemiq.sql.scope import base_tables

    try:
        return tuple(sorted({t.name for t in base_tables(sqlglot.parse_one(sql, read=dialect), dialect)}))
    except Exception:  # noqa: BLE001 -- a name for a warning; the cut itself is already raised
        return ()


class EnrichDeadline:
    """The adapter enrich reads through: `execute` with the time limit, everything else the source's.

    `timeout_s` 0 means none. An adapter whose `execute` takes no `timeout_s` (Superset's, an evaluation source's) is
    read without one, and `unbounded` says so, for enrich to warn once."""

    def __init__(self, inner: Any, timeout_s: float) -> None:
        self._source = inner
        self.timeout_s = timeout_s or None
        self.unbounded = self.timeout_s is not None and not _takes_a_deadline(inner)
        self.timed_out: list[tuple[str, ...]] = []

    def execute(self, sql: str, timeout_s: float | None = None) -> list[tuple]:
        limit = timeout_s if timeout_s is not None else self.timeout_s
        if limit is None or self.unbounded:
            return self._source.execute(sql)
        try:
            return self._source.execute(sql, timeout_s=limit)
        except QueryTimedOut:
            self.timed_out.append(_tables(sql, getattr(self._source, "dialect", None)))
            raise

    def __getattr__(self, name: str) -> Any:
        # Everything but `execute` is the source's: its dictionary, its dialect, its scope.
        return getattr(self._source, name)


def _named(adapter: EnrichDeadline) -> str:
    names = sorted({t for tables in adapter.timed_out for t in tables})
    unnamed = sum(not tables for tables in adapter.timed_out)
    listed = ", ".join(names) + (f" (and {unnamed} statement(s) that would not parse)" if unnamed else "")
    return f"{len(adapter.timed_out)} statement(s) reached the {adapter.timeout_s:g} s limit, reading {listed}"


def time_limit_job(adapter: Any, source_id: str) -> Job | None:
    """The snapshot's record of what the limit cut so far, or None when it cut nothing."""
    if not getattr(adapter, "timed_out", None):
        return None
    return Job(id="enrich:time-limit", source_id=source_id, kind="enrich", status="partial", detail=_named(adapter))


def time_limit_warning(adapter: Any, source_id: str) -> str | None:
    """The line enrich ends with when the limit cut anything, the value index's statements included."""
    if not getattr(adapter, "timed_out", None):
        return None
    return (f"WARNING: source {source_id!r}: {_named(adapter)}. Each cut statement cost what it was reading: a "
            "table's profile, or one grounding or value-index item. MNEMIQ_ENRICH_STATEMENT_TIMEOUT_S sets the limit.")

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mnemiq.contract import DeferralReason


@dataclass
class AnswerRecord:
    deferred: bool
    cached: bool
    total_ms: float
    mode: str | None
    # A source outage used to be recorded as a deferral, so `deferral_rate` -- the number the
    # product claim rests on -- went UP when the database went down (register M6).
    failed: bool = False
    reason_code: DeferralReason | str | None = None


@dataclass
class Metrics:
    answers: int = 0
    deferrals: int = 0
    errors: int = 0  # was hardcoded 0, so a total outage reported as flawless abstention (M2)
    deferral_rate: float = 0.0
    cache_hit_rate: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def aggregate(records: list[AnswerRecord]) -> Metrics:
    if not records:
        return Metrics()
    n = len(records)
    deferrals = sum(1 for r in records if r.deferred and not r.failed)
    errors = sum(1 for r in records if r.failed)
    cached = sum(1 for r in records if r.cached)
    latencies = [r.total_ms for r in records]
    # Deferring is a decision; failing is not. The rate is over the requests we actually decided,
    # so an outage can no longer raise it (M6).
    decided = n - errors
    return Metrics(
        answers=n, deferrals=deferrals, errors=errors,
        deferral_rate=(deferrals / decided if decided else 0.0), cache_hit_rate=cached / n,
        p50_ms=_pct(latencies, 0.50), p95_ms=_pct(latencies, 0.95),
    )


class ObservabilitySink(Protocol):
    def record(self, source_id: str, rec: AnswerRecord) -> None: ...

    def recent(self, source_id: str, limit: int) -> list[AnswerRecord]: ...


class NullSink:
    def record(self, source_id: str, rec: AnswerRecord) -> None:
        return None

    def recent(self, source_id: str, limit: int) -> list[AnswerRecord]:
        return []


class PostgresSink:
    """Fail-soft answer log in the control Postgres."""

    _CREATE = (
        "CREATE TABLE IF NOT EXISTS mnemiq_answer_log ("
        "source_id TEXT, deferred BOOLEAN, cached BOOLEAN, total_ms DOUBLE PRECISION, mode TEXT, "
        "failed BOOLEAN DEFAULT FALSE, reason_code TEXT)"
    )
    # Existing deployments predate the two columns; without this their INSERT would fail and be
    # swallowed by the fail-soft handler below, i.e. observability would silently stop recording.
    _UPGRADE = (
        "ALTER TABLE mnemiq_answer_log "
        "ADD COLUMN IF NOT EXISTS failed BOOLEAN DEFAULT FALSE, "
        "ADD COLUMN IF NOT EXISTS reason_code TEXT"
    )

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def record(self, source_id: str, rec: AnswerRecord) -> None:
        try:
            import psycopg

            with psycopg.connect(self._dsn, autocommit=True) as con:
                con.execute(self._CREATE)
                con.execute(self._UPGRADE)
                con.execute(
                    "INSERT INTO mnemiq_answer_log "
                    "(source_id, deferred, cached, total_ms, mode, failed, reason_code) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (source_id, rec.deferred, rec.cached, rec.total_ms, rec.mode, rec.failed,
                     str(rec.reason_code) if rec.reason_code else None),
                )
        except Exception:
            pass  # fail-soft: observability never breaks a request

    def recent(self, source_id: str, limit: int) -> list[AnswerRecord]:
        try:
            import psycopg

            with psycopg.connect(self._dsn, autocommit=True) as con:
                con.execute(self._CREATE)
                con.execute(self._UPGRADE)
                rows = con.execute(
                    "SELECT deferred, cached, total_ms, mode, failed, reason_code "
                    "FROM mnemiq_answer_log WHERE source_id = %s LIMIT %s",
                    (source_id, limit),
                ).fetchall()
            return [AnswerRecord(deferred=r[0], cached=r[1], total_ms=r[2], mode=r[3],
                                 failed=bool(r[4]), reason_code=r[5])
                    for r in rows]
        except Exception:
            return []

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from mnemiq.eval.report import Report

_COLS = ("source_id", "run_at", "accuracy", "strict_accuracy", "total", "correct",
         "correct_facts", "wrong", "deferred_wrongly", "error")

_DDL = (
    "CREATE TABLE IF NOT EXISTS mnemiq_eval_run ("
    "source_id TEXT, run_at TEXT, accuracy DOUBLE PRECISION, "
    "strict_accuracy DOUBLE PRECISION, total INT, correct INT, "
    "correct_facts INT, wrong INT, deferred_wrongly INT, error INT)"
)


@dataclass
class RunRecord:
    source_id: str
    run_at: str
    accuracy: float
    strict_accuracy: float
    total: int
    correct: int
    correct_facts: int
    wrong: int
    deferred_wrongly: int
    error: int


def _to_record(source_id: str, run_at: str, report: Report) -> RunRecord:
    return RunRecord(
        source_id=source_id, run_at=run_at, accuracy=report.accuracy,
        strict_accuracy=report.strict_accuracy, total=report.total, correct=report.correct,
        correct_facts=report.correct_facts, wrong=report.wrong,
        deferred_wrongly=report.deferred_wrongly, error=report.error,
    )


def record_run(control_dsn: str | None, source_id: str, report: Report,
               path: str | None = None, run_at: str = "") -> None:
    """Append a run to the trend. Postgres when a control DSN is set, else a local JSON.
    Fail-soft: a trend-write error never fails the eval run."""
    rec = _to_record(source_id, run_at, report)
    try:
        if control_dsn:
            import psycopg

            with psycopg.connect(control_dsn, autocommit=True) as con:
                con.execute(_DDL)
                con.execute(
                    f"INSERT INTO mnemiq_eval_run ({', '.join(_COLS)}) "
                    f"VALUES ({', '.join(['%s'] * len(_COLS))})",
                    tuple(asdict(rec)[c] for c in _COLS),
                )
            return
        rows = []
        if path and os.path.exists(path):
            with open(path) as fh:
                rows = json.load(fh)
        rows.append(asdict(rec))
        if path:
            with open(path, "w") as fh:
                json.dump(rows, fh)
    except Exception:
        pass  # fail-soft


def last_run(control_dsn: str | None, source_id: str,
             path: str | None = None) -> RunRecord | None:
    try:
        if control_dsn:
            import psycopg

            with psycopg.connect(control_dsn, autocommit=True) as con:
                con.execute(_DDL)
                row = con.execute(
                    f"SELECT {', '.join(_COLS)} FROM mnemiq_eval_run WHERE source_id = %s "
                    "ORDER BY run_at DESC LIMIT 1",
                    (source_id,),
                ).fetchone()
                return RunRecord(**dict(zip(_COLS, row, strict=True))) if row else None
        if path and os.path.exists(path):
            with open(path) as fh:
                rows = [r for r in json.load(fh) if r["source_id"] == source_id]
            return RunRecord(**rows[-1]) if rows else None
    except Exception:
        return None
    return None


def check_regression(report: Report, previous: RunRecord | None,
                     tolerance: float = 0.02) -> str | None:
    """A message if accuracy dropped more than tolerance below the previous run, else None."""
    if previous is None:
        return None
    if report.accuracy < previous.accuracy - tolerance:
        return (f"accuracy regressed: {report.accuracy:.1%} < previous "
                f"{previous.accuracy:.1%} - {tolerance:.0%} tolerance")
    return None

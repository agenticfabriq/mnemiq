"""Live living-loops shared state against the available Postgres (control plane, test source ids).
Proves the eval-trend and answer-log round-trips. Integration-gated; no LLM."""

import os

import psycopg
import pytest

from mnemiq.eval.report import Report
from mnemiq.eval.trend import last_run, record_run
from mnemiq.observability.metrics import AnswerRecord, PostgresSink, aggregate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured"),
]

_DSN = os.getenv("MNEMIQ_PG_DSN", "")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for t in ("mnemiq_eval_run", "mnemiq_answer_log"):
        # Delete only the test rows; ignore if the table was never created (no wrong-schema
        # CREATE here -- the real schema is owned by trend.py / PostgresSink).
        try:
            with psycopg.connect(_DSN, autocommit=True) as con:
                con.execute(f"DELETE FROM {t} WHERE source_id LIKE 'lltest_%'")
        except Exception:
            pass


def test_eval_trend_roundtrip():
    record_run(_DSN, "lltest_src", Report(total=10, correct=8, wrong=2))
    record_run(_DSN, "lltest_src", Report(total=10, correct=9, wrong=1))
    rec = last_run(_DSN, "lltest_src")
    assert rec is not None and rec.correct in (8, 9)  # a run was persisted and read back


def test_answer_log_sink_roundtrip():
    sink = PostgresSink(_DSN)
    sink.record("lltest_src", AnswerRecord(deferred=False, cached=False, total_ms=12.0, mode="x"))
    sink.record("lltest_src", AnswerRecord(deferred=True, cached=False, total_ms=8.0, mode="x"))
    m = aggregate(sink.recent("lltest_src", 100))
    assert m.answers == 2 and m.deferrals == 1

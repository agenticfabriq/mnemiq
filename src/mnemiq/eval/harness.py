from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import pyarrow as pa

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import EvaluationCase
from mnemiq.eval.grade import normalize, results_match

_PREVIEW_ROWS = 100


class Outcome(StrEnum):
    CORRECT = "correct"  # exact result-set match (BIRD-strict semantics)
    CORRECT_FACTS = "correct_facts"  # right data, different shape (e.g. an extra column).
    # A format difference is NOT a failure -- it is its own category, so strict-EX and
    # got-the-facts can be reported side by side without re-grading.
    WRONG = "wrong"  # answered, and the answer is not right. The only invisible failure.
    DEFERRED_CORRECTLY = "deferred_correctly"
    DEFERRED_WRONGLY = "deferred_wrongly"
    ERROR = "error"


@dataclass
class CaseResult:
    case_id: str
    outcome: Outcome
    question: str = ""
    answer: str = ""
    sql: str = ""
    gold_sql: str = ""
    # Bounded, JSON-safe previews of both result sets: without them, analyzing a failure
    # means re-running the queries by hand. None = the query never ran.
    gold_rows: list[dict] | None = field(default=None)
    engine_rows: list[dict] | None = field(default=None)
    gold_row_count: int | None = None
    engine_row_count: int | None = None
    db_id: str | None = None
    difficulty: str | None = None
    agreement: float | None = None  # self-consistency: winning-cluster fraction, if any
    judge_engaged: bool | None = None  # selector-judge: consulted on this case?
    judge_override: bool | None = None  # ...and picked against the majority?
    candidates_executed: int | None = None  # multi-candidate: how many of N ran
    proposed: bool = False
    approved: bool = False
    executed: bool = False
    ms: float = 0.0


Engine = Callable[[str], AgentAnswer]


def _preview(table: pa.Table) -> list[dict]:
    return [
        {k: normalize(v) for k, v in row.items()}
        for row in table.slice(0, _PREVIEW_ROWS).to_pylist()
    ]


def run_case(case: EvaluationCase, engine: Engine, adapter, gold_adapter=None) -> CaseResult:
    """Ask the engine, then check its answer against the gold query's result set.

    `adapter` executes the engine's SQL (the same executor the engine used); `gold_adapter`
    (defaults to `adapter`) executes the gold SQL. BIRD passes a DuckDB engine adapter and a
    native-SQLite gold adapter; single-engine callers (ACME) pass one adapter for both.
    """
    gold_adapter = gold_adapter or adapter
    result = CaseResult(
        case_id=case.id,
        outcome=Outcome.ERROR,
        question=case.question,
        gold_sql=case.gold_sql or "",
        db_id=case.db_id,
        difficulty=next(
            (t for t in case.tags if t in {"simple", "moderate", "challenging"}), None
        ),
    )
    started = time.perf_counter()

    try:
        answer: AgentAnswer = engine(case.question)
    except Exception as exc:
        result.answer = str(exc)
        result.ms = (time.perf_counter() - started) * 1000
        return result

    result.ms = (time.perf_counter() - started) * 1000
    result.agreement = answer.agreement
    result.judge_engaged = answer.judge_engaged
    result.judge_override = answer.judge_override
    result.candidates_executed = answer.candidates_executed

    gold: pa.Table | None = None
    if case.answerable:
        try:
            gold = gold_adapter.execute_arrow(case.gold_sql, timeout_s=30)
            result.gold_rows = _preview(gold)
            result.gold_row_count = gold.num_rows
        except Exception:
            gold = None  # a gold failure never changes the outcome; the preview just stays empty

    if answer.failed:
        # The source refused to serve us. That is an ERROR, not the engine abstaining -- grading
        # it DEFERRED_WRONGLY reported an outage as "safe: gave up on an answerable question"
        # and inflated every deferral rate we have measured (register M6).
        result.outcome = Outcome.ERROR
        result.answer = answer.answer
        return result

    if answer.deferred:
        # Refusing a question the data cannot answer is the behaviour we built the decider
        # for. Refusing one it *can* answer is a failure -- but a safe one: the user knows.
        result.outcome = Outcome.DEFERRED_CORRECTLY if not case.answerable else Outcome.DEFERRED_WRONGLY
        result.answer = answer.answer
        return result

    result.answer = answer.answer
    result.sql = answer.trace.target_sql if answer.trace else ""
    result.proposed = True
    result.approved = True

    if not case.answerable:
        # It answered a question ACME holds no data for: it invented something.
        result.outcome = Outcome.WRONG
        result.executed = True
        return result

    try:
        if gold is None:  # the gold query itself failed: grading is impossible
            raise RuntimeError(f"gold query failed for {case.id}")
        candidate = adapter.execute_arrow(result.sql, timeout_s=30)
    except Exception as exc:
        result.outcome = Outcome.ERROR
        result.answer = str(exc)
        return result

    result.engine_rows = _preview(candidate)
    result.engine_row_count = candidate.num_rows
    result.executed = True
    if results_match(gold, candidate, allow_extra_columns=False):
        result.outcome = Outcome.CORRECT
    elif results_match(gold, candidate, allow_extra_columns=True):
        result.outcome = Outcome.CORRECT_FACTS
    else:
        result.outcome = Outcome.WRONG
    return result

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import EvaluationCase
from mnemiq.eval.grade import results_match


class Outcome(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"  # answered, and the answer is not right. The only invisible failure.
    DEFERRED_CORRECTLY = "deferred_correctly"
    DEFERRED_WRONGLY = "deferred_wrongly"
    ERROR = "error"


@dataclass
class CaseResult:
    case_id: str
    outcome: Outcome
    answer: str = ""
    sql: str = ""
    proposed: bool = False
    approved: bool = False
    executed: bool = False
    ms: float = 0.0


Engine = Callable[[str], AgentAnswer]


def run_case(case: EvaluationCase, engine: Engine, adapter) -> CaseResult:
    """Ask the engine, then check its answer against the gold query's result set."""
    started = time.perf_counter()
    try:
        answer: AgentAnswer = engine(case.question)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.ERROR,
            answer=str(exc),
            ms=(time.perf_counter() - started) * 1000,
        )

    elapsed = (time.perf_counter() - started) * 1000

    if answer.deferred:
        # Refusing a question the data cannot answer is the behaviour we built the decider
        # for. Refusing one it *can* answer is a failure -- but a safe one: the user knows.
        outcome = Outcome.DEFERRED_CORRECTLY if not case.answerable else Outcome.DEFERRED_WRONGLY
        return CaseResult(case_id=case.id, outcome=outcome, answer=answer.answer, ms=elapsed)

    if not case.answerable:
        # It answered a question ACME holds no data for: it invented something.
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.WRONG,
            answer=answer.answer,
            sql=answer.trace.target_sql if answer.trace else "",
            proposed=True,
            approved=True,
            executed=True,
            ms=elapsed,
        )

    sql = answer.trace.target_sql if answer.trace else ""
    try:
        gold = adapter.execute_arrow(case.gold_sql, timeout_s=30)
        candidate = adapter.execute_arrow(sql, timeout_s=30)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.ERROR,
            answer=str(exc),
            sql=sql,
            proposed=True,
            approved=True,
            ms=elapsed,
        )

    correct = results_match(gold, candidate)
    return CaseResult(
        case_id=case.id,
        outcome=Outcome.CORRECT if correct else Outcome.WRONG,
        answer=answer.answer,
        sql=sql,
        proposed=True,
        approved=True,
        executed=True,
        ms=elapsed,
    )

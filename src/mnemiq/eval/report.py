from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from mnemiq.eval.harness import CaseResult, Outcome


@dataclass
class Report:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    deferred_correctly: int = 0
    deferred_wrongly: int = 0
    error: int = 0
    tokens: int = 0
    llm_calls: int = 0
    results: list[CaseResult] = field(default_factory=list)

    @property
    def answerable(self) -> int:
        return self.correct + self.wrong + self.deferred_wrongly + self.error

    @property
    def accuracy(self) -> float:
        """Correct answers over questions the data can actually answer."""
        return self.correct / self.answerable if self.answerable else 0.0

    def render(self) -> str:
        lines = [
            f"cases            {self.total}",
            "",
            f"  CORRECT            {self.correct}",
            f"  WRONG              {self.wrong}   <- the only failure a user cannot see",
            f"  DEFERRED_WRONGLY   {self.deferred_wrongly}   (safe: gave up on an answerable question)",
            f"  DEFERRED_CORRECTLY {self.deferred_correctly}   (success: refused the unanswerable)",
            f"  ERROR              {self.error}",
            "",
            f"accuracy         {self.accuracy:.1%}  (correct / answerable)",
            f"llm calls        {self.llm_calls}",
            f"tokens           {self.tokens}",
        ]
        failures = [r for r in self.results if r.outcome in (Outcome.WRONG, Outcome.ERROR)]
        if failures:
            lines += ["", "failures:"]
            lines += [f"  [{r.outcome}] {r.case_id}: {r.sql or r.answer}"[:160] for r in failures]
        return "\n".join(lines)


def summarize(results: list[CaseResult], tokens: int = 0, llm_calls: int = 0) -> Report:
    report = Report(total=len(results), tokens=tokens, llm_calls=llm_calls, results=list(results))
    counters = {
        Outcome.CORRECT: "correct",
        Outcome.WRONG: "wrong",
        Outcome.DEFERRED_CORRECTLY: "deferred_correctly",
        Outcome.DEFERRED_WRONGLY: "deferred_wrongly",
        Outcome.ERROR: "error",
    }
    for result in results:
        attr = counters[result.outcome]
        setattr(report, attr, getattr(report, attr) + 1)
    return report


def slice_by(results: list[CaseResult], key: Callable[[CaseResult], str | None]) -> dict[str, Report]:
    """Sub-reports grouped by a key (database, difficulty). None keys are dropped."""
    groups: dict[str, list[CaseResult]] = {}
    for r in results:
        k = key(r)
        if k is not None:
            groups.setdefault(k, []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from mnemiq.eval.harness import CaseResult, Outcome


@dataclass
class Report:
    total: int = 0
    correct: int = 0
    correct_facts: int = 0  # right data, different shape -- a format difference, not a failure
    wrong: int = 0
    deferred_correctly: int = 0
    deferred_wrongly: int = 0
    error: int = 0
    # Answers our executor ran and the gold's engine cannot. Not a wrong answer:
    # excluded from the BIRD-comparable number, kept in the product metric.
    unportable: int = 0
    unportable_exact: int = 0
    tokens: int = 0
    llm_calls: int = 0
    results: list[CaseResult] = field(default_factory=list)

    @property
    def answerable(self) -> int:
        return self.correct + self.correct_facts + self.wrong + self.deferred_wrongly + self.error

    @property
    def accuracy(self) -> float:
        """The product metric: got-the-facts over questions the data can answer.

        Portability does not enter here. Our executor is the one that ran the
        query, and it answered; whether the same SQL would also run on the
        benchmark's engine is a different question from whether the agent found
        the facts.
        """
        return (self.correct + self.correct_facts) / self.answerable if self.answerable else 0.0

    @property
    def strict_accuracy(self) -> float:
        """Exact result-set match, BIRD-comparable -- so portability counts.

        BIRD's protocol executes candidate and gold in one engine. An answer the
        gold's engine cannot run has no comparable result set, so counting it
        here would make the number incomparable to the thing it is named after.
        """
        exact = self.correct - self.unportable_exact
        return exact / self.answerable if self.answerable else 0.0

    def render(self) -> str:
        lines = [
            f"cases            {self.total}",
            "",
            f"  CORRECT            {self.correct}   (exact result-set match)",
            f"  CORRECT_FACTS      {self.correct_facts}   (right data, different shape -- not a failure)",
            f"  WRONG              {self.wrong}   <- the only failure a user cannot see",
            f"  DEFERRED_WRONGLY   {self.deferred_wrongly}   (safe: gave up on an answerable question)",
            f"  DEFERRED_CORRECTLY {self.deferred_correctly}   (success: refused the unanswerable)",
            f"  ERROR              {self.error}",
            f"  unportable         {self.unportable}   ({self.unportable_exact} of them exact -- "
            f"ran here, not on the gold's engine)",
            "",
            f"accuracy         {self.accuracy:.1%}  (got-the-facts / answerable -- the product metric)",
            f"strict accuracy  {self.strict_accuracy:.1%}  (exact match / answerable -- BIRD-comparable)",
            f"llm calls        {self.llm_calls}",
            f"tokens           {self.tokens}",
        ]
        # Every outcome that COST accuracy, which is what this list is for. `DEFERRED_WRONGLY` was
        # missing and it is not a lesser failure: accuracy is got-the-facts over answerable, so a
        # case the engine gave up on subtracts exactly what a wrong one does. The gate printed
        # "accuracy regressed: 88.0% < previous 96.0%" beside a single failure line while three
        # cases were responsible, and `check_regression`'s own advice -- find which case moved
        # rather than widen the tolerance -- was unfollowable from the output that gave it.
        #
        # `DEFERRED_CORRECTLY` stays out: refusing the unanswerable is the product working, and
        # collapsing the two deferral outcomes here would undo the reason there are two.
        failures = [
            r for r in self.results
            if r.outcome in (Outcome.WRONG, Outcome.ERROR, Outcome.DEFERRED_WRONGLY)
        ]
        if failures:
            lines += ["", "failures:"]
            lines += [f"  [{r.outcome}] {r.case_id}: {r.sql or r.answer}"[:160] for r in failures]
        return "\n".join(lines)


def summarize(results: list[CaseResult], tokens: int = 0, llm_calls: int = 0) -> Report:
    report = Report(total=len(results), tokens=tokens, llm_calls=llm_calls, results=list(results))
    counters = {
        Outcome.CORRECT: "correct",
        Outcome.CORRECT_FACTS: "correct_facts",
        Outcome.WRONG: "wrong",
        Outcome.DEFERRED_CORRECTLY: "deferred_correctly",
        Outcome.DEFERRED_WRONGLY: "deferred_wrongly",
        Outcome.ERROR: "error",
    }
    for result in results:
        attr = counters[result.outcome]
        setattr(report, attr, getattr(report, attr) + 1)
        if result.portable_to_gold_engine is False:
            report.unportable += 1
            if result.outcome is Outcome.CORRECT:
                report.unportable_exact += 1
    return report


def slice_by(results: list[CaseResult], key: Callable[[CaseResult], str | None]) -> dict[str, Report]:
    """Sub-reports grouped by a key (database, difficulty). None keys are dropped."""
    groups: dict[str, list[CaseResult]] = {}
    for r in results:
        k = key(r)
        if k is not None:
            groups.setdefault(k, []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}

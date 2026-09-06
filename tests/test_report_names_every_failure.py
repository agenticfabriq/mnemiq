"""The gate's failure list must name every case that cost accuracy.

Found while re-recording the ACME baseline against CI. The gate printed
`accuracy regressed: 88.0% < previous 96.0%` and listed ONE failure, while three cases
were dragging the number down -- the other two were `DEFERRED_WRONGLY`, which counts
against `accuracy` and was never printed. `check_regression`'s own docstring tells the
reader to "find out which case moved instead" of widening the tolerance; the output did
not let them.
"""

from __future__ import annotations

from mnemiq.eval.harness import CaseResult, Outcome
from mnemiq.eval.report import summarize


def _report():
    return summarize([
        CaseResult(case_id="answered", outcome=Outcome.CORRECT, sql="select 1"),
        CaseResult(case_id="fire-count", outcome=Outcome.WRONG, sql="select count(*) from claim"),
        # Answerable, and the engine gave up. It costs exactly as much accuracy as the wrong
        # answer above -- both are one case out of `answerable`.
        CaseResult(case_id="policy-tenure", outcome=Outcome.DEFERRED_WRONGLY,
                   answer="I don't have enough information about tenure"),
        # NOT a failure: refusing the unanswerable is the product working.
        CaseResult(case_id="unanswerable", outcome=Outcome.DEFERRED_CORRECTLY,
                   answer="that is not in this database"),
    ])


def test_a_wrongly_deferred_case_is_named_in_the_failures():
    rendered = _report().render()
    assert "policy-tenure" in rendered, (
        "a case that cost accuracy must be named, or the operator told to find which case "
        f"moved cannot:\n{rendered}"
    )
    assert "fire-count" in rendered


def test_a_correctly_deferred_case_is_not_listed_as_a_failure():
    """The distinction is the whole point of having two deferral outcomes."""
    rendered = _report().render()
    failures_block = rendered.split("failures:")[1]
    assert "unanswerable" not in failures_block, (
        f"refusing the unanswerable is a success and must not be listed:\n{failures_block}"
    )


def test_the_listed_failures_account_for_the_accuracy_shortfall():
    """The count that matters: as many failure lines as answerable cases not got right.

    Asserting the names appear is not enough -- it passes while a whole outcome class is
    missing, which is how this shipped. Three answerable cases, one right, so two lines.
    """
    report = _report()
    listed = [ln for ln in report.render().split("failures:")[1].strip().splitlines() if ln.strip()]
    got_the_facts = report.correct + report.correct_facts
    assert len(listed) == report.answerable - got_the_facts, (
        f"{len(listed)} failure line(s) for {report.answerable - got_the_facts} lost case(s)"
    )

import pyarrow as pa

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import EvaluationCase, IdentityContext, Trace
from mnemiq.eval.harness import Outcome, run_case
from mnemiq.eval.report import summarize


class _GoldAdapter:
    """Runs the gold SQL; returns a scripted table for anything else."""

    def __init__(self, candidate: pa.Table | None = None):
        self._candidate = candidate

    def execute_arrow(self, sql, timeout_s=None):
        if sql == "GOLD":
            return pa.table({"n": [820]})
        if self._candidate is None:
            raise Exception("boom")
        return self._candidate


def _trace(sql="CANDIDATE") -> Trace:
    return Trace(
        question="q",
        plan_sql=sql,
        target_sql=sql,
        result_shape="scalar",
        timing={"total_ms": 1.0},
        enrichment_version="v1",
        identity=IdentityContext(tenant_id="t", principal_id="u"),
        tables_used=["fireclaim"],
    )


def _case(answerable=True) -> EvaluationCase:
    return EvaluationCase(
        id="fire-count",
        question="how many fire claims?",
        gold_sql="GOLD" if answerable else None,
        answerable=answerable,
    )


def _answered(sql="CANDIDATE") -> AgentAnswer:
    return AgentAnswer(answer="820 claims.", trace=_trace(sql), deferred=False)


def _deferred() -> AgentAnswer:
    return AgentAnswer(answer="I cannot answer that.", deferred=True)


def test_a_right_answer_is_correct():
    adapter = _GoldAdapter(candidate=pa.table({"total": [820]}))
    result = run_case(_case(), lambda q: _answered(), adapter)

    assert result.outcome is Outcome.CORRECT
    assert result.approved and result.executed


def test_a_confidently_wrong_answer_is_the_worst_outcome():
    adapter = _GoldAdapter(candidate=pa.table({"total": [819]}))
    result = run_case(_case(), lambda q: _answered(), adapter)
    assert result.outcome is Outcome.WRONG


def test_deferring_an_unanswerable_question_is_a_success():
    result = run_case(_case(answerable=False), lambda q: _deferred(), _GoldAdapter())
    assert result.outcome is Outcome.DEFERRED_CORRECTLY


def test_answering_an_unanswerable_question_is_wrong():
    # the engine invented data ACME does not hold -- the failure the decider exists to stop
    result = run_case(_case(answerable=False), lambda q: _answered(), _GoldAdapter())
    assert result.outcome is Outcome.WRONG


def test_giving_up_on_an_answerable_question_is_a_safe_failure():
    result = run_case(_case(), lambda q: _deferred(), _GoldAdapter())
    assert result.outcome is Outcome.DEFERRED_WRONGLY


def test_a_crash_is_an_error_not_a_wrong_answer():
    def _explode(_q):
        raise RuntimeError("kaboom")

    result = run_case(_case(), _explode, _GoldAdapter())
    assert result.outcome is Outcome.ERROR


def test_a_case_result_carries_everything_a_reviewer_needs():
    # question, gold SQL, our SQL, and BOTH result sets -- without these, analyzing a
    # failure means re-running queries by hand (which is exactly what Plan 08 required)
    adapter = _GoldAdapter(candidate=pa.table({"total": [819]}))
    result = run_case(_case(), lambda q: _answered(), adapter)

    assert result.question == "how many fire claims?"
    assert result.gold_sql == "GOLD"
    assert result.sql == "CANDIDATE"
    assert result.gold_rows == [{"n": 820}]
    assert result.engine_rows == [{"total": 819}]
    assert result.gold_row_count == 1 and result.engine_row_count == 1


def test_a_deferred_case_still_shows_what_gold_would_have_returned():
    # a wrong deferral is only reviewable if the report shows the answer that existed
    result = run_case(_case(), lambda q: _deferred(), _GoldAdapter())
    assert result.outcome is Outcome.DEFERRED_WRONGLY
    assert result.gold_rows == [{"n": 820}]
    assert result.engine_rows is None


def test_result_previews_are_capped():
    big = pa.table({"n": list(range(500))})
    adapter = _GoldAdapter(candidate=big)
    result = run_case(_case(), lambda q: _answered(), adapter)

    assert result.engine_row_count == 500  # the true count is kept
    assert len(result.engine_rows) == 100  # the preview is bounded


def test_the_report_separates_wrong_from_error():
    adapter = _GoldAdapter(candidate=pa.table({"total": [819]}))
    results = [
        run_case(_case(), lambda q: _answered(), _GoldAdapter(pa.table({"n": [820]}))),
        run_case(_case(), lambda q: _answered(), adapter),
        run_case(_case(answerable=False), lambda q: _deferred(), _GoldAdapter()),
    ]
    report = summarize(results)

    assert report.correct == 1
    assert report.wrong == 1
    assert report.deferred_correctly == 1
    assert report.total == 3

    rendered = report.render()
    assert "WRONG" in rendered
    assert "1" in rendered

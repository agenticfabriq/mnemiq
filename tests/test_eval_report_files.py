import json

from mnemiq.eval.artifacts import write_html, write_json
from mnemiq.eval.harness import CaseResult, Outcome
from mnemiq.eval.report import summarize


def _results() -> list[CaseResult]:
    return [
        CaseResult(
            case_id="fire-count",
            outcome=Outcome.WRONG,
            question="how many fire claims are there?",
            answer="There are 0 fire claims.",
            sql="SELECT COUNT(DISTINCT claimnumber) AS n FROM fireclaim",
            gold_sql="SELECT count(*) AS n FROM fireclaim",
            gold_rows=[{"n": 820}],
            engine_rows=[{"n": 0}],
            gold_row_count=1,
            engine_row_count=1,
            proposed=True,
            approved=True,
            executed=True,
            ms=1234.5,
        ),
        CaseResult(
            case_id="claim-count",
            outcome=Outcome.CORRECT,
            question="how many claims <b>total</b>?",  # markup must not execute in the HTML
            answer="2 claims.",
            sql="SELECT count(*) AS n FROM claim",
            gold_sql="SELECT count(*) AS n FROM claim",
            gold_rows=[{"n": 2}],
            engine_rows=[{"n": 2}],
            gold_row_count=1,
            engine_row_count=1,
            executed=True,
        ),
        CaseResult(
            case_id="no-salary",
            outcome=Outcome.DEFERRED_CORRECTLY,
            question="average salary?",
            answer="I cannot answer that from this data.",
        ),
    ]


def test_the_json_holds_the_full_per_case_record(tmp_path):
    report = summarize(_results(), tokens=1000, llm_calls=5)
    path = tmp_path / "results.json"
    write_json(report, str(path), label="test-run")

    data = json.loads(path.read_text())
    assert data["label"] == "test-run"
    assert data["summary"]["accuracy"] == report.accuracy
    assert data["summary"]["tokens"] == 1000

    by_id = {c["case_id"]: c for c in data["cases"]}
    fire = by_id["fire-count"]
    assert fire["question"] == "how many fire claims are there?"
    assert fire["gold_sql"].startswith("SELECT count")
    assert fire["sql"].startswith("SELECT COUNT(DISTINCT")
    assert fire["gold_rows"] == [{"n": 820}]
    assert fire["engine_rows"] == [{"n": 0}]
    assert fire["outcome"] == "wrong"


def test_the_html_shows_failures_first_and_escapes_content(tmp_path):
    report = summarize(_results())
    path = tmp_path / "report.html"
    write_html(report, str(path), label="test-run")

    html = path.read_text()
    assert "how many fire claims are there?" in html
    assert "SELECT count(*) AS n FROM fireclaim" in html
    # the question containing markup is escaped, never rendered as tags
    assert "<b>total</b>" not in html
    assert "&lt;b&gt;total&lt;/b&gt;" in html
    # failures come before successes
    assert html.index("fire-count") < html.index("claim-count")
    assert "820" in html and "wrong" in html

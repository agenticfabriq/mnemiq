"""Per-case eval artifacts: a JSON record for machines, an HTML report for humans.

The terminal report says WHAT failed; these say WHY -- question, gold SQL, our SQL and
both result sets side by side, so no failure ever needs the queries re-run by hand.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict

from mnemiq.eval.harness import CaseResult, Outcome
from mnemiq.eval.report import Report

# Failures first: the reader's time goes where the bugs are.
_ORDER = [
    Outcome.WRONG,
    Outcome.ERROR,
    Outcome.DEFERRED_WRONGLY,
    Outcome.CORRECT_FACTS,
    Outcome.CORRECT,
    Outcome.DEFERRED_CORRECTLY,
]

_COLORS = {
    Outcome.WRONG: "#c0392b",
    Outcome.ERROR: "#e67e22",
    Outcome.DEFERRED_WRONGLY: "#b7950b",
    Outcome.CORRECT_FACTS: "#0e7c7b",
    Outcome.CORRECT: "#1e8449",
    Outcome.DEFERRED_CORRECTLY: "#2471a3",
}


def _sorted(results: list[CaseResult]) -> list[CaseResult]:
    rank = {outcome: i for i, outcome in enumerate(_ORDER)}
    return sorted(results, key=lambda r: rank[r.outcome])


def _summary(report: Report) -> dict:
    return {
        "total": report.total,
        "correct": report.correct,
        "correct_facts": report.correct_facts,
        "wrong": report.wrong,
        "deferred_correctly": report.deferred_correctly,
        "deferred_wrongly": report.deferred_wrongly,
        "error": report.error,
        "accuracy": report.accuracy,
        "strict_accuracy": report.strict_accuracy,
        "llm_calls": report.llm_calls,
        "tokens": report.tokens,
    }


def write_json(report: Report, path: str, label: str) -> None:
    payload = {
        "label": label,
        "summary": _summary(report),
        "cases": [asdict(r) for r in _sorted(report.results)],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _rows_table(rows: list[dict] | None, total: int | None) -> str:
    if rows is None:
        return "<p class='muted'>(not executed)</p>"
    if not rows:
        return "<p class='muted'>0 rows</p>"
    headers = "".join(f"<th>{html.escape(str(k))}</th>" for k in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values()) + "</tr>"
        for row in rows
    )
    note = ""
    if total is not None and total > len(rows):
        note = f"<p class='muted'>showing {len(rows)} of {total} rows</p>"
    return f"<table><tr>{headers}</tr>{body}</table>{note}"


def _case_html(result: CaseResult) -> str:
    color = _COLORS[result.outcome]
    is_failure = result.outcome in (Outcome.WRONG, Outcome.ERROR, Outcome.DEFERRED_WRONGLY)
    open_attr = " open" if is_failure else ""
    sql_block = (
        f"<div class='col'><h4>gold SQL</h4><pre>{html.escape(result.gold_sql or '(none)')}</pre>"
        f"{_rows_table(result.gold_rows, result.gold_row_count)}</div>"
        f"<div class='col'><h4>engine SQL</h4><pre>{html.escape(result.sql or '(none)')}</pre>"
        f"{_rows_table(result.engine_rows, result.engine_row_count)}</div>"
    )
    return f"""<details{open_attr}>
<summary><span class="badge" style="background:{color}">{html.escape(result.outcome)}</span>
<strong>{html.escape(result.case_id)}</strong> — {html.escape(result.question)}
<span class="muted">{result.ms:.0f} ms</span></summary>
<p><em>answer:</em> {html.escape(result.answer)}</p>
<div class="cols">{sql_block}</div>
</details>"""


def write_html(report: Report, path: str, label: str) -> None:
    summary = _summary(report)
    stat_cells = "".join(
        f"<td><b>{summary[k]}</b><br><span class='muted'>{k}</span></td>"
        for k in ("total", "correct", "correct_facts", "wrong", "deferred_wrongly", "deferred_correctly", "error")
    )
    cases = "\n".join(_case_html(r) for r in _sorted(report.results))
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>mnemiq eval — {html.escape(label)}</title>
<style>
body {{ font: 14px/1.5 -apple-system, sans-serif; margin: 2rem auto; max-width: 72rem; padding: 0 1rem; }}
table {{ border-collapse: collapse; margin: .5rem 0; }}
td, th {{ border: 1px solid #ccc; padding: .2rem .6rem; text-align: left; }}
pre {{ background: #f6f6f6; padding: .6rem; overflow-x: auto; white-space: pre-wrap; }}
details {{ border: 1px solid #ddd; border-radius: 6px; padding: .5rem .8rem; margin: .5rem 0; }}
summary {{ cursor: pointer; }}
.badge {{ color: #fff; border-radius: 4px; padding: .05rem .5rem; font-size: .85em; }}
.muted {{ color: #888; font-size: .9em; }}
.cols {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.col {{ flex: 1 1 22rem; min-width: 0; }}
.stats td {{ border: none; padding: .2rem 1.2rem .2rem 0; }}
</style></head><body>
<h1>mnemiq eval — {html.escape(label)}</h1>
<table class="stats"><tr>{stat_cells}</tr></table>
<p>accuracy <b>{summary["accuracy"]:.1%}</b> (got-the-facts) &nbsp;·&nbsp; strict <b>{summary["strict_accuracy"]:.1%}</b>
&nbsp;·&nbsp; llm calls {summary["llm_calls"]} &nbsp;·&nbsp; tokens {summary["tokens"]}</p>
{cases}
</body></html>"""
    with open(path, "w") as fh:
        fh.write(doc)

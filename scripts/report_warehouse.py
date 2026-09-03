"""Turn graded vendor runs into one comparison page against mnemiq's published numbers.

Reads any number of graded JSONL files (from grade_warehouse.py) and emits a single HTML
report: the four buckets per run, per-database and per-difficulty breakdowns, and the mnemiq
baseline alongside.

  uv run python scripts/report_warehouse.py \
      --run "Snowflake Cortex Analyst=eval-reports/cortex-graded.jsonl" \
      --out eval-reports/comparison.html
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import os

# From the mnemiq README: BIRD mini-dev, 500 questions, 11 unseen schemas, single shot.
MNEMIQ_BASELINE = {"exact_match": 40.5, "got_the_facts": 52.2, "n": 500}

_BUCKETS = ("correct", "wrong", "deferred", "error", "gold_failed")


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(rows: list[dict]) -> dict:
    counts = collections.Counter(r["bucket"] for r in rows)
    scored = sum(counts[b] for b in ("correct", "wrong", "deferred", "error"))
    return {
        "counts": counts,
        "scored": scored,
        "accuracy": 100 * counts["correct"] / scored if scored else 0.0,
    }


def group(rows: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row.get(key):
            out[row[key]].append(row)
    return {k: summarize(v) for k, v in sorted(out.items())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="append",
        dest="runs",
        required=True,
        metavar="LABEL=PATH",
        help="a graded JSONL, labelled for the report; repeatable",
    )
    p.add_argument("--out", default="eval-reports/comparison.html")
    args = p.parse_args()

    runs = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"skipping {label}: no such file {path}")
            continue
        rows = load(path)
        runs.append((label, rows, summarize(rows)))

    if not runs:
        print("no graded runs found")
        return 1

    for label, rows, summary in runs:
        counts = summary["counts"]
        print(f"\n{label}")
        print(f"  {'bucket':14}{'n':>6}{'% of scored':>13}")
        for bucket in _BUCKETS:
            n = counts[bucket]
            share = (
                "excluded"
                if bucket == "gold_failed"
                else f"{100 * n / summary['scored']:.1f}%" if summary["scored"] else "-"
            )
            print(f"  {bucket:14}{n:6}{share:>13}")
        print(f"  execution accuracy: {summary['accuracy']:.1f}% of {summary['scored']} scored")

    rows_html = "\n".join(
        f"<tr><td>{html.escape(label)}</td>"
        f"<td class='num'>{s['counts']['correct']}</td>"
        f"<td class='num'>{s['counts']['wrong']}</td>"
        f"<td class='num'>{s['counts']['deferred']}</td>"
        f"<td class='num'>{s['counts']['error']}</td>"
        f"<td class='num'>{s['counts']['gold_failed']}</td>"
        f"<td class='num'><strong>{s['accuracy']:.1f}%</strong></td></tr>"
        for label, _, s in runs
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(
            "<title>Warehouse Bench Results</title>"
            "<style>body{font-family:system-ui;margin:2rem;max-width:60rem}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border-bottom:1px solid #ddd;padding:.5rem .7rem;text-align:left}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}</style>"
            "<h1>BIRD mini-dev — vendor comparison</h1>"
            f"<p>mnemiq baseline: {MNEMIQ_BASELINE['exact_match']}% exact-match, "
            f"{MNEMIQ_BASELINE['got_the_facts']}% got-the-facts, "
            f"{MNEMIQ_BASELINE['n']} questions.</p>"
            "<table><thead><tr><th>run</th><th class='num'>correct</th><th class='num'>wrong</th>"
            "<th class='num'>deferred</th><th class='num'>error</th>"
            "<th class='num'>gold failed</th><th class='num'>accuracy</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

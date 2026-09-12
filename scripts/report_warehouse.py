"""Turn graded vendor runs into one comparison page against mnemiq's published numbers.

Reads any number of graded JSONL files (from grade_warehouse.py) and emits a single HTML
report: one row per run, with every bucket the grader writes and both of its metrics.

No per-database or per-difficulty breakdown. An earlier version of this docstring promised
both, and the only function that could have produced them was defined and never called --
`grade_warehouse.py` prints the per-database table today.

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

# The mnemiq baseline is PASSED IN, not hardcoded. It used to be
# `{"exact_match": 40.5, "got_the_facts": 52.2}` sourced in a comment to "the mnemiq README",
# and neither figure appears in that README or anywhere else in the repo -- so a number nobody
# could re-derive was rendered into a published comparison page as this engine's own result.
# Omitted from the page entirely when not supplied, which is the honest default: a comparison
# with a missing baseline is incomplete, a comparison with an invented one is wrong.

# MUST match what `grade_warehouse.py` writes and counts. It emits a `correct_facts` bucket
# (right facts, non-identical result set) and puts it in its own denominator; this file left it
# out of both, so every such row vanished from the counts, the denominator shrank, and the
# inflated accuracy went into the HTML directly beneath the mnemiq baseline.
_BUCKETS = ("correct", "correct_facts", "wrong", "deferred", "error", "gold_failed")
# `gold_failed` is excluded from the denominator -- the gold query itself did not run, so the
# question was never posed. Every other bucket is a posed question with an answer or a refusal.
_SCORED = ("correct", "correct_facts", "wrong", "deferred", "error")


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(rows: list[dict]) -> dict:
    counts = collections.Counter(r["bucket"] for r in rows)
    scored = sum(counts[b] for b in _SCORED)
    facts = counts["correct"] + counts["correct_facts"]
    return {
        "counts": counts,
        "scored": scored,
        # The same two metrics `grade_warehouse.py` prints, over the same denominator. Reporting
        # one number called "accuracy" hid which of them it was.
        "exact_match": 100 * counts["correct"] / scored if scored else 0.0,
        "got_the_facts": 100 * facts / scored if scored else 0.0,
    }


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
    p.add_argument("--baseline-exact", type=float,
                   help="mnemiq exact-match %% to show alongside; omitted if unset")
    p.add_argument("--baseline-facts", type=float,
                   help="mnemiq got-the-facts %% to show alongside; omitted if unset")
    p.add_argument("--baseline-n", type=int, help="questions behind those two figures")
    p.add_argument("--baseline-source", default="",
                   help="where they came from -- a run id, a report, a commit")
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
        print(f"  exact-match   {summary['exact_match']:.1f}%   "
              f"got-the-facts {summary['got_the_facts']:.1f}%   "
              f"of {summary['scored']} scored")

    rows_html = "\n".join(
        f"<tr><td>{html.escape(label)}</td>"
        f"<td class='num'>{s['counts']['correct']}</td>"
        f"<td class='num'>{s['counts']['correct_facts']}</td>"
        f"<td class='num'>{s['counts']['wrong']}</td>"
        f"<td class='num'>{s['counts']['deferred']}</td>"
        f"<td class='num'>{s['counts']['error']}</td>"
        f"<td class='num'>{s['counts']['gold_failed']}</td>"
        f"<td class='num'>{s['exact_match']:.1f}%</td>"
        f"<td class='num'><strong>{s['got_the_facts']:.1f}%</strong></td></tr>"
        for label, _, s in runs
    )

    # Rendered only when the caller supplies BOTH figures, and it says where they came from.
    # A baseline is a claim about this engine's own result; if it cannot be attributed it does
    # not belong on a page that compares vendors to it.
    if args.baseline_exact is not None and args.baseline_facts is not None:
        n = f", {args.baseline_n} questions" if args.baseline_n else ""
        src = f" (source: {html.escape(args.baseline_source)})" if args.baseline_source else ""
        baseline_html = (
            f"<p>mnemiq baseline: {args.baseline_exact}% exact-match, "
            f"{args.baseline_facts}% got-the-facts{n}.{src}</p>"
        )
    else:
        baseline_html = (
            "<p><em>No mnemiq baseline supplied "
            "(--baseline-exact / --baseline-facts).</em></p>"
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
            + baseline_html +
            "<table><thead><tr><th>run</th><th class='num'>correct</th>"
            "<th class='num'>correct facts</th><th class='num'>wrong</th>"
            "<th class='num'>deferred</th><th class='num'>error</th>"
            "<th class='num'>gold failed</th><th class='num'>exact-match</th>"
            "<th class='num'>got-the-facts</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

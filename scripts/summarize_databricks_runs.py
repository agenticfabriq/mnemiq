"""Aggregate the Databricks benchmark runs: headline numbers, and how stable they are.

Two questions, and the second is the one a single run cannot answer:

  1. what did each run score -- exact match and got-the-facts, per benchmark and tier;
  2. how much of that is real. Repeats of the same benchmark against the same data should
     agree; every case whose bucket changes between runs is either vendor nondeterminism or a
     harness bug, and the two look identical in a headline percentage.

A flip rate near zero means the score is a measurement. A flip rate of several percent means
the headline number carries that much noise and small gaps between tiers are not meaningful.

  uv run python scripts/summarize_databricks_runs.py --dir eval-reports/databricks
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

CORRECT = {"correct", "correct_facts"}


def load_graded(path: str) -> dict[str, str]:
    """{case_id: bucket} for one graded run."""
    out: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["case_id"]] = rec["bucket"]
    return out


def scored(buckets: Counter) -> int:
    """The denominator: everything except golds that could not be established."""
    return sum(
        n for name, n in buckets.items() if name not in ("gold_failed", "gold_missing")
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="eval-reports/databricks")
    p.add_argument("--out", default=None, help="also write the summary to this file")
    args = p.parse_args()

    runs: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.dir, "*-graded.jsonl"))):
        name = os.path.basename(path)
        match = re.match(r"(genie|aiquery)-(bird|spider2|spider)-run(\d+)-graded\.jsonl", name)
        if not match:
            continue
        tier, benchmark, run = match.group(1), match.group(2), int(match.group(3))
        runs[(benchmark, tier)][run] = load_graded(path)

    if not runs:
        print(f"no graded runs under {args.dir}")
        return 1

    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit(f"{'benchmark':9} {'tier':8} {'run':>3} {'scored':>7} {'exact':>8} {'facts':>8} "
         f"{'deferred':>9} {'error':>7} {'excluded':>9}")
    for (benchmark, tier) in sorted(runs):
        for run in sorted(runs[(benchmark, tier)]):
            counts = Counter(runs[(benchmark, tier)][run].values())
            n = scored(counts)
            exact = f"{100 * counts['correct'] / n:.1f}%" if n else "-"
            facts = (
                f"{100 * (counts['correct'] + counts['correct_facts']) / n:.1f}%" if n else "-"
            )
            excluded = counts["gold_failed"] + counts["gold_missing"]
            emit(
                f"{benchmark:9} {tier:8} {run:>3} {n:7} {exact:>8} {facts:>8} "
                f"{counts['deferred']:9} {counts['error']:7} {excluded:9}"
            )

    emit()
    emit("run-to-run stability (cases graded in both runs)")
    emit(f"{'benchmark':9} {'tier':8} {'pair':>7} {'shared':>7} {'flipped':>8} {'flip %':>8}")
    any_pairs = False
    for (benchmark, tier) in sorted(runs):
        ordered = sorted(runs[(benchmark, tier)])
        for first, second in zip(ordered, ordered[1:]):
            a, b = runs[(benchmark, tier)][first], runs[(benchmark, tier)][second]
            shared = set(a) & set(b)
            flipped = [c for c in shared if (a[c] in CORRECT) != (b[c] in CORRECT)]
            if not shared:
                continue
            any_pairs = True
            emit(
                f"{benchmark:9} {tier:8} {f'{first}->{second}':>7} {len(shared):7} "
                f"{len(flipped):8} {100 * len(flipped) / len(shared):7.1f}%"
            )
    if not any_pairs:
        emit("  (only one run per benchmark and tier so far -- nothing to compare)")

    emit()
    emit("cases correct in at least one run but not all (the unstable set)")
    for (benchmark, tier) in sorted(runs):
        ordered = sorted(runs[(benchmark, tier)])
        if len(ordered) < 2:
            continue
        shared = set.intersection(*(set(runs[(benchmark, tier)][r]) for r in ordered))
        unstable = [
            c
            for c in sorted(shared)
            if len({runs[(benchmark, tier)][r][c] in CORRECT for r in ordered}) > 1
        ]
        emit(f"  {benchmark}/{tier}: {len(unstable)} of {len(shared)}")
        for case in unstable[:15]:
            path = " -> ".join(runs[(benchmark, tier)][r][case] for r in ordered)
            emit(f"      {case:18} {path}")
        if len(unstable) > 15:
            emit(f"      ... and {len(unstable) - 15} more")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nsummary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

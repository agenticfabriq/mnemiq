"""Re-grade saved Spider 2.0-lite runs on the CURRENT rule, through Spider's own grading path.

A stored outcome is just a word: nothing in an artifact says which grading rule produced it, and
file mtimes are not evidence because artifacts get copied. So a run's labels are recomputed rather
than dated, which is the only way to put several runs in one table honestly.

**Grade the way Spider grades.** It does not execute the gold SQL: it ships PUBLISHED result CSVs,
and a case is correct against ANY alternative the benchmark accepts (`<id>.csv` plus `<id>_*.csv`).
Two shortcuts were tried while writing this and each returned a confident wrong answer --

  * running the candidate on a bare `sqlite3` connection, where the engine used `SQLiteAdapter`:
    104 of 135 executions failed;
  * executing `gold_sql` once and comparing to that, discarding the alternatives: 70 of 135 labels
    "moved".

A rule change that moves half a corpus is a harness bug, not a finding. The correct pass moves 0
labels on five of six arms. Treat a large movement here as a reason to check this script.

  scripts/regrade_spider2.py --spider2-dir ~/src/dataset/spider2-lite eval-reports/<run>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.eval.harness import Outcome
from mnemiq.eval.spider2 import gold_alternatives, grade_alternatives, spider2_db_path

_GRADED = {"correct", "correct_facts", "wrong"}
_NAME = {Outcome.CORRECT: "correct", Outcome.CORRECT_FACTS: "correct_facts", Outcome.WRONG: "wrong"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--spider2-dir", default=os.environ.get(
        "MNEMIQ_SPIDER2_DIR", os.path.expanduser("~/src/dataset/spider2-lite")))
    p.add_argument("--suffix", default="regraded-2026-09-06",
                   help="written into the output filename, because a re-graded artifact must say so")
    args = p.parse_args()

    for run in args.runs:
        records = [json.loads(line) for line in open(run)]
        n = len(records)
        before = sum(1 for r in records if r["outcome"] == "correct")
        changed = failed = no_gold = 0
        for r in records:
            if r["outcome"] not in _GRADED:
                continue
            alternatives = gold_alternatives(args.spider2_dir, r["case_id"])
            if not alternatives:
                no_gold += 1
                continue
            try:
                adapter = SQLiteAdapter(spider2_db_path(args.spider2_dir, r["db_id"]))
                candidate = adapter.execute_arrow(r["sql"], timeout_s=60)
            except Exception:  # noqa: BLE001 -- an unrunnable candidate keeps the label it has
                failed += 1
                continue
            new = _NAME[grade_alternatives(candidate, alternatives)]
            if new != r["outcome"]:
                r["outcome_as_run"], r["outcome"] = r["outcome"], new
                changed += 1
        after = sum(1 for r in records if r["outcome"] == "correct")
        print(f"{run}: exact-match {before}/{n} ({before / n:.1%}) -> {after}/{n} ({after / n:.1%}); "
              f"{changed} labels changed, {failed} unrunnable, {no_gold} without gold")
        if changed:
            out = run.replace(".jsonl", f".{args.suffix}.jsonl")
            with open(out, "w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            print(f"  wrote {out}")
        else:
            print("  no file written: the labels it has ARE the current rule's")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

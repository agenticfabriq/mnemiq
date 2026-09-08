"""Regrade stored mnemiq engine runs under the CURRENT grade.py, without re-asking anything.

Why this exists: a mnemiq run grades inline, so each CaseResult carries the verdict of
whatever grade.py was imported when that process started. When the grader changes underneath
a series of runs -- as it did on 2026-09-02 21:37, when the pulled grade.py made exact match
order-sensitive -- the stored outcomes are no longer one metric, and a "three-run spread"
silently mixes run-to-run variance with a rule change. File mtimes cannot settle it: the
import happens at process start, not at write time.

So the SQL is re-executed and re-judged here. No model is called and no question is re-asked;
only the comparison is redone, which is the only part that changed.

  uv run python scripts/regrade_engine_runs.py --benchmark bird eval-reports/mnemiq-bird/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow as pa

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.eval.bird import bird_db_path
from mnemiq.eval.grade import results_match
from mnemiq.eval.spider import spider_db_path

_DEFAULTS = {
    "bird": os.environ.get(
        "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
    ),
    "spider": os.environ.get(
        "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
    ),
}
_PATHS = {"bird": bird_db_path, "spider": spider_db_path}


def to_table(rows: list, names: list[str]) -> pa.Table:
    if not names:
        return pa.table({})
    if not rows:
        return pa.table({n: pa.array([]) for n in names})
    columns = list(zip(*rows))
    # Duplicate column names are legal in SQL and illegal in an Arrow schema; suffix them
    # rather than letting the whole case fail as an execution error it never had.
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return pa.table({n: pa.array(list(c)) for n, c in zip(unique, columns)})


class TooManyRows(RuntimeError):
    """A result the run itself would never have materialised."""


def _run(path: str, sql: str, cap: int, timeout: float):
    """Execute one statement read-only, bounded in both rows and wall clock.

    A regrade must not attempt what the run declined to attempt: the harness caps gold at
    1000 rows, so a regrade without the same cap hangs on exactly the pathological queries
    the cap exists for. The interrupt timer bounds the other failure mode -- a query that is
    slow rather than large.
    """
    import sqlite3
    import threading

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        names = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(cap + 1)
        if len(rows) > cap:
            raise TooManyRows(f"more than {cap} rows")
        return rows, names
    finally:
        timer.cancel()
        con.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+")
    p.add_argument("--benchmark", default="bird", choices=["bird", "spider"])
    p.add_argument("--root", default=None, help="dataset root; defaults per benchmark")
    p.add_argument("--suffix", default="-regraded", help="written beside the input")
    p.add_argument(
        "--max-rows", type=int, default=1000,
        help="skip a case whose gold or candidate exceeds this many rows. The harness applies "
             "the same cap when running (_gold_too_big); without it a regrade hangs forever on "
             "the very queries the run itself declined to materialise.",
    )
    p.add_argument(
        "--timeout", type=float, default=20.0,
        help="seconds one statement may run before the case is marked ungradable",
    )
    args = p.parse_args()

    root = os.path.expanduser(args.root or _DEFAULTS[args.benchmark])
    db_path = _PATHS[args.benchmark]

    for path in args.results:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        adapters: dict[str, SQLiteAdapter] = {}
        counts = {"correct": 0, "correct_facts": 0, "wrong": 0, "other": 0, "ungradable": 0}
        changed = 0
        out = []

        for rec in rows:
            was = str(rec.get("outcome", "")).split(".")[-1].lower()
            sql, gold_sql, db_id = rec.get("sql"), rec.get("gold_sql"), rec.get("db_id")

            if not sql or not gold_sql:
                # Deferrals and errors have no SQL to re-judge; they are outcomes the grader
                # never touched, so they are carried through exactly as they were.
                counts["other"] += 1
                out.append({**rec, "regraded": was})
                continue

            if db_id not in adapters:
                adapters[db_id] = SQLiteAdapter(db_path(root, db_id))
            adapter = adapters[db_id]

            try:
                gold_rows, gold_names = _run(
                    db_path(root, db_id), gold_sql, args.max_rows, args.timeout
                )
                cand_rows, cand_names = _run(
                    db_path(root, db_id), sql, args.max_rows, args.timeout
                )
            except Exception:
                counts["ungradable"] += 1
                out.append({**rec, "regraded": "ungradable"})
                continue

            gold = to_table(gold_rows, gold_names)
            cand = to_table(cand_rows, cand_names)
            if results_match(gold, cand, allow_extra_columns=False):
                now = "correct"
            elif results_match(gold, cand, allow_extra_columns=True):
                now = "correct_facts"
            else:
                now = "wrong"
            counts[now] += 1
            changed += (now != was)
            out.append({**rec, "regraded": now})

        target = path.replace(".jsonl", f"{args.suffix}.jsonl")
        with open(target, "w") as fh:
            for rec in out:
                fh.write(json.dumps(rec, default=str) + "\n")

        graded = counts["correct"] + counts["correct_facts"] + counts["wrong"]
        denom = graded + counts["other"]  # deferrals/errors stay in, as answerable does
        exact = 100 * counts["correct"] / denom if denom else 0
        facts = 100 * (counts["correct"] + counts["correct_facts"]) / denom if denom else 0
        print(
            f"{os.path.basename(path):28} n={denom:5} exact {exact:5.1f}%  facts {facts:5.1f}%"
            f"   verdicts changed: {changed}  ungradable: {counts['ungradable']}"
        )
        print(f"    -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

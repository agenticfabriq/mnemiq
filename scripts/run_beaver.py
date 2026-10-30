"""Run BEAVER (MIT's enterprise text-to-SQL benchmark) against the live engine, on BEAVER's MySQL.

Setup (the MySQL server flags, the dataset, the model behind our numbers): docs/beaver.md.
Needs MNEMIQ_MYSQL_DSN, and MNEMIQ_LLM_* unless --control. One JSON line per question goes to
--out; a re-run resumes from it.

Examples:
  uv run python scripts/run_beaver.py --control                 # proves grading, no model
  uv run python scripts/run_beaver.py --sets dw --limit 5
  uv run python scripts/run_beaver.py --annotations all
  uv run python scripts/run_beaver.py --sets <private set> --no-grade   # SQL only, no gold
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mnemiq.eval.beaver import ANNOTATIONS, SETS, RunConfig, control_failures, run, summarize
from mnemiq.eval.mysql_source import MySQLMisconfigured, MySQLUnavailable


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beaver", default=os.environ.get("MNEMIQ_BEAVER_DIR"),
                   help="the BEAVER dataset folder, holding query/data/*.parquet "
                        "(default: $MNEMIQ_BEAVER_DIR)")
    p.add_argument("--sets", nargs="+", default=list(SETS),
                   help=f"which sets to run (default: {' '.join(SETS)})")
    p.add_argument("--limit", type=int, help="the first N questions of each set")
    p.add_argument("--ids", type=Path, help="a file of question ids, one per line: run only these")
    p.add_argument("--annotations", choices=ANNOTATIONS, default="none",
                   help="none: the question alone. all: BEAVER's annotations appended, and the "
                        "question granted only its annotated tables. no-sub-sql: as all, without "
                        "the sub-questions and their SQL")
    p.add_argument("--no-grade", action="store_true",
                   help="record each question's SQL without grading it (a set without gold)")
    p.add_argument("--control", action="store_true",
                   help="answer every question with its own gold SQL; every graded row must be "
                        "correct, which proves the grading path before any model runs")
    p.add_argument("--out", type=Path, default=Path("eval-reports/beaver-results.jsonl"),
                   help="results file, one JSON line per question; a re-run resumes from it")
    p.add_argument("--cache", type=Path, default=Path("eval-reports/beaver-cache"),
                   help="where each database's enrichment snapshot is cached")
    args = p.parse_args()

    if not args.beaver:
        print("set --beaver or MNEMIQ_BEAVER_DIR to the BEAVER dataset folder", file=sys.stderr)
        return 2
    if not os.environ.get("MNEMIQ_MYSQL_DSN"):
        print("set MNEMIQ_MYSQL_DSN (see docs/beaver.md)", file=sys.stderr)
        return 2
    settings = None
    if not args.control:
        from mnemiq.config import Settings

        settings = Settings.from_env()
        if not settings.llm_base_url or not settings.llm_model:
            print("MNEMIQ_LLM_* not configured", file=sys.stderr)
            return 2
        settings.assert_local_only()
    ids = None
    if args.ids:
        ids = frozenset(line.strip() for line in args.ids.read_text().splitlines() if line.strip())

    cfg = RunConfig(beaver_dir=Path(args.beaver), sets=args.sets, out=args.out,
                    annotations=args.annotations, grade=not args.no_grade, control=args.control,
                    limit=args.limit, ids=ids, cache_dir=args.cache)

    def progress(row: dict) -> None:
        verdict = row.get("outcome") or ("answered" if row.get("sql") else "declined")
        print(f"{row['set']:8} {row['id']:16} {verdict:20} {row['s']:6}s", flush=True)

    print(f"-> {args.out} (annotations={args.annotations}, control={args.control}, "
          f"graded={not args.no_grade})", flush=True)
    try:
        rows = run(cfg, settings=settings, on_row=progress)
    except (FileNotFoundError, ValueError, MySQLMisconfigured, MySQLUnavailable) as exc:
        print(exc, file=sys.stderr)
        return 2

    # The whole file, not only this run's rows: a resumed run's file is the measurement.
    rows_in_file = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
    print(f"\nthis run: {len(rows)} questions; {args.out}: {len(rows_in_file)}")
    rows = rows_in_file
    for set_name, counts in summarize(rows).items():
        print(f"  {set_name:8} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if args.control:
        failed = control_failures(rows)
        graded = sum(1 for r in rows if r.get("gold_gradable"))
        print(f"control: {graded - len(failed)} of {graded} gradable rows correct"
              f" ({len(rows) - graded} not gradable: gold failed or over the row limit)")
        if failed:
            print("control FAILED on: " + ", ".join(f"{r['set']}/{r['id']}" for r in failed),
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

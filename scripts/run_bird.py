"""Run BIRD mini-dev against the live engine, sliced by difficulty and database.

Examples:
  uv run python scripts/run_bird.py --limit 30
  uv run python scripts/run_bird.py --db financial --db superhero
  uv run python scripts/run_bird.py --report eval-reports
"""

from __future__ import annotations

import argparse
import sys

from mnemiq.config import Settings
from mnemiq.eval.artifacts import write_html, write_json
from mnemiq.eval.bird import load_bird
from mnemiq.eval.bird_runner import run_bird
from mnemiq.eval.report import slice_by, summarize

_DEFAULT_DIR = "/Users/user/src/dataset/bird-minidev/MINIDEV"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_DIR)
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--difficulty", choices=["simple", "moderate", "challenging"])
    p.add_argument("--no-evidence", action="store_true")
    p.add_argument("--cache", default="eval-reports/bird-cache")
    p.add_argument("--report")
    p.add_argument(
        "--results",
        default="eval-reports/bird-results.jsonl",
        help="checkpoint file; a re-run resumes from it. Delete it to start fresh.",
    )
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent questions per database (LLM-latency-bound; try 4-8). Lower if rate-limited.",
    )
    args = p.parse_args()

    settings = Settings.from_env()
    if not settings.llm_api_key:
        print("set MNEMIQ_LLM_* env", file=sys.stderr)
        return 1

    cases = load_bird(
        args.minidev,
        limit=args.limit,
        db_ids=args.dbs,
        difficulty=args.difficulty,
        with_evidence=not args.no_evidence,
    )
    print(
        f"loaded {len(cases)} BIRD cases across {len({c.db_id for c in cases})} databases",
        flush=True,
    )

    def progress(done, total, result):
        print(f"  [{done:3}/{total}] {result.outcome:20} {result.case_id}", flush=True)

    results, use = run_bird(
        cases,
        args.minidev,
        settings,
        cache_dir=args.cache,
        on_case=progress,
        results_path=args.results,
        workers=args.workers,
    )
    report = summarize(results, tokens=use["tokens"], llm_calls=use["llm_calls"])

    print("\n===== BIRD mini-dev =====")
    print(report.render())
    if use["excluded"]:
        print(f"\nexcluded {len(use['excluded'])} cases (gold > row cap): {use['excluded']}")

    print("\nby difficulty:")
    for name, rep in slice_by(results, lambda r: r.difficulty).items():
        print(f"  {name:12} {rep.accuracy:6.1%}  ({rep.correct}/{rep.answerable})")
    print("\nby database:")
    for name, rep in slice_by(results, lambda r: r.db_id).items():
        print(f"  {name:26} {rep.accuracy:6.1%}  ({rep.correct}/{rep.answerable})")

    if args.report:
        from pathlib import Path

        Path(args.report).mkdir(parents=True, exist_ok=True)
        write_json(report, f"{args.report}/bird.json", label="BIRD mini-dev")
        write_html(report, f"{args.report}/bird.html", label="BIRD mini-dev")
        print(f"\nreport written: {args.report}/bird.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

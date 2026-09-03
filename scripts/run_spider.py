"""Run Spider 1.0 dev against the live mnemiq engine, sliced by database.

The same runner as BIRD, pointed at Spider's SQLite files: identical enrichment, identical
engine, identical single-engine grading (the engine's SQL and the gold both execute on the
same native SQLite file, so a mismatch is a real error and never a cross-dialect artifact).
Only the question source and the database path differ, which is what makes the two benchmarks
comparable to each other rather than to two subtly different harnesses.

Spider carries no evidence hint and no difficulty label, so there is no --no-evidence flag and
the difficulty slice is omitted; Spider's own signal is the database.

  uv run python scripts/run_spider.py --limit 30
  uv run python scripts/run_spider.py --db concert_singer --db world_1
  uv run python scripts/run_spider.py --report eval-reports
"""

from __future__ import annotations

import argparse
import os
import sys

from mnemiq.config import Settings
from mnemiq.eval.artifacts import write_html, write_json
from mnemiq.eval.bird_runner import run_bird
from mnemiq.eval.report import slice_by, summarize
from mnemiq.eval.spider import load_spider, spider_db_path

_DEFAULT_DIR = os.environ.get(
    "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spider-dir", default=_DEFAULT_DIR)
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--cache", default="eval-reports/spider-cache")
    p.add_argument("--report")
    p.add_argument(
        "--results",
        default="eval-reports/spider-engine-results.jsonl",
        help="checkpoint file; a re-run resumes from it. Delete it to start fresh. "
        "Named -engine- so it can never be confused with the vendor spider-results.jsonl.",
    )
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent questions per database (LLM-latency-bound; try 4-8). "
        "Lower if rate-limited.",
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="self-consistency: generate N candidates per question and vote (N=5 typical). "
        "N x generation cost.",
    )
    args = p.parse_args()

    settings = Settings.from_env()
    if not settings.llm_api_key:
        print(
            "set MNEMIQ_LLM_* env (MNEMIQ_LLM_BASE_URL / MNEMIQ_LLM_API_KEY / MNEMIQ_LLM_MODEL) "
            "-- see .env.example",
            file=sys.stderr,
        )
        return 1

    spider_dir = os.path.expanduser(args.spider_dir)
    if args.refresh and os.path.exists(args.results):
        os.remove(args.results)
        meta = args.results + ".meta.json"
        if os.path.exists(meta):
            os.remove(meta)

    cases = load_spider(spider_dir, split=args.split, limit=args.limit, db_ids=args.dbs)
    print(
        f"loaded {len(cases)} Spider {args.split} cases across "
        f"{len({c.db_id for c in cases})} databases",
        flush=True,
    )

    def progress(done, total, result):
        print(f"  [{done:4}/{total}] {result.outcome:20} {result.case_id}", flush=True)

    results, use = run_bird(
        cases,
        spider_dir,
        settings,
        cache_dir=args.cache,
        on_case=progress,
        results_path=args.results,
        workers=args.workers,
        candidates=args.candidates,
        db_path_fn=spider_db_path,
    )
    report = summarize(results, tokens=use["tokens"], llm_calls=use["llm_calls"])

    print(f"\n===== Spider 1.0 {args.split} =====")
    print(report.render())
    if use["excluded"]:
        print(f"\nexcluded {len(use['excluded'])} cases (gold > row cap): {use['excluded']}")

    print("\nby database (strict / got-the-facts):")
    for name, rep in slice_by(results, lambda r: r.db_id).items():
        print(f"  {name:26} {rep.strict_accuracy:6.1%} / {rep.accuracy:6.1%}  (n={rep.answerable})")

    if args.report:
        from pathlib import Path

        label = f"Spider 1.0 {args.split}"
        Path(args.report).mkdir(parents=True, exist_ok=True)
        write_json(report, f"{args.report}/spider-engine.json", label=label)
        write_html(report, f"{args.report}/spider-engine.html", label=label)
        print(f"\nreport written: {args.report}/spider-engine.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

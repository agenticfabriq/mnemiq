"""Run the Spider 2.0-lite LOCAL slice (135 SQLite cases) against the live engine.

Grading is against the benchmark's published gold result CSVs, best-of-alternatives --
see mnemiq/eval/spider2.py for how that differs from the BIRD path and why.

Examples:
  uv run python scripts/run_spider2.py --limit 5 --db chinook
  uv run python scripts/run_spider2.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys

from mnemiq.config import Settings
from mnemiq.eval.report import slice_by, summarize
from mnemiq.eval.spider2 import load_spider2_local, run_spider2

_DEFAULT_DIR = os.environ.get(
    "MNEMIQ_SPIDER2_DIR", os.path.expanduser("~/src/dataset/spider2-lite"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spider2", default=_DEFAULT_DIR)
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--no-knowledge", action="store_true",
                   help="drop the per-question reference docs (13 cases carry one)")
    p.add_argument("--cache", default=None,
                   help="enrichment cache dir; defaults per tier so the two never mix")
    p.add_argument("--no-semantic", action="store_true",
                   help="tier 1: structural enrichment only, no LLM-written descriptions")
    p.add_argument(
        "--results",
        default="eval-reports/spider2-results.jsonl",
        help="checkpoint file; a re-run resumes from it. Delete it to start fresh.",
    )
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent questions per database (LLM-latency-bound; try 4).")
    p.add_argument("--candidates", type=int, default=1,
                   help="self-consistency candidates per question (1 = single-shot).")
    args = p.parse_args()

    settings = Settings()
    if not settings.llm_base_url or not settings.llm_api_key:
        print("MNEMIQ_LLM_* not configured", file=sys.stderr)
        return 2

    cases = load_spider2_local(
        args.spider2, limit=args.limit, db_ids=args.dbs,
        with_knowledge=not args.no_knowledge,
    )
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 2
    print(f"{len(cases)} cases across {len({c.db_id for c in cases})} databases", flush=True)

    def on_case(done: int, total: int, result) -> None:
        print(f"[{done}/{total}] {result.case_id} ({result.db_id}): {result.outcome}",
              flush=True)

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    results, meta = run_spider2(
        cases, args.spider2, settings,
        cache_dir=args.cache or (
            "eval-reports/spider2-cache" if not args.no_semantic
            else "eval-reports/spider2-cache-tier1"
        ),
        results_path=args.results,
        workers=args.workers, candidates=args.candidates, on_case=on_case,
        semantic=not args.no_semantic,
    )

    report = summarize(results, tokens=meta["tokens"], llm_calls=meta["llm_calls"])
    print(f"\n=== overall (n={report.total}) ===")
    print(report.render())
    print("\n=== by database ===")
    for db_id, sub in sorted(slice_by(results, lambda r: r.db_id).items()):
        strict = sub.correct / sub.total * 100 if sub.total else 0.0
        facts = (sub.correct + sub.correct_facts) / sub.total * 100 if sub.total else 0.0
        print(f"  {db_id:32s} {strict:5.1f}% / {facts:5.1f}%  (n={sub.total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

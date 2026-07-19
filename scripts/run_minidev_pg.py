"""Run BIRD mini-dev (PostgreSQL) against the live engine, sliced by difficulty and database.

The engine executes its generated SQL against the loaded `bird_dev` Postgres (via DuckDB);
gold PG SQL is graded on native Postgres. Point generation at a local model by setting
MNEMIQ_LLM_BASE_URL/MODEL to a vLLM server, and hold embeddings on the hosted endpoint with
MNEMIQ_EMBED_BASE_URL/KEY.

Examples:
  uv run python scripts/run_minidev_pg.py --limit 20
  uv run python scripts/run_minidev_pg.py --db financial --db superhero
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse as up

from mnemiq.config import Settings
from mnemiq.eval.bird import load_bird
from mnemiq.eval.minidev_pg import run_minidev_pg
from mnemiq.eval.report import slice_by, summarize

_DEFAULT_DIR = "/Users/user/src/dataset/bird-minidev/MINIDEV"


def _bird_dsn(base_dsn: str, db: str = "bird_dev") -> str:
    """Same Postgres server as MNEMIQ_PG_DSN, but the bird_dev database."""
    p = up.urlparse(base_dsn)
    return p._replace(path=f"/{db}").geturl()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_DIR)
    p.add_argument("--pg-dsn", help="bird_dev DSN; default: MNEMIQ_PG_DSN with db=bird_dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--difficulty", choices=["simple", "moderate", "challenging"])
    p.add_argument("--no-evidence", action="store_true")
    p.add_argument("--cache", default="eval-reports/minidev-pg-cache")
    p.add_argument("--results", default="eval-reports/minidev-pg-results.jsonl",
                   help="checkpoint file; a re-run resumes from it. Delete it to start fresh.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--candidates", type=int, default=1)
    args = p.parse_args()

    settings = Settings.from_env()
    if not settings.llm_api_key or not settings.pg_dsn:
        print("set MNEMIQ_LLM_* and MNEMIQ_PG_DSN env", file=sys.stderr)
        return 1
    pg_dsn = args.pg_dsn or _bird_dsn(settings.pg_dsn)

    cases = load_bird(
        args.minidev,
        dialect="postgresql",
        limit=args.limit,
        db_ids=args.dbs,
        difficulty=args.difficulty,
        with_evidence=not args.no_evidence,
    )
    print(f"loaded {len(cases)} mini-dev PG cases across {len({c.db_id for c in cases})} databases "
          f"| generation model: {settings.llm_model}", flush=True)

    def progress(done, total, result):
        print(f"  [{done:3}/{total}] {result.outcome:20} {result.case_id}", flush=True)

    results, use = run_minidev_pg(
        cases, args.minidev, pg_dsn, settings,
        cache_dir=args.cache, on_case=progress, results_path=args.results,
        workers=args.workers, candidates=args.candidates,
    )
    report = summarize(results, tokens=use["tokens"], llm_calls=use["llm_calls"])

    print("\n===== BIRD mini-dev (PostgreSQL) =====")
    print(f"model: {settings.llm_model}")
    print(report.render())
    print("\nby difficulty (strict-EX / got-the-facts):")
    for name, rep in slice_by(results, lambda r: r.difficulty).items():
        print(f"  {name:12} {rep.strict_accuracy:6.1%} / {rep.accuracy:6.1%}  (n={rep.answerable})")
    print("\nby database (strict-EX / got-the-facts):")
    for name, rep in slice_by(results, lambda r: r.db_id).items():
        print(f"  {name:26} {rep.strict_accuracy:6.1%} / {rep.accuracy:6.1%}  (n={rep.answerable})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

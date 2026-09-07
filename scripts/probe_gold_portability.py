"""How many CORRECT answers use SQL the gold engine cannot parse.

BIRD grades one engine against itself. mnemiq answers over DuckDB-attached Postgres and the gold
runs on Postgres directly, so an answer can be right and still be written in SQL the gold engine
will not accept -- `DOUBLE` for `DOUBLE PRECISION`, `YEAR(d)`, `strftime`, a quoted `"Match"` where
Postgres holds `match`. Those are excluded from the exact-match rate the README publishes and kept
in got-the-facts.

This is a SEPARATE question from the grading rule, and the two compose. A re-grade re-executes the
candidate through the engine's own adapter, where non-portable SQL runs fine and reports no
failure; its "0 execution failures" is therefore not evidence about portability. So this probe has
to be re-run whenever the labels move -- it subtracts from the CORRECT set, and a re-grade changes
which cases those are.

  scripts/probe_gold_portability.py <run>.regraded-2026-09-06.jsonl [...]

Needs mini-dev in Postgres (`MNEMIQ_PG_DSN`; the database name is replaced with `bird_dev`).
"""

from __future__ import annotations

import argparse
import json
import urllib.parse as up

from mnemiq.adapters.pg import PostgresAdapter
from mnemiq.config import Settings

_ANSWERABLE = {"correct", "correct_facts", "wrong", "deferred_wrongly", "error"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--database", default="bird_dev")
    args = p.parse_args()

    dsn = up.urlunparse(up.urlparse(str(Settings().pg_dsn))._replace(path=f"/{args.database}"))
    gold = PostgresAdapter(dsn)

    for run in args.runs:
        records = [json.loads(line) for line in open(run)]
        answerable = sum(1 for r in records if r["outcome"] in _ANSWERABLE)
        correct = [r for r in records if r["outcome"] == "correct"]
        reasons: dict[str, int] = {}
        for r in correct:
            try:
                gold.execute_arrow(r["sql"], timeout_s=30)
            except Exception as exc:  # noqa: BLE001 -- the refusal IS the measurement
                key = str(exc).split("\n")[0][:70]
                reasons[key] = reasons.get(key, 0) + 1
        lost = sum(reasons.values())
        print(f"\n{run}\n  answerable {answerable}, correct {len(correct)} "
              f"({len(correct) / answerable:.1%}) -> {lost} non-portable -> "
              f"portable exact-match {(len(correct) - lost) / answerable:.1%}")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>3}x {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

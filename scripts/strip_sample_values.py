"""Remove sample values from a semantic view whose own samples break Snowflake's parser.

Autopilot embeds a few real values per column to help Cortex Analyst. When one of those
values only looks numeric -- F1 lap times are `1:32.713` -- Snowflake's own YAML reader
rejects the model it wrote, and every question against the view fails with

    Semantic model failed validation with error: Invalid semantic model yaml:
    Malformed numeric value '1:32.713'

The view is otherwise fine, so this drops the `sample_values (...)` clauses and leaves
tables, relationships, descriptions and metrics untouched. BIRD hit the same bug on two of
its eleven databases.

  uv run python scripts/strip_sample_values.py --database SPIDER2 --schema F1
"""

from __future__ import annotations

import argparse
import os
import re

import snowflake.connector

# A quoted SQL string list: 'plain', 'with '' an escaped quote'.
_SAMPLES = re.compile(r" sample_values \((?:'(?:[^']|'')*'(?:,\s*)?)*\)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="SPIDER2")
    p.add_argument("--view", default="SPIDER2_SV")
    p.add_argument("--schema", action="append", dest="schemas", required=True)
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")
    try:
        for schema in args.schemas:
            cur.execute(
                f"SELECT GET_DDL('SEMANTIC_VIEW', '{args.database}.\"{schema}\".{args.view}')"
            )
            ddl = cur.fetchone()[0]
            stripped, count = _SAMPLES.subn("", ddl)
            print(f"{schema:30} {count:4} sample_values clauses removed")
            if count and not args.dry_run:
                cur.execute(f'USE SCHEMA {args.database}."{schema}"')
                cur.execute(stripped)
    finally:
        cur.close()
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

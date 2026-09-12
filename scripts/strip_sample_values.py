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

# A quoted SQL string list: 'plain', 'with '' an escaped quote'.
#
# UNROLLED, because the readable form `(?:'(?:[^']|'')*'(?:,\s*)?)*` backtracks exponentially
# -- GitHub code scanning flags it py/redos, high. It is ambiguous twice over: a run of quotes
# can split between the inner `''` alternative and the outer star, and two adjacent items with
# no comma parse either as one escaped string or as two. Measured on
# `" sample_values ('" + "'" * n`: 0.01 ms at n=14, 0.56 at n=26, 814 at n=46.
#
# The unrolled item `'[^']*(?:''[^']*)*'` has exactly one parse, and items are comma-SEPARATED.
# The trailing group is `(?:\s*,)?\s*` and NOT `\s*,?\s*`: two unbounded `\s*` either side of
# an optional separator is the same class of split point, quadratic rather than exponential,
# and the first attempt at this fix had it.
#
# `tests/test_strip_sample_values.py` holds what this is verified against, including the two
# growth curves and the widening below -- the previous version of this comment asserted a
# verification that lived only in a scratch file.
_SAMPLES = re.compile(
    r" sample_values \((?:"
    r"'[^']*(?:''[^']*)*'"
    r"(?:\s*,\s*'[^']*(?:''[^']*)*')*"
    r"(?:\s*,)?\s*"
    r")?\)"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="SPIDER2")
    p.add_argument("--view", default="SPIDER2_SV")
    p.add_argument("--schema", action="append", dest="schemas", required=True)
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Imported here, not at module scope: the driver is an optional extra, and the regex above
    # is the part worth testing. `capture_rows`, `grade_spider2` and `grade_warehouse` defer it
    # the same way.
    import snowflake.connector

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

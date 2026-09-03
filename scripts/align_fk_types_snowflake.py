"""Cast a foreign-key column to its referenced column's type, so the constraint can exist.

Spider declares some join columns with types that disagree across the two sides of the key
(concert.Stadium_ID as TEXT against stadium.Stadium_ID as NUMBER), and reading a
type-violating table as text widens a few more. Snowflake refuses a FOREIGN KEY whose types
differ, so those relationships would be missing from the semantic view -- and a missing
relationship is the difference between an answer and a refusal.

Rebuilding the child table with the column cast to the parent's type states what the schema
already means. Gold and prediction both run against the result, so the comparison stays fair.

  uv run python scripts/align_fk_types_snowflake.py --database SPIDER
"""

from __future__ import annotations

import argparse
import os
import sys

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mnemiq.eval.spider import load_keys, load_spider  # noqa: E402

_DEFAULT_SPIDER = os.environ.get(
    "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
)


def column_types(cur, database: str, schema: str) -> dict[tuple[str, str], str]:
    cur.execute(
        f"""SELECT table_name, column_name, data_type, numeric_precision, numeric_scale
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = '{schema}'"""
    )
    out = {}
    for table, column, dtype, precision, scale in cur.fetchall():
        if dtype == "NUMBER" and precision is not None:
            dtype = f"NUMBER({precision},{scale or 0})"
        out[(table.upper(), column.upper())] = dtype
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spider", default=_DEFAULT_SPIDER)
    p.add_argument("--database", default="SPIDER")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    args = p.parse_args()

    root = os.path.expanduser(args.spider)
    keys = load_keys(root)
    wanted = sorted({c.db_id for c in load_spider(root)})

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    aligned, failed = 0, []
    try:
        for db_id in wanted:
            schema = db_id.upper()
            foreign = keys.get(db_id, {}).get("foreign", [])
            if not foreign:
                continue
            types = column_types(cur, args.database, schema)

            # Group by child table: one rebuild per table, however many of its columns move.
            recasts: dict[str, dict[str, str]] = {}
            for table, column, ref_table, ref_column in foreign:
                child_key = (table.upper(), column.upper())
                parent_key = (ref_table.upper(), ref_column.upper())
                child = types.get(child_key)
                parent = types.get(parent_key)
                if not child or not parent or child == parent:
                    continue

                # An id is a number. When only one side reads as text -- usually because its
                # table hit the type-violation fallback -- move that side to the numeric type
                # rather than degrading both to text.
                child_numeric = child.startswith(("NUMBER", "FLOAT", "INT"))
                parent_numeric = parent.startswith(("NUMBER", "FLOAT", "INT"))
                if child_numeric and not parent_numeric:
                    recasts.setdefault(ref_table.upper(), {})[ref_column.upper()] = child
                else:
                    recasts.setdefault(table.upper(), {})[column.upper()] = parent

            for table, columns in recasts.items():
                cur.execute(
                    f"""SELECT column_name FROM {args.database}.INFORMATION_SCHEMA.COLUMNS
                        WHERE table_schema = '{schema}' AND table_name = '{table}'
                        ORDER BY ordinal_position"""
                )
                names = [r[0] for r in cur.fetchall()]
                projection = ", ".join(
                    f'TRY_CAST("{n}" AS {columns[n]}) AS "{n}"' if n in columns else f'"{n}"'
                    for n in names
                )
                sql = (
                    f'CREATE OR REPLACE TABLE {args.database}.{schema}."{table}" AS '
                    f'SELECT {projection} FROM {args.database}.{schema}."{table}"'
                )
                try:
                    cur.execute(sql)
                    aligned += len(columns)
                    print(f"  {schema}.{table}: {', '.join(columns)} -> parent types")
                except Exception as exc:
                    failed.append(f"{schema}.{table}: {str(exc).splitlines()[-1][:70]}")
    finally:
        cur.close()
        con.close()

    print(f"\naligned {aligned} columns")
    for line in failed:
        print(f"  FAILED {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

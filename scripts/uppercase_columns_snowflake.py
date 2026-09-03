"""Rename every lower/mixed-case column in BENCH to upper case.

INFER_SCHEMA takes column names verbatim from the Parquet footer, so the loaded tables carry
SQLite's original casing. Snowflake upper-cases unquoted identifiers, so BIRD's gold SQL
(`SELECT ... FROM account WHERE account_id = ...`) resolves to ACCOUNT_ID and fails against a
column actually named "account_id". Renaming to upper case is the Snowflake-native convention
and makes both the gold and the generated SQL resolve without quoting games.

Renames are metadata-only: no data is rewritten and existing constraints follow the column.

  uv run python scripts/uppercase_columns_snowflake.py --connection bench_key
"""

from __future__ import annotations

import argparse
import os
import sys

import snowflake.connector


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="BENCH")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    cur.execute(
        f"""SELECT table_schema, table_name, column_name
            FROM {args.database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema NOT IN ('INFORMATION_SCHEMA', 'PUBLIC')
              AND column_name <> UPPER(column_name)
            ORDER BY table_schema, table_name, ordinal_position"""
    )
    targets = cur.fetchall()
    print(f"{len(targets)} columns to rename")

    renamed, clashes = 0, []
    try:
        for schema, table, column in targets:
            sql = (
                # Quote the schema too: Spider 2.0-lite has hyphenated db_ids.
                f'ALTER TABLE {args.database}."{schema}"."{table}" '
                f'RENAME COLUMN "{column}" TO "{column.upper()}"'
            )
            if args.dry_run:
                print(f"  {sql}")
                continue
            try:
                cur.execute(sql)
                renamed += 1
            except Exception as exc:
                # Two columns differing only by case cannot both become the same upper-case
                # name -- worth reporting rather than silently losing one.
                clashes.append(f"{schema}.{table}.{column}: {str(exc).splitlines()[-1][:80]}")
    finally:
        cur.close()
        con.close()

    print(f"renamed {renamed} columns")
    if clashes:
        print(f"\n{len(clashes)} could not be renamed:")
        for line in clashes:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

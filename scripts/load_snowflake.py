"""Load the BIRD Parquet export into Snowflake: one schema per db_id, one table per file.

Reads the Parquet tree produced by scripts/bird_to_parquet.py:

    out/<db_id>/<table>.parquet   ->   <database>.<DB_ID>.<TABLE>

Connection comes from ~/.snowflake/config.toml (a named connection) unless overridden by
SNOWFLAKE_* environment variables. Browser SSO, key-pair, and password all work -- whatever
the connection is already configured for.

Examples:
  uv run python scripts/load_snowflake.py --connection bench
  uv run python scripts/load_snowflake.py --db financial --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import snowflake.connector


def parquet_tree(root: str, allowed: set[str] | None) -> list[tuple[str, list[str]]]:
    """[(db_id, [parquet paths])] for each database directory under `root`."""
    found: list[tuple[str, list[str]]] = []
    for db_id in sorted(os.listdir(root)):
        db_dir = os.path.join(root, db_id)
        if not os.path.isdir(db_dir) or (allowed is not None and db_id not in allowed):
            continue
        files = sorted(f for f in os.listdir(db_dir) if f.endswith(".parquet"))
        if files:
            found.append((db_id, [os.path.join(db_dir, f) for f in files]))
    return found


def load_db(cur, database: str, db_id: str, files: list[str], *, dry_run: bool) -> int:
    # Quote the schema: Spider 2.0-lite has db_ids with hyphens (Db-IMDB, sqlite-sakila)
    # that are not bare identifiers.
    schema = f'"{db_id.upper()}"'
    stage = f"{database}.{schema}.BIRD_STG"
    fmt = f"{database}.{schema}.PARQUET_FMT"

    def run(sql: str) -> None:
        if dry_run:
            print(f"    {sql.strip().splitlines()[0][:110]}")
            return
        cur.execute(sql)

    run(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}")
    # INFER_SCHEMA takes a *named* file format, not an inline one -- so the format is an object.
    # REPLACE_INVALID_CHARACTERS: Spider's SQLite carries a few latin-1 byte sequences that
    # are not valid UTF-8. Without this, COPY aborts the whole table on the first one.
    run(f"CREATE OR REPLACE FILE FORMAT {fmt} TYPE = PARQUET REPLACE_INVALID_CHARACTERS = TRUE")
    run(f"CREATE STAGE IF NOT EXISTS {stage} FILE_FORMAT = {fmt}")

    for path in files:
        table = os.path.splitext(os.path.basename(path))[0].upper()
        qualified = f"{database}.{schema}.\"{table}\""

        # PUT needs a forward-slashed absolute URI; AUTO_COMPRESS off, Parquet is already compressed.
        run(f"PUT 'file://{os.path.abspath(path)}' @{stage} OVERWRITE = TRUE AUTO_COMPRESS = FALSE")

        # INFER_SCHEMA reads the column names and types straight out of the Parquet footer,
        # so the Snowflake table mirrors the SQLite table without a hand-written DDL.
        run(
            f"""CREATE OR REPLACE TABLE {qualified} USING TEMPLATE (
                    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
                    FROM TABLE(INFER_SCHEMA(
                        LOCATION => '@{stage}/{os.path.basename(path)}',
                        FILE_FORMAT => '{fmt}')))"""
        )
        run(
            f"""COPY INTO {qualified}
                FROM '@{stage}/{os.path.basename(path)}'
                FILE_FORMAT = {fmt}
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE"""
        )

    return len(files)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="out", help="Parquet root from bird_to_parquet.py")
    p.add_argument("--database", default="BENCH")
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument(
        "--connection",
        default=os.environ.get("SNOWFLAKE_CONNECTION", "bench"),
        help="named connection in ~/.snowflake/config.toml",
    )
    p.add_argument("--db", action="append", dest="dbs", help="restrict to these db_ids")
    p.add_argument("--dry-run", action="store_true", help="print the SQL, connect to nothing")
    args = p.parse_args()

    if not os.path.isdir(args.out):
        print(f"no such directory: {args.out} -- run scripts/bird_to_parquet.py first", file=sys.stderr)
        return 1

    databases = parquet_tree(args.out, set(args.dbs) if args.dbs else None)
    if not databases:
        print(f"no Parquet files under {args.out}", file=sys.stderr)
        return 1

    con = cur = None
    if not args.dry_run:
        con = snowflake.connector.connect(connection_name=args.connection)
        cur = con.cursor()
        cur.execute(f"USE WAREHOUSE {args.warehouse}")
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {args.database}")

    total = 0
    try:
        for db_id, files in databases:
            print(f"  {db_id}", flush=True)
            total += load_db(cur, args.database, db_id, files, dry_run=args.dry_run)
            print(f"  {db_id:24} {len(files):3} tables loaded", flush=True)
    finally:
        if con is not None:
            cur.close()
            con.close()

    verb = "would load" if args.dry_run else "loaded"
    print(f"{verb} {total} tables into {args.database} across {len(databases)} schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare row counts in Snowflake against the BIRD SQLite sources, table by table.

A COPY INTO that silently drops or duplicates rows would show up later as unexplained
accuracy loss, not as an error. This catches it before the benchmark run.

  uv run python scripts/verify_snowflake.py --connection bench_key
"""

from __future__ import annotations

import argparse
import os
import sys

import duckdb
import snowflake.connector

from bird_to_parquet import _INTERNAL_PREFIXES, sqlite_dbs

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def sqlite_counts(sqlite_path: str) -> dict[str, int]:
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{sqlite_path}' AS s (TYPE sqlite, READ_ONLY);")
    tables = [
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog = 's'"
        ).fetchall()
        if not row[0].lower().startswith(_INTERNAL_PREFIXES)
    ]
    counts = {
        t.upper(): con.execute(f'SELECT count(*) FROM s.main."{t}"').fetchone()[0] for t in tables
    }
    con.close()
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--databases-subdir", default="dev_databases")
    p.add_argument("--database", default="BENCH")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--db", action="append", dest="dbs")
    args = p.parse_args()

    root = os.path.join(os.path.expanduser(args.minidev), args.databases_subdir)
    allowed = set(args.dbs) if args.dbs else None
    databases = [(d, path) for d, path in sqlite_dbs(root) if allowed is None or d in allowed]
    if not databases:
        print(f"no SQLite databases under {root}", file=sys.stderr)
        return 1

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()

    mismatches: list[str] = []
    checked = 0
    try:
        for db_id, sqlite_path in databases:
            schema = db_id.upper()
            expected = sqlite_counts(sqlite_path)

            actual = {
                row[0].upper(): None
                for row in cur.execute(
                    f"SELECT table_name FROM {args.database}.INFORMATION_SCHEMA.TABLES "
                    f"WHERE table_schema = '{schema}'"
                ).fetchall()
            }
            for table in actual:
                actual[table] = cur.execute(
                    f'SELECT count(*) FROM {args.database}.{schema}."{table}"'
                ).fetchone()[0]

            for table, want in sorted(expected.items()):
                checked += 1
                got = actual.get(table)
                if got is None:
                    mismatches.append(f"{schema}.{table}: MISSING in Snowflake (expected {want})")
                elif got != want:
                    mismatches.append(f"{schema}.{table}: sqlite {want} != snowflake {got}")

            extra = set(actual) - set(expected)
            for table in sorted(extra):
                mismatches.append(f"{schema}.{table}: extra table in Snowflake, not in SQLite")

            print(f"  {db_id:24} {len(expected):3} tables checked", flush=True)
    finally:
        cur.close()
        con.close()

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES:")
        for line in mismatches:
            print(f"  {line}")
        return 1

    print(f"\nall {checked} tables match row-for-row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

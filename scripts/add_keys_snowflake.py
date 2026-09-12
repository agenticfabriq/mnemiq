"""Replay BIRD's primary and foreign keys onto the Snowflake tables.

Parquet carries no constraints, so a Parquet round-trip silently drops every key the SQLite
schema declared. That matters: Cortex Analyst infers a semantic view's *relationships* from
declared keys, and without them it refuses every question needing a join -- "the provided
semantic model does not define any relationships between the tables".

mnemiq reads those keys straight off the SQLite schema during enrichment. Replaying them here
gives Snowflake the same structural information, so the comparison measures the engines rather
than an artifact of how the data was loaded.

Snowflake keeps PK/FK as informational (RELY-less, unenforced) constraints, which is exactly
what a semantic layer reads.

  uv run python scripts/add_keys_snowflake.py --connection bench_key
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc_reason import reason  # noqa: E402
from bird_to_parquet import _INTERNAL_PREFIXES, sqlite_dbs  # noqa: E402

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def keys_for(sqlite_path: str) -> tuple[dict[str, list[str]], list[tuple]]:
    """({table: [pk columns]}, [(table, column, ref_table, ref_column)])"""
    con = sqlite3.connect(sqlite_path)
    tables = [
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not r[0].lower().startswith(_INTERNAL_PREFIXES)
    ]
    primary: dict[str, list[str]] = {}
    foreign: list[tuple] = []
    for table in tables:
        pk = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")') if r[5]]
        if pk:
            primary[table] = pk
        for fk in con.execute(f'PRAGMA foreign_key_list("{table}")'):
            # (id, seq, ref_table, from_col, to_col, ...)
            ref_table, from_col, to_col = fk[2], fk[3], fk[4]
            if to_col is None:  # implicit reference to the target's primary key
                target_pk = [r[1] for r in con.execute(f'PRAGMA table_info("{ref_table}")') if r[5]]
                if not target_pk:
                    continue
                to_col = target_pk[0]
            foreign.append((table, from_col, ref_table, to_col))
    con.close()
    return primary, foreign


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--databases-subdir", default="dev_databases")
    p.add_argument("--database", default="BENCH")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = os.path.join(os.path.expanduser(args.minidev), args.databases_subdir)
    allowed = set(args.dbs) if args.dbs else None
    databases = [(d, path) for d, path in sqlite_dbs(root) if allowed is None or d in allowed]
    if not databases:
        print(f"no SQLite databases under {root}", file=sys.stderr)
        return 1

    con = cur = None
    if not args.dry_run:
        con = snowflake.connector.connect(connection_name=args.connection)
        cur = con.cursor()
        cur.execute(f"USE WAREHOUSE {args.warehouse}")

    skipped: list[str] = []

    def run(sql: str) -> bool:
        """Each constraint stands alone: an existing or unsupported one is noted, not fatal.

        These are informational constraints -- one that will not apply costs a relationship in
        the semantic view, which is worth reporting, but never worth aborting the other ten
        databases over."""
        if args.dry_run:
            print(f"    {sql}")
            return True
        try:
            cur.execute(sql)
            return True
        except Exception as exc:
            skipped.append(f"{sql[:90]}... -> {reason(exc, 80)}")
            return False

    total_pk = total_fk = 0
    try:
        for db_id, sqlite_path in databases:
            schema = db_id.upper()
            primary, foreign = keys_for(sqlite_path)

            for table, columns in primary.items():
                # Table names were uppercased at load; INFER_SCHEMA preserved column case
                # from the Parquet footer, so columns must be quoted verbatim.
                cols = ", ".join(f'"{c}"' for c in columns)
                run(
                    f'ALTER TABLE {args.database}.{schema}."{table.upper()}" '
                    f"ADD PRIMARY KEY ({cols})"
                )
                total_pk += 1

            for table, column, ref_table, ref_column in foreign:
                run(
                    f'ALTER TABLE {args.database}.{schema}."{table.upper()}" '
                    f'ADD FOREIGN KEY ("{column}") '
                    f'REFERENCES {args.database}.{schema}."{ref_table.upper()}" '
                    f'("{ref_column}")'
                )
                total_fk += 1

            print(f"  {db_id:24} {len(primary):2} PK, {len(foreign):2} FK", flush=True)
    finally:
        if con is not None:
            cur.close()
            con.close()

    verb = "would add" if args.dry_run else "added"
    print(f"{verb} {total_pk} primary keys and {total_fk} foreign keys")
    if skipped:
        print(f"\n{len(skipped)} statements did not apply:")
        for line in skipped:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

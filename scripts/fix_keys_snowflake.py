"""Rebuild BIRD's primary/foreign keys in Snowflake, correctly, from a clean slate.

The first replay ran before column names were upper-cased and added the FOREIGN KEY before the
UNIQUE its target needed, so every key pointing at a non-primary column was silently dropped --
card_games ended with none at all, and Cortex Analyst cannot infer a relationship it was never
told about.

This drops every existing constraint in the database and re-adds them in dependency order:
primary keys, then a unique key on any column a foreign key targets, then the foreign keys.

  uv run python scripts/fix_keys_snowflake.py --connection bench_key
"""

from __future__ import annotations

import argparse
import os
import sys

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc_reason import reason  # noqa: E402
from add_keys_snowflake import keys_for  # noqa: E402
from bird_to_parquet import sqlite_dbs  # noqa: E402

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def drop_existing(cur, database: str, schema: str) -> int:
    """Remove every table constraint in the schema, so the rebuild is idempotent.

    The schema is quoted. Spider 2.0-lite ids carry hyphens (DB-IMDB, SQLITE-SAKILA) and reach
    this shared function from the key builders; an unquoted hyphen parses as subtraction, so
    the ALTER raises into the bare `except` below and the constraints outlive the rebuild.

    PRECONDITION, since a quoted identifier is case-exact where an unquoted one folds to upper:
    `schema` must arrive spelled as the catalog stores it. Today it always does -- callers
    either upper-case a db id themselves or read the name back from
    `INFORMATION_SCHEMA.SCHEMATA`. A future caller passing `card_games` fails at the SELECT
    below, not at the ALTER: `constraint_schema = '{schema}'` is a plain string comparison
    against a catalog holding `CARD_GAMES`, so it returns no rows, the loop never runs, and
    the function reports zero dropped having looked at nothing.
    """
    dropped = 0
    cur.execute(
        f"""SELECT constraint_name, constraint_type, table_name
            FROM {database}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS
            WHERE constraint_schema = '{schema}'"""
    )
    # Foreign keys first: a primary or unique key cannot be dropped while one references it.
    order = {"FOREIGN KEY": 0, "UNIQUE": 1, "PRIMARY KEY": 2}
    for name, kind, table in sorted(cur.fetchall(), key=lambda r: order.get(r[1], 3)):
        try:
            cur.execute(
                f'ALTER TABLE {database}."{schema}"."{table}" DROP CONSTRAINT "{name}"'
            )
            dropped += 1
        except Exception:
            pass
    return dropped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--databases-subdir", default="dev_databases")
    p.add_argument("--database", default="BENCH")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--db", action="append", dest="dbs")
    args = p.parse_args()

    root = os.path.join(os.path.expanduser(args.minidev), args.databases_subdir)
    allowed = set(args.dbs) if args.dbs else None
    databases = [(d, path) for d, path in sqlite_dbs(root) if allowed is None or d in allowed]

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    print(f"{'database':26} {'PK':>4} {'FK want':>8} {'FK got':>7}")
    total_missing = []
    try:
        for db_id, sqlite_path in databases:
            schema = db_id.upper()
            primary, foreign = keys_for(sqlite_path)
            drop_existing(cur, args.database, schema)

            pk_done = 0
            for table, columns in primary.items():
                cols = ", ".join(f'"{c.upper()}"' for c in columns)
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}."{schema}"."{table.upper()}" '
                        f"ADD PRIMARY KEY ({cols})"
                    )
                    pk_done += 1
                except Exception:
                    pass

            # Every column a foreign key points at needs a primary or unique key first.
            targets = {
                (rt, rc) for _, _, rt, rc in foreign
                if primary.get(rt, [None])[:1] != [rc]
            }
            for ref_table, ref_column in targets:
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}."{schema}"."{ref_table.upper()}" '
                        f'ADD UNIQUE ("{ref_column.upper()}")'
                    )
                except Exception:
                    pass

            fk_done, missing = 0, []
            for table, column, ref_table, ref_column in foreign:
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}."{schema}"."{table.upper()}" '
                        f'ADD FOREIGN KEY ("{column.upper()}") '
                        f'REFERENCES {args.database}."{schema}"."{ref_table.upper()}" '
                        f'("{ref_column.upper()}")'
                    )
                    fk_done += 1
                except Exception as exc:
                    missing.append(
                        f"{schema}: {table}.{column} -> {ref_table}.{ref_column}: "
                        f"{reason(exc, 70)}"
                    )

            total_missing += missing
            flag = "" if fk_done == len(foreign) else "  <-- incomplete"
            print(f"{db_id:26} {pk_done:4} {len(foreign):8} {fk_done:7}{flag}")
    finally:
        cur.close()
        con.close()

    if total_missing:
        print(f"\n{len(total_missing)} foreign keys still could not be added:")
        for line in total_missing[:15]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

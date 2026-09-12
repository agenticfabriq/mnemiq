"""Replay Spider 2.0-lite's primary and foreign keys into Snowflake from the SQLite schemas.

Unlike Spider 1.0 there is no tables.json here, so the keys come from SQLite's own pragmas.
Cortex Analyst infers a semantic view's relationships from declared constraints, and without
them it refuses every question that needs a join -- the failure mode BIRD hit at 5/5 deferred.

  uv run python scripts/keys_spider2_snowflake.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_keys_snowflake import drop_existing  # noqa: E402

_DEFAULT_ROOT = os.environ.get(
    "MNEMIQ_SPIDER2_DIR", os.path.expanduser("~/src/dataset/spider2-lite")
)


def keys_from_sqlite(path: str) -> tuple[dict[str, list[str]], list[tuple]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            name
            for (name,) in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
            if not name.lower().startswith("sqlite_")
        ]
        primary: dict[str, list[str]] = {}
        foreign: list[tuple] = []
        for table in tables:
            pk = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")') if r[5]]
            if pk:
                primary[table] = pk
            for fk in con.execute(f'PRAGMA foreign_key_list("{table}")'):
                ref_table, from_col, to_col = fk[2], fk[3], fk[4]
                if to_col is None:
                    target = [
                        r[1] for r in con.execute(f'PRAGMA table_info("{ref_table}")') if r[5]
                    ]
                    if not target:
                        continue
                    to_col = target[0]
                foreign.append((table, from_col, ref_table, to_col))
    finally:
        con.close()
    return primary, foreign


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=_DEFAULT_ROOT)
    p.add_argument("--database", default="SPIDER2")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    args = p.parse_args()

    # `databases/`, which is where `spider2_db_path` puts them -- the glob used to sit one
    # level above that, matched nothing, ran its loop zero times and exited 0. A key-replay
    # script that silently replays no keys is the failure its own docstring describes, since
    # Cortex Analyst then refuses every join question and nothing says why.
    root = os.path.expanduser(args.root)
    files = sorted(glob.glob(os.path.join(root, "databases", "*.sqlite")))
    if not files:
        print(f"no .sqlite under {os.path.join(root, 'databases')} -- nothing to replay",
              file=sys.stderr)
        return 2
    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    print(f"{'database':30} {'PK':>4} {'FK want':>8} {'FK got':>7}")
    shortfall = []
    try:
        for path in files:
            db_id = os.path.basename(path)[:-7]
            schema = f'"{db_id.upper()}"'
            primary, foreign = keys_from_sqlite(path)
            drop_existing(cur, args.database, db_id.upper())

            pk_done = 0
            for table, columns in primary.items():
                cols = ", ".join(f'"{c.upper()}"' for c in columns)
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}.{schema}."{table.upper()}" '
                        f"ADD PRIMARY KEY ({cols})"
                    )
                    pk_done += 1
                except Exception:
                    pass

            targets = {
                (rt, rc) for _, _, rt, rc in foreign
                if [c.lower() for c in primary.get(rt, [])][:1] != [rc.lower()]
            }
            for ref_table, ref_column in targets:
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}.{schema}."{ref_table.upper()}" '
                        f'ADD UNIQUE ("{ref_column.upper()}")'
                    )
                except Exception:
                    pass

            fk_done = 0
            for table, column, ref_table, ref_column in foreign:
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}.{schema}."{table.upper()}" '
                        f'ADD FOREIGN KEY ("{column.upper()}") '
                        f'REFERENCES {args.database}.{schema}."{ref_table.upper()}" '
                        f'("{ref_column.upper()}")'
                    )
                    fk_done += 1
                except Exception:
                    pass

            flag = "" if fk_done == len(foreign) else "  <-- incomplete"
            if fk_done != len(foreign):
                shortfall.append((db_id, len(foreign) - fk_done))
            print(f"{db_id:30} {pk_done:4} {len(foreign):8} {fk_done:7}{flag}")
    finally:
        cur.close()
        con.close()

    if shortfall:
        total = sum(n for _, n in shortfall)
        print(f"\n{total} foreign keys not applied, across {len(shortfall)} databases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay Spider's primary and foreign keys into Snowflake from tables.json.

Spider declares its keys in `tables.json` rather than in the SQLite files, as index pairs into
a global column list. Cortex Analyst infers a semantic view's relationships from declared
constraints, so without this every join question is unanswerable -- the same failure BIRD hit.

Runs from a clean slate: drop every constraint, then primary keys, then a unique key on any
column a foreign key targets, then the foreign keys.

  uv run python scripts/keys_spider_snowflake.py
"""

from __future__ import annotations

import argparse
import os
import sys

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc_reason import reason  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fix_keys_snowflake import drop_existing  # noqa: E402
from mnemiq.eval.spider import load_keys, load_spider  # noqa: E402

_DEFAULT_SPIDER = os.environ.get(
    "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spider", default=_DEFAULT_SPIDER)
    p.add_argument("--database", default="SPIDER")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--split", default="dev")
    args = p.parse_args()

    root = os.path.expanduser(args.spider)
    keys = load_keys(root)
    wanted = sorted({c.db_id for c in load_spider(root, split=args.split)})

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    print(f"{'database':30} {'PK':>4} {'FK want':>8} {'FK got':>7}")
    missing: list[str] = []
    # A failed PRIMARY KEY or UNIQUE is the CAUSE of a failed FOREIGN KEY -- Snowflake will not
    # reference a column that carries neither. Both used to be swallowed whole while the
    # foreign key they enable reported its own error, so the operator read the symptom and
    # never the reason. Collected with their messages and printed FIRST, above the failures
    # they explain.
    unapplied: list[str] = []
    try:
        for db_id in wanted:
            schema = db_id.upper()
            spec = keys.get(db_id, {"primary": {}, "foreign": []})
            primary, foreign = spec["primary"], spec["foreign"]
            drop_existing(cur, args.database, schema)

            pk_done = 0
            for table, columns in primary.items():
                cols = ", ".join(f'"{c.upper()}"' for c in columns)
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}.{schema}."{table.upper()}" '
                        f"ADD PRIMARY KEY ({cols})"
                    )
                    pk_done += 1
                except Exception as exc:
                    unapplied.append(
                        f"{schema}: PRIMARY KEY {table}({', '.join(columns)}): "
                        f"{reason(exc, 60)}"
                    )

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
                except Exception as exc:
                    unapplied.append(
                        f"{schema}: UNIQUE {ref_table}.{ref_column}: "
                        f"{reason(exc, 60)}"
                    )

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
                except Exception as exc:
                    missing.append(
                        f"{schema}: {table}.{column} -> {ref_table}.{ref_column}: "
                        f"{reason(exc, 60)}"
                    )

            flag = "" if fk_done == len(foreign) else "  <-- incomplete"
            print(f"{db_id:30} {pk_done:4} {len(foreign):8} {fk_done:7}{flag}")
    finally:
        cur.close()
        con.close()

    if unapplied:
        print(f"\n{len(unapplied)} keys the foreign keys below depend on could not be added:")
        for line in unapplied[:12]:
            print(f"  {line}")
    if missing:
        print(f"\n{len(missing)} foreign keys could not be added:")
        for line in missing[:12]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

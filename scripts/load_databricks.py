"""Load the BIRD Parquet export into Databricks Unity Catalog: one schema per db_id.

Files go to a Unity Catalog volume, then each table is created from its Parquet with
`read_files`. Primary and foreign keys are replayed from the SQLite schema afterwards -- Genie
infers a space's joins from declared constraints, exactly as Cortex Analyst does, and a Parquet
round-trip carries none.

Credentials come from ~/.databrickscfg (a [DEFAULT] or named profile with host + token) or the
DATABRICKS_HOST / DATABRICKS_TOKEN environment variables.

  uv run python scripts/load_databricks.py --db financial      # pilot
  uv run python scripts/load_databricks.py                     # all 11
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from add_keys_snowflake import keys_for  # noqa: E402
from mnemiq.eval.spider import load_keys as spider_keys  # noqa: E402
from mnemiq.eval.warehouse import (  # noqa: E402
    databricks_sql_connection,
    databricks_workspace,
    schema_for,
)
from bird_to_parquet import sqlite_dbs  # noqa: E402

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def parquet_tree(root: str, allowed: set[str] | None) -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    for db_id in sorted(os.listdir(root)):
        db_dir = os.path.join(root, db_id)
        if not os.path.isdir(db_dir) or (allowed is not None and db_id not in allowed):
            continue
        files = sorted(f for f in os.listdir(db_dir) if f.endswith(".parquet"))
        if files:
            found.append((db_id, [os.path.join(db_dir, f) for f in files]))
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="out", help="Parquet root from bird_to_parquet.py")
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--databases-subdir", default="dev_databases")
    p.add_argument("--catalog", default="bench")
    p.add_argument("--volume", default="raw", help="volume name, created per schema")
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))
    p.add_argument(
        "--host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="workspace URL; triggers browser OAuth instead of a stored token",
    )
    p.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--skip-keys", action="store_true")
    p.add_argument(
        "--benchmark",
        default="bird",
        choices=["bird", "spider", "spider2"],
        help="where the keys come from: BIRD declares them in its SQLite files, Spider in "
        "tables.json, Spider 2.0-lite in flat per-database SQLite files. Getting this wrong "
        "loads the tables with no constraints at all, and Genie cannot infer a join it was "
        "never told about.",
    )
    p.add_argument(
        "--spider-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
        ),
    )
    p.add_argument(
        "--spider2-sqlite-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER2_SQLITE", os.path.expanduser("~/src/dataset/spider2-lite")
        ),
        help="spider2 only: directory of flat <db>.sqlite files",
    )
    p.add_argument(
        "--keys-report",
        default=None,
        help="write every constraint that did not apply, with the engine's reason, as JSON",
    )
    args = p.parse_args()

    if not os.path.isdir(args.out):
        print(f"no such directory: {args.out} -- run bird_to_parquet.py first", file=sys.stderr)
        return 1

    databases = parquet_tree(args.out, set(args.dbs) if args.dbs else None)
    if not databases:
        print(f"no Parquet under {args.out}", file=sys.stderr)
        return 1

    workspace = databricks_workspace(host=args.host, profile=args.profile)

    try:
        connection, warehouse_id = databricks_sql_connection(workspace, args.warehouse_id)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"using warehouse {warehouse_id}")
    cur = connection.cursor()

    # {db_id: (primary, foreign)} -- resolved from whichever source this benchmark declares its
    # keys in, so the loop below is identical for both.
    keys_by_db: dict[str, tuple[dict[str, list[str]], list[tuple[str, str, str, str]]]] = {}
    if not args.skip_keys:
        if args.benchmark == "spider2":
            # Spider 2.0-lite ships one flat <db>.sqlite per database rather than the nested
            # <db>/<db>.sqlite layout the other two use, so sqlite_dbs finds nothing here.
            root = os.path.expanduser(args.spider2_sqlite_dir)
            for name in sorted(os.listdir(root)):
                if not name.endswith(".sqlite"):
                    continue
                db_id = name[: -len(".sqlite")]
                try:
                    keys_by_db[db_id] = keys_for(os.path.join(root, name))
                except Exception as exc:
                    print(f"  no keys from {name}: {type(exc).__name__}: {exc}"[:160])
        elif args.benchmark == "spider":
            for db_id, declared in spider_keys(os.path.expanduser(args.spider_dir)).items():
                keys_by_db[db_id] = (declared["primary"], declared["foreign"])
        else:
            sqlite_root = os.path.join(os.path.expanduser(args.minidev), args.databases_subdir)
            for db_id, sqlite_path in sqlite_dbs(sqlite_root):
                keys_by_db[db_id] = keys_for(sqlite_path)

    total = 0
    try:
        # The catalog itself has to exist before any schema in it can: a fresh workspace has
        # only the metastore's defaults, and CREATE SCHEMA alone fails on a missing catalog.
        cur.execute(f"CREATE CATALOG IF NOT EXISTS {args.catalog}")
        load_failures: list[dict] = []
        for db_id, files in databases:
            schema = schema_for(db_id)
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{schema}")
            cur.execute(f"CREATE VOLUME IF NOT EXISTS {args.catalog}.{schema}.{args.volume}")
            volume_root = f"/Volumes/{args.catalog}/{schema}/{args.volume}"

            loaded = 0
            for path in files:
                table = os.path.splitext(os.path.basename(path))[0]
                target = f"{volume_root}/{os.path.basename(path)}"
                try:
                    with open(path, "rb") as fh:
                        workspace.files.upload(target, fh, overwrite=True)

                    # read_files reads the Parquet footer, so the Delta table mirrors the source
                    # schema without hand-written DDL.
                    #
                    # Column mapping is not optional here: BIRD ships column names like
                    # "Free Meal Count (K-12)", and Delta rejects spaces and parentheses in a
                    # column name unless the table opts in. Renaming them instead would break
                    # every gold query that quotes the original name.
                    cur.execute(
                        f"CREATE OR REPLACE TABLE {args.catalog}.{schema}.`{table}` "
                        f"TBLPROPERTIES ("
                        f"'delta.columnMapping.mode' = 'name', "
                        f"'delta.minReaderVersion' = '2', "
                        f"'delta.minWriterVersion' = '5') AS "
                        f"SELECT * FROM read_files('{target}', format => 'parquet')"
                    )
                    loaded += 1
                    total += 1
                except Exception as exc:
                    # One unloadable table must not abort the other ten databases: an overnight
                    # run that dies on the first bad schema wastes the whole night.
                    load_failures.append(
                        {
                            "db_id": db_id,
                            "table": table,
                            "reason": f"{type(exc).__name__}: {exc}"[:300],
                        }
                    )
                    print(f"    {db_id}.{table}: {type(exc).__name__}: {exc}"[:180], flush=True)

            print(f"  {db_id:24} {loaded:3}/{len(files):<3} tables loaded", flush=True)

        if load_failures:
            print(f"\n{len(load_failures)} tables failed to load")

        if not args.skip_keys:
            print("\nreplaying primary and foreign keys")
            applied = {"pk": 0, "fk": 0}
            failures: list[dict] = []
            for db_id, _ in databases:
                schema = schema_for(db_id)
                if db_id not in keys_by_db:
                    failures.append({"db_id": db_id, "kind": "source", "reason": "no declared keys"})
                    continue
                primary, foreign = keys_by_db[db_id]

                # NOT NULL must be set one column at a time: `ALTER COLUMN a, b SET NOT NULL` is
                # not valid Databricks SQL, so a composite primary key silently lost its NOT NULL
                # and then failed the ADD CONSTRAINT that depends on it.
                for table, columns in primary.items():
                    try:
                        for column in columns:
                            cur.execute(
                                f"ALTER TABLE {args.catalog}.{schema}.`{table}` "
                                f"ALTER COLUMN `{column}` SET NOT NULL"
                            )
                        cols = ", ".join(f"`{c}`" for c in columns)
                        cur.execute(
                            f"ALTER TABLE {args.catalog}.{schema}.`{table}` "
                            f"ADD CONSTRAINT `{table}_pk` PRIMARY KEY ({cols})"
                        )
                        applied["pk"] += 1
                    except Exception as exc:
                        failures.append(
                            {
                                "db_id": db_id,
                                "kind": "pk",
                                "table": table,
                                "columns": columns,
                                "reason": f"{type(exc).__name__}: {exc}"[:300],
                            }
                        )

                for table, column, ref_table, ref_column in foreign:
                    name = f"{table}_{column}_fk".replace(".", "_")[:120]
                    try:
                        cur.execute(
                            f"ALTER TABLE {args.catalog}.{schema}.`{table}` "
                            f"ADD CONSTRAINT `{name}` FOREIGN KEY (`{column}`) "
                            f"REFERENCES {args.catalog}.{schema}.`{ref_table}` (`{ref_column}`)"
                        )
                        applied["fk"] += 1
                    except Exception as exc:
                        failures.append(
                            {
                                "db_id": db_id,
                                "kind": "fk",
                                "table": table,
                                "column": column,
                                "references": f"{ref_table}.{ref_column}",
                                "reason": f"{type(exc).__name__}: {exc}"[:300],
                            }
                        )

            print(f"  applied {applied['pk']} primary keys, {applied['fk']} foreign keys")
            print(f"  {len(failures)} constraints did not apply")
            if args.keys_report:
                os.makedirs(os.path.dirname(args.keys_report) or ".", exist_ok=True)
                with open(args.keys_report, "w") as fh:
                    json.dump(
                        {
                            "applied": applied,
                            "failures": failures,
                            "load_failures": load_failures,
                        },
                        fh,
                        indent=2,
                    )
                print(f"  constraint failures -> {args.keys_report}")
    finally:
        cur.close()
        connection.close()

    print(f"\nloaded {total} tables into {args.catalog} across {len(databases)} schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export BIRD mini-dev (or Spider) SQLite databases to Parquet, one directory per db_id.

Layout produced:  <out>/<db_id>/<table>.parquet
That maps directly onto a schema-per-database load in Snowflake or Databricks.

Examples:
  uv run python scripts/bird_to_parquet.py --minidev ~/src/dataset/bird-minidev/MINIDEV
  uv run python scripts/bird_to_parquet.py --db financial --db superhero --out out
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import duckdb


def _lit(value: str) -> str:
    """SQL string literal. DuckDB's ATTACH and COPY..TO take a literal, not a parameter."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


# SQLite's own bookkeeping tables. Not part of any BIRD schema -- exporting them would add
# phantom tables to the warehouse and to every semantic model built over it.
_INTERNAL_PREFIXES = ("sqlite_",)


def sqlite_dbs(root: str) -> list[tuple[str, str]]:
    """(db_id, sqlite_path) for every database directory under `root`."""
    found: list[tuple[str, str]] = []
    for db_id in sorted(os.listdir(root)):
        db_dir = os.path.join(root, db_id)
        if not os.path.isdir(db_dir):
            continue
        path = os.path.join(db_dir, f"{db_id}.sqlite")
        if os.path.exists(path):
            found.append((db_id, path))
    return found


def _attach(sqlite_path: str, *, all_varchar: bool = False):
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    if all_varchar:
        # SQLite is dynamically typed, so a column declared INTEGER can hold "F" (Spider's
        # `Is_male` does). Reading that table as text is lossy but truthful; the alternative is
        # dropping the table entirely.
        con.execute("SET GLOBAL sqlite_all_varchar=true;")
    con.execute(f"ATTACH {_lit(sqlite_path)} AS s (TYPE sqlite, READ_ONLY);")
    return con


def export_db(db_id: str, sqlite_path: str, out_dir: str) -> int:
    con = _attach(sqlite_path)

    # Ask SQLite for the table list, not DuckDB's information_schema: DuckDB binds every
    # view in the file to answer that query, so one broken view definition (Spider 2.0-lite has
    # them) aborts the whole export. sqlite_master is a plain catalog read and cannot fail that
    # way, and restricting to type='table' skips views, which are not data to load anyway.
    catalog = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = [
            name
            for (name,) in catalog.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
            if not name.lower().startswith(_INTERNAL_PREFIXES)
        ]
    finally:
        catalog.close()

    os.makedirs(out_dir, exist_ok=True)
    loosened: list[str] = []
    for table in tables:
        target = os.path.join(out_dir, f"{table}.parquet")

        # SQLite's BLOB is a shrug: Spider 2.0-lite stores wrestler names and dates in BLOB
        # columns alongside actual images. Snowflake refuses to COPY text into a BINARY
        # column, so read every BLOB as text -- TRY_CAST leaves a NULL where the bytes are
        # genuinely not text (the staff photos), which no benchmark question asks about.
        described = con.execute(f'DESCRIBE SELECT * FROM s.main."{table}"').fetchall()
        projection = ", ".join(
            f'TRY_CAST("{name}" AS VARCHAR) AS "{name}"' if str(dtype).upper() == "BLOB"
            else f'"{name}"'
            for name, dtype, *_ in described
        )
        copy = f'COPY (SELECT {projection} FROM s.main."{table}") TO {_lit(target)} (FORMAT parquet);'
        try:
            con.execute(copy)
        except duckdb.Error:
            # Only this table needs the loose read; the rest keep their declared types.
            fallback = _attach(sqlite_path, all_varchar=True)
            fallback.execute(
                f'COPY (SELECT * FROM s.main."{table}") TO {_lit(target)} (FORMAT parquet);'
            )
            fallback.close()
            loosened.append(table)
    con.close()
    if loosened:
        print(f"    {db_id}: read as text due to type violations: {', '.join(loosened)}")
    return len(tables)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--minidev",
        default=os.environ.get(
            "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/MINIDEV")
        ),
        help="BIRD mini-dev root (the directory holding dev_databases/)",
    )
    p.add_argument("--databases-subdir", default="dev_databases")
    p.add_argument("--db", action="append", dest="dbs", help="restrict to these db_ids")
    p.add_argument("--out", default="out", help="output root (default: ./out)")
    args = p.parse_args()

    root = os.path.join(os.path.expanduser(args.minidev), args.databases_subdir)
    if not os.path.isdir(root):
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    allowed = set(args.dbs) if args.dbs else None
    databases = [(d, path) for d, path in sqlite_dbs(root) if allowed is None or d in allowed]
    if not databases:
        print(f"no SQLite databases found under {root}", file=sys.stderr)
        return 1

    total = 0
    for db_id, sqlite_path in databases:
        count = export_db(db_id, sqlite_path, os.path.join(args.out, db_id))
        total += count
        print(f"  {db_id:24} {count:3} tables", flush=True)

    print(f"exported {total} tables from {len(databases)} databases to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

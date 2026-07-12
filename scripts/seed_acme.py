from __future__ import annotations

import csv
import glob
import os
import re

import psycopg

from scripts.acme_ddl_translate import translate


def _table_columns(ddl_stmt: str) -> tuple[str, list[str]]:
    name = re.search(r"CREATE\s+TABLE\s+(\w+)", ddl_stmt, re.IGNORECASE).group(1).lower()
    cols = []
    for line in ddl_stmt.splitlines():
        m = re.match(r"\s*(\w+)\s+(int|bigint|varchar|char|decimal|timestamp)", line, re.IGNORECASE)
        if m and m.group(1).lower() not in {"primary", "foreign"}:
            cols.append(m.group(1).lower())
    return name, cols


def _dedupe_header(raw: list[str]) -> list[str | None]:
    """Lowercase header; mark duplicate/empty names as None so they are skipped."""
    seen: set[str] = set()
    out: list[str | None] = []
    for h in raw:
        h = h.strip().lower()
        if h and h not in seen:
            out.append(h)
            seen.add(h)
        else:
            out.append(None)
    return out


def seed(dsn: str, data_dir: str) -> dict[str, int]:
    """Load every ACME CSV into Postgres.

    Tables in ACME_small.ddl get their declared (typed) columns; the remaining CSVs
    get a table inferred from the header with all-`text` columns. Returns {table: rows}.
    """
    ddl_path = os.path.join(data_dir, "DDL", "ACME_small.ddl")
    stmts = translate(open(ddl_path).read())
    ddl_cols = dict(_table_columns(s) for s in stmts)
    ddl_create = {
        re.search(r"CREATE\s+TABLE\s+(\w+)", s, re.IGNORECASE).group(1).lower(): s for s in stmts
    }
    counts: dict[str, int] = {}
    with psycopg.connect(dsn, autocommit=True) as conn:
        for csv_path in sorted(glob.glob(os.path.join(data_dir, "data", "*.csv"))):
            table = os.path.splitext(os.path.basename(csv_path))[0].lower()
            with open(csv_path, newline="") as fh:
                reader = csv.reader(fh)
                header = _dedupe_header(next(reader))
                if table in ddl_cols:
                    valid = set(ddl_cols[table])
                    idx_to_col = {i: h for i, h in enumerate(header) if h in valid}
                    create_sql = ddl_create[table]
                else:
                    idx_to_col = {i: h for i, h in enumerate(header) if h}
                    cols = [idx_to_col[i] for i in sorted(idx_to_col)]
                    create_sql = f"CREATE TABLE {table} (" + ", ".join(f"{c} text" for c in cols) + ")"

                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                conn.execute(create_sql)

                order = sorted(idx_to_col)
                target = [idx_to_col[i] for i in order]
                placeholders = ",".join(["%s"] * len(target))
                insert = f'INSERT INTO {table} ({",".join(target)}) VALUES ({placeholders})'
                n = 0
                with conn.cursor() as cur:
                    for row in reader:
                        vals = [row[i] if i < len(row) and row[i] != "" else None for i in order]
                        cur.execute(insert, vals)
                        n += 1
                counts[table] = n
    return counts


if __name__ == "__main__":
    from mnemiq.config import Settings

    s = Settings.from_env()
    print(seed(s.pg_dsn, s.acme_data_dir))

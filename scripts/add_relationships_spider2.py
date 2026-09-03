"""Fold declared keys into semantic views that were generated before the keys existed.

Autopilot reads constraints once, at generation time: a view built over a schema with no
FOREIGN KEYs has an empty `relationships (...)` clause forever, and Cortex Analyst then
refuses every question needing a join. Rebuilding those views through the wizard would also
discard the AI-written table and column descriptions, which are part of what is being
measured.

So this edits the view instead. `GET_DDL` returns the whole definition; the keys now declared
on the tables are spliced in as `primary key (...)` on each table entry and a
`relationships (...)` block, and the result is replayed as CREATE OR REPLACE. Descriptions,
sample values, facts and dimensions all survive verbatim.

  uv run python scripts/add_relationships_spider2.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import snowflake.connector


def declared_keys(cur, database: str) -> tuple[dict, dict]:
    """({schema: {table: pk_column}}, {schema: [(child, col, parent, col), ...]})"""
    cur.execute(f"SHOW PRIMARY KEYS IN DATABASE {database}")
    names = [d[0] for d in cur.description]
    s, t, c = names.index("schema_name"), names.index("table_name"), names.index("column_name")
    primary: dict[str, dict[str, str]] = defaultdict(dict)
    for row in cur.fetchall():
        primary[row[s]].setdefault(row[t], row[c])

    cur.execute(f"SHOW IMPORTED KEYS IN DATABASE {database}")
    names = [d[0] for d in cur.description]
    idx = {n: names.index(n) for n in (
        "fk_schema_name", "fk_table_name", "fk_column_name", "pk_table_name", "pk_column_name"
    )}
    foreign: dict[str, list[tuple]] = defaultdict(list)
    for row in cur.fetchall():
        foreign[row[idx["fk_schema_name"]]].append((
            row[idx["fk_table_name"]],
            row[idx["fk_column_name"]],
            row[idx["pk_table_name"]],
            row[idx["pk_column_name"]],
        ))
    return primary, foreign


def splice(ddl: str, primary: dict[str, str], foreign: list[tuple]) -> str | None:
    """Add primary keys and a relationships block to a semantic view's DDL."""
    if "\n\trelationships (" in ddl:
        return None  # already has them; leave it alone

    # A semantic view will only accept a relationship whose target column is that table's
    # declared key, so the key each parent needs is dictated by what points at it -- not by
    # whatever SHOW PRIMARY KEYS happens to report. Where two columns of one table are
    # referenced, only the first can be the key and the other relationships are dropped.
    keys: dict[str, str] = {}
    for _, _, parent, parent_column in sorted(foreign):
        keys.setdefault(parent, primary.get(parent) or parent_column)
        if keys[parent] != parent_column and primary.get(parent) == keys[parent]:
            keys[parent] = parent_column  # prefer a key something actually references

    def add_pk(match: re.Match) -> str:
        qualified, rest = match.group(1), match.group(2)
        table = qualified.rsplit(".", 1)[-1].strip('"')
        column = keys.get(table) or primary.get(table)
        if not column:
            return match.group(0)
        # Autopilot sometimes writes a composite key that sweeps foreign key columns in
        # alongside the real one (SQLITE-SAKILA's STORE gets (MANAGER_STAFF_ID, STORE_ID)).
        # A relationship has to name the whole key, so the key is narrowed to the column
        # that actually identifies the row.
        rest = re.sub(r"^ primary key \([^)]*\)", "", rest)
        return f"\t\t{qualified} primary key ({column}){rest}"

    # Only the table entries: three dotted parts. A dimension or fact line is TABLE.COLUMN.
    ddl = re.sub(r'\t\t((?:[\w"-]+\.){2}[\w"-]+)([^\n]*)(?=\n)', add_pk, ddl)

    used: set[str] = set()
    edges: dict[str, set[str]] = defaultdict(set)

    def reaches(start: str, goal: str) -> bool:
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(edges[node])
        return False

    lines = []
    for child, child_col, parent, parent_col in sorted(foreign):
        if keys.get(parent) != parent_col:
            continue  # points at a column the parent cannot declare as its key
        # Semantic views reject cycles, and Sakila has a real one: STORE names its manager
        # in STAFF, STAFF names the store it works at. Keep the first edge, drop the one
        # that closes the loop -- a join path in one direction beats none in either.
        if reaches(parent, child):
            continue
        edges[child].add(parent)
        name = f"{child}_TO_{parent}"
        suffix = 2
        while name in used:
            name = f"{child}_TO_{parent}_{suffix}"
            suffix += 1
        used.add(name)
        lines.append(f"\t\t{name} as {child}({child_col}) references {parent}({parent_col})")
    if not lines:
        return None

    block = "\trelationships (\n" + ",\n".join(lines) + "\n\t)\n"
    # The clause sits between the tables block and facts/dimensions, as Snowflake writes it.
    for anchor in ("\tfacts (", "\tdimensions (", "\tmetrics ("):
        position = ddl.find(anchor)
        if position != -1:
            return ddl[:position] + block + ddl[position:]
    return ddl.rstrip().rstrip(";") + "\n" + block


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="SPIDER2")
    p.add_argument("--view", default="SPIDER2_SV")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--schema", action="append", dest="schemas")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")
    primary, foreign = declared_keys(cur, args.database)

    cur.execute(f"SHOW SEMANTIC VIEWS IN DATABASE {args.database}")
    names = [d[0] for d in cur.description]
    s, n = names.index("schema_name"), names.index("name")
    views = [r[s] for r in cur.fetchall() if r[n] == args.view]
    if args.schemas:
        views = [v for v in views if v in set(args.schemas)]

    print(f"{'schema':30} {'relationships':>13}")
    changed = 0
    try:
        for schema in sorted(views):
            cur.execute(
                f"SELECT GET_DDL('SEMANTIC_VIEW', '{args.database}.\"{schema}\".{args.view}')"
            )
            ddl = cur.fetchone()[0]
            # A view built through the wizard can end up stored in one schema over another
            # schema's tables (one mis-click, and it is invisible afterwards). Splicing this
            # schema's keys into someone else's tables would fail loudly at best and
            # mislabel a relationship at worst.
            body = ddl[ddl.index("tables ("):]
            referenced = {
                r.strip('"')
                for r in re.findall(rf'{args.database}\.([A-Za-z0-9_-]+|"[^"]+")\.', body)
            }
            if referenced != {schema}:
                print(f"{schema:30} {'WRONG TABLES: ' + ','.join(sorted(referenced)):>13}")
                continue

            patched = splice(ddl, primary.get(schema, {}), foreign.get(schema, []))
            if patched is None:
                print(f"{schema:30} {'unchanged':>13}")
                continue
            count = patched.count(" references ")
            if args.dry_run:
                print(f"{schema:30} {count:13}  (dry)")
                continue
            cur.execute(f'USE SCHEMA {args.database}."{schema}"')
            try:
                cur.execute(patched)
            except Exception as exc:
                print(f"{schema:30} {'FAILED':>13}  {str(exc).splitlines()[-1][:90]}")
                continue
            changed += 1
            print(f"{schema:30} {count:13}")
    finally:
        cur.close()
        con.close()

    print(f"\n{changed} semantic views rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

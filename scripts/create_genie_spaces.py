"""Create one Genie space per benchmark database, over that database's tables and nothing else.

Genie's create API takes a `serialized_space` blob. Its shape, read off an existing space, is
small: a version, a `data_sources.tables` list of fully-qualified identifiers, and optional
`instructions`, `config.sample_questions` and `benchmarks` blocks.

Only the table list is written here. Cloning a real space wholesale would carry its text
instructions and example question/SQL pairs into every benchmark space -- hand-written hints
about an unrelated dataset, which is exactly the kind of help the benchmark is supposed to be
measuring the absence of. A space with tables and no instructions is the honest baseline.

  uv run python scripts/create_genie_spaces.py --benchmark bird --catalog bench
  uv run python scripts/create_genie_spaces.py --dump-template        # inspect an existing space
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mnemiq.eval.warehouse import (  # noqa: E402
    databricks_sql_connection,
    databricks_workspace,
    schema_for,
)


def tables_in(cur, catalog: str, schema: str) -> list[str]:
    cur.execute(
        f"SELECT table_name FROM {catalog}.information_schema.tables "
        f"WHERE table_schema = '{schema}' ORDER BY table_name"
    )
    return [row[0] for row in cur.fetchall()]


def select_tables(cur, catalog: str, schema: str, tables: list[str], cap: int) -> list[str]:
    """At most `cap` tables, chosen without looking at the questions or the gold.

    Genie refuses a space holding more than 30 tables, and Spider 2.0's `oracle_sql` has 38.
    Truncating alphabetically would be arbitrary, and choosing by what the questions mention
    would feed the benchmark its own answer, so the rule is structural: keep the tables that
    participate in a foreign key, largest connected component first, and drop referentially
    isolated ones last. A table nothing joins to is the least likely to be needed for the
    multi-table analytics Spider 2.0 asks, and it is the only signal available that does not
    come from the questions.

    Dropping a table a question does need is a real cost, and it shows up honestly as a wrong
    answer rather than as a missing row.
    """
    if len(tables) <= cap:
        return tables

    cur.execute(
        f"""SELECT kcu.table_name AS fk_table, ccu.table_name AS pk_table
            FROM {catalog}.information_schema.referential_constraints rc
            JOIN {catalog}.information_schema.key_column_usage kcu
              ON rc.constraint_name = kcu.constraint_name
            JOIN {catalog}.information_schema.constraint_column_usage ccu
              ON rc.unique_constraint_name = ccu.constraint_name
            WHERE kcu.table_schema = '{schema}'"""
    )
    adjacency: dict[str, set[str]] = {t: set() for t in tables}
    for child, parent in cur.fetchall():
        if child in adjacency and parent in adjacency:
            adjacency[child].add(parent)
            adjacency[parent].add(child)

    seen: set[str] = set()
    components: list[list[str]] = []
    for table in tables:
        if table in seen:
            continue
        stack, component = [table], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node] - seen)
        components.append(sorted(component))

    # Whole components, biggest first: half a component is a broken join path.
    components.sort(key=lambda c: (-len(c), c[0]))
    kept: list[str] = []
    for component in components:
        if len(kept) + len(component) <= cap:
            kept.extend(component)
    return sorted(kept)


def space_payload(catalog: str, schema: str, tables: list[str]) -> str:
    """The minimal serialized space: this schema's tables, no instructions, no examples."""
    return json.dumps(
        {
            "version": 2,
            "data_sources": {
                "tables": [
                    {"identifier": f"{catalog}.{schema}.{table}"} for table in sorted(tables)
                ]
            },
        }
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="bench")
    p.add_argument("--benchmark", default="bird", choices=["bird", "spider", "spider2"])
    p.add_argument("--out", default=None, help="Parquet root; defaults by benchmark")
    p.add_argument("--dump-template", action="store_true", help="print every space and its blob")
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))
    p.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    p.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    p.add_argument("--spaces", default=None, help="db_id -> space_id map to write")
    p.add_argument("--parent-path", default=None)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument(
        "--max-tables",
        type=int,
        default=30,
        help="Genie rejects a space above this; larger schemas are reduced by select_tables",
    )
    args = p.parse_args()

    out = args.out or {
        "spider": "out-spider",
        "spider2": "out-spider2",
    }.get(args.benchmark, "out")
    spaces_path = args.spaces or f"eval-reports/genie-spaces-{args.benchmark}.json"

    workspace = databricks_workspace(host=args.host, profile=args.profile)

    if args.dump_template:
        for space in (workspace.genie.list_spaces().spaces or []):
            print(f"=== {space.space_id}  {space.title}")
            # workspace.genie.get_space() omits include_serialized_space, so its
            # serialized_space always comes back None; the REST call carries the flag.
            full = workspace.api_client.do(
                "GET",
                f"/api/2.0/genie/spaces/{space.space_id}",
                query={"include_serialized_space": "true"},
            )
            print(full.get("serialized_space"))
        return 0

    connection, warehouse_id = databricks_sql_connection(workspace, args.warehouse_id)
    cur = connection.cursor()

    existing = {s.title: s.space_id for s in (workspace.genie.list_spaces().spaces or [])}
    mapping: dict[str, str] = {}
    if os.path.exists(spaces_path):
        with open(spaces_path) as fh:
            mapping = json.load(fh)

    db_ids = sorted(
        d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))
    )
    if args.dbs:
        db_ids = [d for d in db_ids if d in set(args.dbs)]

    try:
        for db_id in db_ids:
            title = f"bench-{args.benchmark}-{db_id}"
            if db_id in mapping:
                print(f"  {db_id:26} already mapped")
                continue
            if title in existing:
                mapping[db_id] = existing[title]
                print(f"  {db_id:26} reusing {existing[title]}")
                continue

            schema = schema_for(db_id)
            tables = tables_in(cur, args.catalog, schema)
            if not tables:
                print(f"  {db_id:26} no tables in {args.catalog}.{schema} -- skipped")
                continue

            selected = select_tables(cur, args.catalog, schema, tables, args.max_tables)
            if len(selected) < len(tables):
                dropped = sorted(set(tables) - set(selected))
                print(
                    f"  {db_id:26} {len(tables)} tables > Genie's {args.max_tables}-table cap; "
                    f"dropped {len(dropped)} referentially isolated: {', '.join(dropped)}"
                )

            serialized = space_payload(args.catalog, schema, selected)
            try:
                space = workspace.genie.create_space(
                    warehouse_id=warehouse_id,
                    serialized_space=serialized,
                    title=title,
                    description=f"{args.benchmark} benchmark space over {args.catalog}.{schema}",
                    parent_path=args.parent_path,
                )
            except Exception as exc:
                print(f"  {db_id:26} FAILED {type(exc).__name__}: {exc}"[:200])
                continue

            mapping[db_id] = space.space_id
            print(f"  {db_id:26} {space.space_id}  ({len(selected)} tables)", flush=True)
    finally:
        cur.close()
        connection.close()

    os.makedirs(os.path.dirname(spaces_path) or ".", exist_ok=True)
    with open(spaces_path, "w") as fh:
        json.dump(mapping, fh, indent=2, sort_keys=True)
    print(f"\n{len(mapping)} spaces -> {spaces_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

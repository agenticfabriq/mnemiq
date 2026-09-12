"""Infer join keys for the Spider 2.0-lite schemas that declare none, and apply them.

Twenty of the thirty local Spider 2.0-lite databases ship without a single FOREIGN KEY in
their SQLite DDL. Cortex Analyst builds a semantic view's relationships from declared
constraints only, so on those twenty it refuses every multi-table question outright
("the provided semantic model does not define any relationships between the tables") --
the answer is a deferral, not a wrong query, and the benchmark measures nothing.

This script recovers the joins the DDL omits:

  1. candidate key = a column named ID / <TABLE>_ID / <TABLE>ID that is unique and non-null
     in the actual data;
  2. candidate FK = a column in another table with the same name (or <PARENT>_ID form);
  3. accept only if every non-null child value exists in the parent -- referential
     integrity checked against the data, not guessed from the name.

This is an enrichment Snowflake gets and mnemiq does not: mnemiq reads the same FK-less
DDL and has to infer the joins itself at query time. Runs using it must be reported as a
separate, Snowflake-favourable condition.

  uv run python scripts/infer_keys_spider2.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_keys_snowflake import drop_existing  # noqa: E402

# Types that can plausibly carry a key. Floats and booleans are excluded: a float join key
# is a coincidence, not a relationship.
_KEY_TYPES = {"TEXT", "NUMBER", "FIXED"}
# A nullable integer column arrives from Parquet as FLOAT. It cannot identify rows, but it
# can still reference them, so it is allowed on the child side and retyped before the key
# is declared.
_CHILD_TYPES = _KEY_TYPES | {"FLOAT", "REAL", "DOUBLE"}


def fetch_columns(cur, database: str, schema: str) -> dict[str, list[tuple[str, str]]]:
    cur.execute(
        f"""SELECT table_name, column_name, data_type
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position""",
        (schema,),
    )
    columns: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for table, column, dtype in cur.fetchall():
        columns[table].append((column, dtype))
    return columns


def key_shaped(table: str, column: str) -> bool:
    """Does this column name look like it identifies rows of `table`?"""
    singular = re.sub(r"(?<=[A-Z])S$", "", table)  # ORDERS -> ORDER
    return column in {
        "ID",
        f"{table}_ID",
        f"{singular}_ID",
        f"{table}ID",
        f"{singular}ID",
        f"{table}_KEY",
        f"{singular}_KEY",
    }


def id_shaped(column: str) -> bool:
    """A name that reads like an identifier, whatever table it sits on."""
    return bool(re.search(r"(^|_)(ID|KEY|CODE|NO|NUM)(_|$)|ID$", column))


def unique_non_null(cur, database: str, schema: str, table: str, column: str) -> bool:
    cur.execute(
        f'SELECT COUNT(*), COUNT("{column}"), COUNT(DISTINCT "{column}") '
        f'FROM {database}."{schema}"."{table}"'
    )
    total, non_null, distinct = cur.fetchone()
    return total > 0 and non_null == total and distinct == total


def align_column(cur, database: str, schema: str, table: str, column: str, target: str) -> bool:
    """Rebuild `table` with `column` cast to `target`, if the cast is lossless.

    SQLite stores integers and their text spellings interchangeably, so one side of a join
    lands as NUMBER and the other as TEXT. Snowflake refuses a FOREIGN KEY across a type
    mismatch, and a missing relationship is the difference between an answer and a refusal.
    The cast is checked for loss first -- a column that does not convert cleanly is left
    alone and its relationship is simply not declared.
    """
    cur.execute(
        f'SELECT COUNT("{column}"), COUNT(TRY_CAST("{column}" AS {target})), '
        f'COUNT_IF(TRY_CAST("{column}" AS FLOAT) IS NOT NULL '
        f'         AND TRY_CAST("{column}" AS FLOAT) <> TRUNC(TRY_CAST("{column}" AS FLOAT))) '
        f'FROM {database}."{schema}"."{table}"'
    )
    non_null, converted, fractional = cur.fetchone()
    if non_null != converted:
        return False
    # TRY_CAST rounds 3.5 to 4 rather than failing, so a fractional column would convert
    # "cleanly" and silently change its values. An identifier has no fractional part.
    if target.startswith("NUMBER") and fractional:
        return False

    cur.execute(
        f"""SELECT column_name FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position""",
        (schema, table),
    )
    names = [r[0] for r in cur.fetchall()]
    projection = ", ".join(
        f'TRY_CAST("{n}" AS {target}) AS "{n}"' if n == column else f'"{n}"' for n in names
    )
    cur.execute(
        f'CREATE OR REPLACE TABLE {database}."{schema}"."{table}" AS '
        f'SELECT {projection} FROM {database}."{schema}"."{table}"'
    )
    return True


def contained(
    cur,
    database: str,
    schema: str,
    child: tuple[str, str],
    parent: tuple[str, str],
    *,
    max_orphan_rate: float = 0.05,
) -> bool:
    """Does the child column reference the parent key, in the data?

    Not a strict subset test. These databases carry real referential noise -- IPL's
    PLAYER_MATCH names 124 player ids out of 12,495 rows that PLAYER does not contain -- and
    demanding zero orphans throws away a relationship the schema plainly has. Snowflake does
    not enforce foreign keys anyway; the constraint is a statement about meaning, which the
    semantic view reads. A few percent of dangling rows is dirt, a third of them is a
    coincidence, so the cut is at 5%.
    """
    ct, cc = child
    pt, pc = parent
    cur.execute(
        # Both halves must range over the SAME rows. `COUNT(c."cc")` skips NULLs, while a NULL
        # child makes the correlated predicate NULL, so EXISTS is false and the row counted as
        # an orphan -- a nullable key with 500 NULLs and 500 present values scored 500/500 = 1.0
        # and was rejected, though every value it has resolves. Orphans are counted among
        # non-null children only.
        f'SELECT COUNT(c."{cc}"), COUNT_IF(c."{cc}" IS NOT NULL AND NOT EXISTS ('
        f'  SELECT 1 FROM {database}."{schema}"."{pt}" p '
        f'  WHERE TRIM(CAST(p."{pc}" AS VARCHAR)) = TRIM(CAST(c."{cc}" AS VARCHAR)))) '
        f'FROM {database}."{schema}"."{ct}" c'
    )
    non_null, orphans = cur.fetchone()
    # A column of all NULLs contains nothing and proves nothing -- reject it.
    if not non_null:
        return False
    return orphans / non_null <= max_orphan_rate


def infer(cur, database: str, schema: str) -> tuple[list[tuple[str, str]], list[tuple]]:
    columns = fetch_columns(cur, database, schema)

    # A column name that appears on more than one table is the shape a shared key has.
    shared = defaultdict(set)
    for table, cols in columns.items():
        for column, _ in cols:
            shared[column].add(table)

    parents: list[tuple[str, str]] = []
    for table, cols in columns.items():
        for column, dtype in cols:
            if dtype.upper() not in _CHILD_TYPES:
                continue
            named_for_table = key_shaped(table, column)
            shared_id = id_shaped(column) and len(shared[column]) > 1
            # A column carried by most of the schema's tables is that schema's join key even
            # when its name says nothing (STACKING keys every table on NAME).
            widely_shared = len(shared[column]) >= max(3, len(columns) - 1)
            if not (named_for_table or shared_id or widely_shared):
                continue
            # A parent is an entity, not a bridge. DB-IMDB's M_GENRE(INDEX, MID, GID, ID)
            # holds nothing but keys, and every other M_* table's ids sit inside it, so
            # taking it as a parent invents relationships between link tables. Require at
            # least one column carrying an attribute.
            if not any(c != column and not id_shaped(c) and c != "INDEX" for c, _ in cols):
                continue
            try:
                if unique_non_null(cur, database, schema, table, column):
                    parents.append((table, column))
                    break  # one identifying column per table is enough
            except Exception:
                continue

    foreign: list[tuple] = []
    for parent_table, parent_column in parents:
        for child_table, cols in columns.items():
            if child_table == parent_table:
                continue
            for child_column, dtype in cols:
                if dtype.upper() not in _CHILD_TYPES:
                    continue
                if child_column != parent_column and not key_shaped(parent_table, child_column):
                    continue
                if (child_table, child_column) in parents:
                    continue  # its own identity, not a reference
                try:
                    if contained(
                        cur, database, schema,
                        (child_table, child_column), (parent_table, parent_column),
                    ):
                        foreign.append(
                            (child_table, child_column, parent_table, parent_column)
                        )
                except Exception:
                    continue
    return parents, foreign


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="SPIDER2")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--schema", action="append", dest="schemas")
    p.add_argument(
        "--only-keyless",
        action="store_true",
        default=True,
        help="skip schemas that already declare a foreign key (the default)",
    )
    p.add_argument("--all-schemas", dest="only_keyless", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    cur.execute(
        f"""SELECT schema_name FROM {args.database}.INFORMATION_SCHEMA.SCHEMATA
            WHERE schema_name NOT IN ('INFORMATION_SCHEMA', 'PUBLIC')
            ORDER BY schema_name"""
    )
    schemas = [r[0] for r in cur.fetchall()]
    if args.schemas:
        schemas = [s for s in schemas if s in set(args.schemas)]

    if args.only_keyless:
        cur.execute(f"SHOW IMPORTED KEYS IN DATABASE {args.database}")
        names = [d[0] for d in cur.description]
        index = names.index("fk_schema_name")
        have = {r[index] for r in cur.fetchall()}
        schemas = [s for s in schemas if s not in have]

    print(f"{len(schemas)} schemas to infer keys for")
    print(f"{'schema':30} {'PK':>4} {'FK found':>9} {'FK applied':>11}")
    total_fk = 0
    try:
        for schema in schemas:
            parents, foreign = infer(cur, args.database, schema)
            if args.dry_run:
                print(f"{schema:30} {len(parents):4} {len(foreign):9} {'(dry)':>11}")
                for row in foreign:
                    print(f"    {row[0]}.{row[1]} -> {row[2]}.{row[3]}")
                continue

            # Align the two sides of every mismatched pair before declaring anything: the
            # rebuild drops constraints, so it has to happen first.
            types = {
                (t, c): d.upper()
                for t, cols in fetch_columns(cur, args.database, schema).items()
                for c, d in cols
            }
            aligned = 0
            for child_table, child_column, parent_table, parent_column in foreign:
                child_type = types.get((child_table, child_column))
                parent_type = types.get((parent_table, parent_column))
                if child_type == parent_type:
                    continue
                # Bring both sides to one type: numeric where the values allow it, text
                # otherwise. Whichever side already holds the target is left untouched.
                for target, label in (("NUMBER(38,0)", "NUMBER"), ("VARCHAR", "TEXT")):
                    sides = [
                        s for s, t in (
                            ((child_table, child_column), child_type),
                            ((parent_table, parent_column), parent_type),
                        )
                        if t != label
                    ]
                    try:
                        if all(
                            align_column(cur, args.database, schema, s[0], s[1], target)
                            for s in sides
                        ):
                            for s in sides:
                                types[s] = label
                            aligned += len(sides)
                            break
                    except Exception:
                        continue

            drop_existing(cur, args.database, schema)
            pk_done = 0
            for table, column in parents:
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}."{schema}"."{table}" '
                        f'ADD PRIMARY KEY ("{column}")'
                    )
                    pk_done += 1
                except Exception:
                    pass

            fk_done = 0
            for child_table, child_column, parent_table, parent_column in foreign:
                try:
                    cur.execute(
                        f'ALTER TABLE {args.database}."{schema}"."{child_table}" '
                        f'ADD FOREIGN KEY ("{child_column}") '
                        f'REFERENCES {args.database}."{schema}"."{parent_table}" '
                        f'("{parent_column}")'
                    )
                    fk_done += 1
                except Exception:
                    pass
            total_fk += fk_done
            print(
                f"{schema:30} {pk_done:4} {len(foreign):9} {fk_done:11}"
                + (f"   ({aligned} columns retyped)" if aligned else "")
            )
    finally:
        cur.close()
        con.close()

    print(f"\n{total_fk} inferred foreign keys applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

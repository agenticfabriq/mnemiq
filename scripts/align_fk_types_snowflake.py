"""Cast a foreign-key column to its referenced column's type, so the constraint can exist.

Spider declares some join columns with types that disagree across the two sides of the key
(concert.Stadium_ID as TEXT against stadium.Stadium_ID as NUMBER), and reading a
type-violating table as text widens a few more. Snowflake refuses a FOREIGN KEY whose types
differ, so those relationships would be missing from the semantic view -- and a missing
relationship is the difference between an answer and a refusal.

Rebuilding the child table with the column cast to the parent's type states what the schema
already means. Gold and prediction both run against the result, so the comparison stays fair.

  uv run python scripts/align_fk_types_snowflake.py --database SPIDER
"""

from __future__ import annotations

import argparse
import os
import sys

import snowflake.connector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mnemiq.eval.spider import load_keys, load_spider  # noqa: E402

_DEFAULT_SPIDER = os.environ.get(
    "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
)


def column_types(cur, database: str, schema: str) -> dict[tuple[str, str], str]:
    cur.execute(
        f"""SELECT table_name, column_name, data_type, numeric_precision, numeric_scale
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = '{schema}'"""
    )
    out = {}
    for table, column, dtype, precision, scale in cur.fetchall():
        if dtype == "NUMBER" and precision is not None:
            dtype = f"NUMBER({precision},{scale or 0})"
        out[(table.upper(), column.upper())] = dtype
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spider", default=_DEFAULT_SPIDER)
    p.add_argument("--database", default="SPIDER")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    args = p.parse_args()

    root = os.path.expanduser(args.spider)
    keys = load_keys(root)
    wanted = sorted({c.db_id for c in load_spider(root)})

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    aligned, failed, skipped = 0, [], []
    try:
        for db_id in wanted:
            schema = db_id.upper()
            foreign = keys.get(db_id, {}).get("foreign", [])
            if not foreign:
                continue
            types = column_types(cur, args.database, schema)

            # Group by child table: one rebuild per table, however many of its columns move.
            recasts: dict[str, dict[str, str]] = {}
            for table, column, ref_table, ref_column in foreign:
                child_key = (table.upper(), column.upper())
                parent_key = (ref_table.upper(), ref_column.upper())
                child = types.get(child_key)
                parent = types.get(parent_key)
                if not child or not parent or child == parent:
                    continue

                # An id is a number. When only one side reads as text -- usually because its
                # table hit the type-violation fallback -- move that side to the numeric type
                # rather than degrading both to text.
                child_numeric = child.startswith(("NUMBER", "FLOAT", "INT"))
                parent_numeric = parent.startswith(("NUMBER", "FLOAT", "INT"))
                if child_numeric and not parent_numeric:
                    recasts.setdefault(ref_table.upper(), {})[ref_column.upper()] = child
                else:
                    recasts.setdefault(table.upper(), {})[column.upper()] = parent

            for table, columns in recasts.items():
                cur.execute(
                    f"""SELECT column_name FROM {args.database}.INFORMATION_SCHEMA.COLUMNS
                        WHERE table_schema = '{schema}' AND table_name = '{table}'
                        ORDER BY ordinal_position"""
                )
                names = [r[0] for r in cur.fetchall()]
                # TO_VARCHAR here for the same reason the probe below has it, and it MUST be
                # the same expression as the probe: a probe that certifies
                # `TRY_CAST(TO_VARCHAR(x) AS t)` while the rewrite runs `TRY_CAST(x AS t)` has
                # verified something adjacent to what executes. On a numeric source -- which
                # reaches here whenever child and parent are both numeric and unequal, say
                # NUMBER(38,0) against FLOAT -- the probe would pass and the bare TRY_CAST
                # would then be handed a non-string, which Snowflake does not accept.
                projection = ", ".join(
                    f'TRY_CAST(TO_VARCHAR("{n}") AS {columns[n]}) AS "{n}"'
                    if n in columns else f'"{n}"'
                    for n in names
                )
                # CHECK THE CAST LOSES NOTHING BEFORE OVERWRITING THE TABLE. `TRY_CAST`
                # returns NULL where it cannot convert, and `CREATE OR REPLACE` then makes
                # those NULLs the table -- silently, and counted as `aligned`. The population
                # here is the one the docstring names: tables that hit the type-violation
                # fallback and read as text hold precisely the values that violate the numeric
                # type. `infer_keys_spider2` guards the identical rewrite; this did not.
                #
                # Fractional values are rejected for the same reason it rejects them: TRY_CAST
                # rounds 3.5 to 4 rather than failing, so a fractional column converts
                # "cleanly" while changing every value. An identifier has no fractional part.
                # TO_VARCHAR around every source, because Snowflake's TRY_CAST takes a STRING
                # expression and `recasts` can hold a numeric column: NUMBER(38,0) against
                # FLOAT are both numeric and not equal, so they reach the `else` branch above
                # and this would ask TRY_CAST to read a number. Rendering to text first is
                # value-preserving for the question being asked and valid for any source type.
                #
                # The whole probe is inside a try. It is a SAFETY check, so it must not be the
                # thing that stops the run: an unreadable probe means this table cannot be
                # shown to convert losslessly, which is a reason to leave it alone and say so,
                # not a reason to abandon the tables after it. The first version of this ran
                # outside the try below and would have aborted `main()` on any probe error.
                lossy = []
                for name, target in columns.items():
                    try:
                        cur.execute(
                            f'SELECT COUNT("{name}"), '
                            f'COUNT(TRY_CAST(TO_VARCHAR("{name}") AS {target})), '
                            f'COUNT_IF(TRY_CAST(TO_VARCHAR("{name}") AS FLOAT) IS NOT NULL AND '
                            f'  TRY_CAST(TO_VARCHAR("{name}") AS FLOAT) '
                            f'  <> TRUNC(TRY_CAST(TO_VARCHAR("{name}") AS FLOAT))) '
                            f'FROM {args.database}.{schema}."{table}"'
                        )
                        non_null, converted, fractional = cur.fetchone()
                    except Exception as exc:
                        lossy.append(f"{name} (probe failed: {str(exc).splitlines()[-1][:50]})")
                        continue
                    if non_null != converted:
                        lossy.append(f"{name} ({non_null - converted} would become NULL)")
                    elif target.upper().startswith("NUMBER") and fractional:
                        lossy.append(f"{name} ({fractional} fractional, TRY_CAST would round)")
                if lossy:
                    skipped.append(f"{schema}.{table}: {', '.join(lossy)}")
                    continue

                sql = (
                    f'CREATE OR REPLACE TABLE {args.database}.{schema}."{table}" AS '
                    f'SELECT {projection} FROM {args.database}.{schema}."{table}"'
                )
                try:
                    cur.execute(sql)
                    aligned += len(columns)
                    print(f"  {schema}.{table}: {', '.join(columns)} -> parent types")
                except Exception as exc:
                    failed.append(f"{schema}.{table}: {str(exc).splitlines()[-1][:70]}")
    finally:
        cur.close()
        con.close()

    print(f"\naligned {aligned} columns")
    for line in failed:
        print(f"  FAILED {line}")
    # Reported, not silent: a skipped table keeps its data and loses its relationship, and the
    # operator has to know which. Left un-aligned is a missing constraint; aligned lossily is
    # a table of NULLs where values used to be.
    for line in skipped:
        print(f"  SKIPPED (cast would lose data) {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

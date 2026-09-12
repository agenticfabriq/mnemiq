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
import re
import sys

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


def _numeric(t: str) -> bool:
    return t.startswith(("NUMBER", "FLOAT", "INT"))


_NUMERIC_ORDER = ("NUMBER", "DECIMAL", "NUMERIC", "INT", "FLOAT", "DOUBLE", "REAL")


def _numeric_rank(t: str) -> tuple[int, int, int, str]:
    """Order numeric candidates deterministically, WIDEST first within a family.

    A key is an identifier, so the integer families outrank the floating ones -- the same
    reason the loss probe rejects a fractional column.

    Within a family the tie-break is the declared precision and scale, NOT the rendered
    string. A plain string compare put `NUMBER(10,0)` before `NUMBER(38,0)` because "1" sorts
    before "3", so a 38-digit parent was NARROWED to ten digits -- and against `NUMBER(9,0)`
    the same compare widened instead, the direction flipping on the first character of the
    digit count. Widening cannot lose a value; narrowing can, and would be caught only by the
    loss probe refusing the whole component.
    """
    digits = re.findall(r"\d+", t)
    precision = int(digits[0]) if digits else 0
    scale = int(digits[1]) if len(digits) > 1 else 0
    for rank, prefix in enumerate(_NUMERIC_ORDER):
        if t.startswith(prefix):
            return (rank, -precision, -scale, t)
    return (len(_NUMERIC_ORDER), -precision, -scale, t)


def plan_recasts(
    foreign: list[tuple], types: dict[tuple[str, str], str]
) -> list[dict[str, dict[str, str]]]:
    """One target type per CONNECTED COMPONENT of columns, one entry per component.

    Each entry is `{table: {column: target}}` -- the tables that component touches. The
    caller must treat an entry as ALL OR NOTHING: rewriting some of a component's tables and
    skipping another leaves exactly the split this function exists to prevent.

    Foreign keys chain, so the unit is the component and not the pair or even the parent
    group. Deciding a pair at a time destroyed relationships that were already declarable:
    with `P.K` TEXT, `C1.K` TEXT and `C2.K` NUMBER the C1/P pair agrees and is skipped, then
    C2 moves `P.K` to NUMBER and C1 is left TEXT against a NUMBER parent -- its FOREIGN KEY
    no longer declarable, and `aligned 1 columns` printed with nothing said about it.

    GROUPING BY PARENT ONLY MOVED THAT ONE LEVEL DOWN, which a review caught before it
    shipped: a column that moves as a child did not carry ITS children. On the chain
    G1/G2/G3 -> C -> P, the parent group moved C to NUMBER while C's own group read C's
    ORIGINAL text type and left all three grandchildren behind -- three relationships broken
    where the pair rule had broken one. Union-find over the columns is what actually closes
    it: every column reachable through a key ends at the same type.

    An id is a number, so a component holding any numeric member goes numeric rather than
    degrading every side to text; an all-text component is left alone, since casting text to
    text rewrites tables to change nothing. Where a component holds several numeric types the
    choice is `_numeric_rank`, not input order -- one of those relationships loses whatever we
    do, because a column holds one type, and it must at least lose reproducibly.

    Pure: reads the type map, returns a plan. `main` executes it.
    """
    root: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x):
        root.setdefault(x, x)
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Smaller key as the root. This does NOT affect the plan and no test can catch
            # it -- components are disjoint sets of (table, column), so membership and every
            # target are the same whichever end wins, and a mutation to `root[ra] = rb` keeps
            # the suite green. Kept because a stable component id is easier to read in a
            # debugger, not because the output depends on it.
            root[max(ra, rb)] = min(ra, rb)

    for table, column, ref_table, ref_column in foreign:
        child = (table.upper(), column.upper())
        parent = (ref_table.upper(), ref_column.upper())
        # A key naming a column the schema does not have is a stale tables.json entry.
        # Inventing a type for it would rewrite the wrong column.
        if child in types and parent in types:
            union(child, parent)

    components: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in root:
        components.setdefault(find(key), []).append(key)

    plan: list[dict[str, dict[str, str]]] = []
    for members in (sorted(components[r]) for r in sorted(components)):
        numeric = sorted((types[k] for k in members if _numeric(types[k])), key=_numeric_rank)
        if not numeric:
            continue
        target = numeric[0]
        moves: dict[str, dict[str, str]] = {}
        for table, column in members:
            if types[(table, column)] != target:
                moves.setdefault(table, {})[column] = target
        if moves:
            plan.append(moves)
    return plan


def probe_component(cur, database: str, schema: str, component: dict) -> list[str]:
    """Why this component cannot be rewritten losslessly, or [] if it can. WRITES NOTHING.

    ALL OR NOTHING, at COMPONENT scale. The probe used to run per table and skip that one
    table, which split the component it was protecting: with `P.K` TEXT, `C1.K` TEXT and
    `C2.K` NUMBER the plan moves both `P` and `C1` to NUMBER, so a `P` that fails the probe --
    the population this script exists for, text columns holding the values that violate the
    numeric type -- left `C1` rewritten to NUMBER against a still-TEXT parent. `C1` and `P`
    AGREED before the run. The operator saw "C1: K -> parent types" and "SKIPPED ... P" with
    nothing connecting them.

    The sibling states the discipline for the pair case -- "DECIDE BOTH SIDES BEFORE WRITING
    EITHER" in `infer_keys_spider2.align_pairs` -- and a component needs it at its own scale.

    TO_VARCHAR on every source, and it MUST match the rewrite's expression: a probe certifying
    `TRY_CAST(TO_VARCHAR(x) AS t)` while `TRY_CAST(x AS t)` runs has verified something
    adjacent to what executes. A numeric source reaches here whenever two sides are both
    numeric and unequal, and Snowflake's TRY_CAST does not accept one.

    Every probe is inside the try: a safety check must not be the thing that stops the run.
    An unreadable probe means this component cannot be SHOWN to convert, which is a reason to
    leave it alone and say so.
    """
    lossy: list[str] = []
    for table, columns in sorted(component.items()):
        for name, target in sorted(columns.items()):
            try:
                cur.execute(
                    f'SELECT COUNT("{name}"), '
                    f'COUNT(TRY_CAST(TO_VARCHAR("{name}") AS {target})), '
                    f'COUNT_IF(TRY_CAST(TO_VARCHAR("{name}") AS FLOAT) IS NOT NULL AND '
                    f'  TRY_CAST(TO_VARCHAR("{name}") AS FLOAT) '
                    f'  <> TRUNC(TRY_CAST(TO_VARCHAR("{name}") AS FLOAT))) '
                    f'FROM {database}.{schema}."{table}"'
                )
                non_null, converted, fractional = cur.fetchone()
            except Exception as exc:
                lossy.append(f"{table}.{name} (probe failed: {str(exc).splitlines()[-1][:50]})")
                continue
            if non_null != converted:
                lossy.append(f"{table}.{name} ({non_null - converted} would become NULL)")
            elif target.upper().startswith("NUMBER") and fractional:
                lossy.append(f"{table}.{name} ({fractional} fractional, TRY_CAST would round)")
    return lossy


def _reason(exc: Exception) -> str:
    """One short line from an exception, safely.

    `str(exc).splitlines()[-1]` raises IndexError on an empty message -- `"".splitlines()` is
    `[]` -- and every use of that idiom here sits inside a handler whose whole purpose is to
    stop a failure from ending the run. The IndexError would propagate past both loops and do
    exactly that.
    """
    lines = str(exc).splitlines()
    return (lines[-1] if lines else exc.__class__.__name__)[:70]


def apply_component(cur, database: str, schema: str, component: dict) -> dict:
    """Rewrite a whole component. Returns what happened: written / failed / unattempted.

    Extracted from `main` so the hoist and the split report are reachable by a test -- the
    property the read hoist establishes lives HERE, not in `build_rewrites`, which issues only
    a SELECT and so can never be caught writing early by any test of its own.

    ALL READS BEFORE ANY WRITE. The column read used to sit inside the write loop and outside
    any `try`, so failing on the second table -- after the first was rewritten -- escaped every
    loop and ended the run.

    Snowflake commits DDL as it runs and CREATE OR REPLACE cannot be rolled back, so a write
    failing part-way leaves the component inconsistent and nothing here can undo it. Both the
    tables already rewritten AND the ones never attempted now disagree, and the caller is told
    all three sets.
    """
    try:
        rewrites = build_rewrites(cur, database, schema, component)
    except Exception as exc:
        return {"unreadable": _reason(exc), "written": [], "failed": None, "unattempted": []}

    written: list[str] = []
    for i, (table, sql, columns) in enumerate(rewrites):
        try:
            cur.execute(sql)
            written.append((table, columns))
        except Exception as exc:
            return {
                "unreadable": None,
                "written": written,
                "failed": (table, _reason(exc)),
                # Everything after the failure is never attempted, and disagrees with what
                # WAS rewritten just as much as the failed table does. Naming only the failure
                # reads as one unlucky rewrite.
                "unattempted": [t for t, _, _ in rewrites[i + 1:]],
            }
    return {"unreadable": None, "written": written, "failed": None, "unattempted": []}


def build_rewrites(cur, database: str, schema: str, component: dict) -> list[tuple]:
    """Every `CREATE OR REPLACE` this component needs, as (table, sql, columns). READS ONLY.

    Built for the WHOLE component before any of it is executed. The column list came from
    `INFORMATION_SCHEMA` inside the write loop and outside any `try`, so a read that failed on
    the second table -- after the first was already rewritten -- did not merely split the
    component: it escaped every loop and ended the run, leaving the remaining schemas
    unprocessed too. Reading first costs nothing when it fails.

    The projection casts through TO_VARCHAR and MUST match what `probe_component` measured; a
    probe certifying one expression while another executes has verified something adjacent to
    what runs.
    """
    rewrites = []
    for table, columns in sorted(component.items()):
        cur.execute(
            f"""SELECT column_name FROM {database}.INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{schema}' AND table_name = '{table}'
                ORDER BY ordinal_position"""
        )
        names = [r[0] for r in cur.fetchall()]
        projection = ", ".join(
            f'TRY_CAST(TO_VARCHAR("{n}") AS {columns[n]}) AS "{n}"' if n in columns else f'"{n}"'
            for n in names
        )
        rewrites.append((
            table,
            f'CREATE OR REPLACE TABLE {database}.{schema}."{table}" AS '
            f'SELECT {projection} FROM {database}.{schema}."{table}"',
            columns,
        ))
    return rewrites


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

    # Deferred so `plan_recasts` above can be imported and tested without the warehouse
    # extra, the convention `tests/test_ddl_translate.py` sets.
    import snowflake.connector

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    aligned, failed, skipped, split, unreadable = 0, [], [], [], []
    try:
        for db_id in wanted:
            schema = db_id.upper()
            foreign = keys.get(db_id, {}).get("foreign", [])
            if not foreign:
                continue
            types = column_types(cur, args.database, schema)

            for component in plan_recasts(foreign, types):
                # PROBE THE WHOLE COMPONENT FIRST. Rewriting some of its tables and skipping
                # another is the split the component grouping exists to prevent.
                lossy = probe_component(cur, args.database, schema, component)
                if lossy:
                    skipped.append(f"{schema}: {', '.join(lossy)}")
                    continue

                outcome = apply_component(cur, args.database, schema, component)
                if outcome["unreadable"]:
                    unreadable.append(
                        f"{schema}: {', '.join(sorted(component))} "
                        f"({outcome['unreadable']})"
                    )
                    continue
                for table, columns in outcome["written"]:
                    aligned += len(columns)
                    target = next(iter(columns.values()))
                    print(f"  {schema}.{table}: {', '.join(columns)} -> {target}")
                if outcome["failed"]:
                    table, reason = outcome["failed"]
                    failed.append(f"{schema}.{table}: {reason}")
                    if outcome["written"]:
                        left = [t for t, _ in outcome["written"]]
                        behind = [table, *outcome["unattempted"]]
                        split.append(
                            f"{schema}: {', '.join(left)} rewritten; "
                            f"{', '.join(behind)} NOT -- their keys no longer agree"
                        )
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
    # LOUDEST, and last so it is what remains on screen. A split component is the one outcome
    # here that leaves the schema worse than it was found: tables that agreed now do not, and
    # Snowflake commits DDL as it runs so nothing here can undo it. It also changes the exit
    # code, because a caller scripting this needs to know the difference between "some
    # relationships were not declared" and "some relationships were BROKEN".
    # A separate bucket from `skipped`, which is printed as "cast would lose data". A
    # lost-data skip means the values were inspected and refused; a read failure means nothing
    # was inspected at all, and the two want opposite responses -- accept the loss, or retry.
    for line in unreadable:
        print(f"  UNREADABLE (could not read the column list) {line}")
    # LOUDEST, and last so it is what remains on screen. A split component is the one outcome
    # here that leaves the schema worse than it was found: tables that agreed now do not, and
    # Snowflake commits DDL as it runs so nothing here can undo it. It also changes the exit
    # code, because a caller scripting this needs to know the difference between "some
    # relationships were not declared" and "some relationships were BROKEN".
    for line in split:
        print(f"  SPLIT (component left inconsistent) {line}")
    return 1 if split else 0


if __name__ == "__main__":
    raise SystemExit(main())

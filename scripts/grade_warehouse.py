"""Score a vendor run against BIRD gold, using mnemiq's own result-based grader.

Both the vendor's SQL and the gold SQL execute on the *same* warehouse, so float precision,
type coercion, and NULL ordering are the engine's own -- never an artifact of comparing across
two databases. The gold is transpiled from BIRD's SQLite dialect with sqlglot.

Four buckets, and the fourth is the one people forget:

    correct      result sets state the same facts
    wrong        SQL ran, results differ
    deferred     the vendor asked for clarification instead of answering
    gold_failed  the gold could not be transpiled or executed -- EXCLUDED from the
                 denominator, because a sqlglot miss is not a vendor loss

  uv run python scripts/grade_warehouse.py --results eval-reports/cortex-results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import sqlite3

import pyarrow as pa
import sqlglot

from mnemiq.eval.bird import load_bird
from mnemiq.eval.spider import load_spider  # noqa: F401
from mnemiq.eval.grade import results_match
from mnemiq.eval.warehouse import (
    databricks_sql_connection,
    databricks_workspace,
    load_results,
    schema_for,
)

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def connect(engine: str, args):
    """(connection, cursor) for the engine the run was produced on."""
    if engine == "snowflake":
        import snowflake.connector

        con = snowflake.connector.connect(connection_name=args.connection)
        cur = con.cursor()
        cur.execute(f"USE WAREHOUSE {args.warehouse}")
        return con, cur

    workspace = databricks_workspace(host=args.host, profile=args.profile)
    con, _ = databricks_sql_connection(workspace, args.warehouse_id)
    return con, con.cursor()


def run_on_sqlite(sqlite_path: str, sql: str) -> pa.Table:
    """Execute the gold against its own SQLite file, through SQLite itself.

    Last resort, for gold only valid under SQLite's own rules -- backtick-quoted identifiers,
    and bare columns in the SELECT of a GROUP BY query. DuckDB's SQLite reader will not accept
    either, so this uses the stdlib sqlite3 driver: the engine the gold was written for.
    """
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        names = [d[0] for d in cur.description or []]
        rows = cur.fetchall()
    finally:
        con.close()

    if not names:
        return pa.table({})
    columns = list(zip(*rows)) if rows else [() for _ in names]
    return pa.table({name: pa.array(list(col)) for name, col in zip(names, columns)})


def use_schema(cur, database: str, schema: str, engine: str) -> None:
    """Point the session at the question's own schema, so unqualified names resolve.

    Unity Catalog rejects `USE SCHEMA catalog.schema` outright -- "[UC_INVALID_NAMESPACE]
    Nested or empty namespaces are not supported" -- and it has to be said in two statements.
    Getting this wrong is silent and total: the gold falls back to its local SQLite file, every
    candidate execution raises, and every question in the run grades `wrong`, which reads as a
    vendor scoring zero rather than as a broken harness.
    """
    if engine == "snowflake":
        cur.execute(f"USE SCHEMA {database}.{schema}")
    else:
        cur.execute(f"USE CATALOG {database}")
        cur.execute(f"USE SCHEMA {schema}")


def run_sql(cur, sql: str, database: str, schema: str, engine: str) -> pa.Table:
    """Execute in the question's own schema, so unqualified table names resolve."""
    use_schema(cur, database, schema, engine)
    cur.execute(sql)
    if engine == "snowflake":
        table = cur.fetch_arrow_all()
        # fetch_arrow_all returns None for an empty result; an empty table compares correctly.
        return table if table is not None else pa.table({})

    rows = cur.fetchall()
    names = [d[0] for d in cur.description]
    if not rows:
        return pa.table({name: pa.array([]) for name in names})
    columns = list(zip(*rows))
    return pa.table({name: pa.array(list(col)) for name, col in zip(names, columns)})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="eval-reports/cortex-results.jsonl")
    p.add_argument(
        "--engine",
        default="snowflake",
        choices=["snowflake", "databricks"],
        help="which warehouse the run was produced on; also the sqlglot target for the gold",
    )
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))
    p.add_argument(
        "--host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="workspace URL; triggers browser OAuth instead of a stored token",
    )
    p.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    p.add_argument("--database", default=None, help="default: BENCH (snowflake) / bench (databricks)")
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--dialect", default=None, help="sqlglot target; defaults to --engine")
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--benchmark", default="bird", choices=["bird", "spider"])
    p.add_argument(
        "--spider-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
        ),
    )
    p.add_argument(
        "--no-sqlite-fallback",
        action="store_true",
        help="do not fall back to executing gold on its own SQLite file",
    )
    p.add_argument(
        "--no-gold-fallback",
        action="store_true",
        help="do not retry a failed gold from BIRD's hand-refined PostgreSQL gold",
    )
    p.add_argument("--report", default="eval-reports/cortex-graded.jsonl")
    args = p.parse_args()
    args.dialect = args.dialect or args.engine
    if args.database is None:
        args.database = "BENCH" if args.engine == "snowflake" else "bench"

    results = load_results(args.results)
    if not results:
        print(f"no results in {args.results} -- run the vendor runner first")
        return 1

    # BIRD ships the same 500 questions with gold hand-refined for PostgreSQL. Where the
    # SQLite gold will not transpile (JULIANDAY, casts, quoting), the Postgres gold usually
    # will -- and recovering it puts the question back in the denominator instead of dropping
    # it, which is the difference between measuring the vendor and measuring sqlglot.
    # Spider ships a single SQLite gold, so there is no second dialect to fall back to.
    fallback_gold: dict[str, str] = {}
    if not args.no_gold_fallback and args.benchmark == "bird":
        try:
            fallback_gold = {
                case.id: case.gold_sql
                for case in load_bird(os.path.expanduser(args.minidev), dialect="postgresql")
            }
        except Exception as exc:
            print(f"no PostgreSQL gold fallback available: {exc}")

    def sqlite_path_for(db_id: str) -> str | None:
        if args.benchmark == "spider":
            root = os.path.expanduser(args.spider_dir)
            path = os.path.join(root, "database", db_id, f"{db_id}.sqlite")
        else:
            root = os.path.expanduser(args.minidev)
            path = os.path.join(root, "dev_databases", db_id, f"{db_id}.sqlite")
        return path if os.path.exists(path) else None

    con, cur = connect(args.engine, args)

    buckets: Counter[str] = Counter()
    by_db: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_difficulty: defaultdict[str, Counter[str]] = defaultdict(Counter)
    graded: list[dict] = []

    try:
        for index, (case_id, result) in enumerate(sorted(results.items()), 1):
            schema = (
                result.db_id.upper() if args.engine == "snowflake" else schema_for(result.db_id)
            )
            bucket, detail = "", ""

            if result.deferral:
                bucket = "deferred"
                detail = result.deferral[:200]
            elif result.error:
                bucket = "error"
                detail = result.error[:200]
            else:
                # The gold runs first: if it cannot run, this question cannot score the vendor
                # either way, and must leave the denominator entirely.
                gold = None
                failure = ""
                # `mysql` as a read dialect is not a mistake: both BIRD and Spider gold use
                # double quotes for string literals ("France"), which SQLite tolerates but
                # Snowflake reads as an identifier. Parsing as MySQL rewrites them correctly.
                for source, sql in (
                    ("sqlite", result.gold_sql),
                    ("mysql", result.gold_sql),
                    ("postgres", fallback_gold.get(case_id)),
                ):
                    if not sql:
                        continue
                    try:
                        gold_sql = sqlglot.transpile(sql, read=source, write=args.dialect)[0]
                        gold = run_sql(cur, gold_sql, args.database, schema, args.engine)
                        break
                    except Exception as exc:
                        failure = f"{type(exc).__name__}: {exc}"[:200]

                if gold is None and not args.no_sqlite_fallback:
                    path = sqlite_path_for(result.db_id)
                    if path:
                        try:
                            gold = run_on_sqlite(path, result.gold_sql)
                        except Exception as exc:
                            failure = f"sqlite fallback: {type(exc).__name__}: {exc}"[:200]

                if gold is None:
                    bucket, detail = "gold_failed", failure
                else:
                    try:
                        candidate = run_sql(cur, result.sql, args.database, schema, args.engine)
                    except Exception as exc:
                        bucket, detail = "wrong", f"execution failed: {exc}"[:200]
                    else:
                        # mnemiq grades every case twice from one execution: an exact
                        # result-set match, then the same comparison allowing extra context
                        # columns. Reporting only the strict number and comparing it to
                        # mnemiq's got-the-facts figure would pit two different rules
                        # against each other.
                        if results_match(gold, candidate, allow_extra_columns=False):
                            bucket = "correct"
                        elif results_match(gold, candidate, allow_extra_columns=True):
                            bucket = "correct_facts"
                        else:
                            bucket = "wrong"

            buckets[bucket] += 1
            by_db[result.db_id][bucket] += 1
            if result.difficulty:
                by_difficulty[result.difficulty][bucket] += 1
            graded.append(
                {
                    "case_id": case_id,
                    "db_id": result.db_id,
                    "difficulty": result.difficulty,
                    "bucket": bucket,
                    "detail": detail,
                    "sql": result.sql,
                }
            )
            if index % 25 == 0:
                print(f"  graded {index}/{len(results)}", flush=True)
    finally:
        cur.close()
        con.close()

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        for row in graded:
            fh.write(json.dumps(row) + "\n")

    scored = (
        buckets["correct"]
        + buckets["correct_facts"]
        + buckets["wrong"]
        + buckets["deferred"]
        + buckets["error"]
    )
    print(f"\n{'bucket':14} {'n':>5}  {'% of scored':>11}")
    for name in ("correct", "correct_facts", "wrong", "deferred", "error", "gold_failed"):
        n = buckets[name]
        share = "excluded" if name == "gold_failed" else f"{100 * n / scored:.1f}%" if scored else "-"
        print(f"{name:14} {n:5}  {share:>11}")
    facts = buckets["correct"] + buckets["correct_facts"]
    print(
        f"\nexact-match     {100 * buckets['correct'] / scored:.1f}%  "
        f"({buckets['correct']}/{scored})   -- mnemiq's strict metric"
    )
    print(
        f"got-the-facts   {100 * facts / scored:.1f}%  "
        f"({facts}/{scored})   -- mnemiq's product metric"
    )
    print(f"(gold_failed {buckets['gold_failed']} excluded from the denominator)")

    print(f"\n{'database':26} {'scored':>7} {'exact':>8} {'facts':>8}")
    for db_id in sorted(by_db):
        c = by_db[db_id]
        n = c["correct"] + c["correct_facts"] + c["wrong"] + c["deferred"] + c["error"]
        exact = f"{100 * c['correct'] / n:.1f}%" if n else "-"
        fact = f"{100 * (c['correct'] + c['correct_facts']) / n:.1f}%" if n else "-"
        print(f"{db_id:26} {n:7} {exact:>8} {fact:>8}")

    if by_difficulty:
        print(f"\n{'difficulty':26} {'scored':>7} {'exact':>8} {'facts':>8}")
        for level in ("simple", "moderate", "challenging"):
            c = by_difficulty.get(level)
            if not c:
                continue
            n = c["correct"] + c["correct_facts"] + c["wrong"] + c["deferred"] + c["error"]
            exact = f"{100 * c['correct'] / n:.1f}%" if n else "-"
            fact = f"{100 * (c['correct'] + c['correct_facts']) / n:.1f}%" if n else "-"
            print(f"{level:26} {n:7} {exact:>8} {fact:>8}")

    print(f"\nper-case detail -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

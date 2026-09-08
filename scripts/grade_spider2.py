"""Score a Spider 2.0-lite run with Spider 2.0's own evaluator.

Spider 2.0-lite has no gold SQL for the local instances -- correctness is defined by stored
gold *result* CSVs, and several may be acceptable for one question (`local002_a.csv`,
`local002_b.csv`, ...). Its `compare_pandas_table` is column containment: every gold column
must appear somewhere in the prediction, extra prediction columns are allowed, order is
per-question (`ignore_order`), numbers compare to 1e-2. That is mnemiq's got-the-facts rule,
so got-facts is the headline here.

Spider 2.0 stops there, so the strict half of the pair is defined rather than adopted: the
benchmark's own match PLUS the prediction returning the gold's column count, mirroring
results_match's allow_extra_columns on BIRD. Only got-facts compares to a published Spider 2.0
score; the exact column travels no further than this repo.

A prediction is scored against each accepted gold in turn; matching any one of them counts.

    correct        the prediction contains every gold column, in the gold's own shape
    correct_facts  contains every gold column but returns extra ones
    wrong       SQL ran, the columns do not match any accepted gold
    deferred    Cortex asked for clarification instead of answering
    error       the generated SQL failed to execute
    no_gold     no gold CSV shipped for the instance -- EXCLUDED from the denominator

  uv run python scripts/grade_spider2.py --results eval-reports/spider2-results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from mnemiq.eval.spider2_vendor import compare_tables, gold_paths, load_eval_meta
from mnemiq.eval.warehouse import (
    databricks_sql_connection,
    databricks_workspace,
    load_results,
    schema_for,
)

_DEFAULT_SUITE = os.environ.get(
    "MNEMIQ_SPIDER2_SUITE",
    os.path.expanduser("~/src/dataset/spider2-lite/Spider2/spider2-lite/evaluation_suite"),
)


def read_gold(path: str) -> list[list]:
    """A gold CSV as column-major values, matching what the comparison expects."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return []
    body = rows[1:]  # first line is the header
    # Width comes from the widest BODY row, not the header. A gold CSV whose body carries more
    # fields than its header would otherwise have its trailing gold columns silently dropped,
    # and a prediction missing them would still be scored correct.
    width = max([len(rows[0])] + [len(row) for row in body]) if body else len(rows[0])
    return [[row[i] if i < len(row) else None for row in body] for i in range(width)]


def connect(engine: str, args):
    """(connection, cursor) for the warehouse the run was produced on."""
    if engine == "snowflake":
        import snowflake.connector

        con = snowflake.connector.connect(connection_name=args.connection)
        cur = con.cursor()
        cur.execute(f"USE WAREHOUSE {args.warehouse}")
        return con, cur

    workspace = databricks_workspace(host=args.host, profile=args.profile)
    con, _ = databricks_sql_connection(workspace, args.warehouse_id)
    return con, con.cursor()


def run_sql(cur, database: str, schema: str, sql: str, engine: str) -> list[list]:
    # Unity Catalog rejects a dotted namespace -- "[UC_INVALID_NAMESPACE] Nested or empty
    # namespaces are not supported" -- so Databricks needs two statements where Snowflake
    # takes one. Getting this wrong fails every candidate execution while the gold still
    # resolves, which reads as a vendor scoring zero rather than a broken grader.
    if engine == "snowflake":
        cur.execute(f'USE SCHEMA {database}."{schema}"')
    else:
        cur.execute(f"USE CATALOG {database}")
        cur.execute(f"USE SCHEMA {schema}")
    cur.execute(sql)
    rows = cur.fetchall()
    width = len(cur.description)
    return [[row[i] for row in rows] for i in range(width)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="eval-reports/spider2-results.jsonl")
    p.add_argument("--suite", default=_DEFAULT_SUITE)
    p.add_argument("--database", default="SPIDER2")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--out", default="eval-reports/spider2-graded.jsonl")
    p.add_argument(
        "--engine",
        default="snowflake",
        choices=["snowflake", "databricks"],
        help="which warehouse the predictions were generated against and must re-execute on",
    )
    p.add_argument("--catalog", default=None, help="databricks: catalog holding the schemas")
    p.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))
    p.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    p.add_argument("--report", default=None, help="alias for --out")
    args = p.parse_args()
    if args.report:
        args.out = args.report
    if args.engine == "databricks":
        args.database = args.catalog or "bench_spider2"

    suite = os.path.expanduser(args.suite)
    meta = load_eval_meta(os.path.join(suite, "gold", "spider2lite_eval.jsonl"))
    exec_dir = os.path.join(suite, "gold", "exec_result")

    results = load_results(args.results)
    con, cur = connect(args.engine, args)

    buckets: Counter[str] = Counter()
    per_db: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict] = []
    try:
        for index, (case_id, result) in enumerate(sorted(results.items()), 1):
            schema = result.db_id.upper() if args.engine == "snowflake" else schema_for(result.db_id)
            golds = gold_paths(exec_dir, case_id)
            info = meta.get(case_id, {})

            bucket, detail = "wrong", ""
            if not golds:
                bucket = "no_gold"
            elif result.deferral:
                bucket = "deferred"
            elif not result.sql:
                bucket = "error"
                detail = result.error or "no sql"
            else:
                try:
                    predicted = run_sql(cur, args.database, schema, result.sql, args.engine)
                except Exception as exc:
                    bucket, detail = "error", str(exc).splitlines()[0][:200]
                else:
                    # condition_cols is either one list of column indices for every
                    # accepted gold, or one list per gold in the same order as the files.
                    condition = info.get("condition_cols") or []
                    per_gold = bool(condition) and isinstance(condition[0], list)
                    for position, path in enumerate(golds):
                        columns = (
                            (condition[position] if position < len(condition) else [])
                            if per_gold
                            else condition
                        )
                        gold = read_gold(path)
                        if compare_tables(
                            predicted,
                            gold,
                            condition_cols=columns or None,
                            ignore_order=bool(info.get("ignore_order", True)),
                        ):
                            # Spider 2.0 defines only containment, which is got-the-facts.
                            # The strict reading is mnemiq's: the same match AND the
                            # prediction returning the gold's own column count rather than a
                            # wider SELECT -- the allow_extra_columns distinction
                            # results_match draws on BIRD, so exact/facts mean the same thing
                            # on every benchmark. Not comparable to a published Spider 2.0
                            # score, which only ever quotes the tolerant rule.
                            if len(predicted) == len(gold):
                                bucket = "correct"
                                detail = f"{os.path.basename(path)} · gold shape"
                            else:
                                bucket = "correct_facts"
                                detail = (
                                    f"{os.path.basename(path)} · {len(predicted)} cols "
                                    f"vs gold {len(gold)}"
                                )
                            break

            buckets[bucket] += 1
            if bucket != "no_gold":
                per_db[result.db_id][bucket] += 1
                per_db[result.db_id]["scored"] += 1
            rows.append(
                {"case_id": case_id, "db_id": result.db_id, "bucket": bucket, "detail": detail}
            )
            if index % 25 == 0:
                print(f"  graded {index}/{len(results)}", flush=True)
    finally:
        cur.close()
        con.close()

    scored = sum(v for k, v in buckets.items() if k != "no_gold")
    facts_n = buckets["correct"] + buckets["correct_facts"]
    print(f"\n{'bucket':14} {'n':>5}  % of scored")
    for name in ("correct", "correct_facts", "wrong", "deferred", "error"):
        share = f"{buckets[name] / scored:6.1%}" if scored else "     -"
        print(f"{name:14} {buckets[name]:5} {share}")
    if buckets["no_gold"]:
        print(f"{'no_gold':14} {buckets['no_gold']:5}     excluded")

    if scored:
        print(
            f"\nexact-match    {buckets['correct'] / scored:5.1%}  "
            f"({buckets['correct']}/{scored})  -- mnemiq's strict rule: gold's facts AND shape"
        )
        print(
            f"got-the-facts  {facts_n / scored:5.1%}  "
            f"({facts_n}/{scored})  -- Spider 2.0's own metric, column containment"
        )
        print("(Spider 2.0 defines only the second; the first is mnemiq's own and does not")
        print(" compare to a published Spider 2.0 score)")

    print(f"\n{'database':30} {'scored':>7} {'exact':>8} {'facts':>8}")
    for db in sorted(per_db):
        counts = per_db[db]
        n = counts["scored"]
        db_facts = counts["correct"] + counts["correct_facts"]
        print(f"{db:30} {n:7} {counts['correct'] / n:8.1%} {db_facts / n:8.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nper-case detail -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

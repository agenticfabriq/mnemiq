"""Raw-LLM tier on Databricks: ask ai_query() for SQL, with no semantic layer.

The model gets the schema DDL and the question -- nothing else. This is the tier directly
comparable to mnemiq's proposer, and to Snowflake's SNOWFLAKE.CORTEX.COMPLETE tier.

  uv run python scripts/run_ai_query.py --db financial --limit 20
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mnemiq.eval.bird import load_bird  # noqa: E402
from mnemiq.eval.spider import load_spider  # noqa: E402
from mnemiq.eval.spider2_vendor import load_spider2_local  # noqa: E402
from mnemiq.eval.warehouse import (  # noqa: E402
    VendorResult,
    append_result,
    databricks_sql_connection,
    databricks_workspace,
    load_results,
    schema_for,
)

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)

_PROMPT = """You are a Databricks SQL expert. Write one SQL query that answers the question.

Schema of {catalog}.{schema}:
{ddl}

Question: {question}

Reply with only the SQL query, no explanation and no markdown fences. Reference tables as \
{catalog}.{schema}.table_name."""


def schema_ddl(cur, catalog: str, schema: str) -> str:
    cur.execute(
        f"""SELECT table_name, column_name, full_data_type
            FROM {catalog}.information_schema.columns
            WHERE table_schema = '{schema}'
            ORDER BY table_name, ordinal_position"""
    )
    tables: dict[str, list[str]] = {}
    for table, column, dtype in cur.fetchall():
        tables.setdefault(table, []).append(f"  {column} {dtype}")
    lines = [f"CREATE TABLE {t} (\n" + ",\n".join(cols) + "\n);" for t, cols in tables.items()]

    # Every column needs a distinct alias. Without them the result has table_name and
    # column_name twice, Arrow refuses to build a schema with duplicate field names, and the
    # whole DDL comes back empty -- so the model is asked for SQL against no schema at all,
    # which looks like a weak model rather than a broken prompt.
    cur.execute(
        f"""SELECT kcu.table_name  AS fk_table,
                   kcu.column_name AS fk_column,
                   ccu.table_name  AS pk_table,
                   ccu.column_name AS pk_column
            FROM {catalog}.information_schema.referential_constraints rc
            JOIN {catalog}.information_schema.key_column_usage kcu
              ON rc.constraint_name = kcu.constraint_name
            JOIN {catalog}.information_schema.constraint_column_usage ccu
              ON rc.unique_constraint_name = ccu.constraint_name
            WHERE kcu.table_schema = '{schema}'"""
    )
    for fk_table, fk_column, pk_table, pk_column in cur.fetchall():
        lines.append(f"-- {fk_table}.{fk_column} references {pk_table}.{pk_column}")
    return "\n".join(lines)


def extract_sql(text: str) -> str | None:
    if not text:
        return None
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    body = (fenced.group(1) if fenced else text).strip()
    if not re.search(r"\bselect\b|\bwith\b", body, re.I):
        return None
    return body.rstrip().rstrip(";")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--catalog", default="bench")
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))
    p.add_argument(
        "--host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="workspace URL; triggers browser OAuth instead of a stored token",
    )
    p.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    p.add_argument("--model", default="databricks-claude-sonnet-4-5")
    p.add_argument(
        "--benchmark",
        default="bird",
        choices=["bird", "spider", "spider2"],
        help="which question set to ask; spider reads --spider-dir and has no evidence hints, "
        "spider2 reads --spider2-jsonl and has no gold SQL at all",
    )
    p.add_argument(
        "--spider-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
        ),
    )
    p.add_argument("--split", default="dev", help="spider only")
    p.add_argument(
        "--spider2-jsonl",
        default=os.environ.get(
            "MNEMIQ_SPIDER2_JSONL",
            os.path.expanduser("~/src/dataset/spider2-lite/Spider2/spider2-lite/spider2-lite.jsonl"),
        ),
        help="spider2 only: the spider2-lite instance file (local* instances are used)",
    )
    p.add_argument(
        "--no-knowledge",
        action="store_true",
        help="spider2 only: drop the external_knowledge document 13 questions depend on. "
        "Default is to include it, which is Spider 2.0's own protocol.",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--no-evidence", action="store_true")
    p.add_argument("--results", default="eval-reports/ai-query-results.jsonl")
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="questions asked concurrently, each on its own warehouse connection "
        "(a cursor is not thread-safe)",
    )
    args = p.parse_args()

    if args.benchmark == "spider2":
        cases = load_spider2_local(
            os.path.expanduser(args.spider2_jsonl),
            limit=args.limit,
            db_ids=args.dbs,
            with_knowledge=not args.no_knowledge,
        )
    elif args.benchmark == "spider":
        cases = load_spider(
            os.path.expanduser(args.spider_dir),
            split=args.split,
            limit=args.limit,
            db_ids=args.dbs,
        )
    else:
        cases = load_bird(
            os.path.expanduser(args.minidev),
            limit=args.limit,
            db_ids=args.dbs,
            with_evidence=not args.no_evidence,
        )
    done = {} if args.refresh else load_results(args.results)
    remaining = [c for c in cases if c.id not in done]
    print(f"{len(cases)} cases, {len(done)} answered, {len(remaining)} to ask", flush=True)
    if not remaining:
        return 0

    workspace = databricks_workspace(host=args.host, profile=args.profile)
    connection, warehouse_id = databricks_sql_connection(workspace, args.warehouse_id)
    cur = connection.cursor()

    # A cursor is not thread-safe, so each worker gets its own connection. The main one above
    # stays for the DDL prefetch, which is shared and cheap to do once.
    local = threading.local()
    extra_connections: list = []
    conn_lock = threading.Lock()

    def worker_cursor():
        if getattr(local, "cur", None) is None:
            con, _ = databricks_sql_connection(workspace, warehouse_id)
            with conn_lock:
                extra_connections.append(con)
            local.cur = con.cursor()
        return local.cur

    # The DDL is per-schema and identical for every question in that schema; fetching it up
    # front keeps the workers from racing to build the same cache entry N times over.
    ddl_cache: dict[str, str] = {}
    for schema in sorted({schema_for(case.db_id) for case in remaining}):
        try:
            ddl_cache[schema] = schema_ddl(cur, args.catalog, schema)
        except Exception as exc:
            print(f"  no DDL for {schema}: {type(exc).__name__}: {exc}"[:200], flush=True)
            ddl_cache[schema] = ""

    counts = {"sql": 0, "deferred": 0, "error": 0}
    write_lock = threading.Lock()
    done_count = 0

    def ask_one(case) -> VendorResult:
        schema = schema_for(case.db_id)
        prompt = _PROMPT.format(
            catalog=args.catalog,
            schema=schema,
            ddl=ddl_cache.get(schema, ""),
            question=case.question,
        )
        result = VendorResult(
            case_id=case.id,
            db_id=case.db_id,
            question=case.question,
            gold_sql=case.gold_sql,
            difficulty=(case.tags or [""])[0],
        )
        started = time.monotonic()
        try:
            worker = worker_cursor() if args.workers > 1 else cur
            worker.execute("SELECT ai_query(?, ?)", [args.model, prompt])
            answer = worker.fetchone()[0]
            result.latency_ms = int((time.monotonic() - started) * 1000)
            result.sql = extract_sql(answer)
            if not result.sql:
                result.deferral = (answer or "")[:400]
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"[:400]
        return result

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(ask_one, case): case for case in remaining}
            for future in as_completed(futures):
                case = futures[future]
                result = future.result()
                with write_lock:
                    done_count += 1
                    counts[result.outcome] += 1
                    append_result(args.results, result)
                    print(
                        f"  [{done_count:4}/{len(remaining)}] {result.outcome:9} "
                        f"{case.id:16} {case.db_id}",
                        flush=True,
                    )
    finally:
        cur.close()
        connection.close()
        for con in extra_connections:
            try:
                con.close()
            except Exception:
                pass

    print(
        f"\nasked {len(remaining)}: {counts['sql']} sql, "
        f"{counts['deferred']} no-sql, {counts['error']} error"
    )
    print(f"results -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

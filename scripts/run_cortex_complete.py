"""Raw-LLM tier: ask SNOWFLAKE.CORTEX.COMPLETE for SQL, with no semantic layer at all.

This is the one-shot baseline. The model gets the schema DDL and the question -- nothing more,
no curated descriptions, no relationships, no verified queries. It isolates the model from the
product machinery around it, and is the tier directly comparable to mnemiq's proposer.

  uv run python scripts/run_cortex_complete.py --db financial --limit 20
  uv run python scripts/run_cortex_complete.py --model claude-4-sonnet
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import snowflake.connector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mnemiq.eval.bird import load_bird  # noqa: E402
from mnemiq.eval.warehouse import VendorResult, append_result, load_results  # noqa: E402

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)

_PROMPT = """You are a Snowflake SQL expert. Write one SQL query that answers the question.

Schema of database {database}.{schema}:
{ddl}

Question: {question}

Reply with only the SQL query, no explanation and no markdown fences. Reference tables as \
{database}.{schema}.TABLE_NAME."""


def schema_ddl(cur, database: str, schema: str) -> str:
    """CREATE TABLE-shaped text for every table in the schema, plus its foreign keys.

    The same structural facts the semantic-layer tier gets -- columns, types, keys -- so the
    two tiers differ by the product's machinery, not by what they were told about the data.
    """
    cur.execute(
        f"""SELECT table_name, column_name, data_type
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = '{schema}'
            ORDER BY table_name, ordinal_position"""
    )
    tables: dict[str, list[str]] = {}
    for table, column, dtype in cur.fetchall():
        tables.setdefault(table, []).append(f"  {column} {dtype}")

    lines = [f"CREATE TABLE {t} (\n" + ",\n".join(cols) + "\n);" for t, cols in tables.items()]

    cur.execute(f"SHOW IMPORTED KEYS IN SCHEMA {database}.{schema}")
    rows = cur.fetchall()
    if rows:
        columns = {desc[0].lower(): i for i, desc in enumerate(cur.description)}
        seen = set()
        for row in rows:
            fk = (
                row[columns["fk_table_name"]],
                row[columns["fk_column_name"]],
                row[columns["pk_table_name"]],
                row[columns["pk_column_name"]],
            )
            if fk in seen:
                continue
            seen.add(fk)
            lines.append(f"-- {fk[0]}.{fk[1]} references {fk[2]}.{fk[3]}")
    return "\n".join(lines)


def extract_sql(text: str) -> str | None:
    """The model was asked for bare SQL; strip fences anyway and take the statement."""
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
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--database", default="BENCH")
    p.add_argument("--warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    p.add_argument("--model", default="claude-4-sonnet")
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--no-evidence", action="store_true")
    p.add_argument("--results", default="eval-reports/cortex-complete-results.jsonl")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

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

    con = snowflake.connector.connect(connection_name=args.connection)
    cur = con.cursor()
    cur.execute(f"USE WAREHOUSE {args.warehouse}")

    ddl_cache: dict[str, str] = {}
    counts = {"sql": 0, "deferred": 0, "error": 0}
    try:
        for index, case in enumerate(remaining, 1):
            schema = case.db_id.upper()
            if schema not in ddl_cache:
                ddl_cache[schema] = schema_ddl(cur, args.database, schema)

            prompt = _PROMPT.format(
                database=args.database,
                schema=schema,
                ddl=ddl_cache[schema],
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
                cur.execute(
                    "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", (args.model, prompt)
                )
                answer = cur.fetchone()[0]
                result.latency_ms = int((time.monotonic() - started) * 1000)
                result.sql = extract_sql(answer)
                if not result.sql:
                    result.deferral = (answer or "")[:400]
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"[:400]

            counts[result.outcome] += 1
            append_result(args.results, result)
            print(
                f"  [{index:3}/{len(remaining)}] {result.outcome:9} {case.id:14} {case.db_id}",
                flush=True,
            )
    finally:
        cur.close()
        con.close()

    print(
        f"\nasked {len(remaining)}: {counts['sql']} sql, "
        f"{counts['deferred']} no-sql, {counts['error']} error"
    )
    print(f"results -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

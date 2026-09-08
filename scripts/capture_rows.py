"""Re-execute stored vendor SQL and capture rows + ordered columns for the beacon push.

Beacon grades pushed rows; the grading pass discarded them. This re-runs each stored
prediction on the warehouse it was produced against and writes per-case JSONL:
case_id, db_id, question, sql, columns (ordered, from cursor.description), rows
(list of dicts, capped), row_count, deferred, error, latency passthrough.
No model calls; execution only.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

from mnemiq.eval.warehouse import databricks_sql_connection, databricks_workspace, schema_for

CAP = 1000

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    p.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    p.add_argument("--engine", default="databricks", choices=["databricks", "snowflake"])
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--sf-warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE", "BENCH_WH"))
    args = p.parse_args()

    if args.engine == "snowflake":
        import snowflake.connector
        con = snowflake.connector.connect(connection_name=args.connection)
        cur = con.cursor()
        cur.execute(f"USE WAREHOUSE {args.sf_warehouse}")
    else:
        ws = databricks_workspace(host=args.host)
        con, _ = databricks_sql_connection(ws, args.warehouse_id)
        cur = con.cursor()

    done = set()
    out = Path(args.out)
    if out.exists():  # resume
        done = {json.loads(l)["case_id"] for l in out.open() if l.strip()}
    n = 0
    with out.open("a") as fh:
        for line in open(args.results):
            r = json.loads(line)
            cid = r["case_id"]
            if cid in done:
                continue
            rec = {"case_id": cid, "db_id": r.get("db_id"), "question": r.get("question"),
                   "sql": r.get("sql"), "latency_ms": r.get("latency_ms"),
                   "deferred": bool(r.get("deferral")), "error": r.get("error"),
                   "columns": None, "rows": None, "row_count": None}
            sql = (r.get("sql") or "").strip()
            if sql and not rec["deferred"] and not rec["error"]:
                try:
                    if args.engine == "snowflake":
                        cur.execute(f"USE SCHEMA {args.catalog}.{schema_for(r['db_id']).upper()}")
                    else:
                        cur.execute(f"USE CATALOG {args.catalog}")
                        cur.execute(f"USE SCHEMA {schema_for(r['db_id'])}")
                    t0 = time.time()
                    cur.execute(sql)
                    cols = [d[0] for d in (cur.description or [])]
                    raw = cur.fetchmany(CAP + 1)
                    rec["columns"] = cols
                    rec["rows"] = [dict(zip(cols, row)) for row in raw[:CAP]]
                    rec["row_count"] = len(raw)  # CAP+1 signals over-cap
                    rec["exec_ms"] = int((time.time() - t0) * 1000)
                except Exception as e:  # capture, don't die
                    rec["error"] = f"capture: {type(e).__name__}: {e}"[:400]
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"captured {n}", flush=True)
    print(f"DONE {args.out}: {n} new")

if __name__ == "__main__":
    main()

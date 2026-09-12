#!/usr/bin/env python3
"""Run a text-to-SQL SLM through ITS OWN native prompt, not mnemiq's.

This is a POSITIVE CONTROL, not a product measurement. Arctic-Text2SQL-R1 and
OmniSQL are trained on the OmniSQL prompt template; feeding them mnemiq's
prompt and reading a low score would confound "bad model" with "wrong prompt".
So this script reproduces the model's native input as faithfully as it can and
asks one question: does our harness reproduce roughly the score the model's
authors published? It reports EX over every case it ran, and additionally over a
subset if `<out>.excluded.json` exists beside the output.

KNOWN CONFOUND, not fixed: the prompt's stock sentence claims the schema block
carries primary keys, foreign keys and constraints. It does not -- the blocks are
columns and example values only. On a join-heavy benchmark a low reproduction is
therefore partly attributable to this, which is the confound the control exists
to rule out. The wording is verbatim from the reference prompt and changing it
would trade one fidelity gap for another.

If it does NOT, the gap is in the harness, the subset, or the grading -- and
every later number comparing this model to mnemiq is uninterpretable until
that is resolved. Nothing else should be run before this passes.

The prompt format is taken verbatim from OmniSQL's own worked example
(github.com/RUCKBReasoning/OmniSQL, examples/example_1.txt). Its schema block
carries per-column example VALUES -- the model's native input is a
value-grounded schema, which is the profiling step under another name.

Grading reuses `mnemiq.eval.grade.results_match(..., allow_extra_columns=False)`,
the BIRD execution-accuracy reading of the contract agreed with beacon. Output
is the JSONL shape `beacon/scripts/load_eval_reports.py` ingests, so beacon
re-grades every row independently and prints the disagreement table.

Usage:
    source .env                      # MNEMIQ_PG_DSN etc.; Settings reads os.environ only
    .venv/bin/python scripts/run_native_slm.py \\
        --base-url http://HOST:8000/v1 --model arctic --limit 20
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import decimal
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse as up
import urllib.request

import sqlite3

import psycopg
import pyarrow as pa

from mnemiq.eval.bird import load_bird
from mnemiq.eval.grade import normalize, results_match

ROWS_PREVIEW = 1000  # mnemiq caps result sets here; match it so both sides preview alike

MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/MINIDEV"))

# Verbatim from OmniSQL examples/example_1.txt. Do not "improve" the wording:
# the model was trained against these exact strings, and paraphrase is drift.
PROMPT = """Task Overview:
You are a data science expert. Below, you are provided with a database schema \
and a natural language question. Your task is to understand the schema and \
generate a valid SQL query to answer the question.

Database Engine:
{engine}

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, \
primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. \
If the question asks for a specific column, make sure to only include that \
column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the \
question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how \
to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


def bird_dsn(base: str, db: str = "bird_dev") -> str:
    return up.urlparse(base)._replace(path=f"/{db}").geturl()


def tables_by_db() -> dict[str, list[str]]:
    with open(os.path.join(MINIDEV, "dev_tables.json")) as fh:
        return {e["db_id"]: list(e["table_names_original"]) for e in json.load(fh)}


def build_schema(conn: psycopg.Connection, tables: list[str], samples: int = 2) -> str:
    """CREATE TABLE blocks with per-column example values, OmniSQL style.

    Values come from the live `bird_dev`, not from the SQLite originals, so the
    schema the model reads is the schema its SQL will actually run against.
    """
    # `dev_tables.json` keeps BIRD's original casing (`Patient`, `Player_Attributes`) but the
    # loader lowercased every table into `bird_dev`. An exact-name lookup therefore returned
    # nothing and SILENTLY DROPPED the table -- two databases went out with an entirely empty
    # schema, and the model invented plausible names (`patients`, `players`) that could not
    # execute. Resolve case-insensitively, and refuse to emit a schema that lost a table.
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        live = {r[0].lower(): r[0] for r in cur.fetchall()}
    resolved, dropped = [], []
    for t in tables:
        actual = live.get(t.lower())
        (resolved if actual else dropped).append(actual or t)
    if dropped:
        raise RuntimeError(
            f"schema would omit {len(dropped)} declared table(s): {dropped}. "
            "A partial schema is scored as a model error; fix the mapping instead.")

    blocks: list[str] = []
    for t in resolved:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position", (t,))
            cols = cur.fetchall()
        if not cols:
            raise RuntimeError(f"table {t!r} resolved but has no columns -- refusing to continue")
        lines = []
        for name, dtype in cols:
            ex = ""
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f'SELECT DISTINCT "{name}" FROM "{t}" '
                        f'WHERE "{name}" IS NOT NULL LIMIT {samples}')
                    vals = [r[0] for r in cur.fetchall()]
                if vals:
                    # Normalise BEFORE repr. Raw `repr(date(2012, 8, 26))` renders
                    # `datetime.date(2012, 8, 26)` into the schema block -- Python
                    # internals shown to a model that was trained on `'2012-08-26'`,
                    # in every date column of every prompt.
                    rendered = ", ".join(
                        repr(_cell(v) if not isinstance(v, str) else v[:40]) for v in vals)
                    ex = f" -- example: [{rendered}]"
            except Exception:
                conn.rollback()  # a column we cannot sample is not a reason to fail the case
            # Quote anything not a bare lowercase identifier: `Academic Year` and `County Code`
            # are real mini-dev column names, and a model told they are bare will emit
            # unquoted Postgres that cannot parse -- a prompt defect scored as a model error.
            ident = name if re.fullmatch(r"[a-z_][a-z0-9_]*", name) else f'"{name}"'
            lines.append((f"    {ident} {dtype}", ex))
        # Comma separates columns; the comment trails it, as in OmniSQL's example. The last
        # column takes no comma -- that example only gets away with one because PRIMARY KEY
        # and CONSTRAINT lines follow it, and these blocks carry neither.
        body = "\n".join(
            decl + ("," if i < len(lines) - 1 else "") + ex
            for i, (decl, ex) in enumerate(lines))
        blocks.append(f"CREATE TABLE {t} (\n{body}\n);")
    return "\n\n".join(blocks)


def _cell(v):
    """One result cell as a JSON-native value.

    `json.dumps(default=str)` looked harmless and silently stringified every
    `Decimal`, so beacon compared '109398.548387096774' against the NUMBER
    109398.548387096774 and failed 12 answers my own grader had passed. The
    disagreement was mine, not the graders'. Numbers stay numbers.
    """
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """Last fenced sql block wins -- these models reason first and answer last."""
    found = SQL_BLOCK.findall(text or "")
    if found:
        return found[-1].strip()
    # No fence: take the tail from the LAST SELECT/WITH. `re.search` returns the leftmost
    # match, which picks up a discarded draft ("SELECT count(*) ... but that is wrong.
    # Actually: SELECT name ...") and hands back something that cannot execute -- an
    # extraction defect then scored as a model error.
    starts = [m.start() for m in re.finditer(r"(?is)\b(?:select|with)\b", text or "")]
    return (text[starts[-1]:].strip() if starts else "")


class RequestRejected(Exception):
    """The server answered and refused. Not retryable, and not a transport fault."""


class TransportFailed(Exception):
    """The server could not be reached. NOT the same as a model that answered badly.

    Collapsing the two is how a dead SSH tunnel became "the model emitted no SQL" and
    produced a 15.20% that looked like a result. A transport failure must reach the
    summary as its own state and must void the run, not enter the denominator.
    """


def generate(base_url: str, model: str, prompt: str, timeout: float, max_tokens: int,
             attempts: int = 4, n: int = 1, temperature: float = 0.0) -> list[str]:
    """`n` completions in ONE request, so the prompt is prefilled once and only decode scales.

    Returns a list so the caller cannot silently read a single answer out of a vote.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
    }).encode()
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                choices = json.loads(r.read())["choices"]
                return [c["message"]["content"] for c in choices]
        except urllib.error.HTTPError as exc:
            # 4xx is the server understanding us and refusing -- a context-length 400 is a
            # prompt problem, and retrying it four times then reporting "fix the transport"
            # names the wrong fault.
            # NOT `body`: that name holds the encoded request payload, and assigning the
            # error text to it made attempt 2 rebuild the request with `data=<str>`, which
            # raises TypeError inside urlopen before any network call -- so the documented
            # backoff never reached the server on the 5xx path and one transient 503 voided
            # the run with a Python type error as its diagnosis.
            detail = exc.reason
            try:
                detail = exc.read().decode("utf-8", "replace")[:300] or exc.reason
            except Exception:  # noqa: BLE001 - a body we cannot read is not a second failure
                pass
            if 400 <= exc.code < 500:
                raise RequestRejected(f"HTTP {exc.code}: {detail}") from exc
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)  # the 5xx path must back off too, not fall through
        except Exception as exc:  # noqa: BLE001 - any transport fault is retryable here
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s: rides out a tunnel restart
    raise TransportFailed(str(last))


def sqlite_conn(db_id: str):
    """BIRD ships one .sqlite per database; there is no flat shared schema as in `bird_dev`."""
    path = os.path.join(MINIDEV, "dev_databases", db_id, f"{db_id}.sqlite")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no SQLite database for {db_id!r} at {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")  # BIRD has non-UTF8 bytes
    return con


def build_schema_sqlite(con, samples: int = 2) -> str:
    """The same value-grounded CREATE TABLE blocks, read from SQLite's own catalog."""
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    blocks = []
    for t in names:
        cols = [(r[1], r[2] or "text") for r in con.execute(f'PRAGMA table_info("{t}")')]
        if not cols:
            continue
        lines = []
        for name, dtype in cols:
            ex = ""
            try:
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL '
                    f'LIMIT {samples}')]
                if vals:
                    ex = " -- example: [" + ", ".join(
                        repr(_cell(v) if not isinstance(v, str) else v[:40]) for v in vals) + "]"
            except Exception:
                pass
            ident = name if re.fullmatch(r"[a-z_][a-z0-9_]*", name) else f'"{name}"'
            lines.append((f"    {ident} {dtype}", ex))
        body = "\n".join(decl + ("," if i < len(lines) - 1 else "") + ex
                          for i, (decl, ex) in enumerate(lines))
        blocks.append(f"CREATE TABLE {t} (\n{body}\n);")
    return "\n\n".join(blocks)


def run_sql_sqlite(con, sql: str, timeout_s: float = 30.0) -> pa.Table | None:
    """Execute with a hard wall-clock cap.

    A candidate with a missing join condition is a cartesian product, and `financial.trans`
    has about a million rows. Without a cap one such query span at 97% CPU for 1h39m and
    blocked the whole grading pass -- the run looked alive, wrote nothing, and forfeited
    every generation still held in memory. SQLite has no statement timeout, so the abort
    goes through the progress handler, which is the only interrupt point it offers.
    """
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        names = [d[0] for d in cur.description] if cur.description else []
        cols = list(zip(*rows)) if rows else [() for _ in names]  # by position; see run_sql
        return pa.Table.from_arrays([pa.array(list(c)) for c in cols], names=names)
    except Exception:
        return None
    finally:
        con.set_progress_handler(None, 0)


def _result_key(t: pa.Table | None):
    """A hashable canonical form of a result set, for voting.

    Candidates vote on WHAT THEY RETURNED, not on how they spelled it -- two different
    queries with the same rows are one vote, which is the whole point of execution-based
    self-consistency. Row order is discarded (sorted) because BIRD's own EX compares sets.
    `None` means the query did not run, and non-running candidates never win a vote.
    """
    if t is None or t.num_columns == 0:
        return None
    rows = [tuple(normalize(v) for v in r)
            for r in zip(*[c.to_pylist() for c in t.columns])]
    return tuple(sorted(rows, key=repr))


def vote(cands: list[tuple[str, pa.Table | None]]):
    """Pick the SQL whose result set the most candidates agree on.

    Ties and all-failed groups resolve to the FIRST candidate offering that result, so the
    function is deterministic given its input. Returns (sql, table, n_votes, n_groups).
    """
    groups: dict[object, list[tuple[str, pa.Table | None]]] = {}
    for sql, tbl in cands:
        key = _result_key(tbl)
        if key is None:
            continue                      # a query that did not run is not evidence
        groups.setdefault(key, []).append((sql, tbl))
    if not groups:
        return (cands[0][0] if cands else ""), None, 0, 0
    best = max(groups.values(), key=len)
    return best[0][0], best[0][1], len(best), len(groups)


def run_sql(conn: psycopg.Connection, sql: str, timeout_s: float = 30.0) -> pa.Table | None:
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
            cur.execute(sql)
            rows = cur.fetchall()
            names = [d.name for d in cur.description] if cur.description else []
        # By POSITION, not through a dict: `SELECT T2.name, T1.name, ...` is real gold in
        # this corpus (question 901), and a dict collapses the two into one column -- for
        # the gold AND the candidate, so a candidate missing a column grades correct. The
        # engine's own adapter avoids this the same way and says so.
        cols = list(zip(*rows)) if rows else [() for _ in names]
        return pa.Table.from_arrays([pa.array(list(c)) for c in cols], names=names)
    except Exception:
        conn.rollback()
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="vLLM OpenAI-compatible /v1 URL")
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--out", default="eval-reports/native-slm.jsonl")
    ap.add_argument("--engine", default="PostgreSQL", help="the Database Engine line")
    ap.add_argument("--backend", choices=["postgres", "sqlite"], default="postgres",
                    help="sqlite runs BIRD's own per-database files in its native dialect")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--db", action="append", dest="dbs")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--candidates", type=int, default=1,
                    help="samples per question; >1 enables execution-based majority voting")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default 0.0 single-shot, 0.8 when --candidates > 1")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel generation requests; execution stays sequential")
    args = ap.parse_args()
    if args.temperature is None:
        # Voting needs diverse samples; greedy would return the same answer N times and
        # report a unanimous vote that measured nothing. 0.8 is the paper's rollout value.
        args.temperature = 0.0 if args.candidates == 1 else 0.8

    pg = os.environ.get("MNEMIQ_PG_DSN")
    if not pg and args.backend == "postgres":
        print("MNEMIQ_PG_DSN unset -- `source .env` first", file=sys.stderr)
        return 2

    sqlite_mode = args.backend == "sqlite"
    if sqlite_mode and args.engine == "PostgreSQL":
        # The prompt must name the engine the SQL will actually run on; leaving the default
        # here would tell a SQLite-trained model to write Postgres against SQLite.
        args.engine = "SQLite"
    cases = load_bird(MINIDEV, dialect="sqlite" if sqlite_mode else "postgresql",
                      limit=args.limit, db_ids=args.dbs)
    per_db = tables_by_db()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    conns: dict[str, object] = {}
    if sqlite_mode:
        for case in cases:
            db = case.db_id or ""
            if db not in conns:
                conns[db] = sqlite_conn(db)
        conn = None
    else:
        conn = psycopg.connect(bird_dsn(pg))
    exec_sql = (lambda q, db: run_sql_sqlite(conns[db], q)) if sqlite_mode else \
               (lambda q, db: run_sql(conn, q))
    schema_cache: dict[str, str] = {}
    counts = {"correct": 0, "wrong": 0, "error": 0}
    n_empty_sql = 0
    n_transport = 0
    n_votes_for_winner = n_groups_total = 0

    # Schemas first: build_schema samples the database, so it shares the one connection
    # and must finish before any worker thread touches it.
    for case in cases:
        db = case.db_id or ""
        if db not in schema_cache:
            schema_cache[db] = (build_schema_sqlite(conns[db]) if sqlite_mode
                                else build_schema(conn, per_db.get(db, [])))

    # GENERATION is parallel (~15s each, and the server batches happily); EXECUTION and
    # GRADING stay sequential on the single connection. Splitting them this way keeps the
    # concurrency entirely inside HTTP calls that share no state -- a pool of threads all
    # issuing psycopg on one connection is a data race, not a speedup.
    def gen_one(case):
        prompt = PROMPT.format(
            engine=args.engine, schema=schema_cache[case.db_id or ""], question=case.question)
        t0 = time.time()
        try:
            raws = generate(args.base_url, args.model, prompt, args.timeout, args.max_tokens,
                            n=args.candidates, temperature=args.temperature)
        except RequestRejected as exc:
            # The server answered and refused. Not a transport fault, so it does not void
            # the run -- but it is not an answer either, so it grades as no SQL.
            print(f"  REQUEST REJECTED for {case.id}: {exc}", file=sys.stderr)
            return [""], (time.time() - t0) * 1000.0
        except TransportFailed as exc:
            print(f"  TRANSPORT FAILED for {case.id}: {exc}", file=sys.stderr)
            return None, (time.time() - t0) * 1000.0   # None != [] : unreached, not unanswered
        return [extract_sql(r) for r in raws], (time.time() - t0) * 1000.0

    print(f"generating {len(cases)} with {args.concurrency} workers ...", flush=True)
    t_gen = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        generated = list(pool.map(gen_one, cases))  # map preserves input order
    print(f"generation done in {time.time() - t_gen:.0f}s; executing and grading ...", flush=True)
    # Generation is the expensive half and lived only in memory until a grading stall threw
    # it away. Persist it first: a crash after this point costs minutes, not GPU hours.
    gen_path = args.out + ".generations.json"
    with open(gen_path, "w") as gh:
        json.dump([{"case_id": c.id, "sql": g[0], "ms": g[1]}
                   for c, g in zip(cases, generated)], gh)
    print(f"generations saved to {gen_path}", flush=True)

    with open(args.out, "w") as out:
        for i, (case, (sql, ms)) in enumerate(zip(cases, generated), 1):
            db = case.db_id or ""
            if sql is not None:
                # Execute EVERY candidate, then vote on results. With --candidates 1 this is
                # one execution and the vote is a no-op, so the single-shot path is unchanged.
                executed = [(q, exec_sql(q, db) if q else None) for q in sql]
                sql, cand_tbl, votes, groups = vote(executed)
                n_votes_for_winner += votes
                n_groups_total += groups
            if sql is None:
                n_transport += 1
                out.write(json.dumps({
                    "case_id": case.id, "outcome": "error", "sql": "",
                    "db_id": db, "ms": round(ms, 1), "error": "transport",
                }) + "\n")
                counts["error"] = counts.get("error", 0) + 1
                continue
            if not sql:
                n_empty_sql += 1

            cand = cand_tbl
            gold = exec_sql(case.gold_sql, db) if case.gold_sql else None

            if gold is None:
                outcome = "error"          # our gold did not run: not the model's fault
            elif cand is None:
                outcome = "wrong"          # model produced nothing runnable
            else:
                outcome = "correct" if results_match(
                    gold, cand, allow_extra_columns=False) else "wrong"
            counts[outcome] = counts.get(outcome, 0) + 1

            # beacon GRADES BY COMPARISON and never executes SQL, so a push that carries
            # only the query is refused. Ship the rows this run actually returned, bounded,
            # with the true count beside them.
            rows_preview = row_count = None
            if cand is not None:
                as_rows = [list(t) for t in zip(*[c.to_pylist() for c in cand.columns])] \
                    if cand.num_columns else []
                row_count = len(as_rows)
                rows_preview = [[_cell(v) for v in r] for r in as_rows[:ROWS_PREVIEW]]

            out.write(json.dumps({
                "case_id": case.id, "outcome": outcome, "sql": sql,
                "db_id": db, "ms": round(ms, 1),
                "engine_rows": rows_preview, "engine_row_count": row_count,
            }) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(cases):
                # Divide by what was GRADED. Dividing by `i` prints the very quantity the
                # void rule below suppresses, into a log that gets read back.
                ex = 100.0 * counts["correct"] / max(1, i - n_transport)
                print(f"  [{i}/{len(cases)}] EX={ex:.2f}%  {counts}", flush=True)

    n = len(cases)
    print(f"\nmodel={args.model} engine={args.engine} n={n}")
    if args.candidates > 1:
        graded = max(1, n - n_transport)
        print(f"voting: {args.candidates} samples @ T={args.temperature}  "
              f"mean winning votes {n_votes_for_winner / graded:.2f}/{args.candidates}  "
              f"mean distinct result groups {n_groups_total / graded:.2f}")
    print(f"correct={counts['correct']} wrong={counts['wrong']} "
          f"error={counts['error']}  empty-sql={n_empty_sql}  transport-failed={n_transport}")
    print(f"wrote {args.out}")

    # A run that could not reach the server for some of its questions has no score. Printing
    # one anyway is how 244 refused connections became "the model scored 15.20%".
    if n_transport:
        print(f"\nRUN VOID: {n_transport}/{n} questions never reached the model. "
              f"No accuracy is reported. Fix the transport and re-run.", file=sys.stderr)
        print("NATIVE_CONTROL_EXIT=1")
        return 1

    ex = 100.0 * counts["correct"] / n if n else 0.0
    print(f"EX={ex:.2f}% over all {n}")
    # Resolved BESIDE --out, not against the process CWD: a fixed relative path either
    # skipped silently or applied one run's exclusions to another's output.
    excluded_path = args.out + ".excluded.json"
    if os.path.exists(excluded_path):
        with open(excluded_path) as eh:
            excl = set(json.load(eh))
        kept = [c for c in cases if c.id not in excl]
        with open(args.out) as fh:
            rows = {json.loads(line)["case_id"]: json.loads(line) for line in fh}
        corr = sum(1 for c in kept if rows.get(c.id, {}).get("outcome") == "correct")
        print(f"EX={100.0 * corr / len(kept):.2f}% over the {len(kept)} mnemiq also graded "
              f"(gold <= 1000 rows) -- this is the like-for-like number")
    print("NATIVE_CONTROL_EXIT=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
        --base-url http://HOST:8000/v1 --model arctic --limit 20 \\
        --prompt-style arctic --max-tokens 8192

    `--prompt-style` defaults to `omnisql`, which is NOT arctic's own prompt: the model
    name and the prompt style are independent, and the two prompts do not score alike.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import decimal
import itertools
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

# Verbatim from their bird_eval/infer.py, and under the same rule as PROMPT above: these are
# the strings the model was RL-trained against, so a paraphrase measures a different prompt.
# Three things differ from the OmniSQL original we ran, and the paper (§4.2, Table 5) credits
# the combination with +2.6: the task text is a SYSTEM message, the output format is a
# think/answer envelope with stated budgets, and the assistant turn is PREFILLED so the model
# opens in reasoning mode rather than being asked to.
ARCTIC_SYSTEM = (
    "You are a data science expert. Below, you are provided with a database schema and a natural"
    " language question. Your task is to understand the schema and generate a valid SQL query to"
    " answer the question."
)

ARCTIC_INSTRUCT = """Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>"""

# Their infer.py hardcodes SQLite here; `{engine}` is a deliberate divergence so the same
# style can be run against Postgres, and the run header records which was used.
ARCTIC_USER = """Database Engine:
{engine}

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
{instruct}"""

# They append this AFTER apply_chat_template(add_generation_prompt=True), which opens the
# assistant turn mid-sentence. Over the OpenAI chat API the equivalent is a trailing assistant
# message with `continue_final_message`; `add_generation_prompt` must be False or the server
# closes the turn we are trying to continue. Verified against the live server: the completion
# resumes inside the thought and emits the closing `</think>`.
ARCTIC_PREFILL = "Let me solve this step by step. \n<think>"


def all_truncated(reasons: list[str]) -> bool:
    """Did EVERY candidate stop at the token cap?

    One definition on purpose. This lived twice -- once in the progress meter, once in the
    grading loop -- and a mutation of either copy left the other correct, so the two
    counters it feeds could disagree and only one was pinned by a test.
    """
    return bool(reasons) and all(r == "length" for r in reasons)


def usable_candidates(sqls: list[str], reasons: list[str]) -> list[str]:
    """The candidates that may be executed, voted with and graded.

    A generation stopped at the token cap is an UNFINISHED answer, and `extract_sql`'s
    last-SELECT fallback pulls a runnable query out of a cut-off chain of thought often
    enough to matter: 30 of 88 truncations graded CORRECT on the 2026-09-13 envelope run,
    where the flag was metadata and the text was executed like any other. An unfinished
    answer scoring as a right one is worse than it scoring as a wrong one.

    Returning `[""]` rather than `[]` when nothing finished is deliberate: `[]` would make
    `vote` see no candidates and the caller cannot distinguish that from a transport fault,
    which is the value `None` already carries. `[""]` grades as no SQL, which is what it is.
    """
    if not reasons:                       # nothing generated, or a caller with no reasons
        return list(sqls)
    keep = [q for q, why in zip(sqls, reasons, strict=False) if why != "length"]
    return keep if keep else [""]


def build_messages(style: str, engine: str, schema: str, question: str):
    """The request's messages and any transport-level extras. Pure, so it is testable."""
    if style == "omnisql":
        return [{"role": "user", "content": PROMPT.format(
            engine=engine, schema=schema, question=question)}], {}
    if style == "arctic":
        return [
            {"role": "system", "content": ARCTIC_SYSTEM},
            {"role": "user", "content": ARCTIC_USER.format(
                engine=engine, schema=schema, question=question,
                instruct=ARCTIC_INSTRUCT)},
            {"role": "assistant", "content": ARCTIC_PREFILL},
        ], {"continue_final_message": True, "add_generation_prompt": False}
    raise ValueError(f"unknown prompt style {style!r}")


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


def generate(base_url: str, model: str, messages: list, timeout: float, max_tokens: int,
             attempts: int = 4, n: int = 1, temperature: float = 0.0,
             extra: dict | None = None) -> tuple[list[str], list[str]]:
    """`n` completions in ONE request, so the prompt is prefilled once and only decode scales.

    Returns lists so the caller cannot silently read a single answer out of a vote, and
    the finish reasons alongside them: a generation stopped at the token cap is a
    TRUNCATION, not a wrong answer, and `extract_sql`'s last-SELECT fallback will happily
    turn one into a plausible query that grades as an error the model did not make.
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        **(extra or {}),
    }).encode()
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                choices = json.loads(r.read())["choices"]
                return ([c["message"]["content"] for c in choices],
                        [c.get("finish_reason") or "" for c in choices])
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
            # 408 and 429 are the server saying "not now", not "no": they are transport
            # conditions wearing a 4xx, and a rate-limited run that treats them as a
            # refusal publishes a score for questions it never managed to ask.
            if exc.code in (408, 429):
                last = exc
                if attempt < attempts - 1:
                    time.sleep(2 ** attempt)
                continue
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
    ap.add_argument("--prompt-style", choices=["omnisql", "arctic"], default="omnisql",
                    help="omnisql: the original single-turn prompt. arctic: their bird_eval "
                         "prompt -- system role, think/answer envelope, prefilled <think>.")
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
    n_truncated_only = 0
    # Cases where a real comparison happened: gold ran AND the model returned a finished
    # candidate. Invalid SQL counts -- "the model wrote something that does not run" is a
    # measurement. An unfinished generation is not, and neither is a gold that failed.
    n_scored = 0
    n_transport = 0
    n_votes_for_winner = n_groups_total = 0

    # Schemas first: build_schema samples the database, so it shares the one connection
    # and must finish before any worker thread touches it.
    for case in cases:
        db = case.db_id or ""
        if db not in schema_cache:
            schema_cache[db] = (build_schema_sqlite(conns[db]) if sqlite_mode
                                else build_schema(conn, per_db.get(db, [])))

    # A generation cut off at the token cap is not a wrong answer, and nothing downstream
    # can tell the difference: `extract_sql` falls back to the last SELECT, so a truncated
    # chain of thought yields a plausible query. Measured before this mattered -- Arctic's
    # median completion is ~500 tokens against a 2048 cap -- but a longer-CoT model or a
    # wider schema would cross it silently, which is the whole defect class this file exists
    # to avoid. `list.append` is atomic under the GIL, so the worker threads can share it.
    truncated: list[str] = []

    # GENERATION is parallel (~15s each, and the server batches happily); EXECUTION and
    # GRADING stay sequential on the single connection. Splitting them this way keeps the
    # concurrency entirely inside HTTP calls that share no state -- a pool of threads all
    # issuing psycopg on one connection is a data race, not a speedup.
    def gen_one(case):
        messages, extra = build_messages(
            args.prompt_style, args.engine, schema_cache[case.db_id or ""], case.question)
        t0 = time.time()
        try:
            raws, reasons = generate(args.base_url, args.model, messages, args.timeout,
                                     args.max_tokens, n=args.candidates,
                                     temperature=args.temperature, extra=extra)
            for why in reasons:
                if why == "length":
                    truncated.append(case.id)
        except RequestRejected as exc:
            # The server answered and REFUSED. It is not a transport fault -- the network
            # worked -- but the question still went unanswered, so it voids the run under
            # the same rule, and is named apart in the summary because the fix differs.
            print(f"  REQUEST REJECTED for {case.id}: {exc}", file=sys.stderr)
            rejected.append(case.id)
            # NOT `[""]`: that is indistinguishable from a model that answered nothing, and
            # it graded as a wrong answer while `n_transport` stayed 0, so a run whose every
            # request was refused printed a real-looking EX of 0%. The question was never
            # asked; `None` is the value that already means unreached.
            return None, (time.time() - t0) * 1000.0, [], []
        except TransportFailed as exc:
            print(f"  TRANSPORT FAILED for {case.id}: {exc}", file=sys.stderr)
            # None != [] : unreached, not unanswered
            return None, (time.time() - t0) * 1000.0, [], []
        extracted = [extract_sql(r) for r in raws]
        return usable_candidates(extracted, reasons), (time.time() - t0) * 1000.0, reasons, raws

    print(f"generating {len(cases)} with {args.concurrency} workers ...", flush=True)
    t_gen = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Generation is 90% of the wall clock and a HEALTHY run used to print nothing while
        # it ran, so a working run and a hung one looked identical for an hour -- observed:
        # the operator asked whether it was going at all. (Per-case failures already went to
        # stderr; what was missing was any sign of progress.)
        #
        # The count has to separate attempts from ANSWERS. Counting in `finally` alone counts
        # attempts, and every request 400ing finishes generation in seconds and prints
        # `generated 500/500 in 3s (~0 min left)` -- the most reassuring line this meter can
        # emit, for a run that produced no SQL whatsoever. Both error paths in `gen_one`
        # return no finish reasons, which is the signal used here.
        #
        # `itertools.count` rather than `n += 1`: the increment happens on worker threads, and
        # read-modify-write on an int can lose one, which would make every later line
        # under-report. `list.append` is atomic for the same reason, as with `truncated`.
        counter = itertools.count(1)
        rejected: list[str] = []    # the server understood and refused (4xx, not 408/429)
        trunc_only: list[str] = []  # generated, but every candidate hit the token cap
        unreached: list[str] = []   # never generated: transport fault or a refused request
        no_sql: list[str] = []      # generated, but nothing extractable came out

        def gen_one_counted(case):
            result = None
            try:
                result = gen_one(case)
                return result
            finally:
                # The counter FIRST, so the line can never report more failures than
                # completions -- with 8 workers refused at once it read "1/500, 5 FAILED".
                i = next(counter)
                if result is None or not result[2]:
                    unreached.append(case.id)
                elif all_truncated(result[2]):
                    # Its own bucket, not `no_sql`: same symptom, different remedy. It must
                    # still reach the meter -- taking it out of `no_sql` without adding it
                    # here left an all-truncating run printing a clean `generated 30/30`,
                    # which is the silence two earlier commits were written to remove.
                    trunc_only.append(case.id)
                elif not any((q or "").strip() for q in (result[0] or [])):
                    # A 200 carrying no fence and no SELECT is the OTHER way this meter can
                    # reassure about a run that produced nothing: finish reasons are present,
                    # so the unreached check above passes it. Keying on the extracted SQL
                    # already in hand is what distinguishes the two. Kept apart from
                    # truncation above because the operator's next move differs: extraction
                    # or the prompt here, `--max-tokens` there.
                    no_sql.append(case.id)
                # i == 1 as well as every 25th: the first line must not wait for 25
                # completions, because per-case latency is exactly what goes pathological
                # when something is wrong. At the default 180s timeout and 4 attempts, 25
                # completions can be 45 minutes away -- the same silence this is fixing.
                if i == 1 or i % 25 == 0 or i == len(cases):
                    el = time.time() - t_gen
                    bad = "".join([f", {len(unreached)} UNREACHED" if unreached else "",
                                   f", {len(trunc_only)} TRUNCATED" if trunc_only else "",
                                   f", {len(no_sql)} NO-SQL" if no_sql else ""])
                    if i < args.concurrency:
                        # No ETA yet: `i / elapsed` divides by wall time during which
                        # `--concurrency` cases ran in PARALLEL, so it understates the rate
                        # by about the worker count -- at concurrency 8 the first line
                        # projected ~125 min for a run that finished in ~16.
                        print(f"  generated {i}/{len(cases)}{bad} in {el:.0f}s "
                              f"(rate not yet meaningful: {args.concurrency} in flight)",
                              flush=True)
                    else:
                        rate = i / el if el else 0
                        left = (len(cases) - i) / rate / 60 if rate else 0
                        print(f"  generated {i}/{len(cases)}{bad} in {el:.0f}s "
                              f"(~{left:.0f} min left)", flush=True)

        generated = list(pool.map(gen_one_counted, cases))  # map preserves input order
    print(f"generation done in {time.time() - t_gen:.0f}s; executing and grading ...", flush=True)
    # Generation is the expensive half and lived only in memory until a grading stall threw
    # it away. Persist it first: a crash after this point costs minutes, not GPU hours.
    gen_path = args.out + ".generations.json"
    with open(gen_path, "w") as gh:
        json.dump([{"case_id": c.id, "sql": g[0], "ms": g[1],
                    "finish_reasons": g[2], "raw": g[3]}
                   for c, g in zip(cases, generated)], gh)
    print(f"generations saved to {gen_path}", flush=True)

    with open(args.out, "w") as out:
        for i, (case, (sql, ms, reasons, _raw)) in enumerate(zip(cases, generated), 1):
            was_truncated = any(r == "length" for r in reasons)

            db = case.db_id or ""
            if sql is not None:
                # Drop the candidates that never finished. `truncated` used to be metadata
                # only: the unfinished text was executed and graded like any other, and
                # `extract_sql`'s last-SELECT fallback pulls a runnable query out of a
                # cut-off chain of thought often enough that 30 of 88 truncations graded
                # CORRECT on the 2026-09-13 envelope run. An unfinished answer scoring as a
                # right one is worse than it scoring as a wrong one, and it also made the
                # printed "each is scored wrong" bound false in both directions.
                # Already filtered by `gen_one`: unfinished candidates never arrive here.
                # Execute EVERY candidate, then vote on results. With
                # --candidates 1 this is one execution and the vote is a no-op.
                executed = [(q, exec_sql(q, db) if q else None) for q in sql]
                sql, cand_tbl, votes, groups = vote(executed)
                n_votes_for_winner += votes
                n_groups_total += groups
            if sql is None:
                n_transport += 1
                out.write(json.dumps({
                    "case_id": case.id, "outcome": "error", "sql": "",
                    "db_id": db, "ms": round(ms, 1),
                    # For whoever reads this JSONL directly. (NOT for beacon: its
                    # `load_eval_reports.py` declares no `error` field and never reads one,
                    # so the distinction does not survive into beacon's item rows.)
                    "error": "rejected" if case.id in rejected else "transport",
                    "prompt_style": args.prompt_style, "truncated": was_truncated,
                }) + "\n")
                counts["error"] = counts.get("error", 0) + 1
                continue
            if not sql:
                # A fully-truncated case arrives as [""] and would otherwise be reported as
                # `empty-sql`, sending the operator to look at extraction when the fix is
                # `--max-tokens`. Same value, two causes, different remedies.
                if all_truncated(reasons):
                    n_truncated_only += 1
                else:
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
            if gold is not None and sql:
                n_scored += 1

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
                "prompt_style": args.prompt_style, "truncated": was_truncated,
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
    print(f"\nmodel={args.model} engine={args.engine} n={n} "
          f"prompt_style={args.prompt_style} max_tokens={args.max_tokens}")
    if truncated:
        print(f"TRUNCATED at the token cap: {len(truncated)} generation(s). Unfinished "
              f"candidates are DISCARDED before execution, so they cannot vote and cannot "
              f"grade correct; a case with no finished candidate grades `wrong`. Raise "
              f"--max-tokens and re-run to remove the doubt. First few: {truncated[:5]}")
    if args.candidates > 1:
        graded = max(1, n - n_transport)
        print(f"voting: {args.candidates} samples @ T={args.temperature}  "
              f"mean winning votes {n_votes_for_winner / graded:.2f}/{args.candidates}  "
              f"mean distinct result groups {n_groups_total / graded:.2f}")
    print(f"correct={counts['correct']} wrong={counts['wrong']} "
          f"error={counts['error']}  empty-sql={n_empty_sql}  "
          f"no-finished-candidate={n_truncated_only}  unanswered={n_transport}"
          + (f" (of which {len(rejected)} refused 4xx)" if rejected else ""))
    print(f"wrote {args.out}")

    # A run that could not reach the server for some of its questions has no score. Printing
    # one anyway is how 244 refused connections became "the model scored 15.20%".
    if n_transport:
        # Rejections are counted here too: both mean the question was never answered, which
        # is the condition this rule exists for. They are named apart because the fix
        # differs -- a transport fault is the network, a 4xx is the request.
        why = f"{n_transport}/{n} questions went UNANSWERED"
        if rejected:
            why += (f" ({len(rejected)} of them REFUSED by the server with a 4xx -- check "
                    f"context length and the model name, e.g. {rejected[:3]})")
        print(f"\nRUN VOID: {why}. No accuracy is reported. Fix it and re-run.",
              file=sys.stderr)
        print("NATIVE_CONTROL_EXIT=1")
        return 1

    # A run in which NO question produced a gradeable answer has no accuracy either, and
    # the previous rule only caught the unanswered case: 30 of 30 generations cut off at the
    # token cap printed `EX=0.00%` and exited 0. An all-truncated, all-gold-failed or
    # all-empty run measures its own configuration, not the model.
    if n_scored == 0:
        print(f"\nRUN VOID: no question produced a finished answer to grade "
              f"(no-finished-candidate={n_truncated_only}, empty-sql={n_empty_sql}, "
              f"error={counts['error']}). `wrong={counts['wrong']}` is not a score: those "
              f"cases returned no SQL, so nothing was compared. No FINAL accuracy is "
              f"reported; the per-batch `EX=` lines above are provisional and this run has "
              f"no result. Do not scrape them.", file=sys.stderr)
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

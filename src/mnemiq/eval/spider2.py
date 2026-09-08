"""Spider 2.0-lite (local slice) as EvaluationCases, graded against the gold result CSVs.

Why this module exists beside the BIRD path instead of inside it:

- Gold is a set of RESULT TABLES, not a query. Only 24/135 local cases ship gold SQL;
  the official grader (evaluation_suite/evaluate_utils.py) compares result CSVs, and
  134/135 cases publish ALTERNATIVE acceptable results (up to 12 per case). So grading
  here never executes a gold query: it loads every alternative, and the case passes if
  any alternative matches. harness.run_case's single-gold contract cannot express that,
  and generalizing it mid-flight was not worth touching the shared harness -- this file
  mirrors run_case's terminal states exactly and adds the one thing it lacks.
- Enrichment reuses enrich_bird_db UNCHANGED, through a symlinked BIRD-layout shim
  (dev_databases/<db>/<db>.sqlite -> databases/<db>.sqlite): same cache keys, same
  grounding chain, no duplicated pipeline.

Comparison semantics vs the official evaluate_utils: theirs passes if every gold column
appears among predicted columns by value (tolerance 1e-2); ours is results_match -- same
tolerance, multiset rows, candidate may add columns. Our CORRECT additionally requires
equal column count, so got-the-facts is the number comparable to their metric.

External knowledge: 13 local cases carry a hand-written reference document. It is
appended to the question as a hint by default -- the same convention as BIRD's evidence
field, and the same caveat: no deployment gets this.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.csv as pacsv

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase
from mnemiq.eval.bird_runner import (
    _append_result,
    _load_done,
    _load_meta,
    _save_meta,
    enrich_bird_db,
)
from mnemiq.eval.engine import build_engine
from mnemiq.eval.grade import results_match
from mnemiq.eval.harness import CaseResult, Engine, Outcome, _preview
from mnemiq.llm.client import ModelUnavailable


def _repo_dir(spider2_dir: str) -> str:
    return os.path.join(spider2_dir, "repo", "spider2-lite")


def spider2_db_path(spider2_dir: str, db_id: str) -> str:
    return os.path.join(spider2_dir, "databases", f"{db_id}.sqlite")


def load_spider2_local(
    spider2_dir: str,
    *,
    limit: int | None = None,
    db_ids: list[str] | None = None,
    with_knowledge: bool = True,
) -> list[EvaluationCase]:
    """The 135 local-SQLite cases. BigQuery/Snowflake instances need cloud adapters and
    credentials this engine does not have; they are filtered out, not failed."""
    repo = _repo_dir(spider2_dir)
    allowed = set(db_ids) if db_ids else None

    cases: list[EvaluationCase] = []
    with open(os.path.join(repo, "spider2-lite.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            iid = rec["instance_id"]
            if not iid.startswith("local"):
                continue
            if allowed is not None and rec["db"] not in allowed:
                continue

            question = rec["question"]
            doc = rec.get("external_knowledge")
            if with_knowledge and doc:
                doc_path = os.path.join(repo, "resource", "documents", doc)
                if os.path.isfile(doc_path):
                    with open(doc_path) as dfh:
                        question = f"{question}\n\nHint:\n{dfh.read().strip()}"

            # Gold SQL exists for a minority of cases and is documentation here, never
            # executed -- grading is against the result CSVs.
            gold_path = os.path.join(repo, "evaluation_suite", "gold", "sql", f"{iid}.sql")
            gold_sql = ""
            if os.path.isfile(gold_path):
                with open(gold_path) as gfh:
                    gold_sql = gfh.read().strip()

            cases.append(
                EvaluationCase(
                    id=iid,
                    question=question,
                    gold_sql=gold_sql,
                    db_id=rec["db"],
                    answerable=True,
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


# A dead endpoint answers every remaining case identically and instantly. Left alone, a
# run "finishes" as a full-length file of nothing -- which is worse than crashing, because
# it looks like data. Measured: one such run wrote 101 rows before anyone noticed.
MAX_CONSECUTIVE_OUTAGES = 5


def _is_outage(result: CaseResult) -> bool:
    """An engine-side outage, as distinct from a case the engine genuinely could not do."""
    return result.outcome is Outcome.ERROR and "provider did not respond" in (result.answer or "")


def gold_alternatives(spider2_dir: str, instance_id: str) -> list[pa.Table]:
    """Every published acceptable result for a case: `<id>.csv` plus `<id>_*.csv`."""
    er = os.path.join(_repo_dir(spider2_dir), "evaluation_suite", "gold", "exec_result")
    paths = sorted(
        set(glob.glob(os.path.join(er, f"{instance_id}.csv")))
        | set(glob.glob(os.path.join(er, f"{instance_id}_*.csv")))
    )
    tables = []
    for path in paths:
        try:
            tables.append(pacsv.read_csv(path))
        except Exception:  # noqa: BLE001 -- one unparseable alternative must not void the rest
            continue
    return tables


def grade_alternatives(candidate: pa.Table, alternatives: list[pa.Table]) -> Outcome:
    """Best outcome across the acceptable results: a case is CORRECT against the benchmark
    if it is correct against ANY answer the benchmark accepts."""
    if any(results_match(g, candidate, allow_extra_columns=False) for g in alternatives):
        return Outcome.CORRECT
    if any(results_match(g, candidate, allow_extra_columns=True) for g in alternatives):
        return Outcome.CORRECT_FACTS
    return Outcome.WRONG


def run_case_csv(
    case: EvaluationCase, engine: Engine, adapter, alternatives: list[pa.Table]
) -> CaseResult:
    """harness.run_case's flow, with the gold side replaced by pre-loaded result tables.

    Terminal states are kept identical on purpose: failed -> ERROR (an outage is not an
    abstention, M6), deferred -> DEFERRED_WRONGLY (every local case is answerable)."""
    result = CaseResult(
        case_id=case.id,
        outcome=Outcome.ERROR,
        question=case.question,
        gold_sql=case.gold_sql or "",
        db_id=case.db_id,
    )
    if alternatives:
        result.gold_rows = _preview(alternatives[0])
        result.gold_row_count = alternatives[0].num_rows

    started = time.perf_counter()
    try:
        answer = engine(case.question)
    except Exception as exc:  # noqa: BLE001 -- a crashed case is an ERROR row, not a dead run
        result.answer = str(exc)
        result.ms = (time.perf_counter() - started) * 1000
        return result
    result.ms = (time.perf_counter() - started) * 1000

    result.agreement = answer.agreement
    result.judge_engaged = answer.judge_engaged
    result.judge_fell_back = answer.judge_fell_back
    result.verify_confidence = answer.verify_confidence
    result.verify_layer = answer.verify_layer
    result.judge_override = answer.judge_override
    result.candidates_executed = answer.candidates_executed

    if answer.failed:
        result.outcome = Outcome.ERROR
        result.answer = answer.answer
        return result
    if answer.deferred:
        result.outcome = Outcome.DEFERRED_WRONGLY
        result.answer = answer.answer
        return result

    result.answer = answer.answer
    result.sql = answer.trace.target_sql if answer.trace else ""
    result.proposed = True
    result.approved = True

    try:
        if not alternatives:
            raise RuntimeError(f"no gold result published for {case.id}")
        candidate = adapter.execute_arrow(result.sql, timeout_s=30)
    except Exception as exc:  # noqa: BLE001
        result.outcome = Outcome.ERROR
        result.answer = str(exc)
        return result

    result.engine_rows = _preview(candidate)
    result.engine_row_count = candidate.num_rows
    result.executed = True
    # Portability is decided, not skipped. BIRD has two engines and must check by running
    # the SQL on the gold's; here the benchmark's own SQLite file IS the executor, so a
    # query that ran has already run on the gold engine and there is nothing to re-check.
    # Keyed on the adapter rather than hardcoded: run this through a DuckDB attachment and
    # the claim stops being free, and the flag must go back to being earned.
    result.portable_to_gold_engine = getattr(adapter, "dialect", None) == "sqlite"
    result.outcome = grade_alternatives(candidate, alternatives)
    return result


def _bird_layout_shim(spider2_dir: str, shim_root: str, db_ids: list[str]) -> str:
    """enrich_bird_db resolves dev_databases/<db>/<db>.sqlite; give it that layout as
    symlinks so the enrichment pipeline (and its cache keys) is reused byte-for-byte."""
    for db_id in db_ids:
        d = os.path.join(shim_root, "dev_databases", db_id)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, f"{db_id}.sqlite")
        target = spider2_db_path(spider2_dir, db_id)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"no database file for {db_id!r}: {target}")
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(target, link)
    return shim_root


def run_spider2(
    cases: list[EvaluationCase],
    spider2_dir: str,
    settings: Settings,
    *,
    cache_dir: str | None = None,
    results_path: str | None = None,
    workers: int = 1,
    candidates: int = 1,
    on_case: Callable[[int, int, CaseResult], None] | None = None,
    semantic: bool = True,
) -> tuple[list[CaseResult], dict]:
    """Run local Spider 2.0-lite cases grouped by database. Resumable exactly like
    run_bird: every result is checkpointed as it lands and a re-run skips what is done."""
    by_db: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_db.setdefault(case.db_id, []).append(case)

    shim_root = os.path.join(cache_dir or spider2_dir, "_bird-layout-shim")
    _bird_layout_shim(spider2_dir, shim_root, sorted(by_db))

    done = _load_done(results_path) if results_path else {}
    tokens, calls, excluded = _load_meta(results_path) if results_path else (0, 0, [])
    results: list[CaseResult] = list(done.values())
    processed = len(done)
    _outages = 0

    for db_id, db_cases in sorted(by_db.items()):
        remaining = [c for c in db_cases if c.id not in done]
        if not remaining:
            continue

        snapshot = enrich_bird_db(
            shim_root, db_id, settings, cache_dir=cache_dir, semantic=semantic
        )
        alternatives = {c.id: gold_alternatives(spider2_dir, c.id) for c in remaining}

        local = threading.local()
        clients: list = []
        clients_lock = threading.Lock()

        def _engine():
            if not hasattr(local, "e"):
                adapter = SQLiteAdapter(spider2_db_path(spider2_dir, db_id))
                ask, client = build_engine(snapshot, adapter, settings, candidates=candidates)
                local.e = (ask, adapter)
                with clients_lock:
                    clients.append(client)
            return local.e

        def _work(case: EvaluationCase) -> CaseResult:
            ask, adapter = _engine()
            return run_case_csv(case, ask, adapter, alternatives[case.id])

        if workers <= 1:
            out = [_work(c) for c in remaining]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                out = list(pool.map(_work, remaining))

        for result in out:
            processed += 1
            results.append(result)
            if results_path is not None:
                _append_result(results_path, result)
            if on_case is not None:
                on_case(processed, len(cases), result)
            _outages = _outages + 1 if _is_outage(result) else 0
            if _outages >= MAX_CONSECUTIVE_OUTAGES:
                raise ModelUnavailable(
                    f"{_outages} consecutive cases got no answer from the model provider; "
                    f"stopping after {processed} of {len(cases)}. The results file holds the "
                    "cases that did run -- delete the outage rows and re-run to resume."
                )

        for client in clients:
            tokens += client.total_tokens
            calls += client.calls
        if results_path is not None:
            _save_meta(results_path, tokens, calls, excluded)

    return results, {"tokens": tokens, "llm_calls": calls, "excluded": excluded}

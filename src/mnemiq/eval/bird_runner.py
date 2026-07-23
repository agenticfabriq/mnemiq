from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase, Snapshot
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.examples import LLMExampleGenerator, enrich_examples
from mnemiq.enrichment.facts import LLMFactsEnricher, enrich_table_facts
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.eval.bird import bird_db_path
from mnemiq.eval.engine import build_engine
from mnemiq.eval.harness import CaseResult, Outcome, run_case
from mnemiq.llm.client import LLMClient


def _append_result(path: str, result: CaseResult) -> None:
    """Checkpoint one result immediately -- a 500-question live run must survive a kill."""
    with open(path, "a") as fh:
        fh.write(json.dumps(asdict(result), default=str) + "\n")


def _load_done(path: str) -> dict[str, CaseResult]:
    """Reload checkpointed results so a resumed run skips what it already answered."""
    if not os.path.isfile(path):
        return {}
    done: dict[str, CaseResult] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["outcome"] = Outcome(d["outcome"])
            done[d["case_id"]] = CaseResult(**d)
    return done


def _meta_path(results_path: str) -> str:
    return results_path + ".meta.json"


def _load_meta(results_path: str) -> tuple[int, int, list[str]]:
    path = _meta_path(results_path)
    if not os.path.isfile(path):
        return 0, 0, []
    with open(path) as fh:
        m = json.load(fh)
    return m.get("tokens", 0), m.get("llm_calls", 0), m.get("excluded", [])


def _save_meta(results_path: str, tokens: int, calls: int, excluded: list[str]) -> None:
    with open(_meta_path(results_path), "w") as fh:
        json.dump({"tokens": tokens, "llm_calls": calls, "excluded": excluded}, fh)


# Facts/examples default OFF (settings.enrich_facts/enrich_examples): the plan-20 A/B measured both
# phases as regressions on a strong frontier model (facts -2.9, facts+examples -8.0 strict). Parked
# as opt-in plumbing; set MNEMIQ_ENRICH_FACTS/EXAMPLES=1 (a weaker/local model may need scaffolding).
def _enrich_cache_suffix(settings: Settings) -> str:
    parts = []
    if settings.enrich_facts:
        parts.append("facts")
    if settings.enrich_examples:
        parts.append("examples")
    if settings.dictionary_path:
        # a dictionary changes grounded meanings -> must not reuse a no-dict (or other-dict)
        # snapshot. Content edits to the same path still need --refresh (as grounding itself does).
        import hashlib
        parts.append("dict" + hashlib.sha256(settings.dictionary_path.encode()).hexdigest()[:6])
    if settings.ontology_records_path:
        # ontology bindings change grounded meanings AND which columns carry a scheme -> a
        # no-ontology snapshot must never be reused. Content edits still need --refresh.
        import hashlib

        parts.append(
            "onto" + hashlib.sha256(settings.ontology_records_path.encode()).hexdigest()[:6]
        )
    return f"__{'_'.join(parts)}" if parts else ""


def enrich_bird_db(
    minidev_dir: str,
    db_id: str,
    settings: Settings,
    cache_dir: str | None = None,
    refresh: bool = False,
    semantic: bool = True,
) -> Snapshot:
    """Enrich one BIRD database. Cached to disk: BIRD DBs never change, so (db_id, model)
    is the key -- the enriched snapshot depends on the model, so switching models must not
    silently reuse another model's enrichment. The facts/examples toggles enter the key too,
    so an A/B run never reuses another config's enrichment."""
    model_slug = (settings.llm_model or "default").replace("/", "_")
    cache_path = (
        os.path.join(cache_dir, f"{db_id}__{model_slug}{_enrich_cache_suffix(settings)}.json")
        if cache_dir else None
    )
    if cache_path and not refresh and os.path.isfile(cache_path):
        with open(cache_path) as fh:
            return Snapshot.model_validate_json(fh.read())

    from mnemiq.enrichment.dictionary import load_dictionary
    from mnemiq.enrichment.grounding import ground_codes
    from mnemiq.ontology.records import load_records

    adapter = SQLiteAdapter(bird_db_path(minidev_dir, db_id))
    snapshot = enrich_structural(adapter, db_id)
    _dict = load_dictionary(settings.dictionary_path) if settings.dictionary_path else None
    _onto = load_records(settings.ontology_records_path) if settings.ontology_records_path else None
    snapshot = ground_codes(adapter, snapshot, _dict, _onto)
    if semantic:
        snapshot = enrich_semantic(snapshot, LLMEnricher(LLMClient(settings)))
        if settings.enrich_facts:
            snapshot = enrich_table_facts(snapshot, LLMFactsEnricher(LLMClient(settings)))
        if settings.enrich_examples:
            snapshot = enrich_examples(
                snapshot, LLMExampleGenerator(LLMClient(settings)),
                adapter, dialect=adapter.dialect,
            )

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as fh:
            fh.write(snapshot.model_dump_json(by_alias=True))
    return snapshot


def _run_grouped(
    cases: list[EvaluationCase],
    build: Callable[[str], tuple[Callable, object]],
    gold_sql_sentinel: str | None = None,
) -> list[CaseResult]:
    """Route each case to its DB's engine+adapter (built once per DB). Pure: engine injected."""
    by_db: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_db.setdefault(case.db_id, []).append(case)

    results: list[CaseResult] = []
    for db_id, db_cases in by_db.items():
        ask, adapter = build(db_id)
        for case in db_cases:
            if gold_sql_sentinel is not None:  # test hook: fake adapter keys off a sentinel
                case = case.model_copy(update={"gold_sql": gold_sql_sentinel})
            results.append(run_case(case, ask, adapter))
    return results


def _process_db(cases, build_engine_fn, max_rows_cap: int, workers: int):
    """Run one database's cases, optionally across worker threads.

    Thread safety: each worker builds its OWN engine on first use (thread-local) -- its own
    DuckDB store connection, its own SQLite adapter, its own LLM client -- so no non-thread-safe
    connection is ever shared. The OpenAI HTTP client is concurrency-safe. Results are
    consumed single-threaded by the caller (ThreadPoolExecutor.map preserves order), so the
    checkpoint append needs no lock. Returns (list of (kind, payload), per-worker clients).
    """
    local = threading.local()
    clients: list = []
    clients_lock = threading.Lock()

    def engine():
        if not hasattr(local, "e"):
            ask, engine_adapter, gold_adapter, client = build_engine_fn()
            local.e = (ask, engine_adapter, gold_adapter)
            with clients_lock:
                clients.append(client)
        return local.e

    def work(case):
        ask, engine_adapter, gold_adapter = engine()
        if _gold_too_big(gold_adapter, case.gold_sql, max_rows_cap):
            return ("excluded", case.id)
        return ("result", run_case(case, ask, engine_adapter, gold_adapter))

    if workers <= 1:
        out = [work(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(work, cases))
    return out, clients


def _gold_too_big(adapter, gold_sql: str, cap: int) -> bool:
    probe = f"SELECT count(*) FROM (SELECT 1 FROM ({gold_sql}) LIMIT {cap + 1})"
    try:
        return adapter.execute(probe)[0][0] > cap
    except Exception:
        return False  # let run_case surface a real error rather than pre-judging


def run_bird(
    cases: list[EvaluationCase],
    minidev_dir: str,
    settings: Settings,
    *,
    cache_dir: str | None = None,
    max_rows_cap: int = 1000,
    on_case: Callable[[int, int, CaseResult], None] | None = None,
    results_path: str | None = None,
    workers: int = 1,
    candidates: int = 1,
) -> tuple[list[CaseResult], dict]:
    """Run BIRD cases grouped by database. Resumable: with results_path, each result is
    checkpointed as it completes and a re-run skips everything already answered -- a long
    live run survives a kill without re-paying for the questions it already got through."""
    by_db: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_db.setdefault(case.db_id, []).append(case)

    done_results = _load_done(results_path) if results_path else {}
    tokens, calls, excluded = _load_meta(results_path) if results_path else (0, 0, [])
    skip = set(done_results) | set(excluded)

    results: list[CaseResult] = list(done_results.values())
    processed = len(skip)

    for db_id, db_cases in by_db.items():
        remaining = [c for c in db_cases if c.id not in skip]
        if not remaining:
            continue  # whole DB already done in a prior segment -- no enrichment, no client

        snapshot = enrich_bird_db(minidev_dir, db_id, settings, cache_dir=cache_dir)

        def _build():  # each worker builds its own isolated engine (thread-safe connections)
            # BIRD grades single-engine on native SQLite: the engine generates + executes
            # SQLite and gold runs on the same engine, so a wrong answer is a real error,
            # never a cross-engine artifact (parity measured 1.5% otherwise). DuckDB is the
            # executor in the product path (DuckDBAdapter); the benchmark stays apples-to-apples.
            adapter = SQLiteAdapter(bird_db_path(minidev_dir, db_id))
            ask, client = build_engine(snapshot, adapter, settings, candidates=candidates)
            return ask, adapter, adapter, client  # engine + gold: same native SQLite executor

        out, clients = _process_db(remaining, _build, max_rows_cap, workers)

        # collection is single-threaded here -> checkpoint append needs no lock
        for kind, payload in out:
            processed += 1
            if kind == "excluded":
                excluded.append(payload)
                continue
            results.append(payload)
            if results_path is not None:
                _append_result(results_path, payload)
            if on_case is not None:
                on_case(processed, len(cases), payload)

        for client in clients:
            tokens += client.total_tokens
            calls += client.calls
        if results_path is not None:
            _save_meta(results_path, tokens, calls, excluded)

    return results, {"tokens": tokens, "llm_calls": calls, "excluded": excluded}

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase, Snapshot
from mnemiq.enrichment.enricher import LLMEnricher
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
    silently reuse another model's enrichment."""
    model_slug = (settings.llm_model or "default").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{db_id}__{model_slug}.json") if cache_dir else None
    if cache_path and not refresh and os.path.isfile(cache_path):
        with open(cache_path) as fh:
            return Snapshot.model_validate_json(fh.read())

    adapter = SQLiteAdapter(bird_db_path(minidev_dir, db_id))
    snapshot = enrich_structural(adapter, db_id)
    if semantic:
        snapshot = enrich_semantic(snapshot, LLMEnricher(LLMClient(settings)))

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
            results.append(run_case(case, ask, adapter, allow_extra_columns=False))
    return results


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
        adapter = SQLiteAdapter(bird_db_path(minidev_dir, db_id))
        ask, client = build_engine(snapshot, adapter, settings)

        for case in remaining:
            processed += 1
            if _gold_too_big(adapter, case.gold_sql, max_rows_cap):
                # the guard caps the candidate at max_rows_cap; a larger gold would force a
                # false WRONG. Exclude and log -- never silently mark it failed.
                excluded.append(case.id)
                continue
            result = run_case(case, ask, adapter, allow_extra_columns=False)
            results.append(result)
            if results_path is not None:
                _append_result(results_path, result)
            if on_case is not None:
                on_case(processed, len(cases), result)

        tokens += client.total_tokens
        calls += client.calls
        if results_path is not None:
            _save_meta(results_path, tokens, calls, excluded)

    return results, {"tokens": tokens, "llm_calls": calls, "excluded": excluded}

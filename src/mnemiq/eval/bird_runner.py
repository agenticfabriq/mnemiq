from __future__ import annotations

import os
from collections.abc import Callable

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase, Snapshot
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.eval.bird import bird_db_path
from mnemiq.eval.engine import build_engine
from mnemiq.eval.harness import CaseResult, run_case
from mnemiq.llm.client import LLMClient


def enrich_bird_db(
    minidev_dir: str,
    db_id: str,
    settings: Settings,
    cache_dir: str | None = None,
    refresh: bool = False,
    semantic: bool = True,
) -> Snapshot:
    """Enrich one BIRD database. Cached to disk: BIRD DBs never change, so db_id is the key."""
    cache_path = os.path.join(cache_dir, f"{db_id}.json") if cache_dir else None
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
) -> tuple[list[CaseResult], dict]:
    by_db: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_db.setdefault(case.db_id, []).append(case)

    results: list[CaseResult] = []
    excluded: list[str] = []
    tokens = calls = 0
    done = 0

    for db_id, db_cases in by_db.items():
        snapshot = enrich_bird_db(minidev_dir, db_id, settings, cache_dir=cache_dir)
        adapter = SQLiteAdapter(bird_db_path(minidev_dir, db_id))
        ask, client = build_engine(snapshot, adapter, settings)

        for case in db_cases:
            done += 1
            if _gold_too_big(adapter, case.gold_sql, max_rows_cap):
                # the guard caps the candidate at max_rows_cap; a larger gold would force a
                # false WRONG. Exclude and log -- never silently mark it failed.
                excluded.append(case.id)
                continue
            result = run_case(case, ask, adapter, allow_extra_columns=False)
            results.append(result)
            if on_case is not None:
                on_case(done, len(cases), result)

        tokens += client.total_tokens
        calls += client.calls

    return results, {"tokens": tokens, "llm_calls": calls, "excluded": excluded}

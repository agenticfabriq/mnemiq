"""Run BIRD mini-dev on the PostgreSQL dialect.

The engine generates DuckDB and executes it against the loaded `bird_dev` Postgres via
DuckDBAdapter (ATTACH); the gold PG SQL runs on native Postgres (PostgresAdapter). Per-db
scoping: `bird_dev` is one flat public schema holding all 11 DBs' tables (0 collisions), and
each question's engine is built over only its db_id's enriched snapshot, so the model still
sees a shrunk, per-database world -- not all 75 tables.
"""

from __future__ import annotations

from collections.abc import Callable

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.pg import PostgresAdapter
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase
from mnemiq.eval.bird_runner import (
    _append_result,
    _load_done,
    _load_meta,
    _process_db,
    _save_meta,
    enrich_bird_db,
)
from mnemiq.eval.engine import build_engine
from mnemiq.eval.harness import CaseResult


def run_minidev_pg(
    cases: list[EvaluationCase],
    minidev_dir: str,
    pg_dsn: str,
    settings: Settings,
    *,
    enrich_settings: Settings | None = None,
    cache_dir: str | None = None,
    max_rows_cap: int = 1000,
    on_case: Callable[[int, int, CaseResult], None] | None = None,
    results_path: str | None = None,
    workers: int = 1,
    candidates: int = 1,
) -> tuple[list[CaseResult], dict]:
    """Grouped-by-db, resumable mini-dev PG run. Enrichment is per-db (cached, from the SQLite
    dev_databases -- dialect-agnostic); execution is against `bird_dev` (pg_dsn) via DuckDB;
    gold runs on native Postgres.

    `enrich_settings` (default `settings`) lets enrichment run on a *different* model than
    generation -- e.g. hold the semantic layer constant on a hosted model while only generation
    varies across a local-model ladder. Embeddings follow `settings.embed_endpoint()`."""
    enrich_settings = enrich_settings or settings
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
            continue

        snapshot = enrich_bird_db(minidev_dir, db_id, enrich_settings, cache_dir=cache_dir)

        def _build(snapshot=snapshot):  # bind the current db's snapshot; own connections per worker
            engine_adapter = DuckDBAdapter.postgres(pg_dsn, read_only=True)  # engine SQL -> bird_dev
            ask, client = build_engine(snapshot, engine_adapter, settings, candidates=candidates)
            gold_adapter = PostgresAdapter(pg_dsn)  # gold PG SQL on native Postgres
            return ask, engine_adapter, gold_adapter, client

        out, clients = _process_db(remaining, _build, max_rows_cap, workers)

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

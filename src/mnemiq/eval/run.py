from __future__ import annotations

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.config import Settings
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.eval.engine import build_engine
from mnemiq.eval.golden import load_cases
from mnemiq.eval.harness import run_case
from mnemiq.eval.report import summarize
from mnemiq.llm.client import LLMClient


def run_acme(settings: Settings, golden: str = "evals/acme.json",
             gate: bool = False, record: bool = False) -> int:
    """Run the ACME golden set once (enrichment ON) and print the report.

    --record appends the run to the accuracy trend; --gate additionally fails (exit 1) when
    accuracy regressed beyond tolerance versus the last recorded run (the evaluation loop)."""
    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN")
        return 1
    from mnemiq.enrichment.dictionary import load_dictionary
    from mnemiq.enrichment.grounding import ground_codes

    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    _snap = enrich_structural(adapter, settings.source_id)
    _dict = load_dictionary(settings.dictionary_path) if settings.dictionary_path else None
    _snap = ground_codes(adapter, _snap, _dict)
    snapshot = enrich_semantic(_snap, LLMEnricher(LLMClient(settings)))
    ask, client = build_engine(snapshot, adapter, settings)
    results = [run_case(c, ask, adapter) for c in load_cases(golden)]
    report = summarize(results, tokens=client.total_tokens, llm_calls=client.calls)
    print(report.render())

    from mnemiq.eval.trend import check_regression, last_run, record_run

    previous = last_run(settings.control_dsn, settings.source_id, path="evals/trend.json")
    if record or gate:
        record_run(settings.control_dsn, settings.source_id, report, path="evals/trend.json")
    if gate:
        msg = check_regression(report, previous)
        if msg:
            print(f"GATE FAILED: {msg}")
            return 1
    return 0

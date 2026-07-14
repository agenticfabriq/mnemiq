"""Run the golden set against the live engine. With --ab, run it with and without enrichment."""

from __future__ import annotations

import sys
from pathlib import Path

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.config import Settings
from mnemiq.contract import Snapshot
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.eval.engine import build_engine
from mnemiq.eval.golden import load_cases
from mnemiq.eval.harness import run_case
from mnemiq.eval.report import Report, summarize
from mnemiq.llm.client import LLMClient
from mnemiq.semantic.glossary import load_definitions


def _run(snapshot: Snapshot, adapter, settings: Settings, label: str, definitions=()) -> Report:
    ask, client = build_engine(snapshot, adapter, settings, definitions=definitions)

    cases = load_cases("evals/acme.json")
    results = []
    for i, case in enumerate(cases, start=1):
        result = run_case(case, ask, adapter)
        results.append(result)
        print(f"  [{i:2}/{len(cases)}] {result.outcome:20} {case.id}", flush=True)

    report = summarize(results, tokens=client.total_tokens, llm_calls=client.calls)
    print(f"\n===== {label} =====")
    print(report.render())
    return report


def main() -> int:
    settings = Settings.from_env()
    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN", file=sys.stderr)
        return 1

    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    ab = "--ab" in sys.argv

    # The glossary applies to BOTH arms: the A/B isolates enrichment, nothing else.
    glossary_path = Path("glossary/acme.json")
    definitions = load_definitions(str(glossary_path)) if glossary_path.exists() else []

    print("enriching ACME...", flush=True)
    structural = enrich_structural(adapter, "acme")
    enriched = enrich_semantic(structural, LLMEnricher(LLMClient(settings)))

    print("\nrunning the golden set (enrichment ON)...", flush=True)
    on = _run(enriched, adapter, settings, "ENRICHMENT ON", definitions=definitions)

    if not ab:
        return 0

    # The thesis on trial: the same questions, the same model, the same decider -- the only
    # difference is whether the snapshot carries meaning.
    print("\nrunning the golden set (enrichment OFF)...", flush=True)
    off = _run(
        structural, adapter, settings, "ENRICHMENT OFF (structural only)", definitions=definitions
    )

    print("\n===== THE SEMANTIC LAYER, IN A NUMBER =====")
    print(f"  accuracy with enrichment:    {on.accuracy:.1%}")
    print(f"  accuracy without enrichment: {off.accuracy:.1%}")
    print(f"  wrong answers with:          {on.wrong}")
    print(f"  wrong answers without:       {off.wrong}")
    print(f"  tokens with:                 {on.tokens}")
    print(f"  tokens without:              {off.tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

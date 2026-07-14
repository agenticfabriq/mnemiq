"""Run the golden set against the live engine.

--ab            run twice: enrichment ON, then OFF (structural only)
--report DIR    also write per-case artifacts: <label>.json and <label>.html
                (question, gold SQL, engine SQL, both result sets, outcome)
"""

from __future__ import annotations

import argparse
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
from mnemiq.eval.artifacts import write_html, write_json
from mnemiq.eval.report import Report, summarize
from mnemiq.llm.client import LLMClient
from mnemiq.semantic.glossary import load_definitions


def _run(
    snapshot: Snapshot,
    adapter,
    settings: Settings,
    label: str,
    definitions=(),
    report_dir: Path | None = None,
) -> Report:
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

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        slug = label.lower().split("(")[0].strip().replace(" ", "-")
        write_json(report, str(report_dir / f"{slug}.json"), label=label)
        write_html(report, str(report_dir / f"{slug}.html"), label=label)
        print(f"\nreport written: {report_dir / f'{slug}.html'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab", action="store_true", help="run enrichment ON and OFF")
    parser.add_argument("--report", metavar="DIR", help="write per-case JSON + HTML artifacts")
    args = parser.parse_args()

    settings = Settings.from_env()
    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN", file=sys.stderr)
        return 1

    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    report_dir = Path(args.report) if args.report else None

    # The glossary applies to BOTH arms: the A/B isolates enrichment, nothing else.
    glossary_path = Path("glossary/acme.json")
    definitions = load_definitions(str(glossary_path)) if glossary_path.exists() else []

    print("enriching ACME...", flush=True)
    structural = enrich_structural(adapter, "acme")
    enriched = enrich_semantic(structural, LLMEnricher(LLMClient(settings)))

    print("\nrunning the golden set (enrichment ON)...", flush=True)
    on = _run(
        enriched, adapter, settings, "ENRICHMENT ON",
        definitions=definitions, report_dir=report_dir,
    )

    if not args.ab:
        return 0

    # The thesis on trial: the same questions, the same model, the same decider -- the only
    # difference is whether the snapshot carries meaning.
    print("\nrunning the golden set (enrichment OFF)...", flush=True)
    off = _run(
        structural, adapter, settings, "ENRICHMENT OFF (structural only)",
        definitions=definitions, report_dir=report_dir,
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

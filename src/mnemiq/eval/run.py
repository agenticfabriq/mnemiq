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


# One spelling, read and written from the same constant. Two string literals is how a reader
# and a writer come to disagree about where the baseline lives.
TREND_PATH = "evals/trend.json"


def run_acme(settings: Settings, golden: str = "evals/acme.json",
             gate: bool = False, record: bool = False) -> int:
    """Run the ACME golden set once (enrichment ON) and print the report.

    --record appends the run to the accuracy trend; --gate additionally fails (exit 1) when
    accuracy regressed beyond tolerance versus the last recorded run (the evaluation loop).

    --gate also fails when it could not compare at all -- no baseline, or a trend store it
    could not read. Both used to pass, so the gate could not fail and did not, while the
    number it was guarding drifted sixteen points (M34)."""
    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN")
        return 1
    from mnemiq.enrichment.certified import (
        apply_certified_set, fetch_certified_records, require_certified)
    from mnemiq.enrichment.dictionary import load_dictionary
    from mnemiq.enrichment.grounding import apply_dictionary, ground_codes
    from mnemiq.enrichment.pipeline import content_version
    from mnemiq.ontology.records import load_records

    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    _snap = enrich_structural(adapter, settings.source_id)
    _dict = load_dictionary(settings.dictionary_path) if settings.dictionary_path else None
    _onto = load_records(settings.ontology_records_path) if settings.ontology_records_path else None
    _certified_set = fetch_certified_records(settings)
    require_certified(_certified_set, settings)
    # precedence: ontology < correlated < lookup < certified < dictionary
    _snap = ground_codes(adapter, _snap, dictionary=None, ontology=_onto)
    _snap, _protected = apply_certified_set(_snap, _certified_set)
    _snap = enrich_semantic(_snap, LLMEnricher(LLMClient(settings)), protected=_protected)
    if _dict:
        _snap = apply_dictionary(_snap, _dict)
        _snap.version = content_version(_snap)
    snapshot = _snap
    ask, client = build_engine(snapshot, adapter, settings)
    results = [run_case(c, ask, adapter) for c in load_cases(golden)]
    report = summarize(results, tokens=client.total_tokens, llm_calls=client.calls)
    print(report.render())

    from mnemiq.eval.trend import (
        TrendUnavailable, gate_outcome, last_run, record_run, should_record,
    )

    store_error: str | None = None
    previous = None
    try:
        previous = last_run(settings.control_dsn, settings.source_id, path=TREND_PATH)
    except TrendUnavailable as exc:
        store_error = str(exc)
    # Compare BEFORE writing. `--gate` used to imply a write, and wrote first, so a regression
    # became the baseline it had just been rejected against -- one red build, then green
    # forever (M34).
    msg = gate_outcome(report, previous, store_error=store_error) if gate else None
    if should_record(record=record, gate=gate, gate_failed=msg is not None):
        record_run(settings.control_dsn, settings.source_id, report, path=TREND_PATH)
    if gate:
        # A gate that cannot compare FAILS. It used to pass, and passed for as long as no
        # baseline existed -- which was always, because nothing had ever written one.
        if msg:
            print(f"GATE FAILED: {msg}")
            return 1
        print("GATE PASSED: compared against the last recorded run")
    return 0

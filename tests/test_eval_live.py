import os

import pytest

from acme_dsn import acme_dsn, requires_acme

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.config import Settings
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.eval.engine import build_engine
from mnemiq.eval.golden import load_cases
from mnemiq.eval.harness import Outcome, run_case
from mnemiq.eval.report import summarize
from mnemiq.llm.client import LLMClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured"),
    requires_acme,
]

_DSN = acme_dsn()


@pytest.fixture(scope="module")
def acme_engine():
    """The whole engine, enriched and indexed once for the module."""
    settings = Settings.from_env()
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    snapshot = enrich_semantic(
        enrich_structural(adapter, "acme"), LLMEnricher(LLMClient(settings))
    )
    ask, client = build_engine(snapshot, adapter, settings)
    return ask, adapter, client


def test_the_engine_never_invents_an_answer_it_does_not_have(acme_engine):
    """The one result that would sink the project: confidently answering the unanswerable."""
    ask, adapter, _ = acme_engine
    unanswerable = [c for c in load_cases("evals/acme.json") if not c.answerable]

    results = [run_case(case, ask, adapter) for case in unanswerable]
    report = summarize(results)

    assert report.wrong == 0, [r.case_id for r in results if r.outcome is Outcome.WRONG]
    assert report.deferred_correctly == len(unanswerable)


def test_the_engine_answers_most_of_what_acme_can_answer(acme_engine):
    ask, adapter, client = acme_engine
    answerable = [c for c in load_cases("evals/acme.json") if c.answerable]

    results = [run_case(case, ask, adapter) for case in answerable]
    report = summarize(results, tokens=client.total_tokens, llm_calls=client.calls)
    print("\n" + report.render())  # the numbers, on the record

    # A floor, not a target. It exists to catch a regression, and to be raised once we know
    # what "normal" looks like.
    assert report.accuracy >= 0.6, report.render()

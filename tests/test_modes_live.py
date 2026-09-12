"""ACME smoke of the mode surface through the REAL Runtime (spec section 9): enrich into a
temp store, build_runtime, then ask in instant and deep modes. Live-gated like test_ask_live."""

import json
import os

import pytest

from acme_dsn import acme_dsn, requires_acme

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.runtime import build_runtime
from mnemiq.semantic.store import build_index
from mnemiq.semantic.values import build_value_index
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured"),
    requires_acme,
]

_DSN = acme_dsn()


def _identity():
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("modes-live")
    settings = Settings.from_env()
    settings.pg_dsn = settings.pg_dsn or _DSN
    settings.store_path = str(tmp / "store.duckdb")

    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    snap = enrich_semantic(
        enrich_structural(adapter, settings.source_id), LLMEnricher(LLMClient(settings))
    )
    con = init_store(settings.store_path)
    save_snapshot(con, snap)
    build_value_index(adapter, snap, con)
    build_index(con, snap, LLMEmbedder(settings))
    tables = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
    con.close()  # build_runtime reopens the store file

    policy = tmp / "authz.json"
    policy.write_text(json.dumps({"roles": {"analyst": tables}}))
    settings.authz_path = str(policy)
    return build_runtime(settings)


def test_instant_answers_the_simplest_acme_question(runtime):
    ans = runtime.ask("how many claims are there in total?", _identity(), mode="instant")
    assert ans.mode == "instant"
    assert ans.deferred is False
    assert "claim" in " ".join(ans.trace.tables_used)


def test_deep_runs_the_full_multi_candidate_path_or_defers_honestly(runtime):
    ans = runtime.ask("how many claims are there in total?", _identity(), mode="deep")
    assert ans.mode == "deep"
    if ans.deferred:
        assert "agreement" in ans.answer  # only the gate defers here, and it says so
    elif ans.candidates_executed is not None:
        # the min_agreement gate guarantees a full set on any non-deferred N>1 answer
        assert ans.candidates_executed == 5
        assert ans.agreement is not None and ans.agreement >= 0.6


def test_default_mode_is_thinking_end_to_end(runtime):
    ans = runtime.ask("how many claims are there in total?", _identity())
    assert ans.mode == "thinking"
    assert ans.deferred is False

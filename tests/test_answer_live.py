import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured"),
]

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


class _StaticAuthz:
    def __init__(self, objects):
        self.grants = GrantSet(frozenset(objects))

    def grants_for(self, identity):
        return self.grants


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    settings = Settings.from_env()
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    snapshot = enrich_semantic(
        enrich_structural(adapter, "acme"), LLMEnricher(LLMClient(settings))
    )
    con = init_store(str(tmp_path_factory.mktemp("store") / "s.duckdb"))
    build_index(con, snapshot, LLMEmbedder(settings))

    tables = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
    return con, snapshot, adapter, settings, tables


def _ask(engine, question, cache):
    con, snapshot, adapter, settings, tables = engine
    identity = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
    authz = _StaticAuthz(tables)

    packet = retrieve(con, question, identity, authz, LLMEmbedder(settings), k=5)
    agent = Agent(
        generator=LLMGenerator(LLMClient(settings)),
        synthesizer=LLMSynthesizer(LLMClient(settings)),
        adapter=adapter,
        cache=cache,
        budget=Budget(wall_clock_s=120.0),
    )
    return agent.answer(packet, snapshot, authz.grants, identity)


def test_the_engine_answers_a_real_question_with_a_real_number(engine):
    result = _ask(engine, "how many claims are there in total?", TwoTierCache(L1Cache()))

    assert result.deferred is False, result.answer
    assert result.answer.strip()
    assert any(ch.isdigit() for ch in result.answer), f"no number in the answer: {result.answer}"

    trace = result.trace
    assert trace is not None
    assert "claim" in " ".join(trace.tables_used)
    assert trace.result_shape in {"scalar", "table"}
    assert trace.enrichment_version


def test_the_same_question_twice_hits_the_cache(engine):
    cache = TwoTierCache(L1Cache())
    question = "how many claims are there in total?"

    first = _ask(engine, question, cache)
    assert first.cached is False

    second = _ask(engine, question, cache)
    assert second.cached is True  # the database was not asked again
    assert second.deferred is False


def test_the_engine_still_defers_rather_than_inventing(engine):
    result = _ask(engine, "what is the average salary of our employees?", TwoTierCache(L1Cache()))
    assert result.deferred is True, result.answer
    assert result.trace is None  # nothing ran, so there is nothing to trace

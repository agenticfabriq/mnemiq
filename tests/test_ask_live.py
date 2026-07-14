import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.generate.generator import LLMGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_index
from mnemiq.sql.verdict import Approved
from mnemiq.store.bootstrap import init_store

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured"),
]

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


class _StaticAuthz:
    def __init__(self, objects):
        self._grants = GrantSet(frozenset(objects))

    def grants_for(self, identity):
        return self._grants


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    settings = Settings.from_env()
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    snapshot = enrich_semantic(
        enrich_structural(adapter, "acme"), LLMEnricher(LLMClient(settings))
    )
    con = init_store(str(tmp_path_factory.mktemp("store") / "s.duckdb"))
    build_index(con, snapshot, LLMEmbedder(settings))
    return con, snapshot, adapter, settings


def _identity():
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def _ask(engine, question, allowed):
    con, snapshot, adapter, settings = engine
    authz = _StaticAuthz(allowed)
    packet = retrieve(con, question, _identity(), authz, LLMEmbedder(settings), k=5)
    return plan_query(
        packet,
        snapshot,
        authz.grants_for(_identity()),
        LLMGenerator(LLMClient(settings)),
        adapter=adapter,
        target="duckdb",
    )


def test_a_real_question_produces_sql_the_source_accepts(engine):
    con, _snapshot, _adapter, _settings = engine
    all_tables = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]

    outcome = _ask(engine, "how many claims are there in total?", all_tables)
    assert isinstance(outcome, Approved), outcome

    assert "claim" in " ".join(outcome.tables)
    assert "LIMIT" in outcome.target_sql  # imposed, whatever the model wrote
    # EXPLAIN already proved this against the real source inside plan_query


def test_the_engine_defers_instead_of_inventing_an_answer(engine):
    # ACME has no such data. A confident query here would be the worst possible outcome.
    outcome = _ask(engine, "what is the average salary of our employees?", ["claim", "policy"])
    assert isinstance(outcome, Deferred), outcome


def test_access_bounds_the_answer_end_to_end(engine):
    # the question is *about* people, but this identity cannot see the person table
    outcome = _ask(engine, "what are the last names of our policy holders?", ["claim", "policy"])

    if isinstance(outcome, Approved):
        assert "person" not in outcome.tables  # never, under any circumstances
    else:
        assert isinstance(outcome, Deferred)

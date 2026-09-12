import os

import pytest

from acme_dsn import acme_dsn, requires_acme

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured"),
    requires_acme,
]

_DSN = acme_dsn()


class _StaticAuthz:
    def __init__(self, *objects):
        self._grants = GrantSet(frozenset(objects))

    def grants_for(self, identity):
        return self._grants


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    settings = Settings.from_env()
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    snapshot = enrich_semantic(
        enrich_structural(adapter, "acme"), LLMEnricher(LLMClient(settings))
    )
    con = init_store(str(tmp_path_factory.mktemp("store") / "s.duckdb"))
    build_index(con, snapshot, LLMEmbedder(settings))
    return con


def _identity():
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def test_a_business_question_retrieves_the_right_physical_tables(indexed):
    """The premise of the semantic layer: ask in business language, get the right tables."""
    authz = _StaticAuthz(
        *(r[0] for r in indexed.execute("SELECT object_id FROM semantic_object").fetchall())
    )

    packet = retrieve(
        indexed,
        "how much have we paid out on fire damage claims?",
        _identity(),
        authz,
        LLMEmbedder(Settings.from_env()),
        k=5,
    )
    retrieved = {c.object_id for c in packet.cards}
    assert retrieved & {"claim", "fireclaim", "claim_amount", "loss_payment"}, retrieved


def test_grants_bound_what_can_ever_be_retrieved(indexed):
    """The same question, a narrower identity: the person table must stay invisible."""
    packet = retrieve(
        indexed,
        "who is the person on the policy?",  # aimed squarely at the forbidden table
        _identity(),
        _StaticAuthz("claim", "policy"),
        LLMEmbedder(Settings.from_env()),
        k=5,
    )
    retrieved = {c.object_id for c in packet.cards}
    assert retrieved <= {"claim", "policy"}
    assert "person" not in retrieved

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column
from mnemiq.generate.generator import SqlProposal
from mnemiq.generate.plan_query import plan_query
from mnemiq.semantic.federation import FederatedSnapshot
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.verdict import Approved


class _Gen:
    def propose(self, packet, feedback=None):
        return SqlProposal(sql="SELECT id FROM pg.person", reason="")


def test_plan_query_uses_registry_from_federated_snapshot():
    snap = FederatedSnapshot(
        version="fed-x", source_id="federated", created_at="t",
        columns=[Column(id="pg.person.id", object_id="pg.person", name="id")],
        registry={"pg": "public"},
    )
    grants = GrantSet(frozenset({"pg.person"}))
    packet = ContextPacket(
        question="q",
        cards=[RetrievedCard("pg.person", "TABLE pg.person", 1.0)],
        grant_fingerprint=grants.fingerprint,
        enrichment_version="fed-x",
    )
    out = plan_query(packet, snap, grants, _Gen(), dialect="duckdb", target="duckdb")
    assert isinstance(out, Approved)
    assert "pg.public.person" in out.target_sql  # registry drove the expansion

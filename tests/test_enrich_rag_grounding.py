import duckdb

from mnemiq.contract import Column, Definition, Snapshot, SourceBinding
from mnemiq.enrichment.enricher import FakeEnricher
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.ontology.records import OntologyRecords
from mnemiq.semantic.definition_index import DefinitionIndex, DefinitionRetriever, build_definition_index


def _snap():
    return Snapshot(
        version="v1", source_id="acme", created_at="2026-07-26T00:00:00Z",
        source_bindings=[SourceBinding(id="sb:claim", source_id="acme", object_id="claim",
                                       source_object="claim", binding_type="table")],
        columns=[Column(id="claim.premium", object_id="claim", name="premium", data_type="numeric")],
        definitions=[Definition(id="d1", term="Premium", domain="ins",
                                definition="the amount paid for coverage")],
    )


def _retriever(snap):
    con = duckdb.connect()
    build_definition_index(OntologyRecords(), snap, con, FakeEmbedder())
    return DefinitionRetriever(DefinitionIndex(con), FakeEmbedder())


def test_retriever_grounding_reaches_the_enricher():
    snap = _snap()
    enr = FakeEnricher(replies={"claim": '{"columns": [{"name": "premium"}]}'})
    enrich_semantic(snap, enr, retriever=_retriever(snap))
    assert "REFERENCE" in enr.grounding_calls["claim"]


def test_none_retriever_is_byte_identical_to_no_rag():
    snap = _snap()
    replies = {"claim": '{"columns": [{"name": "premium", "description": "x"}]}'}
    baseline = enrich_semantic(snap, FakeEnricher(replies=dict(replies)))
    with_none = enrich_semantic(snap, FakeEnricher(replies=dict(replies)), retriever=None)
    assert with_none.model_dump() == baseline.model_dump()


def test_retriever_error_is_fail_soft():
    class Boom:
        def grounding_for(self, table, columns):
            raise RuntimeError("down")

    snap = _snap()
    enr = FakeEnricher(replies={"claim": '{"columns": [{"name": "premium"}]}'})
    out = enrich_semantic(snap, enr, retriever=Boom())  # must not raise
    assert enr.grounding_calls["claim"] == ""  # fell back to no grounding
    assert out.version  # produced a snapshot

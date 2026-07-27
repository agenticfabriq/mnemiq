import duckdb

from mnemiq.contract import Definition, Snapshot
from mnemiq.llm.embeddings import EMBED_DIM, FakeEmbedder
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords
from mnemiq.semantic.definition_index import build_definition_index, DefinitionIndex


def _snapshot(definitions):
    return Snapshot(version="v1", source_id="acme", created_at="2026-07-26T00:00:00Z",
                    definitions=definitions)


def _scheme(id_, n):
    return ConceptScheme(id=id_, label=id_, concepts=[
        Concept(id=f"{id_}:{i}", notation=str(i), pref_label=f"{id_} term {i}",
                definition=f"definition of {id_} {i}") for i in range(n)])


def test_indexes_definitions_and_in_scale_concepts(tmp_path):
    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    records = OntologyRecords(schemes=[_scheme("small", 3)])
    n = build_definition_index(records, snap, con, FakeEmbedder(), max_concepts=500)
    assert n == 4  # 1 definition + 3 concepts
    rows = con.execute("SELECT count(*), len(embedding) FROM definition_concept "
                       "GROUP BY len(embedding)").fetchall()
    assert rows == [(4, EMBED_DIM)]


def test_oversized_scheme_concepts_are_skipped(tmp_path):
    con = duckdb.connect()
    records = OntologyRecords(schemes=[_scheme("big", 5)])
    n = build_definition_index(records, _snapshot([]), con, FakeEmbedder(), max_concepts=2)
    assert n == 0  # 5 > 2 -> the scheme contributes no concept rows


def test_no_embedder_writes_nothing(tmp_path):
    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins", definition="x")])
    assert build_definition_index(OntologyRecords(), snap, con, None) == 0
    assert DefinitionIndex(con).nearest("premium", None) == []


def test_nearest_returns_k_scored_terms(tmp_path):
    con = duckdb.connect()
    defs = [Definition(id=f"d{i}", term=f"Term{i}", domain="d", definition=f"meaning {i}")
            for i in range(6)]
    build_definition_index(OntologyRecords(), _snapshot(defs), con, FakeEmbedder())
    hits = DefinitionIndex(con).nearest("Term1 meaning", FakeEmbedder(), k=3)
    assert len(hits) == 3
    assert all(isinstance(t, str) and isinstance(x, str) and isinstance(s, float)
               for t, x, s in hits)
    scores = [s for _, _, s in hits]
    assert scores == sorted(scores, reverse=True)  # descending

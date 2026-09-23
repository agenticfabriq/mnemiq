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


def test_the_definition_index_is_built_at_the_embedder_s_width_not_1536(tmp_path):
    # definition_concept has its own DDL, sized independently of semantic_object and example --
    # a fix that only touched those two would leave this table still fixed at 1536.
    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    n = build_definition_index(OntologyRecords(), snap, con, FakeEmbedder(dim=64))
    assert n == 1
    (decl,) = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'definition_concept' AND column_name = 'embedding'"
    ).fetchone()
    assert "64" in decl and "1536" not in decl


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


def test_a_fail_soft_rebuild_still_clears_the_old_rows(tmp_path):
    # Delete-then-insert per source: a rebuild that turns up nothing new must not leave the
    # PREVIOUS build's rows behind, since nearest() reads across every source with no filter.
    # Not a live bug today -- cli.py builds into a fresh connection every run -- but a contract
    # a future caller that reuses a connection across builds needs to hold.
    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    assert build_definition_index(OntologyRecords(), snap, con, FakeEmbedder()) == 1

    assert build_definition_index(OntologyRecords(), snap, con, None) == 0
    assert con.execute("SELECT count(*) FROM definition_concept").fetchone()[0] == 0


def test_an_empty_corpus_rebuild_also_clears_the_old_rows(tmp_path):
    # Same contract, the "real embedder but nothing to embed this time" branch: a source that
    # dropped its last definition should not keep grounding on the one that used to be there.
    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    assert build_definition_index(OntologyRecords(), snap, con, FakeEmbedder()) == 1

    assert build_definition_index(OntologyRecords(), _snapshot([]), con, FakeEmbedder()) == 0
    assert con.execute("SELECT count(*) FROM definition_concept").fetchone()[0] == 0


def test_an_embed_error_also_clears_the_old_rows(tmp_path):
    # Same contract, the third fail-soft branch: an embedder that raises. This is the ordering
    # this task deliberately kept from the function's original behaviour -- see the comment on
    # the DELETE in build_definition_index for the trade it makes.
    class _FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("down")

    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    assert build_definition_index(OntologyRecords(), snap, con, FakeEmbedder()) == 1

    assert build_definition_index(OntologyRecords(), snap, con, _FailingEmbedder()) == 0
    assert con.execute("SELECT count(*) FROM definition_concept").fetchone()[0] == 0


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

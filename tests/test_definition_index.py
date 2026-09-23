import logging

import duckdb
import pytest

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
    # Same contract, the third fail-soft branch: an embedder that raises. See the comment on
    # the DELETE in build_definition_index for the trade this ordering makes.
    class _FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("down")

    con = duckdb.connect()
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    assert build_definition_index(OntologyRecords(), snap, con, FakeEmbedder()) == 1

    assert build_definition_index(OntologyRecords(), snap, con, _FailingEmbedder()) == 0
    assert con.execute("SELECT count(*) FROM definition_concept").fetchone()[0] == 0


def test_a_width_mismatch_against_an_existing_store_is_refused_not_wiped(tmp_path):
    # Same exposure as build_index/build_example_index: CREATE TABLE IF NOT EXISTS no-ops
    # against a store already built at a different width, so a naive rebuild's DELETE would run
    # and autocommit before the INSERT's ConversionException -- an emptied index, no guidance.
    from mnemiq.store.bootstrap import init_store

    path = str(tmp_path / "s.duckdb")
    snap = _snapshot([Definition(id="d1", term="Premium", domain="ins",
                                 definition="the amount paid for coverage")])
    con = init_store(path)
    assert build_definition_index(OntologyRecords(), snap, con, FakeEmbedder()) == 1
    con.close()

    con = init_store(path)
    with pytest.raises(RuntimeError) as exc_info:
        build_definition_index(OntologyRecords(), snap, con, FakeEmbedder(dim=1024))
    message = str(exc_info.value)
    assert "1536" in message and "1024" in message

    # The point of the fix: the original build's row must survive the refused rebuild.
    assert con.execute("SELECT count(*) FROM definition_concept").fetchone()[0] == 1


def test_nearest_on_a_never_built_index_is_silent(tmp_path, caplog):
    # DefinitionIndex.__init__ no longer creates the table (no embedder there to size it with),
    # so a query against a never-built index used to raise duckdb.CatalogException and fall into
    # nearest's broad except, logging a WARNING on every single call -- reachable whenever
    # build_definition_index fail-softs (empty corpus, embed error) and cli.py's enrich path
    # still asks for grounding per table. A never-built index should answer "nothing indexed"
    # exactly as cheaply and quietly as an empty one.
    con = duckdb.connect()
    with caplog.at_level(logging.WARNING):
        hits = DefinitionIndex(con).nearest("premium", FakeEmbedder())
    assert hits == []
    assert caplog.records == []


def test_nearest_still_warns_when_the_table_exists_but_the_query_fails(caplog):
    # A missing table and a missing function raise the SAME duckdb.CatalogException type
    # (confirmed directly: SELECT nosuchfn(1) also raises CatalogException). Catching that type
    # around the query itself would silently swallow both, defeating the WARNING a genuinely
    # broken query -- say, array_cosine_similarity missing after a DuckDB downgrade -- is meant
    # to raise. "Never built" has to be checked explicitly, not inferred from the query's own
    # exception type.
    class _TableExistsButQueryFails:
        def execute(self, sql, params=None):
            if "information_schema.tables" in sql:
                return self  # fetchone() below reports the table as present
            raise duckdb.CatalogException(
                "Scalar Function with name array_cosine_similarity does not exist!"
            )

        def fetchone(self):
            return (1,)

    with caplog.at_level(logging.WARNING):
        hits = DefinitionIndex(_TableExistsButQueryFails()).nearest("premium", FakeEmbedder())
    assert hits == []
    assert any("query failed" in r.message for r in caplog.records)


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

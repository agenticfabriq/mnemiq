import pytest

from mnemiq.contract import Column, Example, Snapshot, SourceBinding
from mnemiq.llm.embeddings import EMBED_DIM, FakeEmbedder
from mnemiq.semantic.store import build_example_index, build_index, indexed_version
from mnemiq.store.bootstrap import init_store


def _snapshot(version="v1") -> Snapshot:
    return Snapshot(
        version=version,
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        source_bindings=[
            SourceBinding(
                id=f"sb:{t}", source_id="acme", object_id=t, source_object=t, binding_type="table"
            )
            for t in ("claim", "person")
        ],
        columns=[
            Column(id="claim.status", object_id="claim", name="status", data_type="text"),
            Column(id="person.age", object_id="person", name="age", data_type="integer"),
        ],
    )


def test_build_index_writes_a_card_per_table(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    assert build_index(con, _snapshot(), FakeEmbedder()) == 2

    rows = con.execute(
        "SELECT object_id, version, len(embedding) FROM semantic_object ORDER BY object_id"
    ).fetchall()
    assert [r[0] for r in rows] == ["claim", "person"]
    assert all(r[1] == "v1" for r in rows)
    assert all(r[2] == EMBED_DIM for r in rows)
    assert indexed_version(con, "acme") == "v1"


def test_rebuilding_the_same_version_is_idempotent(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, _snapshot(), FakeEmbedder())
    build_index(con, _snapshot(), FakeEmbedder())
    assert con.execute("SELECT count(*) FROM semantic_object").fetchone()[0] == 2


def test_a_new_version_replaces_the_old_one(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, _snapshot("v1"), FakeEmbedder())
    build_index(con, _snapshot("v2"), FakeEmbedder())

    versions = con.execute("SELECT DISTINCT version FROM semantic_object").fetchall()
    assert versions == [("v2",)]  # one live index per source; stale cards never linger
    assert indexed_version(con, "acme") == "v2"


def test_an_index_is_built_at_the_embedder_s_width_not_1536(tmp_path):
    # A local embedding model is rarely 1536 wide (BGE-M3 is 1024, nomic-embed 768). The DDL
    # used to bake EMBED_DIM in at import time while every query cast to the real vector
    # length, so a narrower embedder wrote into a column it could not be compared against.
    con = init_store(str(tmp_path / "s.duckdb"))
    n = build_index(con, _snapshot(), FakeEmbedder(dim=64))

    assert n == 2  # same count the other tests assert
    (decl,) = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'semantic_object' AND column_name = 'embedding'"
    ).fetchone()
    assert "64" in decl and "1536" not in decl


def test_the_example_index_is_also_built_at_the_embedder_s_width(tmp_path):
    # example has its own DDL, sized independently of semantic_object -- a fix that only
    # touched one of the two would leave this table still fixed at 1536.
    con = init_store(str(tmp_path / "s.duckdb"))
    snap = _snapshot().model_copy(update={"examples": [
        Example(question="how many claims?", sql="SELECT count(*) FROM claim",
                tables=["claim"], object_id="claim"),
    ]})
    n = build_example_index(con, snap, FakeEmbedder(dim=64))

    assert n == 1
    (decl,) = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'example' AND column_name = 'embedding'"
    ).fetchone()
    assert "64" in decl and "1536" not in decl


def test_a_width_mismatch_against_an_existing_store_is_refused_not_wiped(tmp_path):
    # Reviewer-measured crash: CREATE TABLE IF NOT EXISTS no-ops against a store already built at
    # a different width, so the per-source DELETE runs and autocommits before DuckDB's own
    # ConversionException kills the INSERT on the width mismatch -- an emptied index, no
    # guidance. The fix must catch this before that DELETE, not after.
    path = str(tmp_path / "s.duckdb")
    con = init_store(path)
    build_index(con, _snapshot(), FakeEmbedder())  # 1536-wide, the hosted default
    con.close()

    con = init_store(path)
    with pytest.raises(RuntimeError) as exc_info:
        build_index(con, _snapshot(), FakeEmbedder(dim=1024))
    message = str(exc_info.value)
    assert "1536" in message and "1024" in message

    # The point of the fix: the original build's rows must survive the refused rebuild.
    rows = con.execute("SELECT count(*) FROM semantic_object").fetchone()[0]
    assert rows == 2


def test_an_example_index_width_mismatch_is_also_refused_not_wiped(tmp_path):
    # example has its own DDL and its own DELETE, sized and guarded independently of
    # semantic_object -- a fix that only touched one of the two would leave this table exposed.
    path = str(tmp_path / "s.duckdb")
    snap = _snapshot().model_copy(update={"examples": [
        Example(question="how many claims?", sql="SELECT count(*) FROM claim",
                tables=["claim"], object_id="claim"),
    ]})
    con = init_store(path)
    build_example_index(con, snap, FakeEmbedder())
    con.close()

    con = init_store(path)
    with pytest.raises(RuntimeError) as exc_info:
        build_example_index(con, snap, FakeEmbedder(dim=1024))
    message = str(exc_info.value)
    assert "1536" in message and "1024" in message
    assert con.execute("SELECT count(*) FROM example").fetchone()[0] == 1


def test_the_full_text_index_is_queryable(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, _snapshot(), FakeEmbedder())

    hit = con.execute(
        "SELECT object_id FROM ("
        "  SELECT object_id, fts_main_semantic_object.match_bm25(object_id, 'status') AS score"
        "  FROM semantic_object"
        ") WHERE score IS NOT NULL"
    ).fetchall()
    assert hit == [("claim",)]

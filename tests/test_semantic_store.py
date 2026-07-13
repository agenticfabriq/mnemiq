from mnemiq.contract import Column, Snapshot, SourceBinding
from mnemiq.llm.embeddings import EMBED_DIM, FakeEmbedder
from mnemiq.semantic.store import build_index, indexed_version
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

import pytest

from mnemiq.config import SourceSpec
from mnemiq.contract import Column, Snapshot
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store
from mnemiq.store.federated_build import build_federated_snapshot


class _FakeEmbedder:
    @property
    def dim(self) -> int:
        return 1536  # EMBED_DIM, matched to the vectors this stub returns below

    def embed(self, texts):
        return [[float(len(t) % 7)] * 1536 for t in texts]  # EMBED_DIM vectors, deterministic


def _snap(sid, ver, table):
    return Snapshot(version=ver, source_id=sid, created_at="t",
                    columns=[Column(id=f"{table}.id", object_id=table, name="id")])


def test_federated_build_indexes_qualified_ids(tmp_path):
    con = init_store(str(tmp_path / "store.duckdb"))
    pairs = [
        (SourceSpec(id="a", kind="postgres", target="d", catalog="pg", schema="public"),
         _snap("a", "v1", "person")),
        (SourceSpec(id="b", kind="sqlite", target="f", catalog="ops", schema="main"),
         _snap("b", "v2", "person")),
    ]
    n, _ = build_federated_snapshot(con, pairs, _FakeEmbedder())
    ids = {r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()}
    assert ids == {"pg.person", "ops.person"}  # same table name, no collision
    assert n == 2


def test_federated_rebuild_refuses_a_width_mismatch_before_deleting_the_old_rows(tmp_path):
    """Reviewer-caught failure: a store built single-source at 1536, then pointed at a
    1024-wide local embedder and rebuilt federated, must not have its existing rows deleted
    before the width mismatch is ever noticed. `build_federated_snapshot` has its own
    per-source DELETE ahead of `build_index`'s call, so a refusal that only fired once
    `build_index` reached ITS OWN check would already have run and autocommitted that DELETE --
    emptying the index and then telling the operator to rebuild the very thing just emptied.
    """
    path = str(tmp_path / "store.duckdb")
    con = init_store(path)
    build_index(con, _snap("acme", "v1", "claim"), FakeEmbedder())  # single-source, 1536-wide
    con.close()

    con = init_store(path)
    pairs = [
        (SourceSpec(id="a", kind="postgres", target="d", catalog="pg", schema="public"),
         _snap("a", "v1", "person")),
    ]
    with pytest.raises(RuntimeError) as exc_info:
        build_federated_snapshot(con, pairs, FakeEmbedder(dim=1024))
    message = str(exc_info.value)
    assert "1536" in message and "1024" in message

    # The point of the fix: the single-source build's rows must survive the refused rebuild.
    assert con.execute("SELECT object_id FROM semantic_object").fetchall() == [("claim",)]

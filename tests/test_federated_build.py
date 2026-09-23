import pytest

from mnemiq.config import SourceSpec
from mnemiq.contract import Column, Example, Snapshot, SourceBinding
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.store import build_example_index, build_index
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


def test_federated_rebuild_refuses_an_example_width_mismatch_too(tmp_path):
    """The semantic_object and example checks are independent -- each guards its own DELETE.
    Built so ONLY `example` mismatches (`semantic_object` is already at the width this federated
    rebuild uses), isolating this from the test above: review caught that a version of this fix
    covering just `semantic_object` left `example` unrefused, and the prior test could not have
    caught that gap because its snapshot carries no examples at all.
    """
    path = str(tmp_path / "store.duckdb")
    con = init_store(path)
    build_index(con, _snap("acme", "v1", "claim"), FakeEmbedder(dim=1024))  # matches the dim below
    snap_with_example = _snap("acme", "v1", "claim").model_copy(update={"examples": [
        Example(question="q", sql="SELECT 1", tables=["claim"], object_id="claim"),
    ]})
    build_example_index(con, snap_with_example, FakeEmbedder(dim=1536))  # the one that mismatches
    con.close()

    con = init_store(path)
    pairs = [
        (SourceSpec(id="a", kind="postgres", target="d", catalog="pg", schema="public"),
         Snapshot(version="v1", source_id="a", created_at="t",
                  columns=[Column(id="person.id", object_id="person", name="id")],
                  examples=[Example(question="q2", sql="SELECT 1", tables=["person"],
                                     object_id="person")])),
    ]
    with pytest.raises(RuntimeError) as exc_info:
        build_federated_snapshot(con, pairs, FakeEmbedder(dim=1024))
    message = str(exc_info.value)
    assert "example" in message  # names the mismatched table, not semantic_object
    assert "1536" in message and "1024" in message

    # The point of the fix: the earlier example row must survive the refused rebuild.
    assert con.execute("SELECT object_id FROM example").fetchall() == [("claim",)]


def test_federated_rebuild_still_refuses_when_the_snapshot_has_bindings_but_no_columns(tmp_path):
    """`build_cards` makes one card per `source_bindings` entry, falling back to `columns` only
    when there are NO bindings -- so a first version of this fix that gated on `fed.columns`
    instead of `build_cards(fed)` skipped the width check for exactly this shape while
    `build_index` still produced cards for it, reopening the data-loss window one binding shape
    narrower than the fix that closed it everywhere else. Caught by review before it shipped.
    """
    path = str(tmp_path / "store.duckdb")
    con = init_store(path)
    build_index(con, _snap("acme", "v1", "claim"), FakeEmbedder())  # single-source, 1536-wide
    con.close()

    con = init_store(path)
    pairs = [
        (SourceSpec(id="a", kind="postgres", target="d", catalog="pg", schema="public"),
         Snapshot(version="v1", source_id="a", created_at="t",
                  source_bindings=[SourceBinding(id="sb:person", source_id="a",
                                                  object_id="person", source_object="person",
                                                  binding_type="table")])),  # bindings, no columns
    ]
    with pytest.raises(RuntimeError) as exc_info:
        build_federated_snapshot(con, pairs, FakeEmbedder(dim=1024))
    message = str(exc_info.value)
    assert "1536" in message and "1024" in message

    # The point of the fix: the single-source build's rows must survive the refused rebuild.
    assert con.execute("SELECT object_id FROM semantic_object").fetchall() == [("claim",)]

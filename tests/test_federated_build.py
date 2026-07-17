from mnemiq.config import SourceSpec
from mnemiq.contract import Column, Snapshot
from mnemiq.store.bootstrap import init_store
from mnemiq.store.federated_build import build_federated_snapshot


class _FakeEmbedder:
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

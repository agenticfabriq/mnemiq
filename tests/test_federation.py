from mnemiq.config import SourceSpec
from mnemiq.contract import Column, Example, Snapshot, TableFacts
from mnemiq.semantic.federation import FederatedSnapshot, merge_snapshots, qualify_object_id


def test_qualify_object_id():
    assert qualify_object_id("pg", "person") == "pg.person"


def _snap(source_id, version, table):
    return Snapshot(
        version=version, source_id=source_id, created_at="t",
        columns=[Column(id=f"{table}.id", object_id=table, name="id")],
        table_facts=[TableFacts(object_id=table, grain="one row per id")],
        examples=[Example(question="q", sql=f"SELECT id FROM {table}",
                          tables=[table], object_id=table)],
    )


def test_merge_qualifies_object_ids_and_carries_registry():
    pairs = [
        (SourceSpec(id="a", kind="postgres", target="d", catalog="pg", schema="public"),
         _snap("a", "v1", "person")),
        (SourceSpec(id="b", kind="sqlite", target="f", catalog="ops", schema="main"),
         _snap("b", "v9", "orders")),
    ]
    fed = merge_snapshots(pairs)
    assert isinstance(fed, FederatedSnapshot) and isinstance(fed, Snapshot)
    assert fed.registry == {"pg": "public", "ops": "main"}
    assert {c.object_id for c in fed.columns} == {"pg.person", "ops.orders"}
    assert {tf.object_id for tf in fed.table_facts} == {"pg.person", "ops.orders"}
    ex = {e.object_id: e for e in fed.examples}
    assert ex["pg.person"].tables == ["pg.person"]  # example table refs qualified too
    assert fed.version.startswith("fed-")  # composite, deterministic
    assert merge_snapshots(pairs).version == fed.version  # stable across calls

import pytest

from mnemiq.contract import Column, Relationship, Snapshot
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot, save_snapshot


def _snap(version="v1"):
    return Snapshot(
        version=version,
        source_id="acme",
        created_at="2026-07-12T00:00:00Z",
        columns=[Column(id="party.name", object_id="party", name="name", data_type="text")],
        relationships=[
            Relationship(
                id="claim->party", from_="claim", to="party", cardinality="many_to_one"
            )
        ],
    )


def test_save_load_round_trip(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    snap = _snap()
    save_snapshot(con, snap)
    loaded = load_snapshot(con, "v1")
    assert loaded == snap
    assert loaded.relationships[0].from_ == "claim"  # the "from" alias survives the trip


def test_current_version_is_latest(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, _snap("v1"))
    save_snapshot(con, _snap("v2"))
    assert current_version(con, "acme") == "v2"
    assert current_version(con, "nobody") is None


def test_resaving_a_version_replaces_it(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, _snap("v1"))
    save_snapshot(con, _snap("v1"))
    assert con.execute("SELECT count(*) FROM snapshot").fetchone()[0] == 1


def test_load_unknown_version_raises(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    with pytest.raises(KeyError):
        load_snapshot(con, "nope")

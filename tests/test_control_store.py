from mnemiq.contract import Snapshot
from mnemiq.store.bootstrap import init_store
from mnemiq.store.control import resolve_version
from mnemiq.store.snapshot_store import has_snapshot, save_snapshot


def test_has_snapshot(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, Snapshot(version="v1", source_id="acme", created_at="t"))
    assert has_snapshot(con, "v1") is True
    assert has_snapshot(con, "nope") is False


def test_resolve_version_without_control_dsn_uses_local(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, Snapshot(version="v1", source_id="acme", created_at="t"))
    assert resolve_version(con, None, "acme") == "v1"

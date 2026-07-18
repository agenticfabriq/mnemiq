from mnemiq.config import Settings
from mnemiq.contract import Snapshot
from mnemiq.runtime import Runtime
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot


def _settings(tmp_path, control=None):
    return Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="postgres://x",
                    acme_data_dir=None, source_id="acme",
                    store_path=str(tmp_path / "s.duckdb"), control_dsn=control)


def test_reload_is_noop_without_control_dsn(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, Snapshot(version="v1", source_id="acme", created_at="t"))
    rt = Runtime(con=con, snapshot=Snapshot(version="v1", source_id="acme", created_at="t"),
                 adapter=None, agent=None, embedder=None, authz=None,
                 settings=_settings(tmp_path), loaded_versions={"acme": "v1"})
    rt.reload_if_stale()  # no control dsn -> must not touch the DB or raise
    assert rt.snapshot.version == "v1"

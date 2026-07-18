import mnemiq.cli as cli
from mnemiq.config import Settings


def test_build_does_not_publish_without_control_dsn(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("mnemiq.store.control.publish_version",
                        lambda *a, **k: calls.append(a))
    s = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn=None,
                 acme_data_dir=None, source_id="acme", store_path=str(tmp_path / "s.duckdb"),
                 control_dsn=None)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: s))
    # The build may fail for unrelated reasons (no LLM creds in the test env); the invariant we
    # assert is only that publish is never called when control_dsn is None.
    try:
        cli.main(["build"])
    except Exception:
        pass
    assert calls == []

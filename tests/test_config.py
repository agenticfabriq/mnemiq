import json

from mnemiq.config import Settings, SourceSpec


def test_from_env_reads_vars(monkeypatch):
    monkeypatch.setenv("MNEMIQ_LLM_MODEL", "openai.gpt-5-mini")
    monkeypatch.setenv("MNEMIQ_PG_DSN", "postgresql://u:p@localhost:5432/acme")
    monkeypatch.delenv("MNEMIQ_LLM_API_KEY", raising=False)
    s = Settings.from_env()
    assert s.llm_model == "openai.gpt-5-mini"
    assert s.pg_dsn.endswith("/acme")
    assert s.llm_api_key is None


def test_default_model():
    s = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn=None, acme_data_dir=None
    )
    assert s.llm_model == "openai.gpt-5.5"


def test_source_id_and_store_path_defaults():
    s = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn=None, acme_data_dir=None
    )
    assert s.source_id == "acme"
    assert s.store_path == "mnemiq.duckdb"


def test_default_mode_reads_env(monkeypatch):
    monkeypatch.setenv("MNEMIQ_MODE", "deep")
    assert Settings.from_env().default_mode == "deep"


def test_default_mode_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("MNEMIQ_MODE", raising=False)
    assert Settings.from_env().default_mode is None


def test_write_enabled_from_env(monkeypatch):
    monkeypatch.setenv("MNEMIQ_WRITE_ENABLED", "1")
    assert Settings.from_env().write_enabled is True


def test_write_enabled_defaults_false(monkeypatch):
    monkeypatch.delenv("MNEMIQ_WRITE_ENABLED", raising=False)
    assert Settings.from_env().write_enabled is False


def test_source_specs_synthesizes_single_source_from_pg_dsn():
    s = Settings(llm_base_url=None, llm_api_key=None, llm_model=None,
                 pg_dsn="postgres://x", acme_data_dir=None, source_id="acme")
    specs = s.source_specs()
    assert specs == [SourceSpec(id="acme", kind="postgres", target="postgres://x",
                                catalog="src", schema="public")]


def test_embed_endpoint_defaults_to_chat():
    s = Settings(llm_base_url="http://chat", llm_api_key="ck", llm_model=None, pg_dsn=None,
                 acme_data_dir=None)
    assert s.embed_endpoint() == ("http://chat", "ck")


def test_embed_endpoint_split_overrides_chat():
    s = Settings(llm_base_url="http://local-vllm", llm_api_key="x", llm_model=None, pg_dsn=None,
                 acme_data_dir=None, embed_base_url="http://hosted", embed_api_key="hk")
    assert s.embed_endpoint() == ("http://hosted", "hk")  # generation local, embeddings hosted


def test_control_dsn_from_env(monkeypatch):
    monkeypatch.setenv("MNEMIQ_CONTROL_DSN", "postgresql://ctl")
    assert Settings.from_env().control_dsn == "postgresql://ctl"


def test_control_dsn_defaults_none(monkeypatch):
    monkeypatch.delenv("MNEMIQ_CONTROL_DSN", raising=False)
    assert Settings.from_env().control_dsn is None


def test_source_specs_reads_manifest(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([
        {"id": "sales", "kind": "postgres", "target": "postgres://p",
         "catalog": "pg", "schema": "public"},
        {"id": "ops", "kind": "sqlite", "target": "/tmp/ops.db",
         "catalog": "ops", "schema": "main"},
    ]))
    s = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn=None,
                 acme_data_dir=None, sources_path=str(manifest))
    specs = s.source_specs()
    assert [sp.catalog for sp in specs] == ["pg", "ops"]
    assert specs[1].kind == "sqlite" and specs[1].schema == "main"

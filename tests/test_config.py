from mnemiq.config import Settings


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

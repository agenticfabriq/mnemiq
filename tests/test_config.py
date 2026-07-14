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

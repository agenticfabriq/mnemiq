from mnemiq.config import Settings


def test_verify_defaults_off(monkeypatch):
    for k in ("MNEMIQ_VERIFY", "MNEMIQ_VERIFY_THRESHOLD", "MNEMIQ_VERIFY_JUDGE"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.verify is False and s.verify_threshold == 0.5
    assert s.verify_sanity is True and s.verify_grounding is False and s.verify_judge is False


def test_verify_env_parsed(monkeypatch):
    monkeypatch.setenv("MNEMIQ_VERIFY", "1")
    monkeypatch.setenv("MNEMIQ_VERIFY_THRESHOLD", "0.7")
    monkeypatch.setenv("MNEMIQ_VERIFY_JUDGE", "1")
    monkeypatch.setenv("MNEMIQ_VERIFY_BASE_URL", "http://localhost:8000/v1")
    s = Settings.from_env()
    assert s.verify is True and s.verify_threshold == 0.7 and s.verify_judge is True
    assert s.verify_endpoint()[0] == "http://localhost:8000/v1"


def test_verify_endpoint_falls_back_to_llm(monkeypatch):
    monkeypatch.delenv("MNEMIQ_VERIFY_BASE_URL", raising=False)
    monkeypatch.setenv("MNEMIQ_LLM_BASE_URL", "http://host/v1")
    monkeypatch.setenv("MNEMIQ_LLM_API_KEY", "k")
    s = Settings.from_env()
    assert s.verify_endpoint() == ("http://host/v1", "k")

from mnemiq.config import Settings


def test_sp5b_settings_defaults(monkeypatch):
    for var in ("MNEMIQ_DEFINITION_INDEX_MAX_CONCEPTS", "MNEMIQ_BINDING_SUGGESTIONS_PATH"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.definition_index_max_concepts == 500
    assert s.binding_suggestions_path is None


def test_sp5b_settings_from_env(monkeypatch):
    monkeypatch.setenv("MNEMIQ_DEFINITION_INDEX_MAX_CONCEPTS", "10")
    monkeypatch.setenv("MNEMIQ_BINDING_SUGGESTIONS_PATH", "/tmp/x.json")
    s = Settings()
    assert s.definition_index_max_concepts == 10
    assert s.binding_suggestions_path == "/tmp/x.json"

import json
import pathlib

import pytest

from mnemiq.config import Settings, SourceSpec


def _src_files_reading(var: str) -> list[str]:
    """Engine source files (other than config.py) that still READ `var` from the environment —
    i.e. a line mentioning the var AND os.getenv/os.environ (comments mentioning it are fine)."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "mnemiq"
    hits = []
    for p in root.rglob("*.py"):
        if p.name == "config.py":
            continue
        for line in p.read_text().splitlines():
            if var in line and ("getenv" in line or "environ" in line):
                hits.append(p.name)
                break
    return hits


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


# --- pydantic-settings migration (2026-07-21) ---


def test_env_prefix_maps_existing_names(monkeypatch):
    monkeypatch.setenv("MNEMIQ_VERIFY_THRESHOLD", "0.3")
    monkeypatch.setenv("MNEMIQ_MODE", "deep")
    s = Settings.from_env()
    assert s.verify_threshold == 0.3        # coerced to float
    assert s.default_mode == "deep"


def test_verify_tristate(monkeypatch):
    monkeypatch.delenv("MNEMIQ_VERIFY", raising=False)
    assert Settings.from_env().verify_override is None
    assert Settings.from_env().verify is False
    monkeypatch.setenv("MNEMIQ_VERIFY", "0")
    assert Settings.from_env().verify_override == "0"
    assert Settings.from_env().verify is False
    monkeypatch.setenv("MNEMIQ_VERIFY", "1")
    assert Settings.from_env().verify_override == "1"
    assert Settings.from_env().verify is True


def test_validation_rejects_bad_threshold(monkeypatch):
    monkeypatch.setenv("MNEMIQ_VERIFY_THRESHOLD", "2.0")   # > 1.0 -> pydantic range error
    with pytest.raises(Exception):
        Settings.from_env()
    # NB: a bad MNEMIQ_MODE is rejected at boot by build_runtime (UnknownMode), not here.


def test_folded_fields_read_their_env(monkeypatch):
    monkeypatch.setenv("MNEMIQ_GUIDED_SQL", "1")
    monkeypatch.setenv("MNEMIQ_ASSERTIVE_SQL", "1")
    monkeypatch.setenv("MNEMIQ_RETRIEVAL_K", "6")
    monkeypatch.setenv("MNEMIQ_PRINCIPAL", "alice")
    monkeypatch.setenv("MNEMIQ_ROLES", "analyst,admin")
    monkeypatch.setenv("MNEMIQ_TENANT", "acme")
    s = Settings.from_env()
    assert s.guided_sql is True and s.assertive_sql is True
    assert s.retrieval_k == 6
    assert s.principal == "alice" and s.tenant == "acme"
    assert s.roles == "analyst,admin"


def test_folded_field_defaults(monkeypatch):
    for v in ("MNEMIQ_GUIDED_SQL", "MNEMIQ_ASSERTIVE_SQL", "MNEMIQ_RETRIEVAL_K"):
        monkeypatch.delenv(v, raising=False)
    s = Settings.from_env()
    assert s.guided_sql is False and s.assertive_sql is False
    assert s.retrieval_k == 12   # shipped default


def test_env_example_lists_every_field_without_secrets():
    text = Settings.env_example()
    assert "MNEMIQ_LLM_BASE_URL=" in text
    assert "MNEMIQ_VERIFY_THRESHOLD=0.5" in text
    assert "MNEMIQ_RETRIEVAL_K=12" in text
    assert "MNEMIQ_VERIFY=" in text            # the tri-state alias, not MNEMIQ_VERIFY_OVERRIDE
    assert "MNEMIQ_VERIFY_OVERRIDE" not in text
    for line in text.splitlines():
        if line.startswith("MNEMIQ_") and ("API_KEY" in line or "DSN" in line):
            assert line.split("#")[0].strip().endswith("=")  # secret value blank


def test_no_adhoc_retrieval_k_reads():
    assert _src_files_reading("MNEMIQ_RETRIEVAL_K") == []


def test_no_adhoc_identity_reads():
    for var in ("MNEMIQ_PRINCIPAL", "MNEMIQ_ROLES", "MNEMIQ_TENANT"):
        assert _src_files_reading(var) == [], var


def test_no_adhoc_generation_flag_reads():
    for var in ("MNEMIQ_GUIDED_SQL", "MNEMIQ_ASSERTIVE_SQL",
                "MNEMIQ_ENRICH_FACTS", "MNEMIQ_ENRICH_EXAMPLES"):
        assert _src_files_reading(var) == [], var


def test_cli_config_prints_template(capsys):
    from mnemiq.cli import main
    rc = main(["config", "example"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MNEMIQ_LLM_BASE_URL=" in out and "MNEMIQ_RETRIEVAL_K=12" in out


def test_dictionary_path_reads_env(monkeypatch):
    monkeypatch.setenv("MNEMIQ_DICTIONARY_PATH", "/tmp/dict.json")
    assert Settings().dictionary_path == "/tmp/dict.json"


def test_dictionary_path_defaults_none(monkeypatch):
    monkeypatch.delenv("MNEMIQ_DICTIONARY_PATH", raising=False)
    assert Settings().dictionary_path is None


def test_ontology_records_path_defaults_none(monkeypatch):
    from mnemiq.config import Settings

    monkeypatch.delenv("MNEMIQ_ONTOLOGY_RECORDS_PATH", raising=False)
    assert Settings().ontology_records_path is None
    monkeypatch.setenv("MNEMIQ_ONTOLOGY_RECORDS_PATH", "/x/records.json")
    assert Settings().ontology_records_path == "/x/records.json"

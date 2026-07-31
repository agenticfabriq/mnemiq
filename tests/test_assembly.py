from types import SimpleNamespace

import duckdb

from mnemiq.assembly import build_components
from mnemiq.config import Settings


def _settings() -> Settings:
    return Settings(llm_base_url="http://x", llm_api_key="k", llm_model="m",
                    guided_sql=True, assertive_sql=True)


def test_build_components_wires_generator_from_adapter_dialect_and_settings():
    con = duckdb.connect(":memory:")
    kit = build_components(_settings(), SimpleNamespace(dialect="sqlite"), con)
    assert kit.generator._dialect == "sqlite"
    assert kit.generator._guided_sql is True
    assert kit.generator._assertive is True


def test_build_components_shares_one_client_across_all_llm_components():
    con = duckdb.connect(":memory:")
    kit = build_components(_settings(), SimpleNamespace(dialect="duckdb"), con)
    assert kit.generator._client is kit.client
    assert kit.selector._client is kit.client

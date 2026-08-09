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


def test_guided_decoding_uses_response_format_not_the_vllm_extension():
    """The flag has to work on the endpoint we actually ship against.

    `guided_json` is a vLLM extension; the OpenAI-compatible endpoint answers it with
    `400 Unknown parameter`, so the request FAILS rather than degrading -- the flag was
    unusable on the frontier model, not merely inert. `response_format` is understood by
    both. Measured against both backends before the switch.
    """
    from mnemiq.generate.generator import _guided_extra_body

    assert _guided_extra_body(False) is None

    body = _guided_extra_body(True)
    assert "guided_json" not in body, "the vLLM extension is a 400 on OpenAI-compatible endpoints"
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert "strict" not in fmt["json_schema"], (
        "strict additionally demands additionalProperties:false and every property required, "
        "which the optional `reason` violates -- also a 400"
    )
    schema = fmt["json_schema"]["schema"]
    assert schema["required"] == ["sql"]
    assert schema["properties"]["sql"]["minLength"] == 1, "the non-empty guarantee is the point"

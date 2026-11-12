"""Superset mode's settings and the rules a deployment must meet before it starts in that mode.

Superset mode answers each question through Superset's MCP service as the signed-in person, holding no database
credential of its own. Every configuration that mode cannot honour refuses to start, with a message naming the setting
to change; with no Superset URL set, nothing here applies.
"""

import logging
import os

import pytest

from mnemiq.auth.config import auth_config_from_settings
from mnemiq.config import Settings

URL = "https://superset.example.com/mcp"
OIDC = {"oidc_issuer": "https://idp.example/realms/r", "oidc_audience": "mnemiq-api", "tenant": "acme"}
SUPERSET = {"superset_mcp_url": URL, "superset_database_id": 1, **OIDC}


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment: a variable set in the shell running the tests must not decide them."""
    for name in [n for n in os.environ if n.startswith("MNEMIQ_") and n != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


def _superset_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("mnemiq.superset")]


def _check(**kw) -> None:
    from mnemiq.superset.settings import check_superset_settings

    settings = Settings(**kw)
    return check_superset_settings(settings, auth_config_from_settings(settings))


def _refused(*names: str, **kw) -> str:
    """The refusal `kw` gets, asserted to name each setting in `names`."""
    from mnemiq.superset.settings import SupersetConfigError

    with pytest.raises(SupersetConfigError) as exc:
        _check(**kw)
    message = str(exc.value)
    for name in names:
        assert name in message, message
    return message


def test_url_without_database_id():
    _refused("MNEMIQ_SUPERSET_DATABASE_ID", **(SUPERSET | {"superset_database_id": None}))


def test_database_id_without_url():
    """Half a Superset configuration would otherwise run in direct mode, reading the source with mnemiq's own
    credential, which is what setting Superset up was meant to avoid."""
    _refused("MNEMIQ_SUPERSET_MCP_URL", "MNEMIQ_SUPERSET_DATABASE_ID", superset_database_id=1)


@pytest.mark.parametrize("schema", ["public; drop", "", "2026_sales", "sales.public", 'pub"lic', "sales-2026",
                                    "sales\n", "public "])
def test_schema_must_be_a_plain_identifier(schema):
    _refused("MNEMIQ_SUPERSET_SCHEMA", **SUPERSET, superset_schema=schema)


@pytest.mark.parametrize("schema", ["sales_2026", "public", "_staging", "Sales"])
def test_a_plain_identifier_is_accepted_as_the_schema(schema):
    assert _check(**SUPERSET, superset_schema=schema) is None


def test_needs_an_identity_provider():
    from mnemiq.superset.settings import SupersetConfigError, check_superset_settings

    # No verifier at all: the settings identity would serve every request, with no token to forward.
    _refused("MNEMIQ_OIDC_ISSUER", superset_mcp_url=URL, superset_database_id=1)
    # A verifier, but not an identity provider's: Open WebUI's signed header alone.
    owui = Settings(superset_mcp_url=URL, superset_database_id=1, tenant="acme",
                    openwebui_jwt_secret="s" * 32, openwebui_connection_key_header="X-Mnemiq-Connection",
                    openwebui_connection_key="k" * 32)
    auth = auth_config_from_settings(owui)
    assert auth is not None and auth.oidc is None
    with pytest.raises(SupersetConfigError) as exc:
        check_superset_settings(owui, auth)
    assert "MNEMIQ_OIDC_ISSUER" in str(exc.value)


def test_refuses_mcp_over_http():
    _refused("MNEMIQ_MCP_TRANSPORT", **(SUPERSET | {"oidc_audience": "https://mnemiq.example/mcp"}),
             mcp_transport="http")


@pytest.mark.parametrize("door, name", [
    ({"openwebui_jwt_secret": "s" * 32, "openwebui_connection_key_header": "X-Mnemiq-Connection",
      "openwebui_connection_key": "k" * 32}, "MNEMIQ_OPENWEBUI_JWT_SECRET"),
    ({"openwebui_audience": "open-webui-api"}, "MNEMIQ_OPENWEBUI_AUDIENCE"),
])
def test_refuses_the_open_webui_doors(door, name):
    _refused(name, **SUPERSET, **door)


def test_refuses_writes():
    _refused("MNEMIQ_WRITE_ENABLED", **SUPERSET, write_enabled=True)


def test_refuses_a_sources_manifest(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text("[]")
    _refused("MNEMIQ_SOURCES_PATH", **SUPERSET, sources_path=str(manifest))


@pytest.mark.parametrize("phases, named, not_named", [
    ({"enrich_facts": True}, ["MNEMIQ_ENRICH_FACTS"], ["MNEMIQ_ENRICH_EXAMPLES"]),
    ({"enrich_examples": True}, ["MNEMIQ_ENRICH_EXAMPLES"], ["MNEMIQ_ENRICH_FACTS"]),
    ({"enrich_facts": True, "enrich_examples": True}, ["MNEMIQ_ENRICH_FACTS", "MNEMIQ_ENRICH_EXAMPLES"], []),
])
def test_refuses_facts_and_examples_enrichment(phases, named, not_named):
    message = _refused(*named, **SUPERSET, **phases)
    for name in not_named:
        assert name not in message, message


@pytest.mark.parametrize("url", [
    "ftp://superset.example.com/mcp",               # not http or https
    "superset.example.com/mcp",                     # no scheme
    "http:///mcp",                                  # no host
    "http://superset:notaport/mcp",                 # a port that is not a number
    "http://superset:0/mcp",                        # port 0
    "http://admin@superset:5008/mcp",               # a user name
    "http://superset:5008/mcp?database=1",          # a query
    "http://superset:5008/mcp?",                    # an empty query
    "http://superset:5008/mcp#tools",               # a fragment
    "http://superset:5008/mcp#",                    # an empty fragment
    " http://superset:5008/mcp",                    # whitespace a URL parser drops without a word
    "http://super\tset:5008/mcp",                   # a tab a URL parser drops without a word
    "http://supersét.example.com/mcp",              # outside ASCII
])
def test_url_must_be_http_or_https_without_userinfo_query_or_fragment(url):
    _refused("MNEMIQ_SUPERSET_MCP_URL", **(SUPERSET | {"superset_mcp_url": url}))


def test_a_refused_url_is_named_without_its_credentials():
    """The refusal reaches the console and whatever log the operator keeps."""
    message = _refused("MNEMIQ_SUPERSET_MCP_URL",
                       **(SUPERSET | {"superset_mcp_url": "https://admin:pa55-w0rd-x@superset.example.com/mcp"}))
    assert "pa55-w0rd-x" not in message
    message = _refused("MNEMIQ_SUPERSET_MCP_URL",
                       **(SUPERSET | {"superset_mcp_url": "https://superset.example.com/mcp?token=t0ken-x"}))
    assert "t0ken-x" not in message


@pytest.mark.parametrize("url", ["https://superset.example.com/mcp", "https://superset.example.com:8443/mcp",
                                 "http://127.0.0.1:5008/mcp", "http://[::1]:5008/mcp", "http://localhost:5008/mcp",
                                 "http://superset:5008/mcp"])
def test_an_http_or_https_url_with_a_host_is_accepted(url):
    assert _check(**(SUPERSET | {"superset_mcp_url": url})) is None


def test_a_complete_superset_configuration_passes(caplog):
    settings = Settings(**SUPERSET)
    assert settings.superset_mode is True
    assert settings.superset_schema == "public"
    with caplog.at_level(logging.DEBUG, logger="mnemiq.superset"):
        assert _check(**SUPERSET) is None
    assert _superset_records(caplog) == []


def test_http_to_another_host_is_accepted_with_one_warning(caplog):
    """A Compose service name is the usual case: Superset and mnemiq on one private network, where http is normal.
    Each person's token still crosses that network in the clear, so startup says so once."""
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"):
        assert _check(**(SUPERSET | {"superset_mcp_url": "http://superset:5008/mcp"})) is None
    warnings = _superset_records(caplog)
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "MNEMIQ_SUPERSET_MCP_URL" in warnings[0].getMessage()
    assert "superset" in warnings[0].getMessage()


@pytest.mark.parametrize("url", ["https://superset.example.com/mcp", "http://127.0.0.1:5008/mcp",
                                 "http://[::1]:5008/mcp", "http://localhost:5008/mcp", "http://LOCALHOST:5008/mcp"])
def test_https_or_this_machine_is_accepted_without_a_warning(url, caplog):
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"):
        assert _check(**(SUPERSET | {"superset_mcp_url": url})) is None
    assert _superset_records(caplog) == []


def test_direct_mode_is_not_checked(tmp_path):
    """With no Superset URL nothing here applies, however the rest is set: direct mode starts as it always has."""
    from mnemiq.superset.settings import check_superset_settings

    manifest = tmp_path / "sources.json"
    manifest.write_text("[]")
    everything = Settings(
        superset_schema="public; drop", write_enabled=True, sources_path=str(manifest), enrich_facts=True,
        enrich_examples=True, mcp_transport="http", openwebui_jwt_secret="s" * 32, openwebui_audience="other")
    assert everything.superset_mode is False
    assert check_superset_settings(everything, None) is None
    assert check_superset_settings(Settings(), None) is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_url_is_direct_mode(blank):
    """A sourced copy of `.env.example` sets every unset setting to the empty string, which must read as unset."""
    from mnemiq.superset.settings import check_superset_settings

    settings = Settings(superset_mcp_url=blank)
    assert settings.superset_mode is False
    assert check_superset_settings(settings, None) is None


def test_the_constants_other_modules_share():
    from mnemiq.superset.settings import SUPERSET_DIALECT, SUPERSET_SOURCE_JOB

    assert SUPERSET_DIALECT == "postgres"
    assert SUPERSET_SOURCE_JOB == "superset:source"


def test_superset_url_is_local_only_checked(monkeypatch):
    # The endpoint checks only; which DuckDB extensions this machine holds is test_local_only_extensions.py's.
    monkeypatch.setattr("mnemiq.duckdb_extensions.check_extensions", lambda names: list(names))
    public = Settings(local_only=True, superset_mcp_url="https://superset.example.com/mcp")
    with pytest.raises(RuntimeError) as exc:
        public.assert_local_only()
    assert "superset_mcp_url='https://superset.example.com/mcp'" in str(exc.value)
    # The known-good case: Superset on this network passes.
    Settings(local_only=True, superset_mcp_url="http://10.0.0.5:5008/mcp").assert_local_only()

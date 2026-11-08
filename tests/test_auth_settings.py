"""The settings verified identity reads, the one role normalizer, and where a JWKS is fetched from."""

import pytest

from mnemiq.config import Settings, identity_from_settings


@pytest.fixture(autouse=True)
def _extensions_loaded(monkeypatch):
    # The local-only boot check also loads DuckDB extensions; that half has its own tests.
    monkeypatch.setattr("mnemiq.duckdb_extensions.check_extensions", lambda names: list(names))


def test_normalize_roles_strips_and_drops_empties_without_folding():
    from mnemiq.auth.roles import normalize_roles

    assert normalize_roles([" analyst", "", "Viewer ", "  "]) == ["analyst", "Viewer"]


def test_identity_from_settings_uses_the_normalizer(monkeypatch):
    import mnemiq.auth.roles as roles

    seen = []
    real = roles.normalize_roles
    monkeypatch.setattr(roles, "normalize_roles", lambda values: seen.append(list(values)) or real(values))
    identity = identity_from_settings(Settings(roles="analyst, ,viewer"))
    assert identity.roles == ["analyst", "viewer"]
    assert seen == [["analyst", " ", "viewer"]]


def test_jwks_url_is_derived_from_the_issuer():
    s = Settings(oidc_issuer="https://idp.example/realms/r")
    assert s.oidc_jwks_url == "https://idp.example/realms/r/protocol/openid-connect/certs"
    # A trailing slash on the issuer does not double the separator.
    s = Settings(oidc_issuer="https://idp.example/realms/r/")
    assert s.oidc_jwks_url == "https://idp.example/realms/r/protocol/openid-connect/certs"


def test_a_jwks_file_means_no_jwks_url(tmp_path):
    s = Settings(oidc_issuer="https://idp.example/realms/r", oidc_jwks_file=str(tmp_path / "jwks.json"))
    assert s.oidc_jwks_url is None


def test_an_explicit_jwks_url_is_kept():
    s = Settings(oidc_issuer="https://idp.example/realms/r", oidc_jwks_url="https://keys.example/jwks")
    assert s.oidc_jwks_url == "https://keys.example/jwks"


def test_a_public_jwks_url_is_refused_under_local_only():
    s = Settings(local_only=True, oidc_issuer="https://idp.example/realms/r")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "oidc_jwks_url='https://idp.example/realms/r/protocol/openid-connect/certs'" in str(exc.value)


def test_a_jwks_file_passes_local_only_with_a_public_issuer(tmp_path):
    s = Settings(local_only=True, oidc_issuer="https://idp.example/realms/r",
                 oidc_jwks_file=str(tmp_path / "jwks.json"))
    s.assert_local_only()


def test_secrets_render_empty_in_env_example(monkeypatch):
    monkeypatch.setenv("MNEMIQ_OPENWEBUI_JWT_SECRET", "s3cret-value-that-must-not-print-0001")
    monkeypatch.setenv("MNEMIQ_OPENWEBUI_CONNECTION_KEY", "binding-value-that-must-not-print-01")
    text = Settings.env_example()
    assert "MNEMIQ_OPENWEBUI_JWT_SECRET=  #" in text
    assert "MNEMIQ_OPENWEBUI_CONNECTION_KEY=  #" in text
    assert "must-not-print" not in text


@pytest.mark.parametrize("env, field, value", [
    ("MNEMIQ_OIDC_ISSUER", "oidc_issuer", "https://idp.example/realms/r"),
    ("MNEMIQ_OIDC_AUDIENCE", "oidc_audience", "https://mnemiq.example/mcp"),
    ("MNEMIQ_OIDC_JWKS_FILE", "oidc_jwks_file", "/keys/jwks.json"),
    ("MNEMIQ_OIDC_GROUPS_CLAIM", "oidc_groups_claim", "groups"),
    ("MNEMIQ_OPENWEBUI_AUDIENCE", "openwebui_audience", "mnemiq-openwebui"),
    ("MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER", "openwebui_connection_key_header", "X-Mnemiq-Connection"),
    ("MNEMIQ_IDENTITY_MAP_PATH", "identity_map_path", "/etc/mnemiq/identity-map.json"),
    ("MNEMIQ_MCP_TRANSPORT", "mcp_transport", "http"),
])
def test_each_setting_reads_its_env_var(monkeypatch, env, field, value):
    monkeypatch.setenv(env, value)
    assert getattr(Settings(), field) == value


def test_the_defaults():
    s = Settings()
    assert s.oidc_issuer is None and s.oidc_jwks_url is None
    assert s.oidc_tenant_claim == "tenant"
    assert s.oidc_roles_claim == "realm_access.roles"
    assert s.oidc_groups_claim is None
    assert s.oidc_leeway_s == 30.0
    assert s.openwebui_jwt_header == "X-OpenWebUI-User-Jwt"
    assert s.openwebui_jwt_max_lifetime_s == 300
    assert s.openwebui_user_headers == (
        "X-OpenWebUI-User-Id,X-OpenWebUI-User-Email,X-OpenWebUI-User-Name,X-OpenWebUI-User-Role")
    assert s.mcp_transport == "stdio"


def test_an_unknown_mcp_transport_is_refused():
    with pytest.raises(ValueError):
        Settings(mcp_transport="websocket")

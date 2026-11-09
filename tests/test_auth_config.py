"""Every identity configuration the doors cannot enforce refuses to start, each with a message naming what to change."""

import json

import pytest

from mnemiq.config import Settings

ISSUER = "https://idp.example/realms/r"
SECRET = "s" * 32
KEY = "k" * 32


def _cfg(**kw):
    from mnemiq.auth.config import auth_config_from_settings

    return auth_config_from_settings(Settings(**kw))


def _refused(match: str, **kw) -> str:
    from mnemiq.auth.config import AuthConfigError

    with pytest.raises(AuthConfigError) as exc:
        _cfg(**kw)
    assert match in str(exc.value), str(exc.value)
    return str(exc.value)


IDENTITY_SETTINGS = (
    "tenant", "oidc_issuer", "oidc_audience", "oidc_jwks_url", "oidc_jwks_file", "oidc_tenant_claim",
    "oidc_roles_claim", "oidc_groups_claim", "oidc_leeway_s", "openwebui_audience", "openwebui_jwt_secret",
    "openwebui_jwt_header", "openwebui_jwt_max_lifetime_s", "openwebui_user_headers",
    "openwebui_connection_key_header", "openwebui_connection_key", "identity_map_path", "mcp_transport")
OIDC = {"oidc_issuer": ISSUER, "oidc_audience": "mnemiq-api", "tenant": "acme"}
OWUI = {"openwebui_jwt_secret": SECRET, "openwebui_connection_key_header": "X-Mnemiq-Connection",
        "openwebui_connection_key": KEY, "tenant": "acme"}


def test_no_issuer_means_no_auth_config():
    assert _cfg() is None


def test_a_sourced_env_example_configures_no_verifier(monkeypatch):
    """`.env.example` says "Copy to .env", and a sourced copy sets every unset setting to the empty string: that
    must read as unset, never as an empty secret that turns a door on."""
    import pathlib

    from mnemiq.auth.config import auth_config_from_settings

    example = pathlib.Path(__file__).resolve().parents[1] / ".env.example"
    identity = {f"MNEMIQ_{name.upper()}" for name in IDENTITY_SETTINGS}
    seen = set()
    for line in example.read_text().splitlines():
        name, _, rest = line.partition("=")
        if name in identity:
            monkeypatch.setenv(name, rest.split("  #")[0].strip())
            seen.add(name)
    assert seen == identity
    assert auth_config_from_settings(Settings()) is None
    assert auth_config_from_settings(Settings()) is None


def test_an_oidc_configuration():
    cfg = _cfg(**OIDC)
    assert cfg.tenant == "acme" and cfg.oidc.issuer == ISSUER and cfg.oidc.audience == "mnemiq-api"
    assert cfg.oidc.jwks_url == ISSUER + "/protocol/openid-connect/certs"
    assert cfg.openwebui is None and cfg.binding is None and cfg.mcp_http is False
    assert cfg.user_headers == ("x-openwebui-user-id", "x-openwebui-user-email", "x-openwebui-user-name",
                                "x-openwebui-user-role")


def test_an_open_webui_configuration():
    cfg = _cfg(**OWUI)
    assert cfg.openwebui.secret == SECRET.encode() and cfg.openwebui.max_lifetime_s == 300
    assert cfg.binding == ("x-mnemiq-connection", KEY.encode())
    assert cfg.issuers == {"open-webui"}


def test_tenant_star_starts_with_oidc_alone():
    assert _cfg(**{**OIDC, "tenant": "*"}).tenant == "*"


@pytest.mark.parametrize("tenant", [None, "", "   "])
def test_a_verifier_without_a_tenant_refuses(tenant):
    _refused("MNEMIQ_TENANT", **{**OIDC, "tenant": tenant})
    _refused("MNEMIQ_TENANT", **{**OWUI, "tenant": tenant})


def test_a_short_open_webui_secret_refuses():
    _refused("at least 32 bytes", **{**OWUI, "openwebui_jwt_secret": "s" * 31})


def test_the_signed_header_without_a_binding_refuses():
    _refused("MNEMIQ_OPENWEBUI_CONNECTION_KEY", **{**OWUI, "openwebui_connection_key": None,
                                                   "openwebui_connection_key_header": None})


@pytest.mark.parametrize("header, key", [("X-Mnemiq-Connection", "k" * 31), ("X-Mnemiq-Connection", None),
                                         (None, KEY)])
def test_a_short_or_half_configured_binding_refuses(header, key):
    _refused("MNEMIQ_OPENWEBUI_CONNECTION_KEY",
             **{**OWUI, "openwebui_connection_key_header": header, "openwebui_connection_key": key})


@pytest.mark.parametrize("header", ["Authorization", "authorization", "X-OpenWebUI-User-Jwt",
                                    "x-openwebui-user-email", "X-OpenWebUI-User-Id"])
def test_the_binding_never_shares_an_identity_header(header):
    _refused("MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER", **{**OWUI, "openwebui_connection_key_header": header})


def test_the_binding_never_shares_a_renamed_identity_header():
    _refused("MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER",
             **{**OWUI, "openwebui_jwt_header": "X-Signed-User", "openwebui_connection_key_header": "x-signed-user"})
    _refused("MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER",
             **{**OWUI, "openwebui_user_headers": "X-Plain-User", "openwebui_connection_key_header": "X-Plain-User"})


def test_a_binding_with_no_open_webui_door_refuses():
    _refused("no Open WebUI door", **{**OIDC, "openwebui_connection_key_header": "X-Mnemiq-Connection",
                                      "openwebui_connection_key": KEY})


def test_a_binding_for_option_a_starts():
    cfg = _cfg(**{**OIDC, "openwebui_audience": "mnemiq-openwebui",
                  "openwebui_connection_key_header": "X-Mnemiq-Connection", "openwebui_connection_key": KEY})
    assert cfg.binding == ("x-mnemiq-connection", KEY.encode())
    assert cfg.oidc.openwebui_audience == "mnemiq-openwebui"


def test_the_signed_header_with_any_tenant_refuses():
    _refused("names one tenant", **{**OWUI, "tenant": "*"})


def test_an_issuer_without_an_audience_refuses():
    _refused("MNEMIQ_OIDC_AUDIENCE", **{**OIDC, "oidc_audience": None})


def test_an_open_webui_audience_without_an_issuer_refuses():
    _refused("MNEMIQ_OIDC_ISSUER", **{**OWUI, "openwebui_audience": "mnemiq-openwebui"})


def test_equal_audiences_refuse():
    _refused("must differ", **{**OIDC, "openwebui_audience": "mnemiq-api"})


def test_a_jwks_url_and_a_jwks_file_together_refuse(tmp_path):
    _refused("MNEMIQ_OIDC_JWKS_FILE", **{**OIDC, "oidc_jwks_url": "https://keys.example/jwks",
                                         "oidc_jwks_file": str(tmp_path / "jwks.json")})


def test_mcp_over_http_without_an_issuer_refuses():
    _refused("MNEMIQ_OIDC_ISSUER", mcp_transport="http", tenant="acme")


def test_a_map_with_no_verifier_refuses(tmp_path):
    _refused("MNEMIQ_IDENTITY_MAP_PATH", identity_map_path=str(tmp_path / "map.json"), tenant="acme")


@pytest.mark.parametrize("audience", [
    "mnemiq",  # a bare id is fine while nothing needs an MCP URL
])
def test_a_bare_audience_starts_under_stdio(audience):
    assert _cfg(**{**OIDC, "oidc_audience": audience}).mcp_http is False


@pytest.mark.parametrize("audience", [
    "https://mnemiq.example.com/mcp",
    "https://mnemiq.example.com:8443/mcp",
    "http://127.0.0.1:8123/mcp",
    "http://[::1]:8123/mcp",
])
def test_the_mcp_audience_starts_under_http(audience):
    assert _cfg(**{**OIDC, "oidc_audience": audience, "mcp_transport": "http"}).mcp_http is True


@pytest.mark.parametrize("audience", [
    "mnemiq",
    "http://10.0.0.5/mcp",
    "http://mnemiq.example.com/mcp",
    "http://localhost/mcp",
    "http://localhost.example.com/mcp",
    "http://127.0.0.1.nip.io/mcp",
    "https://h.example/mcp/",
    "https://h.example/mcp?x",
    "https://h.example/mcp?",
    "https://h.example/mcp#x",
    "https://h.example/mcp#",
    "https://h.example/x-mcp",
    "https://h.example/mnemiq/mcp",
    "https://h.example",
    "https:///mcp",
    "https://user@h.example/mcp",
    "https://h.example:notaport/mcp",
    "ftp://h.example/mcp",
])
def test_a_wrong_mcp_audience_refuses_under_http(audience):
    _refused("MNEMIQ_OIDC_AUDIENCE", **{**OIDC, "oidc_audience": audience, "mcp_transport": "http"})


def test_refusals_never_print_a_secret():
    message = _refused("at least 32 bytes", **{**OWUI, "openwebui_jwt_secret": "short-secret-value"})
    assert "short-secret-value" not in message
    message = _refused("MNEMIQ_OPENWEBUI_CONNECTION_KEY", **{**OWUI, "openwebui_connection_key": "short-key-value"})
    assert "short-key-value" not in message


def test_the_error_is_a_clean_exit():
    from mnemiq.auth.config import AuthConfigError

    assert issubclass(AuthConfigError, SystemExit)


def test_a_jwks_file_with_no_usable_key_refuses_to_start(tmp_path):
    for body in ('{"keys": []}', "{broken", '{"keys": [{"kty": "oct", "k": "AAAA"}]}'):
        path = tmp_path / "jwks.json"
        path.write_text(body)
        _refused("MNEMIQ_OIDC_JWKS_FILE", **{**OIDC, "oidc_jwks_file": str(path)})
    _refused("MNEMIQ_OIDC_JWKS_FILE", **{**OIDC, "oidc_jwks_file": str(tmp_path / "missing.json")})


def _keys_the_verifier_cannot_use() -> dict:
    """Keys PyJWT reads but the verifier never checks a signature with, since it pins RS256 and ES256."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from jwt.algorithms import ECAlgorithm, RSAAlgorithm

    from tests.auth_keys import jwk, rsa_key

    p384 = ECAlgorithm.to_jwk(ec.generate_private_key(ec.SECP384R1()).public_key(), as_dict=True)
    return {
        "hs256": {"kty": "oct", "k": "c2VjcmV0LXNlY3JldC1zZWNyZXQtc2VjcmV0LXNlY3JldA", "kid": "hs", "alg": "HS256"},
        "es384": {**p384, "kid": "p384", "use": "sig"},
        "es256-on-p384": {**p384, "kid": "p384-as-es256", "use": "sig", "alg": "ES256"},  # PyJWT builds this one
        "ps256": {**jwk(rsa_key(), "ps"), "alg": "PS256"},
        "rsa-private": {**RSAAlgorithm.to_jwk(rsa_key(), as_dict=True), "kid": "private", "alg": "RS256"},
        "int-kid": {**jwk(rsa_key(), "k1"), "kid": 7},  # a token's `kid` is a string, so it can never name this key
    }


@pytest.mark.parametrize("name", ["hs256", "es384", "es256-on-p384", "ps256", "rsa-private", "int-kid"])
def test_a_jwks_file_of_keys_the_verifier_cannot_use_refuses_to_start(tmp_path, name):
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps({"keys": [_keys_the_verifier_cannot_use()[name]]}))
    _refused("MNEMIQ_OIDC_JWKS_FILE", **{**OIDC, "oidc_jwks_file": str(path)})


def test_a_jwks_file_with_one_usable_key_among_others_starts(tmp_path):
    from tests.auth_keys import jwks_for, rsa_key

    path = tmp_path / "jwks.json"
    path.write_text(json.dumps({"keys": [*_keys_the_verifier_cannot_use().values(), *jwks_for((rsa_key(), "k1"))["keys"]]}))
    assert _cfg(**{**OIDC, "oidc_jwks_file": str(path)}).oidc.jwks_file == str(path)


def test_a_jwks_file_with_a_signing_key_starts(tmp_path):
    from tests.auth_keys import jwks_for, rsa_key

    path = tmp_path / "jwks.json"
    path.write_text(json.dumps(jwks_for((rsa_key(), "k1"))))
    assert _cfg(**{**OIDC, "oidc_jwks_file": str(path)}).oidc.jwks_file == str(path)


def test_an_issuer_named_like_open_webuis_refuses_to_start():
    _refused("MNEMIQ_OIDC_ISSUER", **{**OIDC, "oidc_issuer": "open-webui"})

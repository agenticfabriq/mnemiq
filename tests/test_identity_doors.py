"""One credential per request, judged by the door it arrives at: /v1 and MCP take the identity provider's token for
their audience; /openai/v1 takes Open WebUI's signed header or a token for its own audience, through mnemiq's
connection."""

import json
import logging

import pytest

from mnemiq.config import Settings
from tests.auth_keys import (AUDIENCE, ISSUER, OWUI_AUDIENCE, claims, hs256, jwks_for, mint, owui_claims,
                             rsa_key)

SECRET = "s" * 32
KEY = "k" * 32
BINDING = "x-mnemiq-connection"
JWT = "x-openwebui-user-jwt"


def _doors(tmp_path, rules=None, **settings):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import HttpDoors

    if rules is not None:
        path = tmp_path / "identity-map.json"
        path.write_text(json.dumps(rules))
        settings["identity_map_path"] = str(path)
    cfg = auth_config_from_settings(Settings(**settings))
    return HttpDoors(cfg, fetch=lambda: jwks_for((rsa_key(), "k1")))


OPTION_B = dict(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE, openwebui_jwt_secret=SECRET,
                openwebui_connection_key_header="X-Mnemiq-Connection", openwebui_connection_key=KEY)
OPTION_A = dict(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE, openwebui_audience=OWUI_AUDIENCE)
MAP = {ISSUER: {}, "open-webui": {"sub": {"owui-user-1": ["store1_manager"]}}}


def _bearer(token):
    return {"authorization": [f"Bearer {token}"]}


def _owui(token=None, binding=KEY):
    found = {JWT: [token or hs256(owui_claims(), SECRET.encode())]}
    if binding is not None:
        found[BINDING] = [binding]
    return found


def _refused(call, headers, status, reason):
    from mnemiq.auth.doors import DoorRefusal

    with pytest.raises(DoorRefusal) as exc:
        call(headers)
    assert exc.value.status == status, (exc.value.status, exc.value.reason)
    assert reason in exc.value.reason, exc.value.reason
    return exc.value


# --- Open WebUI's signed header ---------------------------------------------------------------------------------


def test_a_valid_open_webui_token_with_the_binding_yields_the_mapped_identity(tmp_path):
    identity = _doors(tmp_path, MAP, **OPTION_B).openai(_owui())
    assert identity.principal_id == "open-webui:owui-user-1" and identity.roles == ["store1_manager"]


def test_a_valid_open_webui_token_without_the_binding_is_refused(tmp_path):
    _refused(_doors(tmp_path, MAP, **OPTION_B).openai, _owui(binding=None), 401, "mnemiq's connection")


def test_a_wrong_binding_value_is_refused(tmp_path):
    _refused(_doors(tmp_path, MAP, **OPTION_B).openai, _owui(binding="x" * 32), 401, "mnemiq's connection")


def test_a_binding_sent_twice_is_refused(tmp_path):
    headers = {**_owui(), BINDING: [KEY, KEY]}
    _refused(_doors(tmp_path, MAP, **OPTION_B).openai, headers, 401, "")


def test_the_binding_key_and_the_tokens_never_reach_the_log(tmp_path, caplog):
    doors = _doors(tmp_path, MAP, **OPTION_B)
    token = hs256(owui_claims(email="private.person@example.com"), SECRET.encode())
    with caplog.at_level(logging.DEBUG):
        doors.openai(_owui(token))
        for headers in (_owui(token, binding="y" * 32), _owui(token, binding=None)):
            with pytest.raises(Exception):
                doors.openai(headers)
    assert KEY not in caplog.text and "y" * 32 not in caplog.text
    assert token not in caplog.text and "private.person" not in caplog.text


def test_the_signed_header_is_refused_when_not_enabled(tmp_path):
    _refused(_doors(tmp_path, **OPTION_A).openai, _owui(binding=None), 401, "not enabled")


# --- Option A: the identity provider's token for /openai/v1's own audience --------------------------------------


def test_option_a_with_a_binding_needs_the_header(tmp_path):
    doors = _doors(tmp_path, **OPTION_A, openwebui_connection_key_header="X-Mnemiq-Connection",
                   openwebui_connection_key=KEY)
    token = mint(claims(aud=OWUI_AUDIENCE))
    _refused(doors.openai, _bearer(token), 401, "mnemiq's connection")
    assert doors.openai({**_bearer(token), BINDING: [KEY]}).principal_id == "user-1"


def test_option_a_without_a_binding_serves(tmp_path):
    assert _doors(tmp_path, **OPTION_A).openai(_bearer(mint(claims(aud=OWUI_AUDIENCE)))).roles == ["analyst"]


# --- The doors keep their credentials apart (the bypass the commit gate found) ----------------------------------


@pytest.mark.parametrize("binding", [KEY, None])
def test_the_open_webui_header_on_v1_is_refused(tmp_path, binding):
    _refused(_doors(tmp_path, MAP, **OPTION_B).v1, _owui(binding=binding), 401, "/openai/v1 only")


def test_an_open_webui_path_token_on_v1_is_refused(tmp_path):
    _refused(_doors(tmp_path, **OPTION_A).v1, _bearer(mint(claims(aud=OWUI_AUDIENCE))), 401, "audience")


def test_a_v1_token_on_openai_is_refused(tmp_path):
    _refused(_doors(tmp_path, **OPTION_A).openai, _bearer(mint(claims(aud=AUDIENCE))), 401, "audience")


@pytest.mark.parametrize("door", ["v1", "openai"])
def test_a_token_carrying_both_audiences_is_refused_on_each_door(tmp_path, door):
    doors = _doors(tmp_path, **OPTION_A)
    _refused(getattr(doors, door), _bearer(mint(claims(aud=[AUDIENCE, OWUI_AUDIENCE]))), 401, "both")


def test_without_an_open_webui_audience_openai_takes_no_bearer(tmp_path):
    _refused(_doors(tmp_path, MAP, **OPTION_B).openai,
             {**_bearer(mint(claims(aud=AUDIENCE))), BINDING: [KEY]}, 401, "MNEMIQ_OPENWEBUI_AUDIENCE")


def test_v1_takes_its_own_token(tmp_path):
    identity = _doors(tmp_path, **OPTION_A).v1(_bearer(mint(claims())))
    assert identity.principal_id == "user-1" and identity.tenant_id == "acme" and identity.roles == ["analyst"]


def test_an_open_webui_only_deployment_refuses_every_v1_request(tmp_path):
    settings = {k: v for k, v in OPTION_B.items() if not k.startswith("oidc_")}
    _refused(_doors(tmp_path, MAP, **settings).v1, _bearer(mint(claims())), 401, "no identity provider")


# --- Plain headers and two credentials --------------------------------------------------------------------------


@pytest.mark.parametrize("door", ["v1", "openai"])
def test_plain_headers_alone_are_refused_with_the_fallback_reason(tmp_path, door):
    doors = _doors(tmp_path, MAP, **OPTION_B)
    _refused(getattr(doors, door), {"x-openwebui-user-email": ["a@x.example"], BINDING: [KEY]}, 401,
             "unsigned identity header")


@pytest.mark.parametrize("door", ["v1", "openai"])
def test_plain_headers_beside_a_valid_token_are_refused(tmp_path, door):
    doors = _doors(tmp_path, MAP, **OPTION_B)
    _refused(getattr(doors, door), {**_owui(), "x-openwebui-user-id": ["u"]}, 401, "conflicting identity headers")
    _refused(getattr(doors, door), {**_bearer(mint(claims())), "x-openwebui-user-role": ["admin"]}, 401,
             "unsigned identity header")


def test_a_static_key_beside_the_signed_header_is_two_credentials(tmp_path):
    rejected = _refused(_doors(tmp_path, MAP, **OPTION_B).openai, {**_owui(), **_bearer("sk-static-key")}, 401,
                        "two credentials")
    assert "auth type to none" in rejected.reason


def test_an_identity_provider_token_beside_the_signed_header_is_two_credentials(tmp_path):
    doors = _doors(tmp_path, MAP, **{**OPTION_B, "openwebui_audience": OWUI_AUDIENCE})
    rejected = _refused(doors.openai, {**_owui(), **_bearer(mint(claims(aud=OWUI_AUDIENCE)))}, 401,
                        "two credentials")
    assert "system_oauth" in rejected.reason


def test_a_failed_bearer_beside_a_valid_signed_header_is_still_two_credentials(tmp_path):
    """No precedence and no fall-through: the request is refused, never settled by trying the other one."""
    _refused(_doors(tmp_path, MAP, **OPTION_B).openai, {**_owui(), **_bearer("not.a.jwt")}, 401, "two credentials")


# --- Tenant, map and no credential ------------------------------------------------------------------------------


def test_another_tenants_token_on_v1_is_a_403(tmp_path):
    _refused(_doors(tmp_path, **OPTION_A).v1, _bearer(mint(claims(tenant="globex"))), 403, "another tenant")


@pytest.mark.parametrize("door, headers", [("v1", "bearer"), ("openai", "owui")])
def test_an_unreadable_map_is_a_503_on_each_door(tmp_path, door, headers):
    doors = _doors(tmp_path, MAP, **OPTION_B, openwebui_audience=OWUI_AUDIENCE)
    (tmp_path / "identity-map.json").write_text("{broken")
    sent = _bearer(mint(claims(realm_access=None))) if headers == "bearer" else _owui()
    _refused(getattr(doors, door), sent, 503, "identity map unavailable")


@pytest.mark.parametrize("door", ["v1", "openai"])
def test_no_credential_is_a_401(tmp_path, door):
    doors = _doors(tmp_path, MAP, **OPTION_B, openwebui_audience=OWUI_AUDIENCE)
    _refused(getattr(doors, door), {BINDING: [KEY]} if door == "openai" else {}, 401, "no credential")


def test_a_bad_bearer_on_v1_is_a_401(tmp_path):
    _refused(_doors(tmp_path, **OPTION_A).v1, _bearer(mint(claims(aud="another"))), 401, "audience")


# --- Review focus: the header's form, and a credential sent twice -----------------------------------------------


def test_the_bearer_scheme_is_case_insensitive(tmp_path):
    token = mint(claims())
    assert _doors(tmp_path, **OPTION_A).v1({"authorization": [f"bearer {token}"]}).principal_id == "user-1"


@pytest.mark.parametrize("value", ["Bearer", "Bearer ", "Bearer a b", "Basic dXNlcg==", "Token abc", ""])
def test_a_malformed_authorization_header_is_refused_as_such(tmp_path, value):
    _refused(_doors(tmp_path, **OPTION_A).v1, {"authorization": [value]}, 401, "malformed credential")


@pytest.mark.parametrize("headers", [
    {"authorization": ["Bearer a.b.c", "Bearer d.e.f"]},
    {JWT: ["a.b.c", "d.e.f"], BINDING: [KEY]},
])
def test_a_credential_sent_twice_is_refused_as_ambiguous(tmp_path, headers):
    doors = _doors(tmp_path, MAP, **OPTION_B)
    _refused(doors.openai, headers, 401, "ambiguous credential")
    _refused(doors.v1, headers, 401, "ambiguous credential")


def test_every_refusal_is_logged_with_its_reason(tmp_path, caplog):
    doors = _doors(tmp_path, **OPTION_A)
    with caplog.at_level(logging.INFO, logger="mnemiq.auth"), pytest.raises(Exception):
        doors.v1(_bearer(mint(claims(tenant="globex"))))
    assert "another tenant" in caplog.text and "user-1" in caplog.text


def test_the_binding_key_is_compared_in_constant_time():
    """`==` would pass every behavioural test here and leak the key's prefix through timing, so this reads the code:
    the binding check calls hmac.compare_digest, and nothing in the door compares a value with `==` or `!=`."""
    import ast
    import inspect

    from mnemiq.auth import doors

    tree = ast.parse(inspect.getsource(doors.HttpDoors._identity).lstrip().replace("    def", "def", 1))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "compare_digest" and isinstance(n.func.value, ast.Name) and n.func.value.id == "hmac"]
    assert len(calls) == 1
    compared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            for operand in (node.left, *node.comparators):
                if isinstance(operand, ast.Call) and isinstance(operand.func, ast.Name) and operand.func.id == "len":
                    continue  # a count, not a value
                compared |= {name.id for name in ast.walk(operand) if isinstance(name, ast.Name)}
    assert not compared & {"key", "sent"}, compared


def test_v1_needs_the_db_read_scope(tmp_path):
    doors = _doors(tmp_path, **OPTION_A)
    rejected = _refused(doors.v1, _bearer(mint(claims(scope="openid"))), 403, "lacks db.read")
    assert rejected.challenge == 'Bearer error="insufficient_scope", scope="db.read"'
    assert doors.v1(_bearer(mint(claims(scope="openid db.read")))).principal_id == "user-1"


def test_openai_takes_a_bearer_without_db_read_until_its_rule_is_decided(tmp_path):
    """Pinned on purpose: whether /openai/v1's Bearer tokens need db.read, as /v1's do, is decided with the endpoint
    itself, and that decision should change this test rather than pass unnoticed. Open WebUI's signed header carries
    no scopes at all."""
    doors = _doors(tmp_path, **OPTION_A)
    assert doors.openai(_bearer(mint(claims(aud=OWUI_AUDIENCE, scope="openid")))).principal_id == "user-1"

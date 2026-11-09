"""The HTTP API under an issuer: every /v1 route takes the verified caller, /v1/whoami says who that is, and with no
issuer nothing changes."""

import json
import logging

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app
from tests.auth_keys import AUDIENCE, ISSUER, OWUI_AUDIENCE, claims, hs256, jwks_for, mint, owui_claims, rsa_key

SETTINGS_IDENTITY = IdentityContext(tenant_id="local", principal_id="settings-user", roles=["analyst"])


class _RT:
    """A runtime that records whom it served, with `scope` answered by a real access policy when given one."""

    def __init__(self, authz=None):
        self.served: list[IdentityContext] = []
        self._authz = authz

    def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
        self.served.append(identity)
        return AgentAnswer(answer="2.", mode="thinking")

    def scope(self, identity):
        self.served.append(identity)
        tables = sorted(self._authz.grants_for(identity).objects) if self._authz else []
        return {"tables": tables, "starters": []}


def _doors(tmp_path=None, rules=None, **over):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import HttpDoors

    settings = dict(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE)
    settings.update(over)
    if rules is not None:
        path = tmp_path / "identity-map.json"
        path.write_text(json.dumps(rules))
        settings["identity_map_path"] = str(path)
    return HttpDoors(auth_config_from_settings(Settings(**settings)), fetch=lambda: jwks_for((rsa_key(), "k1")))


def _client(rt, doors=None, **kw):
    return TestClient(build_app(rt, SETTINGS_IDENTITY, doors=doors, **kw))


def _auth(**over):
    return {"Authorization": f"Bearer {mint(claims(**over))}"}


ROUTES = [("post", "/v1/ask", {"question": "how many?"}), ("post", "/v1/chat", {"question": "how many?"}),
          ("get", "/v1/schema", None), ("get", "/v1/whoami", None)]


def _call(client, method, path, body, headers=None):
    return getattr(client, method)(path, headers=headers or {}, **({"json": body} if body is not None else {}))


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_v1_route_refuses_without_a_token_under_an_issuer(method, path, body):
    rt = _RT()
    r = _call(_client(rt, _doors()), method, path, body)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["detail"] == "no credential"
    assert rt.served == []


def test_a_refused_chat_is_a_plain_401_not_a_stream():
    r = _client(_RT(), _doors()).post("/v1/chat", json={"question": "q"})
    assert r.status_code == 401 and not r.headers["content-type"].startswith("text/event-stream")


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_v1_route_serves_the_tokens_identity_never_the_settings_one(method, path, body):
    """M9's third criterion. The mutation it kills: falling back to the settings identity."""
    rt = _RT()
    r = _call(_client(rt, _doors()), method, path, body, _auth())
    assert r.status_code == 200
    if path != "/v1/whoami":
        assert [i.principal_id for i in rt.served] == ["user-1"]
        assert all(i != SETTINGS_IDENTITY for i in rt.served)


def test_whoami_returns_the_verified_caller():
    r = _client(_RT(), _doors()).get("/v1/whoami", headers=_auth(sub="user-7"))
    assert r.json() == {"principal_id": "user-7", "tenant_id": "acme", "verified": True, "end_session_url": None}


def test_whoami_returns_the_settings_identity_with_no_issuer():
    assert _client(_RT()).get("/v1/whoami").json() == {"principal_id": "settings-user", "tenant_id": "local",
                                                        "verified": False, "end_session_url": None}


# --- Sign out through the identity provider: whoami carries where it ends ---------------------------------------

END = "https://sso.example/realms/r/protocol/openid-connect/logout?client_id=workbench-proxy"


def test_whoami_returns_the_configured_end_session_url():
    r = _client(_RT(), _doors(oidc_end_session_url=END)).get("/v1/whoami", headers=_auth(sub="user-7"))
    assert r.json()["end_session_url"] == END


def test_whoami_returns_no_end_session_url_when_unset():
    assert _client(_RT(), _doors()).get("/v1/whoami", headers=_auth()).json()["end_session_url"] is None
    assert _client(_RT()).get("/v1/whoami").json()["end_session_url"] is None


def test_whoami_keeps_its_other_fields():
    reply = _client(_RT(), _doors(oidc_end_session_url=END)).get("/v1/whoami", headers=_auth(sub="user-7")).json()
    assert set(reply) == {"principal_id", "tenant_id", "verified", "end_session_url"}
    assert (reply["principal_id"], reply["tenant_id"], reply["verified"]) == ("user-7", "acme", True)


def test_a_signed_out_whoami_is_a_401_that_names_no_end_session_url():
    """Where Sign out ends is told only to a verified caller, and a refusal says nothing of it."""
    r = _client(_RT(), _doors(oidc_end_session_url=END)).get("/v1/whoami")
    assert r.status_code == 401
    assert "sso.example" not in r.text
    assert not any("sso.example" in value for value in r.headers.values())



def test_with_no_issuer_chat_and_schema_serve_the_settings_identity():
    rt = _RT()
    client = _client(rt)
    assert client.get("/v1/schema").status_code == 200
    assert client.post("/v1/chat", json={"question": "q"}).status_code == 200
    assert rt.served and all(i == SETTINGS_IDENTITY for i in rt.served)


def test_with_no_issuer_open_webui_headers_change_nothing():
    rt = _RT()
    r = _client(rt).get("/v1/schema", headers={"X-OpenWebUI-User-Email": "a@x.example"})
    assert r.status_code == 200 and rt.served == [SETTINGS_IDENTITY]


def test_two_users_see_two_scopes(tmp_path):
    """The join: two tokens, mapped to two roles, through a real access policy, list different tables."""
    from mnemiq.authz.grants import FileAuthzProvider

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"roles": {"analyst": ["claim"], "store1_manager": ["store"]}}))
    rules = {ISSUER: {"sub": {"ana": ["analyst"], "sam": ["store1_manager"]}}}
    client = _client(_RT(FileAuthzProvider(str(policy))), _doors(tmp_path, rules))
    ana = client.get("/v1/schema", headers=_auth(sub="ana", realm_access=None)).json()["tables"]
    sam = client.get("/v1/schema", headers=_auth(sub="sam", realm_access=None)).json()["tables"]
    assert ana == ["claim"] and sam == ["store"]


def test_another_tenant_is_a_403():
    r = _client(_RT(), _doors()).get("/v1/schema", headers=_auth(tenant="globex"))
    assert r.status_code == 403 and "www-authenticate" not in r.headers


def test_an_unreadable_map_is_a_503(tmp_path):
    doors = _doors(tmp_path, {ISSUER: {}})
    (tmp_path / "identity-map.json").write_text("{broken")
    r = _client(_RT(), doors).get("/v1/schema", headers=_auth(realm_access=None))
    assert r.status_code == 503 and r.json()["detail"] == "identity map unavailable"


def test_a_credential_sent_twice_over_http_is_ambiguous():
    """Through real HTTP header handling: two Authorization headers reach the door as two values."""
    token = mint(claims())
    client = _client(_RT(), _doors())
    r = client.get("/v1/schema", headers=[("Authorization", f"Bearer {token}"), ("Authorization", f"Bearer {token}")])
    assert r.status_code == 401 and "ambiguous" in r.json()["detail"]


def test_header_names_are_matched_whatever_their_case():
    client = _client(_RT(), _doors())
    assert client.get("/v1/schema", headers={"AUTHORIZATION": f"bearer {mint(claims())}"}).status_code == 200
    r = client.get("/v1/schema", headers={**_auth(), "x-OPENWEBUI-user-ID": "u"})
    assert r.status_code == 401 and "unsigned identity header" in r.json()["detail"]


def test_the_openai_door_dependency_through_http(tmp_path):
    """/openai/v1's routes arrive with the endpoint; its door is reached here through the same dependency."""
    from mnemiq.server.app import caller_dependency

    doors = _doors(tmp_path, {"open-webui": {"sub": {"owui-user-1": ["analyst"]}}},
                   openwebui_audience=OWUI_AUDIENCE, openwebui_jwt_secret="s" * 32,
                   openwebui_connection_key_header="X-Mnemiq-Connection", openwebui_connection_key="k" * 32)
    app = FastAPI()

    @app.get("/openai/v1/probe")
    def probe(who: IdentityContext = Depends(caller_dependency(doors, "openai"))) -> dict:
        return {"principal_id": who.principal_id, "roles": who.roles}

    client = TestClient(app)
    signed = hs256(owui_claims(), b"s" * 32)
    ok = client.get("/openai/v1/probe", headers={"X-OpenWebUI-User-Jwt": signed, "X-Mnemiq-Connection": "k" * 32})
    assert ok.json() == {"principal_id": "open-webui:owui-user-1", "roles": ["analyst"]}
    assert client.get("/openai/v1/probe", headers={"X-OpenWebUI-User-Jwt": signed}).status_code == 401
    both = client.get("/openai/v1/probe", headers={"X-OpenWebUI-User-Jwt": signed, "X-Mnemiq-Connection": "k" * 32,
                                                   "Authorization": "Bearer sk-static"})
    assert both.status_code == 401 and "auth type to none" in both.json()["detail"]


# --- Review focus: what stays open under an issuer --------------------------------------------------------------


def test_healthz_and_the_workbench_serve_without_a_token(tmp_path):
    (tmp_path / "index.html").write_text("<title>mnemiq workbench</title>")
    client = _client(_RT(), _doors(), static_dir=tmp_path)
    assert client.get("/healthz").json() == {"ok": True}
    assert "mnemiq workbench" in client.get("/").text
    for method, path, body in ROUTES:
        assert _call(client, method, path, body).status_code == 401


# --- serve_http ---------------------------------------------------------------------------------------------------


def test_serve_http_refuses_a_bad_identity_config_before_building_the_runtime(monkeypatch):
    import mnemiq.server.app as server_app
    from mnemiq.auth.config import AuthConfigError

    built = []
    monkeypatch.setattr("mnemiq.runtime.build_runtime", lambda s: built.append(s))
    monkeypatch.setattr(server_app, "_run_uvicorn", lambda app, host, port: None)
    with pytest.raises(AuthConfigError):
        server_app.serve_http(Settings(openwebui_jwt_secret="short", tenant="acme"), host="127.0.0.1", port=8080)
    assert built == []


def test_serve_http_passes_doors_under_an_issuer(monkeypatch):
    import mnemiq.server.app as server_app
    from mnemiq.auth.doors import HttpDoors

    seen = {}
    monkeypatch.setattr("mnemiq.runtime.build_runtime", lambda s: _RT())
    monkeypatch.setattr(server_app, "build_app", lambda rt, ident, **kw: seen.update(kw) or "APP")
    monkeypatch.setattr(server_app, "_run_uvicorn", lambda app, host, port: None)
    server_app.serve_http(Settings(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE), host="0.0.0.0",
                          port=8080)
    assert isinstance(seen["doors"], HttpDoors)


def test_serve_http_without_an_issuer_on_a_public_address_warns(monkeypatch, caplog):
    import mnemiq.server.app as server_app

    seen = {}
    monkeypatch.setattr("mnemiq.runtime.build_runtime", lambda s: _RT())
    monkeypatch.setattr(server_app, "build_app", lambda rt, ident, **kw: seen.update(kw) or "APP")
    monkeypatch.setattr(server_app, "_run_uvicorn", lambda app, host, port: None)
    with caplog.at_level(logging.WARNING):
        server_app.serve_http(Settings(), host="0.0.0.0", port=8080)
    assert seen["doors"] is None and "MNEMIQ_OIDC_ISSUER" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        server_app.serve_http(Settings(), host="127.0.0.1", port=8080)
    assert "MNEMIQ_OIDC_ISSUER" not in caplog.text


def test_serve_http_warns_about_a_mapped_role_the_policy_lacks(monkeypatch, caplog, tmp_path):
    import mnemiq.server.app as server_app
    from mnemiq.authz.grants import FileAuthzProvider

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"roles": {"analyst": ["claim"]}}))
    rt = _RT()
    rt.authz = FileAuthzProvider(str(policy))
    path = tmp_path / "identity-map.json"
    path.write_text(json.dumps({ISSUER: {"sub": {"u": ["anlyst"]}}}))
    monkeypatch.setattr("mnemiq.runtime.build_runtime", lambda s: rt)
    monkeypatch.setattr(server_app, "_run_uvicorn", lambda app, host, port: None)
    with caplog.at_level(logging.WARNING, logger="mnemiq.auth"):
        server_app.serve_http(Settings(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE,
                                       identity_map_path=str(path)), host="127.0.0.1", port=8080)
    assert "anlyst" in caplog.text


# --- /v1 needs the db.read scope (the owner's decision, 2026-11-08) ----------------------------------------------


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_v1_route_refuses_a_token_without_db_read(method, path, body):
    """A 403, not a 401: the caller is signed in and not permitted, so the workbench must not loop on sign-in."""
    rt = _RT()
    r = _call(_client(rt, _doors()), method, path, body, _auth(scope="openid profile"))
    assert r.status_code == 403
    assert r.json()["detail"] == "insufficient scope: this token lacks db.read"
    assert r.headers["www-authenticate"] == 'Bearer error="insufficient_scope", scope="db.read"'
    assert rt.served == []


@pytest.mark.parametrize("scope", [None, "", "db.reader", "db.write", ["db.read"]])
def test_only_a_space_separated_scope_naming_db_read_passes(scope):
    r = _client(_RT(), _doors()).get("/v1/schema", headers=_auth(scope=scope))
    assert r.status_code == 403


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_v1_route_serves_a_token_with_db_read(method, path, body):
    r = _call(_client(_RT(), _doors()), method, path, body, _auth(scope="openid db.read profile"))
    assert r.status_code == 200

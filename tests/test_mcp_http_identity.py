"""MCP over streamable HTTP: a real server on loopback, the SDK's own client, and each request judged by its own token.

Runs on mcp 1.x (the lock) and 2.x (the mcp-2x workflow), so it touches the SDK only through names both share.
"""

import json
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager

import anyio
import pytest
import uvicorn

from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from tests.auth_keys import AUDIENCE as V1_AUDIENCE, ISSUER, OWUI_AUDIENCE, claims, hs256, jwks_for, mint, owui_claims
from tests.auth_keys import rsa_key

SETTINGS_IDENTITY = IdentityContext(tenant_id="local", principal_id="settings-user", roles=["analyst"])
TABLES = {"analyst": ["claim"], "store1_manager": ["store"]}


class _RT:
    """Lists the tables of the identity's roles, so two identities are told apart by what they see."""

    def __init__(self):
        self.served: list[IdentityContext] = []

    def schema(self, identity):
        self.served.append(identity)
        return sorted(t for role in identity.roles for t in TABLES.get(role, []))

    def write(self, sql, identity):
        raise RuntimeError("reached the write path")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(port, **over):
    from mnemiq.auth.config import auth_config_from_settings

    settings = dict(tenant="acme", oidc_issuer=ISSUER, oidc_audience=f"http://127.0.0.1:{port}/mcp",
                    openwebui_audience=OWUI_AUDIENCE, mcp_transport="http")
    settings.update(over)
    return auth_config_from_settings(Settings(**settings))


def _fetch():
    return jwks_for((rsa_key(), "k1"))


@contextmanager
def _serving(rt=None, **over):
    """The whole HTTP app, MCP mounted, served by uvicorn on a loopback port; yields (base URL, audience, rt)."""
    from mnemiq.auth.doors import HttpDoors
    from mnemiq.mcp.server import mcp_http_app
    from mnemiq.server.app import build_app

    port = _free_port()
    cfg = _config(port, **over)
    rt = rt or _RT()
    app = build_app(rt, SETTINGS_IDENTITY, doors=HttpDoors(cfg, fetch=_fetch),
                    mcp_app=mcp_http_app(rt, cfg, fetch=_fetch))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port}/mcp", rt
    finally:
        server.should_exit = True
        thread.join(10)


def _token(audience, **over):
    return mint(claims(aud=audience, **over))


class _Bearer:
    """An auth whose token the test can change between two requests of one client."""

    def __init__(self, token):
        self.token = token


def _client(bearer: _Bearer | None, extra_headers=None):
    """The SDK's own HTTP client factory, whichever httpx flavour this mcp uses."""
    import importlib

    from mcp.shared._httpx_utils import create_mcp_http_client

    hx = importlib.import_module(type(create_mcp_http_client()).__module__.split(".")[0])

    class Auth(hx.Auth):
        def auth_flow(self, request):
            if bearer is not None:
                request.headers["Authorization"] = f"Bearer {bearer.token}"
            yield request

    return create_mcp_http_client(headers=extra_headers, auth=Auth())


@asynccontextmanager
async def _session(url, bearer, extra_headers=None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url, http_client=_client(bearer, extra_headers)) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session


def _is_error(result) -> bool:
    return getattr(result, "is_error", getattr(result, "isError", None)) is True


def _text(result) -> str:
    return result.content[0].text


def _tables(url, token) -> list[str]:
    async def run():
        async with _session(url, _Bearer(token)) as session:
            result = await session.call_tool("get_schema", {})
            assert not _is_error(result), _text(result)
            return json.loads(_text(result))["tables"]

    return anyio.run(run)


def _status(url, headers=None, method="post", path=None) -> tuple[int, dict]:
    import httpx

    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    with httpx.Client() as client:
        r = client.request(method.upper(), path or url, json=body if method == "post" else None,
                           headers={"accept": "application/json, text/event-stream", **(headers or {})})
        return r.status_code, dict(r.headers)


def _mapped(tmp_path):
    path = tmp_path / "identity-map.json"
    path.write_text(json.dumps({ISSUER: {"sub": {"ana": ["analyst"], "sam": ["store1_manager"]}}}))
    return str(path)


def test_two_tokens_for_two_roles_see_two_scopes(tmp_path):
    """M9's second criterion, the join: two requests carrying two tokens get their own role's tables."""
    with _serving(identity_map_path=_mapped(tmp_path)) as (base, aud, rt):
        assert _tables(aud, _token(aud, sub="ana", realm_access=None)) == ["claim"]
        assert _tables(aud, _token(aud, sub="sam", realm_access=None)) == ["store"]
        assert SETTINGS_IDENTITY not in rt.served


def test_a_token_changed_between_two_requests_is_judged_by_the_second(tmp_path):
    """Stateless: a session that kept the token it opened with would list ana's tables for sam's request."""
    with _serving(identity_map_path=_mapped(tmp_path)) as (base, aud, rt):
        async def run():
            bearer = _Bearer(_token(aud, sub="ana", realm_access=None))
            async with _session(aud, bearer) as session:
                first = json.loads(_text(await session.call_tool("get_schema", {})))["tables"]
                bearer.token = _token(aud, sub="sam", realm_access=None)
                second = json.loads(_text(await session.call_tool("get_schema", {})))["tables"]
                return first, second

        assert anyio.run(run) == (["claim"], ["store"])


def test_no_token_is_a_401_naming_the_resource_metadata():
    with _serving() as (base, aud, rt):
        status, headers = _status(aud)
        assert status == 401
        assert "resource_metadata" in headers["www-authenticate"]
        assert f"{base}/.well-known/oauth-protected-resource/mcp" in headers["www-authenticate"]
        assert rt.served == []


@pytest.mark.parametrize("make", [
    lambda aud: {"X-OpenWebUI-User-Jwt": hs256(owui_claims(), b"s" * 32)},
    lambda aud: {"Authorization": f"Bearer {_token([aud, OWUI_AUDIENCE])}"},
    lambda aud: {"Authorization": f"Bearer {_token(OWUI_AUDIENCE)}"},
    lambda aud: {"Authorization": f"Bearer {_token(V1_AUDIENCE)}"},
    lambda aud: {"Authorization": f"Bearer {_token(aud, tenant='globex')}"},
    lambda aud: {"Authorization": f"Bearer {_token(aud, realm_access='admin')}"},
])
def test_every_other_credential_is_a_401(make):
    with _serving() as (base, aud, rt):
        assert _status(aud, make(aud))[0] == 401
        assert rt.served == []


def test_a_valid_token_reaches_the_tools():
    with _serving() as (base, aud, rt):
        assert _tables(aud, _token(aud)) == ["claim"]


def test_get_schema_needs_the_read_scope():
    with _serving() as (base, aud, rt):
        async def run():
            async with _session(aud, _Bearer(_token(aud, scope="openid"))) as session:
                return await session.call_tool("get_schema", {})

        result = anyio.run(run)
        assert _is_error(result) and "db.read" in _text(result)
        assert rt.served == []


def test_db_write_needs_the_write_scope():
    with _serving() as (base, aud, rt):
        async def run(scope):
            async with _session(aud, _Bearer(_token(aud, scope=scope))) as session:
                return await session.call_tool("db_write", {"sql": "DELETE FROM claim"})

        refused = anyio.run(run, "openid db.read")
        assert _is_error(refused) and "db.write" in _text(refused)
        reached = anyio.run(run, "openid db.read db.write")
        assert _is_error(reached) and "db.write" not in _text(reached)  # past the scope, into the (stub) write path


def test_the_resource_metadata_names_the_audience():
    import httpx

    with _serving() as (base, aud, rt):
        r = httpx.get(f"{base}/.well-known/oauth-protected-resource/mcp")
        assert r.status_code == 200
        assert r.json()["resource"] == aud
        assert r.json()["authorization_servers"] == [ISSUER] or r.json()["authorization_servers"] == [ISSUER + "/"]


def test_the_door_is_at_mcp_not_mcp_mcp():
    with _serving() as (base, aud, rt):
        assert _status(aud, path=f"{base}/mcp/mcp")[0] == 404


def test_the_v1_routes_still_serve_beside_mcp():
    import httpx

    with _serving() as (base, aud, rt):
        assert httpx.get(f"{base}/healthz").json() == {"ok": True}
        assert httpx.get(f"{base}/v1/whoami").status_code == 401


def test_the_mcp_app_is_stateless():
    from mnemiq.mcp.server import mcp_http_settings

    assert mcp_http_settings(_config(8123))["stateless_http"] is True


def test_dns_rebinding_protection_allows_the_audiences_host_only():
    from mnemiq.mcp.server import mcp_http_settings

    security = mcp_http_settings(_config(8123))["transport_security"]
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["127.0.0.1:8123"]
    assert security.allowed_origins == ["http://127.0.0.1:8123"]


def test_a_foreign_host_header_is_refused():
    with _serving() as (base, aud, rt):
        status, _ = _status(aud, {"Authorization": f"Bearer {_token(aud)}", "Host": "evil.example"})
        assert status in (400, 421)


def test_serve_refuses_the_http_transport():
    from mnemiq.mcp.server import serve

    with pytest.raises(SystemExit, match="mnemiq serve --http"):
        serve(Settings(mcp_transport="http", tenant="acme", oidc_issuer=ISSUER,
                       oidc_audience="https://mnemiq.example.com/mcp"))


def test_the_resolver_refuses_without_a_verified_token():
    """Outside a verified request the per-call resolver refuses; it has no identity of its own to fall back to. The
    transport refuses such a request first, so only a direct call reaches this."""
    from mnemiq.mcp.server import _mcp_classes, http_identity

    _, tool_error = _mcp_classes()
    with pytest.raises(tool_error, match="no verified token"):
        http_identity(_config(8123))("db.read")


# --- The same header rules as /v1, in front of the SDK's own check (final review) --------------------------------


def _raw(url, headers: list[tuple[str, str]]) -> int:
    import httpx

    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    with httpx.Client() as client:
        sent = [("accept", "application/json, text/event-stream"), *headers]
        return client.post(url, json=body, headers=sent).status_code


@pytest.mark.parametrize("extra", [
    [("X-OpenWebUI-User-Id", "u")],
    [("X-OpenWebUI-User-Email", "a@x.example")],
    [("X-OpenWebUI-User-Jwt", "a.b.c")],
    [("Authorization", "Bearer not-a-token")],
])
def test_a_valid_token_beside_another_identity_header_is_refused(extra):
    """The SDK reads the first Authorization and nothing else, so this screen is what refuses the rest on /mcp."""
    with _serving() as (base, aud, rt):
        valid = [("Authorization", f"Bearer {_token(aud)}")]
        assert _raw(aud, valid) == 200
        assert _raw(aud, valid + extra) == 401
        assert rt.served == []


def test_a_malformed_authorization_is_refused_on_mcp():
    with _serving() as (base, aud, rt):
        assert _raw(aud, [("Authorization", "Basic dXNlcg==")]) == 401


def test_the_screen_names_its_reason():
    import httpx

    with _serving() as (base, aud, rt):
        r = httpx.post(aud, headers={"Authorization": f"Bearer {_token(aud)}", "X-OpenWebUI-User-Id": "u",
                                     "accept": "application/json, text/event-stream"}, json={})
        assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer")
        assert "unsigned identity header" in r.json()["error_description"]


def test_the_metadata_route_is_not_screened():
    import httpx

    with _serving() as (base, aud, rt):
        r = httpx.get(f"{base}/.well-known/oauth-protected-resource/mcp", headers={"X-OpenWebUI-User-Id": "u"})
        assert r.status_code == 200

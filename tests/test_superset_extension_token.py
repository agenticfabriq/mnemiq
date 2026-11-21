"""The Superset extension's token route: the pure minting code and the handler behind the Flask route. Neither
module imports Flask or Superset, so these run in `unit` with no install."""

import json
import os
import time
from types import SimpleNamespace

import jwt
import pytest

from mnemiq.config import Settings
from tests.auth_keys import AUDIENCE, ISSUER, jwks_for, rsa_key
from tests.superset_extension_path import BACKEND_SRC, use_extension_source

use_extension_source()

ASK = {"db.read", "superset:sqllab:write", "superset:dataset:read"}  # what the Superset mode asks with
BUILD = ASK | {"superset:chart:read", "superset:chart:write", "superset:dashboard:read",
               "superset:dashboard:write", "superset:dataset:write"}


def _pem(tmp_path, mode=0o600, name="mint.pem"):
    from cryptography.hazmat.primitives import serialization

    path = tmp_path / name
    path.write_bytes(rsa_key().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.NoEncryption()))
    os.chmod(path, mode)
    return path


def _mapping(tmp_path, **over):
    base = {"MNEMIQ_TOKEN_KEY_FILE": str(_pem(tmp_path)), "MNEMIQ_TOKEN_KID": "k1", "MNEMIQ_TOKEN_ISSUER": ISSUER,
            "MNEMIQ_TOKEN_AUDIENCE": AUDIENCE, "MNEMIQ_TOKEN_TENANT": "acme"}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def _cfg(tmp_path, **over):
    from agenticfabriq.mnemiq.token import SigningConfig

    return SigningConfig.from_mapping(_mapping(tmp_path, **over))


def _decode(token):
    return jwt.decode(token, rsa_key().public_key(), algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER)


def test_mint_claims(tmp_path):
    from agenticfabriq.mnemiq.token import mint

    now = 1_800_000_000.0
    token = mint(_cfg(tmp_path), username="ana", purpose="ask", now=now)
    assert jwt.get_unverified_header(token) == {"alg": "RS256", "typ": "JWT", "kid": "k1"}
    found = jwt.decode(token, rsa_key().public_key(), algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER,
                       options={"verify_exp": False, "verify_nbf": False, "verify_iat": False})
    assert found["iss"] == ISSUER and found["aud"] == AUDIENCE
    assert found["sub"] == "ana" and found["preferred_username"] == "ana"
    assert found["iat"] == int(now) and found["exp"] == int(now) + 300
    # Superset's MCP verifier refuses a token whose `nbf` is after its own clock, with no leeway
    # (`superset/mcp_service/jwt_verifier.py`, the not-before check): `nbf` is a few seconds back, so an MCP process
    # whose clock is a little behind the web server's still takes a token minted the moment before.
    assert found["nbf"] == int(now) - 5
    assert found["tenant"] == "acme" and found["jti"]
    again = mint(_cfg(tmp_path), username="ana", purpose="ask", now=now)
    assert jwt.decode(again, options={"verify_signature": False})["jti"] != found["jti"]


def test_lifetime_is_capped_at_300(tmp_path):
    from agenticfabriq.mnemiq.token import mint

    cfg = _cfg(tmp_path, MNEMIQ_TOKEN_LIFETIME_S="9999")
    assert cfg.lifetime_s == 300
    found = jwt.decode(mint(cfg, username="ana", purpose="ask", now=time.time()), options={"verify_signature": False})
    assert found["exp"] - found["iat"] == 300
    assert _cfg(tmp_path, MNEMIQ_TOKEN_LIFETIME_S="60").lifetime_s == 60
    assert _cfg(tmp_path).lifetime_s == 300
    with pytest.raises(ValueError):
        _cfg(tmp_path, MNEMIQ_TOKEN_LIFETIME_S="0")


def test_purpose_scopes(tmp_path):
    from agenticfabriq.mnemiq.token import mint

    cfg = _cfg(tmp_path)
    now = time.time()
    ask = _decode(mint(cfg, username="ana", purpose="ask", now=now))["scope"]
    build = _decode(mint(cfg, username="ana", purpose="build", now=now))["scope"]
    assert len(ask.split()) == len(ASK) and set(ask.split()) == ASK
    assert set(build.split()) == BUILD
    with pytest.raises(ValueError):
        mint(cfg, username="ana", purpose="admin", now=now)  # type: ignore[arg-type]


def test_refuses_a_world_readable_key(tmp_path):
    from agenticfabriq.mnemiq.token import SigningConfig

    for mode in (0o644, 0o640, 0o604):
        loose = _pem(tmp_path, mode, name=f"loose-{mode:o}.pem")
        with pytest.raises(ValueError, match="0600"):
            SigningConfig.from_mapping(_mapping(tmp_path, MNEMIQ_TOKEN_KEY_FILE=str(loose)))
    link = tmp_path / "link.pem"
    link.symlink_to(_pem(tmp_path))
    with pytest.raises(ValueError, match="regular file"):
        SigningConfig.from_mapping(_mapping(tmp_path, MNEMIQ_TOKEN_KEY_FILE=str(link)))
    with pytest.raises(ValueError, match="regular file"):
        SigningConfig.from_mapping(_mapping(tmp_path, MNEMIQ_TOKEN_KEY_FILE=str(tmp_path)))
    _cfg(tmp_path)  # the control: a 0600 file is accepted


@pytest.mark.parametrize("missing", ["MNEMIQ_TOKEN_KEY_FILE", "MNEMIQ_TOKEN_KID", "MNEMIQ_TOKEN_ISSUER",
                                     "MNEMIQ_TOKEN_AUDIENCE", "MNEMIQ_TOKEN_TENANT"])
def test_refuses_missing_settings(tmp_path, missing):
    from agenticfabriq.mnemiq.token import SigningConfig

    with pytest.raises(ValueError, match=missing):
        SigningConfig.from_mapping(_mapping(tmp_path, **{missing: None}))
    with pytest.raises(ValueError, match=missing):
        SigningConfig.from_mapping(_mapping(tmp_path, **{missing: "  "}))


def test_token_verifies_with_mnemiqs_door(tmp_path):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import DoorRefusal, HttpDoors

    from agenticfabriq.mnemiq.token import mint

    cfg = auth_config_from_settings(Settings(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE))
    doors = HttpDoors(cfg, fetch=lambda: jwks_for((rsa_key(), "k1")))
    signing = _cfg(tmp_path)

    def headers(token):
        return {"authorization": [f"Bearer {token}"]}

    fresh = mint(signing, username="ana", purpose="build", now=time.time())
    delegation = doors.v1_delegation(headers(fresh))
    assert delegation.identity.tenant_id == "acme"
    assert delegation.token  # the door hands on the very Bearer it verified
    with pytest.raises(DoorRefusal) as exc:
        doors.v1(headers(mint(signing, username="ana", purpose="build", now=time.time() - 1000)))
    assert exc.value.status == 401


def signed_in(username: str = "ana", user_id=7) -> SimpleNamespace:
    """A signed-in user as Flask-AppBuilder's `User` model is one (`flask_appbuilder/security/sqla/models.py`): a
    positive integer `id`, `is_anonymous` False, and no `is_guest_user`."""
    return SimpleNamespace(id=user_id, username=username, is_anonymous=False, is_authenticated=True, is_active=True)


def guest(username: str, **more) -> SimpleNamespace:
    """rc2's embedded guest as `GuestUser` is one (`superset/security/guest_token.py`): `is_guest_user` True,
    authenticated and not anonymous, no `id`, and the username the embedding host wrote into its guest token."""
    return SimpleNamespace(is_guest_user=True, username=username, is_anonymous=False, is_authenticated=True,
                           is_active=True, **more)


def anonymous() -> SimpleNamespace:
    """Nobody signed in, as flask_login's `AnonymousUserMixin` is (`flask_login/mixins.py`): anonymous, no `id`, no
    username. It reaches the route when the Public role holds the route's permission."""
    return SimpleNamespace(is_anonymous=True, is_authenticated=False, is_active=False)


NOT_SIGNED_IN = [
    pytest.param(guest("admin"), id="a guest naming admin"),
    # `SupersetSecurityManager.guest_user_cls` (`superset/security/manager.py`) is a class a deployment's own security
    # manager may replace, so a guest may carry an id: the guest check stands on its own.
    pytest.param(guest("admin", id=1), id="a guest with an id"),
    pytest.param(anonymous(), id="anonymous"),
    pytest.param(SimpleNamespace(username="admin", is_anonymous=False), id="no id"),
    pytest.param(SimpleNamespace(username="admin"), id="nothing says signed in"),
    *(pytest.param(signed_in("admin", bad), id=f"id {bad!r}") for bad in (0, -1, True, "7", None, 7.0)),
]


@pytest.mark.parametrize("user", NOT_SIGNED_IN)
def test_the_route_mints_only_for_a_signed_in_superset_user(tmp_path, user):
    """An embedded guest's username is whatever its guest token says, and the token the route mints would resolve in
    MCP to the real user of that name; nobody signed in has no name at all. So a token is minted only for a user who
    is not a guest, not anonymous and has a positive integer id, and the refusal comes before the body is read."""
    from agenticfabriq.mnemiq.token import NotSignedIn, mint_for

    cfg = _cfg(tmp_path)
    for body in ({"purpose": "build"}, {"purpose": "ask"}, None):
        with pytest.raises(NotSignedIn):
            mint_for(user, body, cfg, time.time())
    # The control: the same name, signed in.
    assert _decode(mint_for(signed_in("admin"), {"purpose": "build"}, cfg, time.time())["token"])["sub"] == "admin"


def test_api_ignores_a_username_in_the_body(tmp_path):
    from agenticfabriq.mnemiq.token import BadRequest, mint_for

    cfg = _cfg(tmp_path)
    user = signed_in("ana")
    answer = mint_for(user, {"purpose": "ask", "username": "root", "sub": "root", "preferred_username": "root"},
                      cfg, time.time())
    assert set(answer) == {"token", "expires_in"} and answer["expires_in"] == 300
    found = _decode(answer["token"])
    assert found["sub"] == found["preferred_username"] == "ana"
    for bad in ({}, {"purpose": "x"}, {"purpose": ["ask"]}, None, []):
        with pytest.raises(BadRequest):
            mint_for(user, bad, cfg, time.time())


API_SOURCE = BACKEND_SRC / "agenticfabriq" / "mnemiq" / "api.py"


def route(monkeypatch, user, body, config, forwarding=None, requests_module=None):
    """What the token route's handler (`MnemiqApi.mint` in `api.py`) responds, as its status and fields, with Flask,
    Flask-AppBuilder and `superset_core` stood in: `g.user` is `user`, the request's JSON `body`, Superset's config
    `config`. Their decorators pass the handler through, so what runs is the handler alone; authentication and CSRF
    are Superset's and the live suite's. `api.py` is loaded from its file under a name of its own, so no stand-in
    outlives the test."""
    import importlib.util
    import sys
    import types

    def passthrough(*args, **kwargs):
        return lambda target: target

    class RestApi:
        def response(self, status, **fields):
            return status, fields

    class Response:
        def __init__(self, body, status=200, content_type=None, headers=None):
            self.body, self.status, self.content_type, self.headers = body, status, content_type, headers or {}

    stood_in = {
        "flask": {"Response": Response, "current_app": SimpleNamespace(config=config), "g": SimpleNamespace(user=user),
                  "request": SimpleNamespace(get_json=lambda silent=False: body, method="GET", content_length=0,
                                             args=SimpleNamespace(to_dict=lambda flat=True: {}),
                                             get_data=lambda cache=False: b""),
                  "stream_with_context": lambda chunks: chunks},
        "flask_appbuilder": {},
        "flask_appbuilder.api": {"expose": passthrough, "permission_name": passthrough, "protect": passthrough,
                                 "safe": lambda handler: handler},
        "superset_core": {}, "superset_core.rest_api": {},
        "superset_core.rest_api.api": {"RestApi": RestApi},
        "superset_core.rest_api.decorators": {"api": passthrough},
    }
    for name, attributes in stood_in.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    if requests_module is not None:
        monkeypatch.setitem(sys.modules, "requests", requests_module)
        # urllib3 (requests' own dependency, so in Superset's image) for its marker telling it to send no header of
        # its own; the live suite shows the real one keeps urllib3's user agent off the wire.
        util = types.ModuleType("urllib3.util")
        util.SKIP_HEADER = SKIP_HEADER
        monkeypatch.setitem(sys.modules, "urllib3", types.ModuleType("urllib3"))
        monkeypatch.setitem(sys.modules, "urllib3.util", util)
    spec = importlib.util.spec_from_file_location("token_route_under_test", API_SOURCE)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    if forwarding == "module":
        return loaded
    if forwarding is not None:
        answer = loaded.MnemiqForward().forward(forwarding)
        return answer.status, json.loads(answer.body), answer.headers
    return loaded.MnemiqApi().mint()


def test_the_route_answers_403_to_anyone_not_signed_in(monkeypatch, tmp_path):
    config = _mapping(tmp_path)
    for user in (guest("admin"), guest("admin", id=1), anonymous()):
        status, fields = route(monkeypatch, user, {"purpose": "build"}, config)
        assert status == 403 and "token" not in fields, (status, fields)
        assert fields["message"] == "the token route mints only for a signed-in Superset user"
    # The controls: a signed-in user gets a token; a bad body is a 400, a missing setting a 503 naming it.
    status, fields = route(monkeypatch, signed_in("ana"), {"purpose": "build"}, config)
    assert status == 200 and _decode(fields["token"])["sub"] == "ana"
    assert route(monkeypatch, signed_in("ana"), {"purpose": "admin"}, config)[0] == 400
    unset = {k: v for k, v in config.items() if k != "MNEMIQ_TOKEN_TENANT"}
    assert route(monkeypatch, signed_in("ana"), {"purpose": "build"}, unset) == (
        503, {"message": "MNEMIQ_TOKEN_TENANT is not set"})
    # A guest is refused before the route reads its settings: it learns nothing of how the route is set up.
    assert route(monkeypatch, guest("admin"), {"purpose": "build"}, unset)[0] == 403


def test_the_forwarder_unconfigured_answers_503_naming_mnemiq_url(monkeypatch, tmp_path):
    """Without `MNEMIQ_URL` the forwarder answers 503 with the sentence naming it, as `{"detail": ...}` like every
    refusal of its own, before anything is minted or sent; the token settings missing, it names that one."""
    from agenticfabriq.mnemiq.forward import FORWARD_NOT_CONFIGURED

    config = _mapping(tmp_path)
    status, body, headers = route(monkeypatch, signed_in("ana"), None, config, forwarding="schema")
    assert (status, body, headers.get("Cache-Control")) == (503, {"detail": FORWARD_NOT_CONFIGURED}, "no-store")
    unset = {k: v for k, v in config.items() if k != "MNEMIQ_TOKEN_TENANT"}
    status, body, _ = route(monkeypatch, signed_in("ana"), None, {**unset, "MNEMIQ_URL": "http://mnemiq:8080"},
                            forwarding="schema")
    assert (status, body) == (503, {"detail": "MNEMIQ_TOKEN_TENANT is not set"})


def test_the_forwarder_refuses_a_guest_before_reading_any_setting(monkeypatch, tmp_path):
    """A guest, or nobody signed in, is told 403 before the forwarder reads its settings: with none set, it still
    learns nothing of how they are set."""
    for user in (guest("admin"), anonymous()):
        status, body, _ = route(monkeypatch, user, None, {}, forwarding="schema")
        assert (status, body) == (403, {"detail": "the token route mints only for a signed-in Superset user"})


# urllib3's value for `urllib3.util.SKIP_HEADER`.
SKIP_HEADER = "@@@SKIP_HEADER@@@"


class _FakeRequests:
    """`requests` as the forwarder's `_send` uses it: a Session whose `request` records its arguments and answers a
    response of `content_type` yielding `chunks` (an exception among them is raised when reached)."""

    class RequestException(Exception):
        pass

    class ConnectionError(RequestException):
        pass

    class Timeout(RequestException):
        pass

    class ConnectTimeout(ConnectionError, Timeout):
        pass

    class ReadTimeout(Timeout):
        pass

    def __init__(self, *, content_type="application/json", chunks=(b'{"ok": true}',), raises=None):
        self.calls: list[dict] = []
        self.closed = 0
        fake = self

        class Answer:
            status_code = 200
            headers = {"Content-Type": content_type}

            def iter_content(self, chunk_size=None):
                for chunk in chunks:
                    if isinstance(chunk, BaseException):
                        raise chunk
                    yield chunk

            def close(self):
                fake.closed += 1

        class Session:
            trust_env = True

            def __init__(self):
                self.headers = {"User-Agent": "python-requests", "Accept-Encoding": "gzip"}

            def request(self, method, url, **kwargs):
                fake.calls.append({"method": method, "url": url, "trust_env": self.trust_env,
                                   "session_headers": dict(self.headers), **kwargs})
                if raises is not None:
                    raise raises
                return Answer()

            def close(self):
                pass

        self.Session = Session


def _outbound(stream=False, total_s=600.0):
    from agenticfabriq.mnemiq.forward import Outbound

    return Outbound(method="POST", url="http://mnemiq/v1/chat", headers={"authorization": "Bearer t", "accept": "x",
                                                                         "content-type": "application/json"},
                    body=b"{}", timeout_s=30.0, stream=stream, total_s=total_s)


def test_send_sends_the_outbound_exactly_and_reads_nothing_from_the_environment(monkeypatch, tmp_path):
    """What mnemiq receives is the outbound's headers and body, no more: no proxy or `.netrc` from the environment
    (`trust_env` off), no redirect followed; each wait bounded by the outbound's wait."""
    fake = _FakeRequests()
    api_module = route(monkeypatch, signed_in("ana"), None, {}, forwarding="module", requests_module=fake)
    found = api_module._send(_outbound())
    [call] = fake.calls
    assert call["headers"] == {**_outbound().headers, "User-Agent": SKIP_HEADER} and call["data"] == b"{}"
    assert call["trust_env"] is False and call["allow_redirects"] is False and call["timeout"] == (5, 30.0)
    assert call["session_headers"] == {}
    assert b"".join(found.chunks) == b'{"ok": true}' and fake.closed == 1


def test_send_reads_where_mnemiq_fails_as_what_it_is(monkeypatch, tmp_path):
    """Not reaching mnemiq is unreachable, its connection timing out included; a wait for its answer, before or while
    the body is read, is a timeout."""
    from agenticfabriq.mnemiq import forward

    for raised, expected in ((_FakeRequests.ConnectTimeout(), forward.Unreachable),
                             (_FakeRequests.ConnectionError(), forward.Unreachable),
                             (_FakeRequests.ReadTimeout(), forward.TimedOut)):
        fake = _FakeRequests(raises=raised)
        api_module = route(monkeypatch, signed_in("ana"), None, {}, forwarding="module", requests_module=fake)
        with pytest.raises(expected):
            api_module._send(_outbound())

    class ReadTimeoutError(Exception):
        pass

    fake = _FakeRequests(chunks=(b'{"o', _FakeRequests.ConnectionError(ReadTimeoutError("read timed out"))))
    api_module = route(monkeypatch, signed_in("ana"), None, {}, forwarding="module", requests_module=fake)
    with pytest.raises(forward.TimedOut):
        api_module._send(_outbound())


def test_send_passes_on_only_json_or_an_event_stream(monkeypatch, tmp_path):
    from agenticfabriq.mnemiq import forward

    fake = _FakeRequests(content_type="text/html")
    api_module = route(monkeypatch, signed_in("ana"), None, {}, forwarding="module", requests_module=fake)
    with pytest.raises(forward.Unreachable):
        api_module._send(_outbound())


def test_send_cuts_a_stream_at_its_total_deadline(monkeypatch, tmp_path):
    """A stream that keeps sending (mnemiq's heartbeat) is cut once the whole answer has taken its total time."""
    fake = _FakeRequests(content_type="text/event-stream", chunks=[b": beat\n\n"] * 100)
    api_module = route(monkeypatch, signed_in("ana"), None, {}, forwarding="module", requests_module=fake)
    clock = iter(float(t) for t in range(1000))
    monkeypatch.setattr(api_module.time, "monotonic", lambda: next(clock))
    found = api_module._send(_outbound(stream=True, total_s=5.0))
    passed = list(found.chunks)
    assert 0 < len(passed) < 100 and fake.closed == 1


def csrf_problems(source: str) -> list[str]:
    """Whether the token route's API class opts out of Flask-AppBuilder's CSRF exemption, read with `ast`: FAB
    exempts an API's blueprint unless the class sets `csrf_exempt = False` (`BaseApi.csrf_exempt` is True), and
    `superset_core`'s `RestApi` does not set it."""
    import ast

    problems = []
    # Every API class the extension mounts (each a `RestApi`): the token route, the forwarder, and any added later.
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
               and any(getattr(base, "id", None) == "RestApi" for base in node.bases)]
    if {node.name for node in classes} < {"MnemiqApi", "MnemiqForward"}:
        problems.append("the token route's or the forwarder's class is missing")
    for node in classes:
        name = node.name
        values = [stmt.value.value for stmt in node.body if isinstance(stmt, ast.Assign)
                  and any(getattr(t, "id", None) == "csrf_exempt" for t in stmt.targets)
                  and isinstance(stmt.value, ast.Constant)]
        if values != [False]:
            problems.append(f"{name}.csrf_exempt is {values or 'not set'}")
    return problems


def test_the_token_route_needs_csrf():
    assert csrf_problems(API_SOURCE.read_text()) == []
    # The planted defect, as the route first shipped: the class does not say, so FAB exempts the route.
    assert csrf_problems(API_SOURCE.read_text().replace("    csrf_exempt = False\n", "", 1)) == [
        "MnemiqApi.csrf_exempt is not set"]
    source = API_SOURCE.read_text()
    forwarder = source.index("class MnemiqForward")
    assert csrf_problems(source[:forwarder] + source[forwarder:].replace("    csrf_exempt = False\n", "", 1)) == [
        "MnemiqForward.csrf_exempt is not set"]
    assert csrf_problems(API_SOURCE.read_text().replace("csrf_exempt = False", "csrf_exempt = True")) == [
        "MnemiqApi.csrf_exempt is [True]", "MnemiqForward.csrf_exempt is [True]"]

"""The Superset extension's forwarder: the pure code behind `/extensions/agenticfabriq/mnemiq/v1/...`, which forwards
a panel's requests to mnemiq as the signed-in person, each with a token minted for that request. No Flask or
Superset import, so these run in `unit` with no install; the Flask wrapper is held by the live tests."""

import logging
import os
from types import SimpleNamespace

import jwt
import pytest

from tests.auth_keys import AUDIENCE, ISSUER, rsa_key
from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import forward  # noqa: E402
from agenticfabriq.mnemiq.forward import (  # noqa: E402
    BODY_TOO_LARGE,
    FORWARD_NOT_CONFIGURED,
    MAX_BODY,
    MNEMIQ_TIMED_OUT,
    MNEMIQ_UNREACHABLE,
    NOT_FORWARDED,
    QUERY_REFUSED,
    ForwardConfig,
    Inbound,
    Outbound,
    forward_for,
)
from agenticfabriq.mnemiq.token import ASK_SCOPES, BUILD_SCOPES, NOT_SIGNED_IN, SigningConfig  # noqa: E402

NOW = 1_800_000_000.0
BUILD_ID = "0f8fad5b-d9cb-469f-a165-70867728950e"


@pytest.fixture
def signing(tmp_path) -> SigningConfig:
    from cryptography.hazmat.primitives import serialization

    path = tmp_path / "mint.pem"
    path.write_bytes(rsa_key().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.NoEncryption()))
    os.chmod(path, 0o600)
    return SigningConfig.from_mapping({"MNEMIQ_TOKEN_KEY_FILE": str(path), "MNEMIQ_TOKEN_KID": "k1",
                                       "MNEMIQ_TOKEN_ISSUER": ISSUER, "MNEMIQ_TOKEN_AUDIENCE": AUDIENCE,
                                       "MNEMIQ_TOKEN_TENANT": "acme"})


TARGET = ForwardConfig.from_mapping({"MNEMIQ_URL": "http://mnemiq.internal:8080/base"})
ANA = SimpleNamespace(id=7, username="ana", is_anonymous=False, is_guest_user=False)


class Recorder:
    """A `Send` that records each outbound request and answers what it was given."""

    def __init__(self, answer: Inbound | Exception | None = None) -> None:
        self.sent: list[Outbound] = []
        self.answer = answer

    def __call__(self, outbound: Outbound) -> Inbound:
        self.sent.append(outbound)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer or Inbound(200, "application/json", None, iter([b'{"ok": true}']))


def _forward(signing, method="GET", path="/v1/schema", query=None, body=b"", user=ANA, send=None):
    send = send if send is not None else Recorder()
    found = forward_for(user, method, path, query or {}, body, signing=signing, target=TARGET, now=NOW, send=send)
    return found, send


def _claims(outbound: Outbound) -> dict:
    token = outbound.headers["authorization"].removeprefix("Bearer ")
    return jwt.decode(token, rsa_key().public_key(), algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER,
                      options={"verify_exp": False, "verify_nbf": False, "verify_iat": False})


def _text(inbound: Inbound) -> str:
    return b"".join(inbound.chunks).decode()


def test_forwards_with_a_fresh_token_for_the_signed_in_user_only(signing):
    """The token is the signed-in user's: a username in the body changes nothing, and each request gets its own."""
    body = b'{"goal": "sales", "username": "admin"}'
    _, send = _forward(signing, "POST", "/v1/superset/builds", body=body)
    _forward(signing, "POST", "/v1/superset/builds", body=body, send=send)
    first, second = send.sent
    assert _claims(first)["sub"] == "ana" and _claims(first)["preferred_username"] == "ana"
    assert _claims(first)["jti"] != _claims(second)["jti"]
    assert first.url == "http://mnemiq.internal:8080/base/v1/superset/builds" and first.body == body


def test_purpose_follows_the_route(signing):
    _, chat = _forward(signing, "POST", "/v1/chat", body=b"{}")
    _, build = _forward(signing, "GET", f"/v1/superset/builds/{BUILD_ID}")
    assert _claims(chat.sent[0])["scope"].split() == list(ASK_SCOPES)
    assert _claims(build.sent[0])["scope"].split() == list(BUILD_SCOPES)
    assert chat.sent[0].stream is True and build.sent[0].stream is False


# Each path as Superset's router hands it over: decoded once. A browser's `%2f` arrives as `/`, so a name holding one
# is two segments and matches no route; a `%2f` still in the path was encoded twice, and is refused.
@pytest.mark.parametrize("path", [
    "/v1/ask", "/v1/whoami", "/openai/v1/chat/completions", "/mcp", "/v1/superset/builds/../x",
    "/v1/superset/builds/..", "//elsewhere/v1/schema", "/v1/superset/dashboards/12/measures/a/b/edit",
    "/v1/superset/dashboards/12/measures/a%2fb/edit", "/v1/superset/dashboards/12/measures/../edit",
    "/v1/superset/dashboards/12/measures/./edit", "/v1/superset/dashboards/0/measures",
    f"/v1/superset/builds/{BUILD_ID}/next/again", f"/v1/superset/builds/{BUILD_ID}/confirm/again", "/v1/schema/",
    "/V1/schema",
])
def test_unlisted_path_is_refused_without_calling_mnemiq(signing, path):
    found, send = _forward(signing, "POST" if path.endswith(("edit", "again")) else "GET", path)
    assert found.status == 404 and NOT_FORWARDED in _text(found)
    assert send.sent == []


def test_wrong_method_is_refused(signing):
    found, send = _forward(signing, "POST", "/v1/schema")
    assert found.status == 405 and send.sent == []
    found, send = _forward(signing, "GET", "/v1/chat")
    assert found.status == 405 and send.sent == []


def test_unlisted_query_is_refused(signing):
    found, send = _forward(signing, "GET", "/v1/superset/builds", query={"principal": ["bo"]})
    assert found.status == 400 and QUERY_REFUSED in _text(found) and send.sent == []
    found, send = _forward(signing, "GET", "/v1/superset/builds", query={"dashboard_id": ["12"]})
    assert found.status == 200 and send.sent[0].url.endswith("/v1/superset/builds?dashboard_id=12")
    found, send = _forward(signing, "GET", "/v1/superset/builds", query={"dashboard_id": ["12", "13"]})
    assert found.status == 400 and send.sent == []


def test_never_forwards_the_cookie_or_csrf_header(signing):
    """The incoming headers are not even an input: what mnemiq receives is exactly these."""
    _, send = _forward(signing, "POST", "/v1/superset/builds", body=b"{}")
    _, plain = _forward(signing, "GET", "/v1/schema")
    assert set(send.sent[0].headers) == {"authorization", "accept", "content-type"}
    assert set(plain.sent[0].headers) == {"authorization", "accept"}
    assert send.sent[0].headers["content-type"] == "application/json"


def test_answer_carries_only_type_and_challenge(signing):
    answer = Inbound(401, "application/json", 'Bearer error="invalid_token"', iter([b'{"detail": "x"}']))
    found, _ = _forward(signing, send=Recorder(answer))
    assert found.status == 401 and found.content_type == "application/json"
    assert found.www_authenticate == 'Bearer error="invalid_token"'
    assert forward.ANSWER_HEADERS == {"Cache-Control": "no-store"}


def test_unreachable_is_a_502_and_timeout_a_504(signing):
    found, _ = _forward(signing, send=Recorder(forward.Unreachable()))
    assert found.status == 502 and MNEMIQ_UNREACHABLE in _text(found)
    found, _ = _forward(signing, send=Recorder(forward.TimedOut()))
    assert found.status == 504 and MNEMIQ_TIMED_OUT in _text(found)


def test_large_body_is_refused(signing):
    found, send = _forward(signing, "POST", "/v1/superset/builds", body=b"x" * (MAX_BODY + 1))
    assert found.status == 413 and BODY_TOO_LARGE in _text(found) and send.sent == []


@pytest.mark.parametrize("user", [
    SimpleNamespace(id=None, username="guest", is_anonymous=False, is_guest_user=True),
    SimpleNamespace(is_anonymous=True),
    SimpleNamespace(id=7, username="ana", is_anonymous=False, is_guest_user=True),
])
def test_guest_and_anonymous_are_refused_before_anything(signing, user):
    found, send = _forward(signing, "GET", "/not/even/a/route", user=user)
    assert found.status == 403 and NOT_SIGNED_IN in _text(found) and send.sent == []


@pytest.mark.parametrize("config", [
    {}, {"MNEMIQ_URL": ""}, {"MNEMIQ_URL": "ftp://mnemiq"}, {"MNEMIQ_URL": "http://user:pw@mnemiq"},
    {"MNEMIQ_URL": "http://mnemiq/?a=1"}, {"MNEMIQ_URL": "http://mnemiq/#x"},
    {"MNEMIQ_URL": "http://mnemiq", "MNEMIQ_FORWARD_TIMEOUT_S": 0},
    {"MNEMIQ_URL": "http://mnemiq", "MNEMIQ_FORWARD_TIMEOUT_S": 901},
    {"MNEMIQ_URL": "http://mnemiq", "MNEMIQ_FORWARD_IDLE_S": 4},
])
def test_config_refuses_credentials_query_or_missing_url(config):
    with pytest.raises(ValueError):
        ForwardConfig.from_mapping(config)


def test_config_keeps_a_path_prefix_and_names_the_setting():
    assert ForwardConfig.from_mapping({"MNEMIQ_URL": "https://m.example/mnemiq/"}).url == "https://m.example/mnemiq"
    assert "MNEMIQ_URL" in FORWARD_NOT_CONFIGURED


def test_stream_passes_chunks_as_they_come(signing):
    """A stream's chunks are passed on as they arrive: the first is handed back before the second is made."""
    made: list[str] = []

    def chunks():
        made.append("first")
        yield b"data: one\n\n"
        made.append("second")
        yield b"data: two\n\n"

    found, _ = _forward(signing, "POST", "/v1/chat", body=b"{}", send=Recorder(
        Inbound(200, "text/event-stream", None, chunks())))
    first = next(found.chunks)
    assert first == b"data: one\n\n" and made == ["first"]
    assert next(found.chunks) == b"data: two\n\n" and made == ["first", "second"]


def _leaked(records, *secrets: str) -> list[str]:
    """The log messages that hold any of `secrets`."""
    return [record.getMessage() for record in records if any(s and s in record.getMessage() for s in secrets)]


def test_no_token_or_body_is_logged(signing, caplog):
    """The forwarder's own log lines, which fire when mnemiq cannot be reached or times out, hold neither the token
    minted for that very request nor its body. The check is shown a line that does."""
    caplog.set_level(logging.DEBUG)
    body = b'{"goal": "a goal that names a customer"}'
    for failure in (forward.Unreachable(), forward.TimedOut()):
        send = Recorder(failure)
        found, _ = _forward(signing, "POST", "/v1/superset/builds", body=body, send=send)
        assert found.status in (502, 504)
        token = send.sent[0].headers["authorization"].removeprefix("Bearer ")
        assert caplog.records and _leaked(caplog.records, token, body.decode()) == []
    planted = logging.LogRecord("x", logging.WARNING, __file__, 1, "sent %s", (body.decode(),), None)
    assert _leaked([planted], body.decode()) != []


def test_every_listed_route_forwards(signing):
    """Each route the panel uses is forwarded, by its method, with its purpose."""
    cases = [("GET", "/v1/schema"), ("POST", "/v1/chat"), ("GET", "/v1/superset/builds"),
             ("POST", "/v1/superset/builds"), ("GET", f"/v1/superset/builds/{BUILD_ID}"),
             ("POST", f"/v1/superset/builds/{BUILD_ID}/next"), ("POST", f"/v1/superset/builds/{BUILD_ID}/confirm"),
             ("GET", "/v1/superset/dashboards/12/measures"),
             ("POST", "/v1/superset/dashboards/12/measures/policies/edit"),
             ("POST", "/v1/superset/dashboards/12/measures/total_premium@12/remove"),
             ("POST", "/v1/superset/dashboards/12/measures/policies/save"),
             ("POST", "/v1/superset/dashboards/12/changes")]
    for method, path in cases:
        found, send = _forward(signing, method, path, body=b"{}" if method == "POST" else b"")
        assert found.status == 200 and len(send.sent) == 1, (method, path, found.status)


def test_a_measures_name_is_encoded_once_for_mnemiq(signing):
    """The name as the router decoded it reaches mnemiq encoded once: a saved metric's `@` and a literal `%` alike."""
    _, send = _forward(signing, "POST", "/v1/superset/dashboards/12/measures/total_premium@12/remove", body=b"{}")
    assert send.sent[0].url.endswith("/v1/superset/dashboards/12/measures/total_premium%4012/remove")
    _, send = _forward(signing, "POST", "/v1/superset/dashboards/12/measures/growth %25/save", body=b"{}")
    assert send.sent[0].url.endswith("/v1/superset/dashboards/12/measures/growth%20%2525/save")


def test_the_whole_answer_is_bounded_by_the_total_time(signing):
    """Each outbound carries the request's whole time beside its wait for mnemiq: a stream waits the idle time between
    reads, and its whole answer, like any other, the total time."""
    target = ForwardConfig.from_mapping({"MNEMIQ_URL": "http://m", "MNEMIQ_FORWARD_TIMEOUT_S": 120,
                                         "MNEMIQ_FORWARD_IDLE_S": 30})
    send = Recorder()
    forward_for(ANA, "POST", "/v1/chat", {}, b"{}", signing=signing, target=target, now=NOW, send=send)
    forward_for(ANA, "GET", "/v1/schema", {}, b"", signing=signing, target=target, now=NOW, send=send)
    (chat, schema) = send.sent
    assert (chat.timeout_s, chat.total_s) == (30, 120) and (schema.timeout_s, schema.total_s) == (120, 120)

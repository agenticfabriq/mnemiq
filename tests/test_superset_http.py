"""Superset mode at the HTTP door: `/v1` verifies the person's token and hands that same token on, every Superset
refusal reaches the person as the response that says what to do about it, and the entry points that answer as the one
identity their settings name refuse to run in Superset mode.

The deployment is the scripted Superset-mode one `tests/test_superset_runtime.py` builds, served as `mnemiq serve
--http` serves it: `serve_http` checks the settings, builds the runtime and the doors, and hands the app to a stand-in
for uvicorn. Tokens are signed here with a key whose public half is the deployment's key set file, and Superset is
`FakeSuperset`, a real MCP server on loopback, so every call crosses the wire and the bearer it carried is the one the
fake recorded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import mnemiq.adapters.resolve as resolve_module
import mnemiq.runtime as runtime_module
import mnemiq.server.app as app_module
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.superset.client import (
    DelegatedToken,
    SupersetDenied,
    SupersetMcp,
    SupersetPermissionDenied,
    SupersetRefusal,
    SupersetRefusedCaller,
    SupersetResponseTooLarge,
    SupersetToolError,
    SupersetUnavailable,
)
from mnemiq.superset.executor import SupersetCredentialMissing
from mnemiq.superset.grants import NOT_A_BUILDER, GrantsUnreadable, NotABuilder, SupersetMisconfigured
from mnemiq.superset.payload import SupersetAccessRefused
from mnemiq.superset.service import SERVICE_KEY_ENV, service_credential_from_env
from mnemiq.superset.settings import SupersetConfigError, SupersetSnapshotRefused
from tests.auth_keys import AUDIENCE, ISSUER, claims, jwks_for, mint, owui_claims, hs256, public_pem, raw_token, rsa_key
from tests.scripted_deployment import SQL, retrieve_the_one_table
from tests.superset_fake import FakeSuperset
from tests.sweep.stub_model import StubModel
from tests.test_superset_runtime import DATABASE_ID, SOURCE_ID, _PROBE, _answer_rows, _snapshot, superset_deployment

SIGN_IN = "Superset did not accept this sign-in; sign in to Superset again."
DENIED = "Superset refused this request and did not say why; its log records the reason."
ACCESS_REFUSED = "Superset refused access to something this request reads."
GRANTS_UNREADABLE = ("Superset's answer about what this user may read could not be read, so nothing was answered; "
                     "mnemiq's log records why.")
SNAPSHOT_REFUSED = ("This deployment's knowledge is not the one enriched for the Superset database it answers from, so "
                    "nothing is answered until an operator enriches it through Superset; mnemiq's log records why.")
# Superset's own texts, as its MCP service renders them (`superset/mcp_service/auth.py` and
# `superset/mcp_service/middleware.py`, `GlobalErrorHandlerMiddleware`).
ROLE_DENIAL = "Permission denied: can_execute_sql_query on SQLLab for user west (tool: execute_sql)"
GENERIC_DENIAL = "Permission denied for execute_sql: You don't have access to this resource."
UNKNOWN_USER = ("Invalid parameter in execute_sql: JWT authenticated user not found in Superset database. "
                "Ensure the user exists before granting MCP access.")
# Words only Superset says in these tests, so where they end up can be read.
SAID = "superset-said-this"

ROUTES = [("post", "/v1/ask", {"question": "how many?"}), ("post", "/v1/chat", {"question": "how many?"}),
          ("get", "/v1/schema", None), ("get", "/v1/whoami", None)]


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        served.reply("execute_sql", _answer_rows)
        yield served


@pytest.fixture
def model():
    with StubModel(sql=SQL) as stub:
        yield stub
    assert not stub.all_errors()


@pytest.fixture
def jwks(tmp_path) -> str:
    """The identity provider's key set, as a file the deployment reads instead of fetching it."""
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps(jwks_for((rsa_key(), "k1"))))
    return str(path)


@pytest.fixture
def no_direct_source(monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("Superset mode reached a direct source connection")

    monkeypatch.setattr(runtime_module, "source_adapter", reached)
    monkeypatch.setattr(resolve_module, "adapter_for", reached)


def _settings(tmp_path, model, fake, jwks, **over) -> Settings:
    """A Superset-mode deployment whose doors verify tokens from `ISSUER` for `AUDIENCE`, with the key set in a file."""
    return superset_deployment(tmp_path, model.url, fake.url, oidc_issuer=ISSUER, oidc_audience=AUDIENCE,
                               oidc_jwks_file=jwks, tenant="acme", **over)


def _serve(settings: Settings, monkeypatch):
    """What `serve_http` would serve for `settings`, and the runtime it built: uvicorn is replaced, nothing else."""
    built: dict = {}
    real = runtime_module.build_runtime
    monkeypatch.setattr(runtime_module, "build_runtime", lambda s: built.setdefault("runtime", real(s)))
    monkeypatch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
    app_module.serve_http(settings, host="127.0.0.1", port=8080)
    return TestClient(built["app"]), built["runtime"]


@pytest.fixture
def server(tmp_path, model, fake, jwks, monkeypatch, no_direct_source):
    retrieve_the_one_table(monkeypatch)
    return _serve(_settings(tmp_path, model, fake, jwks), monkeypatch)


def _claims(**over) -> dict:
    return claims(**{"sub": "west", "preferred_username": "west", "scope": "db.read", **over})


def _token(**over) -> str:
    return mint(_claims(**over))


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _call(client, method, path, body, headers=None):
    return getattr(client, method)(path, headers=headers or {}, **({"json": body} if body is not None else {}))


def _frames(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


# --- the token, verified and handed on ------------------------------------------------------------------------------


def test_ask_forwards_the_verified_token_unchanged(server, fake):
    client, _ = server
    token = _token()
    r = client.post("/v1/ask", json={"question": "q"}, headers=_bearer(token))
    assert r.status_code == 200, r.text
    assert not r.json()["failed"] and not r.json()["deferred"], r.json()
    assert fake.calls and all(call.authorization == f"Bearer {token}" for call in fake.calls)


def test_schema_forwards_the_verified_token_unchanged(server, fake):
    client, runtime = server
    runtime.con.execute("CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT, card TEXT)")
    runtime.con.execute("INSERT INTO semantic_object (object_id, card) VALUES ('wide', 'wide (col_0 TEXT)')")
    token = _token()
    r = client.get("/v1/schema", headers=_bearer(token))
    assert r.status_code == 200, r.text
    assert [table["object_id"] for table in r.json()["tables"]] == ["wide"]
    assert fake.calls and all(call.authorization == f"Bearer {token}" for call in fake.calls)


def test_whoami_names_the_verified_caller_and_asks_superset_nothing(server, fake):
    client, _ = server
    r = client.get("/v1/whoami", headers=_bearer(_token()))
    assert r.status_code == 200
    assert (r.json()["principal_id"], r.json()["tenant_id"], r.json()["verified"]) == ("west", "acme", True)
    assert fake.calls == []


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_no_token_is_401_before_superset_is_called(server, fake, model, method, path, body):
    client, _ = server
    r = _call(client, method, path, body)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert fake.calls == [] and model.calls == 0


def _expired() -> str:
    now = int(time.time())
    return _token(iat=now - 900, nbf=now - 900, exp=now - 600)


BAD_TOKENS = {
    "expired": _expired,
    "wrong-audience": lambda: _token(aud="another-service"),
    "wrong-issuer": lambda: _token(iss="https://elsewhere.example/realms/r"),
    # The provider's public key used as an HMAC secret, which an RS256 verifier must refuse.
    "hs256": lambda: raw_token(_claims(), alg="HS256", key=public_pem(rsa_key())),
    "key-not-in-the-jwks": lambda: mint(_claims(), rsa_key("elsewhere"), kid="k9"),
}


@pytest.mark.parametrize("make", BAD_TOKENS.values(), ids=BAD_TOKENS.keys())
@pytest.mark.parametrize("method, path, body", ROUTES)
def test_bad_tokens_never_reach_superset(server, fake, model, make, method, path, body):
    client, _ = server
    r = _call(client, method, path, body, _bearer(make()))
    assert r.status_code == 401, r.text
    assert fake.calls == [] and model.calls == 0


def test_the_delegation_is_the_bearer_the_door_verified(tmp_path, jwks):
    """The door hands on the Bearer it verified, from the one Authorization header it read, and verifies exactly as
    `v1` does: same identity, and every refusal the same."""
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import Delegation, DoorRefusal, HttpDoors

    cfg = auth_config_from_settings(Settings(oidc_issuer=ISSUER, oidc_audience=AUDIENCE, oidc_jwks_file=jwks,
                                             tenant="acme"))
    doors = HttpDoors(cfg)
    token = _token()
    headers = {"authorization": [f"Bearer {token}"]}
    found = doors.v1_delegation(headers)
    assert isinstance(found, Delegation)
    assert found.token == DelegatedToken(token) and found.identity == doors.v1(headers)
    refused = [{}, {"authorization": [f"Bearer {token}", f"Bearer {_token(sub='east')}"]},
               {"authorization": [f"Bearer {_expired()}"]},
               {"authorization": [f"Bearer {token}"], "x-openwebui-user-jwt": [hs256(owui_claims(), b"s" * 32)]},
               {"authorization": [f"Bearer {_token(scope='openid')}"]}]
    for sent in refused:
        with pytest.raises(DoorRefusal) as by_v1:
            doors.v1(sent)
        with pytest.raises(DoorRefusal) as by_delegation:
            doors.v1_delegation(sent)
        assert (by_delegation.value.status, by_delegation.value.reason) == (by_v1.value.status, by_v1.value.reason)


def test_service_key_in_the_environment_is_never_sent(tmp_path, model, fake, jwks, monkeypatch, no_direct_source):
    """The service key enrichment uses lives in a file the environment names; nothing on the ask path reads it. The
    file is one enrichment would accept, so the key is there to be read, and still no request carries it."""
    key = tmp_path / "service-key"
    key.write_text("sst_service")
    key.chmod(0o600)
    monkeypatch.setenv(SERVICE_KEY_ENV, str(key))
    assert service_credential_from_env().value == "sst_service"  # in place and readable
    retrieve_the_one_table(monkeypatch)
    client, runtime = _serve(_settings(tmp_path, model, fake, jwks), monkeypatch)
    runtime.con.execute("CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT, card TEXT)")
    tokens = []
    for method, path, body in ROUTES:
        tokens.append(_token())
        assert _call(client, method, path, body, _bearer(tokens[-1])).status_code == 200
    assert fake.calls and not any("sst_service" in (call.authorization or "") for call in fake.calls)
    assert {call.authorization for call in fake.calls} <= {f"Bearer {token}" for token in tokens}


def test_chat_stream_carries_the_delegation(server, fake, monkeypatch):
    """`/v1/chat` runs the ask through the real pump, off the event loop the stream is served on: the client runs its
    own loop with `anyio.run` in the calling thread, which a thread already running one could not."""
    client, _ = server
    in_a_loop: list[bool] = []
    real_call = SupersetMcp.call

    def recording_call(self, tool, request, **kwargs):
        try:
            asyncio.get_running_loop()
            in_a_loop.append(True)
        except RuntimeError:
            in_a_loop.append(False)
        return real_call(self, tool, request, **kwargs)

    monkeypatch.setattr(SupersetMcp, "call", recording_call)
    token = _token()
    r = client.post("/v1/chat", json={"question": "q"}, headers=_bearer(token))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r.text)
    answer = next((f["value"] for f in frames if f.get("name") == "mnemiq.answer"), None)
    assert answer is not None, frames
    assert not answer["failed"] and not answer["deferred"], answer
    assert frames[-1]["type"] == "RUN_FINISHED"
    assert fake.calls and {call.authorization for call in fake.calls} == {f"Bearer {token}"}
    assert in_a_loop and not any(in_a_loop)


# --- Superset's refusals, as the person reads them ------------------------------------------------------------------


def _refusal_in_band(client, token: str) -> dict:
    r = client.post("/v1/chat", json={"question": "q"}, headers=_bearer(token))
    assert r.status_code == 200
    frames = _frames(r.text)
    assert not any(f.get("name") == "mnemiq.answer" for f in frames), frames
    assert frames[-1]["type"] == "RUN_ERROR", frames
    return frames[-1]


def test_not_a_builder_is_403_naming_the_roles(server, fake, model):
    client, _ = server
    fake.reply("execute_sql", lambda request, authorization: RuntimeError(ROLE_DENIAL))
    for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]:
        r = _call(client, method, path, body, _bearer(_token()))
        assert r.status_code == 403, r.text
        assert r.json()["detail"] == NOT_A_BUILDER
        assert "Alpha" in r.json()["detail"] and "sql_lab" in r.json()["detail"]
    assert _refusal_in_band(client, _token())["message"] == NOT_A_BUILDER
    assert model.calls == 0


def test_a_generic_permission_error_claims_no_cause(server, fake, model):
    client, _ = server
    fake.reply("execute_sql", lambda request, authorization: RuntimeError(GENERIC_DENIAL))
    for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]:
        r = _call(client, method, path, body, _bearer(_token()))
        assert r.status_code == 403, r.text
        assert r.json()["detail"] == DENIED
        said = r.text.lower()
        assert "role" not in said and "alpha" not in said and "sign in" not in said
    assert _refusal_in_band(client, _token())["message"] == DENIED
    assert model.calls == 0


@pytest.mark.parametrize("refuse", ["at-the-transport", "unknown-user"])
def test_unknown_superset_user_is_refused_not_500(server, fake, model, refuse):
    """Superset refuses a verified token whose user it has no account for: at its transport with a 401, or with the
    tool error its authentication raises. Either way the person is told to sign in to Superset, never to ask for
    roles."""
    client, _ = server
    token = _token()
    if refuse == "at-the-transport":
        fake.refuse_token(token)
    else:
        fake.reply("execute_sql", lambda request, authorization: RuntimeError(UNKNOWN_USER))
    for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]:
        r = _call(client, method, path, body, _bearer(token))
        assert r.status_code == 401, r.text
        assert r.json()["detail"] == SIGN_IN and r.headers["www-authenticate"] == "Bearer"
        assert "role" not in r.text.lower()
    assert _refusal_in_band(client, token)["message"] == SIGN_IN
    assert model.calls == 0


def test_misconfigured_superset_is_503_naming_the_setting(server, fake, model):
    client, _ = server
    fake.reply("execute_sql", lambda request, authorization: RuntimeError("Unknown tool: execute_sql"))
    for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]:
        r = _call(client, method, path, body, _bearer(_token()))
        assert r.status_code == 503, r.text
        assert "MCP_DISABLED_TOOLS" in r.json()["detail"]
    assert "MCP_DISABLED_TOOLS" in _refusal_in_band(client, _token())["message"]
    assert model.calls == 0


def test_a_permission_denied_statement_ends_the_ask_as_the_refusal_it_is(server, fake, model):
    """Raised by the answer statement itself, a role's denial is read by the runner as the source not answering, and
    the ask still ends as the refusal it is: the 403 naming the roles, never a "try again", with the writer asked once
    and nothing repaired."""
    client, _ = server

    def denying_the_statement(request, authorization):
        if request.get("dry_run") or request.get("force_refresh") or _PROBE.search(request["sql"]):
            return _answer_rows(request, authorization)
        return RuntimeError(ROLE_DENIAL)

    fake.reply("execute_sql", denying_the_statement)
    r = client.post("/v1/ask", json={"question": "q"}, headers=_bearer(_token()))
    assert r.status_code == 403, r.text
    assert r.json()["detail"] == NOT_A_BUILDER and "try again" not in r.text.lower()
    assert model.calls == 1


def _read_only():
    from mnemiq.superset.executor import SupersetReadOnly

    return SupersetReadOnly()


# Each failure a route may meet, raised where the route calls the runtime: the status and the sentence it becomes.
# Superset's own words ride in `.detail` only, and must reach no one.
RAISED = {
    "permission-denied": (lambda: SupersetPermissionDenied("Superset's roles for this user do not permit execute_sql",
                                                           detail=SAID), 403, NOT_A_BUILDER),
    "not-a-builder": (NotABuilder, 403, NOT_A_BUILDER),
    "access-refused": (lambda: SupersetAccessRefused("Superset refused access to something this query reads."), 403,
                       ACCESS_REFUSED),
    "bare-refusal": (lambda: SupersetRefusal("Superset said no", detail=SAID), 403, DENIED),
    "denied": (lambda: SupersetDenied("Superset refused execute_sql and did not say why", detail=SAID), 403, DENIED),
    "refused-caller": (lambda: SupersetRefusedCaller("Superset did not accept the credential this request carries "
                                                     "(HTTP 401)", detail=SAID), 401, SIGN_IN),
    "tool-error": (lambda: SupersetToolError("Superset's execute_sql tool failed", detail=SAID), 503,
                   "Superset's execute_sql tool failed"),
    "too-large": (lambda: SupersetResponseTooLarge("Superset's answer to list_datasets was larger than its MCP service "
                                                   "will send; ask for less", detail=SAID), 503,
                  "Superset's answer to list_datasets was larger than its MCP service will send; ask for less"),
    "unavailable": (lambda: SupersetUnavailable("Superset's MCP service could not be reached", detail=SAID), 503,
                    "Superset's MCP service could not be reached"),
    "misconfigured": (lambda: SupersetMisconfigured("Superset's MCP service is not set up as mnemiq needs: x"), 503,
                      "Superset's MCP service is not set up as mnemiq needs: x"),
    "credential-missing": (lambda: SupersetCredentialMissing("Superset mode answers each request as the person whose "
                                                             "verified token it carries, and this request carries "
                                                             "none"), 401,
                           "Superset mode answers each request as the person whose verified token it carries, and "
                           "this request carries none"),
    "read-only": (_read_only, 403, "read-only"),
    "grants-unreadable": (GrantsUnreadable, 503, GRANTS_UNREADABLE),
    # The detail names the deployment's settings, and must reach no one either.
    "snapshot-refused": (lambda: SupersetSnapshotRefused(f"the snapshot was enriched elsewhere {SAID}"), 503,
                         SNAPSHOT_REFUSED),
}


@pytest.mark.parametrize("make, status, detail", RAISED.values(), ids=RAISED.keys())
def test_no_refusal_is_a_500(server, make, status, detail, monkeypatch):
    client, runtime = server

    def raising(*args, **kwargs):
        raise make()

    monkeypatch.setattr(runtime, "ask", raising)
    monkeypatch.setattr(runtime, "scope", raising)
    for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]:
        r = _call(client, method, path, body, _bearer(_token()))
        assert r.status_code == status, (path, r.text)
        told = r.json()["detail"]
        assert told == detail or (detail == "read-only" and "read-only" in told and "sign in" not in told.lower())
        if status == 401:
            assert r.headers["www-authenticate"] == "Bearer"
        assert SAID not in r.text
    in_band = _refusal_in_band(client, _token())
    assert in_band["message"] == told
    assert SAID not in json.dumps(in_band)


def test_a_write_is_refused_as_read_only_and_never_told_to_sign_in(server, fake, monkeypatch):
    from mnemiq.superset.executor import SupersetReadOnly

    client, runtime = server
    writer = IdentityContext(tenant_id="acme", principal_id="west")
    with pytest.raises(SupersetReadOnly) as refused:
        runtime.write("DELETE FROM wide", writer)
    assert "read-only" in str(refused.value) and "sign in" not in str(refused.value).lower()
    # No route writes; one that reached a write would answer the read-only refusal, never a sign-in.
    monkeypatch.setattr(runtime, "ask", lambda *args, **kwargs: runtime.write("DELETE FROM wide", writer))
    r = client.post("/v1/ask", json={"question": "q"}, headers=_bearer(_token()))
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert "sign in" not in r.text.lower()
    assert fake.calls == []


def test_an_unknown_mode_names_only_the_modes_offered(server, fake, model):
    client, _ = server
    r = client.post("/v1/ask", json={"question": "q", "mode": "deep"}, headers=_bearer(_token()))
    assert r.status_code == 422
    assert r.json()["detail"] == "unknown mode; this deployment offers: ['instant', 'thinking']"
    assert fake.calls == [] and model.calls == 0


def test_the_mcp_door_names_only_the_modes_offered():
    """The MCP door's refusal of a mode is composed from the modes the runtime it serves offers, not from every mode
    mnemiq has."""
    import anyio

    from mnemiq.agent.route import StaticRouter
    from mnemiq.mcp.server import _mcp_classes, build_mcp

    class _Runtime:
        router = StaticRouter(default="thinking", offered=frozenset({"instant", "thinking"}))

        def ask(self, question, identity, mode=None, profile=None):
            return self.router.route(question, mode)

    server = build_mcp(_Runtime(), IdentityContext(principal_id="p", tenant_id="t"))
    _, tool_error = _mcp_classes()

    async def call():
        await server.call_tool("db_read", {"question": "q", "mode": "deep"})

    with pytest.raises(tool_error) as refused:
        anyio.run(call)
    assert "this deployment offers: ['instant', 'thinking']." in str(refused.value)


def test_superset_mode_without_doors_refuses_every_ask_as_unsigned(tmp_path, model, fake, monkeypatch,
                                                                    no_direct_source):
    """Built without an issuer, which startup refuses, the app still answers nobody: no token, no Superset call."""
    retrieve_the_one_table(monkeypatch)
    runtime = runtime_module.build_runtime(superset_deployment(tmp_path, model.url, fake.url))
    client = TestClient(app_module.build_app(runtime, IdentityContext(tenant_id="t", principal_id="u")))
    for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]:
        r = _call(client, method, path, body)
        assert r.status_code == 401, r.text
    r = client.post("/v1/chat", json={"question": "q"})
    assert _frames(r.text)[-1]["type"] == "RUN_ERROR"
    assert fake.calls == [] and model.calls == 0


# --- what Superset mode cannot answer from ------------------------------------------------------------------------


def _every_door(client) -> list:
    """`/v1/ask` and `/v1/schema`, each asked once with a fresh token, and `/v1/chat`'s failure frame: what each door
    answered."""
    told = [_call(client, method, path, body, _bearer(_token()))
            for method, path, body in [("post", "/v1/ask", {"question": "q"}), ("get", "/v1/schema", None)]]
    return [*told, _refusal_in_band(client, _token())]


def test_a_refused_snapshot_swap_is_503_on_every_door(server, fake, model, monkeypatch, caplog):
    """A newer snapshot Superset mode will not serve refuses each request on every door with mnemiq's own sentence: a
    503 on `/v1/ask` and `/v1/schema`, and on `/v1/chat` a failure in the stream. No door goes on serving the snapshot
    that was serving while another refuses; Superset and the model are asked nothing; and why the snapshot was refused
    goes to the log once per request, as a warning, and never to the person."""
    client, runtime = server
    serving = runtime.snapshot
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    monkeypatch.setattr(runtime_module, "load_current_snapshot",
                        lambda settings, con: (_snapshot([], version="v2"), {SOURCE_ID: "v2"}))
    why = "was not enriched through Superset"
    with caplog.at_level(logging.INFO):
        ask, schema, chat = _every_door(client)
    for r in (ask, schema):
        assert r.status_code == 503, r.text
        assert r.json()["detail"] == SNAPSHOT_REFUSED and why not in r.text
    assert chat["message"] == SNAPSHOT_REFUSED and why not in json.dumps(chat)
    logged = [record for record in caplog.records if why in record.getMessage()]
    assert len(logged) == 3 and {record.levelno for record in logged} == {logging.WARNING}
    assert runtime.snapshot is serving
    assert fake.calls == [] and model.calls == 0


def _datasets_restricted(fake: FakeSuperset) -> None:
    """A role Superset gives no access to the datasets' details: it refuses the grants' dry run, and answers
    `list_datasets` with an error in place of a page."""
    fake.reply("execute_sql", lambda request, authorization: {
        "success": False, "rows": None, "columns": None, "error": f"You need access to the following tables: `wide` {SAID}",
        "error_type": "TABLE_SECURITY_ACCESS_ERROR"})
    fake.reply("list_datasets", lambda request, authorization: {
        "error": f"You don't have permission to access dataset details for your role. {SAID}",
        "error_type": "DataModelMetadataRestricted"})


UNREADABLE_GRANTS = {
    "datasets-restricted": _datasets_restricted,
    "no-answer": lambda fake: fake.fail_http(502),
}


@pytest.mark.parametrize("unreadable", UNREADABLE_GRANTS.values(), ids=UNREADABLE_GRANTS.keys())
def test_grants_superset_cannot_answer_are_503_on_every_door(server, fake, model, unreadable):
    """When Superset's answer about what the person may read cannot be read, every door says so in mnemiq's words: a
    503 on `/v1/ask` and `/v1/schema`, and on `/v1/chat` a failure in the stream. Never an empty schema, never the
    access policy's wording (Superset mode holds no policy), and never Superset's own text."""
    client, runtime = server
    runtime.con.execute("CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT, card TEXT)")
    runtime.con.execute("INSERT INTO semantic_object (object_id, card) VALUES ('wide', 'wide (col_0 TEXT)')")
    unreadable(fake)
    ask, schema, chat = _every_door(client)
    for r in (ask, schema):
        assert r.status_code == 503, r.text
        assert r.json()["detail"] == GRANTS_UNREADABLE
        assert SAID not in r.text and "policy" not in r.text.lower()
    assert chat["message"] == GRANTS_UNREADABLE and SAID not in json.dumps(chat)
    assert model.calls == 0


def test_direct_mode_still_defers_an_unreadable_policy_in_its_own_words(tmp_path, model, monkeypatch):
    """The control: in direct mode a policy file that cannot be read is still a deferral, a 200 whose sentence names
    the policy as a configuration fault, and the schema still lists nothing."""
    from tests.scripted_deployment import IDENTITY, deployment

    retrieve_the_one_table(monkeypatch)
    settings = deployment(tmp_path, model.url)
    runtime = runtime_module.build_runtime(settings)
    Path(settings.authz_path).write_text("not json")
    runtime.con.execute("CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT, card TEXT)")
    runtime.con.execute("INSERT INTO semantic_object (object_id, card) VALUES ('wide', 'wide (col_0 TEXT)')")
    client = TestClient(app_module.build_app(runtime, IDENTITY))
    r = client.post("/v1/ask", json={"question": "q"})
    assert r.status_code == 200, r.text
    assert r.json()["deferred"] is True and r.json()["reason_code"] == "policy_unavailable"
    assert r.json()["answer"] == ("The authorization policy could not be read, so no access could be resolved. This is "
                                  "a configuration fault, not a limit on your account.")
    r = client.get("/v1/schema")
    assert r.status_code == 200 and r.json()["tables"] == []
    assert model.calls == 0


# --- a token Superset stops accepting mid-ask -----------------------------------------------------------------------


def _kind(request: dict) -> str:
    """What a call to `execute_sql` was for: the grants' dry run, a catalogue read, the proof's probe, or a statement."""
    if request.get("dry_run"):
        return "grants"
    if request.get("force_refresh"):
        return "catalogue"
    if _PROBE.search(request["sql"]):
        return "proof"
    return "statement"


def _refusing_from(fake: FakeSuperset, index: int, token: str) -> None:
    """Superset answers the first `index` calls and refuses `token` from then on, as it refuses a token that expires
    mid-ask: the call at `index` with Superset's refusal of a user it does not know, and every later one at its
    transport. The call at `index` is refused by its own answer, not at the transport, because the transport refuses a
    request only before the tool runs: refused while a call runs, the token would be refused on that call's own
    remaining requests, one call early."""
    calls = 0
    lock = threading.Lock()

    def answer(request, authorization):
        nonlocal calls
        with lock:
            this, calls = calls, calls + 1
        if this < index:
            return _answer_rows(request, authorization)
        fake.refuse_token(token)
        return RuntimeError(UNKNOWN_USER)

    fake.reply("execute_sql", answer)


# Where in an ask Superset may first refuse the token: the call kind it refuses first, and whether the ask carries an
# earlier turn, whose SQL is run again before the model is asked.
PLACES = {
    "grants": ("grants", False),
    "history": ("statement", True),
    "function-inventory": ("catalogue", False),
    "proof": ("proof", False),
    "answer-statement": ("statement", False),
}


@pytest.mark.parametrize("kind, with_history", PLACES.values(), ids=PLACES.keys())
def test_expiry_mid_ask_is_a_stated_failure(server, fake, model, monkeypatch, kind, with_history):
    """Wherever in the ask Superset stops accepting the token, the person is told to sign in to Superset again: a 401
    on `/v1/ask`, and on `/v1/chat`, whose 200 has already gone out, a failure in the stream carrying the same sentence.
    Never "try again", a repair, a second run of the statement, or a 500."""
    client, _ = server
    attempted: list[dict] = []
    real_call = SupersetMcp.call

    def recording_call(self, tool, request, **kwargs):
        attempted.append(request)
        return real_call(self, tool, request, **kwargs)

    monkeypatch.setattr(SupersetMcp, "call", recording_call)
    body = {"question": "q"}
    if with_history:
        first = client.post("/v1/ask", json=body, headers=_bearer(_token(jti="first"))).json()
        body["history"] = [{"question": "how many", "sql": SQL, "grant_fingerprint": first["grant_fingerprint"]}]
    fake.calls.clear()
    reference = client.post("/v1/ask", json=body, headers=_bearer(_token(jti="reference")))
    assert reference.status_code == 200 and not reference.json()["failed"], reference.text
    kinds = [_kind(call.request) for call in fake.calls]
    index = kinds.index(kind)

    for door in ("ask", "chat"):
        token = _token(jti=door)
        _refusing_from(fake, index, token)
        attempted.clear()
        asked = model.calls
        if door == "ask":
            r = client.post("/v1/ask", json=body, headers=_bearer(token))
            assert r.status_code == 401, r.text
            assert r.json()["detail"] == SIGN_IN and r.headers["www-authenticate"] == "Bearer"
        else:
            r = client.post("/v1/chat", json=body, headers=_bearer(token))
            assert r.status_code == 200
            frames = _frames(r.text)
            assert frames[-1]["type"] == "RUN_ERROR" and frames[-1]["message"] == SIGN_IN, frames
            assert not any(f.get("name") == "mnemiq.answer" for f in frames)
        assert "try again" not in r.text.lower()
        # The writer is asked at most once, and only when the refusal comes after the model is asked: nothing is
        # repaired. The statement the refusal met is not sent again.
        assert model.calls - asked == (0 if kind == "grants" or with_history else 1)
        # The first call refused is the one this place names, and nothing is asked of Superset again in its place.
        assert [_kind(request) for request in attempted[:index + 1]] == kinds[:index + 1]
        statements = [request for request in attempted if _kind(request) == "statement"]
        assert len(statements) <= 1
        fake.reply("execute_sql", _answer_rows)


def _failing_from(fake: FakeSuperset, index: int, failure) -> None:
    """Superset answers the first `index` calls, and every call from then on with `failure()`: an exception, which
    the fake sends as the tool's error, or a payload."""
    calls = 0
    lock = threading.Lock()

    def answer(request, authorization):
        nonlocal calls
        with lock:
            this, calls = calls, calls + 1
        return _answer_rows(request, authorization) if this < index else failure()

    fake.reply("execute_sql", answer)


def _reference(client, fake, with_history: bool) -> tuple[dict, list[str]]:
    """The body an ask sends, with an earlier turn when `with_history`, and the kinds of the calls a whole answer to it
    makes, in order."""
    body = {"question": "q"}
    if with_history:
        first = client.post("/v1/ask", json=body, headers=_bearer(_token(jti="first"))).json()
        body["history"] = [{"question": "how many", "sql": SQL, "grant_fingerprint": first["grant_fingerprint"]}]
    fake.calls.clear()
    reference = client.post("/v1/ask", json=body, headers=_bearer(_token(jti="reference")))
    assert reference.status_code == 200 and not reference.json()["failed"], reference.text
    return body, [_kind(call.request) for call in fake.calls]


# Superset's refusals of a known caller's roles and the one it does not explain, as its MCP service words them; each
# with the status and the sentence it reaches the person as.
TERMINAL = {
    "permission-denied": (ROLE_DENIAL, 403, NOT_A_BUILDER),
    "denied": (GENERIC_DENIAL, 403, DENIED),
}


@pytest.mark.parametrize("place", [p for p in PLACES if p != "grants"])
@pytest.mark.parametrize("said, status, sentence", TERMINAL.values(), ids=TERMINAL.keys())
def test_a_terminal_refusal_after_the_grants_is_its_own_refusal(server, fake, model, monkeypatch, said, status,
                                                               sentence, place):
    """Superset refuses the person's roles, or refuses without saying why, at a call after the grants were read: the
    function inventory, the proof, the answer statement, or an earlier turn run again. A retry meets the same refusal,
    so the person is told what the refusal is, as when the grants meet it: the status and sentence on `/v1/ask`, and
    the same sentence in the stream's failure on `/v1/chat`. Never "try again", a repair, a second statement, or
    Superset's own words."""
    client, _ = server
    attempted: list[dict] = []
    real_call = SupersetMcp.call

    def recording_call(self, tool, request, **kwargs):
        attempted.append(request)
        return real_call(self, tool, request, **kwargs)

    monkeypatch.setattr(SupersetMcp, "call", recording_call)
    kind, with_history = PLACES[place]
    body, kinds = _reference(client, fake, with_history)
    index = kinds.index(kind)

    for door in ("ask", "chat"):
        _failing_from(fake, index, lambda: RuntimeError(said))
        attempted.clear()
        asked = model.calls
        if door == "ask":
            r = client.post("/v1/ask", json=body, headers=_bearer(_token(jti=door)))
            assert r.status_code == status, r.text
            assert r.json()["detail"] == sentence
        else:
            r = client.post("/v1/chat", json=body, headers=_bearer(_token(jti=door)))
            assert r.status_code == 200
            frames = _frames(r.text)
            assert frames[-1]["type"] == "RUN_ERROR" and frames[-1]["message"] == sentence, frames
            assert not any(f.get("name") == "mnemiq.answer" for f in frames)
        assert "try again" not in r.text.lower()
        assert said not in r.text and "can_execute_sql_query" not in r.text and "this resource" not in r.text
        # The writer is asked at most once, and not at all when the earlier turn met the refusal: nothing is repaired.
        assert model.calls - asked == (0 if with_history else 1)
        assert [_kind(request) for request in attempted[:index + 1]] == kinds[:index + 1]
        assert len([request for request in attempted if _kind(request) == "statement"]) <= 1
        fake.reply("execute_sql", _answer_rows)


def test_a_table_superset_refuses_in_the_answer_statement_stays_the_models_to_repair(server, fake, model):
    """The control: Superset refusing a table the statement reads is not a refusal of the person, so it is not raised
    after the answer as one. The model is told so in mnemiq's words and asked again, and the ask ends as a stated
    answer, a 200, carrying none of Superset's words."""
    client, _ = server
    body, kinds = _reference(client, fake, False)
    asked = model.calls
    _failing_from(fake, kinds.index("statement"), lambda: {
        "success": False, "rows": None, "columns": None, "error": f"You need access to the following tables: `wide` "
        f"{SAID}", "error_type": "TABLE_SECURITY_ACCESS_ERROR"})
    r = client.post("/v1/ask", json=body, headers=_bearer(_token()))
    assert r.status_code == 200, r.text
    assert SAID not in r.text
    assert model.calls - asked > 1  # repaired: the writer was asked again


def test_superset_not_answering_the_statement_stays_an_outage(server, fake, model, monkeypatch):
    """The control: Superset not answering the answer statement is an outage, as before: the ask ends `failed`, a 200,
    and is not raised after the answer as a refusal."""
    client, _ = server
    body, _kinds = _reference(client, fake, False)
    real_call = SupersetMcp.call

    def unavailable_for_the_statement(self, tool, request, **kwargs):
        if tool == "execute_sql" and _kind(request) == "statement":
            raise SupersetUnavailable("Superset's MCP service could not be reached", detail=SAID)
        return real_call(self, tool, request, **kwargs)

    monkeypatch.setattr(SupersetMcp, "call", unavailable_for_the_statement)
    r = client.post("/v1/ask", json=body, headers=_bearer(_token()))
    assert r.status_code == 200, r.text
    assert r.json()["failed"] is True and r.json()["reason_code"] == "source_unavailable", r.json()
    assert SAID not in r.text


# --- entry points that answer as one identity -----------------------------------------------------------------------


def _superset_environment(monkeypatch, fake: FakeSuperset) -> None:
    monkeypatch.setenv("MNEMIQ_SUPERSET_MCP_URL", fake.url)
    monkeypatch.setenv("MNEMIQ_SUPERSET_DATABASE_ID", str(DATABASE_ID))


def _never_built(monkeypatch) -> None:
    import mnemiq.cli as cli
    import mnemiq.mcp.server as mcp_server

    def reached(*args, **kwargs):
        raise AssertionError("a runtime was built")

    monkeypatch.setattr(cli, "build_runtime", reached)
    monkeypatch.setattr(mcp_server, "build_runtime", reached)
    monkeypatch.setattr(mcp_server, "build_mcp", reached)


def _refused_by_the_cli(argv, monkeypatch, fake, capsys) -> str:
    from mnemiq.cli import main

    _superset_environment(monkeypatch, fake)
    _never_built(monkeypatch)
    assert main(argv) == 1
    said = capsys.readouterr().err
    assert "Superset mode" in said, said
    assert fake.calls == []
    return said


def test_cli_ask_refuses_in_superset_mode(monkeypatch, fake, capsys):
    said = _refused_by_the_cli(["ask", "how many?"], monkeypatch, fake, capsys)
    assert "mnemiq serve --http" in said


def test_cli_write_refuses_in_superset_mode(monkeypatch, fake, capsys):
    said = _refused_by_the_cli(["write", "DELETE FROM wide"], monkeypatch, fake, capsys)
    assert "read-only" in said and "sign in" not in said.lower()


def test_cli_stdio_serve_refuses_in_superset_mode(monkeypatch, fake, capsys):
    said = _refused_by_the_cli(["serve"], monkeypatch, fake, capsys)
    assert "mnemiq serve --http" in said


def test_the_mcp_server_refuses_superset_mode_before_building_anything(monkeypatch, fake):
    from mnemiq.mcp.server import serve

    _never_built(monkeypatch)
    with pytest.raises(SupersetConfigError, match="Superset mode"):
        serve(Settings(superset_mcp_url=fake.url, superset_database_id=DATABASE_ID))
    assert fake.calls == []


def test_cli_eval_refuses_in_superset_mode(monkeypatch, fake):
    from mnemiq.cli import main

    _superset_environment(monkeypatch, fake)
    with pytest.raises(SupersetConfigError, match="MNEMIQ_SUPERSET_MCP_URL"):
        main(["eval"])
    assert fake.calls == []


@pytest.mark.parametrize("argv", [["ask", "q"], ["write", "DELETE FROM wide"], ["serve"]], ids=["ask", "write", "serve"])
def test_a_database_id_without_the_url_is_refused_by_the_cli(argv, monkeypatch, capsys):
    """Half a Superset configuration, which would otherwise answer directly with mnemiq's own credential."""
    from mnemiq.cli import main

    monkeypatch.setenv("MNEMIQ_SUPERSET_DATABASE_ID", str(DATABASE_ID))
    _never_built(monkeypatch)
    assert main(argv) == 1
    assert "MNEMIQ_SUPERSET_MCP_URL" in capsys.readouterr().err


def test_serve_http_checks_the_settings_before_the_runtime_in_direct_mode(monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("a runtime was built")

    monkeypatch.setattr(runtime_module, "build_runtime", reached)
    with pytest.raises(SupersetConfigError, match="MNEMIQ_SUPERSET_MCP_URL"):
        app_module.serve_http(Settings(superset_database_id=DATABASE_ID), host="127.0.0.1", port=8080)


def test_serve_http_refuses_what_superset_mode_cannot_honour_before_the_runtime(tmp_path, model, fake, jwks,
                                                                                monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("a runtime was built")

    monkeypatch.setattr(runtime_module, "build_runtime", reached)
    policy = tmp_path / "authz.json"
    policy.write_text(json.dumps({"roles": {"analyst": ["wide"]}}))
    for over in ({"write_enabled": True}, {"authz_path": str(policy)}):
        with pytest.raises(SupersetConfigError):
            app_module.serve_http(_settings(tmp_path, model, fake, jwks, **over), host="127.0.0.1", port=8080)
    assert fake.calls == []


def test_direct_mode_serves_as_before(tmp_path, monkeypatch):
    """With no Superset URL, `serve_http` builds the runtime and the app as it always has, a policy file and all."""
    from tests.scripted_deployment import deployment

    built: dict = {}
    monkeypatch.setattr(runtime_module, "build_runtime", lambda s: built.setdefault("runtime", object()))
    monkeypatch.setattr(app_module, "build_app", lambda rt, ident, **kw: ("APP", rt, kw))
    monkeypatch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
    settings = deployment(tmp_path, "http://127.0.0.1:1")
    assert settings.authz_path and not settings.superset_mode
    app_module.serve_http(settings, host="127.0.0.1", port=8080)
    assert built["app"][1] is built["runtime"] and built["app"][2] == {"doors": None, "mcp_app": None}

"""The `openai` Python client is the contract for /openai/v1: what it can list, complete and stream, and how it reads
a refusal, against the app itself."""

import json
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.contract.seams import Trace
from mnemiq.profiles import Profile
from mnemiq.server.app import build_app
from tests.auth_keys import hs256, owui_claims

SECRET, KEY = "s" * 32, "k" * 32
TABLES = {"finance": "ledger", "ops": "shipment"}


class _ByRole:
    """Answers from the caller's roles: finance reads `ledger`, ops reads `shipment`, as two grants would."""

    def __init__(self):
        self.profiles = {name: SimpleNamespace(profile=Profile(name, roles, Settings()))
                         for name, roles in (("default", None), ("finance", frozenset({"finance"})),
                                             ("ops", frozenset({"ops"})))}
        self.default_profile = "default"

    def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
        table = TABLES[identity.roles[0]]
        if emit is not None:
            emit({"phase": "started", "stage": "retrieve", "id": 1})
            emit({"phase": "finished", "stage": "retrieve", "id": 1, "ms": 8.0, "ok": True})
        sql = f"SELECT count(*) FROM {table}"
        trace = Trace(question=question, plan_sql=sql, target_sql=sql, asked_sql=sql, result_shape="scalar",
                      timing={}, enrichment_version="e1", identity=identity)
        return AgentAnswer(answer=f"{table} has 2 rows.", trace=trace, mode="thinking", verify_layer="judge")

    def scope(self, identity):
        return {"tables": [], "starters": []}


@pytest.fixture
def app(tmp_path):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import HttpDoors

    path = tmp_path / "identity-map.json"
    path.write_text(json.dumps({"open-webui": {"sub": {"u-fin": ["finance"], "u-ops": ["ops"]}}}))
    doors = HttpDoors(auth_config_from_settings(Settings(
        tenant="acme", openwebui_jwt_secret=SECRET, openwebui_connection_key_header="X-Mnemiq-Connection",
        openwebui_connection_key=KEY, identity_map_path=str(path))))
    return build_app(_ByRole(), IdentityContext(tenant_id="local", principal_id="settings-user"),
                     static_dir=tmp_path, doors=doors)


def _client(app, sub: str | None = "u-fin") -> openai.OpenAI:
    """The client as Open WebUI's connection would be set up for option B: the signed user header and the connection
    key, no API key. The client insists on one, so it is dropped from each request, and exactly one credential
    arrives."""
    http = TestClient(app)

    def no_api_key(request):
        request.headers.pop("authorization", None)

    http.event_hooks = {"request": [no_api_key], "response": []}
    headers = {"X-Mnemiq-Task": ""}
    if sub is not None:
        headers.update({"X-OpenWebUI-User-Jwt": hs256(owui_claims(sub=sub), SECRET.encode()),
                        "X-Mnemiq-Connection": KEY})
    return openai.OpenAI(base_url="http://testserver/openai/v1", api_key="unused", http_client=http,
                         default_headers=headers, max_retries=0)


def test_the_openai_client_lists_models_and_completes(app):
    client = _client(app)
    assert [m.id for m in client.models.list()] == ["default", "finance"]
    done = client.chat.completions.create(model="finance", messages=[{"role": "user", "content": "how many?"}])
    assert done.choices[0].finish_reason == "stop"
    assert done.choices[0].message.content.startswith("ledger has 2 rows.")
    assert done.id.startswith("chatcmpl-") and done.model == "finance"


def test_the_openai_client_streams(app):
    client = _client(app)
    messages = [{"role": "user", "content": "how many?"}]
    plain = client.chat.completions.create(model="default", messages=messages).choices[0].message.content
    chunks = list(client.chat.completions.create(model="default", messages=messages, stream=True,
                                                 reasoning_effort="high"))
    content = "".join(c.choices[0].delta.content or "" for c in chunks)
    reasoning = [(c.choices[0].delta.model_extra or {}).get("reasoning_content") for c in chunks]
    assert "retrieve (8 ms)\n" in reasoning
    assert chunks[-1].choices[0].finish_reason == "stop"
    # The same reply streamed and not, up to the run id each carries.
    assert content.rsplit("run ", 1)[0] == plain.rsplit("run ", 1)[0]


def test_two_users_see_two_model_lists_and_two_tables(app):
    fin, ops = _client(app, "u-fin"), _client(app, "u-ops")
    assert [m.id for m in fin.models.list()] == ["default", "finance"]
    assert [m.id for m in ops.models.list()] == ["default", "ops"]
    ask = [{"role": "user", "content": "how many?"}]
    assert "FROM ledger" in fin.chat.completions.create(model="default", messages=ask).choices[0].message.content
    assert "FROM shipment" in ops.chat.completions.create(model="default", messages=ask).choices[0].message.content


def test_the_client_sees_a_refusal_as_its_own_error_types(app):
    ask = [{"role": "user", "content": "how many?"}]
    with pytest.raises(openai.AuthenticationError):
        _client(app, sub=None).models.list()
    with pytest.raises(openai.NotFoundError) as missing:
        _client(app, "u-ops").chat.completions.create(model="finance", messages=ask)
    assert missing.value.body["code"] == "model_not_found"
    with pytest.raises(openai.BadRequestError):
        _client(app).chat.completions.create(model="default", messages=ask,
                                             tools=[{"type": "function", "function": {"name": "f"}}])

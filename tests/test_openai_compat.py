"""/openai/v1: the way into mnemiq for Open WebUI, or any OpenAI client, with each request answered as its own
verified caller, one model per profile that caller may pick."""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.profiles import Profile
from mnemiq.server.app import build_app
from tests.auth_keys import hs256, owui_claims

SECRET, KEY = "s" * 32, "k" * 32
SETTINGS_IDENTITY = IdentityContext(tenant_id="local", principal_id="settings-user", roles=["analyst"])
MAP = {"open-webui": {"sub": {"u-fin": ["finance"], "u-ops": ["ops"]}}}
OPTION_B = dict(tenant="acme", openwebui_jwt_secret=SECRET, openwebui_connection_key_header="X-Mnemiq-Connection",
                openwebui_connection_key=KEY)


def _profiles(**roles):
    return {name: SimpleNamespace(profile=Profile(name, frozenset(r) if r is not None else None, Settings()))
            for name, r in roles.items()}


class _RT:
    """A runtime that records each question it was asked, with the profiles the deployment loaded."""

    def __init__(self, answer=None, profiles=None, default_profile="default", raises=None):
        self.answer = answer or AgentAnswer(answer="2.", mode="thinking")
        self.profiles = profiles if profiles is not None else _profiles(default=None, finance=["finance"],
                                                                        ops=["ops"])
        self.default_profile, self.raises = default_profile, raises
        self.asked: list[SimpleNamespace] = []

    @property
    def calls(self) -> int:
        return len(self.asked)

    def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
        self.asked.append(SimpleNamespace(question=question, identity=identity, mode=mode, profile=profile))
        if self.raises is not None:
            raise self.raises
        return self.answer

    def scope(self, identity):
        return {"tables": [], "starters": []}


def _doors(tmp_path, **over):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import HttpDoors

    path = tmp_path / "identity-map.json"
    path.write_text(json.dumps(MAP))
    settings = {**OPTION_B, "identity_map_path": str(path), **over}
    return HttpDoors(auth_config_from_settings(Settings(**settings)))


def _static(tmp_path):
    """A built workbench, so every test sees the `/` mount a real deployment has."""
    built = tmp_path / "static"
    built.mkdir(exist_ok=True)
    (built / "index.html").write_text("<!doctype html><title>workbench</title>")
    return built


def _client(tmp_path, rt=None, doors="option-b", **over):
    if doors == "option-b":
        doors = _doors(tmp_path, **over)
    return TestClient(build_app(rt or _RT(), SETTINGS_IDENTITY, static_dir=_static(tmp_path), doors=doors))


def _as(sub="u-fin", task=""):
    headers = {"X-OpenWebUI-User-Jwt": hs256(owui_claims(sub=sub), SECRET.encode()), "X-Mnemiq-Connection": KEY}
    if task is not None:
        headers["X-Mnemiq-Task"] = task
    return headers


def _error(r, status, code=None):
    assert r.status_code == status, (r.status_code, r.text)
    body = r.json()
    assert set(body) == {"error"} and {"message", "type", "param", "code"} <= set(body["error"]), body
    if code is not None:
        assert body["error"]["code"] == code, body
    return body["error"]


# --- Task 2: the mount, the error shape and the model list -------------------------------------------------------


def test_no_issuer_means_no_endpoint(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="mnemiq.server.openai_compat"):
        client = _client(tmp_path, doors=None)
    _error(client.get("/openai/v1/models"), 404)
    _error(client.post("/openai/v1/chat/completions", json={}), 404)  # never the workbench mount's 405
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "MNEMIQ_OPENWEBUI_JWT_SECRET" in said and "MNEMIQ_OPENWEBUI_AUDIENCE" in said


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_an_unmounted_endpoint_is_a_404_for_every_method(tmp_path, method):
    r = _client(tmp_path, doors=None).request(method, "/openai/v1/chat/completions")
    assert r.status_code == 404, (method, r.status_code)


def test_an_issuer_without_an_openai_verifier_means_no_endpoint(tmp_path):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import HttpDoors

    v1_only = HttpDoors(auth_config_from_settings(Settings(tenant="acme", oidc_issuer="https://idp.example/realms/r",
                                                           oidc_audience="mnemiq-api")))
    client = _client(tmp_path, doors=v1_only)
    _error(client.get("/openai/v1/models", headers=_as()), 404)
    _error(client.post("/openai/v1/chat/completions", headers=_as(), json={}), 404)


def test_models_lists_exactly_what_the_identity_may_pick(tmp_path):
    client = _client(tmp_path)
    for sub, expected in (("u-fin", ["default", "finance"]), ("u-ops", ["default", "ops"])):
        r = client.get("/openai/v1/models", headers=_as(sub))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "list"
        assert [m["id"] for m in body["data"]] == expected
        assert all(m["object"] == "model" and m["owned_by"] == "mnemiq" for m in body["data"])


def test_a_caller_with_no_pickable_profile_gets_an_empty_list(tmp_path):
    rt = _RT(profiles=_profiles(default=["admin"], finance=["admin"]))
    assert _client(tmp_path, rt).get("/openai/v1/models", headers=_as()).json()["data"] == []


def test_a_hand_built_runtime_offers_its_default(tmp_path):
    rt = _RT()
    rt.profiles = None  # built by hand, with no profiles loaded
    assert [m["id"] for m in _client(tmp_path, rt).get("/openai/v1/models", headers=_as()).json()["data"]] == [
        "default"]


def test_models_needs_the_task_header(tmp_path):
    error = _error(_client(tmp_path).get("/openai/v1/models", headers=_as(task=None)), 400, "missing_task_header")
    assert "MNEMIQ_OPENWEBUI_TASK_HEADER" in error["message"]


def test_a_door_refusal_is_in_openai_shape(tmp_path):
    plain = {"X-OpenWebUI-User-Id": "u", "X-Mnemiq-Connection": KEY, "X-Mnemiq-Task": ""}
    r = _client(tmp_path).get("/openai/v1/models", headers=plain)
    error = _error(r, 401)
    assert error["type"] == "authentication_error" and "unsigned identity header" in error["message"]
    assert r.headers["www-authenticate"].startswith("Bearer")


def test_no_credential_is_a_401(tmp_path):
    _error(_client(tmp_path).get("/openai/v1/models", headers={"X-Mnemiq-Task": ""}), 401)


def test_every_openai_error_has_openai_shape(tmp_path):
    client = _client(tmp_path)
    get_chat = client.get("/openai/v1/chat/completions", headers=_as())
    _error(get_chat, 405)
    assert get_chat.headers["allow"] == "POST"
    post_models = client.post("/openai/v1/models", headers=_as())
    _error(post_models, 405)
    assert post_models.headers["allow"] == "GET"
    _error(client.post("/openai/v1/embeddings", headers=_as(), json={}), 404)

    class Broken(_RT):
        @property
        def profiles(self):
            raise RuntimeError("store at /srv/secret/path is gone")

        @profiles.setter
        def profiles(self, value):
            pass

    r = _client(tmp_path, Broken()).get("/openai/v1/models", headers=_as())
    error = _error(r, 500)
    assert "secret" not in r.text and "run " in error["message"]


def test_v1_errors_keep_their_own_shape(tmp_path):
    r = _client(tmp_path, doors=None).post("/v1/ask", json={})
    assert r.status_code == 422 and "detail" in r.json()


# --- Task 3: from request to Runtime.ask ------------------------------------------------------------------------


def _chat(client, body=None, headers=None, **over):
    body = body if body is not None else {"model": "default", "messages": [{"role": "user", "content": "how many?"}]}
    return client.post("/openai/v1/chat/completions", headers=headers if headers is not None else _as(),
                       json={**body, **over})


def _refused(r, rt, status, code, *echo):
    error = _error(r, status, code)
    for text in echo:
        assert text not in r.text, (text, r.text)
    assert rt.calls == 0
    return error


def test_the_question_is_the_last_user_message_only(tmp_path):
    rt = _RT()
    messages = [{"role": "system", "content": "SYSTEM-SECRET"}, {"role": "user", "content": "an earlier turn"},
                {"role": "assistant", "content": "an earlier answer"}, {"role": "user", "content": "how many?"}]
    assert _chat(_client(tmp_path, rt), {"model": "default", "messages": messages}).status_code == 200
    assert [a.question for a in rt.asked] == ["how many?"]


def test_the_last_message_must_be_the_users(tmp_path):
    rt = _RT()
    r = _chat(_client(tmp_path, rt), messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    _refused(r, rt, 400, "invalid_last_message")


def test_an_image_part_is_a_400(tmp_path):
    rt = _RT()
    parts = [{"type": "text", "text": "what is this?"},
             {"type": "image_url", "image_url": {"url": "https://h.example/p.png"}}]
    error = _refused(_chat(_client(tmp_path, rt), messages=[{"role": "user", "content": parts}]), rt, 400,
                     "unsupported_content", "h.example", "what is this")
    assert "image" in error["message"]


def test_any_other_part_type_is_a_400(tmp_path):
    rt = _RT()
    parts = [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]
    _refused(_chat(_client(tmp_path, rt), messages=[{"role": "user", "content": parts}]), rt, 400,
             "unsupported_content")


def test_text_parts_are_joined_by_a_newline(tmp_path):
    rt = _RT()
    parts = [{"type": "text", "text": "claims"}, {"type": "text", "text": "by month"}]
    assert _chat(_client(tmp_path, rt), messages=[{"role": "user", "content": parts}]).status_code == 200
    assert rt.asked[0].question == "claims\nby month"


def test_an_empty_question_is_a_400(tmp_path):
    rt = _RT()
    _refused(_chat(_client(tmp_path, rt), messages=[{"role": "user", "content": "   "}]), rt, 400, "empty_question")


def test_a_message_with_the_rag_templates_markers_is_a_400(tmp_path):
    rt = _RT()
    templated = "### Task:\nRespond using the context.\n<context>\nPolicy 12 was cancelled.\n</context>\nhow many?"
    error = _refused(_chat(_client(tmp_path, rt), messages=[{"role": "user", "content": templated}]), rt, 400,
                     "retrieval_template", "Policy 12")
    assert "knowledge" in error["message"] and "web search" in error["message"]
    assert _chat(_client(tmp_path, rt), messages=[{"role": "user", "content": "### Task: count claims"}]).status_code == 200
    assert rt.calls == 1


@pytest.mark.parametrize("effort, mode", [("low", "instant"), ("medium", "thinking"), ("high", "deep"), (None, None)])
def test_reasoning_effort_picks_the_mode(tmp_path, effort, mode):
    rt = _RT()
    body = {"model": "default", "messages": [{"role": "user", "content": "q"}]}
    if effort is not None:
        body["reasoning_effort"] = effort
    assert _chat(_client(tmp_path, rt), body).status_code == 200
    assert rt.asked[0].mode == mode


def test_another_effort_is_a_400_listing_the_offered(tmp_path):
    rt = _RT()
    error = _refused(_chat(_client(tmp_path, rt), reasoning_effort="minimal"), rt, 400, "invalid_reasoning_effort",
                     "minimal")
    assert "['high', 'low', 'medium']" in error["message"]


@pytest.mark.parametrize("field, value", [("tools", [{"type": "function", "function": {"name": "f"}}]),
                                          ("functions", [{"name": "f"}]), ("function_call", "auto")])
def test_tools_are_refused(tmp_path, field, value):
    rt = _RT()
    _refused(_chat(_client(tmp_path, rt), **{field: value}), rt, 400, "unsupported_parameter")


def test_more_than_one_choice_is_refused(tmp_path):
    rt = _RT()
    _refused(_chat(_client(tmp_path, rt), n=2), rt, 400, "unsupported_parameter")


def test_ignored_parameters_change_nothing(tmp_path):
    rt = _RT()
    r = _chat(_client(tmp_path, rt), temperature=0.1, top_p=0.5, max_tokens=5, stop=["\n"], user="x",
              tool_choice="none", stream_options={"include_usage": True})
    assert r.status_code == 200 and rt.calls == 1


def test_a_background_task_never_reaches_the_engine(tmp_path, caplog):
    rt = _RT()
    with caplog.at_level(logging.INFO, logger="mnemiq.server.openai_compat"):
        r = _chat(_client(tmp_path, rt), headers=_as(task="title_generation"))
    assert r.status_code == 200 and rt.calls == 0
    assert r.json()["choices"][0]["message"]["content"] == "" and r.json()["choices"][0]["finish_reason"] == "stop"
    assert any("title_generation" in rec.getMessage() for rec in caplog.records)


def test_a_task_name_outside_open_webuis_is_logged_as_other(tmp_path, caplog):
    rt = _RT()
    with caplog.at_level(logging.INFO, logger="mnemiq.server.openai_compat"):
        assert _chat(_client(tmp_path, rt), headers=_as(task="<script>alert(1)</script>")).status_code == 200
    said = " ".join(rec.getMessage() for rec in caplog.records)
    assert "<script>" not in said and "(other)" in said and rt.calls == 0


def test_a_chat_turn_carries_an_empty_task_header(tmp_path):
    rt = _RT()
    assert _chat(_client(tmp_path, rt), headers=_as(task="")).status_code == 200 and rt.calls == 1


def test_a_missing_task_header_is_a_400(tmp_path):
    rt = _RT()
    _refused(_chat(_client(tmp_path, rt), headers=_as(task=None)), rt, 400, "missing_task_header")


def test_an_unpickable_model_is_a_404_that_does_not_echo_it(tmp_path):
    rt = _RT()
    error = _refused(_chat(_client(tmp_path, rt), model="secret-profile"), rt, 404, "model_not_found",
                     "secret-profile")
    assert "['default', 'finance']" in error["message"]
    _refused(_chat(_client(tmp_path, rt), headers=_as("u-ops"), model="finance"), rt, 404, "model_not_found")


def test_a_pickable_profile_reaches_ask_by_name(tmp_path):
    rt = _RT()
    assert _chat(_client(tmp_path, rt), model="finance").status_code == 200
    assert rt.asked[0].profile == "finance" and rt.asked[0].identity.principal_id == "open-webui:u-fin"


def test_a_hand_built_runtime_asks_its_default_as_no_profile(tmp_path):
    rt = _RT()
    rt.profiles = None
    assert _chat(_client(tmp_path, rt)).status_code == 200 and rt.asked[0].profile is None


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"model": "default"}', '{"model": "default", "messages": []}',
                                 '{"model": "default", "messages": [{"content": "q"}]}',
                                 '{"messages": [{"role": "user", "content": "q"}]}'])
def test_a_malformed_body_is_an_openai_400(tmp_path, raw):
    rt = _RT()
    r = _client(tmp_path, rt).post("/openai/v1/chat/completions", headers={**_as(), "content-type": "application/json"},
                                   content=raw)
    _refused(r, rt, 400, "invalid_body")


def test_a_malformed_body_without_a_credential_is_still_a_401(tmp_path):
    r = _client(tmp_path).post("/openai/v1/chat/completions", headers={"X-Mnemiq-Task": ""}, content="not json")
    _error(r, 401)

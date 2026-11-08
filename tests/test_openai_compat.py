"""/openai/v1: the way into mnemiq for Open WebUI, or any OpenAI client, with each request answered as its own
verified caller, one model per profile that caller may pick."""

import json
import logging
import re
import threading
import time
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


@pytest.mark.parametrize("path", ["/openai/v1/chat/completions", "/openai/v1"])
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "PROPFIND"])
def test_an_unmounted_endpoint_is_a_404_for_every_method(tmp_path, method, path):
    """Every method, an extension method included, and the bare prefix too: none may fall through to the
    workbench mount, which answers a non-GET with 405."""
    r = _client(tmp_path, doors=None).request(method, path)
    assert r.status_code == 404, (method, path, r.status_code)


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
    _error(client.post("/openai/v1", headers=_as(), json={}), 404)
    trace = client.request("TRACE", "/openai/v1/models", headers=_as())
    assert trace.status_code == 405 and trace.headers["allow"] == "GET"

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


# --- Task 4: the reply -----------------------------------------------------------------------------------------


def _answer(answer="There were 25 claims.", sql="SELECT id FROM claim", columns=("id",), rows=None, row_count=None,
            truncated=False, capped=False, **over):
    from mnemiq.agent.loop import ResultPreview
    from mnemiq.contract.seams import Trace

    rows = rows if rows is not None else [[i] for i in range(3)]
    trace = Trace(question="q", plan_sql=sql, target_sql=sql, asked_sql=sql, result_shape="table", timing={},
                  enrichment_version="e1", identity=IdentityContext(tenant_id="acme", principal_id="p"))
    preview = ResultPreview(columns=list(columns), rows=rows, row_count=len(rows) if row_count is None else row_count,
                            truncated=truncated, capped=capped)
    return AgentAnswer(answer=answer, trace=trace, preview=preview, mode="thinking", **over)


def _content(tmp_path, answer, **body):
    r = _chat(_client(tmp_path, _RT(answer=answer)), **body)
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["message"]["content"]


def test_the_reply_reads_in_the_specs_order(tmp_path):
    r = _chat(_client(tmp_path, _RT(answer=_answer(verify_layer="judge"))))
    content, run_id = r.json()["choices"][0]["message"]["content"], r.json()["id"].removeprefix("chatcmpl-")
    at = [content.index(part) for part in ("There were 25 claims", "**Status:**", "```sql", "| id |", f"run {run_id}")]
    assert at == sorted(at), content
    assert r.json()["object"] == "chat.completion" and r.json()["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in r.json()


@pytest.mark.parametrize("layer, line", [
    ("judge", "**Status:** verified by the judge"),
    ("sanity", "**Status:** basic checks only, not judged"),
    ("judge_unavailable", "**Status:** not verified: the judge could not be reached"),
    ("a-new-layer", "**Status:** verification state unknown"),
    (None, "**Status:** not verified"),
])
def test_the_status_line(tmp_path, layer, line):
    content = _content(tmp_path, _answer(verify_layer=layer))
    assert line in content.splitlines(), content
    if layer != "judge":
        assert "verified by" not in content


def test_a_deferral_and_a_failure_say_so(tmp_path):
    from mnemiq.contract import DeferralReason

    deferred = AgentAnswer(answer="No table holds salary.", deferred=True, reason_code=DeferralReason.NO_TABLES)
    assert "**Status:** deferred (no_tables)" in _content(tmp_path, deferred).splitlines()
    failed = AgentAnswer(answer="The source did not answer.", failed=True,
                         reason_code=DeferralReason.EXECUTION_FAILED)
    assert "**Status:** failed (execution_failed)" in _content(tmp_path, failed).splitlines()


def test_the_table_shows_twenty_rows_of_n(tmp_path):
    content = _content(tmp_path, _answer(rows=[[i] for i in range(25)]))
    assert "| 19 |" in content and "| 20 |" not in content
    assert "20 of 25 rows" in content


def test_a_capped_result_says_at_least(tmp_path):
    content = _content(tmp_path, _answer(rows=[[i] for i in range(25)], row_count=1000, truncated=True, capped=True))
    assert "20 of at least 1,000 rows" in content


def test_a_short_result_says_its_count(tmp_path):
    assert "3 rows" in _content(tmp_path, _answer()).splitlines()
    assert "1 row" in _content(tmp_path, _answer(rows=[[1]])).splitlines()
    assert "0 rows" in _content(tmp_path, _answer(rows=[])).splitlines()


def test_a_cell_cannot_break_the_table(tmp_path):
    content = _content(tmp_path, _answer(columns=("v",), rows=[["a|b"], ["a\nb"], ["a\rb"], ["tail\\"], [None]]))
    table = [line for line in content.splitlines() if line.startswith("|")]
    assert len(table) == 2 + 5, table  # header, separator, five rows: nothing split or merged
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 2 for line in table), table  # an escaped pipe splits nothing
    assert "| a\\|b |" in table and "| a b |" in table and "| tail\\\\ |" in table and "|  |" in table


ESCAPED = "\\`*]!<>|$&~_"


@pytest.mark.parametrize("char", list(ESCAPED))
def test_every_escaped_character_is_inert_in_cells_headers_and_prose(tmp_path, char):
    """Each character that starts a link, an image, HTML, math, a code span, emphasis or strikethrough, or splits a
    cell, arrives backslash-escaped, in the answer text, a column header and a cell alike. Not as a character
    reference, which Open WebUI shows as written in a table cell (found in the live check)."""
    ref = "\\" + char
    lines = _content(tmp_path, _answer(answer=f"a{char}b", columns=(f"h{char}h",), rows=[[f"c{char}c"]])).splitlines()
    assert lines[0] == (f"a{ref}b" if char != "\\" else "a\uff3cb")  # the answer's own backslash goes full-width
    table = [line for line in lines if line.startswith("|")]
    assert table == [f"| h{ref}h |", "| --- |", f"| c{ref}c |"], table


# An odd run of backslashes before `(` or `[`: what Open WebUI's KaTeX reads as the start of math.
MATH_START = re.compile(r"(?<!\\)(?:\\\\)*\\[([]")
# A `$`, `<`, `!` or `]` with no backslash escaping it: math, HTML, an image or the end of a link.
UNESCAPED = re.compile(r"(?<!\\)(?:\\\\)*[$<!\]]")


def test_data_and_prose_never_render_as_a_link_image_html_or_math(tmp_path):
    values = ["![x](https://h.example/?q=1)", "<img src=x>", "$1,200 and $3,400", "a\\(b\\)", "\\[x\\]"]
    content = _content(tmp_path, _answer(answer=" ".join(values), columns=(values[0],), rows=[[v] for v in values]))
    outside = content.split("```")[0] + content.split("```")[-1]
    assert not MATH_START.search(outside), outside
    assert not UNESCAPED.search(outside), outside
    for raw, shown in (("$1,200 and $3,400", "\\$1,200 and \\$3,400"), ("![x](https://h.example/?q=1)",
                                                                       "\\![x\\](https\\://h.example/?q=1)"),
                       ("<img src=x>", "\\<img src=x\\>"), ("a\\(b\\)", "a\\\\(b\\\\)"), ("A&B $5", "A\\&B \\$5")):
        assert shown in _content(tmp_path, _answer(columns=("v",), rows=[[raw]])), (raw, shown)
    # `(` and `[` stay as written: escaping either would open math in Open WebUI. The `]` is escaped, and renders as `]`.
    assert "revenue (net) 1.5 [draft\\]" in _content(tmp_path, _answer(answer="revenue (net) 1.5 [draft]"))


def test_an_answer_never_carries_a_backslash_that_can_open_math(tmp_path):
    """Open WebUI's KaTeX also scans whole paragraphs for `\\[`, and Marked splits a paragraph there, so a doubled
    backslash from the answer can be cut in two and open math (found in review, reproduced with Marked 9 and Open
    WebUI's extension). In the answer text every ASCII backslash is this layer's own escape, before a character
    that opens nothing: a backslash the answer itself holds becomes the full-width one."""
    for raw in ("\\[x\\]", "see \\[x+1\\] here", "\\begin{equation}x\\end{equation}", "a\\(b\\)", "C:\\code\\path", "\\\\[x]"):
        prose = _content(tmp_path, _answer(answer=raw)).split("\n\n**Status:**")[0]
        assert not re.search(r"\\(?![`*\]!<>|$&~_:@.])", prose), (raw, prose)
    assert _content(tmp_path, _answer(answer="\\[x\\]")).startswith("\uff3c[x\uff3c\\]")
    # A cell keeps the data's backslashes, doubled: Marked lexes a cell inline only, where a doubled one is safe.
    assert "| a\\\\(b\\\\) |" in _content(tmp_path, _answer(columns=("v",), rows=[["a\\(b\\)"]]))


@pytest.mark.parametrize("raw, shown", [("go to https://evil.example/x", "go to https\\://evil.example/x"),
                                        ("ftp://evil.example/f", "ftp\\://evil.example/f"),
                                        ("go to www.evil.example", "go to www\\.evil.example"),
                                        ("WWW.evil.example", "WWW\\.evil.example"),
                                        ("mail a@evil.example", "mail a\\@evil.example")])
def test_a_bare_url_or_email_never_becomes_a_link(tmp_path, raw, shown):
    """GFM turns a bare `scheme://`, `www.` or email address into a link, in a cell as in a paragraph (found in
    review, reproduced with Marked 9). An escaped `:`, `.` after `www` or `@` breaks each, and opens no math."""
    content = _content(tmp_path, _answer(answer=raw, columns=(raw,), rows=[[raw]]))
    assert content.splitlines()[0] == shown
    assert f"| {shown} |" in content.splitlines()  # the header
    assert content.count(shown) == 3  # answer, header and cell


def test_the_sql_fence_outlasts_backticks_in_the_sql(tmp_path):
    content = _content(tmp_path, _answer(sql="SELECT '```' AS ticks"))
    assert "````sql\nSELECT '```' AS ticks\n````" in content
    assert "```sql\nSELECT id FROM claim\n```" in _content(tmp_path, _answer())


def test_the_answer_keeps_its_lines_without_indentation(tmp_path):
    content = _content(tmp_path, _answer(answer="First line.\n        indented line\r\nlast"))
    assert content.startswith("First line.\nindented line\nlast")


def test_a_decline_is_a_200_whose_text_is_the_reason(tmp_path):
    from mnemiq.contract import DeferralReason

    declined = _answer(answer="No table holds salary.", deferred=True, reason_code=DeferralReason.NO_TABLES)
    content = _content(tmp_path, declined)
    assert content.startswith("No table holds salary.") and "```" not in content and "| id |" not in content


def test_model_error_never_reaches_the_wire(tmp_path):
    r = _chat(_client(tmp_path, _RT(answer=_answer(model_error="connect: 10.0.0.5:8000"))))
    assert r.status_code == 200 and "10.0.0.5" not in r.text and "connect:" not in r.text


def test_an_over_long_question_is_a_400(tmp_path):
    from mnemiq.contract import DeferralReason

    text = "Your question is 9,001 characters long; the limit is 9,000."
    rt = _RT(answer=AgentAnswer(answer=text, deferred=True, reason_code=DeferralReason.QUESTION_TOO_LONG))
    error = _error(_chat(_client(tmp_path, rt)), 400, "question_too_long")
    assert error["message"] == text and rt.calls == 1


@pytest.mark.parametrize("raised, status, code", [(None, 404, "model_not_found"), ("mode", 400, None),
                                                  ("boom", 500, None)])
def test_engine_refusals_are_composed(tmp_path, raised, status, code):
    from mnemiq.agent.route import UnknownMode
    from mnemiq.profiles import UnknownProfile

    exc = {None: UnknownProfile(["default"]), "mode": UnknownMode("the store at /srv/x is gone"),
           "boom": RuntimeError("the store at /srv/x is gone")}[raised]
    r = _chat(_client(tmp_path, _RT(raises=exc)))
    error = _error(r, status, code)
    assert "/srv/x" not in r.text
    if status == 500:
        assert "run " in error["message"]


# --- Task 5: the stream ----------------------------------------------------------------------------------------


class _Streaming(_RT):
    """Emits `events` (a float is a pause, in seconds), then answers, or raises `raises_after`."""

    def __init__(self, events=(), raises_after=None, **kw):
        super().__init__(**kw)
        self.events, self.raises_after = list(events), raises_after

    def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
        self.asked.append(SimpleNamespace(question=question, identity=identity, mode=mode, profile=profile))
        if self.raises is not None:
            raise self.raises
        for event in self.events:
            if isinstance(event, float):
                time.sleep(event)
            else:
                emit(event)
        if self.raises_after is not None:
            raise self.raises_after
        return self.answer


STAGES = [{"phase": "started", "stage": "retrieve", "id": 1},
          {"phase": "finished", "stage": "retrieve", "id": 1, "ms": 12.3456, "ok": True},
          {"kind": "result", "trace": None, "preview": None, "notes": []},
          {"phase": "finished", "stage": "generate", "id": 2, "ms": 340.0, "ok": True}]


def _sse(r) -> list:
    out = []
    for line in r.text.splitlines():
        assert not line.startswith(":"), "no SSE comment lines: Open WebUI's handling of them is not verified"
        if line.startswith("data: "):
            payload = line.removeprefix("data: ")
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def _deltas(chunks) -> list[dict]:
    return [c["choices"][0]["delta"] for c in chunks if c != "[DONE]"]


def test_a_stream_is_role_progress_answer_stop_done(tmp_path):
    rt = _Streaming(STAGES, answer=_answer(verify_layer="judge"))
    r = _chat(_client(tmp_path, rt), stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    chunks = _sse(r)
    run_id = chunks[0]["id"].removeprefix("chatcmpl-")
    assert all(c["object"] == "chat.completion.chunk" and c["model"] == "default" for c in chunks[:-1])
    deltas = _deltas(chunks)
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert deltas[1:3] == [{"reasoning_content": "retrieve (12 ms)\n"}, {"reasoning_content": "generate (340 ms)\n"}]
    from mnemiq.server.openai_compat import render
    from mnemiq.server.serialize import answer_payload

    assert deltas[3] == {"content": render(answer_payload(rt.answer), run_id)}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop" and deltas[-1] == {}
    assert chunks[-1] == "[DONE]" and len(chunks) == 6


def test_a_failed_stage_says_so(tmp_path):
    failed = {"phase": "finished", "stage": "execute", "id": 3, "ms": 5.0, "ok": False}
    deltas = _deltas(_sse(_chat(_client(tmp_path, _Streaming([failed])), stream=True)))
    assert {"reasoning_content": "execute failed (5 ms)\n"} in deltas


def test_a_slow_engine_gets_empty_reasoning_keepalives(tmp_path):
    rt = _Streaming([STAGES[1], 0.15])
    client = TestClient(build_app(rt, SETTINGS_IDENTITY, heartbeat_s=0.02, static_dir=_static(tmp_path),
                                  doors=_doors(tmp_path)))
    deltas = _deltas(_sse(_chat(client, stream=True)))
    assert deltas.count({"reasoning_content": ""}) >= 2, deltas


def test_an_engine_failure_mid_stream_is_a_composed_message_then_stop(tmp_path):
    rt = _Streaming([STAGES[1]], raises_after=RuntimeError("the store at /srv/secret is gone"))
    r = _chat(_client(tmp_path, rt), stream=True)
    chunks = _sse(r)
    run_id = chunks[0]["id"].removeprefix("chatcmpl-")
    content = [d["content"] for d in _deltas(chunks) if d.get("content")]
    assert content == [f"The engine could not complete this request (run {run_id})."]
    assert "/srv/secret" not in r.text and chunks[-1] == "[DONE]"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"


def test_a_task_streams_empty(tmp_path):
    rt = _Streaming(STAGES)
    chunks = _sse(_chat(_client(tmp_path, rt), headers=_as(task="title_generation"), stream=True))
    assert _deltas(chunks) == [{"role": "assistant", "content": ""}, {}] and chunks[-1] == "[DONE]"
    assert rt.calls == 0


def test_a_streamed_over_long_question_is_a_400(tmp_path):
    from mnemiq.contract import DeferralReason

    text = "Your question is 9,001 characters long; the limit is 9,000."
    rt = _Streaming(answer=AgentAnswer(answer=text, deferred=True, reason_code=DeferralReason.QUESTION_TOO_LONG))
    assert _error(_chat(_client(tmp_path, rt), stream=True), 400, "question_too_long")["message"] == text


def test_a_streamed_refusal_before_any_event_has_its_status(tmp_path):
    from mnemiq.profiles import UnknownProfile

    r = _chat(_client(tmp_path, _Streaming(raises=UnknownProfile(["default"]))), stream=True)
    _error(r, 404, "model_not_found")


def test_a_stream_of_an_engine_that_reports_nothing_still_answers(tmp_path):
    rt = _Streaming([], answer=_answer())
    deltas = _deltas(_sse(_chat(_client(tmp_path, rt), stream=True)))
    assert deltas[0] == {"role": "assistant", "content": ""} and "There were 25 claims" in deltas[1]["content"]


@pytest.mark.parametrize("how", ["disconnect", "send-raises"])
def test_a_client_hanging_up_mid_stream_logs_no_error(tmp_path, caplog, how):
    """TestClient runs the app to completion before it answers, so the hang-up is driven through ASGI directly:
    a `receive` that reports the disconnect after the first chunk (ASGI 2.3, where Starlette listens for it), or a
    `send` that raises as a closed socket does (ASGI 2.4)."""
    import anyio

    release, finished = threading.Event(), threading.Event()

    class Held(_RT):
        """Emits a stage, waits until the client has gone, emits another, and answers."""

        def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
            self.asked.append(SimpleNamespace(question=question))
            emit(STAGES[1])
            release.wait(5)
            emit(STAGES[3])  # onto a queue nobody reads any more
            finished.set()
            return self.answer

    rt = Held()
    app = build_app(rt, SETTINGS_IDENTITY, static_dir=_static(tmp_path), doors=_doors(tmp_path))
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": "q"}], "stream": True}).encode()
    headers = [(k.lower().encode(), v.encode()) for k, v in _as().items()] + [(b"content-type", b"application/json")]
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3" if how == "disconnect" else "2.4"},
             "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/openai/v1/chat/completions",
             "raw_path": b"/openai/v1/chat/completions", "root_path": "", "query_string": b"", "headers": headers,
             "client": ("127.0.0.1", 1), "server": ("testserver", 80), "state": {}}

    async def scenario():
        bodies, read, first = [], [False], anyio.Event()

        async def receive():
            if not read[0]:
                read[0] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await first.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                bodies.append(message["body"])
                if how == "send-raises" and len(bodies) >= 2:
                    raise OSError("the client went away")
                first.set()

        try:
            await app(scope, receive, send)
        except Exception:  # noqa: BLE001 -- what the server does with a hang-up is the server's; logging is ours
            pass
        release.set()
        await anyio.to_thread.run_sync(finished.wait, 5)
        return bodies

    with caplog.at_level(logging.DEBUG):
        bodies = anyio.run(scenario)
    assert bodies and finished.is_set()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR and r.name.startswith("mnemiq")]

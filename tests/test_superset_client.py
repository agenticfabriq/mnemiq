"""The client mnemiq calls Superset's MCP service with: one caller's bearer on every request, and every failure read
into mnemiq's own sentence, with Superset's words kept in `.detail` for the log.

Every test runs against `FakeSuperset`, a real MCP server over streamable HTTP on loopback, with the SDK's own
transport in between. Runs on mcp 1.x (the lock) and 2.x (the mcp-2x workflow). Superset puts `Error: ` before every
error's text, as the fake does by default; each way an error's text is read is checked with and without it.
"""

from __future__ import annotations

import socket
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mnemiq.superset.client import (
    Bearer,
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
from tests.superset_fake import FakeSuperset

TOKEN = "tok"
REQUEST = {"database_id": 1, "sql": "SELECT 1"}
PAYLOAD = {"success": True, "rows": [{"n": 1}], "columns": [{"name": "n", "type": "int64"}], "row_count": 1}

# Superset's own texts, as its MCP service renders them (`superset/mcp_service/auth.py` and
# `superset/mcp_service/middleware.py`, `GlobalErrorHandlerMiddleware`).
ROLE_DENIAL = "Permission denied: can_execute_sql_query on SQLLab for user west (tool: execute_sql)"
GENERIC_DENIAL = "Permission denied for execute_sql: You don't have access to this resource."
UNKNOWN_USER = ("Invalid parameter in execute_sql: JWT authenticated user not found in Superset database. "
                "Ensure the user exists before granting MCP access.")
# The error Superset's response size guard raises for an answer too large to send that it does not cut
# (`format_size_limit_error` in `superset/mcp_service/utils/response_size_utils.py`).
TOO_LARGE = ("Response too large: 61,234 bytes (limit: 50,000)\n\nThis response would overwhelm the LLM context window.\n"
             "Please modify your query to reduce the response size:\n\n1. Add a LIMIT clause to your SQL query\n\n"
             "Tool: execute_sql\nReduction needed: ~18%")


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        yield served


@pytest.fixture(params=["prefixed", "bare"])
def any_shape(request):
    """The fake Superset sending each error's text as Superset does, after `Error: `, and without it."""
    with FakeSuperset(bare_errors=request.param == "bare") as served:
        yield served


def _client(url: str, token: str = TOKEN, **kwargs) -> SupersetMcp:
    return SupersetMcp(url, DelegatedToken(token), **kwargs)


def _raised_for(fake, error: Exception, tool: str = "execute_sql") -> Exception:
    """What the client raises when the tool answers with `error`'s text as an error result."""
    fake.reply(tool, lambda request, authorization: error)
    with pytest.raises(Exception) as caught:
        _client(fake.url).call(tool, REQUEST)
    return caught.value


def _holds_no_credential(exc: BaseException) -> None:
    """The bearer appears in none of what an exception shows: its message, its repr or its detail."""
    for shown in (str(exc), repr(exc), getattr(exc, "detail", "")):
        assert TOKEN not in shown, shown


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sends_the_bearer_and_wraps_the_request(fake):
    fake.reply("execute_sql", lambda request, authorization: PAYLOAD)
    _client(fake.url).call("execute_sql", REQUEST)
    assert len(fake.calls) == 1
    assert fake.calls[0].tool == "execute_sql"
    assert fake.calls[0].authorization == "Bearer tok"
    assert fake.calls[0].request == {"database_id": 1, "sql": "SELECT 1"}


def test_returns_the_json_payload(fake):
    fake.reply("list_datasets", lambda request, authorization: {"datasets": [{"id": 3}], "count": 1})
    assert _client(fake.url).call("list_datasets", {"page": 1}) == {"datasets": [{"id": 3}], "count": 1}


def test_reads_the_text_when_superset_sends_no_structured_content():
    """Superset can be configured to send a tool's answer only as JSON text."""
    with FakeSuperset(structured_output=False) as fake:
        fake.reply("get_database_info", lambda request, authorization: {"id": 1, "backend": "postgresql"})
        assert _client(fake.url).call("get_database_info", {"identifier": 1}) == {"id": 1, "backend": "postgresql"}


def test_an_answer_that_is_not_a_json_object_is_a_tool_error():
    with FakeSuperset(structured_output=False) as fake:
        fake.reply("execute_sql", lambda request, authorization: ["not", "an", "object"])
        with pytest.raises(SupersetToolError) as caught:
            _client(fake.url).call("execute_sql", REQUEST)
    assert str(caught.value) == "Superset's answer to execute_sql was not a JSON object"
    _holds_no_credential(caught.value)


def test_401_is_a_refused_caller(fake):
    fake.refuse_token(TOKEN)
    with pytest.raises(SupersetRefusedCaller) as caught:
        _client(fake.url).call("execute_sql", REQUEST)
    assert "HTTP 401" in str(caught.value)
    assert fake.calls == []  # refused at the transport, before any tool ran
    _holds_no_credential(caught.value)


def test_403_at_the_transport_is_a_refused_caller(fake):
    fake.fail_http(403)
    with pytest.raises(SupersetRefusedCaller) as caught:
        _client(fake.url).call("execute_sql", REQUEST)
    assert "HTTP 403" in str(caught.value)
    _holds_no_credential(caught.value)


def test_a_user_superset_does_not_know_is_a_refused_caller(any_shape):
    exc = _raised_for(any_shape, ValueError(UNKNOWN_USER))
    assert type(exc) is SupersetRefusedCaller
    assert str(exc) == "Superset has no user for the credential this request carries"
    assert exc.detail == any_shape.error_text(UNKNOWN_USER)
    _holds_no_credential(exc)


def test_permission_denied_is_its_own_refusal(any_shape):
    exc = _raised_for(any_shape, PermissionError(ROLE_DENIAL))
    assert type(exc) is SupersetPermissionDenied
    assert isinstance(exc, SupersetRefusal)
    # Superset's text names the permission, the view and the user; it rides only in `.detail`, whole.
    assert exc.detail == any_shape.error_text(ROLE_DENIAL)
    assert "SQLLab" not in str(exc) and "west" not in str(exc)
    _holds_no_credential(exc)


def test_a_generic_permission_error_claims_no_cause(any_shape):
    generic = _raised_for(any_shape, PermissionError(GENERIC_DENIAL))
    assert type(generic) is SupersetDenied
    assert str(generic) == "Superset refused execute_sql and did not say why"
    assert generic.detail == any_shape.error_text(GENERIC_DENIAL)
    _holds_no_credential(generic)

    any_shape.refuse_token(TOKEN)
    with pytest.raises(SupersetRefusal) as caught:
        _client(any_shape.url).call("execute_sql", REQUEST)
    assert type(caught.value) is SupersetRefusedCaller
    for exc in (generic, caught.value):
        assert not isinstance(exc, SupersetPermissionDenied)


def test_a_denial_text_is_read_only_at_its_start(any_shape):
    """A tool error that merely quotes a denial is not one: the shapes are matched where Superset writes them."""
    exc = _raised_for(any_shape, RuntimeError(f"Invalid parameter in execute_sql: {ROLE_DENIAL}"))
    assert type(exc) is SupersetToolError


def test_one_error_prefix_is_read_past_and_no_more(fake):
    """Superset puts one `Error: ` before an error's text; a text that opens with another after it does not open with
    the denial."""
    exc = _raised_for(fake, PermissionError(f"Error: {ROLE_DENIAL}"))
    assert type(exc) is SupersetToolError
    assert exc.detail == f"Error: Error: {ROLE_DENIAL}"


def test_unknown_tool_text_names_the_tool_search_setting(any_shape):
    """The text Superset's MCP service sends for a tool it does not offer by name, read in either shape."""
    exc = _raised_for(any_shape, RuntimeError("Unknown tool: execute_sql"))
    assert type(exc) is SupersetToolError
    assert 'MCP_TOOL_SEARCH_CONFIG["enabled"] = False' in str(exc)
    assert exc.detail == any_shape.error_text("Unknown tool: execute_sql")


def test_a_response_too_large_is_its_own_tool_error(any_shape):
    """An answer larger than Superset's MCP service will send, refused whole: a tool error of its own kind, which a
    caller paging a listing reads as a page to ask for again smaller. Superset's text, with its sizes, rides only in
    `.detail`."""
    exc = _raised_for(any_shape, RuntimeError(TOO_LARGE), tool="list_datasets")
    assert type(exc) is SupersetResponseTooLarge
    assert isinstance(exc, SupersetToolError)
    assert str(exc) == "Superset's answer to list_datasets was larger than its MCP service will send; ask for less"
    assert exc.detail == any_shape.error_text(TOO_LARGE)
    assert "61,234" not in str(exc)


def test_unknown_tool_names_the_tool_search_setting(fake):
    with pytest.raises(SupersetToolError) as caught:
        _client(fake.url).call("search_tools_hid_this", {})
    assert 'MCP_TOOL_SEARCH_CONFIG["enabled"] = False' in str(caught.value)
    assert "search_tools_hid_this" in str(caught.value)
    _holds_no_credential(caught.value)


def test_any_other_tool_error_is_a_tool_error(any_shape):
    text = "Datasource error in execute_sql: the query against the underlying datasource failed (secret_table)"
    exc = _raised_for(any_shape, RuntimeError(text))
    assert type(exc) is SupersetToolError
    assert str(exc) == "Superset's execute_sql tool failed"
    assert exc.detail == any_shape.error_text(text)
    _holds_no_credential(exc)


def test_a_server_error_is_unavailable(fake):
    fake.fail_http(502)
    with pytest.raises(SupersetUnavailable) as caught:
        _client(fake.url).call("execute_sql", REQUEST)
    assert "HTTP 502" in str(caught.value)
    _holds_no_credential(caught.value)


@pytest.mark.parametrize("content_type", ["text/html", "application/json"])
def test_an_answer_that_is_not_json_rpc_is_unavailable_at_once(fake, content_type):
    """A 200 that is not a JSON-RPC message (a proxy's sign-in page, or JSON of some other shape) is no answer at all.
    mcp 2.x reports it as a JSON-RPC error of its own making, and 1.x hands it to the session's message handler;
    neither is Superset saying something about the tool, and neither should wait out the deadline."""
    fake.fail_http(200, content_type=content_type)
    started = time.monotonic()
    with pytest.raises(SupersetUnavailable) as caught:
        _client(fake.url, timeout_s=30).call("execute_sql", REQUEST)
    # Half the deadline: a call that fails at once takes a few seconds at most, even on a loaded machine.
    assert time.monotonic() - started < 15
    assert str(caught.value) == "Superset's MCP service gave no usable answer to execute_sql"
    _holds_no_credential(caught.value)


def test_an_event_stream_closed_before_its_result_is_unavailable(fake):
    """mcp 2.x resolves the call with a JSON-RPC "connection closed" error of its own; 1.x waits out the deadline."""
    fake.close_stream_before_result()
    with pytest.raises(SupersetUnavailable) as caught:
        _client(fake.url, timeout_s=3).call("execute_sql", REQUEST)
    assert str(caught.value) in ("Superset's MCP service gave no usable answer to execute_sql",
                                 "Superset's MCP service did not answer execute_sql within 3 seconds")
    _holds_no_credential(caught.value)


def test_no_json_rpc_error_reads_as_superset_answering():
    """Superset sends its refusals and tool failures as error results, never as JSON-RPC errors. A JSON-RPC error is
    the SDK's own report of a transport problem (a closed connection, an answer of the wrong type, one it could not
    parse) or a protocol-level failure, so whatever its code and text, it reads as Superset being unavailable."""
    import mcp
    from mcp.types import CONNECTION_CLOSED, INVALID_PARAMS, INVALID_REQUEST, PARSE_ERROR, ErrorData

    from mnemiq.superset.client import _Exchange

    error_class = getattr(mcp, "MCPError", None) or getattr(mcp, "McpError")

    def jsonrpc_error(code: int, message: str) -> Exception:
        try:
            return error_class(code, message)  # mcp 2.x
        except TypeError:
            return error_class(ErrorData(code=code, message=message))  # mcp 1.x

    errors = [(CONNECTION_CLOSED, "Connection closed"), (INVALID_REQUEST, "Unexpected content type: text/html"),
              (PARSE_ERROR, "Failed to parse JSON response"), (INVALID_PARAMS, ROLE_DENIAL),
              (INVALID_PARAMS, "Unknown tool: execute_sql")]
    client = _client("http://127.0.0.1:1/mcp")
    for code, message in errors:
        for statuses, sentence in (([200, 202], "Superset's MCP service gave no usable answer to execute_sql"),
                                   ([], "Superset's MCP service could not be reached")):
            exchange = _Exchange(statuses=statuses, error=ExceptionGroup("t", [jsonrpc_error(code, message)]))
            exc = client._failure("execute_sql", exchange)
            assert type(exc) is SupersetUnavailable, (code, message, type(exc))
            assert str(exc) == sentence, (code, message)


def test_connection_refused_is_unavailable():
    with pytest.raises(SupersetUnavailable) as caught:
        _client(f"http://127.0.0.1:{_closed_port()}/mcp").call("execute_sql", REQUEST)
    assert str(caught.value) == "Superset's MCP service could not be reached"
    assert caught.value.detail  # what the transport said, for the log
    _holds_no_credential(caught.value)


def test_slow_server_is_unavailable_within_the_timeout(fake):
    def slow(request, authorization):
        time.sleep(3)
        return PAYLOAD

    fake.reply("execute_sql", slow)
    started = time.monotonic()
    with pytest.raises(SupersetUnavailable) as caught:
        _client(fake.url, timeout_s=1).call("execute_sql", REQUEST)
    assert time.monotonic() - started < 2.5
    assert str(caught.value) == "Superset's MCP service did not answer execute_sql within 1 seconds"
    _holds_no_credential(caught.value)


def test_a_call_can_wait_longer_than_the_client_deadline(fake):
    """A call's own deadline replaces the client's, so a statement given more time than the client's default (an
    enrichment query, say) is not cut off by it."""
    def slow(request, authorization):
        time.sleep(2)
        return PAYLOAD

    fake.reply("execute_sql", slow)
    assert _client(fake.url, timeout_s=1).call("execute_sql", REQUEST, timeout_s=30) == PAYLOAD


def test_a_call_can_set_a_shorter_deadline(fake):
    def slow(request, authorization):
        time.sleep(3)
        return PAYLOAD

    fake.reply("execute_sql", slow)
    started = time.monotonic()
    with pytest.raises(SupersetUnavailable) as caught:
        _client(fake.url, timeout_s=30).call("execute_sql", REQUEST, timeout_s=1)
    assert time.monotonic() - started < 2.5
    assert str(caught.value) == "Superset's MCP service did not answer execute_sql within 1 seconds"


def test_a_call_from_inside_an_event_loop_is_not_read_as_an_outage(fake):
    """The client blocks, so it is called from a thread with no event loop; one called from a coroutine is a mistake
    in mnemiq, and raises as itself rather than as Superset being unavailable."""
    import anyio

    async def inside():
        return _client(fake.url).call("execute_sql", REQUEST)

    with pytest.raises(RuntimeError) as caught:
        anyio.run(inside)
    assert type(caught.value) is RuntimeError
    assert fake.calls == []


def test_two_clients_never_share_a_bearer(fake):
    """Each call opens its own session with its own client's bearer, so calls made at once from two threads each
    reach Superset carrying only their own."""
    fake.reply("execute_sql", lambda request, authorization: {"seen": authorization})
    clients = {"Bearer alpha": _client(fake.url, "alpha"), "Bearer beta": _client(fake.url, "beta")}
    jobs = [(expected, client) for expected, client in clients.items()] * 6

    def run(job):
        expected, client = job
        return expected, client.call("execute_sql", {**REQUEST, "expected": expected})

    with ThreadPoolExecutor(max_workers=6) as pool:
        answers = list(pool.map(run, jobs))
    assert all(answer == {"seen": expected} for expected, answer in answers)
    assert len(fake.calls) == 12
    assert all(call.authorization == call.request["expected"] for call in fake.calls)


def test_the_older_transport_name_still_serves(fake, monkeypatch):
    """mcp 1.x also offers the streamable-HTTP opener under its earlier name, which 2.x removed; the client accepts
    either, so this drives a call through that name where the installed mcp has it."""
    from mcp.client import streamable_http

    from mnemiq.superset import client as module

    legacy = getattr(streamable_http, "streamablehttp_client", None)
    if legacy is None:
        pytest.skip("this mcp has only the current name")
    used = []

    def spy(*args, **kwargs):
        used.append(args)
        return legacy(*args, **kwargs)

    earlier_only = SimpleNamespace(streamablehttp_client=spy)
    monkeypatch.setattr(module, "_opener", lambda: module._transport_opener(earlier_only))
    fake.reply("execute_sql", lambda request, authorization: PAYLOAD)
    with pytest.warns(DeprecationWarning):  # the SDK marks the earlier name deprecated
        assert _client(fake.url).call("execute_sql", REQUEST) == PAYLOAD
    assert used == [(fake.url,)]
    assert fake.calls[0].authorization == "Bearer tok"


def test_a_credential_never_prints():
    token = DelegatedToken("tok")
    assert "tok" not in repr(token)
    assert "tok" not in str(token)
    assert "tok" not in f"{token}"
    assert repr(token) == str(token) == "DelegatedToken(<redacted>)"
    assert str(Bearer("tok")) == "Bearer(<redacted>)"
    assert "tok" not in repr(SupersetMcp("http://127.0.0.1:1/mcp", token))
    assert SupersetMcp("http://127.0.0.1:1/mcp", token).credential is token


def test_a_credential_holds_a_header_value_only():
    """A bearer is sent as a header, so a value a header cannot carry is refused when the credential is made, and
    the refusal does not repeat it."""
    for bad in ("", "two words", "line\nbreak", "café"):
        with pytest.raises(ValueError) as caught:
            DelegatedToken(bad)
        if bad:
            assert bad not in str(caught.value)


def test_a_plain_string_is_not_a_credential():
    with pytest.raises(TypeError):
        SupersetMcp("http://127.0.0.1:1/mcp", "tok")  # type: ignore[arg-type]

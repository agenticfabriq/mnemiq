"""A stand-in for Superset's MCP service: a real MCP server over streamable HTTP on loopback, served in a thread.

It offers the three tools mnemiq calls, each taking `request: dict` as Superset's do, and answers each call from a
handler the test sets. What a handler returns reaches the client the way Superset sends it: a dict as structured
content beside its JSON text, an exception as an error result whose text is `Error: ` and then the exception's message.
That is the shape Superset's outermost middleware, `ToolResultCompatibilityMiddleware` in
`superset/mcp_service/middleware.py`, gives every error that reaches it, a refusal from `GlobalErrorHandlerMiddleware`
included. `bare_errors=True` sends the message alone.

Like Superset with authentication on, the transport answers 401 before any tool runs when a request carries no bearer,
or one the test refuses. Runs on mcp 1.x and 2.x.

`RecordingClient` stands in for the client instead, without a wire, where a test needs only what mnemiq sends and what
it makes of the answer.
"""

from __future__ import annotations

import contextvars
import inspect
import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio
import uvicorn

TOOLS = ("execute_sql", "list_datasets", "get_database_info")

# A handler answers one tool call: given the call's `request` and the Authorization header it arrived with, it returns
# the payload, or an exception whose message becomes the error result's text.
Handler = Callable[[dict, "str | None"], Any]

# The Authorization header of the HTTP request a tool call arrived on, set by `_Transport` for that request.
_authorization: contextvars.ContextVar[str | None] = contextvars.ContextVar("authorization", default=None)


@dataclass(frozen=True)
class Call:
    tool: str
    request: dict
    authorization: str | None


@dataclass(frozen=True)
class Sent:
    tool: str
    request: dict
    timeout_s: float | None


class RecordingClient:
    """Stands in for `SupersetMcp`: records each call, and answers it with `answer`, a payload to return or an
    exception to raise, or a function of the call's request that returns or raises one."""

    def __init__(self, answer: dict | Exception | Callable[[dict], dict]) -> None:
        self.answer = answer
        self.calls: list[Sent] = []

    def call(self, tool: str, request: dict, *, timeout_s: float | None = None) -> dict:
        self.calls.append(Sent(tool, request, timeout_s))
        answer = self.answer(request) if callable(self.answer) else self.answer
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeSuperset:
    """`with FakeSuperset() as fake:` serves at `fake.url` until the block ends.

    `structured_output=False` sends each answer as JSON text only, as Superset does when its structured output is
    off. `bare_errors=True` sends an error result's text without the `Error: ` Superset puts before it."""

    def __init__(self, *, structured_output: bool = True, bare_errors: bool = False) -> None:
        self.calls: list[Call] = []
        self.url: str | None = None
        self._structured = structured_output
        self._bare_errors = bare_errors
        self._handlers: dict[str, Handler] = {}
        self._refused: set[str] = set()
        self._http_failure: tuple[int, str] | None = None
        self._close_streams = False
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def reply(self, tool: str, fn: Handler) -> None:
        """Answer every later call of `tool` with `fn(request, authorization)`."""
        if tool not in TOOLS:
            raise ValueError(f"the fake serves only {TOOLS}")
        self._handlers[tool] = fn

    def error_text(self, message: str) -> str:
        """The text of the error result this fake sends for an exception whose message is `message`."""
        return message if self._bare_errors else f"Error: {message}"

    def refuse_token(self, value: str) -> None:
        """Answer 401 at the transport for a request carrying this bearer, before any tool runs."""
        self._refused.add(value)

    def fail_http(self, status: int, *, content_type: str = "application/json") -> None:
        """Answer every later request with this HTTP status at the transport, before any tool runs: a small JSON
        object, or with `content_type="text/html"` a sign-in page, as a proxy in front of Superset might send."""
        self._http_failure = (status, content_type)

    def close_stream_before_result(self) -> None:
        """Answer every later tool call with an event stream that ends before it carries the call's result."""
        self._close_streams = True

    async def _answer(self, tool: str, request: dict):
        from mcp.types import CallToolResult, TextContent

        authorization = _authorization.get()
        self.calls.append(Call(tool, request, authorization))
        handler = self._handlers.get(tool)
        if handler is None:
            outcome: Any = RuntimeError(f"the fake has no reply for {tool}")
        else:
            # In a worker thread, so a slow handler holds up only its own call.
            outcome = await anyio.to_thread.run_sync(handler, request, authorization)
        if isinstance(outcome, Exception):
            return CallToolResult(content=[TextContent(type="text", text=self.error_text(str(outcome)))], isError=True)
        structured = outcome if self._structured and isinstance(outcome, dict) else None
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(outcome))],
                              structuredContent=structured)

    def _app(self):
        from mnemiq.mcp.server import _mcp_classes

        server_class, _ = _mcp_classes()
        # mcp 2.x moved statelessness from the server's constructor to `streamable_http_app()`, as `mcp_http_app` in
        # `mnemiq/mcp/server.py` handles for mnemiq's own door.
        moved = "stateless_http" in inspect.signature(server_class.streamable_http_app).parameters
        server = server_class("superset-fake") if moved else server_class("superset-fake", stateless_http=True)
        for name in TOOLS:
            server.tool(name=name, structured_output=False)(self._tool(name))
        app = server.streamable_http_app(stateless_http=True) if moved else server.streamable_http_app()
        return _Transport(app, self)

    def _tool(self, name: str):
        async def tool(request: dict):
            return await self._answer(name, request)

        tool.__name__ = name
        return tool

    def __enter__(self) -> FakeSuperset:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(self._app(), log_level="warning", lifespan="on"))
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the fake Superset did not start")
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.should_exit = True
        self._thread.join(30)


class _Transport:
    """ASGI in front of the MCP app: refuses at the HTTP layer as Superset's transport does, and hands each request's
    Authorization header to the tool call it carries."""

    def __init__(self, app, fake: FakeSuperset) -> None:
        self.app, self.fake = app, fake

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        authorization = next((value.decode("latin-1") for name, value in scope.get("headers", [])
                              if name.lower() == b"authorization"), None)
        scheme, _, bearer = (authorization or "").partition(" ")
        if self.fake._http_failure is not None:
            status, content_type = self.fake._http_failure
            if content_type == "text/html":
                await _send(send, status, b"text/html; charset=utf-8", b"<html><body><h1>Sign in</h1></body></html>")
            else:
                await _respond(scope, receive, send, status, {"error": "the fake failed this request"})
            return
        if scheme.lower() != "bearer" or not bearer or bearer in self.fake._refused:
            await _respond(scope, receive, send, 401, {"error": "invalid_token",
                                                       "error_description": "Authentication required"},
                           headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})
            return
        if self.fake._close_streams and scope.get("method") == "POST":
            body, receive = await _buffered(receive)
            if _jsonrpc_method(body) == "tools/call":
                await _send(send, 200, b"text/event-stream", b"")
                return
        token = _authorization.set(authorization)
        try:
            await self.app(scope, receive, send)
        finally:
            _authorization.reset(token)


async def _respond(scope, receive, send, status: int, body: dict, headers: dict | None = None) -> None:
    from starlette.responses import JSONResponse

    await JSONResponse(body, status_code=status, headers=headers)(scope, receive, send)


async def _send(send, status: int, content_type: bytes, body: bytes) -> None:
    """A complete response with this status, content type and body; an empty event stream ends at once."""
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", content_type)]})
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _buffered(receive):
    """Read the whole request body, and return it with a `receive` that hands it on to the app unchanged."""
    chunks, more = [], True
    while more:
        message = await receive()
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    body, replayed = b"".join(chunks), False

    async def replay():
        nonlocal replayed
        if not replayed:
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return body, replay


def _jsonrpc_method(body: bytes) -> str | None:
    try:
        message = json.loads(body)
    except ValueError:
        return None
    return message.get("method") if isinstance(message, dict) else None

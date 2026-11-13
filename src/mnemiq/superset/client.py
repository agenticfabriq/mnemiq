"""A synchronous client for Superset's MCP service that carries exactly one caller's bearer.

Each `SupersetMcp` holds one credential, and each call opens its own MCP session over streamable HTTP, sends one tool
call and closes the session. Nothing outlives a call, so one person's bearer never travels on another person's request.
The call runs on `anyio.run` in the calling thread: the HTTP routes run synchronous handlers in worker threads and
enrichment is synchronous, so neither has an event loop to share.

Every failure is raised as one of the exceptions below, whose message is mnemiq's own sentence. Superset's text rides
only in `.detail`, which the client logs once as it raises: a refusal can name a table, a role or a user the caller
may not see, and an exception's message is what reaches the model and the person asking.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A bearer is sent as an HTTP header value, so it is held to visible ASCII without spaces: a value a header cannot
# carry is refused when the credential is made, before an HTTP library can quote it back in an error.
_HEADER_SAFE = re.compile(r"[!-~]+")

# What Superset's outermost middleware, `ToolResultCompatibilityMiddleware` (`superset/mcp_service/middleware.py`), puts
# before the text of every error that reaches it, the refusals below included: it catches the error and answers with
# the text `Error: <message>`. The shapes below are read after one such prefix, wherever they say they open the text.
_ERROR_PREFIX = "Error: "
# The refusal `MCPPermissionDeniedError` (`superset/mcp_service/auth.py`) carries when a known caller's roles lack a
# permission a tool needs. `GlobalErrorHandlerMiddleware` (`superset/mcp_service/middleware.py`) passes its text on
# unwrapped: "Permission denied: <permission> on <view>", then the user and the tool.
_ROLE_DENIAL = re.compile(r"Permission denied: \S+ on \S+")
# The prefix that middleware writes for any other denial: a plain `PermissionError` (an invalid or expired API key is
# one), and a `ForbiddenError` or `SupersetSecurityException` (a signed-in caller denied a resource). The text cannot
# tell these causes apart, so mnemiq claims neither.
_DENIAL = re.compile(r"Permission denied for \S+:")
# Superset's refusal of a verified token whose user it has no account for (`superset/mcp_service/auth.py`).
_UNKNOWN_USER = "not found in Superset database"
# How the error opens that Superset's response size guard raises for an answer too large to send and that it does not
# cut instead (`format_size_limit_error` in `superset/mcp_service/utils/response_size_utils.py`).
_TOO_LARGE = "Response too large:"
# A tool the server does not serve: an MCP server built on the SDK answers "Unknown tool: <name>". Superset's tool search
# (`MCP_TOOL_SEARCH_CONFIG` in `superset/mcp_service/mcp_config.py`, on by default) hides tools from the catalogue a
# client lists and still serves each one called by name, so it is not the cause; a Superset without the tool is, and
# `MCP_DISABLED_TOOLS`, which removes a tool from the service.
_UNKNOWN_TOOL = re.compile(r"unknown tool", re.IGNORECASE)
# How long a call waits for Superset when neither the client nor the call says otherwise.
DEFAULT_TIMEOUT_S = 45.0


@dataclass(frozen=True, repr=False)
class Bearer:
    """A credential sent to Superset as `Authorization: Bearer <value>`. It never prints: its repr, its str and an
    f-string show only its kind, so a log line, a trace or an exception that mentions one carries no secret."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _HEADER_SAFE.fullmatch(self.value):
            raise ValueError(f"a {type(self).__name__} must be visible ASCII characters with no spaces")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class DelegatedToken(Bearer):
    """The asking person's own verified token, passed on to Superset unchanged, so Superset judges the call as them."""


class _SupersetFailure(RuntimeError):
    """A failure with mnemiq's own sentence as its message and what Superset or the transport said in `.detail`."""

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class SupersetUnavailable(_SupersetFailure):
    """Superset gave no usable answer: the transport failed, the call ran out of time, Superset answered with an HTTP
    error, or the database asked for a sign-in of its own (an OAuth2 redirect) that mnemiq cannot give."""


class SupersetCallTimedOut(SupersetUnavailable):
    """The call's deadline passed after Superset took the tool call and before it answered: its response to the call
    had started, had not ended, and its result had not come. What the tool does took longer than the call allowed,
    and Superset may still be doing it. A deadline that passes before Superset takes the call, after its response
    ended without the result, or after its result is in, is `SupersetUnavailable`.

    A reader that knows what the tool was doing may read this more closely than as an outage: Superset's own time limit
    on a statement takes effect only once the database call returns (`SigalrmTimeout` in `superset/utils/core.py`), so
    a statement still running past its limit and the transport's margin ends the call here."""


class SupersetRefusal(_SupersetFailure):
    """Superset answered, and said no."""


class SupersetRefusedCaller(SupersetRefusal):
    """Superset did not accept who is calling: an HTTP 401 or 403 at the transport, or a verified token naming a user
    Superset has no account for."""


class SupersetPermissionDenied(SupersetRefusal):
    """A known caller's roles lack a permission the tool needs."""


class SupersetDenied(SupersetRefusal):
    """Superset refused and did not say why: the same text covers a bad credential and a denied resource."""


class SupersetToolError(_SupersetFailure):
    """The tool reported an error that is not a refusal, or answered with something other than a JSON object."""


class SupersetResponseTooLarge(SupersetToolError):
    """The tool's answer was larger than Superset's MCP service will send, and it sent an error in its place: the
    request is the caller's to make smaller."""


class SupersetMcp:
    """Calls Superset's MCP tools at `url`, carrying `credential` and nothing else, each call within `timeout_s`
    unless the call names a deadline of its own."""

    def __init__(self, url: str, credential: Bearer, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        if not isinstance(credential, Bearer):
            raise TypeError("SupersetMcp carries a Bearer, never a bare string")
        self._url = url
        self._credential = credential
        self._timeout_s = timeout_s
        self._open = _opener()

    @property
    def credential(self) -> Bearer:
        return self._credential

    def call(self, tool: str, request: dict, *, timeout_s: float | None = None) -> dict:
        """Call `tool` with the arguments `{"request": request}`, as Superset's tools take them, and return its
        payload: the structured content when Superset sends it, else the first text content read as a JSON object.

        `timeout_s` is this call's own deadline, in place of the client's: a caller that gives Superset a time limit
        of its own waits longer than that limit, so Superset's answer that the statement ran out of time arrives
        before the transport gives up on it.

        A failed call raises one of the exceptions above, and logs its `.detail` once, at WARNING, as it does."""
        import anyio

        exchange = _Exchange(deadline_s=self._timeout_s if timeout_s is None else timeout_s)
        result = anyio.run(self._exchange, tool, request, exchange)
        if exchange.error is not None:
            failure = self._failure(tool, exchange)
            _log(tool, failure)
            raise failure from exchange.error
        try:
            return _payload(tool, result)
        except _SupersetFailure as failure:
            _log(tool, failure)
            raise

    async def _exchange(self, tool: str, request: dict, exchange: _Exchange):
        """The call itself. A failure of the exchange with Superset is kept on `exchange` for `_failure` to read;
        anything that fails before the exchange begins (a call from inside an event loop, a broken install) raises as
        itself, never as Superset being unavailable."""
        import anyio
        from mcp import ClientSession

        http_client = self._http_client(exchange)
        # One deadline over the whole call (connecting, the handshake and the tool call), so a slow Superset costs
        # at most the call's deadline whichever step it is slow in.
        deadline = anyio.move_on_after(exchange.deadline_s)
        # Cancelled when the transport reports an answer it could not read (a proxy's page, say): mcp 1.x hands that
        # to the session's message handler and leaves the call waiting for a result that will never come.
        abandon = anyio.CancelScope()

        async def on_message(message) -> None:
            if isinstance(message, Exception):
                exchange.error = exchange.error or message
                abandon.cancel()

        try:
            with deadline:
                async with self._open(self._url, http_client) as (read, write):
                    async with ClientSession(read, write, message_handler=on_message) as session:
                        with abandon:
                            await session.initialize()
                            return await session.call_tool(tool, {"request": request})
        except Exception as exc:
            exchange.error = exchange.error or exc
            return None
        finally:
            exchange.timed_out = deadline.cancel_called
        if exchange.error is None:
            exchange.error = TimeoutError(f"no answer within {_seconds(exchange.deadline_s)}")
        return None

    def _http_client(self, exchange: _Exchange):
        # The SDK's own factory, because it builds the HTTP client of the library the installed mcp speaks (httpx on
        # 1.x, httpx2 on 2.x); the transport accepts no other.
        from mcp.shared._httpx_utils import create_mcp_http_client

        client = create_mcp_http_client(headers={"Authorization": f"Bearer {self._credential.value}"})

        async def sent(request) -> None:
            # The SDK sends a session's requests one after another, so a request sent after the tool call means the
            # call's result is in: the `tools/list` the SDK sends to read the tool's output schema, say.
            if exchange.call_taken and _jsonrpc_request(request) not in (None, "tools/call"):
                exchange.call_answered = True

        async def record(response) -> None:
            exchange.statuses.append(response.status_code)
            # Superset starts its answer to a tool call (an event stream) as it takes the call, before the tool runs.
            if _jsonrpc_request(response.request) == "tools/call" and 200 <= response.status_code < 300:
                exchange.call_taken = True
                _on_end(response, exchange.mark_call_ended)

        # mcp 2.x turns an HTTP error into a JSON-RPC error that no longer says which status it was, so the client
        # reads the status of each response itself.
        client.event_hooks = {"request": [sent], "response": [record]}
        return client

    def _failure(self, tool: str, exchange: _Exchange) -> _SupersetFailure:
        """Read a failed exchange: a call that ended without a tool result.

        Superset sends a refusal or a tool's failure as an error result, which `_payload` reads. A call that ends
        without any result is therefore never Superset answering about the tool: it is the transport, whatever the
        SDK raised for it. That includes the JSON-RPC errors the SDK makes up on the client side for a closed
        connection, an answer of the wrong content type or one it could not parse, so their text is never read.

        The HTTP status comes first: mcp 1.x raises an HTTP error as such, but 2.x reports it as a JSON-RPC error of
        its own making that no longer says which status it was."""
        detail = _describe(exchange.error)
        status = exchange.statuses[-1] if exchange.statuses else None
        if status in (401, 403):
            return SupersetRefusedCaller(f"Superset did not accept the credential this request carries (HTTP {status})",
                                         detail=detail)
        if status is not None and not 200 <= status < 300:
            return SupersetUnavailable(f"Superset's MCP service answered HTTP {status}", detail=detail)
        if exchange.timed_out:
            # Superset had taken the tool call, was still sending its response, and had not answered: the time ran out
            # in the tool, not in reaching Superset, in a response that ended without the result (mcp 1.x leaves such
            # a call waiting) or in what the SDK asks after the answer.
            running = exchange.call_taken and not exchange.call_ended and not exchange.call_answered
            late = SupersetCallTimedOut if running else SupersetUnavailable
            return late(f"Superset's MCP service did not answer {tool} within {_seconds(exchange.deadline_s)}",
                        detail=detail)
        if exchange.statuses:  # it answered something: an unreadable answer, or a connection lost mid-call
            return SupersetUnavailable(f"Superset's MCP service gave no usable answer to {tool}", detail=detail)
        return SupersetUnavailable("Superset's MCP service could not be reached", detail=detail)


@dataclass
class _Exchange:
    """One call's deadline, and what the call observed beside its result: each HTTP response's status in order,
    whether Superset took the tool call (its response to the call had started), whether that response then ended (its
    body came to an end or could not be read), whether Superset answered the call (a later request was sent), whether
    time ran out, and the exception that ended it, if one did."""

    deadline_s: float = DEFAULT_TIMEOUT_S
    statuses: list[int] = field(default_factory=list)
    call_taken: bool = False
    call_ended: bool = False
    call_answered: bool = False
    timed_out: bool = False
    error: Exception | None = None

    def mark_call_ended(self) -> None:
        self.call_ended = True


def _seconds(value: float) -> str:
    """`value` seconds as a sentence says it: `1 second`, `0.5 seconds`, `45 seconds`."""
    return f"{value:g} second" if value == 1 else f"{value:g} seconds"


def _opener():
    """The streamable-HTTP transport of the installed mcp."""
    from mcp.client import streamable_http

    return _transport_opener(streamable_http)


def _transport_opener(transport):
    """Open the streamable-HTTP transport by whichever name `transport` (mcp's module) offers, yielding the read and
    write streams. `streamable_http_client` takes the HTTP client it is given and leaves closing it to the caller;
    mcp 1.x also has it under the earlier name `streamablehttp_client`, which builds the client from a factory and
    closes it itself, and which 2.x removed."""
    current = getattr(transport, "streamable_http_client", None)
    if current is not None:
        @asynccontextmanager
        async def open_streams(url: str, http_client):
            async with http_client, current(url, http_client=http_client) as streams:
                yield streams[0], streams[1]

        return open_streams
    earlier = getattr(transport, "streamablehttp_client", None)
    if earlier is None:
        raise ImportError("the installed mcp has no streamable-HTTP client transport")

    @asynccontextmanager
    async def open_earlier(url: str, http_client):
        async with earlier(url, httpx_client_factory=lambda **_: http_client) as streams:
            yield streams[0], streams[1]

    return open_earlier


def _payload(tool: str, result) -> dict:
    """The JSON object a tool answered, or the exception its error result reads as."""
    text = _first_text(result)
    if getattr(result, "is_error", getattr(result, "isError", False)):  # 2.x's field name, then 1.x's
        raise _tool_failure(tool, text)
    structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    if isinstance(structured, dict):
        return structured
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        raise SupersetToolError(f"Superset's answer to {tool} was not a JSON object", detail=text)
    return payload


def _log(tool: str, failure: _SupersetFailure) -> None:
    """Log what Superset or the transport said about a failed call of `tool`, beside mnemiq's sentence for it: the
    exception's message, which is all that reaches the model and the person asking, says only the sentence."""
    logger.warning("Superset's MCP service: %s failed (%s): %s", tool, failure, failure.detail or "no detail")


def _jsonrpc_request(request) -> str | None:
    """The JSON-RPC method an HTTP request carries when it is a request (it has an id), else None: a notification, a
    body that is not JSON, and a request with no body read as None."""
    try:
        message = json.loads(request.content)
    except Exception:  # noqa: BLE001 -- a body this cannot read is not a tool call
        return None
    return message.get("method") if isinstance(message, dict) and "id" in message else None


def _on_end(response, ended: Callable[[], None]) -> None:
    """Call `ended` once the body of `response` comes to an end or a read of it fails: the server ended the stream,
    the connection closed, or a read timed out. A read that a cancellation stops (the call's own deadline) is neither,
    so a response still open when the deadline passes never calls it.

    mcp 1.x reads an event stream that ends before the call's result as nothing at all: it logs it and leaves the call
    waiting for a result that will not come, until the deadline. This is how the client still knows the response
    ended. The body is wrapped in the stream type of the HTTP library the response comes from (httpx on mcp 1.x,
    httpx2 on 2.x), which that library requires of any body it reads."""
    library = importlib.import_module(type(response).__module__.partition(".")[0])
    body = response.stream

    class _Watched(library.AsyncByteStream):
        async def __aiter__(self):
            chunks = body.__aiter__()
            try:
                async for chunk in chunks:
                    yield chunk
            except Exception:  # a cancellation is a BaseException, and passes through unmarked
                ended()
                raise
            else:
                ended()
            finally:
                if isinstance(chunks, AsyncGenerator):
                    await chunks.aclose()

        async def aclose(self) -> None:
            await body.aclose()

    response.stream = _Watched()


def _first_text(result) -> str:
    return next((block.text for block in result.content if getattr(block, "type", None) == "text"), "")


def _tool_failure(tool: str, text: str) -> _SupersetFailure:
    """Read an error Superset's MCP service sent about `tool`, after the one `Error: ` its outermost middleware puts
    before it; the whole text, prefix and all, is kept as the detail only."""
    message = text.removeprefix(_ERROR_PREFIX)
    if _UNKNOWN_USER in message:
        return SupersetRefusedCaller("Superset has no user for the credential this request carries", detail=text)
    if _ROLE_DENIAL.match(message):
        return SupersetPermissionDenied(f"Superset's roles for this user do not permit {tool}", detail=text)
    if _DENIAL.match(message):
        return SupersetDenied(f"Superset refused {tool} and did not say why", detail=text)
    if message.startswith(_TOO_LARGE):
        return SupersetResponseTooLarge(f"Superset's answer to {tool} was larger than its MCP service will send; ask "
                                        "for less", detail=text)
    if _UNKNOWN_TOOL.search(message):
        return SupersetToolError(f"Superset's MCP service does not offer {tool}, which Superset mode calls, and which "
                                 "Superset 7.0.0rc2 and later offer unless MCP_DISABLED_TOOLS names it", detail=text)
    return SupersetToolError(f"Superset's {tool} tool failed", detail=text)


def _leaves(exc: BaseException) -> Iterator[BaseException]:
    """The exceptions inside `exc`: a transport's task group raises its failures as a group, nested at any depth."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaves(inner)
    else:
        yield exc


def _describe(exc: BaseException) -> str:
    return "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in _leaves(exc))

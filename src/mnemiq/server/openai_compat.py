"""`/openai/v1`: an OpenAI-compatible model list and chat completions over `Runtime.ask`, for Open WebUI or any
OpenAI client. Each request is answered as its own verified caller, and each profile that caller may pick is one
model.

Mounted only where the `/openai/v1` door can verify a caller (Open WebUI's signed header, or a provider token for
its own audience). Without one, every path here answers 404, so a deployment cannot quietly become a one-identity
backend. The door's credential rules are `mnemiq.auth.doors`'; this module adds the request rules: the question is
the last user message's text, a background task never reaches the engine, and nothing a caller sent is repeated
in an error.

Every error under `/openai/v1` is in OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`, and every
message is composed here from what this layer knows, never from an exception's text.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from mnemiq.agent.modes import MODES
from mnemiq.agent.route import UnknownMode
from mnemiq.contract import DeferralReason, IdentityContext
from mnemiq.profiles import UnknownProfile, offered_to
from mnemiq.server.pump import HEARTBEAT, EnginePump
from mnemiq.server.serialize import answer_payload

logger = logging.getLogger(__name__)

PREFIX = "/openai/v1"
_ROUTES = {"models": "GET", "chat/completions": "POST"}  # each path this endpoint serves, and its one method
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]  # a route with none is GET-only
_TYPE = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
         404: "invalid_request_error", 405: "invalid_request_error", 503: "server_error", 500: "server_error"}
_OPAQUE = "The engine could not complete this request"

EFFORT_TO_MODE = {"low": "instant", "medium": "thinking", "high": "deep"}
# Open WebUI's default retrieval template (v0.13.0, `DEFAULT_RAG_TEMPLATE`) opens with this header and carries what
# it retrieved in this block. Together they mean the message holds more than the user's own words.
_RAG_MARKERS = ("### Task:", "<context>", "</context>")
# Open WebUI v0.13.0's own background task names (`TASKS`, its `constants.py`). The task header is unsigned and any
# client may set it, so a value outside these is logged as `other`, never as sent.
_OPEN_WEBUI_TASKS = frozenset({"title_generation", "follow_up_generation", "tags_generation", "emoji_generation",
                               "query_generation", "image_prompt_generation", "autocomplete_generation",
                               "function_calling", "moa_response_generation"})


PREVIEW_ROWS = 20
# Every character that starts a link, an image, HTML, math, a code span, emphasis or strikethrough, or splits a
# table cell, as its numeric character reference. Not a backslash escape: Open WebUI renders `\(`, `\[` and `$...$`
# as math, so an escaped bracket would itself open a formula. Without `[` there is no link or image, without `<` no
# HTML, and without `$` and `\` no math; parentheses, `.`, `-` and `#` stay as they are.
_INERT = {char: f"&#{ord(char)};" for char in "\\`*[]!<>|$&~"}
_STATUS = {"judged": "verified by the judge", "basic": "basic checks only, not judged",
           "unavailable": "not verified: the judge could not be reached", "unknown": "verification state unknown"}


def _inert(text: str, *, cell: bool) -> str:
    """`text` with nothing in it that renders: one pass, so a reference is never escaped again. In a table cell a
    line break becomes a space, since a row is one line."""
    return "".join(_INERT.get(char) or (" " if cell and char in "\r\n" else char) for char in text)


def _prose(text: str) -> str:
    # Each line's leading whitespace goes, so no line of the answer becomes an indented code block, where a
    # character reference would show as written.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(_inert(line.lstrip(), cell=False) for line in lines)


def _cell(value) -> str:
    return "" if value is None else _inert(str(value), cell=True)


def _fence(sql: str) -> str:
    """The SQL in a fence longer than any run of backticks inside it, and at least three. Nothing renders in a fence."""
    ticks = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", sql)), default=0))
    return f"{ticks}sql\n{sql}\n{ticks}"


def _status(payload: dict) -> str:
    # Only a judged answer says "verified": `verified_state` keeps the others apart so that none reads as a check.
    for flag, word in (("failed", "failed"), ("deferred", "deferred")):
        if payload[flag]:
            return f"**Status:** {word} ({payload['reason_code']})" if payload["reason_code"] else f"**Status:** {word}"
    state = payload["verified"]
    return "**Status:** " + (_STATUS.get(state, _STATUS["unknown"]) if state is not None else "not verified")


def _table(preview: dict) -> list[str]:
    """The first `PREVIEW_ROWS` rows as a Markdown table, then how many rows the query returned. A capped result's
    count is a floor, so it reads "at least"."""
    columns, shown, total = preview["columns"], preview["rows"][:PREVIEW_ROWS], preview["row_count"]
    lines = ["| " + " | ".join(_cell(c) for c in columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in shown]
    if preview["capped"]:
        count = f"{len(shown)} of at least {total:,} rows"
    elif total > len(shown):
        count = f"{len(shown)} of {total:,} rows"
    else:
        count = f"{total:,} row" + ("" if total == 1 else "s")
    return ["\n".join(lines), count]


def render(payload: dict, run_id: str) -> str:
    """A completion's content, from `answer_payload`: the answer, its status, its SQL, up to `PREVIEW_ROWS` rows and
    the run id, in that order. A decline is its reason and its status alone."""
    parts = [_prose(payload["answer"] or ""), _status(payload)]
    if not payload["deferred"] and not payload["failed"]:
        if payload["sql"]:
            parts.append(_fence(payload["sql"]))
        if payload["preview"] and payload["preview"]["columns"]:
            parts += _table(payload["preview"])
    parts.append(f"run {run_id}")
    return "\n\n".join(parts)


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any = None


class ChatBody(BaseModel):
    """A chat completion request, as far as this endpoint reads one. Every other field is ignored, `temperature`,
    `top_p`, `max_tokens` and `stop` among them: the profile decides those."""

    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[_Message] = Field(min_length=1)
    stream: bool = False
    reasoning_effort: str | None = None
    tools: list | None = None
    functions: list | None = None
    function_call: Any = None
    n: int | None = None


class OpenAIError(Exception):
    """A refusal on `/openai/v1`, answered in OpenAI's error shape. `message` is composed by this module."""

    def __init__(self, status: int, message: str, *, code: str | None = None, param: str | None = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.code, self.param, self.headers = status, message, code, param, headers


def _shaped(status: int, message: str, *, code=None, param=None, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": _TYPE.get(status, "invalid_request_error"),
                                   "param": param, "code": code}}, status_code=status, headers=headers)


def _under_prefix(request: Request) -> bool:
    path = request.url.path
    return path == PREFIX or path.startswith(PREFIX + "/")


def _install_handlers(app: FastAPI) -> None:
    async def on_openai_error(request: Request, exc: OpenAIError):
        return _shaped(exc.status, exc.message, code=exc.code, param=exc.param, headers=exc.headers)

    async def on_http_error(request: Request, exc: StarletteHTTPException):
        # The door's refusals arrive as HTTPException, from `caller_dependency`; the router's 404 and 405 as
        # Starlette's. Under /openai/v1 both take OpenAI's shape, keeping their status and challenge.
        if not _under_prefix(request):
            return await http_exception_handler(request, exc)
        return _shaped(exc.status_code, str(exc.detail), headers=getattr(exc, "headers", None))

    app.add_exception_handler(OpenAIError, on_openai_error)
    app.add_exception_handler(StarletteHTTPException, on_http_error)


def _catch_all(app: FastAPI, *, mounted: bool) -> None:
    """Claims every other /openai/v1 path and method. Registered after the real routes, and before the workbench's
    `/` mount, which would otherwise answer a POST with 405. A full path match wins over the router's method check,
    so this answers the known paths' wrong methods itself."""

    def elsewhere(rest: str, request: Request):
        allowed = _ROUTES.get(rest.strip("/")) if mounted else None
        if allowed is not None:
            raise OpenAIError(405, f"{PREFIX}/{rest.strip('/')} takes {allowed} only", headers={"Allow": allowed})
        raise OpenAIError(404, "not found" if mounted else "this deployment does not serve /openai/v1")

    app.add_api_route(PREFIX + "/{rest:path}", elsewhere, methods=_METHODS, include_in_schema=False)


def _offered(runtime, identity: IdentityContext) -> list[str]:
    """The models this caller may pick: each profile `may_pick` offers them. A runtime built without profiles
    offers its default alone."""
    if runtime.profiles is None:
        return [runtime.default_profile]
    return offered_to(identity, {name: loaded.profile for name, loaded in runtime.profiles.items()})


def _task_value(request: Request, header: str) -> str:
    sent = request.headers.getlist(header)
    if not sent:
        raise OpenAIError(400, "every /openai/v1 request must carry the task header named in "
                               "MNEMIQ_OPENWEBUI_TASK_HEADER: add it to mnemiq's connection in Open WebUI with the "
                               "value {{TASK}}", code="missing_task_header")
    if len(sent) > 1:
        raise OpenAIError(400, "the task header was sent more than once", code="ambiguous_task_header")
    return sent[0].strip()


async def _json_object(request: Request) -> dict:
    try:
        data = json.loads(await request.body())
    except ValueError:
        raise OpenAIError(400, "the request body is not JSON", code="invalid_body") from None
    if not isinstance(data, dict):
        raise OpenAIError(400, "the request body must be a JSON object", code="invalid_body")
    return data


def _chat_body(data: dict) -> ChatBody:
    try:
        return ChatBody.model_validate(data)
    except ValidationError:
        raise OpenAIError(400, "the body is not a chat completion request: it needs `model` and a non-empty "
                               "`messages` list, each message with a `role`", code="invalid_body") from None


def _refuse_parameters(body: ChatBody) -> None:
    for field in ("tools", "functions", "function_call"):
        if getattr(body, field):
            raise OpenAIError(400, "tools are not supported: mnemiq answers the question itself, so turn off function "
                                   "calling for mnemiq's models", code="unsupported_parameter", param=field)
    if body.n is not None and body.n > 1:
        raise OpenAIError(400, "only one choice is returned", code="unsupported_parameter", param="n")


def _question(messages: list[_Message]) -> str:
    """The last user message's text, and nothing else: system messages and earlier turns are not read."""
    last = messages[-1]
    if last.role != "user":
        raise OpenAIError(400, "the last message must be the user's: this endpoint answers one question, the last "
                               "user message", code="invalid_last_message", param="messages")
    content = last.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif kind == "image_url":
                raise OpenAIError(400, "images are not read: send the question as text", code="unsupported_content",
                                  param="messages")
            else:
                raise OpenAIError(400, "only text parts are read: send the question as text",
                                  code="unsupported_content", param="messages")
        text = "\n".join(texts)
    else:
        text = ""
    if not text.strip():
        raise OpenAIError(400, "the last user message holds no text", code="empty_question", param="messages")
    if all(marker in text for marker in _RAG_MARKERS):
        raise OpenAIError(400, "this message carries Open WebUI's retrieval template, so it holds more than the "
                               "question: turn off knowledge, files and web search for mnemiq's models in Open WebUI",
                          code="retrieval_template", param="messages")
    return text


def _mode(effort: str | None) -> str | None:
    if effort is None:
        return None
    if effort not in EFFORT_TO_MODE:
        raise OpenAIError(400, f"unknown reasoning_effort; this endpoint offers: {sorted(EFFORT_TO_MODE)}",
                          code="invalid_reasoning_effort", param="reasoning_effort")
    return EFFORT_TO_MODE[effort]


def _completion(run_id: str, model: str, content: str) -> dict:
    return {"id": f"chatcmpl-{run_id}", "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


_END = object()
_STREAM_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}


def _refusal(exc: Exception, run_id: str) -> OpenAIError:
    """What a caller is told when `ask` raised: composed from the exception's type, never its text."""
    if isinstance(exc, UnknownProfile):
        return _not_offered(exc.offered)
    if isinstance(exc, UnknownMode):
        return OpenAIError(400, f"unknown mode; this deployment offers: {sorted(MODES)}",
                           code="invalid_reasoning_effort", param="reasoning_effort")
    logger.error("run %s: the engine failed", run_id, exc_info=exc)
    return OpenAIError(500, f"{_OPAQUE} (run {run_id}).")


def _checked(answer):
    if answer.reason_code == DeferralReason.QUESTION_TOO_LONG:
        # The engine's own decline, which states the cap and recorded the question as declined: a 400 here.
        raise OpenAIError(400, answer.answer, code="question_too_long", param="messages")
    return answer


def _chunk(run_id: str, created: int, model: str, delta: dict, finish: str | None = None) -> str:
    return "data: " + json.dumps({"id": f"chatcmpl-{run_id}", "object": "chat.completion.chunk", "created": created,
                                  "model": model, "choices": [{"index": 0, "delta": delta,
                                                               "finish_reason": finish}]}) + "\n\n"


def _stage(event: dict) -> str:
    failed = "" if event.get("ok", True) else " failed"
    return f"{event.get('stage')}{failed} ({round(event.get('ms') or 0)} ms)\n"


async def chat_chunks(pump: EnginePump, events, first, run_id: str, model: str):
    """The stream: the role, a reasoning delta per finished stage, an empty one as each heartbeat's keep-alive (not
    an SSE comment, whose handling in Open WebUI is not known), the answer as one content delta, then `stop` and
    `[DONE]`. A failure after the stream opened becomes a composed content delta."""
    created = int(time.time())
    try:
        yield _chunk(run_id, created, model, {"role": "assistant", "content": ""})
        item = first
        while item is not _END:
            if item is HEARTBEAT:
                yield _chunk(run_id, created, model, {"reasoning_content": ""})
            elif item.get("phase") == "finished":  # the early `result` event carries no phase, and is not shown
                yield _chunk(run_id, created, model, {"reasoning_content": _stage(item)})
            item = await anext(events, _END)
        try:
            content = render(answer_payload(pump.result()), run_id)
        except Exception as exc:  # noqa: BLE001 -- composed below; the traceback goes to the log with the run id
            content = _refusal(exc, run_id).message
        yield _chunk(run_id, created, model, {"content": content})
        yield _chunk(run_id, created, model, {}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        await events.aclose()


async def _empty_stream(run_id: str, model: str):
    created = int(time.time())
    yield _chunk(run_id, created, model, {"role": "assistant", "content": ""})
    yield _chunk(run_id, created, model, {}, "stop")
    yield "data: [DONE]\n\n"


def _not_offered(offered: list[str]) -> OpenAIError:
    # Composed from what this caller may pick: an unknown model and a forbidden one read alike, and neither is named.
    return OpenAIError(404, f"that model does not exist or you may not use it; you may pick: {offered}",
                       code="model_not_found", param="model")


def mount_openai(app: FastAPI, runtime, doors, *, heartbeat_s: float) -> bool:
    """Adds `/openai/v1` to `app` when its door can verify a caller, and a 404 for every path under it otherwise.
    Returns whether the endpoint is served."""
    _install_handlers(app)
    if doors is None or not doors.openai_enabled:
        logger.info("/openai/v1 is not served: it needs a verifier of its own, MNEMIQ_OPENWEBUI_JWT_SECRET (Open "
                    "WebUI's signed user header) or MNEMIQ_OPENWEBUI_AUDIENCE (a provider token for this endpoint)")
        _catch_all(app, mounted=False)
        return False

    from mnemiq.server.app import caller_dependency  # here: app.py imports this module

    # The door's refusals are HTTPExceptions, which `_install_handlers` answers in OpenAI's shape under the prefix.
    caller, task_header = caller_dependency(doors, "openai"), doors.task_header

    @app.get(PREFIX + "/models", include_in_schema=False)
    def models(request: Request, who: IdentityContext = Depends(caller)) -> dict:
        _task_value(request, task_header)
        run_id = uuid4().hex
        try:
            offered = _offered(runtime, who)
        except Exception:
            logger.exception("run %s: listing models failed", run_id)
            raise OpenAIError(500, f"{_OPAQUE} (run {run_id}).") from None
        return {"object": "list",
                "data": [{"id": name, "object": "model", "created": 0, "owned_by": "mnemiq"} for name in offered]}

    @app.post(PREFIX + "/chat/completions", include_in_schema=False)
    async def chat(request: Request, who: IdentityContext = Depends(caller)):
        run_id = uuid4().hex
        sent_task = _task_value(request, task_header)
        data = await _json_object(request)
        if sent_task:
            # An Open WebUI background task (a title, tags, follow-ups): answered empty, never asked, never a question.
            name = sent_task if sent_task in _OPEN_WEBUI_TASKS else "other"
            logger.info("run %s: a background task (%s), answered with an empty completion", run_id, name)
            if data.get("stream") is True:
                return StreamingResponse(_empty_stream(run_id, str(data.get("model", ""))),
                                         media_type="text/event-stream", headers=_STREAM_HEADERS)
            return _completion(run_id, str(data.get("model", "")), "")
        body = _chat_body(data)
        _refuse_parameters(body)
        offered = _offered(runtime, who)
        if body.model not in offered:
            raise _not_offered(offered)
        question, mode = _question(body.messages), _mode(body.reasoning_effort)
        profile = None if runtime.profiles is None else body.model
        if body.stream:
            pump = EnginePump(lambda emit: runtime.ask(question, who, mode=mode, emit=emit, profile=profile),
                              heartbeat_s)
            events = pump.events()
            # The status stays open until the run shows its first event, or a heartbeat passes: a run that ends
            # before emitting anything (a decline before retrieval, an unknown profile) gets the status it would
            # get unstreamed, not a 200 already on the wire.
            first = await anext(events, _END)
            if first is _END:
                try:
                    _checked(pump.result())
                except OpenAIError:
                    raise
                except Exception as exc:
                    raise _refusal(exc, run_id) from None
            return StreamingResponse(chat_chunks(pump, events, first, run_id, body.model),
                                     media_type="text/event-stream", headers=_STREAM_HEADERS)
        try:
            answer = await run_in_threadpool(lambda: runtime.ask(question, who, mode=mode, profile=profile))
        except Exception as exc:
            raise _refusal(exc, run_id) from None
        return _completion(run_id, body.model, render(answer_payload(_checked(answer)), run_id))

    _catch_all(app, mounted=True)
    return True

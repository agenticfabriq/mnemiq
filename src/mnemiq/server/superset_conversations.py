"""The Superset panel's conversations: each person's own, kept in the control Postgres (`mnemiq.superset.conversations`),
each message routed by the profile's intent step (`mnemiq.superset.intent`) and recorded as it settles, and every write
waiting for the person's confirm.

**A message** (`POST .../conversations/{id}/messages`) answers an event stream. Its first frame names what the message
was routed to and the index the person's message was kept at; then, by intent:
- an ask streams as `/v1/chat` streams it, the Ask path unchanged, the conversation's earlier answers as its history
  (question, SQL and grant fingerprint, never a row); its answer is kept as its sentence and SQL;
- a build is planned and kept as a proposal, writing nothing to Superset (or answered as an offer of a dashboard
  that covers it, a refused plan, or a dashboard that already shows it all);
- a change is previewed, writing nothing;
- an unclear message, or one the step could not read, is a question back with three choices.

**A confirm** (`.../messages/{n}/confirm`) acts on the proposal as shown, from the dashboard it is for. It claims the
message before it writes (a compare-and-set on its state), so a second confirm, from a double click or another tab,
does nothing; a build whose dashboard is busy or changed since its plan is planned again as a new message; a change is
applied exactly as previewed, or not at all where the dashboard or a chart changed since.

**A reopened answer's table** (`.../messages/{n}/results`) is its SQL decided and run again as the person, now, and
kept nowhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from mnemiq.agent.loop import result_preview
from mnemiq.contract import HistoryTurn
from mnemiq.execute.runner import ExecutionError
from mnemiq.generate.plan_query import deferral_of_refusal
from mnemiq.server.sse import chat_stream, frame
from mnemiq.server.superset_build import (
    BuildAlreadyStarted,
    BuildKit,
    PlanStale,
    _build,
    _builder,
    _changed,
    _refusal,
    confirm_proposed_build,
)
from mnemiq.sql.verdict import Refusal
from mnemiq.superset.build_store import MAX_ID, BuildNotFound, BuildStoreMissing, DashboardBusy
from mnemiq.superset.builder import build_state
from mnemiq.superset.build_store import PROPOSED
from mnemiq.superset.chart_change import (
    STALE_SINCE_PREVIEW,
    ChangeAnswer,
    ChangeInterrupted,
    ChangePreview,
    apply_change,
    preview_change,
)
from mnemiq.superset.conversations import (
    LIST_LIMIT,
    MAX_LIST,
    MAX_MESSAGES,
    MAX_TEXT,
    Conversation,
    ConversationFull,
    ConversationNotFound,
    ConversationStore,
    cursor_of,
)
from mnemiq.superset.executor import superset_executor
from mnemiq.superset.history import run_decided
from mnemiq.superset.intent import CHOOSE_QUESTION, OFF_DASHBOARD_QUESTION, Intent, PageFacts, Routed, recent_of
from mnemiq.superset.measure_actions import DEFAULT_TIMEOUT_S

__all__ = ["mount_superset_conversations"]

logger = logging.getLogger(__name__)

PREFIX = "/v1/superset/conversations"
NOT_YOURS = "This person has no conversation of that id."
NO_SUCH_MESSAGE = "This conversation has no message of that number."
NOT_WAITING = "Nothing in this message waits for a confirm."
ALREADY_DONE = "This was already done."
MAYBE_DONE = "This may or may not have been done; check the dashboard."
SET_ASIDE = "This was set aside; ask again to have it planned anew."
OPEN_IT = "Open the dashboard this was for, then confirm."
REPLANNED = {DashboardBusy: "The dashboard is being built; this was planned again.",
             PlanStale: "The dashboard changed since this plan; it was planned again."}
NOT_REPLANNED = "It could not be planned again: {detail}"
CHOOSE_AN_OFFER = "Choose add or new for an offered dashboard."
REMOVE_ONLY_PROPOSED = "Only the charts this change would remove can be confirmed for removal."
NO_ANSWER = "This message has no answer to show again."
RERUN_REFUSED = ("This answer's query no longer passes mnemiq's checks, so it was not run again; ask the question "
                 "again for a fresh answer.")
RERUN_FAILED = "The query did not run again; ask the question again for a fresh answer."
CONVERSATION_FULL = "This conversation is full; start a new one."
_OPAQUE = "This could not be done; the server's log says why."
CHOICES = ["ask", "build", "change"]
# The earlier answers an ask is reminded of, newest last, and the rows a reopened answer shows.
HISTORY_TURNS = 5
RESULT_ROWS = 100
_KIND = {Intent.ASK: "ask", Intent.BUILD_ONTO: "build", Intent.BUILD_NEW: "build", Intent.CHANGE: "change",
         Intent.UNCLEAR: "note"}
_SETTLED = frozenset({"done", "refused", "stale", "partial", "replanned", "chosen"})
_ANSWERED = {"applying": MAYBE_DONE, "declined": SET_ASIDE, **{state: ALREADY_DONE for state in _SETTLED}}
_STREAM_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}
_Id = Annotated[int, Path(ge=0, lt=MAX_MESSAGES)]


def _open_conversations(settings) -> ConversationStore:  # seam for tests
    return ConversationStore.for_settings(settings)


class MessageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[StrictStr, Field(min_length=1, max_length=MAX_TEXT)]
    # The dashboard on screen when the message was sent, as the panel shows it; none off a dashboard. The page's
    # facts steer the intent step only: every write is checked against Superset when it is confirmed.
    dashboard_id: Annotated[StrictInt, Field(gt=0, le=MAX_ID)] | None = None
    dashboard_title: Annotated[StrictStr, Field(max_length=500)] | None = None
    dashboard_has_charts: StrictBool | None = None
    # The person's own choice after a question back: the intent step is not asked.
    choose: Literal["ask", "build", "change"] | None = None
    mode: StrictStr | None = None
    profile: StrictStr | None = None


class ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The dashboard on screen when the person confirms.
    dashboard_id: Annotated[StrictInt, Field(gt=0, le=MAX_ID)] | None = None
    # The charts the person confirmed removing, of those a change's preview would remove.
    remove_chart_ids: Annotated[list[Annotated[StrictInt, Field(gt=0, le=MAX_ID)]],
                                Field(min_length=1, max_length=250)] | None = None
    # An offered dashboard's answer: add this goal's charts to it, or build a new one.
    choice: Literal["add", "new"] | None = None


def _conversation(found: Conversation) -> dict:
    return {"id": found.conversation_id, "title": found.title, "created_at": found.created_at,
            "updated_at": found.updated_at,
            "messages": [{"index": message.index, **message.doc} for message in found.messages]}


def _conflict(detail: str, **more) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": detail, **more})


def _custom(name: str, value: dict) -> str:
    return frame({"type": "CUSTOM", "name": name, "value": value})


def _chosen(choose: str, text: str, page: PageFacts) -> Routed:
    """The person's own choice after a question back, the message as they typed it."""
    if choose == "ask":
        return Routed(Intent.ASK, text)
    if choose == "build":
        return Routed(Intent.BUILD_ONTO if page.on_dashboard else Intent.BUILD_NEW, text)
    if not page.on_dashboard:
        return Routed(Intent.UNCLEAR, text, OFF_DASHBOARD_QUESTION)
    return Routed(Intent.CHANGE, text)


def _history(conversation: Conversation) -> list[HistoryTurn]:
    """The conversation's latest answers with SQL, each as the question it answered, its SQL and its grant
    fingerprint: what a follow-up ask is reminded of. Never a row: the Ask path runs the newest again as the person."""
    turns, question = [], None
    asked = {message.index: message.doc.get("text") for message in conversation.messages
             if message.doc.get("role") == "person"}
    for message in conversation.messages:
        doc = message.doc
        if doc.get("role") == "person":
            question = doc.get("text")
            continue
        # Paired by the question it replies to, so two tabs asking at once are never crossed; by order only for an
        # answer kept without it.
        replied = asked.get(doc["replies_to"]) if isinstance(doc.get("replies_to"), int) else question
        if doc.get("kind") == "answer" and replied and (doc.get("answer") or {}).get("sql"):
            question = replied
            answer = doc["answer"]
            turns.append(HistoryTurn(question=question, sql=answer["sql"],
                                     tables_used=list(answer.get("tables_used") or []),
                                     grant_fingerprint=answer.get("grant_fingerprint") or ""))
    return turns[-HISTORY_TURNS:]


def _planned_said(proposal: dict) -> str:
    titles = [chart.get("title") or "a chart" for chart in (proposal.get("plan") or {}).get("charts") or []]
    count = f"{len(titles)} chart{'s' if len(titles) != 1 else ''}"
    where = "this dashboard" if proposal.get("dashboard_id") is not None else "a new dashboard"
    return f"I'll add {count} to {where}: {'; '.join(titles)}." if titles else f"Nothing to add to {where}."


async def _off(work: Callable, *args) -> Any:
    """`work(*args)` on a worker thread that carries none of this request's context variables: the Superset client
    starts an event loop of its own there, which it refuses to do on a thread its async library reads as running one
    (`asyncio.to_thread` copies that reading across)."""
    return await asyncio.get_running_loop().run_in_executor(None, lambda: work(*args))


# The asks being read to their end, held so a task whose reader left is not collected before it keeps its answer.
_reading: set[asyncio.Task] = set()


async def _ask_and_keep(stream: AsyncIterator[str], *, keep_answer: Callable[[dict], int],
                        keep_note: Callable[[str], int], out: asyncio.Queue) -> None:
    """Read the Ask path's `stream` to its end, putting each frame on `out` for whoever is reading the response, with
    the answer kept (`keep_answer`, then a `mnemiq.recorded` frame) as it is given, or the run's error kept as a note.
    `out` ends with None. Nothing here waits on the reader: one that left stops reading, and the answer is kept."""
    try:
        async for chunk in stream:
            event = json.loads(chunk[6:]) if chunk.startswith("data: ") else {}
            if event.get("type") == "CUSTOM" and event.get("name") == "mnemiq.answer":
                out.put_nowait(chunk)
                out.put_nowait(_custom("mnemiq.recorded", {"index": await _off(keep_answer, event.get("value") or {})}))
                continue
            if event.get("type") == "RUN_ERROR":
                kept = await _off(keep_note, str(event.get("message") or _OPAQUE))
                out.put_nowait(_custom("mnemiq.recorded", {"index": kept}))
            out.put_nowait(chunk)
    except Exception:
        logger.exception("an ask in a conversation could not be kept")
        out.put_nowait(frame({"type": "RUN_ERROR", "message": _OPAQUE}))
    finally:
        out.put_nowait(None)


def _outcome(state: str, answer: ChangeAnswer) -> dict:
    return {"state": state, "changed": list(answer.changed),
            "left": [{"chart_id": chart_id, "reason": reason} for chart_id, reason in answer.left],
            "dropped": [{"chart_id": chart_id, "key": key} for chart_id, key in answer.dropped],
            "not_editor": [{"chart_id": e.chart_id, "editors": list(e.editors)} for e in answer.not_editor],
            "removed": list(answer.removed), "permanent": answer.permanent}


def _refusal_detail(exc: Exception, doors) -> str | None:
    refusal = _refusal(exc, doors)
    if refusal is None:
        return None
    detail = refusal.detail
    return detail if isinstance(detail, str) else str(detail.get("message") or detail)


def mount_superset_conversations(app: FastAPI, runtime, doors, asker, build: BuildKit, *,
                                 heartbeat_s: float = 15.0) -> None:
    """Mount the conversation routes on `app`, each behind `asker`, the `/v1` door's dependency, sharing `build`'s
    store and flows. The conversations are kept in the deployment's control Postgres (`MNEMIQ_CONTROL_DSN`); without
    one, every conversation route answers 503 naming it, as the build routes do."""
    deployment = build.deployment
    try:
        talks = _open_conversations(deployment)
    except BuildStoreMissing:
        talks = None
    answered = build.answered

    def held() -> ConversationStore:
        if talks is None:
            raise BuildStoreMissing()
        return talks

    def mine(asking_, conversation_id: str):
        """The asker and their conversation, or a 404 that says nothing of whose it is."""
        who, _ = _builder(asking_, building=False)
        found = held().load(who.tenant_id, conversation_id, who.principal_id)
        if found is None:
            raise HTTPException(status_code=404, detail=NOT_YOURS)
        return who, found

    def message_at(found: Conversation, index: int):
        if index >= len(found.messages):
            raise HTTPException(status_code=404, detail=NO_SUCH_MESSAGE)
        return found.messages[index]

    def keep(who, conversation_id: str, doc: dict) -> int:
        try:
            return held().append(who.tenant_id, conversation_id, who.principal_id, doc)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail=NOT_YOURS) from None
        except ConversationFull:
            raise HTTPException(status_code=409, detail=CONVERSATION_FULL) from None

    def settle(who, conversation_id: str, index: int, outcome: dict, expect: str) -> bool:
        return held().settle(who.tenant_id, conversation_id, who.principal_id, index, outcome, expect=expect)

    def kept_start(who, conversation_id: str, answer: dict, *, goal: str, dashboard,
                   replies_to: int | None = None) -> tuple[str, dict, dict]:
        """A start's answer kept as mnemiq's message: its frame's name, the frame's value, and the message as kept."""
        if "proposal" in answer:
            proposal = answer["proposal"]
            doc = {"role": "mnemiq", "kind": "build", "text": _planned_said(proposal), "dashboard": dashboard,
                   "build_id": proposal["build_id"], "goal": proposal["goal"], "proposal": proposal,
                   "outcome": {"state": "proposed"}}
            name, value = "mnemiq.proposal", dict(proposal)
        elif "offer" in answer:
            offer = answer["offer"]
            named = repr(offer["title"]) if offer.get("title") else f"dashboard {offer['dashboard_id']}"
            said = (f"{named} already covers this. Open it, add these charts to it, or build a new dashboard.")
            doc = {"role": "mnemiq", "kind": "build", "text": said, "dashboard": dashboard, "goal": goal,
                   "offer": {"dashboard_id": offer["dashboard_id"], "title": offer.get("title"),
                             "reason": offer.get("reason")},
                   "outcome": {"state": "offered"}}
            name, value = "mnemiq.offer", {**offer, "choices": ["open", "add", "new"]}
        elif "refused" in answer:
            refused = answer["refused"]
            doc = {"role": "mnemiq", "kind": "note", "text": refused["reason"], "dashboard": dashboard,
                   "refused": refused}
            name, value = "mnemiq.refused", dict(refused)
        elif "shown" in answer:
            shown = answer["shown"]
            doc = {"role": "mnemiq", "kind": "note", "text": shown["message"], "dashboard": dashboard,
                   "shown": {"dashboard_id": shown["dashboard_id"], "left_out": shown["left_out"]}}
            name, value = "mnemiq.shown", dict(shown)
        else:
            raise RuntimeError("a proposed start answered with something a conversation does not keep")
        if replies_to is not None:
            doc["replies_to"] = replies_to
        index = keep(who, conversation_id, doc)
        return name, {"index": index, **value}, {"index": index, **doc}

    def proposed(asking_, who, conversation_id: str, goal: str, *, dashboard_id, dashboard, fresh=False,
                 profile=None, replies_to: int | None = None) -> tuple[str, dict, dict]:
        _, answer = build.started(asking_, goal, dashboard_id=dashboard_id, fresh=fresh, profile_name=profile,
                                  propose=True)
        return kept_start(who, conversation_id, answer, goal=goal, dashboard=dashboard, replies_to=replies_to)

    def note(who, conversation_id: str, text: str, dashboard, replies_to: int | None = None) -> int:
        doc = {"role": "mnemiq", "kind": "note", "text": text[:MAX_TEXT], "dashboard": dashboard}
        if replies_to is not None:
            doc["replies_to"] = replies_to
        return keep(who, conversation_id, doc)

    # --- the conversations

    @app.get(PREFIX)
    def listed(before: Annotated[str | None, Query(max_length=200)] = None,
               limit: Annotated[int, Query(ge=1, le=MAX_LIST)] = LIST_LIMIT, asking_=Depends(asker)) -> dict:
        def work() -> dict:
            who, _ = _builder(asking_, building=False)
            try:
                found = held().list(who.tenant_id, who.principal_id, limit=limit, before=before)
            except ValueError:
                raise HTTPException(status_code=422, detail="That is not a cursor this list gave.") from None
            return {"conversations": [{"id": c.conversation_id, "title": c.title, "updated_at": c.updated_at}
                                      for c in found],
                    "next": cursor_of(found[-1]) if len(found) == limit else None}

        return answered(work)

    @app.post(PREFIX, status_code=201)
    def start(asking_=Depends(asker)) -> dict:
        def work() -> dict:
            who, _ = _builder(asking_, building=False)
            return {"conversation": _conversation(held().create(who.tenant_id, who.principal_id, ""))}

        return answered(work)

    @app.get(PREFIX + "/{conversation_id}")
    def read(conversation_id: str, asking_=Depends(asker)) -> dict:
        return answered(lambda: {"conversation": _conversation(mine(asking_, conversation_id)[1])})

    @app.post(PREFIX + "/{conversation_id}/delete", status_code=204)
    def delete(conversation_id: str, asking_=Depends(asker)) -> Response:
        """Delete the conversation. The builds it proposed and nobody confirmed are withdrawn with it; a build it
        started keeps its own record."""

        def work() -> Response:
            who, found = mine(asking_, conversation_id)
            for message in found.messages:
                doc = message.doc
                if (doc.get("outcome") or {}).get("state") == "proposed" and doc.get("build_id"):
                    build.kept().withdraw_proposal(who.tenant_id, doc["build_id"], who.principal_id)
            held().delete(who.tenant_id, conversation_id, who.principal_id)
            return Response(status_code=204)

        return answered(work)

    # --- a message

    @app.post(PREFIX + "/{conversation_id}/messages")
    def message(conversation_id: str, body: MessageBody, asking_=Depends(asker)) -> StreamingResponse:
        """The person's message, routed, recorded and answered as an event stream (the module's docstring). Who asks,
        whose conversation it is, and the profile are settled before the stream starts, so each is a plain refusal."""

        def ready():
            who, found = mine(asking_, conversation_id)
            return who, found, runtime.build_profile(body.profile, who)

        who, found, profile = answered(ready)
        return StreamingResponse(converse(asking_, who, found, profile, body), media_type="text/event-stream",
                                 headers=_STREAM_HEADERS)

    async def waited(work: Callable[[], Any]) -> AsyncIterator:
        """`work` run on a worker thread, a keep-alive comment yielded while it runs, then its result (an exception
        is yielded as itself): planning can take longer than a proxy waits on a quiet stream."""
        task = asyncio.ensure_future(_off(work))
        while True:
            # The heartbeat is told from the work's own outcome by whether the task is done, never by an exception's
            # type: work that raises TimeoutError is a failure to answer, not a beat to wait out again.
            done, _ = await asyncio.wait({task}, timeout=heartbeat_s)
            if not done:
                yield ": keep-alive\n\n"
                continue
            failure = task.exception()
            yield failure if failure is not None else task.result()
            return

    async def converse(asking_, who, found: Conversation, profile, body: MessageBody) -> AsyncIterator[str]:
        conversation_id = found.conversation_id
        dashboard = ({"id": body.dashboard_id, "title": body.dashboard_title} if body.dashboard_id is not None
                     else None)
        page = PageFacts(dashboard_title=body.dashboard_title if body.dashboard_id is not None else None,
                         dashboard_has_charts=(body.dashboard_has_charts is not False)
                         if body.dashboard_id is not None else None)
        if body.choose is not None:
            routed = _chosen(body.choose, body.text, page)
        elif profile.intent is None:
            routed = None
        else:
            routed = None
            async for item in waited(lambda: profile.intent.read(body.text, page, recent_of(found.messages))):
                if isinstance(item, str):
                    yield item
                elif not isinstance(item, Exception):
                    routed = item
        intent = routed.intent if routed is not None else Intent.UNCLEAR
        person = {"role": "person", "kind": _KIND[intent], "text": body.text, "dashboard": dashboard,
                  "intent": intent.value}
        if intent in (Intent.BUILD_ONTO, Intent.BUILD_NEW, Intent.CHANGE):
            person["goal"] = routed.text
        try:
            index = await _off(keep, who, conversation_id, person)
        except HTTPException as refused:
            yield frame({"type": "RUN_ERROR", "message": refused.detail})
            return
        yield _custom("mnemiq.routed", {"intent": intent.value, "text": routed.text if routed else body.text,
                                        "index": index})
        if intent is Intent.ASK:
            async for chunk in asked(asking_, who, found, body, dashboard, index):
                yield chunk
            return
        yield frame({"type": "RUN_STARTED", "runId": f"{conversation_id}:{index}"})
        if intent is Intent.UNCLEAR:
            question = routed.question if routed is not None and routed.question else CHOOSE_QUESTION
            clarify = {"question": question, "choices": CHOICES}
            kept = await _off(keep, who, conversation_id, {
                "role": "mnemiq", "kind": "clarify", "text": question, "dashboard": dashboard, "clarify": clarify,
                "replies_to": index})
            yield _custom("mnemiq.clarify", {"index": kept, **clarify})
            yield frame({"type": "RUN_FINISHED", "runId": f"{conversation_id}:{index}"})
            return

        def acted() -> list[str]:
            if intent is Intent.CHANGE:
                return changed(asking_, who, conversation_id, routed.text, body, dashboard, index)
            name, value, _ = proposed(asking_, who, conversation_id, routed.text,
                                      dashboard_id=body.dashboard_id if intent is Intent.BUILD_ONTO else None,
                                      dashboard=dashboard, profile=body.profile, replies_to=index)
            return [_custom(name, value)]

        async for item in waited(acted):
            if isinstance(item, str):
                yield item
                continue
            if isinstance(item, Exception):
                detail = _refusal_detail(item, doors)
                if detail is None:
                    logger.error("a conversation's message failed", exc_info=item)
                    detail = _OPAQUE
                await _off(note, who, conversation_id, detail, dashboard, index)
                yield frame({"type": "RUN_ERROR", "message": detail})
                return
            for chunk in item:
                yield chunk
        yield frame({"type": "RUN_FINISHED", "runId": f"{conversation_id}:{index}"})

    def changed(asking_, who, conversation_id: str, request: str, body: MessageBody, dashboard,
                replies_to: int) -> list[str]:
        """A change previewed and kept waiting for "Apply"; a chart it asks for proposed beside it as a build."""
        ask = build.asking(asking_, body.dashboard_id, changing=True)
        interpreter = runtime.build_profile(body.profile, who).interpreter
        if interpreter is None:
            raise RuntimeError("this runtime was built with no change interpreter, so it changes no dashboard")
        preview = preview_change(ask, request, interpreter=interpreter)
        record = preview.to_json()
        if preview.edits or preview.would_remove:
            doc = {"role": "mnemiq", "kind": "change", "text": preview.message[:MAX_TEXT], "dashboard": dashboard,
                   "proposal": record, "outcome": {"state": "proposed"}, "replies_to": replies_to}
        else:
            doc = {"role": "mnemiq", "kind": "note", "text": preview.message[:MAX_TEXT], "dashboard": dashboard,
                   "replies_to": replies_to}
        frames = [_custom("mnemiq.preview", {"index": keep(who, conversation_id, doc), **record})]
        if preview.add_chart:
            name, value, _ = proposed(asking_, who, conversation_id, request, dashboard_id=body.dashboard_id,
                                      dashboard=dashboard, profile=body.profile, replies_to=replies_to)
            frames.append(_custom(name, value))
        return frames

    async def asked(asking_, who, found: Conversation, body: MessageBody, dashboard,
                    question: int) -> AsyncIterator[str]:
        """The Ask path's frames, unchanged, with the answer kept as its sentence and SQL once it is given. The stream
        is read to its end on a task of its own, so an answer whose reader left (the panel closed or reloaded) is
        kept all the same."""
        conversation_id = found.conversation_id

        def keep_answer(value: dict) -> int:
            return keep(who, conversation_id, {
                "role": "mnemiq", "kind": "answer", "text": "", "dashboard": dashboard, "replies_to": question,
                "answer": {"sentence": str(value.get("answer") or "")[:MAX_TEXT], "sql": value.get("sql") or "",
                           "grant_fingerprint": value.get("grant_fingerprint"),
                           "tables_used": [str(t) for t in value.get("tables_used") or []]}})

        stream = chat_stream(runtime, asking_.identity, body.text, body.mode, heartbeat_s=heartbeat_s,
                             history=_history(found), profile=body.profile, **asking_.handed)
        out: asyncio.Queue = asyncio.Queue()
        task = asyncio.ensure_future(_ask_and_keep(
            stream, keep_answer=lambda value: keep_answer(value),
            keep_note=lambda text: note(who, conversation_id, text, dashboard, question), out=out))
        _reading.add(task)
        task.add_done_callback(_reading.discard)
        while (chunk := await out.get()) is not None:
            yield chunk

    # --- a confirm

    @app.post(PREFIX + "/{conversation_id}/messages/{index}/confirm")
    def confirm(conversation_id: str, index: _Id, body: ConfirmBody, asking_=Depends(asker)):
        def work():
            who, found = mine(asking_, conversation_id)
            doc = message_at(found, index).doc
            state = (doc.get("outcome") or {}).get("state")
            if state == "offered":
                return chose(asking_, who, conversation_id, index, doc, body)
            if state == "applying" and doc.get("kind") == "build" and recovered(who, conversation_id, index, doc):
                state = "proposed"
            if state in _ANSWERED:
                return _conflict(_ANSWERED[state])
            if state != "proposed":
                return _conflict(NOT_WAITING)
            if doc.get("kind") == "change":
                return applied(asking_, who, conversation_id, index, doc, body)
            return started_build(asking_, who, conversation_id, index, doc, body)

        return answered(work)

    def recovered(who, conversation_id: str, index: int, doc: dict) -> bool:
        """A build's message left `applying` (a server stopped between claiming "Build it" and recording it) set right
        from the build store, which knows: still proposed, the message waits again (True); started since, it was
        done."""
        build_id = (doc.get("proposal") or {}).get("build_id")
        found = build.kept().load_build(who.tenant_id, build_id, who.principal_id) if build_id else None
        if found is not None and found.status == PROPOSED:
            return settle(who, conversation_id, index, {"state": "proposed"}, "applying")
        if found is not None:
            settle(who, conversation_id, index, {"state": "done", "build_id": build_id}, "applying")
        return False

    def started_build(asking_, who, conversation_id: str, index: int, doc: dict, body: ConfirmBody):
        """The person's "Build it": the build confirmed as the build route confirms it, or planned again."""
        proposal = doc["proposal"]
        if proposal.get("dashboard_id") is not None and body.dashboard_id != proposal["dashboard_id"]:
            return _conflict(OPEN_IT)
        if not settle(who, conversation_id, index, {"state": "applying"}, "proposed"):
            return _conflict(ALREADY_DONE)
        builds = build.kept()
        try:
            stored = confirm_proposed_build(asking_, builds, proposal["build_id"], deployment=deployment)
        except (DashboardBusy, PlanStale) as exc:
            builds.withdraw_proposal(who.tenant_id, proposal["build_id"], who.principal_id)
            settle(who, conversation_id, index, {"state": "replanned"}, "applying")
            said = REPLANNED[type(exc)]
            try:
                _, _, again = proposed(asking_, who, conversation_id, proposal["goal"],
                                       dashboard_id=proposal.get("dashboard_id"), dashboard=doc.get("dashboard"))
            except Exception as failed:
                detail = _refusal_detail(failed, doors)
                if detail is None:
                    raise
                text = NOT_REPLANNED.format(detail=detail)
                again = {"index": note(who, conversation_id, text, doc.get("dashboard")), "role": "mnemiq",
                         "kind": "note", "text": text}
            return _conflict(said, message=again)
        except (BuildAlreadyStarted, BuildNotFound):
            settle(who, conversation_id, index, {"state": "done", "build_id": proposal["build_id"]}, "applying")
            return _conflict(ALREADY_DONE)
        except Exception:
            # Nothing was confirmed: the store's confirm is one transaction, and the build's own state is read first
            # by any later confirm. The person may confirm again.
            settle(who, conversation_id, index, {"state": "proposed"}, "applying")
            raise
        settle(who, conversation_id, index, {"state": "done", "build_id": proposal["build_id"]}, "applying")
        return {"build": _build(build_state(builds, stored))}

    def applied(asking_, who, conversation_id: str, index: int, doc: dict, body: ConfirmBody):
        """The person's "Apply": the preview written exactly, or not at all where it no longer holds."""
        preview = ChangePreview.from_json(doc["proposal"])
        if body.dashboard_id != preview.dashboard_id:
            return _conflict(OPEN_IT)
        removing = tuple(dict.fromkeys(body.remove_chart_ids or ()))
        if not set(removing) <= {chart_id for chart_id, _ in preview.would_remove}:
            raise HTTPException(status_code=422, detail=REMOVE_ONLY_PROPOSED)
        ask = build.asking(asking_, preview.dashboard_id, changing=True)
        if not settle(who, conversation_id, index, {"state": "applying"}, "proposed"):
            return _conflict(ALREADY_DONE)
        try:
            answer = apply_change(ask, preview, strict=True, remove=removing)
        except ChangeInterrupted as stopped:
            # Written in part before Superset stopped answering: recorded as it is known, never applied again.
            settle(who, conversation_id, index, _outcome("partial", stopped.answer), "applying")
            raise stopped.__cause__ or stopped from None
        except Exception:
            # Failed before its first write: nothing was written, so the preview waits for its confirm again.
            settle(who, conversation_id, index, {"state": "proposed"}, "applying")
            raise
        stale = [chart_id for chart_id, why in answer.left if why == STALE_SINCE_PREVIEW]
        state = ("partial" if answer.changed or answer.removed else "stale") if stale else "done"
        settle(who, conversation_id, index, _outcome(state, answer), "applying")
        return {"change": _changed(answer), "state": state}

    def chose(asking_, who, conversation_id: str, index: int, doc: dict, body: ConfirmBody):
        """An offered dashboard's answer: this goal's charts proposed onto it, or a new dashboard proposed."""
        if body.choice is None:
            raise HTTPException(status_code=422, detail=CHOOSE_AN_OFFER)
        if not settle(who, conversation_id, index, {"state": "chosen"}, "offered"):
            return _conflict(ALREADY_DONE)
        onto = doc["offer"]["dashboard_id"] if body.choice == "add" else None
        try:
            _, _, again = proposed(asking_, who, conversation_id, doc["goal"], dashboard_id=onto,
                                   dashboard=doc.get("dashboard"), fresh=body.choice == "new")
        except Exception:
            settle(who, conversation_id, index, {"state": "offered"}, "chosen")
            raise
        return {"message": again}

    @app.post(PREFIX + "/{conversation_id}/messages/{index}/decline")
    def decline(conversation_id: str, index: _Id, asking_=Depends(asker)):
        """The person's "Not now": the proposal set aside, a build's proposal withdrawn."""

        def work():
            who, found = mine(asking_, conversation_id)
            doc = message_at(found, index).doc
            state = (doc.get("outcome") or {}).get("state")
            if state not in ("proposed", "offered"):
                return _conflict(_ANSWERED.get(state, NOT_WAITING))
            if not settle(who, conversation_id, index, {"state": "declined"}, state):
                return _conflict(ALREADY_DONE)
            if doc.get("build_id"):
                build.kept().withdraw_proposal(who.tenant_id, doc["build_id"], who.principal_id)
            return {"state": "declined"}

        return answered(work)

    # --- a reopened answer

    @app.post(PREFIX + "/{conversation_id}/messages/{index}/results")
    def results(conversation_id: str, index: _Id, asking_=Depends(asker)) -> dict:
        """The answer's SQL decided and run again as the person, now: its rows answered, and kept nowhere. A refusal
        is the Ask path's own sentence."""

        def work() -> dict:
            who, found = mine(asking_, conversation_id)
            answer = message_at(found, index).doc.get("answer") or {}
            if not answer.get("sql"):
                raise HTTPException(status_code=404, detail=NO_ANSWER)
            _, delegation = _builder(asking_, building=False)
            adapter = superset_executor(deployment, delegation)
            grants = runtime.builder_grants(adapter)
            settings = runtime.build_profile(None, who).settings
            try:
                outcome = run_decided(answer["sql"], adapter=adapter, snapshot=runtime.snapshot, grants=grants,
                                      timeout_s=DEFAULT_TIMEOUT_S, guard_fanout=settings.guard_fanout)
            except ExecutionError as exc:
                logger.warning("a reopened answer's query did not run again: %s", exc.full_text)
                return {"refused": RERUN_FAILED}
            if isinstance(outcome, Refusal):
                deferred = deferral_of_refusal(outcome)
                return {"refused": deferred.reason if deferred is not None else RERUN_REFUSED}
            cap = getattr(outcome.approved, "row_cap", None)
            preview = result_preview(outcome.table, RESULT_ROWS,
                                     capped=cap is not None and outcome.table.num_rows >= cap)
            return {"columns": preview.columns, "rows": preview.rows, "row_count": preview.row_count,
                    "truncated": preview.truncated, "capped": preview.capped}

        return answered(work)

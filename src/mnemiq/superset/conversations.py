"""Each person's conversations with mnemiq in the Superset panel, in a table of mnemiq's control Postgres
(`MNEMIQ_CONTROL_DSN`) beside the builds and the dashboards' measures, made the first time a store connects.

A conversation is its tenant's and principal's alone: every read and write names both, and another person's id finds
nothing, as a build that is not yours does. It holds a title, taken from the person's first message, and its messages
in the order the store took them, as one jsonb list: an append is one `UPDATE` that adds to the list as it stands, so
two tabs sending at once both land, in an order, and neither overwrites the other.

Nothing kept is a row of results. A message is held to `MESSAGE_JSON_SHAPE`, an allow-list of its keys at every depth,
and beneath it the build store's refusal of any key a value, a row or a sample is kept under (`VALUE_SHAPED_KEYS`). An
answer keeps its question, mnemiq's sentence and its SQL; the sentence can hold numbers, and is kept because the answer
is already its owner's own, in their conversation alone. Its table is run again as the person when they ask for it,
and kept nowhere. A build's proposal keeps its plan as the build store does, a filter's typed operand under `operand`.

A proposal's outcome is settled by compare-and-set (`settle`): only while its state is the one expected, so a second
confirm of one proposal, from a double click or another tab, finds it settled and does nothing.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.types.json import Jsonb

from mnemiq.superset.build_store import MAX_ID, BuildStoreMissing, _bound_waits, _iso, _plain_json, _shaped, _text
from mnemiq.superset.planner import PLAN_JSON_SHAPE, SCALAR

if TYPE_CHECKING:
    from mnemiq.config import Settings

__all__ = ["CONVERSATION_TABLE", "INTENTS", "KINDS", "LIST_LIMIT", "MAX_LIST", "MAX_MESSAGES", "MAX_TEXT",
           "MESSAGE_JSON_SHAPE", "ROLES", "STATES", "TITLE_LENGTH",
           "Conversation", "ConversationFull", "ConversationNotFound", "ConversationStore", "ConversationSummary",
           "Message", "cursor_of", "message_shape", "outcome_shape", "read_cursor", "title_of"]

CONVERSATION_TABLE = "mnemiq_superset_conversation"

# The most messages one conversation keeps: each append rewrites the row's list, so a conversation past this asks the
# person to start a new one rather than grow without bound.
MAX_MESSAGES = 400
MAX_TEXT = 8_000
_MAX_SQL = 100_000
# A message as kept, serialised: room for a build's plan of many charts, and a bound on any one row.
_MAX_MESSAGE_CHARS = 256_000
TITLE_LENGTH = 80
LIST_LIMIT = 30
MAX_LIST = 100

ROLES = frozenset({"person", "mnemiq"})
KINDS = frozenset({"ask", "answer", "build", "change", "clarify", "note"})
# What the person's message was routed to, as the intent step answers it.
INTENTS = frozenset({"ask", "build_onto", "build_new", "change", "unclear"})
# A proposal's states: `proposed` until the person confirms, `applying` while a confirm runs, then what came of it.
STATES = frozenset({"proposed", "applying", "done", "refused", "stale", "partial", "replanned", "offered", "chosen",
                    "declined"})

_DASHBOARD_JSON = {"id": SCALAR, "title": SCALAR}
_LEFT_OUT_JSON = [{"part": SCALAR, "chart_id": SCALAR, "title": SCALAR}]
_BUILD_PROPOSAL_JSON = {"build_id": SCALAR, "goal": SCALAR, "dashboard_id": SCALAR, "plan": PLAN_JSON_SHAPE,
                        "left_out": _LEFT_OUT_JSON}
_OUTCOME_JSON = {"state": SCALAR, "detail": SCALAR, "build_id": SCALAR, "changed": [SCALAR],
                 "left": [{"chart_id": SCALAR, "reason": SCALAR}], "dropped": [{"chart_id": SCALAR, "key": SCALAR}],
                 "not_editor": [{"chart_id": SCALAR, "editors": [SCALAR]}], "removed": [SCALAR],
                 "permanent": SCALAR}
# What a kept message may hold, in `planner.PLAN_JSON_SHAPE`'s form: an object is a dict of the keys it may hold, each
# with its value's shape; a list is a one-item list of its items' shape; a tuple is any one of its shapes.
MESSAGE_JSON_SHAPE: dict = {
    "role": SCALAR, "kind": SCALAR, "text": SCALAR, "at": SCALAR,
    # The dashboard the message was sent from, as it was shown then; null off a dashboard.
    "dashboard": (_DASHBOARD_JSON, SCALAR),
    # The person's message: what it was routed to, and for a build or a change the goal or request as it stands alone.
    "intent": SCALAR, "goal": SCALAR,
    "answer": {"sentence": SCALAR, "sql": SCALAR, "grant_fingerprint": SCALAR, "tables_used": [SCALAR]},
    "build_id": SCALAR,
    "proposal": (_BUILD_PROPOSAL_JSON,),
    "offer": {"dashboard_id": SCALAR, "title": SCALAR, "reason": SCALAR},
    "refused": {"reason": SCALAR, "parts": [{"part": SCALAR, "reason": SCALAR}]},
    "shown": {"dashboard_id": SCALAR, "left_out": _LEFT_OUT_JSON},
    "clarify": {"question": SCALAR, "choices": [SCALAR]},
    "outcome": _OUTCOME_JSON,
}

# Two servers starting at once would both run `CREATE TABLE IF NOT EXISTS`; the table is made under this lock instead.
_DDL_LOCK = int.from_bytes(hashlib.sha256(b"mnemiq.superset.conversations").digest()[:8], "big", signed=True)
_CREATE_CONVERSATION = f"""
CREATE TABLE IF NOT EXISTS {CONVERSATION_TABLE} (
    tenant_id text NOT NULL,
    conversation_id uuid NOT NULL,
    principal_id text NOT NULL,
    title text NOT NULL DEFAULT '',
    messages jsonb NOT NULL DEFAULT '[]',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, conversation_id))"""
_CREATE_LIST_INDEX = (f"CREATE INDEX IF NOT EXISTS {CONVERSATION_TABLE}_list ON {CONVERSATION_TABLE} "
                      "(tenant_id, principal_id, updated_at DESC, conversation_id DESC)")
# A message's `at`, on Postgres's clock, written as `_iso` writes a time, so the two compare as text.
_NOW_ISO = "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"+00:00\"')"


class ConversationNotFound(LookupError):
    """No conversation of this id is the person's: none of the id, or another person's, which reads the same."""


class ConversationFull(Exception):
    """The conversation holds `MAX_MESSAGES`: the person starts a new one."""


@dataclass(frozen=True)
class Message:
    index: int
    role: str
    kind: str
    text: str
    at: str
    doc: Mapping[str, Any]  # the message as kept, `MESSAGE_JSON_SHAPE`'s form


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    title: str
    updated_at: str


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    messages: tuple[Message, ...]


# --- What a message may hold

def message_shape(doc: Mapping) -> dict:
    """A copy of `doc` as a message is kept, or ValueError when it holds anything `MESSAGE_JSON_SHAPE` does not admit,
    a key a value, a row or a sample is kept under at any depth, or a field out of its range."""
    if not isinstance(doc, Mapping):
        raise ValueError("a message is a mapping")
    copied = _plain_json(doc, "message", holder="a conversation")
    _shaped(copied, MESSAGE_JSON_SHAPE, "message", writer="mnemiq", kept="a conversation keeps only its messages")
    for required in ("role", "kind", "text"):
        if required not in copied:
            raise ValueError(f"a message names its {required}")
    if copied["role"] not in ROLES:
        raise ValueError("a message's role is the person or mnemiq")
    if copied["kind"] not in KINDS:
        raise ValueError("a message's kind is not one mnemiq keeps")
    _bounded(copied["text"], "message.text", MAX_TEXT)
    if (dashboard := copied.get("dashboard")) is not None:
        if not isinstance(dashboard, dict) or not _is_id(dashboard.get("id")):
            raise ValueError("a message's dashboard is its id and title, or null")
        if dashboard.get("title") is not None:
            _bounded(dashboard["title"], "message.dashboard.title", MAX_TEXT)
    if "intent" in copied and copied["intent"] not in INTENTS:
        raise ValueError("a message's intent is not one the intent step answers")
    if "goal" in copied:
        _bounded(copied["goal"], "message.goal", MAX_TEXT)
    if (answer := copied.get("answer")) is not None:
        _bounded(answer.get("sentence", ""), "message.answer.sentence", MAX_TEXT)
        _bounded(answer.get("sql", ""), "message.answer.sql", _MAX_SQL)
        if not all(isinstance(table, str) for table in answer.get("tables_used", [])):
            raise ValueError("an answer's tables are named by text")
    if "outcome" in copied:
        _outcome_state(copied["outcome"])
    if len(json.dumps(copied)) > _MAX_MESSAGE_CHARS:
        raise ValueError("a message is too large to keep")
    return copied


def outcome_shape(outcome: Mapping) -> dict:
    """A copy of a proposal's outcome as it is kept, or ValueError, as `message_shape` holds a message's."""
    held = message_shape({"role": "mnemiq", "kind": "note", "text": "", "outcome": outcome})
    return held["outcome"]


def _outcome_state(outcome: Any) -> None:
    if not isinstance(outcome, dict) or outcome.get("state") not in STATES:
        raise ValueError("an outcome names one of a proposal's states")


def _bounded(text: Any, path: str, limit: int) -> None:
    if not isinstance(text, str):
        raise ValueError(f"{path} is text")
    if len(text) > limit:
        raise ValueError(f"{path} is longer than {limit} characters")


def _is_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_ID


def title_of(text: str) -> str:
    """A conversation's title, from its first message: its words on one line, at most `TITLE_LENGTH` characters."""
    return " ".join(text.split())[:TITLE_LENGTH]


# --- Paging the list: a cursor names the last conversation a page held, by when it was updated and its id, so
# conversations updated at the same moment are neither skipped nor shown twice.

def cursor_of(summary: ConversationSummary) -> str:
    return f"{summary.updated_at}|{summary.conversation_id}"


def read_cursor(before: str) -> tuple[datetime, str]:
    """`cursor_of`'s two parts, or ValueError for anything else."""
    if not isinstance(before, str) or before.count("|") != 1:
        raise ValueError("a list's cursor is one the list gave")
    moment, conversation_id = before.split("|")
    try:
        updated_at = datetime.fromisoformat(moment)
        conversation = str(uuid.UUID(conversation_id))
    except ValueError:
        raise ValueError("a list's cursor is one the list gave") from None
    if updated_at.tzinfo is None:
        raise ValueError("a list's cursor is one the list gave")
    return updated_at, conversation


def _limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= MAX_LIST:
        raise ValueError(f"a list's limit is from 1 to {MAX_LIST}")
    return limit


def _conversation_id(value: Any) -> str | None:
    """The id as Postgres's `uuid` reads it, or None when it is not one, which no conversation has."""
    try:
        return str(uuid.UUID(value)) if isinstance(value, str) else None
    except ValueError:
        return None


def _index(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < MAX_MESSAGES else None


class ConversationStore:
    """The conversations in the control Postgres at `dsn`.

    One connection, made on the first call and again after it breaks, never when the store is made; each call holds it
    alone, so one store is shared safely across a server's threads. Its session's waits are bounded as the build
    store's are. Every argument is checked before anything is sent."""

    def __init__(self, dsn: str | None) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise BuildStoreMissing()
        self._dsn = dsn
        self._con: psycopg.Connection | None = None
        self._lock = threading.Lock()

    @classmethod
    def for_settings(cls, settings: Settings) -> ConversationStore:
        """The store at the deployment's `control_dsn`, or BuildStoreMissing naming MNEMIQ_CONTROL_DSN."""
        return cls(getattr(settings, "control_dsn", None))

    def close(self) -> None:
        with self._lock:
            self._discard()

    def __enter__(self) -> ConversationStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create(self, tenant: str, principal: str, title: str) -> Conversation:
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        _bounded(title, "a conversation's title", MAX_TEXT)
        with self._connection() as con:
            row = con.execute(
                f"INSERT INTO {CONVERSATION_TABLE} (tenant_id, conversation_id, principal_id, title) "
                "VALUES (%s, %s, %s, %s) RETURNING conversation_id, title, created_at, updated_at",
                (tenant, uuid.uuid4(), principal, title_of(title))).fetchone()
        return Conversation(conversation_id=str(row[0]), title=row[1], created_at=_iso(row[2]),
                            updated_at=_iso(row[3]), messages=())

    def list(self, tenant: str, principal: str, *, limit: int = LIST_LIMIT,
             before: str | None = None) -> list[ConversationSummary]:
        """The person's conversations, the last updated first, `limit` of them after the cursor `before`."""
        tenant, principal, limit = _text(tenant, "tenant"), _text(principal, "principal"), _limit(limit)
        after = read_cursor(before) if before is not None else None
        where = "tenant_id = %s AND principal_id = %s"
        values: list[Any] = [tenant, principal]
        if after is not None:
            where += " AND (updated_at, conversation_id) < (%s, %s)"
            values += [after[0], after[1]]
        with self._connection() as con:
            rows = con.execute(
                f"SELECT conversation_id, title, updated_at FROM {CONVERSATION_TABLE} WHERE {where} "
                "ORDER BY updated_at DESC, conversation_id DESC LIMIT %s", (*values, limit)).fetchall()
        return [ConversationSummary(conversation_id=str(row[0]), title=row[1], updated_at=_iso(row[2]))
                for row in rows]

    def load(self, tenant: str, conversation_id: str, principal: str) -> Conversation | None:
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        if (conversation := _conversation_id(conversation_id)) is None:
            return None
        with self._connection() as con:
            row = con.execute(
                f"SELECT conversation_id, title, created_at, updated_at, messages FROM {CONVERSATION_TABLE} "
                "WHERE tenant_id = %s AND conversation_id = %s AND principal_id = %s",
                (tenant, conversation, principal)).fetchone()
        if row is None:
            return None
        return Conversation(conversation_id=str(row[0]), title=row[1], created_at=_iso(row[2]),
                            updated_at=_iso(row[3]), messages=_messages(row[4]))

    def append(self, tenant: str, conversation_id: str, principal: str, message: Mapping) -> int:
        """Keep `message` at the end of the person's conversation, stamped with the store's time, and answer its
        index. The person's first message titles a conversation with none."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        doc = message_shape(message)
        doc.pop("at", None)
        title = title_of(doc["text"]) if doc["role"] == "person" else ""
        if (conversation := _conversation_id(conversation_id)) is None:
            raise ConversationNotFound()
        with self._connection() as con:
            row = con.execute(
                f"UPDATE {CONVERSATION_TABLE} SET "
                f"messages = messages || jsonb_build_array(%s::jsonb || jsonb_build_object('at', {_NOW_ISO})), "
                "title = CASE WHEN title = '' THEN %s ELSE title END, updated_at = now() "
                "WHERE tenant_id = %s AND conversation_id = %s AND principal_id = %s "
                "AND jsonb_array_length(messages) < %s RETURNING jsonb_array_length(messages) - 1",
                (Jsonb(doc), title, tenant, conversation, principal, MAX_MESSAGES)).fetchone()
            if row is not None:
                return row[0]
            held = con.execute(
                f"SELECT 1 FROM {CONVERSATION_TABLE} "
                "WHERE tenant_id = %s AND conversation_id = %s AND principal_id = %s",
                (tenant, conversation, principal)).fetchone()
        raise ConversationFull() if held is not None else ConversationNotFound()

    def settle(self, tenant: str, conversation_id: str, principal: str, index: int, outcome: Mapping, *,
               expect: str) -> bool:
        """Record `outcome` as the proposal's at `index`, only while its state is `expect`: True when it was recorded,
        False when the proposal is not the person's, has no outcome, or is in another state."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        kept = outcome_shape(outcome)
        if expect not in STATES:
            raise ValueError("a settle expects one of a proposal's states")
        conversation, at = _conversation_id(conversation_id), _index(index)
        if conversation is None or at is None:
            return False
        with self._connection() as con:
            row = con.execute(
                f"UPDATE {CONVERSATION_TABLE} SET messages = jsonb_set(messages, %s, %s), updated_at = now() "
                "WHERE tenant_id = %s AND conversation_id = %s AND principal_id = %s "
                "AND messages -> %s -> 'outcome' ->> 'state' = %s RETURNING 1",
                ([str(at), "outcome"], Jsonb(kept), tenant, conversation, principal, at, expect)).fetchone()
        return row is not None

    def delete(self, tenant: str, conversation_id: str, principal: str) -> bool:
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        if (conversation := _conversation_id(conversation_id)) is None:
            return False
        with self._connection() as con:
            row = con.execute(
                f"DELETE FROM {CONVERSATION_TABLE} "
                "WHERE tenant_id = %s AND conversation_id = %s AND principal_id = %s RETURNING 1",
                (tenant, conversation, principal)).fetchone()
        return row is not None

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        with self._lock:
            if self._con is None or self._con.closed or self._con.broken:
                self._discard()
                con = psycopg.connect(self._dsn, autocommit=True)
                try:
                    _bound_waits(con)
                    _make_tables(con)
                except BaseException:
                    con.close()
                    raise
                self._con = con
            try:
                yield self._con
            except psycopg.OperationalError:
                self._discard()  # the next call connects afresh
                raise

    def _discard(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None


def _make_tables(con: psycopg.Connection) -> None:
    with con.transaction():
        con.execute("SELECT pg_advisory_xact_lock(%s)", (_DDL_LOCK,))
        con.execute(_CREATE_CONVERSATION)
        con.execute(_CREATE_LIST_INDEX)


def _messages(docs: list) -> tuple[Message, ...]:
    return tuple(Message(index=index, role=doc["role"], kind=doc["kind"], text=doc["text"], at=doc.get("at", ""),
                         doc=doc) for index, doc in enumerate(docs))

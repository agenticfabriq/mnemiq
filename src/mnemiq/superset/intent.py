"""The intent step of the Superset panel's one conversation: what a message the person typed asks mnemiq to do.

One small call to the profile's model reads the message with the page it was sent from and the conversation's last
few messages, and answers one of five intents: a question about the data (`ask`), charts added to the dashboard on
screen (`build_onto`), a new dashboard (`build_new`), a change of the dashboard on screen (`change`), or that it cannot
tell (`unclear`, with one short question back). For a build or a change it also rewrites the message so it stands
alone, a reference back ("add that as a chart here") replaced by what it refers to; that rewrite is what the planner
or the change interpreter is given, and the person sees it in the plan or the preview.

mnemiq's own rules read the model's reply before anything acts on it (`routed_from`):
- an ask goes on to the Ask path exactly as the person typed it, whatever the model wrote;
- a change, or charts added to the dashboard on screen, needs a dashboard on screen, or it is a question back;
- a build or a change with nothing to build or change is a question back;
- a reply that cannot be read routes nowhere (None), and the person is asked to choose.

The step decides nothing that writes: every build and change it routes to waits for the person's confirm. What it is
shown of the conversation is the people's and mnemiq's words, never an answer's SQL, and a conversation keeps no row.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from mnemiq.superset.planner import MAX_GOAL

__all__ = ["CHOOSE_QUESTION", "INTENT_MAX_TOKENS", "OFF_DASHBOARD_QUESTION", "RECENT_MESSAGES", "FakeIntentReader",
           "Intent", "LLMIntentReader", "PageFacts", "Routed", "intent_prompt", "recent_of", "routed_from"]

logger = logging.getLogger(__name__)

INTENT_MAX_TOKENS = 600
# How many of the conversation's latest messages the step is shown, and how much of each.
RECENT_MESSAGES = 6
_RECENT_CHARS = 400
_MAX_QUESTION = 300

OFF_DASHBOARD_QUESTION = "Which dashboard? Open it, then ask again."
CHOOSE_QUESTION = "Should I answer this as a question about the data, build charts for it, or change this dashboard?"


class Intent(StrEnum):
    ASK = "ask"
    BUILD_ONTO = "build_onto"
    BUILD_NEW = "build_new"
    CHANGE = "change"
    UNCLEAR = "unclear"


_WRITES_ON_SCREEN = frozenset({Intent.BUILD_ONTO, Intent.CHANGE})


@dataclass(frozen=True)
class PageFacts:
    """The page a message was sent from: the dashboard on screen, by its title, and whether it holds charts. Both are
    None off a dashboard."""

    dashboard_title: str | None
    dashboard_has_charts: bool | None

    @property
    def on_dashboard(self) -> bool:
        return self.dashboard_has_charts is not None


@dataclass(frozen=True)
class Routed:
    """What a message was routed to. `text` is what the routed flow is given: the message exactly as typed for an ask
    and for a question back, the standalone rewrite for a build or a change. `question` is a question back's."""

    intent: Intent
    text: str
    question: str | None = None


_SYSTEM = """You read one message a person typed in mnemiq's chat panel inside Superset, and say what they want \
mnemiq to do with it. Answer with one JSON object: {"intent": ..., "text": ..., "question": ...}

intent is one of:
- "ask": a question about the data, answered in words and a table: what, how many, which, compare, list, trend, why. \
"show me ..." is a question. Asking never changes a dashboard.
- "build_onto": add charts to the dashboard on screen ("add a chart of ... here", "put ... on this dashboard"), or \
describe what an empty dashboard on screen should show.
- "build_new": make a new dashboard ("build me a dashboard of ...", "create a new dashboard for ...").
- "change": change what is already on the dashboard on screen: a chart's type, grouping, time grain, filters, \
order, row limit or number format, or remove a chart.
- "unclear": you cannot tell which of these the person means, or none of them fits.

text: for build_onto and build_new, the goal as it stands alone: what to show, naming the measures and how they are \
broken down, with anything that refers back to earlier messages ("that", "the same", "it") replaced by what it \
refers to. For change, the request as it stands alone, naming the chart when the person named it. For ask and \
unclear, the message as it is.
question: for unclear, one short question back to the person; otherwise null.

build_onto and change act on the dashboard on screen, so only answer them when one is on screen. When you are not \
sure the person wants a dashboard built or changed, answer ask if it reads as a question, and unclear otherwise: a \
question back is always safe."""

_RESPONSE_FORMAT = {"response_format": {"type": "json_schema", "json_schema": {"name": "message_intent", "schema": {
    "type": "object", "additionalProperties": False, "required": ["intent", "text", "question"],
    "properties": {"intent": {"type": "string", "enum": [intent.value for intent in Intent]},
                   "text": {"type": "string"}, "question": {"type": ["string", "null"]}}}}}}


def _page_line(page: PageFacts) -> str:
    if not page.on_dashboard:
        return "Page: not on a dashboard."
    title = json.dumps(page.dashboard_title or "untitled")
    holds = "which has charts" if page.dashboard_has_charts else "which has no charts yet"
    return f"Page: on the dashboard {title}, {holds}."


def intent_prompt(message: str, page: PageFacts, recent: Sequence[str]) -> tuple[str, str]:
    """The step's system prompt and its user message: the page, the conversation's latest messages (oldest first),
    and the message."""
    earlier = ("Earlier in this conversation, oldest first:\n" + "\n".join(f"- {line}" for line in recent)
               if recent else "This is the conversation's first message.")
    return _SYSTEM, f"{_page_line(page)}\n{earlier}\n\nMessage: {message}"


def recent_of(messages: Sequence) -> list[str]:
    """What the step is shown of a conversation's latest `RECENT_MESSAGES` messages: who said each, and its words, an
    answer by its sentence. Never an answer's SQL."""
    lines = []
    for message in list(messages)[-RECENT_MESSAGES:]:
        doc = message.doc
        words = doc.get("answer", {}).get("sentence") if doc.get("kind") == "answer" else None
        words = " ".join(str(words or doc.get("text") or "").split())
        if words:
            lines.append(f"{doc.get('role')}: {words[:_RECENT_CHARS]}")
    return lines


def routed_from(doc, message: str, page: PageFacts) -> Routed | None:
    """The model's reply read by mnemiq's rules (the module's docstring); None when it cannot be read."""
    if not isinstance(doc, Mapping):
        return None
    try:
        intent = Intent(doc.get("intent"))
    except ValueError:
        return None
    text, question = doc.get("text", ""), doc.get("question")
    if not isinstance(text, str) or (question is not None and not isinstance(question, str)):
        return None
    if intent is Intent.ASK:
        return Routed(Intent.ASK, message)
    if intent is Intent.UNCLEAR:
        asked = " ".join((question or "").split())
        return Routed(Intent.UNCLEAR, message, asked if 0 < len(asked) <= _MAX_QUESTION else CHOOSE_QUESTION)
    if intent in _WRITES_ON_SCREEN and not page.on_dashboard:
        return Routed(Intent.UNCLEAR, message, OFF_DASHBOARD_QUESTION)
    rewrite = " ".join(text.split())[:MAX_GOAL]
    if not rewrite:
        return Routed(Intent.UNCLEAR, message, CHOOSE_QUESTION)
    return Routed(intent, rewrite)


class LLMIntentReader:
    """Reads a message with the profile's model, its reply held to `_RESPONSE_FORMAT`. A model that fails, or answers
    what cannot be read, routes nowhere: the conversation then asks the person to choose."""

    def __init__(self, client, max_tokens: int = INTENT_MAX_TOKENS) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def read(self, message: str, page: PageFacts, recent: Sequence[str]) -> Routed | None:
        system, user = intent_prompt(message, page, recent)
        try:
            raw = self._client.complete(system, user, max_tokens=self._max_tokens, extra_body=_RESPONSE_FORMAT)
        except Exception as exc:  # an outage asks the person, as an unreadable reply does
            logger.warning("the model did not answer the intent step (%s): the person is asked to choose",
                           type(exc).__name__)
            return None
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        try:
            doc = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            doc = None
        routed = routed_from(doc, message, page)
        if routed is None:
            logger.warning("the intent step's reply could not be read: the person is asked to choose")
        return routed


@dataclass
class FakeIntentReader:
    """Answers the routings it was given, in order, the last again for any later read, recording each read."""

    routed: Routed | None | Sequence[Routed | None]
    reads: list = field(default_factory=list)

    def read(self, message: str, page: PageFacts, recent: Sequence[str]) -> Routed | None:
        self.reads.append((message, page, list(recent)))
        if isinstance(self.routed, (list, tuple)):
            return self.routed.pop(0) if len(self.routed) > 1 else self.routed[0]
        return self.routed

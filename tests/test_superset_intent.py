"""The intent step of the Superset panel's one conversation: a message routed to an ask, a build onto the dashboard on
screen, a new dashboard, a change of the dashboard on screen, or a question back. The model's reply is read by
mnemiq's own rules before anything acts on it: an ask goes on exactly as typed, a write needs the dashboard it acts on
to be on screen, and a reply that cannot be read routes nowhere, so the person is asked."""

from __future__ import annotations

import json

import pytest

from mnemiq.superset.conversations import INTENTS, Message
from mnemiq.superset.intent import (
    CHOOSE_QUESTION,
    INTENT_MAX_TOKENS,
    OFF_DASHBOARD_QUESTION,
    FakeIntentReader,
    Intent,
    LLMIntentReader,
    PageFacts,
    Routed,
    intent_prompt,
    recent_of,
    routed_from,
)

ON_CHARTS = PageFacts(dashboard_title="Q3 pipeline", dashboard_has_charts=True)
ON_EMPTY = PageFacts(dashboard_title="Untitled", dashboard_has_charts=False)
OFF = PageFacts(dashboard_title=None, dashboard_has_charts=None)


class _Client:
    """The profile's model client, answering `reply` (or raising it) and recording what it was sent."""

    def __init__(self, reply) -> None:
        self.reply = reply
        self.sent: list[dict] = []

    def complete(self, system, user, *, max_tokens, extra_body=None):
        self.sent.append({"system": system, "user": user, "max_tokens": max_tokens, "extra_body": extra_body})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _reply(intent: str, text: str = "", question: str | None = None) -> str:
    return json.dumps({"intent": intent, "text": text, "question": question})


def test_the_intents_are_the_ones_a_conversation_keeps():
    assert {intent.value for intent in Intent} == set(INTENTS)


def test_an_ask_keeps_the_message_exactly():
    message = "  revenue by   product line?  "
    routed = routed_from(json.loads(_reply("ask", "revenue grouped by product line")), message, ON_CHARTS)
    assert routed == Routed(Intent.ASK, message)


@pytest.mark.parametrize("intent", ["change", "build_onto"])
def test_a_write_onto_the_dashboard_off_a_dashboard_is_unclear(intent):
    routed = routed_from(json.loads(_reply(intent, "make the bookings chart a bar chart")), "make it a bar", OFF)
    assert routed == Routed(Intent.UNCLEAR, "make it a bar", OFF_DASHBOARD_QUESTION)
    assert OFF_DASHBOARD_QUESTION == "Which dashboard? Open it, then ask again."


def test_a_new_dashboard_needs_none_on_screen():
    routed = routed_from(json.loads(_reply("build_new", "monthly revenue by region")), "build a dashboard", OFF)
    assert routed == Routed(Intent.BUILD_NEW, "monthly revenue by region")


@pytest.mark.parametrize("doc", [None, [], "ask", {"intent": "delete"}, {"intent": "ask", "text": 3},
                                 {"intent": 1}, {"text": "x"}])
def test_an_unreadable_reply_is_none(doc):
    assert routed_from(doc, "anything", ON_CHARTS) is None


@pytest.mark.parametrize("intent", ["build_onto", "build_new", "change"])
def test_an_empty_rewrite_for_a_write_is_unclear(intent):
    routed = routed_from(json.loads(_reply(intent, "   ")), "do that", ON_CHARTS)
    assert routed == Routed(Intent.UNCLEAR, "do that", CHOOSE_QUESTION)


def test_unclear_carries_the_models_question_or_mnemiqs():
    asked = routed_from(json.loads(_reply("unclear", "", "Which chart do you mean?")), "fix it", ON_CHARTS)
    assert asked == Routed(Intent.UNCLEAR, "fix it", "Which chart do you mean?")
    silent = routed_from(json.loads(_reply("unclear", "", None)), "fix it", ON_CHARTS)
    assert silent == Routed(Intent.UNCLEAR, "fix it", CHOOSE_QUESTION)
    long = routed_from(json.loads(_reply("unclear", "", "x" * 2000)), "fix it", ON_CHARTS)
    assert long.question == CHOOSE_QUESTION


def test_a_reference_is_rewritten_standalone():
    """A build or change reaches the planner or interpreter as the rewrite, which stands alone."""
    reader = LLMIntentReader(_Client(_reply("build_onto", "a bar chart of revenue by product line last quarter")))
    routed = reader.read("add that as a chart here", ON_CHARTS,
                         ["person: revenue by product line last quarter", "mnemiq: Classic Cars led with $3.9M."])
    assert routed == Routed(Intent.BUILD_ONTO, "a bar chart of revenue by product line last quarter")


def test_the_reader_sends_the_page_the_recent_messages_and_a_response_format():
    client = _Client(_reply("ask"))
    LLMIntentReader(client).read("how many deals closed?", ON_EMPTY, ["person: hello"])
    [sent] = client.sent
    assert sent["max_tokens"] == INTENT_MAX_TOKENS
    assert sent["extra_body"]["response_format"]["type"] == "json_schema"
    assert '"Untitled"' in sent["user"] and "no charts" in sent["user"]
    assert "person: hello" in sent["user"] and "how many deals closed?" in sent["user"]


@pytest.mark.parametrize("reply", ["not json at all", "", None, RuntimeError("the model did not answer")])
def test_a_model_that_fails_routes_nowhere(reply):
    assert LLMIntentReader(_Client(reply)).read("anything", ON_CHARTS, []) is None


def test_the_prompt_never_carries_a_row_or_the_sql():
    """What the intent step is shown of earlier messages: the person's words and mnemiq's, never an answer's SQL, and
    a conversation keeps no row to show."""
    messages = (
        Message(0, "person", "ask", "revenue by product line", "t", {"role": "person", "kind": "ask",
                                                                    "text": "revenue by product line"}),
        Message(1, "mnemiq", "answer", "", "t", {"role": "mnemiq", "kind": "answer", "text": "", "answer": {
            "sentence": "Classic Cars leads, $3.9M.", "sql": "SELECT product_line, SUM(amount) FROM sales",
            "grant_fingerprint": "g", "tables_used": ["sales"]}}),
        Message(2, "mnemiq", "build", "I'll add 1 chart.", "t", {"role": "mnemiq", "kind": "build",
                                                               "text": "I'll add 1 chart.", "proposal": {
                                                                   "build_id": "b", "goal": "revenue by line",
                                                                   "dashboard_id": 3, "plan": {"charts": []}},
                                                               "outcome": {"state": "proposed"}}),
    )
    recent = recent_of(messages)
    assert recent == ["person: revenue by product line", "mnemiq: Classic Cars leads, $3.9M.",
                      "mnemiq: I'll add 1 chart."]
    system, user = intent_prompt("add that as a chart here", ON_CHARTS, recent)
    assert "SELECT" not in system + user and "SUM(amount)" not in user


def test_recent_is_the_last_six_each_bounded():
    messages = tuple(Message(i, "person", "ask", f"q{i} " + "x" * 1000, "t",
                             {"role": "person", "kind": "ask", "text": f"q{i} " + "x" * 1000}) for i in range(10))
    recent = recent_of(messages)
    assert [line.split()[1] for line in recent] == ["q4", "q5", "q6", "q7", "q8", "q9"]
    assert all(len(line) <= 420 for line in recent)


def test_the_fake_reader_answers_and_records():
    fake = FakeIntentReader(Routed(Intent.ASK, "x"))
    assert fake.read("x", OFF, []) == Routed(Intent.ASK, "x")
    assert fake.reads == [("x", OFF, [])]


def test_every_profile_has_an_intent_reader(tmp_path):
    from mnemiq.runtime import build_runtime
    from tests.scripted_deployment import deployment
    from tests.sweep.stub_model import StubModel

    with StubModel(sql="SELECT 1") as model:
        runtime = build_runtime(deployment(tmp_path, model.url))
        assert all(isinstance(profile.intent, LLMIntentReader) for profile in runtime.profiles.values())

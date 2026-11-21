"""A person's conversations in the Superset panel, kept by mnemiq: each is its owner's alone, its messages are kept in
the order the store took them, a proposal's outcome is settled once, and nothing kept is a row of results.

These run on the in-memory store. `test_superset_conversation_store.py` runs the same tests on the control Postgres,
by importing them with a fixture of its own, so the two stores are held to one contract."""

from __future__ import annotations

import threading

import pytest

from mnemiq.superset.conversations import (
    MAX_MESSAGES,
    ConversationFull,
    ConversationNotFound,
    cursor_of,
    message_shape,
)
from tests.superset_build_world import Clock, MemoryConversationStore

TENANT = "acme"
ME = "builder@acme"
SOMEONE_ELSE = "viewer@acme"


@pytest.fixture
def conversations():
    return MemoryConversationStore(Clock())


def _ask(text: str = "revenue by product line") -> dict:
    return {"role": "person", "kind": "ask", "text": text, "dashboard": {"id": 3, "title": "Q3 pipeline"}}


def _answer() -> dict:
    return {"role": "mnemiq", "kind": "answer", "text": "",
            "answer": {"sentence": "Classic Cars leads, $3.9M.", "sql": "SELECT product_line, SUM(amount) FROM sales "
                                                                        "GROUP BY 1",
                       "grant_fingerprint": "g1", "tables_used": ["sales"]}}


def _plan() -> dict:
    """A proposed build's plan as the person is shown it (`plan_summary`'s form)."""
    return {"charts": [{"key": "c1", "title": "Revenue by product line", "family": "bar", "measures": ["Revenue"]}],
            "measures": [{"name": "revenue", "label": "Revenue", "expression": "SUM(amount)", "kind": "sum",
                          "currency": "USD"}]}


def _proposal(**plan_over) -> dict:
    return {"role": "mnemiq", "kind": "build", "text": "I'll add 1 chart to \"Q3 pipeline\".",
            "proposal": {"build_id": "b1", "goal": "revenue by product line", "dashboard_id": 3,
                         "plan": {**_plan(), **plan_over}},
            "outcome": {"state": "proposed"}}


def _settle(conversations, conversation_id, index, principal=ME, *, state="applying", expect="proposed"):
    return conversations.settle(TENANT, conversation_id, principal, index, {"state": state}, expect=expect)


# --- Whose a conversation is

def test_a_conversation_is_its_owners_alone(conversations):
    mine = conversations.create(TENANT, ME, "")
    index = conversations.append(TENANT, mine.conversation_id, ME, _proposal())

    assert conversations.load(TENANT, mine.conversation_id, SOMEONE_ELSE) is None
    assert conversations.load("other-tenant", mine.conversation_id, ME) is None
    with pytest.raises(ConversationNotFound):
        conversations.append(TENANT, mine.conversation_id, SOMEONE_ELSE, _ask())
    assert _settle(conversations, mine.conversation_id, index, SOMEONE_ELSE) is False
    assert conversations.delete(TENANT, mine.conversation_id, SOMEONE_ELSE) is False
    assert conversations.list(TENANT, SOMEONE_ELSE) == []

    kept = conversations.load(TENANT, mine.conversation_id, ME)
    assert [message.kind for message in kept.messages] == ["build"]
    assert kept.messages[0].doc["outcome"] == {"state": "proposed"}
    assert conversations.delete(TENANT, mine.conversation_id, ME) is True
    assert conversations.load(TENANT, mine.conversation_id, ME) is None


def test_an_unknown_conversation_is_not_found(conversations):
    missing = "00000000-0000-4000-8000-000000000000"
    assert conversations.load(TENANT, missing, ME) is None
    with pytest.raises(ConversationNotFound):
        conversations.append(TENANT, missing, ME, _ask())
    assert conversations.delete(TENANT, missing, ME) is False


# --- Messages

def test_messages_append_in_order_and_concurrent_appends_both_land(conversations):
    talk = conversations.create(TENANT, ME, "")
    assert conversations.append(TENANT, talk.conversation_id, ME, _ask("first")) == 0
    assert conversations.append(TENANT, talk.conversation_id, ME, _answer()) == 1

    indexes: list[int] = []
    start = threading.Barrier(8)

    def send(n: int) -> None:
        start.wait()
        indexes.append(conversations.append(TENANT, talk.conversation_id, ME, _ask(f"at once {n}")))

    threads = [threading.Thread(target=send, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    kept = conversations.load(TENANT, talk.conversation_id, ME)
    assert sorted(indexes) == list(range(2, 10))
    assert [message.index for message in kept.messages] == list(range(10))
    assert [message.text for message in kept.messages[:2]] == ["first", ""]
    assert sorted(message.text for message in kept.messages[2:]) == sorted(f"at once {n}" for n in range(8))
    # Each index the store answered is the message it took.
    for index, message in zip(sorted(indexes), kept.messages[2:], strict=True):
        assert message.index == index


def test_the_store_stamps_each_message_with_its_own_time(conversations):
    talk = conversations.create(TENANT, ME, "")
    sent = {**_ask(), "at": "1999-01-01T00:00:00+00:00"}
    conversations.append(TENANT, talk.conversation_id, ME, sent)
    [message] = conversations.load(TENANT, talk.conversation_id, ME).messages
    assert message.at != "1999-01-01T00:00:00+00:00"
    assert message.at >= talk.created_at


def test_the_first_persons_message_titles_the_conversation(conversations):
    talk = conversations.create(TENANT, ME, "")
    conversations.append(TENANT, talk.conversation_id, ME, {"role": "mnemiq", "kind": "note", "text": "Hello."})
    assert conversations.load(TENANT, talk.conversation_id, ME).title == ""
    conversations.append(TENANT, talk.conversation_id, ME, _ask("  revenue   by\nproduct line " + "x" * 200))
    conversations.append(TENANT, talk.conversation_id, ME, _ask("a later question"))
    title = conversations.load(TENANT, talk.conversation_id, ME).title
    assert title.startswith("revenue by product line x")
    assert len(title) <= 80


def test_a_conversation_holds_at_most_max_messages(conversations):
    talk = conversations.create(TENANT, ME, "")
    for _ in range(MAX_MESSAGES):
        conversations.append(TENANT, talk.conversation_id, ME, _ask())
    with pytest.raises(ConversationFull):
        conversations.append(TENANT, talk.conversation_id, ME, _ask())
    assert len(conversations.load(TENANT, talk.conversation_id, ME).messages) == MAX_MESSAGES


# --- A proposal's outcome

def test_a_settled_proposal_refuses_a_second_settle(conversations):
    talk = conversations.create(TENANT, ME, "")
    conversations.append(TENANT, talk.conversation_id, ME, _ask())
    index = conversations.append(TENANT, talk.conversation_id, ME, _proposal())

    assert _settle(conversations, talk.conversation_id, index) is True
    assert _settle(conversations, talk.conversation_id, index) is False
    assert _settle(conversations, talk.conversation_id, index, state="done", expect="applying") is True
    kept = conversations.load(TENANT, talk.conversation_id, ME).messages[index]
    assert kept.doc["outcome"] == {"state": "done"}
    # A message with no outcome, or past the end, is never settled.
    assert _settle(conversations, talk.conversation_id, 0) is False
    assert _settle(conversations, talk.conversation_id, 99) is False
    assert _settle(conversations, talk.conversation_id, -1) is False


def test_two_settles_at_once_settle_once(conversations):
    talk = conversations.create(TENANT, ME, "")
    index = conversations.append(TENANT, talk.conversation_id, ME, _proposal())
    won: list[bool] = []
    start = threading.Barrier(6)

    def claim() -> None:
        start.wait()
        won.append(_settle(conversations, talk.conversation_id, index))

    threads = [threading.Thread(target=claim) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(won) == [False] * 5 + [True]


def test_an_outcome_is_held_to_the_shape(conversations):
    talk = conversations.create(TENANT, ME, "")
    index = conversations.append(TENANT, talk.conversation_id, ME, _proposal())
    with pytest.raises(ValueError):
        conversations.settle(TENANT, talk.conversation_id, ME, index, {"state": "done", "rows": [[1]]},
                             expect="proposed")
    with pytest.raises(ValueError):
        conversations.settle(TENANT, talk.conversation_id, ME, index, {"state": "whatever"}, expect="proposed")
    assert conversations.load(TENANT, talk.conversation_id, ME).messages[index].doc["outcome"] == {
        "state": "proposed"}


# --- The list

def test_list_is_newest_first_and_pages_by_before(conversations):
    clock = conversations.clock
    made = []
    for n in range(5):
        made.append(conversations.create(TENANT, ME, ""))
        clock.advance(1)
    conversations.create(TENANT, SOMEONE_ELSE, "")
    # Touching the oldest makes it the newest.
    conversations.append(TENANT, made[0].conversation_id, ME, _ask("touched"))

    first = conversations.list(TENANT, ME, limit=2)
    assert [c.conversation_id for c in first] == [made[0].conversation_id, made[4].conversation_id]
    second = conversations.list(TENANT, ME, limit=2, before=cursor_of(first[-1]))
    assert [c.conversation_id for c in second] == [made[3].conversation_id, made[2].conversation_id]
    third = conversations.list(TENANT, ME, limit=2, before=cursor_of(second[-1]))
    assert [c.conversation_id for c in third] == [made[1].conversation_id]
    assert first[0].title == "touched"


def test_list_pages_through_conversations_updated_at_the_same_moment(conversations):
    made = {conversations.create(TENANT, ME, "").conversation_id for _ in range(5)}  # the clock never moves
    seen: list[str] = []
    before = None
    while page := conversations.list(TENANT, ME, limit=2, before=before):
        seen += [c.conversation_id for c in page]
        before = cursor_of(page[-1])
    assert sorted(seen) == sorted(made)


@pytest.mark.parametrize("before", ["", "not a cursor", "2026-11-14T12:00:00+00:00", "x|y"])
def test_a_malformed_cursor_is_refused(conversations, before):
    with pytest.raises(ValueError):
        conversations.list(TENANT, ME, before=before)


@pytest.mark.parametrize("limit", [0, -1, 101, True])
def test_a_list_limit_out_of_range_is_refused(conversations, limit):
    with pytest.raises(ValueError):
        conversations.list(TENANT, ME, limit=limit)


# --- What a message may hold

@pytest.mark.parametrize("doc", [
    {**_ask(), "rows": [[1, 2]]},
    {**_answer(), "answer": {**_answer()["answer"], "data": [[1]]}},
    {**_answer(), "answer": {**_answer()["answer"], "Rows": [[1]]}},
    _proposal(measures=[{"name": "revenue", "value": 3}]),
    _proposal(charts=[{"key": "c1", "filters": [{"column": "region", "op": "=", "operand": "west"}]}]),
    {**_proposal(), "outcome": {"state": "done", "results": [1]}},
    {**_ask(), "anything": "else"},
    {**_ask(), "role": "system"},
    {**_ask(), "kind": "sql"},
    {**_ask(), "text": 3},
    {**_ask(), "text": "x" * 20_000},
    {**_ask(), "dashboard": {"id": 3, "title": "t", "rows": []}},
    {**_answer(), "answer": {**_answer()["answer"], "tables_used": [{"rows": 1}]}},
])
def test_the_shape_refuses_a_row_anywhere(doc):
    with pytest.raises(ValueError):
        message_shape(doc)


def test_a_proposed_build_keeps_its_plan_as_the_person_is_shown_it():
    """The plan as `plan_summary` writes it is kept; anything more of the plan (a filter, its operand) is not, since
    the build store keeps the plan itself."""
    from mnemiq.superset.builder import plan_summary
    from mnemiq.superset.planner import plan_to_json
    from tests.superset_shared import ONE_BAR

    shown = plan_summary(plan_to_json(ONE_BAR))
    kept = message_shape({**_proposal(), "proposal": {**_proposal()["proposal"], "plan": shown}})
    assert kept["proposal"]["plan"] == shown


def test_the_sentence_and_sql_are_kept(conversations):
    talk = conversations.create(TENANT, ME, "")
    conversations.append(TENANT, talk.conversation_id, ME, _ask())
    conversations.append(TENANT, talk.conversation_id, ME, _answer())
    [question, answer] = conversations.load(TENANT, talk.conversation_id, ME).messages
    assert question.doc["dashboard"] == {"id": 3, "title": "Q3 pipeline"}
    assert answer.doc["answer"] == _answer()["answer"]


def test_a_refused_message_is_not_kept(conversations):
    talk = conversations.create(TENANT, ME, "")
    with pytest.raises(ValueError):
        conversations.append(TENANT, talk.conversation_id, ME, {**_ask(), "rows": [[1]]})
    assert conversations.load(TENANT, talk.conversation_id, ME).messages == ()

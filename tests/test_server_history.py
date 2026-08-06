"""History crosses the wire as structure, and the engine -- not the client -- decides
which turns it may be reminded of."""

import json

from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


class _RT:
    """Records what the server handed the engine."""

    def __init__(self):
        self.history = "unset"

    def ask(self, question, identity, mode=None, emit=None, history=None):
        self.history = history
        return AgentAnswer(answer="ok", grant_fingerprint="fp-analyst")


_TURN = {
    "question": "which region has the most claims?",
    "sql": "SELECT region FROM policy",
    "tables_used": ["policy"],
    "columns": ["region"],
    "rows": [["west"]],
    "grant_fingerprint": "fp-analyst",
}


def test_ask_forwards_history_to_the_engine():
    rt = _RT()
    r = TestClient(build_app(rt, _identity())).post(
        "/v1/ask", json={"question": "and how many claims does it have?", "history": [_TURN]}
    )

    assert r.status_code == 200
    assert [t.question for t in rt.history] == ["which region has the most claims?"]
    assert rt.history[0].rows == [["west"]], "the antecedent rides with the turn"
    assert rt.history[0].grant_fingerprint == "fp-analyst"


def test_a_question_without_history_still_works():
    rt = _RT()
    r = TestClient(build_app(rt, _identity())).post("/v1/ask", json={"question": "how many?"})

    assert r.status_code == 200
    assert rt.history is None


def test_the_answer_carries_the_boundary_so_a_client_can_echo_it():
    rt = _RT()
    body = TestClient(build_app(rt, _identity())).post(
        "/v1/ask", json={"question": "q"}
    ).json()

    # Without this the client has nothing to stamp onto the turn it sends back, and the
    # engine could not tell which policy an echoed turn was answered under.
    assert body["grant_fingerprint"] == "fp-analyst"


def test_malformed_history_is_a_422_not_a_silent_drop():
    rt = _RT()
    r = TestClient(build_app(rt, _identity())).post(
        "/v1/ask", json={"question": "q", "history": [{"rows": "not-a-list"}]}
    )

    assert r.status_code == 422
    assert rt.history == "unset", "the engine was never called"


def test_chat_forwards_history_too():
    rt = _RT()
    r = TestClient(build_app(rt, _identity())).post(
        "/v1/chat", json={"question": "and how many?", "history": [_TURN]}
    )

    assert r.status_code == 200
    types = [
        json.loads(line[len("data: "):])["type"]
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    assert types[-1] == "RUN_FINISHED"
    assert [t.question for t in rt.history] == ["which region has the most claims?"]

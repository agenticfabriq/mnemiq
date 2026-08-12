from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.agent.route import UnknownMode
from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


class _RT:
    def __init__(self, answer=None, cards=(), raises=None, starters=()):
        self._answer, self._cards, self._raises = answer, list(cards), raises
        self._starters = list(starters)
        self.asked = None

    def ask(self, question, identity, mode=None, emit=None, history=None):
        if self._raises:
            raise self._raises
        self.asked = (question, identity.principal_id, mode)
        return self._answer

    def scope(self, identity):
        return {"tables": self._cards, "starters": self._starters}


def _client(rt):
    return TestClient(build_app(rt, _identity()))


def test_ask_returns_the_full_payload():
    rt = _RT(answer=AgentAnswer(answer="2.", mode="thinking"))
    r = _client(rt).post("/v1/ask", json={"question": "how many?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "2."
    assert rt.asked == ("how many?", "u", None)


def test_ask_deferral_is_a_200_not_an_error():
    rt = _RT(answer=AgentAnswer(answer="No table holds salary.", deferred=True))
    r = _client(rt).post("/v1/ask", json={"question": "avg salary?"})
    assert r.status_code == 200
    assert r.json()["deferred"] is True


def test_unknown_mode_is_422():
    rt = _RT(raises=UnknownMode("no such mode"))
    r = _client(rt).post("/v1/ask", json={"question": "q", "mode": "warp"})
    assert r.status_code == 422


def test_schema_and_healthz():
    rt = _RT(cards=[{"object_id": "claim", "card": "..."}])
    c = _client(rt)
    assert c.get("/healthz").json() == {"ok": True}
    assert c.get("/v1/schema").json() == {
        "tables": [{"object_id": "claim", "card": "..."}],
        "starters": [],
    }


def test_the_opening_questions_ride_on_the_schema_response():
    """M32: they are access-scoped alongside the cards, so they travel with them rather than
    on an endpoint of their own that could answer to a different grant resolution."""
    rt = _RT(cards=[{"object_id": "payment", "card": "..."}],
             starters=["What is the total amount in payment?"])
    body = _client(rt).get("/v1/schema").json()
    assert body["starters"] == ["What is the total amount in payment?"]

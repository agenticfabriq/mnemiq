from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.agent.route import UnknownMode
from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


class _RT:
    def __init__(self, answer=None, cards=(), raises=None):
        self._answer, self._cards, self._raises = answer, list(cards), raises
        self.asked = None

    def ask(self, question, identity, mode=None):
        if self._raises:
            raise self._raises
        self.asked = (question, identity.principal_id, mode)
        return self._answer

    def schema(self, identity):
        return self._cards


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
    assert c.get("/v1/schema").json() == {"tables": [{"object_id": "claim", "card": "..."}]}

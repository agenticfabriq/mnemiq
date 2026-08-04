import json
import time

from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=[])


class _RT:
    def __init__(self, answer=None, raises=None, delay_s=0.0):
        self._answer, self._raises, self._delay = answer, raises, delay_s

    def ask(self, question, identity, mode=None, emit=None):
        time.sleep(self._delay)
        if self._raises:
            raise self._raises
        return self._answer


def _events(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):])
            for line in text.splitlines() if line.startswith("data: ")]


def test_chat_streams_the_agui_sequence():
    rt = _RT(answer=AgentAnswer(answer="2 claims.", mode="thinking"))
    r = TestClient(build_app(rt, _identity())).post("/v1/chat", json={"question": "q"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    types = [e["type"] for e in _events(r.text)]
    assert types == ["RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT",
                     "TEXT_MESSAGE_END", "CUSTOM", "RUN_FINISHED"]
    custom = next(e for e in _events(r.text) if e["type"] == "CUSTOM")
    assert custom["name"] == "mnemiq.answer"
    assert custom["value"]["answer"] == "2 claims."


def test_chat_engine_failure_ends_in_run_error():
    rt = _RT(raises=RuntimeError("boom"))
    r = TestClient(build_app(rt, _identity())).post("/v1/chat", json={"question": "q"})
    types = [e["type"] for e in _events(r.text)]
    assert types == ["RUN_STARTED", "RUN_ERROR"]
    assert "boom" in _events(r.text)[-1]["message"]


def test_chat_emits_keepalive_comments_while_the_engine_works():
    rt = _RT(answer=AgentAnswer(answer="ok"), delay_s=0.15)
    app = build_app(rt, _identity(), heartbeat_s=0.05)
    r = TestClient(app).post("/v1/chat", json={"question": "q"})
    assert ": keep-alive" in r.text
    assert [e["type"] for e in _events(r.text)][-1] == "RUN_FINISHED"

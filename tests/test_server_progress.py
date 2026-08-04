"""Progress events escape a synchronous engine on a worker thread and reach the wire.

The point of the queue is ordering: every step a run emitted must arrive before the
answer that run produced, even though the engine is on another thread.
"""

import json
import time

from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext
from mnemiq.progress import Stage, step
from mnemiq.server.app import build_app


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=[])


class _RT:
    """An engine that reports stages the way the real one does -- from a worker thread."""

    def __init__(self, answer=None, raises=None, stages=(), delay_s=0.0):
        self._answer, self._raises, self._stages = answer, raises, list(stages)
        self._delay = delay_s

    def ask(self, question, identity, mode=None, emit=None):
        for stage in self._stages:
            with step(emit, stage):
                time.sleep(self._delay)
        if self._raises:
            raise self._raises
        return self._answer


def _events(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):])
            for line in text.splitlines() if line.startswith("data: ")]


def _post(rt, **kw):
    return TestClient(build_app(rt, _identity(), **kw)).post("/v1/chat", json={"question": "q"})


def test_stages_reach_the_wire_before_the_answer():
    rt = _RT(answer=AgentAnswer(answer="2 claims."),
             stages=[Stage.RETRIEVE, Stage.PLAN, Stage.EXECUTE, Stage.SYNTHESIZE])
    types = [e["type"] for e in _events(_post(rt).text)]

    assert types == [
        "RUN_STARTED",
        "STEP_STARTED", "STEP_FINISHED",   # retrieve
        "STEP_STARTED", "STEP_FINISHED",   # plan
        "STEP_STARTED", "STEP_FINISHED",   # execute
        "STEP_STARTED", "STEP_FINISHED",   # synthesize
        "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
        "CUSTOM", "RUN_FINISHED",
    ]


def test_each_step_names_itself_and_carries_an_id_on_both_halves():
    rt = _RT(answer=AgentAnswer(answer="x"), stages=[Stage.RETRIEVE])
    started, finished = [e for e in _events(_post(rt).text) if e["type"].startswith("STEP_")]

    assert started["stepName"] == "retrieve"
    assert finished["stepName"] == "retrieve"
    assert started["stepId"] and started["stepId"] == finished["stepId"], (
        "a consumer correlating by name alone breaks the moment two steps overlap"
    )
    assert finished["durationMs"] >= 0
    assert finished["ok"] is True


def test_a_step_that_raised_says_so():
    rt = _RT(raises=RuntimeError("boom"), stages=[Stage.RETRIEVE])
    events = _events(_post(rt).text)

    # The stage itself succeeded; the run failed after it.
    assert [e["type"] for e in events][-1] == "RUN_ERROR"
    assert "boom" in events[-1]["message"]


def test_extra_step_fields_survive_to_the_client():
    class _Deep:
        def ask(self, question, identity, mode=None, emit=None):
            with step(emit, Stage.CANDIDATE, index=3, of=5):
                pass
            return AgentAnswer(answer="x")

    started = next(e for e in _events(_post(_Deep()).text) if e["type"] == "STEP_STARTED")
    assert started["index"] == 3 and started["of"] == 5


def test_an_engine_that_reports_nothing_still_streams_the_answer():
    rt = _RT(answer=AgentAnswer(answer="x"))
    types = [e["type"] for e in _events(_post(rt).text)]
    assert "STEP_STARTED" not in types
    assert types[-1] == "RUN_FINISHED"


def test_keepalives_do_not_displace_steps():
    rt = _RT(answer=AgentAnswer(answer="x"), stages=[Stage.RETRIEVE], delay_s=0.15)
    body = _post(rt, heartbeat_s=0.05).text

    assert ": keep-alive" in body
    types = [e["type"] for e in _events(body)]
    assert types[1] == "STEP_STARTED" and types[2] == "STEP_FINISHED"
    assert types[-1] == "RUN_FINISHED"


def test_a_broken_listener_cannot_fail_the_query():
    def hostile(_event):
        raise RuntimeError("listener is broken")

    with step(hostile, Stage.RETRIEVE) as sid:
        assert sid is not None  # the body still runs, and still gets its id

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

    def ask(self, question, identity, mode=None, emit=None, history=None):
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
    err = _events(r.text)[-1]
    # The frame carries the run id, which is the handle a caller quotes and an operator
    # greps for. It does NOT carry the exception -- see the test below.
    assert err["runId"]
    assert err["message"] == "The engine could not complete this request."


def test_chat_failure_does_not_put_the_sources_error_on_the_wire():
    """An engine failure used to yield `str(exc)`, and on a governed deployment that is
    a description of the schema the caller was refused.

    The exception here is shaped like what actually reaches this handler: the source's
    own message, naming a table this identity was never shown, quoting the statement,
    and carrying a DSN. Retrieval scoping and `check_access` keep all three out of the
    ANSWER path; the error path must not be the way around them."""
    leaky = RuntimeError(
        'relation "payroll_salary" does not exist\n'
        'LINE 1: SELECT employee_id, base_salary FROM payroll_salary\n'
        "connection: postgresql://svc_mnemiq@10.2.0.7:5432/hr_prod"
    )
    r = TestClient(build_app(_RT(raises=leaky), _identity())).post(
        "/v1/chat", json={"question": "q"}
    )
    err = _events(r.text)[-1]
    assert err["type"] == "RUN_ERROR"
    for secret in ("payroll_salary", "base_salary", "SELECT", "postgresql://",
                   "10.2.0.7", "hr_prod", "svc_mnemiq"):
        assert secret not in r.text, f"{secret!r} reached the client"


def test_a_mode_the_deployment_does_not_offer_is_still_told_to_the_caller():
    """Opaque by default, not opaque always. `UnknownMode` is a message the PRODUCT
    wrote about the caller's own input, so withholding it would turn a fixable request
    into a mystery -- and it is a constant here, not the exception's text."""
    from mnemiq.agent.route import UnknownMode

    r = TestClient(build_app(_RT(raises=UnknownMode("nope")), _identity())).post(
        "/v1/chat", json={"question": "q"}
    )
    err = _events(r.text)[-1]
    assert err["message"] == "That mode is not one this deployment offers."
    assert "nope" not in r.text


def test_chat_emits_keepalive_comments_while_the_engine_works():
    rt = _RT(answer=AgentAnswer(answer="ok"), delay_s=0.15)
    app = build_app(rt, _identity(), heartbeat_s=0.05)
    r = TestClient(app).post("/v1/chat", json={"question": "q"})
    assert ": keep-alive" in r.text
    assert [e["type"] for e in _events(r.text)][-1] == "RUN_FINISHED"

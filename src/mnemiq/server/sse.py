"""AG-UI-vocabulary SSE framing.

The engine's progress reaches this generator through `server/pump.py`'s `EnginePump`: the worker thread, the
queue, the heartbeat and the completion sentinel. This module frames what it yields.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from mnemiq.agent.route import UnknownMode
from mnemiq.profiles import UnknownProfile
from mnemiq.server.pump import HEARTBEAT, EnginePump
from mnemiq.server.serialize import answer_payload, result_payload
from mnemiq.server.superset_refusals import superset_refusal


def frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def step_frame(event: dict) -> dict:
    """One engine progress event -> one AG-UI STEP_STARTED / STEP_FINISHED frame.

    `stepId` rides on both halves. Correlating a finish to a start by name alone breaks
    as soon as two steps of the same name overlap, which is exactly what deep mode's
    candidates would do if they were ever run concurrently.
    """
    common = {
        "stepName": event.get("stage"),
        "stepId": event.get("id"),
        **{k: v for k, v in event.items() if k not in {"phase", "stage", "id", "ms", "ok"}},
    }
    if event.get("phase") == "started":
        return {"type": "STEP_STARTED", **common}
    return {"type": "STEP_FINISHED", "durationMs": event.get("ms"), "ok": event.get("ok"),
            **common}


logger = logging.getLogger(__name__)

# What a caller may be told when the run fails, keyed by exception TYPE.
#
# Every value is a constant. Nothing is interpolated from the exception, because the
# thing being kept off the wire is its text. What reaches this handler is whatever the
# agent did NOT convert to an answer -- a store error from `retrieve` naming the DuckDB
# path, an authz provider failing open on a file it can name, an embedder or provider
# transport error carrying an endpoint and its request headers. Those strings are made
# of deployment configuration, and `str(exc)` puts them in front of whoever asked a
# question. The engine refusing to answer must not become a way to read the engine.
#
# Note this is the narrower half. A source rejection never arrives here: `ExecutionError`
# is caught in the agent loop and turned into a failed `AgentAnswer`, so it leaves through
# the answer path, which `/v1/ask` shares -- see the fall-through in `_answer_single`.
#
# The default is deliberately uninformative TO THE CALLER and fully informative to the
# operator: the traceback is logged against the run id, which the client already has
# and can quote. An opaque message with a correlation handle costs a support round
# trip; a leaked one cannot be taken back.
_CALLER_FACING: dict[type[BaseException], str] = {
    UnknownMode: "That mode is not one this deployment offers.",
    UnknownProfile: "That profile is not one this deployment offers you.",
}
_OPAQUE = "The engine could not complete this request."


async def chat_stream(runtime, identity, question: str, mode: str | None,
                      heartbeat_s: float = 15.0, history=None, profile: str | None = None, delegation=None):
    """The run's frames. `delegation`, in Superset mode, is the token the door verified, handed to the runtime with the
    question; the pump runs the ask on a worker thread, where the Superset client may start an event loop of its own.
    """
    run_id, msg_id = uuid4().hex, uuid4().hex
    yield frame({"type": "RUN_STARTED", "runId": run_id})

    handed = {} if delegation is None else {"delegation": delegation}
    pump = EnginePump(
        lambda emit: runtime.ask(question, identity, mode=mode, emit=emit, history=history, profile=profile,
                                 **handed),
        heartbeat_s,
    )
    async for item in pump.events():
        if item is HEARTBEAT:  # the engine is just slow: keep the proxy path open
            yield ": keep-alive\n\n"
        elif item.get("kind") == "result":  # the verified rows, ahead of the prose
            yield frame({"type": "CUSTOM", "name": "mnemiq.result", "value": result_payload(item)})
        else:
            yield frame(step_frame(item))

    try:
        ans = pump.result()
    except Exception as exc:  # engine failure -> in-band error, stream ends
        # Logged with the run id BEFORE anything is yielded, so a failure is diagnosable even if
        # the client hangs up on the next frame. A refusal from Superset is a stated outcome, said
        # with the sentence the same refusal gets on /v1/ask; anything else keeps its traceback.
        refused = superset_refusal(exc)
        if refused is not None:
            logger.info("chat run %s ended as %d for %s: %s", run_id, refused.status, type(exc).__name__,
                        refused.detail)
            message = refused.detail
        else:
            logger.exception("chat run %s failed", run_id)
            message = _CALLER_FACING.get(type(exc), _OPAQUE)
        yield frame({"type": "RUN_ERROR", "message": message, "runId": run_id})
        return

    yield frame({"type": "TEXT_MESSAGE_START", "messageId": msg_id, "role": "assistant"})
    yield frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": msg_id, "delta": ans.answer})
    yield frame({"type": "TEXT_MESSAGE_END", "messageId": msg_id})
    yield frame({"type": "CUSTOM", "name": "mnemiq.answer", "value": answer_payload(ans)})
    yield frame({"type": "RUN_FINISHED", "runId": run_id})

"""AG-UI-vocabulary SSE framing.

The engine is synchronous and runs on a worker thread, so progress reaches this
generator through one asyncio.Queue per request. The engine's `emit` callback is a
plain sync function whose whole body schedules a thread-safe put; the queue is then
the single ordering point for everything that reaches the client, and the only thing
the HTTP layer has to know about.

The engine's completion is itself queued, as a sentinel pushed by a done-callback.
That keeps ordering honest -- every step already emitted sits ahead of it in the FIFO
-- and removes any race between "is the future done" and "is the queue drained".
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from mnemiq.server.serialize import answer_payload

_DONE = object()


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


async def chat_stream(runtime, identity, question: str, mode: str | None,
                      heartbeat_s: float = 15.0):
    run_id, msg_id = uuid4().hex, uuid4().hex
    yield frame({"type": "RUN_STARTED", "runId": run_id})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: dict) -> None:
        # Called from the engine's worker thread -- never touch the queue directly.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    fut = loop.run_in_executor(
        None, lambda: runtime.ask(question, identity, mode=mode, emit=emit)
    )
    fut.add_done_callback(lambda _: queue.put_nowait(_DONE))

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
        except TimeoutError:  # the engine is just slow: keep the proxy path open
            yield ": keep-alive\n\n"
            continue
        if item is _DONE:
            break
        yield frame(step_frame(item))

    try:
        ans = fut.result()
    except Exception as exc:  # engine failure -> in-band error, stream ends
        yield frame({"type": "RUN_ERROR", "message": str(exc), "runId": run_id})
        return

    yield frame({"type": "TEXT_MESSAGE_START", "messageId": msg_id, "role": "assistant"})
    yield frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": msg_id, "delta": ans.answer})
    yield frame({"type": "TEXT_MESSAGE_END", "messageId": msg_id})
    yield frame({"type": "CUSTOM", "name": "mnemiq.answer", "value": answer_payload(ans)})
    yield frame({"type": "RUN_FINISHED", "runId": run_id})

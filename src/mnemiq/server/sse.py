"""AG-UI-vocabulary SSE framing. One coarse pass today (the engine returns whole answers);
if the loop later emits live progress, the same event types stream incrementally.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from mnemiq.server.serialize import answer_payload


def frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def chat_stream(runtime, identity, question: str, mode: str | None,
                      heartbeat_s: float = 15.0):
    run_id, msg_id = uuid4().hex, uuid4().hex
    yield frame({"type": "RUN_STARTED", "runId": run_id})
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, lambda: runtime.ask(question, identity, mode=mode))
    while True:
        try:
            ans = await asyncio.wait_for(asyncio.shield(fut), timeout=heartbeat_s)
            break
        except TimeoutError:  # the engine is just slow: keep the proxy path open, keep waiting
            yield ": keep-alive\n\n"
        except Exception as exc:  # engine failure -> in-band error, stream ends
            yield frame({"type": "RUN_ERROR", "message": str(exc), "runId": run_id})
            return
    yield frame({"type": "TEXT_MESSAGE_START", "messageId": msg_id, "role": "assistant"})
    yield frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": msg_id, "delta": ans.answer})
    yield frame({"type": "TEXT_MESSAGE_END", "messageId": msg_id})
    yield frame({"type": "CUSTOM", "name": "mnemiq.answer", "value": answer_payload(ans)})
    yield frame({"type": "RUN_FINISHED", "runId": run_id})

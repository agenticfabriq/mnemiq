"""The engine's progress, carried from its worker thread to an async stream. Shared by every stream the server
sends, so a fix to it lands once.

The engine is synchronous and runs on a worker thread, so progress reaches the stream through one asyncio.Queue
per request. The engine's `emit` callback is a plain sync function whose whole body schedules a thread-safe put;
the queue is then the single ordering point for everything that reaches the client, and the only thing the HTTP
layer has to know about.

The engine's completion is itself queued, as a sentinel pushed by a done-callback. That keeps ordering honest --
every step already emitted sits ahead of it in the FIFO -- and removes any race between "is the future done" and
"is the queue drained".

Nothing here cancels the worker. A stream that stops reading closes the iterator, and the engine runs to its end,
its later emits landing on a queue nobody reads.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar

T = TypeVar("T")

HEARTBEAT = object()  # yielded after each `heartbeat_s` with nothing from the engine
_DONE = object()


class EnginePump(Generic[T]):
    """`run(emit)` on a worker thread, its emitted events as an async iterator, and its result."""

    def __init__(self, run: Callable[[Callable[[dict], None]], T], heartbeat_s: float) -> None:
        self._run, self._heartbeat_s = run, heartbeat_s
        self._future: asyncio.Future | None = None

    async def events(self) -> AsyncIterator[dict | object]:
        """Starts the worker on first iteration, so a stream's opening frame goes out before the engine starts.
        Yields each emitted event in order, `HEARTBEAT` after each quiet `heartbeat_s`, and ends when the run does."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:
            # Called from the engine's worker thread -- never touch the queue directly.
            loop.call_soon_threadsafe(queue.put_nowait, event)

        self._future = loop.run_in_executor(None, lambda: self._run(emit))
        self._future.add_done_callback(lambda _: queue.put_nowait(_DONE))

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=self._heartbeat_s)
            except TimeoutError:  # the engine is just slow
                yield HEARTBEAT
                continue
            if item is _DONE:
                return
            yield item

    def result(self) -> T:
        """The run's return value, or its exception raised. Read once `events()` has ended."""
        if self._future is None:
            raise RuntimeError("the pump has not run: iterate events() first")
        return self._future.result()

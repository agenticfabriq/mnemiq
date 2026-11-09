"""The engine's progress pump: the worker thread, the queue, the heartbeat and the done sentinel that the AG-UI
stream runs on, and that a second stream can share."""

import logging
import threading
import time

import anyio
import pytest

from mnemiq.server.pump import HEARTBEAT, EnginePump


async def _drain(pump):
    with anyio.fail_after(5):  # a pump that never reaches its done sentinel fails here instead of hanging
        return [item async for item in pump.events()]


def test_events_arrive_in_order_and_end_at_done():
    def run(emit):
        for i in range(50):
            emit({"i": i})
        return "answer"

    pump = EnginePump(run, heartbeat_s=5)
    assert anyio.run(_drain, pump) == [{"i": i} for i in range(50)]
    assert pump.result() == "answer"


def test_a_quiet_engine_gets_a_heartbeat_per_interval():
    def run(emit):
        time.sleep(0.15)
        emit({"stage": "late"})

    items = anyio.run(_drain, EnginePump(run, heartbeat_s=0.02))
    late = items.index({"stage": "late"})
    assert sum(item is HEARTBEAT for item in items[:late]) >= 3
    assert not any(item is HEARTBEAT for item in items[late + 1:])


def test_the_worker_starts_on_first_iteration():
    """A stream sends its opening frame before the engine starts, so constructing the pump must start nothing."""
    calls = []

    async def scenario():
        pump = EnginePump(lambda emit: calls.append(1), heartbeat_s=5)
        await anyio.sleep(0.05)
        assert calls == []
        await _drain(pump)
        assert calls == [1]

    anyio.run(scenario)


def test_a_worker_exception_reaches_the_caller_through_result():
    class Boom(Exception):
        pass

    def run(emit):
        emit({"stage": "a"})
        raise Boom("the engine failed")

    pump = EnginePump(run, heartbeat_s=5)
    assert anyio.run(_drain, pump) == [{"stage": "a"}]
    with pytest.raises(Boom):
        pump.result()


def test_a_hang_up_closes_cleanly_and_the_worker_finishes(caplog):
    """Nothing cancels the worker: the engine is synchronous on its own thread. A stream that stops reading closes
    without an error, and the engine runs to its end, its later emits landing on a queue nobody reads."""
    release, finished, emit_errors = threading.Event(), threading.Event(), []

    def run(emit):
        emit({"stage": "first"})
        release.wait(5)
        try:
            emit({"stage": "after the hang-up"})
        except Exception as exc:  # noqa: BLE001 -- the assertion below is that there is none
            emit_errors.append(exc)
        finished.set()
        return "answer"

    async def scenario():
        events = EnginePump(run, heartbeat_s=5).events()
        assert await events.__anext__() == {"stage": "first"}
        await events.aclose()
        release.set()
        await anyio.to_thread.run_sync(finished.wait, 5)

    with caplog.at_level(logging.DEBUG):
        anyio.run(scenario)
    assert finished.is_set() and emit_errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_pump_runs_once():
    """A second iteration would start a second worker and orphan the first run's answer. `events()` is an async
    generator, so its body, and the refusal, run only when it is iterated."""
    calls = []

    def run(emit):
        calls.append(1)
        return "first"

    async def scenario():
        pump = EnginePump(run, heartbeat_s=5)
        await _drain(pump)
        with pytest.raises(RuntimeError, match="runs once"):
            await _drain(pump)
        return pump

    pump = anyio.run(scenario)
    assert calls == [1] and pump.result() == "first"


def test_result_before_the_run_ends_says_so():
    """After a hang-up the worker is still running: `result()` says the run has not ended, rather than asyncio's
    InvalidStateError. After a full drain it returns, since the done sentinel follows the run's end."""
    release = threading.Event()

    def run(emit):
        emit({"stage": "first"})
        release.wait(5)
        return "answer"

    async def scenario():
        pump = EnginePump(run, heartbeat_s=5)
        events = pump.events()
        await events.__anext__()
        await events.aclose()
        with pytest.raises(RuntimeError, match="has not ended"):
            pump.result()
        release.set()

    anyio.run(scenario)

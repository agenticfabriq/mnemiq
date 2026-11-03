"""A decision's deadline (M227).

sqlglot 30.12's DuckDB parser never returns on a struct literal holding a bare `NOT`, and a
decision parses the model's SQL, each row filter and texts it derives from both. A parse that never
returns held the deciding thread at full CPU for as long as the process lived. Python cannot stop a
thread, so the deadline is raised INTO it: a timer, past the deadline, asks the interpreter to raise
`DecisionTimeout` in the deciding thread at its next bytecode (CPython's `PyThreadState_SetAsyncExc`),
and the decision unwinds out of the parser. sqlglot runs as pure Python here, which is what makes
that possible; a loop inside native code would not see it.

Time spent waiting on the source does not count: a source call has its own timeout, and a busy
source would otherwise refuse statements the decider had no trouble with. The adapter a decision
is given is wrapped so the clock stops while any of its methods runs (`paused_source`).
"""

from __future__ import annotations

import ctypes
import functools
import inspect
import logging
import threading
import time
from contextlib import contextmanager

from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)

# The decider's own work, source time excluded: a decision takes milliseconds (3.5 ms median measured
# on three ordinary statements), so ten seconds refuses only a decision that is not going to end.
DECISION_DEADLINE_S = 10.0


class DecisionTimeout(BaseException):
    """Raised into a deciding thread past its deadline. Not an Exception: the deciders refuse on any
    Exception a check raises, and a probe that raises reads as a place a name does not fit (M222),
    so an Exception landing mid-check could approve what the check never finished deciding."""


_active = threading.local()


def _raise_in(thread_id: int, exc: type[BaseException] | None) -> None:
    modified = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id),
                                                           ctypes.py_object(exc) if exc else None)
    if modified > 1:  # the documented recovery: undo, since more than one thread was hit
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)


class _Clock:
    """One thread's deadline: armed, paused while the source runs, resumed, and disarmed."""

    def __init__(self, seconds: float) -> None:
        self.thread_id = threading.get_ident()
        self.remaining = seconds
        self.lock = threading.Lock()
        self.live = True
        self.fired = False
        self.timer: threading.Timer | None = None
        self.started = 0.0

    def _fire(self) -> None:
        with self.lock:
            if self.live and self.timer is not None:
                self.fired = True
                _raise_in(self.thread_id, DecisionTimeout)

    def start(self) -> None:
        with self.lock:
            self.started = time.monotonic()
            self.timer = threading.Timer(max(self.remaining, 0.0), self._fire)
            self.timer.daemon = True
            self.timer.start()

    def stop(self) -> None:
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
                self.remaining -= time.monotonic() - self.started

    def disarm(self) -> None:
        with self.lock:
            self.live = False
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            if self.fired:
                # Fired as the block ended: the timeout may be pending, not yet raised. Cleared here,
                # so it never lands in whatever the thread does next.
                _raise_in(self.thread_id, None)


@contextmanager
def within(seconds: float):
    """Run the block under a deadline of `seconds` of this thread's own work; past it,
    `DecisionTimeout` is raised in this thread. Nested, the outer deadline holds and the inner one
    is a no-op: an inner one disarming would clear the outer's pending timeout."""
    if getattr(_active, "clock", None) is not None:
        yield
        return
    clock = _Clock(seconds)
    _active.clock = clock
    clock.start()
    try:
        yield
    finally:
        try:
            clock.disarm()
        finally:
            _active.clock = None


@contextmanager
def paused():
    """Stop this thread's deadline for the block, a call to the source; a no-op outside one."""
    clock = getattr(_active, "clock", None)
    if clock is None or not clock.live:
        yield
        return
    clock.stop()
    try:
        yield
    finally:
        clock.start()


class _PausedSource:
    """The source a decision is given, with the deadline stopped while any method of it runs. Every
    other attribute reads through, so `hasattr` and `getattr` see the source's own."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        found = getattr(self._inner, name)
        if not callable(found):
            return found

        @functools.wraps(found)
        def call(*args, **kwargs):
            with paused():
                return found(*args, **kwargs)
        return call


def paused_source(adapter):
    """`adapter` wrapped so its calls do not count against the deadline; None stays None."""
    return None if adapter is None else _PausedSource(adapter)


def bounded(decider):
    """`decider` run under `DECISION_DEADLINE_S`, read at each call, with its adapter's calls paused;
    past the deadline, a refusal. Every door decides through `decide` or `decide_write`, so each is
    bounded here rather than at each door."""
    signature = inspect.signature(decider)

    @functools.wraps(decider)
    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if bound.arguments.get("adapter") is not None:
            bound.arguments["adapter"] = paused_source(bound.arguments["adapter"])
        seconds = DECISION_DEADLINE_S
        try:
            with within(seconds):
                return decider(*bound.args, **bound.kwargs)
        except DecisionTimeout:
            logger.warning("a decision ran past its %.1f s deadline and was refused (M227)", seconds)
            return Refusal(
                code=RefusalCode.DECISION_TIMEOUT,
                message=("This statement took longer to check than this engine allows, so it was "
                         "not run. Ask a simpler question."),
            )

    return run

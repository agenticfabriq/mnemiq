"""The progress seam: how work deep inside the agent reports what it is doing.

`Emit` is a plain synchronous one-argument callback threaded down the call chain and
defaulting to None. It is deliberately not a contextvar (which does not survive the
thread-pool hop the HTTP server makes) and not a global bus (which would force every
caller to invent request correlation).

An emit that raises must never break an answer -- progress reporting is not part of
the engine's contract with the caller, and a UI concern cannot be allowed to fail a
query. Every call is guarded.

Every step carries an id. Without one, a consumer has to correlate a finish with a
start by name, which breaks the moment two steps of the same name overlap -- which is
exactly what deep mode's candidates would do if they ever run concurrently.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from uuid import uuid4

Emit = Callable[[dict], None]


class Stage(StrEnum):
    """The phases an answer passes through, named for what the reader would call them."""

    RETRIEVE = "retrieve"  # find the tables this question could be answered from
    PLAN = "plan"  # write the SQL, decide it, repair what the decider rejects
    CANDIDATE = "candidate"  # one of N independently generated attempts (deep mode)
    EXECUTE = "execute"  # run it against the source
    VERIFY = "verify"  # the result-verifier's gate
    SYNTHESIZE = "synthesize"  # turn the result into an answer


def _guarded(emit: Emit, event: dict) -> None:
    try:
        emit(event)
    except Exception:  # noqa: BLE001 -- a broken listener must not fail the query
        pass


@contextmanager
def step(emit: Emit | None, stage: Stage, **fields: object) -> Iterator[str | None]:
    """Bracket one stage: a started event, then a finished event with its duration.

    `ok` is False when the body raised, so a consumer can tell a stage that failed from
    one that merely took a while -- the engine retries around several of these.
    """
    if emit is None:
        yield None
        return

    step_id = uuid4().hex[:12]
    _guarded(emit, {"phase": "started", "stage": str(stage), "id": step_id, **fields})
    started = time.perf_counter()
    ok = True
    try:
        yield step_id
    except BaseException:
        ok = False
        raise
    finally:
        _guarded(
            emit,
            {
                "phase": "finished",
                "stage": str(stage),
                "id": step_id,
                "ms": (time.perf_counter() - started) * 1000,
                "ok": ok,
                **fields,
            },
        )

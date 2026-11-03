"""A decision ends within its deadline, whatever its parser does (M227).

sqlglot 30.12's DuckDB parser never returns on a struct literal holding a bare `NOT`:
`SELECT struct(id, not) FROM claim` held the thread that decided it, at full CPU, for as long as the
process lived. The SQL a door decides is the model's, which a caller can ask it to write, and a row
filter is parsed the same way. Every decision now runs under a deadline: past it a timeout is raised
into the deciding thread, the decision unwinds, and the door refuses. Time the decision spends
waiting on its source does not count, since a source call has its own timeout.

Each case also runs under an alarm, so a deadline that fails fails its test rather than hanging the run.
"""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager

import pytest

import mnemiq.sql.deadline as deadline
from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

SCHEMA = {"claim": {"id", "region"}, "scratch": {"id", "region"}}
LOOPED = ("struct(id, not)", "struct(id, id, not)", "struct(id, not, id)")
SAFETY_S = 15


class _Hung(BaseException):
    pass


@contextmanager
def _safety_net():
    """An alarm past which the test fails rather than spinning on: the deadline under test is
    what should end the decision long before it."""
    def _hung(signum, frame):
        raise _Hung
    previous = signal.signal(signal.SIGALRM, _hung)
    signal.alarm(SAFETY_S)
    try:
        yield
    except _Hung:
        pytest.fail(f"the decision did not end within {SAFETY_S} s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(autouse=True)
def _short_deadline(monkeypatch):
    monkeypatch.setattr(deadline, "DECISION_DEADLINE_S", 1.0)


def _timed_out(verdict):
    return isinstance(verdict, Refusal) and verdict.code is RefusalCode.DECISION_TIMEOUT


@pytest.mark.parametrize("shape", LOOPED)
def test_a_read_whose_parse_never_returns_is_refused_at_the_deadline(shape):
    started = time.monotonic()
    with _safety_net():
        verdict = decide(f"SELECT {shape} FROM claim", SCHEMA, dialect="duckdb", target="duckdb")
    assert _timed_out(verdict), verdict
    assert time.monotonic() - started < 5, "refused, but long after the deadline"


def test_a_write_whose_parse_never_returns_is_refused_at_the_deadline():
    grants = GrantSet(frozenset(SCHEMA), writable=frozenset(SCHEMA))
    with _safety_net():
        verdict = decide_write("INSERT INTO scratch (id, region) SELECT struct(id, not), region FROM claim",
                               SCHEMA, grants, dialect="duckdb", target="duckdb",
                               policy=AccessPolicy(policy_schema=SCHEMA), writes_enabled=True)
    assert _timed_out(verdict), verdict


def test_a_row_filter_whose_parse_never_returns_is_refused_at_the_deadline():
    policy = AccessPolicy(row_filters={"claim": "struct(id, not) IS NOT NULL"}, policy_schema=SCHEMA)
    with _safety_net():
        verdict = decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target="duckdb", policy=policy)
    assert _timed_out(verdict), verdict


def test_off_the_main_thread_as_a_door_runs_it_the_thread_is_freed():
    # The server decides in a thread pool, where no signal reaches: the deadline is per thread.
    out = []
    worker = threading.Thread(target=lambda: out.append(
        decide("SELECT struct(id, not) FROM claim", SCHEMA, dialect="duckdb", target="duckdb")), daemon=True)
    worker.start()
    worker.join(SAFETY_S)
    assert not worker.is_alive(), "the deciding thread is still running"
    assert out and _timed_out(out[0]), out


def test_the_thread_decides_normally_after_a_timeout():
    with _safety_net():
        assert _timed_out(decide("SELECT struct(id, not) FROM claim", SCHEMA, dialect="duckdb", target="duckdb"))
        assert isinstance(decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target="duckdb"), Approved)


def test_a_decision_that_ends_in_time_leaves_no_timeout_behind(monkeypatch):
    # Cancelled and cleared on the way out: a timeout fired as a decision returned must not land
    # in whatever the thread does next.
    monkeypatch.setattr(deadline, "DECISION_DEADLINE_S", 0.004)
    with _safety_net():
        for _ in range(300):
            decide("SELECT id FROM claim WHERE region = 'west'", SCHEMA, dialect="duckdb", target="duckdb")
        ends = time.monotonic() + 0.3
        while time.monotonic() < ends:
            sum(range(1000))  # Python work, where a stray timeout would land


def test_a_nested_deadline_does_not_clear_the_outer_one():
    with _safety_net():
        with pytest.raises(deadline.DecisionTimeout):
            with deadline.within(0.3):
                with deadline.within(10):
                    pass
                ends = time.monotonic() + 3
                while time.monotonic() < ends:
                    sum(range(1000))


class _SlowSource:
    """A source whose EXPLAIN takes longer than the deadline, as a busy one may."""
    def __init__(self, seconds):
        self.seconds = seconds

    def execute(self, sql):
        time.sleep(self.seconds)
        return []


def test_time_waiting_on_the_source_does_not_count():
    with _safety_net():
        verdict = decide("SELECT id FROM claim", SCHEMA, adapter=_SlowSource(1.5), dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Approved), verdict


def test_a_write_s_source_time_does_not_count_either():
    grants = GrantSet(frozenset(SCHEMA), writable=frozenset(SCHEMA))
    with _safety_net():
        verdict = decide_write("DELETE FROM scratch WHERE id = 1", SCHEMA, grants, adapter=_SlowSource(1.5),
                               dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=SCHEMA),
                               writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite), verdict


def test_the_refusal_is_a_dead_end_not_a_repair():
    # A retry is the model's next statement, which may loop the same way: refused once, not N times.
    from mnemiq.sql.verdict import REPAIRABLE
    assert RefusalCode.DECISION_TIMEOUT not in REPAIRABLE


def _spin(seconds):
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        sum(range(1000))


def test_the_deadline_counts_the_work_on_both_sides_of_a_source_call():
    # Paused, not restarted: 0.35 s of work, a source call, and 0.35 s more is past a 0.5 s deadline.
    with _safety_net():
        with pytest.raises(deadline.DecisionTimeout):
            with deadline.within(0.5):
                _spin(0.35)
                with deadline.paused():
                    time.sleep(0.6)
                _spin(0.35)
        with deadline.within(0.5):  # the control: the same with no work after the call ends in time
            _spin(0.35)
            with deadline.paused():
                time.sleep(0.6)


def test_after_a_nested_deadline_the_outer_one_still_pauses_for_the_source():
    # A nested deadline arming its own clock left the thread with none once it ended, so the outer
    # decision's later source calls counted against it.
    with _safety_net():
        with deadline.within(0.5):
            with deadline.within(10):
                pass
            with deadline.paused():
                time.sleep(0.8)


def test_a_timeout_fired_as_the_decision_ends_is_cleared_before_it_lands():
    # Deterministic, where a race is not: the deciding thread waits in a C call, where a pending
    # timeout cannot land; the timer fires and the way out runs meanwhile; then the thread resumes.
    out = {}
    armed, go = threading.Event(), threading.Event()

    def deciding():
        clock = deadline._Clock(60)
        clock.start()
        out["clock"] = clock
        armed.set()
        try:
            go.wait()
            sum(range(1000))  # bytecode, where a pending timeout would be raised
            out["landed"] = False
        except deadline.DecisionTimeout:
            out["landed"] = True

    thread = threading.Thread(target=deciding, daemon=True)
    thread.start()
    assert armed.wait(5)
    out["clock"]._fire()   # the timer, firing as the decision ends
    out["clock"].disarm()  # the decision's way out, before the thread runs again
    go.set()
    thread.join(5)
    assert out.get("landed") is False, out


class _SlowIndex:
    """A value index whose reads take longer than the deadline, as DuckDB's may on a busy store."""
    def __init__(self, seconds):
        self.seconds, self.asked = seconds, 0

    def has(self, table, column):
        time.sleep(self.seconds)
        self.asked += 1
        return True

    def contains(self, table, column, literal):
        time.sleep(self.seconds)
        return True

    def nearest(self, *args, **kwargs):
        time.sleep(self.seconds)
        return []


def test_time_waiting_on_the_value_index_does_not_count():
    index = _SlowIndex(1.2)
    with _safety_net():
        verdict = decide("SELECT id FROM claim WHERE region = 'west'", SCHEMA, values=index,
                         dialect="duckdb", target="duckdb")
    assert index.asked, "the value index was never asked, so the test shows nothing"
    assert isinstance(verdict, Approved), verdict

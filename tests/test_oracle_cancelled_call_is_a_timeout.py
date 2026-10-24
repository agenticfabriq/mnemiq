"""A cancelled Oracle call is a timeout, not the statement's failure (M187): `_is_timeout`
reads the driver's error, and ORA-01013, the spelling a call timeout takes in thick mode
and under a server-side cancel by the driver's own error list, was not among the spellings it
matched, so it reached the runner as a plain source fault while DPY-4024 (thin mode) and
ORA-03156 were reported as "query timed out". ORA-01013 is the cancel itself, which anyone
holding the connection may raise, so it is a timeout only once the deadline has passed; the
code is read off the driver's error object where it carries one, so an application error
quoting a timeout code in its message is none (Codex's first review of #176). No database
here: the matcher reads the error object, or the text of one without a code."""

from __future__ import annotations

import time

import pytest

oracledb = pytest.importorskip("oracledb")

from oracledb import errors as oracledb_errors  # noqa: E402

from mnemiq.adapters.base import QueryCancelled  # noqa: E402
from mnemiq.adapters.oracle import OracleAdapter, _is_timeout  # noqa: E402


def _driver_error(message: str, code: int, kind=None):
    """An error as the driver raises it: `args[0]` the driver's own object carrying the code."""
    return (kind or oracledb.DatabaseError)(oracledb_errors._Error(message=message, code=code))


@pytest.mark.parametrize("text", [
    "ORA-01013: user requested cancel of current operation",
    "DPY-4024: call timeout of 300 ms exceeded",
    "DPY-4024: reworded by a later driver",   # the code, not the wording (the gate)
    "ORA-03156: the statement timed out",   # the code is what the matcher reads
    "DPY-4011: the database or network closed the connection",
])
def test_a_bounded_calls_end_is_a_timeout_under_each_spelling(text):
    assert _is_timeout(oracledb.DatabaseError(text)), text   # text alone: an error with no code
    code = text.split(":")[0]
    assert _is_timeout(_driver_error(text, int(code.split("-")[1]))), text


@pytest.mark.parametrize("text", [
    "ORA-00942: table or view does not exist",
    "ORA-01017: invalid username/password; logon denied",
    "DPY-4033: the database closed the connection because the session was idle",
])
def test_a_fault_that_is_no_timeout_is_not_read_as_one(text):
    assert not _is_timeout(oracledb.DatabaseError(text)), text
    assert not _is_timeout(_driver_error(text, int(text.split(":")[0].split("-")[1]))), text


def test_the_cancel_is_a_timeout_only_once_the_deadline_has_passed():
    # Codex's first review of #176: an explicit cancel on a connection the caller owns raises
    # ORA-01013 at once, and read as a timeout its own detail was replaced by "query timed out".
    cancel = _driver_error("ORA-01013: user requested cancel of current operation", 1013, oracledb.OperationalError)
    assert _is_timeout(cancel, deadline_met=True)
    assert not _is_timeout(cancel, deadline_met=False)
    assert not _is_timeout(oracledb.DatabaseError("ORA-01013: user requested cancel of current operation"), deadline_met=False)
    # A closed connection reads as the cancel does (M188): the bound's end at the deadline, and the
    # source's before it, an outage the runner reports through the adapter's own classifier.
    closed = _driver_error("DPY-4011: the database or network closed the connection", 4011)
    assert _is_timeout(closed, deadline_met=True)
    assert not _is_timeout(closed, deadline_met=False)
    assert not _is_timeout(oracledb.DatabaseError("DPY-4011: the database or network closed the connection"), deadline_met=False)
    # the driver's own timeout codes do not depend on the clock
    assert _is_timeout(_driver_error("DPY-4024: call timeout of 300 ms exceeded", 4024), deadline_met=False)
    assert _is_timeout(_driver_error("ORA-03156: the statement timed out", 3156), deadline_met=False)


def test_the_code_is_the_errors_own_not_one_its_message_quotes():
    # The same review: an application error whose message mentions ORA-01013, and a parse
    # error naming a column TIMEOUT, were read as timeouts by their text; the error's own code
    # decides where it carries one, the text only where it does not.
    assert not _is_timeout(_driver_error("ORA-20001: raised after ORA-01013 was seen", 20001))
    assert not _is_timeout(_driver_error('ORA-00904: "TIMEOUT": invalid identifier', 904))
    assert _is_timeout(oracledb.DatabaseError("a wrapped error whose text says timeout"))   # no code: text


class _Cursor:
    """A cursor whose execute raises, on a connection that takes a call timeout."""

    def __init__(self, raising):
        self.connection = type("Connection", (), {"call_timeout": 0})()
        self._raising = raising

    def execute(self, sql):
        self._raising()


def _bare_adapter(raising):
    """`execute_arrow` alone, over a lease that waits (one tick of the clock) and a cursor
    that raises: no connection, no database."""
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._oracledb = oracledb

    def with_cursor(work, timeout_ms=None):
        time.monotonic()  # the lease's lock wait and setup round trip, before the call
        return work(_Cursor(raising))

    adapter._refuse_unless_read = lambda sql: None
    adapter._with_cursor = with_cursor
    return adapter


@pytest.mark.parametrize("elapsed, timed_out", [(0.01, False), (4.9985, False), (4.9992, True), (5.0, True), (7.5, True)])
def test_execute_arrow_reads_the_cancel_against_its_own_deadline(monkeypatch, elapsed, timed_out):
    # The join: `execute_arrow` measures the call against `timeout_s`, from right before the
    # execute (the gate on the second commit: a clock started before the lease counted its
    # lock wait, so a cancel queued behind a long wait read as the deadline), and hands the
    # answer to the matcher, so a cancel before the deadline reaches the caller as the cancel
    # it is.
    def raising():
        raise _driver_error("ORA-01013: user requested cancel of current operation", 1013, oracledb.OperationalError)
    clock = iter([0.0, 100.0, 100.0 + elapsed])   # the lease's wait, the call's start, its end
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    adapter = _bare_adapter(raising)
    # the deadline is the driver's, armed in whole milliseconds: 4.9995 s arms 4999 ms (the gate)
    if timed_out:
        with pytest.raises(RuntimeError, match="query timed out after 4.9995"):
            adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=4.9995)
    else:
        # Before the deadline the cancel is anyone's: raised as the cancel it is, the driver's
        # ORA-01013 its cause, so the runner does not read it as an outage (the
        # review gate on M188).
        with pytest.raises(QueryCancelled, match="ORA-01013") as caught:
            adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=4.9995)
        assert isinstance(caught.value.__cause__, oracledb.OperationalError)


class _FetchingCursor:
    """A cursor whose execute answers in one second and whose fetch is several round trips, one
    row each, taking the given seconds; the last trip raises `error` when one is given. `fetchall`
    is the driver's own loop over the same trips."""

    description = [("N", None, None, None, None, None, None)]

    def __init__(self, clock, trips, error=None):
        self.connection = type("Connection", (), {"call_timeout": 0})()
        self._clock, self._trips, self._error = clock, list(trips), error

    def execute(self, sql):
        self._clock[0] += 1.0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._trips:
            raise StopIteration
        seconds = self._trips.pop(0)
        self._clock[0] += seconds
        if not self._trips and self._error is not None:
            raise self._error
        return (seconds,)

    def fetchall(self):
        return list(self)


def _fetching_adapter(monkeypatch, trips, error=None):
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._oracledb = oracledb
    adapter._refuse_unless_read = lambda sql: None
    adapter._with_cursor = lambda work, timeout_ms=None: work(_FetchingCursor(clock, trips, error))
    return adapter


_CANCEL = ("ORA-01013: user requested cancel of current operation", 1013, oracledb.OperationalError)
_CLOSED = ("DPY-4011: the database or network closed the connection", 4011, None)


@pytest.mark.parametrize("trips", [[3.0, 3.0, 3.0, 0.5], [4.5]], ids=["late", "first-fetch"])
def test_a_cancel_in_the_fetch_is_the_cancel_not_the_bounds_end(monkeypatch, trips):
    # Codex's review of #179 at f97f7749: `call_timeout` bounds each round trip, not the
    # statement, and the clock ran from the execute through the whole fetch. Three 3 s trips and
    # a cancel 0.5 s into the fourth, 10.5 s after the execute under a 5 s bound: no trip reached
    # the bound, so it is not the bound's end. Read as one it was repairable, and the model could
    # run the statement again. And the first fetch trip is its own too: 4.5 s after a 1 s
    # execute is 5.5 s from the execute, short of the bound from the trip's start.
    adapter = _fetching_adapter(monkeypatch, trips, _driver_error(*_CANCEL))
    with pytest.raises(QueryCancelled, match="ORA-01013"):
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)


def test_a_closed_connection_late_in_a_long_fetch_is_the_sources(monkeypatch):
    # The same trips ending in DPY-4011: the source dropped the connection, so the driver's error
    # reaches the runner as it is and the adapter's classifier reads it as the outage it is.
    adapter = _fetching_adapter(monkeypatch, [3.0, 3.0, 3.0, 0.5], _driver_error(*_CLOSED))
    with pytest.raises(oracledb.DatabaseError, match="DPY-4011") as caught:
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert adapter.source_unavailable(caught.value)


@pytest.mark.parametrize("fault", [_CANCEL, _CLOSED], ids=["cancel", "closed"])
def test_a_fetch_round_trip_that_reaches_the_bound_is_a_timeout(monkeypatch, fault):
    # The other side: one trip that itself lasts the bound is the bound's end, however short the
    # trips before it.
    adapter = _fetching_adapter(monkeypatch, [0.5, 0.5, 5.0], _driver_error(*fault))
    with pytest.raises(RuntimeError, match="query timed out after 5.0"):
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)


def test_a_long_fetch_with_no_fault_returns_every_row(monkeypatch):
    # The per-trip clock changes how the fetch is driven, not what it returns.
    adapter = _fetching_adapter(monkeypatch, [3.0, 3.0, 3.0, 0.5])
    table = adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert table.column_names == ["N"]
    assert table.column(0).to_pylist() == [3.0, 3.0, 3.0, 0.5]

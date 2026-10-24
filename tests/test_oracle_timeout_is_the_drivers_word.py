"""An Oracle call is a timeout only when the driver says so: DPY-4024, ORA-03156, or DPY-4011 with
the driver's own line saying its bound ended the call, never inferred from a clock. M187 matched
ORA-01013 as a timeout, and Codex's first review of #176 made it one only once the deadline had
passed, measured by the adapter. Codex's reviews of #179 showed the adapter cannot measure that:
`call_timeout` bounds each round trip, and the driver makes round trips of its own between rows (it
reads large XMLTYPE values while converting a whole batch), so a cancel late in a long fetch read
as the deadline, was repairable, and could run again. python-oracledb 4.0.2 reports its own bound
as DPY-4024 in thin mode, cancelling the call itself, and maps thick mode's DPI-1067 to the same
code; so ORA-01013 is always someone else's cancel (the adapter never cancels), and DPY-4011 a
closed connection, the source's. One DPY-4011 is the bound's: when the break the driver sends at
its bound goes unanswered within the same bound, it closes the connection and says so on the line
under the message, measured on Oracle Free 23 with a query that was only slow, on a server that was
answering. The code is read off the driver's error object where it carries one, so an application
error quoting a timeout code in its message is none (Codex's first review of #176). No database
here: the matcher reads the error object, or the text of one without a code."""

from __future__ import annotations

import itertools
import time

import pytest

oracledb = pytest.importorskip("oracledb")

from oracledb import errors as oracledb_errors  # noqa: E402

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut  # noqa: E402
from mnemiq.adapters.oracle import OracleAdapter, _is_timeout  # noqa: E402


def _driver_error(message: str, code: int, kind=None):
    """An error as the driver raises it: `args[0]` the driver's own object carrying the code."""
    return (kind or oracledb.DatabaseError)(oracledb_errors._Error(message=message, code=code))


@pytest.mark.parametrize("text", [
    "DPY-4024: call timeout of 300 ms exceeded",
    "DPY-4024: reworded by a later driver",   # the code, not the wording (the gate)
    "ORA-03156: the statement timed out",   # the code is what the matcher reads
])
def test_the_drivers_own_timeout_is_a_timeout_under_each_spelling(text):
    assert _is_timeout(oracledb.DatabaseError(text)), text   # text alone: an error with no code
    code = text.split(":")[0]
    assert _is_timeout(_driver_error(text, int(code.split("-")[1]))), text


@pytest.mark.parametrize("text", [
    "ORA-00942: table or view does not exist",
    "ORA-01017: invalid username/password; logon denied",
    "DPY-4033: the database closed the connection because the session was idle",
    "ORA-01013: user requested cancel of current operation",   # someone else's cancel
    "DPY-4011: the database or network closed the connection",   # the source's
])
def test_a_fault_that_is_no_timeout_is_not_read_as_one(text):
    assert not _is_timeout(oracledb.DatabaseError(text)), text
    assert not _is_timeout(_driver_error(text, int(text.split(":")[0].split("-")[1]))), text


# The driver's own lines under DPY-4011 (python-oracledb 4.0.2 thin, `_process_message`).
_RECOVERY_TIMED_OUT = "socket timed out while recovering from previous socket timeout"
_BREAK_UNANSWERED = "socket timed out while awaiting break response from server"
_CLOSED_TEXT = "DPY-4011: the database or network closed the connection"


def test_a_closed_connection_the_driver_says_its_bound_ended_is_a_timeout():
    # The bound ran out, the driver's break went unanswered within the same bound, and the driver
    # closed the connection, saying so on the line under DPY-4011. Measured on Oracle Free 23 with
    # a query that was only slow, so a server that answers can end this way: a timeout the model
    # may repair, not an outage that ends the turn.
    text = f"{_CLOSED_TEXT}\n{_RECOVERY_TIMED_OUT}"
    assert _is_timeout(_driver_error(text, 4011))
    assert _is_timeout(oracledb.DatabaseError(text))   # no code: the text says timeout


@pytest.mark.parametrize("text", [
    _CLOSED_TEXT,                                   # closed, with no reason given
    f"{_CLOSED_TEXT}\n{_BREAK_UNANSWERED}",          # someone else's break, unanswered
    f"{_CLOSED_TEXT}\nconnection reset by peer {_RECOVERY_TIMED_OUT}",   # not the driver's line
], ids=["no-reason", "a-cancel-unanswered", "quoted-in-another-line"])
def test_any_other_closed_connection_is_no_timeout(text):
    assert not _is_timeout(_driver_error(text, 4011))


def test_the_recovery_line_under_another_code_is_no_timeout():
    # An application error can carry any text; only the driver's DPY-4011 carries this line.
    assert not _is_timeout(_driver_error(f"ORA-20001: raised\n{_RECOVERY_TIMED_OUT}", 20001))


def test_the_code_is_the_errors_own_not_one_its_message_quotes():
    # Codex's first review of #176: an application error whose message mentions ORA-01013, and a
    # parse error naming a column TIMEOUT, were read as timeouts by their text; the error's own
    # code decides where it carries one, the text only where it does not.
    assert not _is_timeout(_driver_error("ORA-20001: raised after DPY-4024 was seen", 20001))
    assert not _is_timeout(_driver_error('ORA-00904: "TIMEOUT": invalid identifier', 904))
    assert _is_timeout(oracledb.DatabaseError("a wrapped error whose text says timeout"))   # no code: text


_CANCEL = ("ORA-01013: user requested cancel of current operation", 1013, oracledb.OperationalError)
_CLOSED = ("DPY-4011: the database or network closed the connection", 4011, None)
_TIMED_OUT = ("DPY-4024: call timeout of 5000 ms exceeded", 4024, None)
_CLOSED_AT_THE_BOUND = (f"{_CLOSED_TEXT}\n{_RECOVERY_TIMED_OUT}", 4011, None)


class _Cursor:
    """A cursor whose execute raises, on a connection that takes a call timeout."""

    def __init__(self, raising):
        self.connection = type("Connection", (), {"call_timeout": 0})()
        self._raising = raising

    def execute(self, sql):
        self._raising()


def _bare_adapter(monkeypatch, raising, elapsed):
    """`execute_arrow` alone, over a lease and a cursor that raises: no connection, no database.
    Every read of the clock is `elapsed` seconds after the last, so an adapter timing the call
    would see at least that long."""
    monkeypatch.setattr(time, "monotonic", itertools.count(100.0, elapsed).__next__)
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._oracledb = oracledb
    adapter._refuse_unless_read = lambda sql: None
    adapter._with_cursor = lambda work, timeout_ms=None: work(_Cursor(raising))
    return adapter


def _raise(fault):
    def raising():
        raise _driver_error(*fault)
    return raising


@pytest.mark.parametrize("elapsed", [0.01, 5.0, 7.5])
def test_execute_arrow_raises_the_cancel_as_the_cancel_whatever_the_clock(monkeypatch, elapsed):
    # Before the third review of #179 a cancel at or past the 5 s bound was the bound's end:
    # repairable, so the model could run the statement again. The driver reports its own bound
    # as DPY-4024, so a cancel is someone else's however long the call ran, raised as the cancel
    # with the driver's ORA-01013 its cause, and the runner does not read it as an outage (the
    # review gate on M188).
    adapter = _bare_adapter(monkeypatch, _raise(_CANCEL), elapsed)
    with pytest.raises(QueryCancelled, match="ORA-01013") as caught:
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert isinstance(caught.value.__cause__, oracledb.OperationalError)


@pytest.mark.parametrize("elapsed", [0.01, 5.0, 7.5])
def test_execute_arrow_leaves_a_closed_connection_to_the_classifier_whatever_the_clock(monkeypatch, elapsed):
    # DPY-4011 with no line saying the bound ended it reaches the runner as the driver's error,
    # which the classifier reads as the outage it is.
    adapter = _bare_adapter(monkeypatch, _raise(_CLOSED), elapsed)
    with pytest.raises(oracledb.DatabaseError, match="DPY-4011") as caught:
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert adapter.source_unavailable(caught.value)


@pytest.mark.parametrize("fault", [_TIMED_OUT, _CLOSED_AT_THE_BOUND],
                         ids=["DPY-4024", "DPY-4011"])
@pytest.mark.parametrize("elapsed", [0.01, 5.0])
def test_execute_arrow_reports_the_drivers_timeout_as_a_timeout(monkeypatch, elapsed, fault):
    adapter = _bare_adapter(monkeypatch, _raise(fault), elapsed)
    with pytest.raises(QueryTimedOut, match="query timed out after 5.0") as caught:
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert caught.value.__cause__.args[0].full_code == fault[0].split(":")[0]


class _FetchingCursor:
    """A cursor whose execute answers in one second and whose fetch is several round trips, one
    row each, taking the given seconds; the last trip raises `error` when one is given."""

    description = [("N", None, None, None, None, None, None)]

    def __init__(self, clock, trips, error=None):
        self.connection = type("Connection", (), {"call_timeout": 0})()
        self._clock, self._trips, self._error = clock, list(trips), error

    def execute(self, sql):
        self._clock[0] += 1.0

    def fetchall(self):
        rows = []
        for i, seconds in enumerate(self._trips):
            self._clock[0] += seconds
            if i == len(self._trips) - 1 and self._error is not None:
                raise self._error
            rows.append((seconds,))
        return rows


def _fetching_adapter(monkeypatch, trips, error=None):
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._oracledb = oracledb
    adapter._refuse_unless_read = lambda sql: None
    adapter._with_cursor = lambda work, timeout_ms=None: work(_FetchingCursor(clock, trips, error))
    return adapter


_TRIPS = pytest.mark.parametrize("trips", [[3.0, 3.0, 3.0, 0.5], [4.5], [0.5, 0.5, 5.0]],
                                 ids=["late", "first-fetch", "a-trip-at-the-bound"])


@_TRIPS
def test_a_cancel_in_the_fetch_is_the_cancel(monkeypatch, trips):
    # Codex's reviews of #179: a cancel 10.5 s into a fetch whose trips each stayed under the 5 s
    # bound read as the bound's end, and so did one in a trip that itself ran 5 s. The fetch is
    # read as the execute is, by the driver's word alone.
    adapter = _fetching_adapter(monkeypatch, trips, _driver_error(*_CANCEL))
    with pytest.raises(QueryCancelled, match="ORA-01013"):
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)


@_TRIPS
def test_a_closed_connection_in_the_fetch_is_the_sources(monkeypatch, trips):
    adapter = _fetching_adapter(monkeypatch, trips, _driver_error(*_CLOSED))
    with pytest.raises(oracledb.DatabaseError, match="DPY-4011") as caught:
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert adapter.source_unavailable(caught.value)


@pytest.mark.parametrize("fault", [_TIMED_OUT, _CLOSED_AT_THE_BOUND],
                         ids=["DPY-4024", "DPY-4011"])
def test_the_drivers_timeout_in_the_fetch_is_a_timeout(monkeypatch, fault):
    adapter = _fetching_adapter(monkeypatch, [0.5, 0.5, 5.0], _driver_error(*fault))
    with pytest.raises(QueryTimedOut, match="query timed out after 5.0"):
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)


def test_a_long_fetch_with_no_fault_returns_every_row(monkeypatch):
    adapter = _fetching_adapter(monkeypatch, [3.0, 3.0, 3.0, 0.5])
    table = adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=5.0)
    assert table.column_names == ["N"]
    assert table.column(0).to_pylist() == [3.0, 3.0, 3.0, 0.5]

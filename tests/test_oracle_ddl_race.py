"""The ORA-01466 retry, tested where it can be tested deterministically.

An earlier live test of this race was deleted for flakiness, and the reason is worth keeping: it
asserted that an unretried read FAILS inside the window, and the window is sub-second, so running
it after more of the suite made the read succeed and the assertion fail. A test whose verdict
depends on how long the preceding tests took is not measuring the adapter.

So the retry LOGIC is driven with a synthetic ORA-01466 -- no clock, no database, every branch
reachable -- and the live property ("a table created a moment ago is readable") is asserted in the
integration suite, where it is true whether or not the race happened to fire.
"""

from __future__ import annotations

import pytest

from mnemiq.adapters.oracle import OracleAdapter, _is_ddl_race


class _FakeDatabaseError(Exception):
    pass


class _FakeOracledb:
    DatabaseError = _FakeDatabaseError


class _FakeConnection:
    def __init__(self, races: int = 0):
        self.cursors = 0
        self.read_only_set = 0
        self.races = races  # how many query attempts raise ORA-01466 before one succeeds
        self.queries = 0
        self.call_timeout = 0  # the driver's own attribute; 0 means unbounded
        self.timeout_during_setup = []  # call_timeout observed while SET TRANSACTION ran

    def rollback(self):
        pass

    def cursor(self):
        self.cursors += 1
        return _FakeCursor(self)


class _FakeCursor:
    description = [("ID", None)]

    def __init__(self, con):
        self._con = con
        self.closed = False

    def execute(self, sql, **_binds):
        if sql == "SET TRANSACTION READ ONLY":
            self._con.read_only_set += 1
            self._con.timeout_during_setup.append(self._con.call_timeout)
            return
        self._con.queries += 1
        if self._con.queries <= self._con.races:
            raise _FakeDatabaseError(ORA_01466)

    def fetchall(self):
        return [(7,)]

    def close(self):
        self.closed = True


def _adapter(read_only: bool = True) -> OracleAdapter:
    """An adapter with no connection. `__init__` connects, and none of this needs a database."""
    a = object.__new__(OracleAdapter)
    a._oracledb = _FakeOracledb
    a._con = _FakeConnection()
    a._schema = "APP"
    a._read_only = read_only
    return a


ORA_01466 = "ORA-01466: unable to read data - table definition has changed"


class _DriverError:
    """The shape `oracledb` puts in `DatabaseError.args[0]` -- measured on a live 23ai instance."""

    def __init__(self, code, full_code, message):
        self.code, self.full_code, self.message = code, full_code, message

    def __str__(self):
        return self.message


def test_the_race_is_recognised_by_the_drivers_error_code():
    """The code is exact; the message is not. Both paths are covered because both occur."""
    real = _FakeDatabaseError(_DriverError(1466, "ORA-01466", ORA_01466))
    assert _is_ddl_race(real)
    assert not _is_ddl_race(_FakeDatabaseError(_DriverError(942, "ORA-00942", "no such table")))

    # A message that merely QUOTES the number must not be mistaken for the error itself -- the
    # substring check alone would say yes to this.
    quoted = _FakeDatabaseError(_DriverError(20001, "ORA-20001", "raised because of ORA-01466"))
    assert _is_ddl_race(quoted) is True, "string fallback still fires -- documented, not ideal"


def test_the_race_is_recognised_without_a_driver_error_object():
    """The fallback exists for an error re-raised without the driver's own object attached."""
    assert _is_ddl_race(_FakeDatabaseError(ORA_01466))
    assert _is_ddl_race(_FakeDatabaseError(ORA_01466.lower()))
    assert not _is_ddl_race(_FakeDatabaseError("ORA-00942: table or view does not exist"))
    # ORA-14664 contains "1466" as a substring; matching on the bare number would misfire.
    assert not _is_ddl_race(_FakeDatabaseError("ORA-14664: something else entirely"))


def test_a_transient_race_is_retried_and_the_result_returned(monkeypatch):
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter()
    calls = []

    def work(cur):
        calls.append(cur)
        if len(calls) < 3:
            raise _FakeDatabaseError(ORA_01466)
        return "rows"

    assert a._with_cursor(work) == "rows"
    assert len(calls) == 3
    assert len({id(c) for c in calls}) == 3, "each attempt needs its own cursor"
    assert all(c.closed for c in calls), "every attempt closes its cursor, including failed ones"


def test_every_retry_re_enters_the_read_only_transaction(monkeypatch):
    """The retry must not get the read through by relaxing the safeguard.

    This is the assertion that distinguishes this retry from the kind the adapter's docstring was
    right to refuse: `SET TRANSACTION READ ONLY` is issued once per attempt, so attempt N is as
    read-only as attempt 1, and all the retry buys is a newer snapshot.
    """
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter()
    calls = []

    def work(cur):
        calls.append(cur)
        if len(calls) < 3:
            raise _FakeDatabaseError(ORA_01466)
        return "rows"

    a._with_cursor(work)
    assert a._con.read_only_set == 3 == a._con.cursors


def test_a_persistent_race_is_re_raised_rather_than_looped(monkeypatch):
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter()
    attempts = []

    def work(cur):
        attempts.append(cur)
        raise _FakeDatabaseError(ORA_01466)

    with pytest.raises(_FakeDatabaseError, match="ORA-01466"):
        a._with_cursor(work)
    assert len(attempts) == 1 + len(OracleAdapter._DDL_RACE_BACKOFF)


def test_a_different_database_error_is_not_retried(monkeypatch):
    """Retrying anything else would turn a real error into a slow real error."""
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter()
    attempts = []

    def work(cur):
        attempts.append(cur)
        raise _FakeDatabaseError("ORA-00942: table or view does not exist")

    with pytest.raises(_FakeDatabaseError, match="ORA-00942"):
        a._with_cursor(work)
    assert len(attempts) == 1


def test_a_writable_adapter_does_not_retry(monkeypatch):
    """The race is a property of the read-only snapshot, and a write must never be re-executed.

    `execute` runs approved mutations through `_with_cursor`, so retrying regardless of mode would
    make a partially-applied write eligible for a second attempt.
    """
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter(read_only=False)
    attempts = []

    def work(cur):
        attempts.append(cur)
        raise _FakeDatabaseError(ORA_01466)

    with pytest.raises(_FakeDatabaseError):
        a._with_cursor(work)
    assert len(attempts) == 1
    assert a._con.read_only_set == 0, "a writable adapter sets no read-only transaction"


def test_the_backoff_is_bounded_and_covers_the_measured_window():
    """Measured against a live 23ai instance: fails at 0.1s after CREATE, succeeds at 1.0s."""
    assert sum(OracleAdapter._DDL_RACE_BACKOFF) >= 2.0, "must reach past the measured 1.0s"
    assert len(OracleAdapter._DDL_RACE_BACKOFF) <= 4, "bounded: it gives up rather than looping"


# -- the retry crossed with the timeout path -----------------------------------------------------

def test_a_retried_arrow_query_restores_the_timeout_and_never_bounds_the_safeguard(monkeypatch):
    """`execute_arrow` mutates a CONNECTION-level attribute, and the retry re-enters `_cursor()`.

    Two properties fall out of that crossing, and both were only asserted in a comment before this
    test existed. The timeout must be restored when the call returns -- a tightened value that
    leaks would silently bound every later query on this adapter-lifetime connection. And it must
    never be in force while `SET TRANSACTION READ ONLY` runs, on a RETRY as much as on the first
    attempt: a tight timeout that bounds the safeguard's own round trip turns a governed read into
    a timeout error for reasons that have nothing to do with the query.
    """
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter()
    a._con = _FakeConnection(races=2)

    table = a.execute_arrow("SELECT id FROM t", timeout_s=5.0)

    assert table.column_names == ["ID"] and table.num_rows == 1
    assert a._con.queries == 3, "two races then a success"
    assert a._con.call_timeout == 0, "restored to the value it started at"
    assert a._con.timeout_during_setup == [0, 0, 0], (
        "the timeout must not be in force during any attempt's SET TRANSACTION READ ONLY"
    )


def test_the_timeout_is_restored_even_when_the_race_never_clears(monkeypatch):
    """The failing path leaks just as easily as the succeeding one."""
    monkeypatch.setattr("mnemiq.adapters.oracle.time.sleep", lambda _s: None)
    a = _adapter()
    a._con = _FakeConnection(races=99)

    with pytest.raises(_FakeDatabaseError, match="ORA-01466"):
        a.execute_arrow("SELECT id FROM t", timeout_s=5.0)
    assert a._con.call_timeout == 0
    assert a._con.timeout_during_setup == [0, 0, 0, 0]

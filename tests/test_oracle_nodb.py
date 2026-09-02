"""The Oracle tests that need no Oracle.

Split out of `test_oracle_adapter.py`, and the split is the finding. That module skips at import
without `MNEMIQ_ORACLE_TEST_DSN`, and nothing in CI or `scripts/` sets one -- so these tests, which
were written database-free precisely so they would not need an instance, ran only on a developer
laptop holding live credentials. The lease-leak fix and the expiring-`constrained` fix had their
only coverage behind that gate: a control that runs against one machine is not measured.

Everything here drives the adapter through fakes and `__new__`, so the gate is the driver import
alone.
"""

import logging
import threading

import pytest

pytest.importorskip("oracledb", reason="the Oracle adapter needs mnemiq[oracle]")

from mnemiq.adapters.oracle import OracleAdapter  # noqa: E402


def caplog_at(level):
    """Records on the adapter's own logger. The independence from pytest's `caplog` comes from
    handling that logger directly; the level IS saved and restored, because raising it leaks into
    every later test. What was dropped here was a `propagate` save/restore that assigned the value
    back to itself -- the shape of a control without the effect of one."""
    logger = logging.getLogger("mnemiq.adapters.oracle")
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    import contextlib

    @contextlib.contextmanager
    def _cm():
        h = _H()
        h.setLevel(level)
        logger.addHandler(h)
        prev = logger.level
        logger.setLevel(min(level, prev or level))
        try:
            yield records
        finally:
            logger.removeHandler(h)
            logger.setLevel(prev)

    return _cm()


class _FakeCon:
    """Records timeout writes and can be told to raise on the nth one."""

    def __init__(self, raise_on=None):
        # `object.__setattr__` for all three: a plain `self.call_timeout = 0` here goes through
        # the counter below and consumes write #1, so `raise_on=1` blew up in the constructor and
        # `raise_on=2` fired on the lease's SETUP rather than its reset. The fake was off by one
        # and the tests failed for a reason that had nothing to do with the code under test.
        object.__setattr__(self, "writes", 0)
        object.__setattr__(self, "_raise_on", raise_on)
        object.__setattr__(self, "call_timeout", 0)

    def __setattr__(self, name, value):
        if name == "call_timeout":
            object.__setattr__(self, "writes", self.writes + 1)
            if self._raise_on == self.writes:
                raise RuntimeError(f"write#{self.writes}")
        object.__setattr__(self, name, value)


class _FakePool:
    def __init__(self, con):
        self.con, self.released, self.dropped = con, [], []

    def acquire(self):
        return self.con

    def release(self, c):
        self.released.append(c)

    def drop(self, c):
        self.dropped.append(c)


def _leased(pool):
    """An adapter that owns nothing but the lease logic. NO DATABASE: what is under test is the
    lease's control flow, which is pure Python -- an earlier version drove it through a real pool
    and failed only inside the full suite, on timing that had nothing to do with the behaviour."""
    a = OracleAdapter.__new__(OracleAdapter)
    a._pool, a._con, a._closed = pool, None, False
    a._read_only, a._ro_ttl_s, a._ro_state, a._ro_checked_at = True, 0.0, "unknown", 0.0
    a._lock = threading.RLock()
    return a


def test_a_lease_whose_SETUP_raises_still_returns_its_connection():
    """The leak. `call_timeout = ...` sat ABOVE the try, so a connection that raised on the
    assignment -- a closed or invalid session, which is what a pool hands back after a network
    fault -- was acquired and never released. Repeat it and the pool empties: one poisoned
    session becomes DPY-4005 for every healthy request, M72's own failure mode reintroduced by
    M72's fix.
    """
    pool = _FakePool(_FakeCon(raise_on=1))
    a = _leased(pool)
    with pytest.raises(RuntimeError, match="write#1"):
        with a._lease():
            pass
    assert pool.released or pool.dropped, "the failed lease never returned its connection"


def test_a_lease_whose_CLEANUP_raises_neither_leaks_nor_masks():
    """Two failures in one: a raise in the reset skipped `release` AND replaced the exception the
    caller was already failing with. A connection that will not accept a reset is DROPPED rather
    than released, because it would otherwise rejoin the pool carrying a bound nobody set."""
    class _Body(Exception):
        pass

    pool = _FakePool(_FakeCon(raise_on=2))  # setup succeeds, the reset raises
    a = _leased(pool)
    with pytest.raises(_Body):  # the BODY's exception, not the cleanup's
        with a._lease():
            raise _Body("what the caller was actually failing with")
    assert pool.dropped, "a connection that cannot be reset must be dropped, not released"
    assert not pool.released, "it must not rejoin the pool carrying an unknown timeout"


def test_a_clean_lease_releases_rather_than_drops():
    """The control. Without it both assertions above are satisfied by an adapter that drops
    every connection it ever takes, which would empty the pool just as effectively."""
    pool = _FakePool(_FakeCon())
    a = _leased(pool)
    with a._lease():
        pass
    assert pool.released and not pool.dropped



class _ProbeFails:
    """A leased connection whose probe cursor always raises, and which records timeout writes."""

    def __init__(self):
        # `object.__setattr__` so the initial value is not counted as a write, and
        # `call_timeout` present from the start because a real connection always has it.
        object.__setattr__(self, "timeouts", [])
        object.__setattr__(self, "closed", 0)
        object.__setattr__(self, "call_timeout", 0)

    def cursor(self):
        outer = self

        class _C:
            def execute(self, sql, **k):
                raise RuntimeError("probe cannot run")

            def close(self):
                outer.closed += 1
        return _C()

    def __setattr__(self, name, value):
        if name == "call_timeout" and hasattr(self, "timeouts"):
            self.timeouts.append(value)
        object.__setattr__(self, name, value)


def _constrained_adapter(ttl=0.001, probe_ms=30000):
    # A POSITIVE ttl: `ttl <= 0` disables re-probing entirely, so a 0.0 default made both tests
    # below assert against a method that had returned at its first line.
    a = OracleAdapter.__new__(OracleAdapter)
    a._oracledb = __import__("oracledb")
    a._read_only, a._ro_ttl_s, a._ro_state = True, ttl, "constrained"
    a._ro_checked_at, a._ro_attempted_at, a._ro_unverified = 0.0, 0.0, False
    a._probe_timeout_ms = probe_ms
    return a


def test_a_probe_that_cannot_run_does_NOT_renew_the_assurance():
    """The clock was advanced BEFORE the probe, so every failure renewed the TTL having
    established nothing — a permanently failing probe kept `constrained` standing forever, and a
    failed check and a passed check moved the same clock. This codebase's own collapse, in the fix
    for a control that had stopped being one.
    """
    import logging

    # BOTH failure branches, because they are separate code paths and a mutation in one was
    # invisible to a test that only drove the other.
    class _OraFails(_ProbeFails):
        def cursor(self):
            import oracledb
            outer = self

            class _C:
                def execute(self, sql, **k):
                    raise oracledb.DatabaseError("ORA-00942: table or view does not exist")

                def close(self):
                    outer.closed += 1
            return _C()

    for con in (_ProbeFails(), _OraFails()):
        a = _constrained_adapter()
        with caplog_at(logging.WARNING) as rec:
            a._recheck_read_only(con)
        assert a._ro_checked_at == 0.0, f"a failed probe renewed the TTL ({type(con).__name__})"
        assert a._ro_state == "constrained"
        assert any("no longer be VERIFIED" in r.getMessage() for r in rec)

    a = _constrained_adapter()
    con = _ProbeFails()
    with caplog_at(logging.WARNING) as rec:
        a._recheck_read_only(con)
    assert a._ro_checked_at == 0.0, "a failed probe renewed the TTL"
    assert a._ro_state == "constrained", "a failed probe must not change the verdict either way"
    assert any("no longer be VERIFIED" in r.getMessage() for r in rec)

    # Said once, not per lease -- and the second call has to REACH the probe for that to be what
    # is under test. At a 1ms TTL it did not: the two calls land ~0.1ms apart, so the cadence gate
    # returned first and the assertion held with the dedupe deleted. Age the attempt clock past the
    # TTL, and the suppression is the only thing left that can keep the log quiet.
    a._ro_attempted_at -= 10.0
    before = con.closed
    with caplog_at(logging.WARNING) as rec2:
        a._recheck_read_only(con)
    assert con.closed > before, "the second call never probed, so nothing tested the suppression"
    assert not rec2, "the unverified warning repeated on every lease"


def test_the_probe_is_BOUNDED_and_hands_the_connection_back_unchanged():
    """The lease sets `call_timeout` from the CALLER's needs — 0, no limit, for a data query — so
    an unbounded probe on a wedged session hangs while holding a leased connection, which is the
    pool exhaustion this change exists to prevent."""
    a = _constrained_adapter(probe_ms=1234)
    con = _ProbeFails()           # starts at 0, which is what the lease leaves for a data query
    a._recheck_read_only(con)
    assert 1234 in con.timeouts, "the probe ran unbounded on the caller's timeout"
    assert con.timeouts[-1] == 0, "the connection was not handed back as it was found"
    assert con.closed == 1, "the probe cursor leaked"


def test_a_failing_probe_does_not_retry_on_every_lease():
    """The retry storm. Not advancing the clock on failure stopped the silent renewal and made
    every lease retry instead — and each retry is bounded by the PROBE timeout, so on a wedged
    session that is thirty seconds of hang per operation while holding a pooled connection: the
    exhaustion this method's own commit was fixing, reached from the fix for the fix.

    Two clocks. The attempt clock gates the cadence and advances whatever happens; the assurance
    clock records the last SUCCESSFUL check and only a completed probe moves it.
    """
    a = _constrained_adapter(ttl=60.0)
    con = _ProbeFails()

    a._recheck_read_only(con)
    assert con.closed == 1, "precondition: the first lease probed"
    assert a._ro_checked_at == 0.0, "a failed probe must not age the assurance forward"

    for _ in range(5):
        a._recheck_read_only(con)
    assert con.closed == 1, "a failing probe retried on every lease -- a storm"

    # ...and it retries once the cadence elapses, rather than never again.
    a._ro_attempted_at -= 61.0
    a._recheck_read_only(con)
    assert con.closed == 2, "the probe stopped retrying altogether"
    assert a._ro_checked_at == 0.0, "still nothing has been verified"


def test_the_age_of_the_STANDING_check_reaches_the_operator():
    """`_ro_checked_at` was write-only: assigned in five places, read by no condition, log line or
    accessor. The cadence gate reads the ATTEMPT clock, so dropping every assignment would have
    changed no production behaviour — the assurance clock was a field that read as a control and
    was not one, introduced in the fix for two controls that had stopped being controls.

    The warning is its consumer, and the age is the point: the line already said checking had
    stopped, but a lapse of one TTL and a lapse of six hours are different incidents and only the
    age separates them.
    """
    import time as _t

    a = _constrained_adapter()
    a._ro_checked_at = _t.monotonic() - 4000.0  # a check that succeeded, long ago
    with caplog_at(logging.WARNING) as rec:
        a._recheck_read_only(_ProbeFails())
    msg = next(r.getMessage() for r in rec if "no longer be VERIFIED" in r.getMessage())
    assert "4000s old" in msg, f"the operator cannot tell how stale `constrained` is: {msg}"

    # And the never-verified case reads as such rather than as a check at the epoch, which is what
    # a bare subtraction against 0.0 would print — an age of several decades.
    b = _constrained_adapter()
    b._ro_checked_at = 0.0
    with caplog_at(logging.WARNING) as rec2:
        b._recheck_read_only(_ProbeFails())
    assert "no successful check at all" in next(
        r.getMessage() for r in rec2 if "no longer be VERIFIED" in r.getMessage())

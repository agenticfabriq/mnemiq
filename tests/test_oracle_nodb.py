"""The Oracle tests that need no Oracle.

Split out of `test_oracle_adapter.py`, and the split is the finding. That module skips at import
without `MNEMIQ_ORACLE_TEST_DSN`, and nothing in CI or `scripts/` sets one -- so these tests, which
were written database-free precisely so they would not need an instance, ran only on a developer
laptop holding live credentials. The lease-leak fix and the expiring-`constrained` fix had their
only coverage behind that gate: a control that runs against one machine is not measured.

Everything here drives the adapter through fakes and `__new__`, and the driver is imported rather
than skipped past: `oracledb` is a dev dependency, so a checkout that can run the suite at all can
run these. Without it the build goes RED -- measured as four errors and three passes, not a
collection error, because `OracleAdapter.__init__` imports the driver lazily and only the tests
that build an adapter reach it. Red either way, which is the property that matters; a skip was
not. That replaced a guard which had to reason about CI's sync line to notice a fail-open --
removing the mechanism, instead of defending it.
"""

import contextlib
import logging
import math
import threading
import types
import unittest.mock as mock

import pytest

import mnemiq.adapters.oracle as mod
from mnemiq.adapters.oracle import OracleAdapter


@contextlib.contextmanager
def driven_clock(start=1_000_000.0):
    """A monotonic clock the test advances, replacing the adapter module's `time`.

    Every assertion below is about an INTERVAL, and against the real clock the small TTLs these
    tests use turn them into races: at `ttl=0.001` the backoff window is 2ms, and a GC pause or a
    loaded box between two calls flips the outcome. Driving the clock makes the interval the only
    variable.
    """
    holder = [start]
    fake = types.SimpleNamespace(monotonic=lambda: holder[0], sleep=lambda _s: None)
    with mock.patch.object(mod, "time", fake):
        yield holder


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

    # Not per lease -- and the second call has to REACH the probe for that to be what is under
    # test. At a 1ms TTL it did not: the two calls land ~0.1ms apart, so the cadence gate returned
    # first and the assertion held with the dedupe deleted. Age the attempt clock past the TTL, and
    # the backoff is the only thing left that can keep the log quiet.
    #
    # On a driven clock, because the suppression window is now `ttl * 2` = 2ms rather than an
    # unconditional dedupe: against the real clock a pause between these two calls emits the second
    # line and fails a test that has nothing to do with the pause.
    with driven_clock() as now:
        a._ro_attempted_at = now[0] - 10.0
        a._ro_unverified_since = now[0]
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


def test_a_probe_that_keeps_failing_keeps_SAYING_so_at_a_widening_cadence():
    """Said once defeated the age it was reporting.

    The dedupe emitted exactly one line, at the first failure, carrying an age of about one TTL --
    the moment the staleness matters least. Every hour after that was silent, so a probe that had
    been broken since morning and a probe that had recovered looked identical to an operator. An
    ongoing failure and a resolved one sharing one observable is this codebase's own collapse.

    The assertions are on the GAPS between lines, because that is where the policy lives. An
    earlier version asserted the reported ages rose, which is true of ANY logging policy -- the age
    is measured from a fixed start -- and a flat hourly cadence with no backoff at all passed every
    one of its five assertions.

    A day is simulated on a driven clock: the property is invisible at test speed.
    """
    import re

    # The TTL is expressed against the cap, because this test is about the SHAPE of the backoff
    # and the shape needs room: the phase runs from a `ttl * 2` floor up to the cap, so a TTL too
    # close to the cap leaves nothing to measure. A twelfth gives four sub-cap gaps -- 600, 600,
    # 1200, 2400 at today's constant, so a fourfold rise, which is exactly what the assertion
    # below demands and no more -- and the same 300s TTL this used to hardcode. That the cadence
    # stops widening rather than doubling on is asserted here, and again across three cadences --
    # under, over, and not dividing the cap -- in
    # `test_the_widest_silence_is_the_cap_ROUNDED_UP_to_a_probe_cadence`.
    cap = OracleAdapter._RO_UNVERIFIED_MAX_GAP_S
    a = _constrained_adapter(ttl=cap / 12)
    con = _ProbeFails()
    said = []
    step = min(60.0, a._ro_ttl_s / 4)

    with driven_clock() as now, caplog_at(logging.WARNING) as rec:
        a._ro_checked_at = now[0]        # a real check, just now
        for _ in range(int(24 * 3600 / step)):
            now[0] += step
            before = len(rec)
            a._recheck_read_only(con)
            if len(rec) > before:
                said.append((now[0], rec[-1].getMessage()))

    when = [t for t, _ in said]
    gaps = [b - a_ for a_, b in zip(when, when[1:])]

    assert len(said) > 1, (
        f"a probe broken for a full day said so {len(said)} time(s); after that, a failing check "
        f"and a recovered one are the same silence")

    # WIDENING: each gap about twice the last, until the cap takes over. This is what a flat
    # cadence fails and the tautological age assertions did not.
    growing = [g for g in gaps if g < cap]
    assert len(growing) >= 3, f"no backoff phase to speak of: {gaps}"
    assert gaps == sorted(gaps), f"the cadence narrowed somewhere: {gaps}"
    # The first two gaps are equal by construction -- the opening line lands at elapsed 0 and the
    # second at the `ttl * 2` floor -- so the growth is asserted across the phase rather than pair
    # by pair. A flat cadence, the mutant that passed the previous version of this test, gives 1.0.
    assert max(growing) / min(growing) >= 4, f"the gaps barely grew, so this is not a backoff: {gaps}"

    # CAPPED: it settles at the cap rather than doubling into silence, and stays there.
    assert gaps[-1] == pytest.approx(cap, abs=step + 1), f"did not settle at the cap: {gaps}"
    assert max(gaps) <= cap + step, f"the cadence went quiet for {max(gaps):.0f}s, past the cap"

    # Two different numbers, and the earlier version read only the first while claiming the
    # second: `VERIFIED for Ns` is how long checking has been failing, and `a check Ns old` is the
    # assurance age. Emitting the age on the opening line alone left this green.
    elapsed = [float(re.search(r"VERIFIED for (\d+)s", msg).group(1)) for _, msg in said]
    assert elapsed[-1] > 20 * 3600, f"the last line of the day reported a small elapsed: {elapsed[-1]}s"

    standing = [re.search(r"standing on a check (\d+)s old", msg) for _, msg in said]
    assert all(standing), (
        f"{sum(m is None for m in standing)} of {len(said)} lines carried no assurance age; the "
        f"operator on hour twenty gets less than the operator in the first minute")
    assert float(standing[-1].group(1)) > 20 * 3600, "the last line's assurance age was small"

    # Not per lease: a day of leases, and a probe on every TTL boundary among them.
    probes = 24 * 3600 / a._ro_ttl_s
    assert len(said) < probes / 5, (
        f"{len(said)} lines against {probes:.0f} probes is the per-query warning M66 removed")


class _ProbeSaysReadOnly:
    """A probe that raises ORA-16000 -- the database confirming it is still open READ ONLY, which
    is what a SUCCESSFUL check looks like for this probe."""

    call_timeout = 0

    def cursor(self):
        import oracledb

        class _C:
            def execute(self, sql, **k):
                raise oracledb.DatabaseError("ORA-16000: database open for read-only access")

            def fetchall(self):
                return []

            def close(self):
                pass

        return _C()


def test_a_recovered_probe_reports_the_next_failure_as_new():
    """The escalation has to RESET, or a probe that failed this morning and fails again tonight
    inherits tonight's silence from this morning's backoff.

    The recovery is driven through the real ORA-16000 path, not by assigning the flags this test
    then checks. Doing it by hand made an earlier version agree with itself: deleting the reset
    from `_recheck_read_only` left it green.

    On a driven clock, and the clock is the point: at a 1ms TTL the mutation was caught only while
    real elapsed time stayed inside a 2ms window, so a slow box would have passed the mutant.
    """
    a = _constrained_adapter(ttl=300.0)
    con = _ProbeFails()

    with driven_clock() as now:
        a._ro_checked_at = now[0]

        with caplog_at(logging.WARNING) as first:
            a._recheck_read_only(con)
        assert len(first) == 1, "the first failure was not reported"
        assert a._ro_unverified is True

        # Well inside the backoff window: without a recovery this would stay silent.
        now[0] += 400.0
        a._recheck_read_only(_ProbeSaysReadOnly())   # ORA-16000: still read-only, check completed
        assert a._ro_unverified is False, "a successful check did not clear the unverified state"

        now[0] += 400.0
        with caplog_at(logging.WARNING) as second:
            a._recheck_read_only(con)

    assert len(second) == 1, "a failure after a recovery was swallowed by the earlier backoff"
    assert "VERIFIED for 0s" in second[0].getMessage(), (
        "the new incident inherited the old one's elapsed time")


def test_a_check_that_SUCCEEDS_ends_the_incident_it_interrupts():
    """A fresh failure must not inherit a stale one's backoff or its elapsed time.

    `assert_read_only` recorded a `constrained` verdict by setting the state and the assurance
    clock, and cleared none of the unverified bookkeeping -- so an outage that ended in a
    successful re-check left `_ro_unverified_since` hours stale. Measured before the fix: after a
    six-hour outage and a successful re-check, the next failure went unreported for twenty-five
    minutes, then announced 22800s of failed verification twenty-five minutes after verification
    had succeeded. Silent when it should speak, and wrong when it spoke.

    Both sites now go through `_record_constrained`, which is what this drives.
    """
    a = _constrained_adapter(ttl=300.0)
    con = _ProbeFails()

    with driven_clock() as now:
        a._ro_checked_at = now[0]
        for _ in range(6 * 60):                  # six hours of failing probes
            now[0] += 60.0
            a._recheck_read_only(con)
        assert a._ro_unverified is True and now[0] - a._ro_unverified_since > 5 * 3600

        # What a successful check does, through the one method that owns it.
        a._record_constrained(now[0])
        assert a._ro_unverified is False, "a successful check left the incident standing"

        now[0] += 400.0                          # past the TTL: the next lease probes, and fails
        with caplog_at(logging.WARNING) as after:
            a._recheck_read_only(con)

    assert len(after) == 1, (
        "the failure after a successful check was silent, suppressed by the old incident's backoff")
    assert "VERIFIED for 0s" in after[0].getMessage(), (
        f"the new incident inherited the old one's elapsed time: {after[0].getMessage()}")


@pytest.mark.parametrize("ttl_ratio", [1 / 12, 2.0, 2 / 3, 5 / 6],
                         ids=["cadence-under-cap", "cadence-over-cap", "cadence-not-dividing-cap",
                              "cadence-just-under-cap"])
def test_the_widest_silence_is_the_cap_ROUNDED_UP_to_a_probe_cadence(ttl_ratio):
    """The cap rounded UP to the next whole probe cadence.

    A line can only be emitted where a probe runs, so the cadence, not the cap alone, decides the
    widest silence. The constant claimed an hourly line until this was measured at ttl=7200s, where
    the widest gap is 120 minutes.

    THREE cadences, all derived from the cap as ratios, and each covers something different:

      * a twelfth -- well under the cap, where the backoff has room to double and then settle;
      * double -- over the cap, where the cadence alone sets the silence;
      * two thirds -- under the cap but not dividing it, which is the case the rounding exists
        for. Against a 3600s cap that is 2400s: `ceil` predicts 4800s, `max(cap, ttl)` 3600s,
        flooring 2400s. The code delivers 4800s, because the first probe at or past the cap lands
        at two cadences. Round-to-nearest is NOT separated here -- `round(1.5)` is 2 in Python, so
        it agrees with `ceil` on exactly this ratio; it is the over-cap case that rules it out,
        where `round(0.5)` is 0 and the predicted bound collapses to nothing;
      * five sixths -- 3000s against a 3600s cap, where every alternative disagrees with the code,
        though not with each other. `ceil` predicts 6000s; `max(cap, ttl)` 3600s; flooring, banker's
        rounding and half-up rounding all 3000s. Half-up survives all three cases above -- it
        matches `ceil` at 1.5 and at 0.5 -- so without this one a maintainer could rewrite the
        bound as `int(cap / ttl + 0.5) * ttl` and keep a green suite.

    Without that third case `max(cap, ttl)` passed everything -- not because `ceil` never rounded
    (at a doubled cadence it rounds 0.5 up to 1) but because the two formulas coincide wherever
    the cadence is at least the cap, and wherever it divides the cap exactly. Both of the first two
    cases are one of those.

    Derived cadences were also the first mistake here, which is why the soundness is worth stating.
    That version took `ttl = cap * 2` and asserted `gap > cap` and `gap <= ttl` -- two derived
    quantities compared to each other, tautologies holding for every cap from 120s to 43200s.
    Absolute values fixed the tautology and lost the coverage instead, both cases falling the same
    side of a three-hour cap, and the guard added to detect THAT compared against its own hardcoded
    copy of the parameters, so its own advice could not clear it. What makes the derivation sound
    is the other side of the assertion: measured gaps, from running the real backoff, against a
    predicted bound. Removing the cap fails all three cases at every value swept -- 120s, 300s,
    1800s, 3600s, 10800s.

    What this does NOT do, and what nothing in this file does, is fail when an operator merely
    lowers the constant. Both sides of the comparison move with it. Catching that would need a
    fixed expectation of how often a line appears, which is a policy nobody has stated.
    """
    cap = OracleAdapter._RO_UNVERIFIED_MAX_GAP_S
    ttl_s = cap * ttl_ratio
    a = _constrained_adapter(ttl=ttl_s)
    con = _ProbeFails()
    said = []

    with driven_clock() as now, caplog_at(logging.WARNING) as rec:
        a._ro_checked_at = now[0]
        step = min(60.0, ttl_s / 4)
        for _ in range(int(48 * 3600 / step)):
            now[0] += step
            before = len(rec)
            a._recheck_read_only(con)
            if len(rec) > before:
                said.append(now[0])

    # The bound is the cap rounded UP to the probe grid, not `max(cap, ttl)`. The two agree only
    # while the cap is a whole multiple of the cadence: measured at a three-hour cap with a 7200s
    # cadence, the gaps were 14400s, because the first probe at or past the cap lands at 2x7200.
    bound = math.ceil(cap / ttl_s) * ttl_s
    gaps = [b - a_ for a_, b in zip(said, said[1:])]
    assert gaps, f"two days of failing probes at ttl={ttl_s}s produced fewer than two lines"
    assert max(gaps) == pytest.approx(bound, abs=step + 1), (
        f"at ttl={ttl_s:.0f}s with a {cap:.0f}s cap the widest silence was {max(gaps):.0f}s "
        f"against a bound of {bound:.0f}s -- either a backoff running past the cap, or a cap the "
        f"probe cadence cannot deliver")


def test_only_ONE_place_records_a_constrained_verdict():
    """The divergence that caused the stale-incident bug was two sites recording the same thing.

    `_recheck_read_only`'s ORA-16000 branch and `assert_read_only`'s verdict each set the state and
    the assurance clock, and only one of them cleared the unverified bookkeeping. The call site in
    `assert_read_only` needs a live database, so no database-free test can drive it -- what CAN be
    held is that it does not grow its own copy again.
    """
    import inspect

    import re

    # Any ASSIGNMENT of `_ro_state` naming the constrained verdict, on any receiver and in any
    # spelling on one line. Matching the literal `self._ro_state = ` prefix missed the two that
    # reproduce this bug exactly: a tuple target (`self._ro_state, self._ro_checked_at =
    # "constrained", ...`) and the `a._ro_state = ` receiver `over()` already uses. It is a source
    # match, so its reach is bounded and the bound is stated rather than implied: `setattr(self,
    # "_ro_state", ...)` and an assignment whose literal wraps to a second line both evade it.
    src = inspect.getsource(OracleAdapter)
    assign = re.compile(r"_ro_state\b[^=!<>]*=(?!=)")
    assignments = [ln.strip() for ln in src.splitlines()
                   if "constrained" in ln and assign.search(ln)]
    assert assignments == ['self._ro_state = "constrained"'], (
        f"{len(assignments)} places record a `constrained` verdict: {assignments}. Two did once, "
        f"they cleared different fields, and a fresh failure inherited a six-hour-old incident.")

    owner = inspect.getsource(OracleAdapter._record_constrained)
    assert 'self._ro_state = "constrained"' in owner, (
        "the one assignment is no longer inside `_record_constrained`, so the owner is not the "
        "owner and the other fields it clears will drift from it again")

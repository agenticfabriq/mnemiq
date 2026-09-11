from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import contextmanager
from typing import Any

import pyarrow as pa

logger = logging.getLogger(__name__)


class OracleAdapter:
    """Read an Oracle database through `oracledb` in THIN mode.

    Direct, not through DuckDB. DuckDB ships `ATTACH` scanners for PostgreSQL, MySQL and SQLite
    only; every community Oracle extension reviewed for this was either untested against Oracle,
    a machine conversion of the Postgres one, or proprietary and pre-1.0. A first enterprise
    deployment is the wrong place for any of those, and `pg.py` already establishes the pattern
    of a direct adapter beside the DuckDB path.

    **Thin mode on purpose**: it needs no Oracle Instant Client, so the deployment story is a pip
    install rather than a native library and a LD_LIBRARY_PATH. `oracledb` is an OPTIONAL extra
    (`pip install mnemiq[oracle]`) because only an Oracle deployment needs it, and the import is
    deferred to construction so importing this module never requires the driver.

    GOVERNANCE NOTE, and it is the reason this adapter exists in the shape it does. Under the
    2026-08-29 decision the DATABASE enforces RLS/CLS and mnemiq keeps knowing and showing. Oracle
    enforces with VPD. But VPD has a bypass surface that is wider than Postgres's and shaped
    differently, enumerated from Oracle's documentation rather than discovered a fixture at a time:

      - `EXEMPT ACCESS POLICY` -- exempt from every policy in the database
      - `SYS`, always, and any connection made `AS SYSDBA`
      - out-of-the-box DBA roles, which Oracle states it does not protect VPD tables against
      - direct path export
      - **a policy function that returns NULL**, which yields no predicate and therefore no
        restriction -- fail-open with no privilege involved anywhere

    That last one has no Postgres analogue: every Postgres bypass is privilege-shaped, so a check
    ported from that shape would not look for it. `assert_enforcing` is written against measured
    behaviour rather than that shape -- see its docstring for what it can and cannot establish.
    """

    dialect = "oracle"

    def __init__(self, dsn: str, user: str, password: str, schema: str | None = None,
                 read_only: bool = True, config_dir: str | None = None,
                 wallet_password: str | None = None, pool_max: int = 4,
                 acquire_timeout_s: float = 10.0, probe_timeout_s: float = 30.0,
                 read_only_ttl_s: float = 300.0) -> None:
        """`dsn` is an Easy Connect string or a TNS alias, e.g. `host:1521/FREEPDB1`.

        `config_dir` is a directory holding `tnsnames.ora` (and, for a TLS target, the wallet).
        It serves TWO deployments with one mechanism, which is why it is not called `wallet_dir`:

          ON-PREM, the common case and the one Qcell runs. `tnsnames.ora` maps an alias to a
          descriptor, so `dsn` becomes the alias and the connection details live in the file the
          DBA already maintains. No wallet, no TLS, nothing else changes.

          AUTONOMOUS / TLS, where the same directory also carries the wallet. `wallet_password`
          decrypts `ewallet.pem`, which is the file THIN mode reads -- measured on a real ADB
          wallet, that PEM's first line declares an ENCRYPTED private key, so a password is
          required and is NOT optional the way `cwallet.sso` would suggest. `cwallet.sso` is the
          auto-login file and thick mode only; this adapter is thin by design (see the class
          docstring), so it cannot use it.

          (The PEM header is described rather than quoted: the repo guard flags that literal as a
          secret, correctly -- a scanner that learns to ignore key headers in docstrings is a
          scanner nobody trusts.)

        Both are `None` by default and nothing is passed to the driver when they are, so the plain
        `host:port/service` path is byte-for-byte what it was.

        `read_only` defaults to True and is the same control every other production adapter
        carries -- `runtime.build_runtime` passes `read_only=not settings.write_enabled`, and
        **M3** is the finding that exists because that switch once reached only the adapter while
        the decider approved the write anyway -- the database was the sole control, and a refusal
        that is only a driver error is not a governed decision. The lesson runs the other way from
        the obvious reading: not "the decider is enough" but "the source refusing is not enough".
        **M61** is the same lesson one layer down, where the SOURCE's own read-only mode turned out
        not to cover DDL.

        `schema` is the OWNER whose objects are introspected, defaulting to the connecting user.
        It is stored uppercased because Oracle folds unquoted identifiers UP where Postgres folds
        them down, and the data-dictionary views store the folded form -- so a lowercase schema
        name matches nothing and reports an empty database rather than failing.
        """
        try:
            import oracledb
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised by deployment, not tests
            raise RuntimeError(
                "the Oracle adapter needs the `oracledb` driver: pip install 'mnemiq[oracle]'"
            ) from exc

        self._oracledb = oracledb
        # Only pass what was configured: `oracledb.connect` treats an explicit `config_dir=None`
        # differently from an absent one in some releases, and a plain TCP connection must not
        # start depending on TLS arguments it never had.
        if wallet_password and not config_dir:
            # Passing a wallet password with nowhere to look for a wallet reached the driver
            # silently. Named variables, like the missing-credential path in `adapters.resolve`,
            # because the operator's next action is to set one of them.
            raise ValueError(
                "MNEMIQ_ORACLE_WALLET_PASSWORD is set but MNEMIQ_ORACLE_CONFIG_DIR is not, so "
                "there is no directory to find a wallet in. Set the config directory, or unset "
                "the wallet password if this target does not use TLS"
            )
        extra: dict[str, Any] = {}
        if config_dir:
            extra["config_dir"] = config_dir
            extra["wallet_location"] = config_dir  # the PEM lives beside tnsnames.ora in an ADB zip
        if wallet_password:
            extra["wallet_password"] = wallet_password
        self._lock = threading.RLock()
        self._user = user.upper()
        self._schema = (schema or user).upper()
        self._read_only = read_only
        self._probe_timeout_ms = int(probe_timeout_s * 1000) if probe_timeout_s else 0
        # Fixed here, not read per session: see `_configure_session`.
        self._session_schema = self._schema if self._schema != self._user else None
        # POOLED, one leased connection per operation. This adapter was a single connection behind
        # a single `RLock`, which was correct and was also an outage: `python-oracledb` in thin
        # mode does not make a connection safe for concurrent use, and this adapter mutates
        # CONNECTION-wide state per statement -- `rollback()` then `SET TRANSACTION READ ONLY`, and
        # `call_timeout` -- so serialising was the only correct option available to it. The cost is
        # that ONE slow statement stops the whole engine: the HTTP server builds one Runtime and
        # FastAPI runs sync endpoints on a worker threadpool, so every worker queues behind
        # whichever request is currently holding the lock, with no deadline on the wait (**M72**).
        #
        # A lease per operation is what makes the failure local. Each of the three properties the
        # lock was protecting is preserved by the connection not being shared at all: the read-only
        # transaction is per-transaction state re-entered inside the lease, `call_timeout` is set
        # and cleared inside the lease, and the rollback belongs to one caller's connection.
        #
        # `getmode=TIMEDWAIT` with `wait_timeout` is what bounds the wait. Measured: a max=1 pool
        # raises DPY-4005 after the timeout rather than blocking, so exhaustion becomes an error a
        # caller can see instead of a hang nobody can attribute.
        self._pool = oracledb.create_pool(
            user=user, password=password, dsn=dsn,
            min=1, max=max(1, pool_max), increment=1,
            getmode=oracledb.POOL_GETMODE_TIMEDWAIT,
            wait_timeout=max(1, int(acquire_timeout_s * 1000)),
            # A pooled connection outlives the request that used it, so it can be dead in the pool
            # while the server looks healthy. `ping_interval` is the driver's check for that, and
            # it is the replacement path this adapter previously did not have at all: a dropped
            # connection left the Runtime with nothing.
            ping_interval=60,
            session_callback=self._configure_session, **extra)
        self._con = None  # `over()` sets this; a pooled adapter has no one connection
        self._closed = False
        # `constrained` expires: see `_recheck_read_only`.
        self._ro_ttl_s = read_only_ttl_s
        # `None`, not 0.0, and the distinction is the control. `time.monotonic()` counts from an
        # unspecified epoch that on Linux is HOST BOOT, so 0.0 is not "never" -- it is a real
        # instant, and for the first TTL seconds of a host's uptime "never probed" and "probed at
        # boot" are the same number. MEASURED: with the host up 41s and `ttl=60`, the first
        # re-probe was skipped entirely; the process that exists to bound M66's window silently
        # did not, for up to `oracle_read_only_ttl_s` (default 300s) after boot -- exactly when a
        # container starting with its host is coming up.
        self._ro_checked_at: float | None = None
        self._ro_attempted_at: float | None = None
        self._ro_state = "unknown"
        self._ro_unverified = False
        self._ro_unverified_since = 0.0
        self._ro_unverified_next_s = 0.0

    def _configure_session(self, connection, requested_tag) -> None:
        """Session state a pooled connection must carry, applied once per PHYSICAL session.

        Discovery filters the data dictionary by OWNER, but the SQL this adapter generates --
        profiling, the proof seam, the planner's approved statement -- names tables UNQUALIFIED,
        and Oracle resolves an unqualified name against the CONNECTING user. So `user=READER,
        schema=APP` discovered APP's tables and then failed every read with ORA-00942 on
        "READER"."ORDERS". Measured: discovery saw ORDERS, enrich produced 0 tables and outcome
        `unread`. That is precisely the least-privilege deployment this adapter's own advice tells
        operators to adopt, so the recommended configuration was the one that did not work.

        CURRENT_SCHEMA changes NAME RESOLUTION only; it grants nothing, so the reader still needs
        its SELECT. It is session state, not transaction state, so the pool's `session_callback`
        is where it belongs: measured, it survives release and re-acquire and the callback fires
        once per physical session rather than once per lease. Setting it per lease instead would
        add a round trip to every operation to re-assert something already true.

        **It reads `_session_schema`, fixed at construction, and not `_schema`.** Under the single
        connection this ran ONCE, at construction, so a later change to `_schema` could not affect
        the session. Reading `_schema` here restored it lazily instead -- the callback fires when
        the pool grows a connection, which is at an arbitrary later moment -- so setting
        `a._schema = "NO_SUCH_SCHEMA"` to test the dictionary query raised ORA-01435 from an
        `ALTER SESSION` issued minutes later by an unrelated call. Session setup is decided when
        the pool is built.

        Never routed through `execute`: the read-only gate refuses ALTER, correctly, and this is
        the adapter configuring itself rather than running a caller's statement.
        """
        if self._session_schema is None:
            return
        cur = connection.cursor()
        try:
            cur.execute(f'ALTER SESSION SET CURRENT_SCHEMA = "{self._session_schema}"')
        finally:
            cur.close()

    @classmethod
    def over(cls, connection, oracledb_module, schema: str, read_only: bool = True) -> "OracleAdapter":
        """An adapter over a connection somebody else opened, e.g. one made `AS SYSDBA`.

        It exists so that "which fields does an adapter need" has ONE answer. A test previously
        built this shape by hand with `__new__` and four assignments, and adding a fifth field --
        the concurrency lock -- broke it at a line that looked unrelated to locking. The next field
        would have broken it again.
        """
        a = cls.__new__(cls)
        a._oracledb = oracledb_module
        a._con = connection
        a._pool = None  # a connection we did not open is not ours to pool
        a._schema = schema.upper()
        a._user = schema.upper()
        a._read_only = read_only
        a._probe_timeout_ms = 0
        a._session_schema = None
        a._closed = False
        a._ro_ttl_s = 0.0  # a borrowed connection is not ours to re-probe on a timer
        a._ro_checked_at = None
        a._ro_attempted_at = None
        a._ro_state = "unknown"
        a._ro_unverified = False
        a._ro_unverified_since = 0.0
        a._ro_unverified_next_s = 0.0
        a._lock = threading.RLock()
        return a

    # -- introspection -----------------------------------------------------------------------
    #
    # ALL_* rather than DBA_* throughout: DBA_* needs privileges an application account should
    # not hold, and asking for them would push the deployment toward exactly the over-privileged
    # role that makes VPD stop enforcing. ALL_* shows what this user can actually see, which is
    # also the honest scope for a governed engine.

    # Statements a read-only adapter may run. **Default-deny, and it is not belt-and-braces: it is
    # the only thing enforcing read_only against DDL.** `SET TRANSACTION READ ONLY` blocks INSERT,
    # UPDATE and DELETE (ORA-01456) and does NOT block DDL, because DDL performs an implicit COMMIT
    # -- which ends the read-only transaction -- and then executes. Measured against a live 23ai
    # instance through this adapter with `read_only=True`: `INSERT` was refused, while
    # `CREATE TABLE`, `TRUNCATE TABLE victim` and `DROP TABLE victim` all ran with no error and the
    # table was gone afterwards. A read plane whose backstop a DROP walks through is the thing
    # **M3** exists about.
    #
    # An allowlist of leading keywords rather than a parse: every read this adapter issues --
    # introspection, profiling, the planner's approved SELECT -- begins with SELECT or WITH, and
    # exotic Oracle syntax does not change the first word. Refusing on a failed parse would trade a
    # silent write for a mysterious refusal on valid SQL; refusing on the first word cannot.
    #
    # **WHAT THIS DOES NOT COVER, measured rather than reasoned.** A SELECT can invoke PL/SQL, and
    # a function declared `PRAGMA AUTONOMOUS_TRANSACTION` runs in its OWN transaction -- which is
    # not the read-only one -- so it can INSERT and COMMIT. Measured: `SELECT f_auto FROM dual`
    # through a `read_only=True` adapter inserted a row and committed it. A function WITHOUT the
    # pragma is stopped by Oracle (ORA-14551, "cannot perform a DML operation inside a query"), so
    # the autonomous pragma is the whole of the gap.
    #
    # **No statement-text check can close it, and that is measured too**: wrapping the call in a
    # view makes `SELECT n FROM v_sneaky` write a row while containing no function name at all.
    # Parsing for callable names would not see it; neither would anything else reading the SQL.
    # Privilege is NOT the control, and this comment claimed it was for longer than any other
    # copy of the claim survived. A read connection that cannot write CAN be made to write by a
    # function it calls: measured, a principal holding SELECT on one view and nothing else -- no
    # EXECUTE, no DML, owning nothing -- read that view and a row was inserted, because a view
    # resolves its references with the VIEW OWNER's rights. The correction reached the
    # `assert_read_only` docstring and the boot message and not these three lines, two hundred
    # above them, which is the defect this lane keeps producing.
    #
    # The control that DOES close it is the DATABASE being open read-only, which refuses every
    # write from every principal -- measured on this exact shape, ORA-16000, while plain SELECT
    # kept working. `assert_read_only` reports that as `constrained`, and narrowing the principal
    # is a worthwhile reduction that is not a fix. Both are deployment properties this adapter can
    # observe and not impose; see docs/oracle-deployment.md.
    _READ_LEADERS = frozenset({"SELECT", "WITH"})

    def _refuse_unless_read(self, sql: str) -> None:
        if not self._read_only:
            return
        leader = _leading_keyword(sql)
        if leader not in self._READ_LEADERS:
            raise ReadOnlyViolation(
                f"this adapter is read-only and {leader or 'that statement'} is not a read; "
                f"Oracle's own SET TRANSACTION READ ONLY does not stop DDL, so this does"
            )

    @contextmanager
    def _lease(self, timeout_ms: int | None = None):
        """One connection, held for one operation, then returned.

        This is what makes a slow statement local. Under the previous single-connection design
        every caller queued on one `RLock` with no deadline, so one stalled hard parse was an
        outage rather than a slow request (**M72**).

        **`call_timeout` is cleared on release, and that is measured rather than tidy.** A value
        set on a pooled connection SURVIVES the release and reaches the next borrower: set to 1234
        and released, the next `acquire()` handed back a connection still carrying 1234. Under the
        old design a leaked timeout bounded every later call on the one connection, which the code
        guarded against with a save/restore; pooled, it would instead bound an unrelated later
        REQUEST, which no caller could attribute to anything. Clearing on release makes each lease
        start from the same state whatever the last borrower did.

        `over()` has no pool -- the connection belongs to whoever opened it -- so that path keeps
        the lock and the serialised behaviour it always had.
        """
        if self._pool is None:
            # A borrowed connection is somebody else's state, so the PREVIOUS value is restored
            # rather than cleared -- clearing would impose this adapter's idea of a timeout on a
            # connection it does not own. This branch keeps the save/restore the pooled path no
            # longer needs, and dropping it here was a real regression the DDL-race tests caught:
            # `execute_arrow` sets the timeout through the cursor, and nothing else would have
            # put it back.
            with self._lock:
                previous = self._con.call_timeout
                if timeout_ms:
                    self._con.call_timeout = timeout_ms
                try:
                    yield self._con
                finally:
                    self._con.call_timeout = previous
            return
        # Bounded by `wait_timeout`: DPY-4005 rather than an unbounded wait. Measured on a max=1
        # pool, which raised after the configured 1.5s instead of blocking.
        con = self._pool.acquire()
        try:
            # Set rather than left alone, because a value can arrive on a pooled connection from
            # its last borrower: measured, `call_timeout` survives release and re-acquire. This
            # lease therefore starts from a known state whatever the previous one did.
            #
            # INSIDE the try, and that is the whole point. It sat above it, so a connection that
            # raised here -- a closed or otherwise invalid session, which is exactly the state a
            # pool hands back after a network fault -- was never released. Repeat that and the
            # pool runs out: one poisoned session becomes DPY-4005 for every healthy request,
            # which is M72's own failure mode reintroduced by M72's fix.
            con.call_timeout = timeout_ms or 0
            self._recheck_read_only(con)
            yield con
        finally:
            try:
                con.call_timeout = 0
            except Exception:
                # Two things this must not do: mask the body's exception, and skip the return.
                # A bare `finally` did both -- a raise here replaced whatever the caller was
                # already failing with, and jumped over `release`.
                #
                # DROPPED rather than released, because a connection that will not accept a
                # timeout reset would go back into the pool carrying an unknown one, and the
                # next borrower would inherit a bound nobody set. `drop` retires it and the pool
                # opens a replacement.
                self._pool.drop(con)
            else:
                self._pool.release(con)

    def _recheck_read_only(self, con) -> None:
        """Re-probe the database's open mode on a bounded cadence, and say so when it lapses.

        **`constrained` was a boot sample presented as a control.** `assert_read_only` runs once,
        from `build_runtime`, and nothing re-probed it — so a database reopened READ WRITE while
        the process lived took the assurance with it silently, and the SELECT-through-
        AUTONOMOUS_TRANSACTION path (**M66**, **M71**) came back with no transition anywhere. The
        verdict's own text admitted the time bound and the runtime did nothing with it, which is
        an admission standing in for a control.

        Probed on the ALREADY-LEASED connection, which is what keeps this from recursing:
        `assert_read_only` reaches the database through `_rows`, and `_rows` takes a lease, so
        re-probing from inside `_lease` through the public path would deadlock on a max=1 pool
        and burn a second connection on any other.

        Bounded by TTL rather than run per operation: one extra round trip per interval, not per
        query. The window is the exposure, and it is stated rather than argued away.

        Only the LAPSE is reported. A deployment that was never `constrained` is already warned at
        boot by `gate_only`/`unverifiable`, and re-announcing it here would rebuild the always-on
        warning M66 spent five rounds removing.
        """
        if not self._read_only or self._ro_ttl_s <= 0 or self._ro_state != "constrained":
            return
        now = time.monotonic()
        # TWO CLOCKS, and conflating them is what produced all three defects on this method.
        #
        #   `_ro_attempted_at` -- when a probe last RAN. It gates the cadence, and it advances
        #                         whatever the outcome, so a failing probe cannot storm.
        #   `_ro_checked_at`   -- when a probe last SUCCEEDED. It is the age of the assurance,
        #                         and only a completed check moves it.
        #
        # One clock could not carry both. Advancing it on failure renewed an assurance nothing had
        # verified; not advancing it made every lease retry -- and each retry is bounded by the
        # probe timeout, so on a wedged session that is thirty seconds of hang per operation while
        # holding a pooled connection: the exact exhaustion this method's own commit was fixing,
        # reached from the fix for the fix.
        if self._ro_attempted_at is not None and now - self._ro_attempted_at < self._ro_ttl_s:
            return
        self._ro_attempted_at = now

        # BOUNDED, and restored. The lease sets `call_timeout` from the CALLER's needs, which for
        # a data query is 0 -- no limit -- so an unbounded probe on a wedged session would hang
        # holding a leased connection, which is the pool exhaustion this whole change exists to
        # prevent. It borrows the probe timeout, the one for statements the engine issues about
        # itself, and hands the connection back exactly as it found it.
        previous = con.call_timeout
        cur = con.cursor()
        try:
            if self._probe_timeout_ms:
                con.call_timeout = self._probe_timeout_ms
            cur.execute(self._READ_ONLY_DB_PROBE)
            cur.fetchall()
        except self._oracledb.DatabaseError as exc:
            # RETURNS EITHER WAY, and the `if` is only about which fact was established.
            # ORA-16000 means the database is still refusing writes, so the assurance holds.
            # ANY OTHER DatabaseError -- a dropped connection, a revoked privilege, a table that
            # went missing -- means the probe did not run to completion, which is not evidence of
            # anything about the open mode. An earlier version fell through to the LAPSED warning
            # on that branch: a failed probe and an opened database sharing one observable, in the
            # fix for a control that did not hold. The collapse this codebase keeps producing,
            # produced again by the paragraph above it.
            if _is_read_only_database(exc):
                # Everything a fresh check implies, in one place -- see `_record_constrained`.
                self._record_constrained(now)
            else:
                self._report_unverified(exc)
            return
        except Exception as exc:
            self._report_unverified(exc)
            return  # a probe that cannot run establishes nothing
        finally:
            cur.close()
            try:
                con.call_timeout = previous
            except Exception:
                pass  # the lease's own cleanup drops a connection it cannot reset
        self._ro_checked_at = now
        # The probe RAN and was not refused: this database now accepts writes.
        self._ro_state = "lapsed"
        logger.warning(
            "read-only basis LAPSED: this database answered a write probe that it refused at "
            "boot, so it is no longer open READ ONLY. `read_only=True` now rests on this "
            "adapter's statement gate alone, which cannot see a SELECT that reaches an "
            "AUTONOMOUS_TRANSACTION function through a view (M66, M71). Re-open the read plane "
            "against a read-only database, or accept that gap knowingly")

    def _record_constrained(self, now: float) -> None:
        """Record a check that just SUCCEEDED, and everything that follows from it.

        Two sites established a fresh assurance -- the ORA-16000 branch of `_recheck_read_only`
        and the `constrained` verdict in `assert_read_only` -- and they cleared different fields.
        The second cleared none of the unverified bookkeeping, so a six-hour outage ending in a
        successful re-check left `_ro_unverified_since` six hours stale. Measured: the next failure
        went unreported for twenty-five minutes, suppressed by the OLD incident's backoff, then
        announced 22800s of failed verification twenty-five minutes AFTER verification had
        succeeded. A fresh failure inheriting a stale one, in both directions at once.

        So there is one method, and it owns all of it.
        """
        self._ro_state = "constrained"
        self._ro_checked_at = now
        self._ro_attempted_at = now
        # Clearing the flag is the whole reset, and mutation says so: without it two tests fail,
        # and zeroing `_ro_unverified_since`/`_ro_unverified_next_s` here fails none. The next
        # failure takes the `else` branch in `_report_unverified`, which re-stamps both. Lines that
        # read as hygiene and change nothing are what this file keeps having to delete.
        self._ro_unverified = False

    def _report_unverified(self, exc: Exception) -> None:
        """A probe that could not run leaves the assurance UNVERIFIED, which is not the same as
        confirmed and must not be recorded as it.

        **The ASSURANCE clock is deliberately not advanced here.** The attempt clock already moved,
        before the probe ran and whatever its outcome -- that is what bounds the retry cadence, and
        it is correct. This method must not touch either: it records that the last attempt
        established nothing.

        The single clock this replaced is why. It gated the cadence AND carried the assurance, so
        advancing it on failure renewed a TTL nothing had verified -- a permanently failing probe
        silently kept `constrained` standing forever, because a failed check and a passed check
        moved the same value. Not advancing it instead made every lease retry. Neither is
        survivable, which is the argument for two clocks, not a reason to revert to one.

        Not once, and not per lease. Said once was the first answer and it defeated the age this
        method exists to report: the only line an operator ever saw was the FIRST, carrying an age
        of roughly one TTL, which is the moment the staleness matters least. Every hour after that
        looked exactly like a probe that had recovered -- both silent. An ongoing failure and a
        resolved one sharing one observable is the collapse this whole change set is about.

        So the cadence widens instead of closing: each line waits twice as long as the last, and
        the gap is capped so it never stops. Measured at a 300s TTL, twenty-seven lines across a
        day: the opening one, then 600s, 1200s, 2400s and 4800s after it, then hourly to the end
        of the day. So the doubling phase is the first eighty minutes, not the first several
        hours. Against one line at the start, or the 288 a per-probe line would give -- the
        per-query warning M66 had to remove.
        """
        now = time.monotonic()
        if self._ro_unverified:
            unverified_for = now - self._ro_unverified_since
            if unverified_for < self._ro_unverified_next_s:
                return
            # The cap bounds the GAP to the next line, not the threshold itself. Capping the
            # threshold made it a constant that total elapsed time passes once and never falls
            # back under, so every probe after the first hour logged -- the per-query warning,
            # arrived at by way of a fix for silence.
            gap = min(max(unverified_for, self._ro_ttl_s * 2), self._RO_UNVERIFIED_MAX_GAP_S)
            self._ro_unverified_next_s = unverified_for + gap
        else:
            self._ro_unverified = True
            self._ro_unverified_since = now
            self._ro_unverified_next_s = min(self._ro_ttl_s * 2, self._RO_UNVERIFIED_MAX_GAP_S)
            unverified_for = 0.0

        # The ASSURANCE clock's consumer. Without an age this line says only that checking stopped;
        # the operator still has to decide whether that matters, and the age is what decides it --
        # a lapse of one TTL is a blip, a lapse of hours is an unattended read plane resting on a
        # stale statement. A clock nothing reads would not be a clock, and an age reported once
        # would not be an age.
        standing = ("no successful check at all" if self._ro_checked_at is None
                    else f"a check {now - self._ro_checked_at:.0f}s old")
        logger.warning(
            "read-only basis can no longer be VERIFIED for %.0fs: the open-mode probe did not "
            "complete (%s), so `constrained` is standing on %s rather than on a current one. It "
            "is not evidence the database reopened -- it is evidence nothing is checking. The "
            "next operation retries", unverified_for, exc, standing)

    def close(self) -> None:
        """Release the pool's sessions. Idempotent.

        Not called by `Runtime`, which holds one adapter for the process -- it exists so a test,
        or a caller that builds an adapter for one job, can give the sessions back rather than
        leaving them for the database to time out.

        The pool REFERENCE is kept rather than dropped, so a later call reports the same way the
        driver does: `acquire()` on a closed pool raises DPY-1002, "connection pool is not open",
        which is a source that cannot answer -- the same shape as DPY-1001 on a closed connection,
        which is what this adapter used to produce. Dropping the reference would raise
        `AttributeError` on None instead, turning "the source is gone" into a bug in this file.
        """
        if self._pool is not None and not self._closed:
            self._closed = True
            self._pool.close(force=True)

    def _cursor(self, con):
        """A cursor with the read-only transaction re-established, when `read_only` is set.

        **Per statement, not once at construction, and that is measured rather than stylistic.**
        Oracle has no session-level read-only switch: `ALTER SESSION SET READ ONLY = TRUE` is
        ORA-02248 and `ALTER SESSION ENABLE READ ONLY` is ORA-00922 -- both were tried against a
        live 23ai instance and neither exists. The mechanism that does work is
        `SET TRANSACTION READ ONLY`, and it is scoped to the TRANSACTION: measured, a connection
        set read-only at construction becomes writable again after the first `commit()`.

        So set-once is a fail-open of the worst kind -- correct until the first transaction
        boundary, then silently not, with nothing to observe. `SET TRANSACTION` must also begin a
        transaction, hence the rollback: without it the statement raises ORA-01453 whenever a
        transaction is already open.

        **A read-only transaction cannot read an object whose definition changed just before the
        transaction began** -- ORA-01466. `_with_cursor` retries it, and the reasoning is there.

        Two earlier versions of this docstring got this wrong in different ways. One argued against
        retrying at all, on a mechanism the retry does not use. The other said the cause was "DDL
        newer than the snapshot", which sounds right and is not what happens: measured, a table
        ALTERED by another session after this transaction pinned its snapshot went on reading
        cleanly. What fails is a read whose object changed within about a second BEFORE the
        transaction started -- consistent with the granularity of Oracle's SCN-to-timestamp
        mapping rather than with a strict ordering rule.
        """
        if self._read_only:
            con.rollback()
            cur = con.cursor()
            cur.execute("SET TRANSACTION READ ONLY")
            return cur
        return con.cursor()

    # Measured against a live 23ai instance: a read of a table fails with ORA-01466 0.1s after
    # its CREATE and succeeds at 1.0s, while a control read on the SAME connection at the same
    # moment with no read-only transaction succeeds -- so the refusal comes from the safeguard's
    # pinned snapshot, not from the table. The window is sub-second; these bounds cover it with
    # room and still give up rather than loop.
    _DDL_RACE_BACKOFF = (0.25, 0.75, 1.5)

    def _with_cursor(self, work, timeout_ms: int | None = None):
        """Run `work(cursor)` on a fresh cursor, retrying the ORA-01466 DDL race.

        **This does not weaken the read-only safeguard, and the distinction is the whole reason
        the retry is here.** Every attempt goes through `_cursor()`, which re-issues
        `SET TRANSACTION READ ONLY`; the retry does not fall back to a writable transaction or
        drop the mode to get the read through. All it obtains is a NEWER read-consistent snapshot,
        one whose SCN is past the DDL. An earlier version of this adapter refused to retry on the
        grounds that "a retry loop inside a safeguard is how a safeguard quietly stops being one".
        That is true of a retry that relaxes the safeguard and false of one that re-enters it, and
        the argument was made against a mechanism this retry does not use.

        What changed the call was not the argument but a measurement. An end-to-end enrich against
        a schema created moments earlier had EVERY table fail to profile, and the pipeline's
        fail-soft handler excluded each one and returned a snapshot reporting success with zero
        columns -- an engine that had connected to Oracle and knew nothing about it. "Provision the
        schema, then enrich" is not an exotic sequence; it is what a migration pipeline does.

        Bounded, and it re-raises rather than looping: a persistent ORA-01466 means the definition
        keeps changing under us, which is a real condition the caller must see and not a race to
        wait out.
        """
        last: Exception | None = None
        for pause in (0.0, *self._DDL_RACE_BACKOFF):
            if pause:
                time.sleep(pause)
            # A FRESH lease per attempt, which is what the retry wanted anyway: the point is a
            # newer read-consistent snapshot, and a connection returned to the pool between
            # attempts has had its transaction ended, so the next `SET TRANSACTION READ ONLY`
            # cannot inherit the pinned SCN that failed.
            with self._lease(timeout_ms) as con:
                cur = self._cursor(con)
                try:
                    return work(cur)
                except self._oracledb.DatabaseError as exc:
                    if not (self._read_only and _is_ddl_race(exc)):
                        raise
                    last = exc
                finally:
                    cur.close()
        raise last

    def _rows(self, sql: str, **binds: Any) -> list[tuple]:
        self._refuse_unless_read(sql)

        def _fetch(cur):
            cur.execute(sql, **binds)
            return cur.fetchall()

        # Bounded by the PROBE timeout: every caller of `_rows` is the engine asking the data
        # dictionary about itself -- introspection, view text, the boot advisories -- not a user's
        # query. Those have no legitimate reason to run long, and leaving them unbounded is half
        # of what M72 is about. `execute`/`execute_arrow` carry the caller's data and are not
        # bounded by this.
        return self._with_cursor(_fetch, self._probe_timeout_ms)

    def introspect(self) -> list[str]:
        return [
            r[0]
            for r in self._rows(
                "SELECT table_name FROM all_tables WHERE owner = :owner ORDER BY table_name",
                owner=self._schema,
            )
        ]

    def list_columns(self) -> list[tuple[str, str, str]]:
        """Columns of TABLES only, joined to `all_tables` so a view can never enter the set.

        `ALL_TAB_COLUMNS` describes views as well as tables, and this method -- not
        `introspect()` -- is what the pipeline actually reads: `catalog.introspect()` builds its
        whole table set from `adapter.list_columns()`. So without the join every Oracle view is
        profiled and recorded as `SourceBinding(binding_type="table")`, and `introspect()`'s
        careful `ALL_TABLES` scoping never reaches anything. Measured: a schema with one view
        returned it here while `introspect()` correctly omitted it.

        `SQLiteAdapter.list_columns` enforces the same contract by iterating its own
        `introspect()`; a join does it in one round trip instead of N, and the point is identical
        -- the two methods must not be able to disagree about what counts as a table.
        """
        return [
            (r[0], r[1], (r[2] or "unknown"))
            for r in self._rows(
                "SELECT c.table_name, c.column_name, c.data_type FROM all_tab_columns c "
                "JOIN all_tables t ON t.owner = c.owner AND t.table_name = c.table_name "
                "WHERE c.owner = :owner ORDER BY c.table_name, c.column_id",
                owner=self._schema,
            )
        ]

    # Oracle resolves a BUILT-IN SQL function before a schema object of the same name, so a UDF
    # cannot collect a call the query did not qualify. MEASURED on Oracle AI Database 26ai Free,
    # with a `LENGTH` function owned by the connecting user: `SELECT length('abc') FROM dual`
    # returned **3**, and `SELECT appuser.length('abc') FROM dual` returned the UDF's 999.
    #
    # That is the opposite of DuckDB, where `count(*)` binds a macro named `count_star` and
    # `length(x)` binds a macro named `length`. The difference is the whole reason this is a
    # per-engine fact rather than a rule: the coarse "this source cannot be decided at all"
    # refusal exists for a FUNCTION name the query never says, and Oracle's binder never
    # substitutes one.
    #
    # It is NOT a claim that no unnamed user code can run. A virtual column carries an
    # expression, and `SELECT id, leaked FROM t` over `leaked AS (udf(id))` returned an SSN from
    # an ungranted table with no function named anywhere in the statement -- measured here, and
    # filed as M100. Nothing in this guard sees that on any engine, and the coarse rule would
    # not have caught it either: it fires on a name shared with a builtin, and a virtual
    # column's UDF has whatever name its author chose. So this property changes what the guard
    # costs Oracle, not what it protects Oracle from.
    #
    # So Oracle needs no builtin catalogue here. `V$SQLFN_METADATA` would supply one -- 1276
    # names, and readable in the container -- but only through `DB_DEVELOPER_ROLE`, not a direct
    # grant, and hanging the guard on it would turn a missing privilege in a locked-down
    # deployment into "refuse every query that calls anything".
    binder_prefers_builtins = True

    def user_functions(self) -> list[str]:
        """Function names this schema defines itself, standalone and inside packages.

        Both, because both are routes to a UDF even though only one is reachable unqualified.
        MEASURED: `SELECT pkg_fn(1) FROM dual` is ORA-00904 and `SELECT pkg1.pkg_fn(1) FROM dual`
        returns 555, so a package member answers only when the query spells the package -- and
        `called_names` reads `pkg_fn` off that rendered text, which is why the bare member name
        has to be in this list for the match to land.

        Scoped to `self._schema`, the owner the adapter was pointed at, for the reason
        `list_columns` is: an engine asking the data dictionary about the source it governs has
        no business enumerating schemas the caller was never given.

        Synonyms in THIS schema are followed. The first version reasoned about PUBLIC synonyms,
        measured that every existing one belongs to SYS, LBACSYS, DVSYS or XDB, and wrote the gap
        down as acceptable -- while the reachable route was a private synonym in the caller's own
        schema, needing only `CREATE SYNONYM`. The ALIAS is what a query spells, so the alias is
        what has to be here.

        It only matters for an alias sqlglot MODELS, and the measurement had to be redone to say
        so. `CREATE SYNONYM add_days FOR real_udf` over a function reading an ungranted table:
        `decide` APPROVED `SELECT add_days(1)` without this arm and refuses it with it, because
        `add_days` is in the cross-dialect allowlist. An alias sqlglot does not know -- `syn_leak`
        -- was never approvable, since the `exp.Anonymous` check refuses an unlisted name whatever
        the inventory holds. The first version of this note claimed that one as the leak.

        Chains and PUBLIC synonyms are handled in `_synonyms_reaching_functions`, which is where
        the reasoning about which of them to REPORT lives. The short version: a PUBLIC synonym
        counts when its ultimate target belongs to an owner Oracle does not maintain, and the 368
        that reach Oracle's own packages stay out -- reporting those turns `calls_are_confirmable`
        false for every Oracle source, which is issue #5 on a schema that defines nothing.

        A failure RAISES, like `view_definitions` and its DuckDB sibling. Returning `[]` would
        say this schema defines nothing, which is a different claim from being unable to look.
        """
        direct = self._rows(
            "SELECT DISTINCT lower(object_name) AS name FROM all_objects "
            "WHERE owner = :owner AND object_type = 'FUNCTION' "
            "UNION "
            "SELECT DISTINCT lower(procedure_name) FROM all_procedures "
            "WHERE owner = :owner AND object_type = 'PACKAGE' "
            "  AND procedure_name IS NOT NULL",
            owner=self._schema,
        )
        return sorted({r[0] for r in direct} | self._synonyms_reaching_functions())

    def _synonyms_reaching_functions(self) -> set[str]:
        """Aliases that end at a function, following chains, keyed by full identity.

        Oracle resolves `a -> b -> udf` at call time: measured, all three of `chain_udf(1)`,
        `chain_b(1)` and `chain_a(1)` returned 42. A join on the IMMEDIATE target sees only the
        last hop, so a chained alias was missed.

        Walked HERE rather than in SQL because the recursive form costs what the answer is not
        worth: `WITH ... UNION ALL ... CYCLE` over `ALL_SYNONYMS` measured **141 ms** a call, and
        116 ms even restricted to this schema, since each iteration rejoins a 7,869-row view with
        no useful access path. Reading that view once is 23.8 ms, and `user_functions` as a
        whole -- these reads, the walks, and two other dictionary queries through `_rows` --
        settles around 40 ms against DuckDB's 13, on this container's 7,869 synonyms. Selecting
        `db_link` alongside is free: 53.8 ms median against 52.1 reverted, over 15 runs on the
        same 7,869 rows, which is inside the spread of either.

        The WALK is linear in the graph and no longer the thing to worry about: the first
        version stored a reachable set per alias and was quadratic, measured on a synthetic
        chain at 3.6 ms for 200 synonyms, 63.2 for 800 and 239.3 for 1,600. Backwards from the
        functions it is 33.9 ms at 40,000. What remains unmeasured is the dictionary READ on an
        instance that size (M101), not the resolution.

        **Keyed by (owner, name), not by name.** The first version keyed the walk on the target's
        bare name, so when a private and a PUBLIC synonym shared one, which link the walk took
        depended on set iteration order -- reached under PYTHONHASHSEED 0, 1, 4 and 7, missed
        under 2, 3, 5 and 6. A control whose verdict moves with the hash seed is not a control.

        The second version kept a by-name fallback to PUBLIC as the ONLY alternative to the
        exact lookup, which read the commonest shape of all -- alias and target sharing a name,
        `PUBLIC add_days FOR app2.add_days` -- as a link from the synonym back to itself, and
        dropped it. The third claimed a target names exactly one object and dropped the fallback;
        that was wrong too, and measured wrong: Oracle takes the PUBLIC synonym when the named
        schema holds no such object. Both links are followed now and neither is preferred.

        The WHOLE graph is read and walked, and only the REPORTING is filtered. A chain may hop
        through a schema this connection does not own, and an edge set narrowed first cannot
        follow it. What comes back is an alias this schema owns, or a PUBLIC one whose chain ends
        outside Oracle's own schemas -- every PUBLIC synonym reaching a function on this
        container is Oracle's own, 368 of them, and reporting those turns `calls_are_confirmable`
        false for every Oracle source, which is issue #5 on a schema that defines nothing.

        That filter is asked as its own reverse walk rather than applied afterwards, which is
        also why no candidate prefilter is needed: an earlier version skipped PUBLIC aliases
        whose IMMEDIATE target was Oracle-maintained, purely to bound a quadratic walk, and gave
        up a chain that leaves Oracle's schemas on a later hop. A fuzz over random graphs found
        68 differences between the two and every one was that gain.

        Termination is `seen` over a set of visited objects, not a hop count. A planted `loop_a -> loop_b -> loop_a` raised
        ORA-32044 from the recursive query, so a cycle is a real shape; a depth cap on top of
        `seen` adds nothing to termination and silently drops a chain longer than the cap.

        A synonym over a DB-LINK object has no local `all_objects` row, so no walk from here
        can tell whether it ends at a function -- and it is REPORTED for exactly that reason
        rather than dropped. Skipping it was a measured fail-open (M102): a call through such
        an alias returned an ungranted SSN and `decide` approved the statement.
        """
        edges: dict[tuple[str, str], tuple[str, str]] = {}
        over_a_link: set[tuple[str, str]] = set()
        for owner, name, target_owner, target_name, db_link in self._rows(
            "SELECT lower(s.owner), lower(s.synonym_name), "
            "       lower(s.table_owner), lower(s.table_name), s.db_link FROM all_synonyms s"
        ):
            # A DB-LINK target cannot be resolved from here at all, so the alias is assumed to
            # reach a function rather than dropped (M102). Dropping it was a MEASURED fail-open:
            # `CREATE SYNONYM add_days FOR m102_udf@zz_loop` over a function reading an
            # ungranted table, `SELECT add_days(1)` returned the SSN through the link, and
            # `decide` APPROVED `SELECT add_days(1) FROM m102_claim` because the alias was
            # missing from `names`.
            #
            # BOTH remote shapes are wrong to follow, which is why this branches on the link and
            # not on the NULL owner it used to test. `ADD_DAYS -> (None, M102_UDF, ZZ_LOOP)` is
            # the unqualified one and has no owner to join on; `ZZ_LINK_SYN2 -> (APPUSER,
            # WEST_F, ZZ_LOOP)` names an owner, and joining on it resolves against the LOCAL
            # `appuser.west_f`, a different object in a different database -- right by accident
            # when a local function shares the name, fail-open when none does.
            #
            # WHAT REPORTING AN ALIAS COSTS, in the two places `names` is read:
            #
            #   the refusal -- `may_shadow_a_builtin` is inert on this adapter
            #     (`binder_prefers_builtins`), so `called & names` is the only rule left, and
            #     `called` is a LEXICAL scan for an identifier before an open paren. So
            #     `SELECT * FROM orders` through `PUBLIC orders FOR orders@ERP_LINK` is still
            #     answered and only `orders(...)` is refused. A scan, not a parse: it
            #     over-collects on purpose, and `FROM customer AS c(id, name)` yields `c`.
            #
            #   the label -- `calls_are_confirmable` is `not names`, so ONE alias downgrades
            #     the whole source. Measured on that same `orders` row against a schema
            #     defining nothing of its own: `user_functions() == ['orders']` and the flag
            #     goes True -> False, `completeness='unknown'` on every answer carrying a call.
            #     A link to a TABLE is the common enterprise shape and nothing local tells it
            #     from a link to a function, so this is the issue-#5 blanket downgrade that the
            #     PUBLIC/Oracle-maintained filter below exists to avoid, by another route.
            #
            # The second is the one worth arguing about, and it is taken anyway: a downgraded
            # label is honest about an answer, and the alternative is handing back an ungranted
            # SSN. Narrowing it means a name that must refuse without being evidence the source
            # defines functions -- a third category `FunctionInventory` does not model, filed
            # as M104 so this stays the security fix.
            if db_link is not None:
                over_a_link.add((owner, name))
                continue
            # Defensive, and no longer the DB-link case above. A NULL here used to reach
            # `sorted()` and raise TypeError, which `inventory_from` turns into `unavailable`
            # and the guard turns into refusing EVERY statement on the source -- one such row
            # anywhere in the dictionary would have stopped the whole deployment.
            if target_owner is None or target_name is None:
                continue
            edges[(owner, name)] = (target_owner, target_name)
        if not edges and not over_a_link:
            return set()
        oracle_owned = {
            r[0] for r in self._rows(
                "SELECT lower(username) FROM all_users WHERE oracle_maintained = 'Y'")
        }
        mine = self._schema.lower()

        # Forward and backward both, because each answers a question the other cannot.
        nodes = set(edges) | set(edges.values())
        forward: dict[tuple[str, str], list[tuple[str, str]]] = {}
        backwards: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for at in nodes:
            for step in (edges.get(at), edges.get(("public", at[1]))):
                if step is not None:
                    forward.setdefault(at, []).append(step)
                    backwards.setdefault(step, []).append(at)

        def spread(seeds, adjacency) -> set[tuple[str, str]]:
            """Everything `seeds` reaches. `seen` is what makes a cycle terminate."""
            seen = set(seeds)
            stack = list(seen)
            while stack:
                for nxt in adjacency.get(stack.pop(), ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        # WHICH OWNERS TO ASK ABOUT, and this is where the cost lives rather than in the walk.
        # Taking every owner named by an edge pulls SYS in, and `all_objects` for SYS alone is
        # 107 ms against 3.5 ms for the app schema -- `user_functions` went from 30 ms to 157 ms
        # that way, which is the same regression the removed prefilter had been hiding, moved
        # from the walk into the read.
        #
        # Only two sets can matter. What THIS schema's aliases reach, which is a small forward
        # walk. And every non-Oracle-maintained owner named anywhere, since a PUBLIC alias counts
        # only where its chain ends outside Oracle's own schemas, so SYS functions can never
        # decide that arm.
        mine_reaches = spread({t for (o, _), t in edges.items() if o == mine}, forward)
        owners = sorted(
            {owner for owner, _ in mine_reaches}
            | {owner for owner, _ in edges.values() if owner not in oracle_owned}
        )
        # No early-out on an empty `owners`: the chunked loop below simply does not run, and
        # the link aliases still have to be reported without any `all_objects` read at all.
        # Chunked, because an Oracle IN list is capped at 1,000 items (ORA-01795) and this one
        # grows with every non-Oracle owner any synonym targets -- a schema-per-tenant instance
        # passes that mark. 900 leaves room without needing a second measurement.
        functions: set[tuple[str, str]] = set()
        for start in range(0, len(owners), 900):
            chunk = owners[start:start + 900]
            binds = {f"o{i}": owner for i, owner in enumerate(chunk)}
            placeholders = ", ".join(f":{k}" for k in binds)
            functions.update(
                (owner, name)
                for owner, name in self._rows(
                    "SELECT lower(owner), lower(object_name) FROM all_objects "
                    "WHERE object_type IN ('FUNCTION', 'PACKAGE') "
                    f"  AND lower(owner) IN ({placeholders})",
                    **binds,
                )
            )
        # BACKWARDS from the functions, instead of forwards from every alias. An earlier version
        # stored a reachable set per candidate and was quadratic: measured on a synthetic
        # chain, 3.6 ms at 200 synonyms, 63.2 at 800 and 239.3 at 1,600, which is the wrong shape
        # for the dictionaries M101 is about. Each node is visited once here.
        #
        # A link alias seeds the walk as ITSELF, since there is no local node standing for its
        # target: unprovable, therefore assumed to reach a function. `via_public_link` is the
        # same assumption one hop earlier -- `a -> b` where this schema holds no `b` and the
        # PUBLIC `b` is a link alias, which the `forward`/`backwards` fallback below cannot
        # carry because a link alias contributes no edge to follow.
        unprovable = over_a_link | {at for at in nodes if ("public", at[1]) in over_a_link}
        reaches_any = spread((functions & nodes) | unprovable, backwards)
        # PUBLIC aliases count only where the chain ends outside Oracle's own schemas, so that
        # question is asked of its own smaller target set rather than filtered afterwards. A
        # link alias counts for this arm too: a remote target cannot sit in an Oracle-maintained
        # schema of THIS database, which is the only thing the arm excludes.
        reaches_free = spread(
            {f for f in functions & nodes if f[0] not in oracle_owned} | unprovable, backwards)

        # Keyed on the ALIAS rather than on its target, because a link alias has no target node
        # to test. NOT equivalent for every other row, which an earlier version of this comment
        # claimed: `backwards` also carries the PUBLIC same-bare-name fallback, so an alias can
        # land in `reaches_any` by a link its own target does not have. Measured on
        # `[("app","d","sys","c"), ("public","d","sys","sys_udf")]` with `sys_udf` a function --
        # target-keyed returns `set()`, alias-keyed returns `{"d"}`, because `app.d` collects
        # the PUBLIC `d`'s edge even though Oracle would resolve `d` through the private synonym
        # and never consult the PUBLIC one.
        #
        # Over-reporting, so fail-closed, and a 6,000-graph randomised comparison found 9
        # divergences all in that direction. Kept because the alternative is special-casing the
        # link aliases at the return, and a second code path for the shape this function most
        # recently got wrong is the worse trade.
        return {
            name
            for owner, name in set(edges) | over_a_link
            if (owner == mine and (owner, name) in reaches_any)
            or (owner == "public" and (owner, name) in reaches_free)
        }

    def virtual_columns(self) -> list[tuple[str, str, str]]:
        """(table, column, expression) for every VIRTUAL column in this schema.

        A virtual column runs its expression on read, so it is a route to user code the
        statement never names. MEASURED: `leaked AS (vc_udf(id))` over a function reading
        another table, and `SELECT id, leaked FROM vc_t` returned an SSN with no function named
        anywhere in the SQL (M100). `check_unmodelled_calls` walks call nodes, `called_names`
        reads the rendered text, and `check_access` sees a column the snapshot lists -- all
        three pass it.

        The EXPRESSION comes back, not a verdict, because which of them is opaque is a question
        about this source's function inventory and that belongs in the decider rather than here.
        `ALL_TAB_COLS.DATA_DEFAULT` is a LONG and arrives quoted: `"APPUSER"."VC_UDF"("ID")` for
        the dangerous one, `"QTY"*"PRICE"` for arithmetic, which is exactly the distinction the
        caller needs to draw.

        0.5 ms on this container, and scoped to `self._schema` for the reason `list_columns` is.

        A failure RAISES, like its siblings.
        """
        return [
            (table, column, expression or "")
            for table, column, expression in self._rows(
                "SELECT lower(table_name), lower(column_name), data_default "
                "FROM all_tab_cols WHERE owner = :owner AND virtual_column = 'YES'",
                owner=self._schema,
            )
        ]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        """Declared FKs: (from_table, from_col, to_table, to_col, constraint_id).

        `constraint_id` is the constraint NAME, which groups a composite FK's columns and keeps
        two independent FKs to the same parent distinct -- the same contract SQLite's adapter
        gets from its per-constraint `id`. Joined on `r_constraint_name` and ordered by
        `position` so a composite's columns pair up correctly rather than by luck of row order.
        """
        return [
            (r[0], r[1], r[2], r[3], r[4])
            for r in self._rows(
                "SELECT c.table_name, cc.column_name, rc.table_name, rcc.column_name, c.constraint_name "
                "FROM all_constraints c "
                "JOIN all_cons_columns cc "
                "  ON cc.owner = c.owner AND cc.constraint_name = c.constraint_name "
                "JOIN all_constraints rc "
                "  ON rc.owner = c.r_owner AND rc.constraint_name = c.r_constraint_name "
                "JOIN all_cons_columns rcc "
                "  ON rcc.owner = rc.owner AND rcc.constraint_name = rc.constraint_name "
                " AND rcc.position = cc.position "
                "WHERE c.owner = :owner AND c.constraint_type = 'R' "
                "ORDER BY c.table_name, c.constraint_name, cc.position",
                owner=self._schema,
            )
        ]

    def view_definitions(self) -> list[tuple[str, str, str]]:
        """(view, body, dialect). A failure RAISES, and that is load-bearing.

        `enrich_structural` wraps this call precisely so a source that will not answer records
        `discover:views` as failed, and the governance layer can tell "this source has no views"
        from "nobody could ask" (M52). The DuckDB adapter used to swallow every exception here and
        return `[]`, which meant the pipeline's handler could never fire and an unreadable
        inventory was reported as a complete empty one. Do not add a bare except to this method.

        `ALL_VIEWS.TEXT` is a LONG column, and LONG cannot be filtered, joined, or read after
        another LONG in the same fetch. `oracledb` returns it as a string when fetched directly,
        which is why this selects it alone and orders by a non-LONG column.
        """
        rows = self._rows(
            "SELECT view_name, text FROM all_views WHERE owner = :owner ORDER BY view_name",
            owner=self._schema,
        )
        return [(r[0], r[1], "oracle") for r in rows if r[1]]

    # -- execution ---------------------------------------------------------------------------

    def execute(self, sql: str) -> list[tuple]:
        """Rows for a query; `[]` for a statement that returns none, which the WRITE PATH needs.

        `Runtime.write` runs the approved mutation through this method, and `oracledb` raises
        DPY-1003 from `fetchall()` on a statement that returns no rows -- where sqlite3 answers
        `[]` and DuckDB answers a row count. Without the `description` check an Oracle write
        crashes inside the adapter instead of completing. Measured, not assumed.

        And it COMMITS when not read-only. `oracledb` does not autocommit (`PostgresAdapter`
        passes `autocommit=True` for the same reason), so an approved write would execute, return
        cleanly, and be rolled back on close -- data loss with a success reported to the caller.
        Under `read_only` there is nothing to commit: the transaction is read-only by
        construction, and committing would end it.
        """
        self._refuse_unless_read(sql)

        def _run(cur):
            cur.execute(sql)
            if cur.description is not None:
                rows = cur.fetchall()
            else:
                # A DML statement: report the affected-row count, which is what `Runtime.write`
                # reads to fill `rows_affected` (`result[0][0]` when it is a single int). Returning
                # `[]` here parsed cleanly and made every Oracle write report no row count at all,
                # silently -- the DuckDB path this adapter is measured against answers a count.
                # `rowcount` is read BEFORE the commit and verified to survive it.
                rows = [(cur.rowcount,)] if cur.rowcount is not None and cur.rowcount >= 0 else []
            if not self._read_only:
                # The LEASED connection, not an adapter-wide one -- `cur.connection` is the only
                # correct referent once connections are per-operation, and committing anything
                # else would commit a different caller's transaction.
                cur.connection.commit()
            return rows

        # Retried only under read_only, where the DDL race lives; a write is never re-executed by
        # `_with_cursor`, which is what makes retrying safe to apply on this shared method.
        return self._with_cursor(_run)

    # Statements `validate` will hand to Oracle's parser. Default-deny for a second, sharper
    # reason than `_READ_LEADERS`: **`cursor.parse()` EXECUTES DDL.** Measured -- 
    # `parse("CREATE TABLE parse_ddl_probe (id NUMBER)")` returned without error and the table
    # existed afterwards, and it did so INSIDE a read-only transaction, because the DDL's implicit
    # commit ends that transaction first. A validation seam that runs what it is asked to check is
    # worse than no validation, so the allowlist is what stands between the two.
    _VALIDATABLE = frozenset({"SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "MERGE"})

    def validate(self, sql: str) -> None:
        """Prove a statement is executable WITHOUT executing it. Raises if it is not.

        The deciders prove every plan against the source before approving it, because a snapshot
        can be stale and only the source knows the truth. They did that by issuing
        `EXPLAIN <sql>`, which is Postgres and DuckDB syntax and **is not Oracle's**: measured,
        `EXPLAIN SELECT id FROM t` is ORA-02000, "missing PLAN keyword". Every Oracle plan would
        have been refused as EXPLAIN_FAILED before execution -- the adapter's own tests never
        traversed the deciders, so nothing caught it.

        Oracle's documented form, `EXPLAIN PLAN FOR <sql>`, is not the fix either: it INSERTS the
        plan into PLAN_TABLE, so on the read-only adapter that does the proving it fails with
        ORA-01456, "may not perform insert, delete, update operation inside a READ ONLY
        transaction". Measured both ways -- it succeeds on a writable connection and fails on the
        read one, which is the one that needs it.

        `cursor.parse()` is the mechanism that fits: it validates syntax, column existence and
        object existence -- ORA-00936, ORA-00904, ORA-00942 respectively -- inside a read-only
        transaction, and measured, parsing `INSERT INTO t VALUES (99)` left the row count
        unchanged. Its one sharp edge is DDL, which it runs; `_VALIDATABLE` is the guard.
        """
        leader = _leading_keyword(sql)
        if leader not in self._VALIDATABLE:
            raise ReadOnlyViolation(
                f"refusing to validate a {leader or 'statement'} statement: Oracle's parser "
                f"EXECUTES DDL, so validation is only safe for {sorted(self._VALIDATABLE)}"
            )
        self._refuse_unless_read(sql)

        def _parse(cur):
            cur.parse(sql)

        # Probe-bounded. A parse is the engine asking the source about a statement, and a hard
        # parse that stalls used to block every other caller; now it bounds itself and holds only
        # its own leased connection while it does.
        self._with_cursor(_parse, self._probe_timeout_ms)

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """Run a query and return Arrow, bounding it by the driver's own call timeout.

        `Connection.call_timeout` is milliseconds and is the driver's supported way to bound a
        round trip; a `threading.Timer` calling `cancel()` -- the shape the SQLite and DuckDB
        adapters use -- is not equivalent here, because it races the fetch rather than the call.
        The lease clears the timeout on release, so one bounded query cannot bound the next.
        """
        # ONE ordering still matters, and it is why the timeout is applied inside `_fetch` rather
        # than handed to `_lease`: `_cursor()` issues its own `SET TRANSACTION READ ONLY` round
        # trip, and a tight query timeout applied before that would bound the SAFEGUARD's setup
        # instead of the query -- on the first attempt and on every DDL-race retry alike.
        #
        # What is gone is the save/restore of a previous value. That existed because one
        # connection was shared for the adapter's lifetime, so a leaked timeout bounded every
        # later call and a `previous` read outside the lock could capture another thread's
        # tightened value and write it back as the original. A leased connection is held by one
        # caller and cleared on release, so there is no previous value to preserve and no other
        # thread to race -- which is the concurrency bug removed rather than guarded.
        self._refuse_unless_read(sql)

        def _fetch(cur):
            if timeout_s is not None:
                cur.connection.call_timeout = int(timeout_s * 1000)
            cur.execute(sql)
            names = [d[0] for d in cur.description] if cur.description else []
            return names, cur.fetchall()

        try:
            names, rows = self._with_cursor(_fetch)
        except self._oracledb.DatabaseError as exc:
            # DPY-4011/ORA-03156 surface a cancelled call; report it as a timeout rather than
            # letting a driver code reach the caller as an opaque source error.
            if timeout_s is not None and _is_timeout(exc):
                raise RuntimeError(f"query timed out after {timeout_s}s") from exc
            raise

        # Column-major, by position, so duplicate output names survive -- a dict would collapse
        # `SELECT a AS x, b AS x` into one column and silently change the result.
        columns = list(zip(*rows)) if rows else [() for _ in names]
        arrays = [pa.array(list(col)) for col in columns]
        return pa.Table.from_arrays(arrays, names=names)


    # -- governance ---------------------------------------------------------------------------

    # The target gap between two UNVERIFIED lines. What is DELIVERED is this rounded UP to the
    # next whole probe cadence, because a line can only be emitted where a probe runs -- so with
    # `read_only_ttl_s` at 2400s against this 3600s cap the widest silence is 4800s, two cadences,
    # not 3600s and not one cadence. "Whichever is longer" was the earlier statement of this and
    # it is wrong wherever the cadence does not divide the cap. Doubling alone goes quiet for a
    # day after a day, which is the silence the cap does prevent, within that rounding.
    _RO_UNVERIFIED_MAX_GAP_S = 3600.0

    # `SELECT ... FOR UPDATE` is the only candidate of five that FLIPPED with the open mode, and
    # it was measured against both -- READ WRITE and READ ONLY -- because a check that only ever
    # runs against one case is not measured, which is the mistake this method's privilege query
    # made four times. `WHERE 1=0` is what makes it safe to run at boot against a customer table:
    # it matches no rows, so it takes no row lock, and it still raises. The others and why they
    # cannot serve: `SYS_CONTEXT('USERENV','DATABASE_ROLE')` reports PRIMARY for a read-only PDB,
    # seeing only a Data Guard standby; a harmless `DELETE ... WHERE 1=0` is refused for PRIVILEGE
    # (ORA-41900) in BOTH modes, because Oracle checks privilege first; `LOCK TABLE` and
    # `SET TRANSACTION READ WRITE` both SUCCEED on a read-only database.
    _READ_ONLY_DB_PROBE = "SELECT 1 FROM dual WHERE 1=0 FOR UPDATE"

    def _database_refuses_writes(self) -> bool:
        """Whether the DATABASE itself refuses every write, measured rather than inferred.

        Runs on the normal read path, inside `SET TRANSACTION READ ONLY` like every other read.
        That looked like it would defeat the probe -- a write inside a read-only transaction is
        ORA-01456, which would be the answer in both modes and so distinguish nothing. Measured
        instead of assumed, and it is not: Oracle checks the open mode FIRST, so the two modes
        return different codes (ORA-01456 writable, ORA-16000 read-only) and the probe never has to
        leave the read-only transaction to tell them apart.

        Any other error answers False. This is a boot advisory; a probe that cannot run must not
        become an assurance, and `unverifiable` below is the honest verdict when it does not.
        """
        try:
            self._rows(self._READ_ONLY_DB_PROBE)
        except self._oracledb.DatabaseError as exc:
            return _is_read_only_database(exc)
        except Exception:
            return False
        return False

    def assert_read_only(self) -> tuple[str, str]:
        """Whether `read_only` rests on this adapter's gate alone, or on privilege as well.

        `_refuse_unless_read` stops direct DML and DDL, and it is the ONLY thing that stops DDL --
        Oracle's `SET TRANSACTION READ ONLY` does not. What it cannot stop is a write reached
        through PL/SQL: an `AUTONOMOUS_TRANSACTION` function runs in its own transaction, and a
        view can hide the call so that the statement text names nothing. Both measured.

        So the honest question at boot is not "is the gate on" but **"could this connection write
        if something got past the gate"**, and that is answerable: a principal that owns the schema
        can always write it, and object or ANY-table grants do the same -- **including grants that
        arrive through a ROLE, which `user_tab_privs` does not show at all**. When the answer is
        yes,
        `read_only` is one bug away from not holding, and narrowing the principal is the deployment
        response -- **though it is a reduction, not a fix, and this docstring said otherwise until
        it was measured.** A principal holding SELECT on one view and nothing else caused a row to
        be inserted, because a view resolves its references with the VIEW OWNER's rights. Dropping
        to SELECT removes every DIRECT write; it does not make the connection unable to cause one.

        Three verdicts. `gate_only` and `unverifiable` both come from auditing the CALLER, which
        cannot establish read-onlyness at all: a view resolves its references with the VIEW OWNER's
        rights, and a principal holding SELECT on one view and nothing else read it and a row was
        inserted, holding no EXECUTE, no DML and owning nothing.

        `constrained` does not audit the caller. It asks whether the DATABASE is open read-only,
        which is a different question and the only one whose answer survives the view. This method
        previously carried a paragraph explaining that no third verdict was reachable; that was
        drawn from two attempts which were BOTH privilege queries, in a method whose own evidence
        is that privilege is the wrong instrument. The generalisation was the error, not the two
        measurements it came from.

        Reports; never refuses. Refusing here would break every deployment that reads as its own
        schema owner, which is most of them, over a risk that requires hostile PL/SQL to realise.
        """
        if not self._read_only:
            return ("writable", "this adapter is not read-only, so the question does not apply")

        # Asked BEFORE any privilege query, because it answers a strictly stronger question and the
        # privilege answer is irrelevant once it holds. M66 deleted `constrained` as unreachable
        # after two attempts, and the conclusion drawn was that no verdict above `unverifiable`
        # exists for a minimal read principal. That conclusion was too broad: both attempts were
        # PRIVILEGE queries, and the row's own evidence is that privilege is the wrong instrument.
        # The open mode is not a privilege question. Measured on the shape M66 could not close --
        # a definer-rights view over an AUTONOMOUS_TRANSACTION function, which writes for a caller
        # holding SELECT and nothing else -- the write is refused with ORA-16000 on a read-only
        # database, while plain SELECT keeps working.
        if self._database_refuses_writes():
            # Recorded so `_recheck_read_only` knows there is an assurance to lose. Without this
            # the verdict is a string a human read once.
            self._record_constrained(time.monotonic())
            return ("constrained", (
                "this database is open READ ONLY, so it refuses every write from every principal, "
                "including the one path this adapter's gate cannot see: a SELECT that reaches an "
                "AUTONOMOUS_TRANSACTION function through a view. Measured on exactly that shape -- "
                "refused with ORA-16000 while plain reads kept working. This is the only "
                "deployment in which read_only is enforced by the database rather than by this "
                "process. Note the scope: it was true when this connection opened, and it is re-probed "
                "on a bounded TTL thereafter -- reopening the database READ WRITE ends it and "
                "is reported as a LAPSED warning within that window, which is the exposure"))
        # Object privileges are read through `ALL_TAB_PRIVS`, across THREE grantee routes, and
        # scoped to the governed schema. Every part of that is a measured correction of a wrong
        # earlier version.
        #
        #   ROUTE. `USER_TAB_PRIVS` does not show role-granted privileges -- 0 rows even
        #   unfiltered for a principal that could insert -- and shows nothing for a grant made to
        #   PUBLIC either, which a principal can also use. Both measured, both reported
        #   `constrained` while the principal wrote.
        #
        #   PRIVILEGE. **EXECUTE belongs in this list and its absence was the worst hole**, because
        #   it is the threat itself: a principal holding EXECUTE on an AUTONOMOUS_TRANSACTION
        #   function owns nothing, holds no DML, and writes anyway. Measured -- verdict
        #   `constrained`, and `SELECT owner.f_writes FROM dual` inserted a row. The method's own
        #   docstring described that attack while the query was blind to it.
        #
        #   DROPPED OBJECTS. `BIN$...` is Oracle's recycle-bin naming, and a dropped table keeps
        #   its grants there. Measured: a SELECT-only principal reported `gate_only` on the
        #   strength of a PUBLIC INSERT held by a table that no longer exists. That is the failure
        #   that matters most for an advisory -- not a missed warning but an unearned one, because
        #   a verdict that cannot reach `constrained` is a warning nobody reads.
        #
        #   SCOPE. Restricted to `table_schema = :owner` because PUBLIC holds **1829** EXECUTE
        #   grants outside it on this instance alone -- DBMS_* and friends. Counting those makes
        #   every connection report `gate_only` forever, and a warning that is always on is a
        #   warning nobody reads. What matters is who can write the schema under governance.
        owned, direct, public, via_role, sysprivs = self._rows(
            "SELECT (SELECT count(*) FROM all_tables "
            "         WHERE owner = SYS_CONTEXT('USERENV','SESSION_USER')) AS owned, "
            "       (SELECT count(*) FROM all_tab_privs WHERE table_schema = :owner "
            "         AND privilege IN ('INSERT','UPDATE','DELETE','EXECUTE','ALTER') AND table_name NOT LIKE 'BIN$%' "
            "         AND grantee = SYS_CONTEXT('USERENV','SESSION_USER')) AS direct, "
            "       (SELECT count(*) FROM all_tab_privs WHERE table_schema = :owner "
            "         AND privilege IN ('INSERT','UPDATE','DELETE','EXECUTE','ALTER') AND table_name NOT LIKE 'BIN$%' "
            "         AND grantee = 'PUBLIC') AS public_grants, "
            "       (SELECT count(*) FROM all_tab_privs WHERE table_schema = :owner "
            "         AND privilege IN ('INSERT','UPDATE','DELETE','EXECUTE','ALTER') AND table_name NOT LIKE 'BIN$%' "
            "         AND grantee IN (SELECT role FROM session_roles)) AS via_role, "
            "       (SELECT count(*) FROM session_privs WHERE privilege IN "
            "         ('INSERT ANY TABLE','UPDATE ANY TABLE','DELETE ANY TABLE', "
            "          'CREATE ANY TABLE','DROP ANY TABLE','ALTER ANY TABLE','CREATE TABLE', "
            "          'CREATE PROCEDURE','CREATE ANY PROCEDURE','EXECUTE ANY PROCEDURE', "
            "          'CREATE ANY TRIGGER','CREATE JOB','CREATE ANY JOB')) AS sysprivs "
            "FROM dual", owner=self._schema
        )[0]
        granted = direct + public + via_role
        if owned or granted or sysprivs:
            return ("gate_only", (
                f"this read-only connection CAN write {self._schema}: it owns {owned} table(s), "
                f"holds {granted} write-shaped object privilege(s) there ({direct} direct, "
                f"{via_role} through a role, {public} granted to PUBLIC -- INSERT/UPDATE/DELETE/"
                f"ALTER, and EXECUTE, which is enough on its own), and holds {sysprivs} "
                "write-shaped system privilege(s). Direct writes are refused by this adapter, but "
                "a SELECT that reaches an AUTONOMOUS_TRANSACTION function -- possibly through a "
                "view, where the statement text names nothing -- is not something any statement "
                "check can see. Narrowing this principal to SELECT removes every direct write "
                "and is worth doing, but it does NOT make the connection unable to cause one: "
                "measured, SELECT on a single view was enough, because a view resolves its "
                "references with the VIEW OWNER's rights. Read-only here is the DATABASE's to "
                "enforce"))

        # No write-shaped privilege found. That is NOT "cannot write", and no further query about
        # THIS CALLER would make it one -- arrived at by deleting two attempts rather than by
        # reasoning. What that does not license is the stronger claim this comment used to make,
        # that no query of any kind could: the open-mode probe above reaches `constrained` and is
        # not a question about the caller at all.
        #
        # Measured: a principal holding SELECT on ONE VIEW and nothing else -- no EXECUTE, no DML,
        # owning nothing -- read that view and a row was inserted, because a view resolves its
        # references with the VIEW OWNER's rights and the function inside ran as the owner. So
        # auditing the CALLER cannot establish read-onlyness at any level of thoroughness.
        #
        # Two verdicts were tried HERE -- both about the caller -- and both removed as unreachable,
        # each verified against a live instance rather than argued. Both remain unreachable; what
        # was wrong was concluding from them that the property itself could not be established:
        #
        #   `constrained` ("no write path exists") -- needs the schema's source to prove absence,
        #   and `ALL_SOURCE` shows a non-owner nothing. DBA_SOURCE shows it but needs SELECT ANY
        #   DICTIONARY, the privilege this method's own advice says not to grant.
        #
        #   `gate_only` by CODE ("the schema declares AUTONOMOUS_TRANSACTION") -- same wall from
        #   the other side. Every principal that can see the source is already `gate_only` by
        #   privilege above: the owner (owns tables), an EXECUTE holder (EXECUTE is write-shaped).
        #   Enumerated over four principal shapes and none reached it. I added that branch in the
        #   same commit that deleted `constrained` for being unreachable.
        return ("unverifiable", (
            f"no write-shaped privilege on {self._schema} was found for this connection, and that "
            "is NOT the same as this connection being unable to write. A view resolves its "
            "references with the VIEW OWNER's rights, so SELECT on one view is enough to reach a "
            "subprogram that writes in its own transaction -- measured, a principal holding "
            "exactly that caused a row to be inserted. **Do not treat read_only=True as a "
            "guarantee here.** This is the best verdict a minimal read principal can produce, so "
            "it is not a misconfiguration to fix; the action is to enforce read-only in the "
            "DATABASE -- a genuinely restricted account, or an accepted and documented risk"))

    def assert_enforcing(self) -> tuple[str, str]:
        """Can this CONNECTION be trusted to have VPD applied to it? -> (verdict, reason).

        M57. Under the 2026-08-29 decision the database enforces row security, which makes the
        connecting principal the single point of failure: a connection privileged enough to bypass
        VPD means nothing enforces, silently, and no adapter previously inspected this at all.

        FOUR verdicts, and `bypassing` is the point. (This said "three" while listing four,
        which is the kind of drift a docstring acquires when a verdict is added to the list and
        not to the sentence above it.)

          `bypassing`    -- measured to bypass. **The caller WARNS; nothing refuses**, and this
                            line said "Refuse." until **M74**. That is a decision and not an
                            oversight: under the 2026-08-29 direction the ENGINE still enforces
                            RLS/CLS, so a bypassing connection today is one where mnemiq's own
                            filters are still in force, and refusing to boot would take a working
                            deployment down over a control that is not yet load-bearing. When
                            delegation lands -- **M57** -- this must become fail-closed. What DID
                            change: the verdict cannot be acknowledged away, because an
                            acknowledgement made while it is latent would still be set on
                            the day enforcement moves to the database and it stops being
                            latent.
          `unverifiable` -- no policy is attached to anything this connection can see, so there is
                            nothing to enforce and nothing to confirm.
          `partial`      -- some tables carry a policy and some do not. NOT an acceptance: the
                            ungoverned ones are ungoverned.
          `attached`     -- every table this connection can see carries an enabled policy, and
                            this principal holds no bypass privilege. **Still NOT a guarantee of
                            enforcement**, and the name says so deliberately.

        `partial` exists because counting policies per SCHEMA is a false positive. Measured: with
        a policy on one table and none on the table actually queried, a schema-level count
        reported `attached` while that query returned every row. That is the same flaw the
        Postgres check documents as `rls_tables` counting enabled-ness rather than coverage --
        which this adapter carried across without carrying the caveat, until a review asked.
        Coverage is per TABLE and the counts are in the reason string, because "3 of 47 governed"
        is the fact an operator needs and a single verdict word cannot hold it.

        Why `attached` is the strongest honest answer, and the reason this is not a port of the
        Postgres check: **a VPD policy function that returns NULL yields no predicate.** Measured
        on 23ai -- 2 of 2 rows returned, no privilege involved anywhere -- while `ALL_POLICIES`
        still reports that policy `enable = 'YES'`. Returning `''` behaves identically. So a
        catalog enumeration sees a healthy, attached, enabled policy that restricts nothing, and
        the obvious implementation -- count the policies, call it enforcing -- is wrong in a way
        that looks right. Establishing enforcement would mean executing the policy function, which
        is arbitrary PL/SQL belonging to the deployment, not to us.

        The privilege test is `SESSION_PRIVS`, not a role name. Measured across five principals:
        `EXEMPT ACCESS POLICY` present is EXACTLY the set that bypassed -- an explicit grantee and
        `SYS`, which holds it implicitly -- while the schema OWNER and a `DBA`-role user both had
        VPD APPLIED and both showed 0. So the Postgres intuitions do not transfer in either
        direction: Oracle's owner does not bypass where Postgres's does, and Oracle's DBA does not
        bypass despite documentation that reads as though it might.

        **A SYSDBA connection is caught by the same privilege test, and is not checked
        separately.** An earlier draft added an `ISDBA` branch as belt-and-braces; mutation showed
        removing it broke nothing, and probing showed why: any `AS SYSDBA` connection becomes
        `SESSION_USER = SYS` and holds EXEMPT ACCESS POLICY implicitly, so the privilege test
        always fires first -- including for a non-SYS user granted SYSDBA, measured as
        `exempt=1 isdba=TRUE session_user=SYS`. The branch was unreachable. An unreachable branch
        in a security check is worse than no branch: it reads as protection, no test can cover it,
        and it rots unnoticed. The FACT is kept here; the dead code is not.
        """
        row = self._rows(
            "SELECT (SELECT count(*) FROM session_privs "
            "         WHERE privilege = 'EXEMPT ACCESS POLICY') AS exempt, "
            "       (SELECT count(*) FROM all_tables WHERE owner = :owner) AS tables, "
            "       (SELECT count(DISTINCT p.object_name) FROM all_policies p "
            "         JOIN all_tables t ON t.owner = p.object_owner "
            "                          AND t.table_name = p.object_name "
            "         WHERE p.object_owner = :owner AND p.enable = 'YES' "
            "           AND p.sel = 'YES') AS governed "
            "FROM dual",
            owner=self._schema,
        )[0]
        exempt, tables, governed = int(row[0]), int(row[1]), int(row[2])

        if exempt:
            return ("bypassing", "this principal holds EXEMPT ACCESS POLICY, which exempts it "
                                 "from every VPD policy in the database. A SYSDBA connection "
                                 "reaches this branch too: it holds the privilege implicitly")
        caveat = ("Not proof of enforcement in any case: a policy function returning NULL, or "
                  "'', is reported enabled and restricts nothing.")
        if governed == 0:
            return ("unverifiable", f"no table owned by {self._schema} that this connection can "
                                    f"see carries an enabled VPD policy on SELECT "
                                    f"({tables} table(s) visible), so nothing is enforcing row "
                                    f"security here")
        if governed < tables:
            return ("partial", f"{governed} of {tables} tables owned by {self._schema} carry an "
                               f"enabled SELECT policy; the other {tables - governed} are "
                               f"ungoverned. {caveat}")
        return ("attached", f"all {tables} visible tables owned by {self._schema} carry an "
                            f"enabled SELECT policy and this principal holds no bypass "
                            f"privilege. {caveat}")


class ReadOnlyViolation(RuntimeError):
    """A read-only adapter was asked to run something that is not a read."""


_LEAD_COMMENT = re.compile(r"\A(?:\s+|--[^\n]*|/\*.*?\*/)+", re.S)
_LEAD_WORD = re.compile(r"\A[A-Za-z_][A-Za-z_0-9]*")


def _leading_keyword(sql: str) -> str:
    """The statement's first keyword, upper-cased, with leading comments stripped.

    Comments are stripped in a loop rather than once: `/* a */ -- b\n DROP` interleaves the two
    forms, and a single pass over either one leaves the other in front of the keyword. A prefix
    scan that a comment can hide DDL behind is not a gate.
    """
    prev = None
    while sql != prev:
        prev = sql
        sql = _LEAD_COMMENT.sub("", sql, count=1)
    m = _LEAD_WORD.match(sql)
    return m.group(0).upper() if m else ""


def _is_read_only_database(exc: Exception) -> bool:
    """ORA-16000: the DATABASE refused a write, as opposed to this connection lacking privilege.

    The distinction is the whole value. Every other instrument in this file asks what the CALLER
    may do, and M66 established that the caller's privileges cannot answer whether a read plane can
    cause a write -- a principal holding SELECT on one view and nothing else caused a row to be
    inserted, because a view resolves its references with the VIEW OWNER's rights. ORA-16000 is not
    a privilege verdict. It is the database saying that nothing writes here, whoever asks.

    Same code-first, string-fallback shape as `_is_ddl_race`, for the same reason.
    """
    err = exc.args[0] if exc.args else None
    if getattr(err, "code", None) == 16000 or getattr(err, "full_code", None) == "ORA-16000":
        return True
    return "ORA-16000" in str(exc)


def _is_ddl_race(exc: Exception) -> bool:
    """ORA-01466: this read-only transaction cannot read a definition that changed a moment ago.

    The CODE is checked first because it is exact. Measured against a live 23ai instance, the real
    exception is `oracledb.exceptions.DatabaseError` carrying `code=1466` and
    `full_code="ORA-01466"`; matching a substring of the rendered message would also be satisfied
    by a message that merely quotes the number. The string remains as a fallback for an error
    re-raised without the driver's own error object, which is the shape a wrapper would produce.
    """
    err = exc.args[0] if exc.args else None
    if getattr(err, "code", None) == 1466 or getattr(err, "full_code", None) == "ORA-01466":
        return True
    return "ora-01466" in str(exc).lower()


def _is_timeout(exc: Exception) -> bool:
    text = str(exc).lower()
    return "dpy-4011" in text or "ora-03156" in text or "timeout" in text

from __future__ import annotations

import re
import threading
import time
from typing import Any

import pyarrow as pa


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
                 read_only: bool = True) -> None:
        """`dsn` is an Easy Connect string or a TNS alias, e.g. `host:1521/FREEPDB1`.

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
        self._con = oracledb.connect(user=user, password=password, dsn=dsn)
        # ONE connection, shared by every caller, and `python-oracledb` in thin mode does not make
        # a connection safe for concurrent use. This adapter also mutates CONNECTION-wide state per
        # statement -- `rollback()` then `SET TRANSACTION READ ONLY` in `_cursor`, and
        # `call_timeout` in `execute_arrow` -- so two threads interleaving there is not a slow path
        # but a wrong one: one request can roll back another's transaction between its cursor
        # acquisition and its fetch, and a restored `call_timeout` can be another request's value.
        # The HTTP server builds ONE Runtime and FastAPI runs sync endpoints on a worker
        # threadpool, so this is the deployed shape, not a hypothetical.
        #
        # Serialised rather than pooled. A pool is the right answer for THROUGHPUT and is v2 work
        # with a real cost to size (see the attach-cost question in the governance decision); a
        # lock is the right answer for CORRECTNESS and is available now. RLock so a future method
        # that composes two of these does not deadlock on itself.
        self._lock = threading.RLock()
        self._schema = (schema or user).upper()
        self._read_only = read_only

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
        a._schema = schema.upper()
        a._read_only = read_only
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
    # The control that DOES close it is privilege -- a read connection that cannot write cannot be
    # made to write by any function it calls -- which is what `assert_read_only` reports on, and
    # which is a deployment property this adapter can observe but not impose.
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

    def _cursor(self):
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
            self._con.rollback()
            cur = self._con.cursor()
            cur.execute("SET TRANSACTION READ ONLY")
            return cur
        return self._con.cursor()

    # Measured against a live 23ai instance: a read of a table fails with ORA-01466 0.1s after
    # its CREATE and succeeds at 1.0s, while a control read on the SAME connection at the same
    # moment with no read-only transaction succeeds -- so the refusal comes from the safeguard's
    # pinned snapshot, not from the table. The window is sub-second; these bounds cover it with
    # room and still give up rather than loop.
    _DDL_RACE_BACKOFF = (0.25, 0.75, 1.5)

    def _with_cursor(self, work):
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
            cur = self._cursor()
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

        with self._lock:
            return self._with_cursor(_fetch)

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
                self._con.commit()
            return rows

        # Retried only under read_only, where the DDL race lives; a write is never re-executed by
        # `_with_cursor`, which is what makes retrying safe to apply on this shared method.
        with self._lock:
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

        with self._lock:
            self._with_cursor(_parse)

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """Run a query and return Arrow, bounding it by the driver's own call timeout.

        `Connection.call_timeout` is milliseconds and is the driver's supported way to bound a
        round trip; a `threading.Timer` calling `cancel()` -- the shape the SQLite and DuckDB
        adapters use -- is not equivalent here, because it races the fetch rather than the call.
        The timeout is restored afterwards so one bounded query cannot silently bound the next.
        """
        # Two orderings matter here and both are preserved by running the query inside `_fetch`.
        # The timeout must be set only once a cursor is in hand, because `_cursor()` issues its
        # own `SET TRANSACTION READ ONLY` round trip and a tight timeout would otherwise bound the
        # safeguard's setup rather than the query. And it must be restored on every path,
        # including one where `_cursor()` itself raises -- a tightened value that leaks would
        # silently bound every later call on this adapter-lifetime connection.
        self._refuse_unless_read(sql)
        with self._lock:
            return self._arrow_locked(sql, timeout_s)

    def _arrow_locked(self, sql: str, timeout_s: float | None) -> pa.Table:
        # Split out so the lock covers the CAPTURE of `call_timeout` as well as its restore.
        # Reading `previous` outside the lock lets another thread's tightened value be captured
        # and then written back as this call's "original", making the leak permanent.
        previous = self._con.call_timeout

        def _fetch(cur):
            # The timeout is set with the cursor already in hand, and restored before this
            # returns, so it bounds the QUERY and never `_cursor()`'s own SET TRANSACTION round
            # trip -- on the first attempt and on every retry alike. Restoring here rather than
            # only in the outer `finally` is what keeps a retry's setup unbounded.
            if timeout_s is not None:
                self._con.call_timeout = int(timeout_s * 1000)
            try:
                cur.execute(sql)
                names = [d[0] for d in cur.description] if cur.description else []
                return names, cur.fetchall()
            finally:
                self._con.call_timeout = previous

        try:
            names, rows = self._with_cursor(_fetch)
        except self._oracledb.DatabaseError as exc:
            # DPY-4011/ORA-03156 surface a cancelled call; report it as a timeout rather than
            # letting a driver code reach the caller as an opaque source error.
            if timeout_s is not None and _is_timeout(exc):
                raise RuntimeError(f"query timed out after {timeout_s}s") from exc
            raise
        finally:
            # Belt and braces: `_cursor()` can raise before `_fetch` is ever entered.
            self._con.call_timeout = previous

        # Column-major, by position, so duplicate output names survive -- a dict would collapse
        # `SELECT a AS x, b AS x` into one column and silently change the result.
        columns = list(zip(*rows)) if rows else [() for _ in names]
        arrays = [pa.array(list(col)) for col in columns]
        return pa.Table.from_arrays(arrays, names=names)


    # -- governance ---------------------------------------------------------------------------

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
        `read_only` is one bug away from not holding, and the fix is a deployment one -- connect
        the read plane as a principal with SELECT and nothing else.

        Two verdicts, `gate_only` and `unverifiable`, and the absence of a third is the finding:
        auditing the CALLER cannot establish read-onlyness at all, because a view resolves its
        references with the VIEW OWNER's rights. Measured -- a principal holding SELECT on one view
        and nothing else read it and a row was inserted, holding no EXECUTE, no DML and owning
        nothing.

        Reports; never refuses. Refusing here would break every deployment that reads as its own
        schema owner, which is most of them, over a risk that requires hostile PL/SQL to realise.
        """
        if not self._read_only:
            return ("writable", "this adapter is not read-only, so the question does not apply")
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
                "check can see. Connect the read plane as a principal holding SELECT and nothing "
                "else"))

        # No write-shaped privilege. That is NOT the same as "cannot cause a write", and the
        # difference is measured: a principal holding SELECT on a VIEW and nothing else -- no
        # EXECUTE, no DML, owning nothing -- read that view and a row was inserted. A view resolves
        # its references with the VIEW OWNER's rights, so the function inside ran as the owner and
        # the caller needed no privilege on it at all. Auditing the CALLER can therefore never
        # establish read-onlyness; the hazard lives in the schema's own code.
        #
        # So the absence of writing code has to be shown, not assumed -- and the source may not be
        # readable. Measured: the owner sees 15 lines of `ALL_SOURCE` for this schema and one
        # AUTONOMOUS_TRANSACTION hit, while a SELECT-only principal sees **zero lines** and
        # therefore zero hits. Reading that zero as "no writing code" is the absence/failure
        # collapse this codebase keeps closing, so the two are separate verdicts.
        #
        # DISTINCT program unit, not row: `ALL_SOURCE` holds one row per LINE, so a plain count is
        # a line count and "3 subprograms" for a package with one autonomous procedure spanning
        # three matching lines is simply false in operator-facing text. It counts units whose
        # source MENTIONS the pragma, which a comment also satisfies -- deliberately conservative,
        # since this decides whether to warn.
        lines, autonomous = self._rows(
            "SELECT (SELECT count(*) FROM all_source WHERE owner = :owner) AS lines, "
            "       (SELECT count(DISTINCT name || '.' || type) FROM all_source "
            "         WHERE owner = :owner "
            "         AND UPPER(text) LIKE '%AUTONOMOUS\\_TRANSACTION%' ESCAPE '\\') AS autonomous "
            "FROM dual", owner=self._schema
        )[0]
        if autonomous:
            return ("gate_only", (
                f"this connection holds no write privilege on {self._schema}, but "
                f"{autonomous} of its program unit(s) mention AUTONOMOUS_TRANSACTION in their "
                "source. "
                "Such a subprogram writes in its own transaction, and reaching it needs no "
                "privilege on it -- a view resolves references with the VIEW OWNER's rights, so "
                "SELECT on the view is enough. Read-only cannot be established by privilege here"))
        # There is deliberately NO "constrained" verdict, and it was removed rather than never
        # written: an earlier version returned one and it could not be reached by any principal
        # that would legitimately BE the read plane. Measured -- `ALL_SOURCE` shows a non-owner
        # zero lines of another schema's code, `DBA_SOURCE` shows 14 but needs SELECT ANY
        # DICTIONARY (the privilege this very method tells operators not to grant), and the one
        # non-owner who does see source is a holder of EXECUTE, which is write-shaped and lands in
        # `gate_only` above. So "no write path exists" is not establishable from here, and a
        # verdict asserting it would have been a branch that shipped unreachable -- the shape this
        # adapter already deleted an ISDBA check for.
        return ("unverifiable", (
            f"this connection holds no write privilege on {self._schema}, and {lines} line(s) of "
            "that schema's source are visible to it, so subprograms that write in their own "
            "transaction cannot be ruled out. Reaching one needs no privilege on it: a view "
            "resolves references with the VIEW OWNER's rights, so SELECT on a view is enough -- "
            "measured, a principal holding SELECT on one view and nothing else caused a row to be "
            "inserted. This is the expected verdict for a correctly minimal read principal, and it "
            "reports what cannot be seen rather than reading an unreadable catalog as empty"))

    def assert_enforcing(self) -> tuple[str, str]:
        """Can this CONNECTION be trusted to have VPD applied to it? -> (verdict, reason).

        M57. Under the 2026-08-29 decision the database enforces row security, which makes the
        connecting principal the single point of failure: a connection privileged enough to bypass
        VPD means nothing enforces, silently, and no adapter previously inspected this at all.

        Three verdicts, and the third is the point:

          `bypassing`    -- measured to bypass. Refuse.
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

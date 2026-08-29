from __future__ import annotations

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
        the decider approved the write anyway. A read plane must not depend on the decider being
        the only thing between it and a write.

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
        self._schema = (schema or user).upper()
        self._read_only = read_only

    # -- introspection -----------------------------------------------------------------------
    #
    # ALL_* rather than DBA_* throughout: DBA_* needs privileges an application account should
    # not hold, and asking for them would push the deployment toward exactly the over-privileged
    # role that makes VPD stop enforcing. ALL_* shows what this user can actually see, which is
    # also the honest scope for a governed engine.

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

        **A read-only transaction cannot read a table whose definition changed a moment ago** --
        ORA-01466, "table definition has changed", because `SET TRANSACTION READ ONLY` pins a
        read-consistent snapshot and DDL newer than it is unreadable. `_with_cursor` retries it;
        the reasoning is there, and an earlier version of this docstring argued the opposite on a
        mechanism that does not hold.
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
        def _fetch(cur):
            cur.execute(sql, **binds)
            return cur.fetchall()

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
        return self._with_cursor(_run)

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


def _is_ddl_race(exc: Exception) -> bool:
    """ORA-01466: this read-only transaction's snapshot is older than the table's definition."""
    return "ora-01466" in str(exc).lower()


def _is_timeout(exc: Exception) -> bool:
    text = str(exc).lower()
    return "dpy-4011" in text or "ora-03156" in text or "timeout" in text

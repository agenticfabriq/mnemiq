from __future__ import annotations

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
    ported from that shape would not look for it. `assert_enforcing` below is deliberately NOT
    written yet for that reason -- it needs measuring against a live instance, and guessing at it
    is how the Postgres version needed four drafts, each missing the next.
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

        **Known cost, measured: a read-only transaction cannot read a table whose definition
        changed in the same second** -- ORA-01466, "table definition has changed", because
        `SET TRANSACTION READ ONLY` pins a read-consistent snapshot and DDL newer than it is
        unreadable. Measured: the same read fails immediately after a CREATE, succeeds two seconds
        later, and succeeds immediately through a writable adapter. It is a real edge for
        "provision the schema then enrich at once", and it is accepted rather than retried around:
        a retry loop inside a safeguard is how a safeguard quietly stops being one, and the
        alternative -- no read-only enforcement at all -- is the thing M3 exists about.
        """
        if self._read_only:
            self._con.rollback()
            cur = self._con.cursor()
            cur.execute("SET TRANSACTION READ ONLY")
            return cur
        return self._con.cursor()

    def _rows(self, sql: str, **binds: Any) -> list[tuple]:
        cur = self._cursor()
        try:
            cur.execute(sql, **binds)
            return cur.fetchall()
        finally:
            cur.close()

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
        cur = self._cursor()
        try:
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
        finally:
            cur.close()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """Run a query and return Arrow, bounding it by the driver's own call timeout.

        `Connection.call_timeout` is milliseconds and is the driver's supported way to bound a
        round trip; a `threading.Timer` calling `cancel()` -- the shape the SQLite and DuckDB
        adapters use -- is not equivalent here, because it races the fetch rather than the call.
        The timeout is restored afterwards so one bounded query cannot silently bound the next.
        """
        # The cursor is acquired BEFORE the timeout is set, and the timeout is set INSIDE the
        # try. Both matter. `_cursor()` issues its own `SET TRANSACTION READ ONLY` round trip, so
        # setting the timeout first would bound the safeguard's own setup; and if `_cursor()`
        # raised between the mutation and the `try`, the `finally` would never run and the
        # tightened value would leak to every later call on this adapter-lifetime connection --
        # exactly what this docstring says it prevents.
        cur = self._cursor()
        previous = self._con.call_timeout
        try:
            if timeout_s is not None:
                self._con.call_timeout = int(timeout_s * 1000)
            cur.execute(sql)
            names = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        except self._oracledb.DatabaseError as exc:
            # DPY-4011/ORA-03156 surface a cancelled call; report it as a timeout rather than
            # letting a driver code reach the caller as an opaque source error.
            if timeout_s is not None and _is_timeout(exc):
                raise RuntimeError(f"query timed out after {timeout_s}s") from exc
            raise
        finally:
            cur.close()
            self._con.call_timeout = previous

        # Column-major, by position, so duplicate output names survive -- a dict would collapse
        # `SELECT a AS x, b AS x` into one column and silently change the result.
        columns = list(zip(*rows)) if rows else [() for _ in names]
        arrays = [pa.array(list(col)) for col in columns]
        return pa.Table.from_arrays(arrays, names=names)


def _is_timeout(exc: Exception) -> bool:
    text = str(exc).lower()
    return "dpy-4011" in text or "ora-03156" in text or "timeout" in text

"""An eval-only MySQL source, so mnemiq can answer BEAVER's questions on BEAVER's own engine.

Not a product adapter. MySQL is out of scope for mnemiq v1 (decided 2026-08-29: MySQL has no
native row-level security for the access policy to lean on), and nothing here is wired into
`adapters.resolve`. It exists so the gold SQL and mnemiq's SQL run on the same MySQL with the same
type rules: a DuckDB copy of BEAVER disagreed with MySQL on most of BEAVER's own gold queries,
because BEAVER stores many numbers as text and relies on MySQL's implicit casts.

Transport: DuckDB's `mysql` extension and its `mysql_query` passthrough, so every statement runs
on MySQL untranslated and comes back as Arrow, with no new driver in mnemiq's dependencies.

The server must run with two settings, and the adapter refuses to start without them:

* `ANSI_QUOTES` in `sql_mode`. Enrichment quotes identifiers with double quotes, which MySQL reads
  as strings without it. No BEAVER gold query uses a double quote.
* `lower_case_table_names=1`. BEAVER's gold spells table names in a different case than its dump
  creates them. MySQL accepts this setting only when the data directory is first created.

Connection: `MNEMIQ_MYSQL_DSN`, in the extension's `key=value` form without a database (for
example `host=127.0.0.1 port=3307 user=beaver password=...`). The adapter adds the database.
"""

from __future__ import annotations

import os
import re
import threading
import time

import duckdb
import pyarrow as pa

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.duckdb_extensions import connect_duckdb, load_extension
from mnemiq.execute.runner import ExecutionError
from mnemiq.redact import scrub_credentials

DSN_ENV = "MNEMIQ_MYSQL_DSN"


class MySQLUnavailable(ConnectionError):
    """MySQL did not answer: a connection that dropped and did not come back after a reattach and
    one more try. A `ConnectionError`, so the proof seam and the runner read it as an outage, not
    as a refusal the model should repair."""


class MySQLMisconfigured(RuntimeError):
    """The server runs without a setting this adapter depends on. See the module docstring."""


class MySQLRejected(ExecutionError):
    """A statement MySQL refused, in the runner's own form for a refusal, which the runner passes
    on as it stands. Left as the driver's error, it would meet the runner's last reading, which
    takes any text naming an interrupt for a timeout: MySQL quotes the statement it refuses, so a
    query it could not parse would go back to the model as one to make cheaper. This source types
    its own timeouts, cancels and outages, so what reaches here is a refusal."""

    def __str__(self) -> str:
        # The harness records `str()` of an answer that fails when the grader re-runs it; with
        # MySQL's words (scrubbed already) the results file says why.
        return self.repair_text


class _Run:
    """One statement's timeout state, so a late cancel cannot mark or kill the statement after it."""

    def __init__(self, mysql_id):
        self.mysql_id, self.timed_out, self.interrupted = mysql_id, False, False


def _without(text: str, *statements: str) -> str:
    """`text` with each statement's own text taken out. A source error can quote the statement,
    so a statement holding the words `lost connection` must not read as an outage, nor one
    holding `query execution was interrupted` as a cancel."""
    for statement in statements:
        if statement:
            text = text.replace(statement, "")
    return text


class MySQLSource:
    dialect = "mysql"  # the engine transpiles its plan to this before running it

    # How long an attach waits for the kill connection. The connection is optional (a timeout
    # reopens a missing one), and the same stalled handshake that can hang a reopen would
    # otherwise hold a working source behind the lock.
    _KILLER_OPEN_S = 10.0

    # How long a statement's caller waits for a cancel already under way. Bounded, because
    # reopening the kill connection during an outage can hang, and every thread behind `_lock`
    # waits with the caller.
    _CANCEL_WAIT_S = 15.0

    # How long a reattach keeps trying after the server went away.
    _REATTACH_S = 180.0

    def __init__(self, database: str, dsn: str | None = None, *, local_only: bool = False) -> None:
        dsn = dsn if dsn is not None else os.environ.get(DSN_ENV, "")
        if not dsn.strip():
            raise MySQLMisconfigured(
                f"set {DSN_ENV} to the MySQL connection, e.g. "
                "'host=127.0.0.1 port=3307 user=beaver password=...'")
        if re.search(r"(?:^|\s)(?:database|db)\s*=", dsn):
            raise MySQLMisconfigured(
                f"{DSN_ENV} names a database; leave it out, each BEAVER set picks its own")
        self._db = database
        self._dsn = f"{dsn.strip()} database={database}"
        self._local_only = local_only  # then the extension must already be installed
        self._lock = threading.Lock()  # one DuckDB connection; the engine may call from threads
        self._con = connect_duckdb(local_only=local_only)
        load_extension(self._con, "mysql", local_only=local_only)
        self._killer = None  # a second connection that only ever runs KILL QUERY (see _cancel)
        self._current = None  # the statement now running, as a _Run; a cancel acts on its own only
        self._retired = False  # a late KILL may still name this session; reattach before reuse
        # Held only for the moments a cancel and its statement's release must not interleave: a
        # cancel installs a kill connection or interrupts only while its statement is current, and
        # `_run` releases the statement (and retires its session) under the same lock, so neither
        # can land after the release.
        self._state = threading.Lock()
        failure = None
        try:
            self._attach()
        except duckdb.Error as exc:
            failure = self._unavailable(exc)
        if failure is not None:  # raised outside the handler: see _clean
            raise failure

    def _scrub(self, text: str) -> str:
        # The extension's connect error quotes the whole connection string, password included.
        return scrub_credentials(text)

    def _clean(self, exc: BaseException) -> BaseException:
        """`exc` rebuilt with the credential scrubbed from its text and from its cause's, and no
        context. Every error leaving this source goes through here, not only an outage's: a
        connect error can surface inside any statement, and the readings that put the driver's
        words in their own message (a cancel, the server's time limit) carry it on.

        The caller raises the copy outside its `except` block. Raised inside, the original rides
        along as `__context__`, which `from None` hides from a printed traceback but does not
        remove."""
        text = self._scrub(str(exc))
        try:
            clean = type(exc)(text)
        except Exception:  # noqa: BLE001 -- a type built otherwise; its words matter, not its type
            clean = RuntimeError(f"{type(exc).__name__}: {text}")
        if exc.__cause__ is not None:
            clean.__cause__ = self._clean(exc.__cause__)
        return clean

    def _unavailable(self, exc: BaseException) -> MySQLUnavailable:
        return MySQLUnavailable(self._scrub(str(exc)))

    def _attach_sql(self, alias: str, read_only: bool) -> str:
        mode = ", READ_ONLY" if read_only else ""
        return f"ATTACH '{self._dsn.replace(chr(39), chr(39) * 2)}' AS {alias} (TYPE mysql{mode})"

    def _attach(self) -> None:
        self._con.execute(self._attach_sql("src", read_only=True))
        # The extension keeps one MySQL connection per attach (measured: the same CONNECTION_ID()
        # across queries), so this id names the session every statement runs on, until a reattach
        # replaces it.
        row = self._con.execute(
            "SELECT * FROM mysql_query('src', "
            "'SELECT CONNECTION_ID(), @@SESSION.sql_mode, @@lower_case_table_names')").fetchone()
        self._mysql_id = row[0]
        self._check_settings(row[1], row[2])
        self._retired = False
        # Opened now, while MySQL is known to answer, so a timeout only sends KILL over a live
        # connection: the extension takes no connect_timeout (measured), and a connect made inside
        # a timeout could hang on a server that accepts connections but never answers.
        self._killer = self._open_killer_within(self._KILLER_OPEN_S)

    @staticmethod
    def _check_settings(sql_mode, lower_case) -> None:
        if isinstance(sql_mode, (bytes, bytearray)):
            sql_mode = sql_mode.decode("utf-8", errors="replace")
        missing = []
        if "ANSI_QUOTES" not in str(sql_mode or "").upper().split(","):
            missing.append("sql_mode must include ANSI_QUOTES")
        if str(lower_case) != "1":
            missing.append("lower_case_table_names must be 1 (set when the data directory is "
                           "first created)")
        if missing:
            raise MySQLMisconfigured("the MySQL server is not set up for BEAVER: "
                                     + "; ".join(missing))

    def _open_killer_within(self, wait_s: float):
        """A new kill connection, or None when opening it fails or takes longer than `wait_s`. A
        connection that finishes opening after that closes itself, so it never replaces one the
        adapter holds by then."""
        guard, box = threading.Lock(), {"killer": None, "abandoned": False}

        def open_one():
            try:
                killer = self._new_killer()
            except duckdb.Error:
                return
            with guard:
                if box["abandoned"]:
                    killer.close()
                else:
                    box["killer"] = killer
        opener = threading.Thread(target=open_one, daemon=True)
        opener.start()
        opener.join(wait_s)
        with guard:
            box["abandoned"] = True
            return box["killer"]

    def _new_killer(self):
        killer = connect_duckdb(local_only=self._local_only)
        failure = None
        try:
            killer.execute("LOAD mysql")
            # Not read-only: the extension refuses `mysql_execute` on a read-only attach.
            killer.execute(self._attach_sql("kill", read_only=False))
        except duckdb.Error as exc:
            killer.close()
            failure = self._clean(exc)
        if failure is not None:  # raised outside the handler: see _clean
            raise failure
        return killer

    # Messages that can mean the server went away, not that the statement was wrong; `_lost`
    # confirms with a probe before believing one. Docker Desktop
    # restarting took MySQL with it twice in one day: without a reconnect the attached connection
    # stays dead and every remaining question errors, which can void a run on a 30-second blip.
    # "Lost connection" is also what MySQL says when a heavy statement trips its own network
    # limits; such a statement is re-run once, which treats gold and answers alike. The last entry
    # is the adapter's own state after an outage longer than `_reattach` waits: `src` stays
    # detached, and the next statement reconnects.
    _GONE = ("can't connect", "gone away", "lost connection", "failed to connect",
             "connection refused", "tls/ssl", "shutdown in progress", "server shutdown",
             'failed to find attached database "src"')

    def _lost(self, exc: BaseException, *statements: str) -> bool:
        """Whether the connection is gone: the error reads like an outage AND the session no
        longer answers. The words alone do not decide, since MySQL quotes the part of a statement
        it cannot parse, and a statement can hold any words."""
        text = _without(str(exc), *statements).lower()
        return any(marker in text for marker in self._GONE) and not self._answers()

    def _answers(self) -> bool:
        try:
            self._con.execute("SELECT * FROM mysql_query('src', 'SELECT 1')").fetchall()
        except duckdb.Error:
            return False
        return True

    def _reattach(self) -> None:
        """Reconnect `src`, trying for `_REATTACH_S`. Raises `MySQLUnavailable` when it cannot."""
        self._drop_killer()  # the outage that ends `src` ends the kill connection too
        deadline = time.monotonic() + self._REATTACH_S
        while True:
            try:
                self._con.execute("DETACH DATABASE IF EXISTS src")
                self._attach()
                return
            except duckdb.Error as exc:
                if time.monotonic() > deadline:
                    raise self._unavailable(exc) from None
                time.sleep(5)

    # -- the passthrough ---------------------------------------------------------------------

    def _table(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """`sql` run on MySQL as it stands. Whatever it raises is rebuilt by `_clean` first."""
        try:
            return self._passthrough(sql, timeout_s)
        except Exception as exc:  # noqa: BLE001 -- rebuilt and raised, not handled
            failure = self._clean(exc)
        raise failure  # outside the handler: see _clean

    def _passthrough(self, sql: str, timeout_s: float | None) -> pa.Table:
        statement = sql.strip().rstrip(";")
        literal = "'" + statement.replace("'", "''") + "'"
        query = f"SELECT * FROM mysql_query('src', {literal})"
        with self._lock:
            if self._retired:
                self._reattach()  # see _run: the old session may yet receive a stalled KILL
            try:
                return self._run(query, timeout_s, statement)
            except duckdb.Error as exc:
                if not self._lost(exc, statement, literal):
                    raise  # a statement MySQL rejected: the caller's error, never retried
            self._reattach()  # outside any timer: the outage is not the statement's time
            try:
                return self._run(query, timeout_s, statement)
            except duckdb.Error as exc:
                if self._lost(exc, statement, literal):
                    raise self._unavailable(exc) from None
                raise

    def _cancel(self, run: _Run) -> None:
        """A timeout: stop the statement on MySQL, which ends DuckDB's wait for it.

        DuckDB's interrupt alone neither stops the statement nor frees this connection: the
        caller waits until MySQL finishes, while mysqld spills temporary files it has already
        deleted, which once filled a disk mid-run. The kill goes through a second connection
        because the passthrough connection is busy with the statement.

        DuckDB is interrupted only when the kill fails. An interrupt sent alongside the kill once
        landed in the tail of a call MySQL had already ended (a killed SLEEP ends with a row), so
        the following, unrelated statement raised "Interrupted!". DuckDB drops an interrupt sent
        while idle (measured); the danger is the tail.

        Everything it marks belongs to `run`. It checks that `run` is still the statement running
        before it sends the kill and before it interrupts, so a cancel `_run` stopped waiting for
        during a slow reopen reaches nothing; it kills the session `run` started on, and a KILL
        that stalls past that check is `_run`'s to contain (it retires the session). A connection
        it reopens or drops is its own: it installs one only while `run` is current, and a failure
        closes the one it used, never one a reattach opened since."""
        run.timed_out = True
        killer = self._killer
        try:
            if killer is None:  # dropped after a failure or a reattach
                killer = self._new_killer()
                with self._state:
                    current = self._current is run
                    if current:
                        self._killer = killer
                if not current:  # the reopen was slow and its statement has ended
                    killer.close()
                    return
            if self._current is not run:
                return
            killer.execute(f"CALL mysql_execute('kill', 'KILL QUERY {int(run.mysql_id)}')")
        except duckdb.Error:
            if killer is not None:  # a reopen that failed opened nothing to drop
                self._drop_killer(killer)  # the next timeout reopens it
            with self._state:
                if self._current is run:
                    run.interrupted = True
                    self._con.interrupt()  # DuckDB stops waiting; MySQL may run on

    def _drop_killer(self, killer=None) -> None:
        """Close `killer` (the adapter's own when not given), and forget it if the adapter still
        holds it."""
        with self._state:
            if killer is None:
                killer = self._killer
            if self._killer is killer:
                self._killer = None
        if killer is not None and hasattr(killer, "close"):
            try:
                killer.close()
            except duckdb.Error:
                pass

    def _run(self, query: str, timeout_s: float | None, statement: str = "") -> pa.Table:
        run = self._current = _Run(self._mysql_id)
        timer = threading.Timer(timeout_s, self._cancel, args=(run,)) if timeout_s else None
        if timer:
            timer.daemon = True  # a cancel stuck reaching a dead server must not keep us alive
            timer.start()
        try:
            table = self._con.execute(query).to_arrow_table()
            if run.timed_out:  # ended past its deadline, possibly by the kill
                raise QueryTimedOut(f"query timed out after {timeout_s}s")
            return table
        except duckdb.Error as exc:
            # A killed statement arrives as an I/O error ("Failed to fetch result row"), not as
            # DuckDB's own InterruptException, so the deadline decides, not the error's type.
            if run.timed_out:
                raise QueryTimedOut(f"query timed out after {timeout_s}s") from exc
            # MySQL's own words end the message, after any statement text it quotes, so only the
            # end is read: a statement holding these words cannot pass for a cancel.
            said = _without(str(exc), statement, query).strip().lower()
            if said.endswith("maximum statement execution time exceeded"):
                # The server's own bound (`max_execution_time`), not ours: still the cost.
                raise QueryTimedOut(f"the source ended the query at its time limit: {exc}") from exc
            if said.endswith("query execution was interrupted"):
                # Ended by a KILL this adapter did not send: an operator's or another session's.
                raise QueryCancelled(f"the source cancelled the query: {exc}") from exc
            raise
        finally:
            if timer:
                timer.cancel()
                # A cancel already started finishes before this returns and the lock frees: its
                # kill, or its interrupt, must not land on the next statement on this session.
                # Bounded for a cancel stuck reaching a dead server; once this returns it is no
                # longer current, so it interrupts nothing and sends no kill it has not sent.
                timer.join(self._CANCEL_WAIT_S)
            with self._state:
                if timer and timer.is_alive():
                    # It may have sent its KILL already, stalled, and deliver it yet: the next
                    # statement must not run on the session it names. `_table` reattaches first,
                    # which closes that session, so a late KILL names one that no longer exists.
                    # The kill connection is left to the stalled cancel, not closed under it.
                    self._retired, self._killer = True, None
                self._current = None
            if run.interrupted:
                try:  # absorb an interrupt that landed in the statement's tail
                    self._con.execute("SELECT 1")
                except duckdb.Error:
                    pass

    def _rows(self, sql: str) -> list[tuple]:
        """Rows for catalog and profiling reads. MySQL's catalog text (`column_type`, for one)
        arrives through the extension as bytes, and enrichment parses it as text, so bytes are
        decoded here. Answer and gold results go through `execute_arrow` untouched, the same path
        for both sides of a grade."""
        t = self._table(sql)
        cols = [t.column(i).to_pylist() for i in range(t.num_columns)]

        def text(v):
            return v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else v
        return [tuple(text(v) for v in row) for row in zip(*cols, strict=True)] if cols else []

    def _lit(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    # -- SourceAdapter -----------------------------------------------------------------------

    def introspect(self) -> list[str]:
        return [r[0] for r in self._rows(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = "
            f"{self._lit(self._db)} AND table_type = 'BASE TABLE' ORDER BY table_name")]

    def list_columns(self) -> list[tuple[str, str, str]]:
        return [(r[0], r[1], r[2] or "unknown") for r in self._rows(
            "SELECT table_name, column_name, column_type FROM information_schema.columns "
            f"WHERE table_schema = {self._lit(self._db)} ORDER BY table_name, ordinal_position")]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        return [(r[0], r[1], r[2], r[3], f"{r[0]}:{r[4]}") for r in self._rows(
            "SELECT table_name, column_name, referenced_table_name, referenced_column_name, "
            "constraint_name FROM information_schema.key_column_usage WHERE table_schema = "
            f"{self._lit(self._db)} AND referenced_table_name IS NOT NULL "
            "ORDER BY table_name, ordinal_position")]

    def view_definitions(self) -> list[tuple[str, str, str]]:
        return [(r[0], r[1], "mysql") for r in self._rows(
            "SELECT table_name, view_definition FROM information_schema.views WHERE table_schema = "
            f"{self._lit(self._db)} ORDER BY table_name") if r[1]]

    def user_functions(self) -> list[str]:
        # Asked of the source, live: BEAVER's three databases define none.
        return [r[0].lower() for r in self._rows(
            "SELECT routine_name FROM information_schema.routines WHERE routine_schema = "
            f"{self._lit(self._db)} AND routine_type = 'FUNCTION'")]

    def reachable_user_functions(self) -> list[str]:
        return self.user_functions()

    def virtual_columns(self) -> list[tuple[str, str, str]]:
        return [(r[0], r[1], r[2]) for r in self._rows(
            "SELECT table_name, column_name, generation_expression FROM information_schema.columns "
            f"WHERE table_schema = {self._lit(self._db)} AND extra LIKE '%GENERATED%'")]

    def validate(self, sql: str) -> None:
        """The proof `mnemiq.sql.prove` asks for, in place of `EXPLAIN`: the passthrough cannot
        run `EXPLAIN` (it must know a statement's result columns up front, and EXPLAIN gives
        none), and `mysql_execute` is refused on a read-only attach. Wrapping the plan as a
        zero-row derived table makes MySQL parse it and resolve every table and column without
        returning rows. A result with two columns of one name is refused here, as the
        passthrough refuses it when the statement runs."""
        self._table(f"SELECT * FROM ({sql.strip().rstrip(';')}) AS mnemiq_probe LIMIT 0",
                    timeout_s=30)

    def source_unavailable(self, exc: BaseException) -> bool:
        """The proof seam's question: did the source not answer, rather than answer no. Outages
        reach here as `MySQLUnavailable`, typed by `_table` only once the session stops answering
        too, so a statement whose error merely quotes `lost connection` is never read as one. A
        timeout at the proof is unanswered too; the runner asks its timeout first, so there it
        stays the query's cost."""
        return isinstance(exc, (ConnectionError, TimeoutError, QueryTimedOut,
                                duckdb.InterruptException))

    def execute(self, sql: str) -> list[tuple]:
        return self._rows(sql)

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        try:
            return self._table(sql, timeout_s)
        except duckdb.InterruptException:
            raise  # the runner reads DuckDB's interrupt by its type, as a timeout
        except duckdb.Error as exc:  # already scrubbed, with no context, by `_table`
            raise MySQLRejected("The source rejected this query.", source_detail=str(exc)) \
                from exc

    def close(self) -> None:
        self._drop_killer()
        self._con.close()

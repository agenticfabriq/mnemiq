"""The eval-only MySQL source: its timeout reaches MySQL, and a timeout, a cancel, an outage and a
refusal each reach the engine as what they are.

No MySQL in the first part: a fake DuckDB connection that blocks until the server kills its
statement or DuckDB interrupts it, and a fake side connection that records the KILL. The live
tests at the end need a MySQL server (BEAVER's, set up as docs/beaver.md says) and run only with
MNEMIQ_TEST_MYSQL_DSN set.
"""

import os
import re
import threading
import time

import duckdb
import pytest

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.eval import mysql_source
from mnemiq.eval.mysql_source import MySQLMisconfigured, MySQLSource, MySQLUnavailable
from mnemiq.execute.runner import (
    ExecutionError,
    QueryCancelledError,
    SourceUnavailableError,
)
from mnemiq.execute.runner import run as run_statement
from mnemiq.sql.prove import Proof, prove_outcome

KILL_455 = "CALL mysql_execute('kill', 'KILL QUERY 455')"


class Result:
    def to_arrow_table(self):
        return "rows"

    def fetchall(self):
        return [(1,)]


class Blocking:
    """Stands in for the DuckDB connection: `execute` waits until the server kills the statement
    (an I/O error, as the extension reports a killed statement) or DuckDB is interrupted, or
    returns after `finish_s`. An interrupt sent while no statement runs stays pending here until
    a statement consumes it: the worst case. Real DuckDB drops one sent while idle."""

    def __init__(self, finish_s=None):
        self.finish_s, self.killed, self.interrupted = finish_s, threading.Event(), threading.Event()
        self.interrupts = 0

    def execute(self, query):
        if query == "SELECT 1" and not self.interrupted.is_set():
            return Result()  # instant, as DuckDB's own SELECT 1 is (the interrupt absorber)
        deadline = time.monotonic() + (self.finish_s if self.finish_s is not None else 5)
        while time.monotonic() < deadline:
            if self.killed.is_set():
                raise duckdb.IOException(
                    "Failed to fetch result row for MySQL query: Query execution was interrupted")
            if self.interrupted.is_set():
                self.interrupted.clear()  # consumed by the statement it stops, or the next one
                raise duckdb.InterruptException("INTERRUPT Error: Interrupted!")
            time.sleep(0.01)
        return Result()

    def interrupt(self):
        self.interrupts += 1
        self.interrupted.set()


class Killer:
    """Records the KILL and, like MySQL, ends the statement it names."""

    def __init__(self, con):
        self.con, self.sent, self.closed = con, [], False

    def execute(self, sql):
        self.sent.append(sql)
        self.con.killed.set()

    def close(self):
        self.closed = True


class Fetched:
    def __init__(self, *values):
        self.values = values

    def fetchone(self):
        return self.values


class Session(Blocking):
    """Blocking, plus the MySQL session the attach holds, as the extension keeps one: DETACH ends
    it, ATTACH opens a new one with a new CONNECTION_ID, and a KILL QUERY ends a statement only on
    the session it names."""

    def __init__(self, finish_s=None, mysql_id=455, sql_mode="ANSI_QUOTES,STRICT_TRANS_TABLES",
                 lower_case=1):
        super().__init__(finish_s)
        self.id, self.attached, self.running = mysql_id, True, False
        self.sql_mode, self.lower_case = sql_mode, lower_case

    def execute(self, query):
        if query.startswith("DETACH"):
            self.attached = False
            return Result()
        if query.startswith("ATTACH"):
            self.attached, self.id = True, self.id + 1
            return Result()
        if "CONNECTION_ID()" in query:
            return Fetched(self.id, self.sql_mode, self.lower_case)
        if not self.attached:
            raise duckdb.CatalogException('Catalog Error: Failed to find attached database "src"')
        self.running = True
        try:
            return super().execute(query)
        finally:
            self.running = False

    def kill(self, sql):
        named = int(re.search(r"KILL QUERY (\d+)", sql).group(1))
        if named == self.id and self.attached and self.running:
            self.killed.set()


class LateKill:
    """A kill connection whose KILL reaches MySQL only after `hold_s`."""

    def __init__(self, con, hold_s):
        self.con, self.hold_s = con, hold_s

    def execute(self, sql):
        time.sleep(self.hold_s)
        self.con.kill(sql)


def adapter(con, mysql_id=455):
    a = object.__new__(MySQLSource)
    a._con, a._killer, a._mysql_id, a._lock, a._current = con, Killer(con), mysql_id, \
        threading.Lock(), None
    a._dsn, a._db, a._retired, a._state = "host=fake", "d", False, threading.Lock()
    a._local_only = False
    return a


# -- the timeout reaches MySQL ---------------------------------------------------------------


def test_a_timeout_kills_the_statement_on_mysql_and_raises_the_typed_timeout():
    a = adapter(Blocking())
    started = time.monotonic()
    with pytest.raises(QueryTimedOut, match="timed out after 0.2s"):
        a._run("SELECT * FROM mysql_query('src', 'SELECT SLEEP(60)')", 0.2)
    assert time.monotonic() - started < 2
    assert a._killer.sent == [KILL_455]
    # The kill ended it: DuckDB is not interrupted too, so no interrupt waits for the next one.
    assert a._con.interrupts == 0


def test_a_statement_that_ends_normally_after_its_timeout_still_reports_the_timeout():
    # MySQL ends a killed SLEEP with a row, not an error. The deadline passed, so it is a timeout,
    # and the next statement must not inherit an interrupt.
    con = Blocking(finish_s=0.4)
    a = adapter(con)

    class NoOp:
        sent = []

        def execute(self, sql):
            self.sent.append(sql)
    a._killer = NoOp()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    assert a._killer.sent and con.interrupts == 0


def test_a_statement_that_finishes_in_time_is_not_killed():
    a = adapter(Blocking(finish_s=0.05))
    assert a._run("SELECT 1", 2) == "rows"
    time.sleep(0.1)
    assert a._killer.sent == []


def test_without_a_timeout_nothing_is_armed():
    a = adapter(Blocking(finish_s=0.05))
    assert a._run("SELECT 1", None) == "rows"
    assert a._killer.sent == []


def test_a_failing_kill_falls_back_to_interrupting_duckdb_and_still_reports_a_timeout():
    a = adapter(Blocking())

    class Broken:
        def execute(self, sql):
            raise duckdb.IOException("Failed to connect to MySQL")
    a._killer = Broken()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(60)", 0.2)
    assert a._con.interrupts == 1


def test_an_error_the_statement_raises_on_its_own_is_not_called_a_timeout():
    a = adapter(Blocking())

    class Rejects(Blocking):
        def execute(self, query):
            raise duckdb.IOException("Unknown column 'x' in 'field list'")
    a._con = Rejects()
    with pytest.raises(duckdb.IOException, match="Unknown column"):
        a._run("SELECT x", 2)
    assert a._killer.sent == []


def test_a_kill_still_in_flight_when_the_statement_ends_completes_before_the_next_can_start():
    con = Blocking(finish_s=0.15)
    a = adapter(con)

    class Slow:
        sent = []

        def execute(self, sql):
            time.sleep(0.3)
            self.sent.append(sql)
    a._killer = Slow()
    started = time.monotonic()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    assert a._killer.sent and time.monotonic() - started >= 0.4


def test_a_kill_that_fails_late_leaves_no_interrupt_for_the_next_statement_and_drops_itself():
    con = Blocking(finish_s=0.15)
    a = adapter(con)

    class FailsLate:
        def execute(self, sql):
            time.sleep(0.2)
            raise duckdb.IOException("Lost connection to MySQL server")
    a._killer = FailsLate()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    assert a._killer is None
    assert a._run("SELECT 1", None) == "rows"


def test_a_cancel_stuck_reaching_mysql_holds_the_caller_only_for_the_bounded_wait():
    con = Blocking(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S = 0.2

    class Hangs:
        def execute(self, sql):
            time.sleep(2)
    a._killer = Hangs()
    started = time.monotonic()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    assert time.monotonic() - started < 1


def test_a_dropped_kill_connection_is_reopened_by_the_next_timeout():
    a = adapter(Blocking())
    a._killer = None  # as a failed kill or a reattach leaves it
    a._new_killer = lambda: Killer(a._con)
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(60)", 0.1)
    assert a._killer.sent == [KILL_455] and a._con.interrupts == 0


def test_a_cancel_abandoned_by_the_bounded_wait_touches_nothing_of_the_next_statement():
    con = Blocking(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S = 0.2

    class HangsThenFails:
        def execute(self, sql):
            time.sleep(0.6)
            raise duckdb.IOException("Lost connection to MySQL server")
    a._killer = HangsThenFails()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    con.finish_s = 0.4  # the next statement is running when the stale cancel wakes
    assert a._run("SELECT 2", None) == "rows"
    time.sleep(0.4)
    assert con.interrupts == 0 and not con.killed.is_set()


def test_a_kill_connection_reopened_too_late_neither_kills_the_next_statement_nor_installs():
    con = Blocking(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S, a._killer, late = 0.2, None, []

    def slow_reopen():
        time.sleep(0.6)
        late.append(Killer(con))
        return late[-1]
    a._new_killer = slow_reopen
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    con.finish_s = 0.8  # the next statement is running when the reopen completes
    assert a._run("SELECT 2", None) == "rows"
    assert late and late[0].sent == [] and not con.killed.is_set()
    assert a._killer is None and late[0].closed


def test_a_kill_delivered_after_the_bounded_wait_cannot_end_the_next_statement():
    # The statement ends while its KILL is stalled, `_run` stops waiting, and the KILL reaches
    # MySQL later. The session it names is retired, so the next statement runs on a fresh one.
    con = Session(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S = 0.2
    a._killer = LateKill(con, 0.6)
    a._new_killer = lambda: Killer(con)
    with pytest.raises(QueryTimedOut):
        a._table("SELECT SLEEP(10)", 0.1)
    con.finish_s = 0.8  # the next statement is running when the old KILL lands
    assert a._table("SELECT 2") == "rows"
    assert a._mysql_id == 456 and con.interrupts == 0


def test_an_attach_whose_kill_connection_stalls_finishes_without_it():
    con = Session()
    a = adapter(con)
    a._KILLER_OPEN_S, a._killer, late = 0.2, None, []

    def stalls():
        time.sleep(0.6)
        late.append(Killer(con))
        return late[-1]
    a._new_killer = stalls
    started = time.monotonic()
    a._reattach()
    assert time.monotonic() - started < 0.5 and a._killer is None and a._mysql_id == 456
    newer = a._killer = Killer(con)  # as a later timeout's reopen leaves it
    time.sleep(0.6)
    assert a._killer is newer and late and late[0].closed


def test_a_cancel_stuck_for_good_leaves_no_thread_that_keeps_the_process_alive():
    con = Blocking(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S = 0.1
    release = threading.Event()

    class Stuck:
        def execute(self, sql):
            release.wait(5)
    a._killer = Stuck()
    try:
        with pytest.raises(QueryTimedOut):
            a._run("SELECT SLEEP(10)", 0.05)
        alive = [t for t in threading.enumerate()
                 if t is not threading.main_thread() and t.is_alive()]
        assert alive and all(t.daemon for t in alive)
    finally:
        release.set()


def test_a_stale_cancel_whose_kill_fails_closes_only_its_own_connection():
    con = Blocking(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S = 0.2

    class HangsThenFails:
        closed = False

        def execute(self, sql):
            time.sleep(0.6)
            raise duckdb.IOException("Lost connection to MySQL server")

        def close(self):
            self.closed = True
    old = a._killer = HangsThenFails()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    newer = a._killer = Killer(con)  # as the next statement's reattach leaves it
    time.sleep(0.6)
    assert a._killer is newer and not newer.closed and old.closed


def test_an_interrupt_under_way_when_the_wait_expires_lands_before_the_statement_is_released():
    class SlowInterrupt(Blocking):
        def interrupt(self):
            time.sleep(0.3)
            super().interrupt()
    con = SlowInterrupt(finish_s=0.15)
    a = adapter(con)
    a._CANCEL_WAIT_S = 0.2

    class FailsLate:
        def execute(self, sql):
            time.sleep(0.15)
            raise duckdb.IOException("Lost connection to MySQL server")
    a._killer = FailsLate()
    with pytest.raises(QueryTimedOut):
        a._run("SELECT SLEEP(10)", 0.1)
    con.finish_s = 0.4  # the next statement would be running when the interrupt lands
    assert a._run("SELECT 2", None) == "rows"


# -- each fault reaches the engine as what it is ---------------------------------------------


def test_the_runner_reads_the_timeout_as_the_querys_cost():
    a = adapter(Blocking())
    with pytest.raises(ExecutionError, match="timed out after 0.2s") as raised:
        run_statement(a, "SELECT SLEEP(60)", timeout_s=0.2)
    assert type(raised.value) is ExecutionError  # not an outage, not a cancel


def test_a_kill_the_adapter_did_not_send_is_a_cancel_and_ends_the_turn():
    # An operator's KILL QUERY: the statement ends with MySQL's "interrupted" and no deadline
    # passed. Read as a timeout the model would rewrite and rerun the stopped statement.
    class OperatorKills(Blocking):
        def execute(self, query):
            raise duckdb.IOException(
                "Failed to fetch result row for MySQL query: Query execution was interrupted")
    a = adapter(OperatorKills())
    with pytest.raises(QueryCancelledError):
        run_statement(a, "SELECT 1", timeout_s=5)
    assert a._killer.sent == []


def test_the_servers_own_time_limit_is_a_timeout():
    class ServerLimit(Blocking):
        def execute(self, query):
            raise duckdb.IOException("Failed to fetch result row for MySQL query: Query execution "
                                     "was interrupted, maximum statement execution time exceeded")
    with pytest.raises(ExecutionError, match="timed out") as raised:
        run_statement(adapter(ServerLimit()), "SELECT 1", timeout_s=5)
    assert type(raised.value) is ExecutionError


class Gone(Session):
    """A server that went away: every statement and every reattach fails to connect."""

    def execute(self, query):
        if query.startswith("ATTACH") or "mysql_query" in query:
            raise duckdb.IOException("Failed to connect to MySQL database with parameters "
                                     "\"host=h password=hunter2 database=d\": Can't connect")
        return super().execute(query)


def _gone_adapter():
    a = adapter(Gone())
    a._REATTACH_S = 0.0
    a._new_killer = lambda: pytest.fail("no kill connection during an outage")
    return a


def test_an_outage_reaches_the_runner_as_unavailable_with_the_password_scrubbed(monkeypatch):
    monkeypatch.setattr(mysql_source.time, "sleep", lambda s: None)
    with pytest.raises(SourceUnavailableError) as raised:
        run_statement(_gone_adapter(), "SELECT 1", timeout_s=5)
    assert isinstance(raised.value.__cause__, MySQLUnavailable)
    assert "hunter2" not in raised.value.full_text
    assert raised.value.__cause__.__cause__ is None  # the driver's text, password and all, is dropped


def test_an_outage_at_the_proof_is_unavailable_not_a_refusal(monkeypatch):
    monkeypatch.setattr(mysql_source.time, "sleep", lambda s: None)
    outcome = prove_outcome(_gone_adapter(), "SELECT 1")
    assert outcome.proof is Proof.UNAVAILABLE and "hunter2" not in outcome.detail


def test_a_dropped_connection_is_reattached_and_the_statement_run_once_more():
    class DropsOnce(Session):
        dropped, dead = False, False

        def execute(self, query):
            if query.startswith("ATTACH"):
                self.dead = False
            elif "mysql_query" in query and "CONNECTION_ID()" not in query:
                if self.dead:
                    raise duckdb.IOException("MySQL server has gone away")
                if not self.dropped:
                    self.dropped = self.dead = True
                    raise duckdb.IOException("Lost connection to MySQL server during query")
            return super().execute(query)
    con = DropsOnce(finish_s=0.01)
    a = adapter(con)
    a._new_killer = lambda: Killer(con)
    assert a._table("SELECT 1") == "rows" and a._mysql_id == 456


class SyntaxRejects(Session):
    """MySQL refusing a statement it cannot parse, quoting part of it in its own words, as its
    syntax errors do: a statement holding outage or cancel words puts them in the message."""

    def execute(self, query):
        if "mysql_query" in query and "CONNECTION_ID()" not in query and "'SELECT 1'" not in query:
            inner = query.split("mysql_query('src', '", 1)[1][:-2].replace("''", "'")
            raise duckdb.IOException(
                "Failed to prepare MySQL query: You have an error in your SQL syntax; check the "
                f"manual for the right syntax to use near '{inner[inner.find('note'):]}' at line 1")
        return super().execute(query)


@pytest.mark.parametrize("statement", [
    "SELECT x FROM t WHERE note = 'lost connection'",
    "SELECT x FROM t WHERE note = 'query execution was interrupted'",
])
def test_a_statement_holding_outage_or_cancel_words_is_the_models_to_repair(statement):
    # Only the source's own state decides an outage (the session still answers a probe), and
    # only MySQL's trailing words a cancel, so neither ends the turn and nothing is reattached.
    # The runner itself reads leftover text naming an interrupt as a timeout, which, like a
    # rejection, goes back to the model; that reading is the runner's, not this adapter's.
    a = adapter(SyntaxRejects())
    a._reattach = lambda: pytest.fail("a refusal is never reattached")
    with pytest.raises(ExecutionError) as raised:
        run_statement(a, statement, timeout_s=5)
    assert type(raised.value) is ExecutionError  # not an outage, not a cancel
    words = statement.split("'")[1]
    assert isinstance(raised.value.__cause__, duckdb.IOException)
    assert words in str(raised.value.__cause__)  # the words did reach the message
    proof = prove_outcome(a, statement)
    assert proof.proof is Proof.REJECTED and words in proof.detail


def test_validate_runs_the_plan_as_a_zero_row_derived_table_under_a_bound():
    a = adapter(Blocking())
    seen = []
    a._table = lambda sql, timeout_s=None: seen.append((sql, timeout_s))
    a.validate("SELECT a FROM t;")
    assert seen == [("SELECT * FROM (SELECT a FROM t) AS mnemiq_probe LIMIT 0", 30)]


def test_only_outages_and_timeouts_are_unanswered():
    a = adapter(Blocking())
    assert a.source_unavailable(MySQLUnavailable("gone"))
    assert a.source_unavailable(QueryTimedOut("slow"))
    assert not a.source_unavailable(QueryCancelled("stopped"))
    assert not a.source_unavailable(duckdb.IOException("Unknown column 'x' in 'field list'"))


# -- configuration ---------------------------------------------------------------------------


def test_no_dsn_is_refused_before_anything_connects(monkeypatch):
    monkeypatch.delenv("MNEMIQ_MYSQL_DSN", raising=False)
    monkeypatch.setattr(mysql_source.duckdb, "connect", lambda *a, **k: pytest.fail("connected"))
    with pytest.raises(MySQLMisconfigured, match="MNEMIQ_MYSQL_DSN"):
        MySQLSource("dw")


def test_a_dsn_naming_a_database_is_refused(monkeypatch):
    monkeypatch.setattr(mysql_source.duckdb, "connect", lambda *a, **k: pytest.fail("connected"))
    with pytest.raises(MySQLMisconfigured, match="names a database"):
        MySQLSource("dw", dsn="host=h user=u database=other")


@pytest.mark.parametrize("sql_mode, lower_case, missing", [
    ("STRICT_TRANS_TABLES", 1, "ANSI_QUOTES"),
    ("ANSI_QUOTES", 0, "lower_case_table_names"),
    (b"ANSI_QUOTES,STRICT_TRANS_TABLES", 2, "lower_case_table_names"),
])
def test_a_server_without_the_settings_beaver_needs_is_refused(sql_mode, lower_case, missing):
    with pytest.raises(MySQLMisconfigured, match=missing):
        MySQLSource._check_settings(sql_mode, lower_case)


def test_a_server_with_both_settings_is_accepted():
    MySQLSource._check_settings(b"ONLY_FULL_GROUP_BY,ANSI_QUOTES", 1)
    MySQLSource._check_settings("ANSI_QUOTES", "1")


def test_a_kill_connection_that_cannot_open_says_so_without_the_password(monkeypatch):
    class Refuses:
        def execute(self, sql):
            if sql.startswith("ATTACH"):
                raise duckdb.IOException(
                    'Failed to connect to MySQL database with parameters "host=h '
                    'password=hunter2 database=d": Access denied')

        def close(self):
            pass
    monkeypatch.setattr(mysql_source.duckdb, "connect", lambda *a, **k: Refuses())
    a = adapter(Blocking())
    a._dsn = "host=h password=hunter2 database=d"
    with pytest.raises(duckdb.IOException) as raised:
        a._new_killer()
    assert "hunter2" not in str(raised.value) and raised.value.__cause__ is None


# -- live, against a MySQL server --------------------------------------------------------------

LIVE_DSN = os.getenv("MNEMIQ_TEST_MYSQL_DSN")
LIVE_DB = os.getenv("MNEMIQ_TEST_MYSQL_DATABASE", "dw")
needs_mysql = pytest.mark.skipif(
    not LIVE_DSN, reason="no MySQL configured (MNEMIQ_TEST_MYSQL_DSN); docs/beaver.md sets one up")


@pytest.fixture
def live_source():
    source = MySQLSource(LIVE_DB, dsn=LIVE_DSN)
    yield source
    source.close()


@pytest.mark.integration
@needs_mysql
def test_live_the_catalogue_reads_as_text(live_source):
    tables = live_source.introspect()
    columns = live_source.list_columns()
    assert tables and columns
    assert all(isinstance(c[2], str) for c in columns)  # column_type arrives as bytes, decoded
    assert live_source.user_functions() == []


@pytest.mark.integration
@needs_mysql
def test_live_a_timeout_kills_the_statement_and_the_next_one_runs(live_source):
    started = time.monotonic()
    with pytest.raises(QueryTimedOut):
        live_source.execute_arrow("SELECT SLEEP(20) AS s", timeout_s=0.5)
    assert time.monotonic() - started < 5
    assert live_source.execute_arrow("SELECT 1 AS one").to_pylist() == [{"one": 1}]


@pytest.mark.integration
@needs_mysql
def test_live_an_operators_kill_is_a_cancel(live_source):
    operator = duckdb.connect()
    operator.execute("LOAD mysql")
    operator.execute(live_source._attach_sql("op", read_only=False))
    session = live_source._mysql_id
    killer = threading.Timer(0.5, lambda: operator.execute(
        f"CALL mysql_execute('op', 'KILL QUERY {int(session)}')"))
    killer.start()
    # Not SLEEP: MySQL ends a killed SLEEP (and a killed BENCHMARK) with a row, not an error.
    endless = ("SELECT COUNT(*) AS n FROM information_schema.columns a, "
               "information_schema.columns b, information_schema.columns c, "
               "information_schema.columns d")
    try:
        with pytest.raises(QueryCancelledError):
            run_statement(live_source, endless, timeout_s=30)
    finally:
        killer.join()
        operator.close()
    assert live_source.execute_arrow("SELECT 1 AS one").to_pylist() == [{"one": 1}]


@pytest.mark.integration
@needs_mysql
def test_live_the_proof_accepts_a_valid_plan_and_rejects_an_unknown_column(live_source):
    table = live_source.introspect()[0]
    assert prove_outcome(live_source, f'SELECT COUNT(*) FROM "{table}"').proof is Proof.ACCEPTED
    assert prove_outcome(live_source, f'SELECT "no_such_col" FROM "{table}"').proof \
        is Proof.REJECTED


@pytest.mark.integration
@needs_mysql
def test_live_a_wrong_password_is_unavailable_and_never_shown():
    if "password=" not in LIVE_DSN:
        pytest.skip("the DSN carries no password to replace")
    dsn = re.sub(r"password=\S+", "password=not-the-password-7f3a", LIVE_DSN)
    with pytest.raises(MySQLUnavailable) as raised:
        MySQLSource(LIVE_DB, dsn=dsn)
    assert "not-the-password-7f3a" not in str(raised.value)

"""`execute` takes the deadline `execute_arrow` already takes, on every adapter enrich reads through.

Enrich reads a source through `execute`, which had no time limit: on a production database one profiling statement ran
for hours. With `timeout_s` it ends as `execute_arrow` ends a bounded statement, raising `QueryTimedOut`; without it,
nothing changes. No database here but files: the Oracle cases drive the adapter's own lease and retry loop over a cursor
that raises what the driver raises.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

import pytest

from mnemiq.adapters.base import QueryTimedOut
from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.scoped import TableScopedAdapter
from mnemiq.adapters.sqlite import SQLiteAdapter

# A statement that runs for far longer than any bound below and never yields a first row.
_DUCKDB_SLOW = "SELECT count(*) FROM range(100000000000) a"
_SQLITE_SLOW = ("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 10000000000) "
                "SELECT count(*) FROM n")


def _duckdb(tmp_path) -> DuckDBAdapter:
    import duckdb
    path = tmp_path / "d.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2)")
    con.close()
    return DuckDBAdapter.duckdb(str(path))


def _sqlite(tmp_path) -> SQLiteAdapter:
    path = tmp_path / "s.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2);")
    con.close()
    return SQLiteAdapter(str(path))


@pytest.mark.parametrize("make", [_duckdb, _sqlite], ids=["duckdb", "sqlite"])
def test_execute_without_a_deadline_is_unchanged(tmp_path, make):
    adapter = make(tmp_path)
    rows = adapter.execute("SELECT x FROM t ORDER BY x")
    assert rows == [(1,), (2,)]
    assert adapter.execute("SELECT x FROM t ORDER BY x", timeout_s=None) == rows
    assert adapter.execute("SELECT x FROM t ORDER BY x", timeout_s=30) == rows


@pytest.mark.parametrize("make,slow", [(_duckdb, _DUCKDB_SLOW), (_sqlite, _SQLITE_SLOW)], ids=["duckdb", "sqlite"])
def test_execute_times_out_past_its_deadline(tmp_path, make, slow):
    adapter = make(tmp_path)
    started = time.monotonic()
    with pytest.raises(QueryTimedOut):
        adapter.execute(slow, timeout_s=0.5)
    assert time.monotonic() - started < 5
    # The connection still answers afterwards: the bound ended the statement, not the adapter.
    assert adapter.execute("SELECT 1") == [(1,)]


def test_postgres_execute_times_out_past_its_deadline():
    # Gated: runs where MNEMIQ_PG_DSN names a reachable Postgres (the grading side's own adapter).
    import os
    psycopg = pytest.importorskip("psycopg")
    from mnemiq.adapters.pg import PostgresAdapter
    dsn = os.getenv("MNEMIQ_PG_DSN")
    if not dsn:
        pytest.skip("no MNEMIQ_PG_DSN")
    try:
        adapter = PostgresAdapter(dsn)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    with pytest.raises(QueryTimedOut):
        adapter.execute("SELECT pg_sleep(5)", timeout_s=0.5)
    assert adapter.execute("SHOW statement_timeout") != [("500ms",)], "the bound outlived its statement"
    assert adapter.execute("SELECT 1") == [(1,)]


def test_the_table_scope_passes_the_deadline_on():
    seen = []

    class Inner:
        def execute(self, sql, timeout_s=None):
            seen.append(timeout_s)
            return []

    TableScopedAdapter(Inner(), ("*",), "src").execute("SELECT 1", timeout_s=7)
    assert seen == [7]


# -- Oracle, without a database ---------------------------------------------------------------------------------------

oracledb = pytest.importorskip("oracledb")
from oracledb import errors as oracledb_errors  # noqa: E402

from mnemiq.adapters.oracle import OracleAdapter  # noqa: E402

_TIMED_OUT = ("DPY-4024: call timeout of 500 ms exceeded", 4024)
_TABLE_MISSING = ("ORA-00942: table or view does not exist", 942)


def _driver_error(message: str, code: int):
    """An error as python-oracledb's thin driver raises it: the driver's own error object, carrying the code."""
    return oracledb.DatabaseError(oracledb_errors._Error(message=message, code=code))


class _Connection:
    call_timeout = 0

    def is_healthy(self):
        return True


class _Cursor:
    def __init__(self, connection, fault):
        self.connection, self._fault, self.description = connection, fault, None

    def execute(self, sql):
        if self._fault:
            raise _driver_error(*self._fault)

    def fetchall(self):
        return [(1,)]

    def close(self):
        pass


def _oracle(fault, attempts: list, timeouts: list) -> OracleAdapter:
    """The adapter's own `execute` and `_with_cursor` (its read-only retry loop included) over a lease that hands out a
    connection whose cursor raises `fault`."""
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._oracledb = oracledb
    adapter._read_only = True
    adapter._refuse_unless_read = lambda sql: None

    @contextmanager
    def lease(timeout_ms=None):
        attempts.append(timeout_ms)
        yield _Connection()

    def cursor(con):
        timeouts.append(con)
        return _Cursor(con, fault)

    adapter._lease = lease
    adapter._cursor = cursor
    return adapter


def test_oracle_execute_reports_the_drivers_timeout_as_a_timeout():
    attempts, connections = [], []
    adapter = _oracle(_TIMED_OUT, attempts, connections)
    with pytest.raises(QueryTimedOut):
        adapter.execute("SELECT 1 FROM dual", timeout_s=0.5)
    assert connections[0].call_timeout == 500, "the bound is the driver's call timeout, in milliseconds"


def test_oracle_a_timed_out_statement_is_attempted_once():
    # `_with_cursor` retries a read-only statement on ORA-01466 only. The driver's timeout is not that error, so the
    # statement that ran to its deadline is not run again.
    attempts, connections = [], []
    adapter = _oracle(_TIMED_OUT, attempts, connections)
    with pytest.raises(QueryTimedOut):
        adapter.execute("SELECT 1 FROM dual", timeout_s=0.5)
    assert len(attempts) == 1


def test_oracle_execute_without_a_deadline_raises_the_drivers_error_as_before():
    attempts, connections = [], []
    adapter = _oracle(_TIMED_OUT, attempts, connections)
    with pytest.raises(oracledb.DatabaseError, match="DPY-4024"):
        adapter.execute("SELECT 1 FROM dual")
    adapter = _oracle(_TABLE_MISSING, [], [])
    with pytest.raises(oracledb.DatabaseError, match="ORA-00942"):
        adapter.execute("SELECT 1 FROM missing", timeout_s=0.5)


def test_oracle_execute_with_a_deadline_returns_rows():
    adapter = _oracle(None, [], [])

    def rows_cursor(con):
        cursor = _Cursor(con, None)
        cursor.description = (("X",),)
        return cursor

    adapter._cursor = rows_cursor
    assert adapter.execute("SELECT 1 FROM dual", timeout_s=5) == [(1,)]

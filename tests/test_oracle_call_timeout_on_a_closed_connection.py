"""No call reaches a connection the driver calls unhealthy, and no `call_timeout` write reaches one
it has closed, because that write ends the process.

python-oracledb 4.0.2 closes a connection itself when a call outlives `call_timeout` and the server
does not answer the break that follows within the same bound: DPY-4011, "socket timed out while
recovering from previous socket timeout". Measured on Oracle Free 23 in thin mode with a query that
was only slow (0.4 s a row under a 1 s bound), on a standalone and a pooled connection alike. On
that connection `call_timeout = ...` dereferences the socket the driver dropped and segfaults, and
no `except` catches a segfault: the lease's reset ended the process for every caller.

`is_healthy()` is a local check, False for that dropped socket and also for a session the server
said it is ending while the socket stays open. On the second, a write would be safe, and any call
made to tell the two apart -- a ping -- runs under whatever bound the connection already carries,
which can be none (Codex's review of #182). So the adapter makes no call on an unhealthy connection
at all: it raises `ConnectionUnusable`, a `ConnectionError`, which `unavailable_fault` reads as the
source's. The live reproduction is in `test_oracle_adapter.py`.

No database here: the fake records a write made with the socket gone, which is what crashed,
rather than raising, which the lease's `except` would have caught and so passed on the code that
crashed; and it counts every call that would cross the network while it is unhealthy.
"""

from __future__ import annotations

import ast
import inspect
import threading

import pytest

oracledb = pytest.importorskip("oracledb")

import mnemiq.adapters.oracle as oracle_module  # noqa: E402
from mnemiq.adapters.oracle import (  # noqa: E402
    ConnectionUnusable,
    OracleAdapter,
    unavailable_fault,
)

_CLOSED_BY_THE_DRIVER = "DPY-4011: the database or network closed the connection"


class _Connection:
    """A connection the driver can close mid-call. It records every `call_timeout` write, the ones
    made with the socket gone, and every network call made while unhealthy. `healthy` is what
    `is_healthy()` says; `socket` is whether the driver still holds one, which the write needs."""

    def __init__(self, healthy: bool = True, socket: bool | None = None, ending_after=None):
        object.__setattr__(self, "ending_after", ending_after)   # the server ends it after this
        object.__setattr__(self, "healthy", healthy)
        object.__setattr__(self, "socket", healthy if socket is None else socket)
        object.__setattr__(self, "written_while_closed", [])
        object.__setattr__(self, "written", [])
        object.__setattr__(self, "calls_while_unhealthy", [])
        object.__setattr__(self, "call_timeout", 0)

    def __setattr__(self, name, value):
        if name == "call_timeout":
            (self.written if self.socket else self.written_while_closed).append(value)
        object.__setattr__(self, name, value)

    def is_healthy(self):
        return self.healthy

    def close_by_the_driver(self):
        object.__setattr__(self, "healthy", False)
        object.__setattr__(self, "socket", False)

    def say_the_session_is_ending(self):
        object.__setattr__(self, "healthy", False)   # the socket stays open

    def network(self, call):
        if not self.healthy:
            self.calls_while_unhealthy.append(call)
        if not self.socket:
            raise oracledb.InterfaceError("DPY-1001: not connected to database")

    def ping(self):
        self.network("ping")

    def rollback(self):
        self.network("rollback")

    def cursor(self):
        return _Cursor(self)


class _Cursor:
    description = [("N", None, None, None, None, None, None)]

    def __init__(self, con):
        self.connection = con

    def execute(self, sql, **_binds):
        self.connection.network("execute")
        if sql == self.connection.ending_after:
            self.connection.say_the_session_is_ending()
        if "slow" in sql:
            # The bound ran out and the break went unanswered: the driver closes the connection.
            self.connection.close_by_the_driver()
            raise oracledb.DatabaseError(_CLOSED_BY_THE_DRIVER)

    def fetchall(self):
        return [(1,)]

    def close(self):
        pass


class _Pool:
    def __init__(self, con):
        self.con, self.released, self.dropped = con, [], []

    def acquire(self):
        return self.con

    def release(self, con):
        self.released.append(con)

    def drop(self, con):
        self.dropped.append(con)


def _pooled(con, read_only=True):
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._oracledb = oracledb
    adapter._pool, adapter._con, adapter._closed = _Pool(con), None, False
    adapter._read_only, adapter._ro_ttl_s, adapter._ro_state = read_only, 0.0, "unknown"
    adapter._ro_checked_at = adapter._ro_attempted_at = None
    adapter._lock = threading.RLock()
    return adapter


_UNHEALTHY = pytest.mark.parametrize("socket", [False, True],
                                     ids=["socket-dropped", "session-ending"])


def test_a_pooled_connection_closed_during_the_lease_is_dropped_without_a_write():
    con = _Connection()
    adapter = _pooled(con)
    with pytest.raises(oracledb.DatabaseError, match="DPY-4011"):
        with adapter._lease(timeout_ms=1000) as leased:
            leased.cursor().execute("SELECT slow(1) FROM dual")
    assert con.written_while_closed == [], "the reset wrote to a closed connection: a segfault"
    assert adapter._pool.dropped == [con] and adapter._pool.released == []


def test_a_pooled_session_ending_during_the_lease_is_dropped_with_no_call_at_its_reset():
    # The pool drops it anyway, on return or at the next acquire, and a call to reset a bound
    # nobody will use could stall on that session (the review gate on the second commit).
    con = _Connection()
    adapter = _pooled(con)
    with adapter._lease(timeout_ms=1000) as leased:
        leased.say_the_session_is_ending()
    assert con.written == [1000] and con.calls_while_unhealthy == [], "the reset wrote or called"
    assert adapter._pool.dropped == [con] and adapter._pool.released == []


@_UNHEALTHY
def test_an_unhealthy_connection_the_pool_hands_back_gets_no_call_and_is_dropped(socket):
    con = _Connection(healthy=False, socket=socket)
    adapter = _pooled(con)
    with pytest.raises(ConnectionUnusable):
        with adapter._lease(timeout_ms=1000) as leased:
            leased.cursor().execute("SELECT 1 FROM dual")
    assert con.written == con.written_while_closed == con.calls_while_unhealthy == []
    assert adapter._pool.dropped == [con] and adapter._pool.released == []


def test_a_healthy_lease_still_bounds_its_connection_and_releases_it():
    # The control: an adapter that never wrote the timeout, or dropped every connection, passes
    # the tests around it.
    con = _Connection()
    adapter = _pooled(con)
    with adapter._lease(timeout_ms=1000) as leased:
        assert leased.call_timeout == 1000
    assert con.written == [1000, 0]
    assert adapter._pool.released == [con] and adapter._pool.dropped == []


def test_a_borrowed_connection_closed_during_the_lease_is_not_written_back():
    con = _Connection()
    adapter = OracleAdapter.over(con, oracledb, schema="APP")
    with pytest.raises(oracledb.DatabaseError, match="DPY-4011"):   # the body's, not the restore's
        with adapter._lease(timeout_ms=1000) as leased:
            leased.cursor().execute("SELECT slow(1) FROM dual")
    assert con.written == [1000]
    assert con.written_while_closed == [], "the restore wrote to a closed connection: a segfault"


@_UNHEALTHY
@pytest.mark.parametrize("read_only", [True, False], ids=["read-only", "writable"])
def test_an_unhealthy_borrowed_connection_gets_no_call_and_reads_as_the_sources(socket, read_only):
    # Read-only, the first call would be `_cursor`'s rollback; writable, `execute_arrow`'s bound.
    # On a session the server is ending either is a round trip under whatever bound the
    # connection carries, here none (Codex's review of #182).
    con = _Connection(healthy=False, socket=socket)
    adapter = OracleAdapter.over(con, oracledb, schema="APP", read_only=read_only)
    with pytest.raises(ConnectionUnusable) as caught:
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=1.0)
    assert unavailable_fault(caught.value) and adapter.source_unavailable(caught.value)
    with pytest.raises(ConnectionUnusable):
        with adapter._lease(timeout_ms=1000):   # a lease that bounds on entry
            pass
    assert con.written == con.written_while_closed == con.calls_while_unhealthy == []


def test_a_session_ending_before_its_query_gets_no_bound_and_no_query():
    # Unhealthy after the lease's setup, during `_cursor`'s SET TRANSACTION READ ONLY: the next
    # call is `execute_arrow`'s bound, then the query, each on whatever bound the lease left.
    con = _Connection(ending_after="SET TRANSACTION READ ONLY")
    adapter = _pooled(con)
    with pytest.raises(ConnectionUnusable):
        adapter.execute_arrow("SELECT 1 FROM dual", timeout_s=1.0)
    assert con.written == [0] and con.calls_while_unhealthy == []
    assert adapter._pool.dropped == [con] and adapter._pool.released == []


@pytest.mark.parametrize("healthy, bounded", [(True, [1234]), (False, [])],
                         ids=["closed-by-the-probe", "closed-before-it"])
def test_a_read_only_probe_on_a_closed_connection_writes_no_timeout(healthy, bounded):
    con = _Connection(healthy=healthy)
    adapter = _pooled(con)
    adapter._ro_ttl_s, adapter._ro_state, adapter._probe_timeout_ms = 0.001, "constrained", 1234
    adapter._ro_unverified, adapter._ro_unverified_since = False, 0.0
    adapter._ro_unverified_next_s = 0.0
    adapter._READ_ONLY_DB_PROBE = "SELECT slow(1) FROM dual"
    adapter._recheck_read_only(con)
    assert con.written == bounded
    assert con.written_while_closed == [], "the probe wrote a timeout to a closed connection"
    assert con.calls_while_unhealthy == [], "a call after the close"


_TIMEOUT_ATTRIBUTES = {"call_timeout", "callTimeout"}   # the driver keeps the old name as an alias


def _could_name_the_timeout(name, constants) -> bool:
    """A `setattr` name that is, or may be, the timeout: a literal or a module constant naming
    it, or any other expression, which could."""
    if isinstance(name, ast.Constant):
        return name.value in _TIMEOUT_ATTRIBUTES
    if isinstance(name, ast.Name) and name.id in constants:
        return constants[name.id] in _TIMEOUT_ATTRIBUTES
    return True


def test_every_call_timeout_write_in_the_adapter_goes_through_the_guard():
    # The next write added beside these would bring the crash back, and no test above would see
    # a site it does not drive: every assignment to `.call_timeout`, under either of the driver's
    # names or through `setattr` or `__setattr__`, sits inside the one function that checks the
    # connection first. A `setattr` bound to another name is beyond it.
    tree = ast.parse(inspect.getsource(oracle_module))
    # The module's own constants, so `setattr(exc, _ACQUIRE_FAULT, True)` reads as what it is.
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets if isinstance(target, ast.Name)
    }
    owners = {}  # innermost function: the walk reaches an outer function before a nested one
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                owners[id(inner)] = node.name
    writes = [
        (target.lineno, owners.get(id(target)))
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute) and target.attr in _TIMEOUT_ATTRIBUTES
    ] + [
        (node.lineno, owners.get(id(node)))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "setattr" and len(node.args) >= 2
        and _could_name_the_timeout(node.args[1], constants)
    ] + [
        (node.lineno, owners.get(id(node)))   # `object.__setattr__`, `con.__setattr__`
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__setattr__" and len(node.args) >= 2
        # the name is the second of three arguments on `object`, the first of two on the instance
        and _could_name_the_timeout(node.args[1 if len(node.args) >= 3 else 0], constants)
    ]
    assert writes, "no write found: the walk is not reading the adapter"
    assert {owner for _, owner in writes} == {"_set_call_timeout"}, writes

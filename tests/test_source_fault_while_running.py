"""A source that stops answering while a query runs is an outage, not the query's failure (M188).

The proof seam reads an unanswered proof as SOURCE_UNAVAILABLE (#159, #168), but a connection that
dropped after the proof, or mid-fetch, reached the runner as "The source rejected this query." and
the loop handed it to the model as repair feedback: a correct query rewritten because the network
failed. Measured on the runtime's Postgres path (DuckDB's attach of a live Postgres, reconnects
refused): a duckdb IOException, which the adapter's own `source_unavailable` already called an
outage, and nobody asked it there. The turn now ends on it, reported as the outage it is, and the
statement is never run again: a first draft re-ran it once, and Codex's review of #179 showed a cancel or
a statement timeout worded in another language, and an administrator's kill, all read as outages, so the
re-run repeated the very query someone had stopped.

Real DuckDB sources whose connection closes at a chosen run; Oracle's and SQLite's timeouts through
their own `execute_arrow`.
"""

from __future__ import annotations

import sqlite3
import sys
import time

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot
from mnemiq.execute.runner import ExecutionError, SourceUnavailableError, run
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.server.serialize import answer_payload

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t", principal_id="u", roles=["r"])
# Distinct statements, so each deep-mode candidate runs rather than reading the one before it from
# the cache.
_REPLIES = [f'{{"sql": "SELECT id FROM claim WHERE id > {n}", "reason": "ok"}}' for n in range(-5, 7)]


def _duckdb(tmp_path) -> DuckDBAdapter:
    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT); INSERT INTO claim VALUES (1), (2)")
    con.close()
    return DuckDBAdapter.duckdb(path, read_only=False)


class _DropsAtRun:
    """The real DuckDB source, whose connection closes as run number `at` starts: that run, every
    later one and every later proof fail with DuckDB's own fault for a closed connection."""

    def __init__(self, inner, at: int):
        self._inner, self._at, self.runs = inner, at, 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute_arrow(self, sql, timeout_s=None):
        self.runs += 1
        if self.runs == self._at:
            self._inner._con.close()
        return self._inner.execute_arrow(sql, timeout_s=timeout_s)


class _DropsOnceAtRun(_DropsAtRun):
    """A connection that drops during run number `at` and is there again for the next, as the
    Postgres attach reconnects: DuckDB's own IOException, once."""

    def execute_arrow(self, sql, timeout_s=None):
        self.runs += 1
        if self.runs == self._at:
            raise duckdb.IOException("IO Error: Failed to fetch header for COPY: FATAL: terminating "
                                     "connection due to administrator command")
        return self._inner.execute_arrow(sql, timeout_s=timeout_s)


def _ask(adapter, candidates: int = 1):
    generator = FakeGenerator(list(_REPLIES))
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=adapter,
                  cache=TwoTierCache(L1Cache()), candidates=candidates)
    packet = ContextPacket(question="which claims?", grant_fingerprint=_GRANTS.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-21T00:00:00Z",
                        columns=[Column(id="claim.id", object_id="claim", name="id")])
    return agent.answer(packet, snapshot, _GRANTS, _IDENTITY), generator


def _is_the_outage(answer):
    assert answer.failed is True, answer
    assert answer.deferred is False, "an outage counted as a deferral raises the deferral rate (M6)"
    assert answer.reason_code is DeferralReason.SOURCE_UNAVAILABLE, answer.reason_code
    assert "did not answer" in answer.answer, answer.answer
    assert "closed" not in answer.answer.lower(), "the driver's words stay in the server log"


# -- the runner --------------------------------------------------------------------------------


def test_a_connection_lost_while_running_is_the_sources_and_not_the_querys(tmp_path):
    adapter = _duckdb(tmp_path)
    adapter._con.close()
    with pytest.raises(SourceUnavailableError) as caught:
        run(adapter, "SELECT id FROM claim")
    err = caught.value
    assert isinstance(err, ExecutionError), "every caller catching ExecutionError still catches it"
    assert "did not answer" in err.message
    assert "closed" not in err.message.lower(), "the driver's words ride in source_detail"
    assert "closed" in (err.source_detail or "").lower()


def test_a_timeout_stays_a_timeout_though_the_classifier_calls_an_interrupt_unavailable(tmp_path):
    # The premise: DuckDB's classifier reads its interrupt as unavailable, the proof seam's reading
    # of a proof that never came back. The runner's timeout branch is asked first.
    adapter = _duckdb(tmp_path)
    assert adapter.source_unavailable(duckdb.InterruptException("x"))
    with pytest.raises(ExecutionError) as caught:
        run(adapter, "SELECT count(*) FROM range(100000000000)", timeout_s=0.3)
    assert not isinstance(caught.value, SourceUnavailableError)
    assert "timed out" in caught.value.message


def test_the_querys_own_fault_is_still_a_rejection_the_model_repairs(tmp_path):
    with pytest.raises(ExecutionError) as caught:
        run(_duckdb(tmp_path), "SELECT nope FROM claim")
    assert not isinstance(caught.value, SourceUnavailableError)
    assert caught.value.message == "The source rejected this query."
    assert "nope" in caught.value.repair_text


class _Raising:
    def __init__(self, exc, classifier=None):
        self._exc = exc
        if classifier is not None:
            self.source_unavailable = classifier

    def execute_arrow(self, sql, timeout_s=None):
        raise self._exc


def test_an_adapter_without_a_classifier_is_read_as_the_proof_reads_it():
    # The proof's fallback, the same function: Python's own connection fault is an outage, any
    # other exception keeps its reading.
    with pytest.raises(SourceUnavailableError):
        run(_Raising(ConnectionResetError("peer reset")), "SELECT 1")
    with pytest.raises(ExecutionError) as caught:
        run(_Raising(ValueError("bad")), "SELECT 1")
    assert not isinstance(caught.value, SourceUnavailableError)


def test_a_classifier_that_fails_leaves_the_rejection():
    def broken(exc):
        raise RuntimeError("the classifier itself failed")

    with pytest.raises(ExecutionError) as caught:
        run(_Raising(ConnectionResetError("peer reset"), classifier=broken), "SELECT 1")
    assert not isinstance(caught.value, SourceUnavailableError)


def test_sqlites_own_timeout_reaches_the_model_as_a_timeout(tmp_path):
    # It raised a bare RuntimeError("query timed out ..."), which the runner, reading "interrupt"
    # alone, reported as "The source rejected this query."
    from mnemiq.adapters.sqlite import SQLiteAdapter

    path = str(tmp_path / "s.sqlite")
    sqlite3.connect(path).close()
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    with pytest.raises(ExecutionError) as caught:
        run(SQLiteAdapter(path), runaway, timeout_s=0.2)
    assert "timed out after 0.2s" in caught.value.message, caught.value.message


# -- Oracle: its classifier and its timeout through its own execute_arrow ------------------------


@pytest.fixture
def oracle():
    oracledb = pytest.importorskip("oracledb")
    from oracledb import errors as oracledb_errors

    from mnemiq.adapters.oracle import OracleAdapter

    def driver_error(message, code, kind=None):
        return (kind or oracledb.DatabaseError)(oracledb_errors._Error(message=message, code=code))

    def adapter(raising, healthy=True):
        """`execute_arrow` alone, over a lease and a cursor that raises: no database. `healthy` is
        what the driver's `is_healthy()` says of the connection."""
        bare = OracleAdapter.__new__(OracleAdapter)
        bare._oracledb = oracledb

        class Cursor:
            connection = type("Connection", (), {"call_timeout": 0,
                                                 "is_healthy": lambda self: healthy})()

            def execute(self, sql):
                raising()

        def with_cursor(work, timeout_ms=None):
            time.monotonic()  # the lease's wait, before the call
            return work(Cursor())

        bare._refuse_unless_read = lambda sql: None
        bare._with_cursor = with_cursor
        return bare

    return oracledb, driver_error, adapter


def _raise(exc):
    def raising():
        raise exc
    return raising


def test_oracles_network_fault_while_running_is_an_outage(oracle):
    oracledb, driver_error, adapter = oracle
    lost = driver_error("DPY-6005: cannot connect to database", 6005, oracledb.OperationalError)
    with pytest.raises(SourceUnavailableError):
        run(adapter(_raise(lost)), "SELECT 1 FROM dual", timeout_s=5)


def test_oracles_parse_refusal_is_still_a_rejection(oracle):
    _, driver_error, adapter = oracle
    with pytest.raises(ExecutionError) as caught:
        run(adapter(_raise(driver_error("ORA-00942: table or view does not exist", 942))),
            "SELECT 1 FROM nope", timeout_s=5)
    assert not isinstance(caught.value, SourceUnavailableError)
    assert caught.value.message == "The source rejected this query."


def test_oracles_call_timeout_reaches_the_model_as_a_timeout(oracle):
    _, driver_error, adapter = oracle
    with pytest.raises(ExecutionError) as caught:
        run(adapter(_raise(driver_error("DPY-4024: call timeout of 300 ms exceeded", 4024))),
            "SELECT 1 FROM dual", timeout_s=0.3)
    assert not isinstance(caught.value, SourceUnavailableError)
    assert "timed out after 0.3s" in caught.value.message, caught.value.message


@pytest.mark.parametrize("elapsed", [0.5, 5.0, 7.5])
def test_oracles_closed_connection_is_an_outage_whatever_the_clock(oracle, monkeypatch, elapsed):
    # DPY-4011, the database or network closed the connection, was read as a timeout whatever the
    # clock until the runner read the adapter's outage classification (M187's note), then as one
    # at or past the deadline the adapter measured (Codex's reviews of #179: it could not measure
    # it). The driver reports its own bound as DPY-4024, or as DPY-4011 with a line saying so
    # (`_BOUND_ENDED_CLOSING`); a closed connection with no such line is the source's.
    oracledb, driver_error, adapter = oracle
    closed = driver_error("DPY-4011: the database or network closed the connection", 4011,
                          oracledb.DatabaseError)
    clock = iter([0.0, 100.0, 100.0 + elapsed])   # a clock that would have read the deadline
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(SourceUnavailableError):
        run(adapter(_raise(closed)), "SELECT 1 FROM dual", timeout_s=5)


def test_oracles_refusal_of_a_connection_its_driver_gave_up_is_an_outage(oracle):
    # The adapter refuses a connection python-oracledb calls unhealthy before writing its bound,
    # since that write segfaults on one the driver closed (#182, M214). The refusal is
    # `ConnectionUnusable`, and the statement was never sent: the source's, not the query's.
    from mnemiq.adapters.oracle import ConnectionUnusable

    _, _, adapter = oracle
    sent = []
    with pytest.raises(SourceUnavailableError) as caught:
        run(adapter(lambda: sent.append("execute"), healthy=False), "SELECT 1 FROM dual",
            timeout_s=5)
    assert sent == [], "nothing is sent on a connection the driver gave up"
    assert isinstance(caught.value.__cause__, ConnectionUnusable), caught.value.__cause__


# -- the loop --------------------------------------------------------------------------------


def test_a_source_down_at_the_run_fails_the_turn_without_asking_the_model_to_repair(tmp_path):
    source = _DropsAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source)
    assert len(generator.calls) == 1, generator.calls
    assert source.runs == 1, "the statement is not run again"
    _is_the_outage(answer)


def test_a_connection_oracles_driver_gave_up_after_the_proof_ends_the_turn(tmp_path, oracle):
    # The proof answers, here DuckDB's EXPLAIN, and the run meets Oracle's own refusal of a
    # connection its driver calls unhealthy (#182, M214): the turn ends as the source's, and the
    # model is not asked to repair a query nothing was wrong with.
    _, _, adapter = oracle
    proved, sent = _duckdb(tmp_path), []
    unusable = adapter(lambda: sent.append("execute"), healthy=False)

    class _ProvedThenUnusable:
        """DuckDB for everything the proof asks; Oracle's `execute_arrow` and its classifier for
        the run."""

        runs = 0

        def __getattr__(self, name):
            return getattr(proved, name)

        def execute_arrow(self, sql, timeout_s=None):
            self.runs += 1
            return unusable.execute_arrow(sql, timeout_s=timeout_s)

        def source_unavailable(self, exc):
            return unusable.source_unavailable(exc)

    source = _ProvedThenUnusable()
    answer, generator = _ask(source)
    _is_the_outage(answer)
    assert source.runs == 1, "the proof answered and the run was reached, once"
    assert sent == []
    assert len(generator.calls) == 1, generator.calls


def test_a_connection_that_drops_once_is_not_run_again(tmp_path):
    # The source would answer the next run, as the Postgres attach reconnects, and it still ends the
    # turn: the drop cannot be told apart from an administrator's kill or a cancel in another language.
    source = _DropsOnceAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source)
    _is_the_outage(answer)
    assert len(generator.calls) == 1
    assert source.runs == 1


def test_the_caller_reads_failed_and_not_deferred(tmp_path):
    answer, _ = _ask(_DropsAtRun(_duckdb(tmp_path), at=1))
    out = answer_payload(answer)
    assert (out["failed"], out["deferred"], out["reason_code"]) == (True, False, "source_unavailable")


def test_deep_mode_stops_at_the_first_run_the_source_does_not_answer(tmp_path):
    # Before, the candidate dropped out of the vote, the next ones were asked of a source that was
    # down, and with none run the single path was tried as well.
    source = _DropsAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source, candidates=5)
    assert len(generator.calls) == 1, generator.calls
    assert source.runs == 1
    _is_the_outage(answer)
    assert answer.candidates_executed == 0


def test_deep_mode_fails_the_turn_when_the_source_goes_down_after_a_candidate_ran(tmp_path):
    source = _DropsAtRun(_duckdb(tmp_path), at=2)
    answer, generator = _ask(source, candidates=5)
    assert len(generator.calls) == 2, generator.calls
    _is_the_outage(answer)
    assert answer.candidates_executed == 1


def test_deep_mode_stops_on_a_drop_the_source_would_answer_next_time(tmp_path):
    source = _DropsOnceAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source, candidates=5)
    _is_the_outage(answer)
    assert answer.candidates_executed == 0
    assert len(generator.calls) == 1
    assert source.runs == 1


# -- a cancel the bound did not send is a cancel (the review gate on M188) ------------------------
#
# Measured on the Postgres attach: `pg_cancel_backend` from another session raises DuckDB's
# IOException ("canceling statement due to user request"), which the classifier calls unavailable,
# and Oracle's ORA-01013 is an OperationalError, which its classifier calls
# unavailable too. Read as outages they would be reported as the source not answering. A cancel ends
# the turn, reported as the cancel it is: handed to the repair loop, as before M188, a model that wrote
# the same SQL again ran the stopped statement a second time (Codex's second review of #179).


def test_a_postgres_cancel_through_the_attach_is_a_cancel():
    from mnemiq.adapters.base import QueryCancelled
    from mnemiq.adapters.duckdb import execute_bounded

    class Con:
        def interrupt(self):
            pass

        def execute(self, sql):
            raise duckdb.IOException("IO Error: Failed to fetch header for COPY: ERROR:  canceling "
                                     "statement due to user request")

    with pytest.raises(QueryCancelled, match="user request"):
        execute_bounded(Con(), "SELECT 1", timeout_s=20)


def test_the_runner_reads_a_cancel_as_neither_an_outage_nor_a_timeout():
    from mnemiq.adapters.base import QueryCancelled
    from mnemiq.execute.runner import QueryCancelledError

    class Cancelled:
        def source_unavailable(self, exc):
            return True  # what the classifiers answer for the driver's own cancel

        def execute_arrow(self, sql, timeout_s=None):
            raise QueryCancelled("the source cancelled the query: canceling statement due to user request")

    with pytest.raises(QueryCancelledError) as caught:
        run(Cancelled(), "SELECT 1")
    assert isinstance(caught.value, ExecutionError), "every caller catching ExecutionError still catches it"
    assert not isinstance(caught.value, SourceUnavailableError)
    assert "cancelled" in caught.value.message and "timed out" not in caught.value.message


@pytest.mark.parametrize("elapsed", [0.5, 5.0, 7.5])
def test_oracles_cancel_is_a_cancel_and_not_an_outage_whatever_the_clock(oracle, monkeypatch, elapsed):
    from mnemiq.execute.runner import QueryCancelledError

    oracledb, driver_error, adapter = oracle
    cancel = driver_error("ORA-01013: user requested cancel of current operation", 1013,
                          oracledb.OperationalError)
    clock = iter([0.0, 100.0, 100.0 + elapsed])   # a clock that would have read the deadline
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(QueryCancelledError) as caught:
        run(adapter(_raise(cancel)), "SELECT 1 FROM dual", timeout_s=5)
    assert not isinstance(caught.value, SourceUnavailableError)
    assert "cancelled" in caught.value.message
    assert "ORA-01013" in (caught.value.source_detail or "")


class _CancelledAtRun(_DropsAtRun):
    """A statement cancelled from outside during run number `at`: the attach's own cancel, as
    `execute_bounded` raises it."""

    def __init__(self, inner, at):
        super().__init__(inner, at)
        self.sqls = []

    def execute_arrow(self, sql, timeout_s=None):
        from mnemiq.adapters.base import QueryCancelled

        self.runs += 1
        self.sqls.append(sql)
        if self.runs == self._at:
            raise QueryCancelled("the source cancelled the query: canceling statement due to user request")
        return self._inner.execute_arrow(sql, timeout_s=timeout_s)


_SAME = ['{"sql": "SELECT id FROM claim WHERE id > 0", "reason": "ok"}'] * 12


def _is_the_cancel(answer):
    assert answer.failed is True and answer.deferred is False, answer
    assert answer.reason_code is DeferralReason.EXECUTION_FAILED, answer.reason_code
    assert "cancelled" in answer.answer, answer.answer


@pytest.mark.parametrize("candidates", [1, 5], ids=["single", "deep"])
def test_a_cancelled_statement_ends_the_turn_though_the_model_would_write_it_again(tmp_path, monkeypatch, candidates):
    # The model writes the same SQL every time, as it often does once repaired: the cancelled
    # statement runs once, and nothing after it.
    monkeypatch.setattr(sys.modules[__name__], "_REPLIES", _SAME)
    source = _CancelledAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source, candidates=candidates)
    _is_the_cancel(answer)
    assert source.runs == 1, source.sqls
    assert len(generator.calls) == 1, generator.calls
    assert answer.candidates_executed in (None, 0)


def test_deep_mode_ends_the_turn_on_a_cancel_after_a_candidate_ran(tmp_path):
    source = _CancelledAtRun(_duckdb(tmp_path), at=2)
    answer, generator = _ask(source, candidates=5)
    _is_the_cancel(answer)
    assert answer.candidates_executed == 1
    assert source.runs == 2 and len(generator.calls) == 2


# -- worded in another language: read as an outage, and never run again (Codex's review of #179) ------
#
# The attach carries no SQLSTATE, only the server's message, so a cancel or a statement timeout from a
# Postgres whose `lc_messages` is not English matches neither English guard and reads as an outage. That
# is a misreport this accepts and states, and it costs the turn, never a second run of the statement.


_GERMAN_CANCEL = ("IO Error: Failed to fetch header for COPY: FEHLER:  storniere Anfrage wegen "
                  "Benutzeranforderung")
_GERMAN_TIMEOUT = ("IO Error: Failed to fetch header for COPY: FEHLER:  storniere Anfrage wegen "
                   "Zeitüberschreitung der Anweisung")


class _AttachFaultAtRun(_DropsAtRun):
    """Run number `at` ends with the attach's own fault for the given server message, raised
    through `execute_bounded` as a Postgres source's would be."""

    def __init__(self, inner, at, message):
        super().__init__(inner, at)
        self._message = message

    def execute_arrow(self, sql, timeout_s=None):
        from mnemiq.adapters.duckdb import execute_bounded

        self.runs += 1
        if self.runs == self._at:
            message = self._message

            class Con:
                def interrupt(self):
                    pass

                def execute(self, q):
                    raise duckdb.IOException(message)

            return execute_bounded(Con(), sql, timeout_s)
        return self._inner.execute_arrow(sql, timeout_s=timeout_s)


@pytest.mark.parametrize("message", [_GERMAN_CANCEL, _GERMAN_TIMEOUT], ids=["cancel", "statement-timeout"])
def test_a_cancel_or_timeout_in_another_language_ends_the_turn_and_is_not_run_again(tmp_path, message):
    source = _AttachFaultAtRun(_duckdb(tmp_path), at=1, message=message)
    answer, generator = _ask(source)
    _is_the_outage(answer)
    assert source.runs == 1, "the stopped statement ran again"
    assert len(generator.calls) == 1


@pytest.mark.parametrize("message", [_GERMAN_CANCEL, _GERMAN_TIMEOUT], ids=["cancel", "statement-timeout"])
def test_deep_mode_does_not_run_a_stopped_statement_again_either(tmp_path, message):
    source = _AttachFaultAtRun(_duckdb(tmp_path), at=1, message=message)
    answer, generator = _ask(source, candidates=5)
    _is_the_outage(answer)
    assert source.runs == 1
    assert answer.candidates_executed == 0


# -- the driver's words never outweigh its type (Codex's fifth review of #179) ------------------
#
# The runner read "interrupt" anywhere in a fault's text as the timeout, before the cancel and the
# outage. The attach's message quotes the statement, so a cancel or a dropped connection on a query
# naming `service_interruptions` read as a timeout, the model's to repair, and the stopped statement
# could run again. The text is now asked only of a fault nothing typed has claimed.

_QUOTED = 'IO Error: Failed to execute query "SELECT id FROM service_interruptions": '
_CANCEL_QUOTING = _QUOTED + "ERROR:  canceling statement due to user request"
_DROP_QUOTING = _QUOTED + "server closed the connection unexpectedly"


class _Attach:
    """The attach's `execute_bounded` over a connection raising `message`, with DuckDB's own
    classifier."""

    def __init__(self, message):
        self._message = message

    def execute_arrow(self, sql, timeout_s=None):
        from mnemiq.adapters.duckdb import execute_bounded

        message = self._message

        class Con:
            def interrupt(self):
                pass

            def execute(self, q):
                raise duckdb.IOException(message)

        return execute_bounded(Con(), sql, timeout_s)

    def source_unavailable(self, exc):
        from mnemiq.adapters.duckdb import unavailable_fault

        return unavailable_fault(exc)


def test_a_cancel_quoting_a_statement_that_says_interrupt_is_still_the_cancel():
    from mnemiq.execute.runner import QueryCancelledError

    with pytest.raises(QueryCancelledError) as caught:
        run(_Attach(_CANCEL_QUOTING), "SELECT id FROM service_interruptions", timeout_s=20)
    assert "timed out" not in caught.value.message


def test_a_drop_quoting_a_statement_that_says_interrupt_is_still_the_outage():
    with pytest.raises(SourceUnavailableError):
        run(_Attach(_DROP_QUOTING), "SELECT id FROM service_interruptions", timeout_s=20)


def test_an_interrupt_nothing_typed_claims_is_still_read_as_the_timeout():
    # The control: the words still decide where no type does, a driver or a wrapper raising an
    # untyped error, and there the other reading is a rejection, repairable either way. A fault
    # an adapter's classifier calls an outage is the outage first, whatever its words: DuckDB's
    # IOException through the attach, a MySQL server's "Query execution was interrupted" among
    # them if it arrives so (unmeasured).
    class Untyped:
        def execute_arrow(self, sql, timeout_s=None):
            raise RuntimeError("Query execution was interrupted")

    with pytest.raises(ExecutionError) as caught:
        run(Untyped(), "SELECT 1", timeout_s=20)
    assert "timed out after 20s" in caught.value.message, caught.value.message


@pytest.mark.parametrize("candidates", [1, 5], ids=["single", "deep"])
def test_a_cancel_quoting_interrupt_ends_the_turn_though_the_model_would_write_it_again(
        tmp_path, monkeypatch, candidates):
    monkeypatch.setattr(sys.modules[__name__], "_REPLIES", _SAME)
    source = _AttachFaultAtRun(_duckdb(tmp_path), at=1, message=_CANCEL_QUOTING)
    answer, generator = _ask(source, candidates=candidates)
    _is_the_cancel(answer)
    assert source.runs == 1, "the cancelled statement ran again"
    assert len(generator.calls) == 1, generator.calls


@pytest.mark.parametrize("candidates", [1, 5], ids=["single", "deep"])
def test_a_drop_quoting_interrupt_ends_the_turn_and_is_not_run_again(
        tmp_path, monkeypatch, candidates):
    monkeypatch.setattr(sys.modules[__name__], "_REPLIES", _SAME)
    source = _AttachFaultAtRun(_duckdb(tmp_path), at=1, message=_DROP_QUOTING)
    answer, generator = _ask(source, candidates=candidates)
    _is_the_outage(answer)
    assert source.runs == 1, "the statement ran again after the source dropped it"
    assert len(generator.calls) == 1, generator.calls

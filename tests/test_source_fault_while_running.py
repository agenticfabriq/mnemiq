"""A source that stops answering while a query runs is an outage, not the query's failure (M188).

The proof seam reads an unanswered proof as SOURCE_UNAVAILABLE (#159, #168), but a connection that
dropped after the proof, or mid-fetch, reached the runner as "The source rejected this query." and
the loop handed it to the model as repair feedback: a correct query rewritten because the network
failed. Measured on the runtime's Postgres path (DuckDB's attach of a live Postgres, reconnects
refused): a duckdb IOException, which the adapter's own `source_unavailable` already called an
outage, and nobody asked it there. With reconnects allowed the very next run succeeded, which is
why the loop re-runs the same approved statement once, without the model, before it gives up.

Real DuckDB sources whose connection closes at a chosen run; Oracle's and SQLite's timeouts through
their own `execute_arrow`.
"""

from __future__ import annotations

import sqlite3
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

    def adapter(raising):
        """`execute_arrow` alone, over a lease and a cursor that raises: no database."""
        bare = OracleAdapter.__new__(OracleAdapter)
        bare._oracledb = oracledb

        class Cursor:
            connection = type("Connection", (), {"call_timeout": 0})()

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


@pytest.mark.parametrize("elapsed, reading", [(0.5, "outage"), (5.0, "timeout")])
def test_oracles_closed_connection_is_an_outage_before_the_deadline(oracle, monkeypatch, elapsed, reading):
    # DPY-4011, the database or network closed the connection, was read as a timeout whatever the
    # clock until the runner read the adapter's outage classification (M187's note). Before the
    # deadline nothing the bound did closed it, so it is the source's; at or after the deadline it
    # stays the call's end, as ORA-01013 does.
    oracledb, driver_error, adapter = oracle
    closed = driver_error("DPY-4011: the database or network closed the connection", 4011,
                          oracledb.DatabaseError)
    clock = iter([0.0, 100.0, 100.0 + elapsed])   # the lease's wait, the call's start, its end
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(ExecutionError) as caught:
        run(adapter(_raise(closed)), "SELECT 1 FROM dual", timeout_s=5)
    if reading == "outage":
        assert isinstance(caught.value, SourceUnavailableError), caught.value
    else:
        assert not isinstance(caught.value, SourceUnavailableError)
        assert "timed out" in caught.value.message


# -- the loop --------------------------------------------------------------------------------


def test_a_source_down_at_the_run_fails_the_turn_without_asking_the_model_to_repair(tmp_path):
    source = _DropsAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source)
    assert len(generator.calls) == 1, generator.calls
    assert source.runs == 2, "one re-run of the same statement, then the outage"
    _is_the_outage(answer)


def test_a_connection_that_drops_once_is_answered_by_the_rerun(tmp_path):
    source = _DropsOnceAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source)
    assert answer.failed is False and answer.deferred is False, answer
    assert len(generator.calls) == 1, "the re-run is the same statement: no model call"
    assert source.runs == 2


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
    assert source.runs == 2
    _is_the_outage(answer)
    assert answer.candidates_executed == 0


def test_deep_mode_fails_the_turn_when_the_source_goes_down_after_a_candidate_ran(tmp_path):
    source = _DropsAtRun(_duckdb(tmp_path), at=2)
    answer, generator = _ask(source, candidates=5)
    assert len(generator.calls) == 2, generator.calls
    _is_the_outage(answer)
    assert answer.candidates_executed == 1


def test_deep_mode_keeps_a_candidate_whose_rerun_answers(tmp_path):
    source = _DropsOnceAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source, candidates=5)
    assert answer.failed is False, answer
    assert answer.candidates_executed == 5
    assert len(generator.calls) == 5


# -- a cancel the bound did not send: neither an outage nor re-run (the review gate on M188) ------
#
# Measured on the Postgres attach: `pg_cancel_backend` from another session raises DuckDB's
# IOException ("canceling statement due to user request"), which the classifier calls unavailable,
# and Oracle's ORA-01013 before the deadline is an OperationalError, which its classifier calls
# unavailable too. Read as outages they were re-run, undoing a cancel aimed at a runaway query. A
# cancel keeps the reading it had before M188, a failure the repair loop handles, now named.


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

    class Cancelled:
        def source_unavailable(self, exc):
            return True  # what the classifiers answer for the driver's own cancel

        def execute_arrow(self, sql, timeout_s=None):
            raise QueryCancelled("the source cancelled the query: canceling statement due to user request")

    with pytest.raises(ExecutionError) as caught:
        run(Cancelled(), "SELECT 1")
    assert not isinstance(caught.value, SourceUnavailableError)
    assert "cancelled" in caught.value.message and "timed out" not in caught.value.message


def test_oracles_cancel_before_the_deadline_is_a_cancel_and_not_an_outage(oracle, monkeypatch):
    oracledb, driver_error, adapter = oracle
    cancel = driver_error("ORA-01013: user requested cancel of current operation", 1013,
                          oracledb.OperationalError)
    clock = iter([0.0, 100.0, 100.5])   # ended half a second into a 5 s bound
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(ExecutionError) as caught:
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


def test_a_cancelled_statement_is_not_rerun_as_it_was(tmp_path):
    source = _CancelledAtRun(_duckdb(tmp_path), at=1)
    answer, generator = _ask(source)
    assert answer.reason_code is not DeferralReason.SOURCE_UNAVAILABLE, answer
    assert len(generator.calls) == 2, "the repair loop's next attempt, as before M188"
    assert source.sqls[0] != source.sqls[1], "the cancelled statement itself was run again"

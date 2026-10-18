"""Codex's review of #159: a proof the source could not answer was read as a refusal.

`prove` read every exception as the source saying no. So a connection fault on the second proof
of a policy's rewrite blamed the policy (POLICY_REWRITE_REJECTED, a configuration fault reported to
the operator), and one on the proof of the statement as asked sent a statement no source had read
back to the model as EXPLAIN_FAILED, M158's waste again. A proof now has three outcomes, and the
third, unavailable, is the adapter's reading of its driver's errors: SOURCE_UNAVAILABLE, not
repairable, nobody's statement to fix. Both measured through `prove_governed` with scripted
faults, and through both deciders and the agent with a DuckDB source whose connection is closed.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.duckdb import unavailable_fault as duckdb_unavailable
from mnemiq.adapters.oracle import unavailable_fault as oracle_unavailable
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.prove import Proof, prove, prove_governed, prove_outcome
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import REPAIRABLE, Refusal, RefusalCode


class _Scripted:
    """A source answering each proof as scripted: None accepts, an exception is raised."""

    dialect = "duckdb"

    def __init__(self, *script):
        self.script, self.proofs = list(script), []

    def execute(self, sql):
        self.proofs.append(sql)
        outcome = self.script.pop(0)
        if outcome is not None:
            raise outcome
        return []


REWRITE = "SELECT id FROM (SELECT * FROM claim WHERE who = 1) AS claim"
ASKED = "SELECT id FROM claim"


def _stale():
    return duckdb.BinderException('Binder Error: Referenced column "who" not found')


def _down():
    return ConnectionError("connection reset by peer")


def _slow():
    return TimeoutError("the proof timed out")


@pytest.mark.parametrize("script, code, proofs", [
    # Codex's first case began here: unanswered on the first proof, which used to be proved twice
    # more and, when the source answered the second time, approved.
    ([_down()], RefusalCode.SOURCE_UNAVAILABLE, 1),
    # Codex's second case: EXPLAIN_FAILED before, a statement no source read sent to the model.
    ([_stale(), _down()], RefusalCode.SOURCE_UNAVAILABLE, 2),
    # The re-proof unanswered: POLICY_REWRITE_REJECTED before, the operator told to fix a policy.
    ([_stale(), None, _slow()], RefusalCode.SOURCE_UNAVAILABLE, 3),
    ([_stale(), None, _stale()], RefusalCode.POLICY_REWRITE_REJECTED, 3),
    ([_stale(), _stale()], RefusalCode.EXPLAIN_FAILED, 2),
    ([_stale(), None, None], None, 3),
    ([None], None, 1),
])
def test_each_proof_outcome_lands_where_the_fault_is(script, code, proofs):
    source = _Scripted(*script)
    verdict = prove_governed(source, REWRITE, ASKED)
    if code is None:
        assert verdict is None, verdict
    else:
        assert isinstance(verdict, Refusal) and verdict.code is code, verdict
    assert len(source.proofs) == proofs, source.proofs


def test_an_unanswered_proof_is_nobody_s_to_repair_and_keeps_the_driver_s_words_for_the_log():
    verdict = prove_governed(_Scripted(_down()), ASKED, ASKED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SOURCE_UNAVAILABLE
    assert not verdict.repairable and RefusalCode.SOURCE_UNAVAILABLE not in REPAIRABLE
    assert "reset by peer" not in verdict.message
    assert "reset by peer" in verdict.source_detail
    assert prove(_Scripted(_down()), ASKED).code is RefusalCode.SOURCE_UNAVAILABLE


def test_an_exception_nobody_classified_keeps_its_reading_as_a_refusal():
    # The conservative default, without an adapter's word: only Python's own connection and
    # timeout errors are unavailability, so an unknown fault still reaches the repair loop.
    assert prove_outcome(_Scripted(RuntimeError("something else")), ASKED).proof is Proof.REJECTED
    assert prove_outcome(_Scripted(_down()), ASKED).proof is Proof.UNAVAILABLE


def test_the_adapter_s_reading_of_its_driver_is_the_whole_answer():
    class _Told(_Scripted):
        def source_unavailable(self, exc):
            return "pool" in str(exc)

    assert prove_outcome(_Told(RuntimeError("pool exhausted")), ASKED).proof is Proof.UNAVAILABLE
    assert prove_outcome(_Told(_down()), ASKED).proof is Proof.REJECTED, "the adapter's word, whole"


def test_duckdb_tells_a_closed_connection_and_an_interrupt_from_a_statement_s_faults():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT)")
    faults = {}
    for name, sql in (("binder", "EXPLAIN SELECT whom FROM claim"),
                      ("catalog", "EXPLAIN SELECT id FROM nope"),
                      ("parser", "EXPLAIN SELECT FROM")):
        with pytest.raises(duckdb.Error) as caught:
            con.execute(sql)
        faults[name] = caught.value
    con.close()
    with pytest.raises(duckdb.Error) as caught:
        con.execute("EXPLAIN SELECT 1")
    faults["closed"] = caught.value
    assert [name for name, exc in faults.items() if duckdb_unavailable(exc)] == ["closed"]
    # What a timeout raises here (`execute_bounded`), and a locked file.
    assert duckdb_unavailable(duckdb.InterruptException("INTERRUPT Error: Interrupted!"))
    assert duckdb_unavailable(duckdb.IOException("IO Error: Could not set lock on file"))
    assert duckdb_unavailable(ConnectionError("reset"))


def _dpy(number):
    """The driver's own exception for one of its DPY codes, typed as the driver types it."""
    from oracledb import errors

    error = errors._create_err(number)
    return error.exc_type(error)


def test_oracle_tells_its_network_faults_and_call_timeout_from_a_parse_refusal():
    pytest.importorskip("oracledb")
    from oracledb import errors

    def ora(code, message):
        error = errors._Error(message=message, code=code)
        return error.exc_type(error)

    from mnemiq.adapters.oracle import _UNAVAILABLE_DPY

    # The set was read off this table. A driver whose table differs may name a connection fault
    # the set does not, so its size is pinned: on a driver change, re-read the table, then move
    # both numbers.
    assert len(errors.ERR_MESSAGE_FORMATS) == 190, "the driver's error table changed: re-read it"
    # The driver's own faults that are the statement's or the caller's, by the same classes the
    # unavailable ones carry: invalid number, cannot convert, a cursor not open, an unsupported
    # Python type, too many batch errors.
    refused = [ora(904, 'ORA-00904: "WHOM": invalid identifier'),
               ora(942, "ORA-00942: table or view does not exist"),
               ora(936, "ORA-00936: missing expression"),
               ora(1, "ORA-00001: unique constraint violated"),
               _dpy(4004), _dpy(4007), _dpy(1006), _dpy(3002), _dpy(4029),
               RuntimeError("refusing to validate a CREATE statement")]
    # Every code the adapter names, listed here on its own so the two lists must agree, built as
    # the driver builds it: the pool's acquisition timeout DPY-4005 (`_lease` under contention)
    # and the idle disconnect DPY-4033 are plain non-recoverable DatabaseErrors by the driver's
    # typing (Codex's second and third reviews of #159).
    named = [_dpy(n) for n in (1001, 1002, 4000, 4001, 4005, 4011, 4017, 4018, 4019, 4021, 4024,
                               4026, 4027, 4028, 4030, 4032, 4033, 4034, 4035, 5004, 5005, 5006)]
    assert {str(exc)[:8] for exc in named} == _UNAVAILABLE_DPY
    unanswered = [*named, _dpy(6005),
                  ora(3113, "ORA-03113: end-of-file on communication channel"),
                  ora(1013, "ORA-01013: user requested cancel of current operation"),
                  ora(3156, "ORA-03156: the call timed out"),
                  # What a growing pool meets on acquire after the credentials changed under it.
                  ora(1017, "ORA-01017: invalid credential or not authorized; logon denied"),
                  ora(28000, "ORA-28000: The account is locked."),
                  ora(28001, "ORA-28001: the password has expired"),
                  ora(12541, "ORA-12541: TNS:no listener"),
                  ora(12170, "ORA-12170: TNS:Connect timeout occurred")]
    assert [str(exc)[:9] for exc in refused if oracle_unavailable(exc)] == []
    assert [str(exc)[:9] for exc in unanswered if not oracle_unavailable(exc)] == []


class _OracleScripted(_Scripted):
    """A source proving as Oracle does (`validate`, the parse-only path) and reading its faults
    with the Oracle adapter's own classifier."""

    dialect = "oracle"

    def validate(self, sql):
        self.execute(sql)

    @staticmethod
    def source_unavailable(exc):
        return oracle_unavailable(exc)


@pytest.mark.parametrize("code", [4005, 4033], ids=["pool exhausted", "idle disconnect"])
@pytest.mark.parametrize("script", [
    ["fault"], ["stale", "fault"], ["stale", None, "fault"],
], ids=["first proof", "proof as asked", "re-proof of the rewrite"])
def test_an_exhausted_pool_or_an_idle_disconnect_is_unavailable_at_every_proof_position(
        script, code):
    """Before, DPY-4005 and DPY-4033 read as a refusal: EXPLAIN_FAILED at the first two
    positions, a statement no session parsed sent to the model; POLICY_REWRITE_REJECTED at the
    third, the operator told to fix a policy."""
    pytest.importorskip("oracledb")
    faults = [_dpy(code) if step == "fault" else (_stale() if step else None)
              for step in script]
    source = _OracleScripted(*faults)
    verdict = prove_governed(source, REWRITE, ASKED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SOURCE_UNAVAILABLE, verdict
    assert not verdict.repairable
    assert len(source.proofs) == len(script), source.proofs


def test_through_both_deciders_an_exhausted_oracle_pool_is_unavailable_not_the_model_s():
    pytest.importorskip("oracledb")
    snapshot = _snapshot()
    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}),
                      row_filters={"claim": ENTITLED})
    read = decide("SELECT id FROM claim", visible_schema(snapshot, grants),
                  adapter=_OracleScripted(_dpy(4005)), dialect="oracle", target="oracle",
                  policy=build_access_policy(snapshot, grants))
    assert isinstance(read, Refusal) and read.code is RefusalCode.SOURCE_UNAVAILABLE, read
    assert not read.repairable
    # No row filter on the write: on Oracle an UPDATE of a filtered table is refused before any
    # proof, as a limit of the deployment, so the proof is reached only on a plain grant.
    plain = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    write = decide_write("UPDATE claim SET amount = 0 WHERE id = 1",
                         visible_schema(snapshot, plain), plain,
                         policy=build_access_policy(snapshot, plain),
                         adapter=_OracleScripted(_dpy(4005)), dialect="oracle", target="oracle",
                         writes_enabled=True)
    assert isinstance(write, Refusal) and write.code is RefusalCode.SOURCE_UNAVAILABLE, write


def test_postgres_tells_a_lost_connection_and_a_cancelled_statement_from_a_statement_s_faults():
    psycopg = pytest.importorskip("psycopg")
    from mnemiq.adapters.pg import unavailable_fault as postgres_unavailable

    refused = [psycopg.errors.UndefinedColumn('column "whom" does not exist'),
               psycopg.errors.UndefinedTable('relation "nope" does not exist'),
               psycopg.errors.SyntaxError('syntax error at end of input'),
               psycopg.errors.InsufficientPrivilege("permission denied for table claim")]
    unanswered = [psycopg.OperationalError("server closed the connection unexpectedly"),
                  psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
                  psycopg.errors.AdminShutdown("terminating connection due to administrator command"),
                  ConnectionError("reset")]
    assert [type(exc).__name__ for exc in refused if postgres_unavailable(exc)] == []
    assert [type(exc).__name__ for exc in unanswered if not postgres_unavailable(exc)] == []


def _snapshot():
    names = [("claim", "id"), ("claim", "amount"), ("entitlement", "id"), ("entitlement", "who")]
    return Snapshot(version="v1", source_id="s", created_at="2026-10-17T00:00:00Z",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in names])


class _DropsAtTheProof:
    """The real DuckDB source, whose connection closes as the first proof is asked: the fault
    DuckDB then raises is its own. (Closed before the decider starts, the source fails earlier,
    at the function inventory, which is another seam's reading of the same outage.)"""

    def __init__(self, inner):
        self._inner, self.proofs = inner, []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute(self, sql):
        if sql.startswith("EXPLAIN "):
            self.proofs.append(sql)
            self._inner._con.close()
        return self._inner.execute(sql)


def _closed_source(tmp_path, name="s"):
    path = str(tmp_path / f"{name}.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, amount INT); CREATE TABLE entitlement(id INT, who VARCHAR)")
    con.close()
    return _DropsAtTheProof(DuckDBAdapter.duckdb(path, read_only=False))


ENTITLED = "id IN (SELECT id FROM entitlement WHERE who = 'r')"


def test_through_both_deciders_a_closed_source_is_unavailable_not_the_model_s(tmp_path):
    adapter = _closed_source(tmp_path)
    snapshot = _snapshot()
    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}),
                      row_filters={"claim": ENTITLED})
    read = decide("SELECT id FROM claim", visible_schema(snapshot, grants), adapter=adapter,
                  dialect="duckdb", target="duckdb", policy=build_access_policy(snapshot, grants))
    assert isinstance(read, Refusal) and read.code is RefusalCode.SOURCE_UNAVAILABLE, read
    assert not read.repairable
    adapter = _closed_source(tmp_path, "w")  # its own source: the first one's connection is gone
    write = decide_write("UPDATE claim SET amount = 0 WHERE id = 1",
                         visible_schema(snapshot, grants), grants,
                         policy=build_access_policy(snapshot, grants), adapter=adapter,
                         dialect="duckdb", target="duckdb", writes_enabled=True)
    assert isinstance(write, Refusal) and write.code is RefusalCode.SOURCE_UNAVAILABLE, write


def test_through_the_agent_one_model_call_and_a_deferral_that_says_the_source_did_not_answer(tmp_path):
    """Before, the closed connection read as EXPLAIN_FAILED: three model calls repairing a sound
    statement, and the caller told the model had failed."""
    adapter = _closed_source(tmp_path)
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": ENTITLED})
    generator = FakeGenerator(['{"sql": "SELECT id FROM claim", "reason": "ok"}'] * 3)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=adapter,
                  cache=TwoTierCache(L1Cache()))
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    answer = agent.answer(packet, _snapshot(), grants,
                          IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    assert len(generator.calls) == 1, generator.calls
    assert answer.deferred, answer
    assert "did not answer" in answer.answer, answer.answer
    assert "already closed" not in answer.answer, "the driver's words stay in the server log"

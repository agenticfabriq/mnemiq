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


def _ora(code, message):
    """The driver's exception for a server error, typed as the driver types it."""
    from oracledb import errors

    error = errors._Error(message=message, code=code)
    return error.exc_type(error)


FORCED_LOGOFF = (2399, "ORA-02399: exceeded maximum connect time, you are being logged off")
NO_AUTH_PROTOCOL = (28040, "ORA-28040: No matching authentication protocol")
WRAPPED_REFUSAL = (604, "ORA-00604: error occurred at recursive SQL level 1\n"
                        "ORA-00942: table or view does not exist")
QUOTED_OUTAGE = (604, "ORA-00604: error occurred at recursive SQL level 1\n"
                      "ORA-20001: statement refused; diagnostic example ORA-03113\n"
                      "ORA-06512: at line 2")
WRAPPED_OUTAGE = (604, "ORA-00604: error occurred at recursive SQL level 1\n"
                       "ORA-03113: end-of-file on communication channel")


def _fault(spec):
    return _dpy(spec) if isinstance(spec, int) else _ora(*spec)


def test_oracle_tells_its_network_faults_and_call_timeout_from_a_parse_refusal():
    pytest.importorskip("oracledb")
    from oracledb import errors

    def ora(code, message):
        error = errors._Error(message=message, code=code)
        return error.exc_type(error)

    from mnemiq.adapters.oracle import _UNAVAILABLE_DPY

    def spans(*pairs):
        return {n for lo, hi in pairs for n in range(lo, hi + 1)}

    # Every code in the driver's table, classified: the adapter's set, the OperationalErrors
    # (unavailable by type), or this list, reviewed as the statement's or the caller's. The
    # cursor and fetch states; the caller's usage (ProgrammingError) and the unsupported types,
    # less the connect-time ones among them; the data faults, bind errors and Arrow conversions;
    # the driver's internal checks that decode a value or belong to the driver or the caller,
    # less the protocol faults a proof can meet; the warning and the data errors. By exact span,
    # so a code the driver adds, renumbers or removes fails the equality: re-read the table, then
    # move both lists.
    refused_codes = spans((1003, 1007), (2000, 2017), (2019, 2023), (2025, 2032), (2034, 2048),
                          (2050, 2055), (2058, 2075), (3000, 3007), (3009, 3010), (3012, 3014),
                          (3016, 3041),
                          (4002, 4004), (4006, 4010), (4012, 4016), (4036, 4041), (5011, 5013),
                          (8000, 8002)) | {4022, 4025, 4029, 4031, 5007, 5015, 7000}
    operational_codes = spans((6000, 6007))
    unavailable_codes = {int(code.removeprefix("DPY-")) for code in _UNAVAILABLE_DPY}
    assert unavailable_codes.isdisjoint(refused_codes)
    assert unavailable_codes | operational_codes | refused_codes == set(errors.ERR_MESSAGE_FORMATS), (
        "a driver code is unclassified or gone: re-read the driver's error table")
    assert [n for n in sorted(refused_codes) if oracle_unavailable(_dpy(n))] == []
    assert [n for n in sorted(unavailable_codes | operational_codes)
            if not oracle_unavailable(_dpy(n))] == []

    # The driver's own faults that are the statement's or the caller's, by the same classes the
    # unavailable ones carry: invalid number, cannot convert, a cursor not open, an unsupported
    # Python type, too many batch errors, an unexpected XMLType flag, a number too long to decode,
    # a feature not implemented.
    refused = [ora(904, 'ORA-00904: "WHOM": invalid identifier'),
               ora(942, "ORA-00942: table or view does not exist"),
               # A SERVERERROR trigger failing on a missing table wraps the statement's own
               # refusal (Codex's eighth review): no outage among the wrapped errors.
               ora(604, "ORA-00604: error occurred at recursive SQL level 1\n"
                        "ORA-00942: table or view does not exist\n"
                        "ORA-06512: at line 4"),
               # Codex's ninth review: a code quoted in an application error's text is prose.
               ora(604, "ORA-00604: error occurred at recursive SQL level 1\n"
                        "ORA-20001: statement refused; diagnostic example ORA-03113\n"
                        "ORA-06512: at line 2"),
               ora(936, "ORA-00936: missing expression"),
               ora(1, "ORA-00001: unique constraint violated"),
               _dpy(4004), _dpy(4007), _dpy(1006), _dpy(3002), _dpy(4029), _dpy(5007), _dpy(5015),
               _dpy(5012),
               RuntimeError("refusing to validate a CREATE statement")]
    # Every code the adapter names, listed here on its own so the two lists must agree, built as
    # the driver builds it: the pool's acquisition timeout DPY-4005 (`_lease` under contention)
    # and the idle disconnect DPY-4033 are plain non-recoverable DatabaseErrors by the driver's
    # typing (Codex's second and third reviews of #159); the response parser's unknown message
    # type and piggyback opcode (DPY-5000, 5009) are InternalErrors (its fourth); the unsupported
    # in-band notification the read buffer disconnects on (DPY-3008) a NotSupportedError (its
    # fifth).
    named = [_dpy(n) for n in (1001, 1002, 2018, 2033, 2049, 2056, 2057, 3008, 3015, 4000, 4001,
                               4005, 4011, 4017, 4018, 4019, 4021, 4024, 4026, 4027, 4028, 4030,
                               4032, 4033, 4034, 4035, 5000, 5001, 5002, 5003, 5004, 5005, 5006,
                               5009, 5010, 5014)]
    assert {str(exc)[:8] for exc in named} == _UNAVAILABLE_DPY
    unanswered = [*named, _dpy(6005),
                  ora(3113, "ORA-03113: end-of-file on communication channel"),
                  ora(1013, "ORA-01013: user requested cancel of current operation"),
                  ora(3156, "ORA-03156: the call timed out"),
                  # What a growing pool meets on acquire: the credentials and the account, the
                  # instance's state, the session and process limits (Codex's sixth review of
                  # #159 found ORA-01045 among them).
                  ora(1017, "ORA-01017: invalid credential or not authorized; logon denied"),
                  ora(1045, "ORA-01045: user APP lacks CREATE SESSION privilege; logon denied"),
                  ora(28000, "ORA-28000: The account is locked."),
                  ora(28001, "ORA-28001: the password has expired"),
                  ora(1033, "ORA-01033: ORACLE initialization or shutdown in progress"),
                  ora(1034, "ORA-01034: ORACLE not available"),
                  ora(1035, "ORA-01035: ORACLE only available to users with RESTRICTED SESSION privilege"),
                  ora(1089, "ORA-01089: immediate shutdown or close in progress - no operations are permitted"),
                  ora(1090, "ORA-01090: shutdown in progress - connection is not permitted"),
                  ora(1092, "ORA-01092: ORACLE instance terminated. Disconnection forced"),
                  ora(18, "ORA-00018: maximum number of sessions exceeded"),
                  ora(20, "ORA-00020: maximum number of processes (300) exceeded"),
                  ora(2391, "ORA-02391: exceeded simultaneous SESSIONS_PER_USER limit"),
                  # Codex's seventh review: the profile's connect time, and an authentication
                  # protocol the server no longer accepts.
                  ora(2399, "ORA-02399: exceeded maximum connect time, you are being logged off"),
                  ora(28040, "ORA-28040: No matching authentication protocol"),
                  # Codex's eighth review: ORA-00604 wraps the recursive SQL's own error, which
                  # the driver's OperationalError typing of the outer code says nothing about.
                  ora(604, "ORA-00604: error occurred at recursive SQL level 1\n"
                           "ORA-03113: end-of-file on communication channel"),
                  ora(604, "ORA-00604: error occurred at recursive SQL level 1\n"
                           "ORA-00028: your session has been killed"),
                  # Naming no other code, it keeps the driver's OperationalError reading.
                  ora(604, "ORA-00604: error occurred at recursive SQL level 1"),
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


@pytest.mark.parametrize("code", [4005, 4033, 5000, 5009, 3008, FORCED_LOGOFF, NO_AUTH_PROTOCOL,
                                  WRAPPED_OUTAGE],
                         ids=["pool exhausted", "idle disconnect", "unknown message type",
                              "unknown piggyback", "in-band notification disconnect",
                              "forced logoff", "no authentication protocol",
                              "ORA-00604 wrapping a lost channel"])
@pytest.mark.parametrize("script", [
    ["fault"], ["stale", "fault"], ["stale", None, "fault"],
], ids=["first proof", "proof as asked", "re-proof of the rewrite"])
def test_a_pool_session_or_protocol_fault_is_unavailable_at_every_proof_position(script, code):
    """Before, DPY-4005 and DPY-4033 (Codex's second and third reviews), DPY-5000 and DPY-5009
    (its fourth) and ORA-02399 and ORA-28040 (its seventh) read as a refusal: EXPLAIN_FAILED at
    the first two positions, a statement no session parsed sent to the model;
    POLICY_REWRITE_REJECTED at the third, the operator told to fix a policy."""
    pytest.importorskip("oracledb")
    faults = [_fault(code) if step == "fault" else (_stale() if step else None)
              for step in script]
    source = _OracleScripted(*faults)
    verdict = prove_governed(source, REWRITE, ASKED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SOURCE_UNAVAILABLE, verdict
    assert not verdict.repairable
    assert len(source.proofs) == len(script), source.proofs


@pytest.mark.parametrize("code", [4005, FORCED_LOGOFF, NO_AUTH_PROTOCOL],
                         ids=["pool exhausted", "forced logoff", "no authentication protocol"])
def test_through_both_deciders_an_exhausted_oracle_pool_is_unavailable_not_the_model_s(code):
    pytest.importorskip("oracledb")
    snapshot = _snapshot()
    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}),
                      row_filters={"claim": ENTITLED})
    read = decide("SELECT id FROM claim", visible_schema(snapshot, grants),
                  adapter=_OracleScripted(_fault(code)), dialect="oracle", target="oracle",
                  policy=build_access_policy(snapshot, grants))
    assert isinstance(read, Refusal) and read.code is RefusalCode.SOURCE_UNAVAILABLE, read
    assert not read.repairable
    # No row filter on the write: on Oracle an UPDATE of a filtered table is refused before any
    # proof, as a limit of the deployment, so the proof is reached only on a plain grant.
    plain = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    write = decide_write("UPDATE claim SET amount = 0 WHERE id = 1",
                         visible_schema(snapshot, plain), plain,
                         policy=build_access_policy(snapshot, plain),
                         adapter=_OracleScripted(_fault(code)), dialect="oracle", target="oracle",
                         writes_enabled=True)
    assert isinstance(write, Refusal) and write.code is RefusalCode.SOURCE_UNAVAILABLE, write


@pytest.mark.parametrize("wrapped", [WRAPPED_REFUSAL, QUOTED_OUTAGE],
                         ids=["a missing table", "an application error quoting an outage code"])
@pytest.mark.parametrize("script", [
    ["fault"], ["stale", "fault"], ["stale", None, "fault"],
], ids=["first proof", "proof as asked", "re-proof of the rewrite"])
def test_a_statement_s_refusal_wrapped_in_ora_00604_keeps_its_reading(script, wrapped):
    """Codex's eighth review of #159: the driver types ORA-00604 OperationalError, and the
    classifier read every OperationalError as the source being down, so a SERVERERROR trigger
    failing on a missing table turned a refusal into an outage: repair suppressed, or a policy
    rewrite blamed. Read by the errors it wraps instead; marked on acquire it is still the
    source's."""
    pytest.importorskip("oracledb")
    faults = [_ora(*wrapped) if step == "fault" else (_stale() if step else None)
              for step in script]
    source = _OracleScripted(*faults)
    verdict = prove_governed(source, REWRITE, ASKED)
    assert isinstance(verdict, Refusal) and verdict.code is not RefusalCode.SOURCE_UNAVAILABLE, verdict
    from mnemiq.adapters.oracle import _mark_acquire_fault

    marked = _ora(*wrapped)
    _mark_acquire_fault(marked)
    assert oracle_unavailable(marked)


def test_a_fault_on_the_pool_s_acquire_is_unavailable_whatever_its_code():
    """Codex's seventh review of #159: a pool that cannot authenticate should not depend on the
    code list naming its error. No session was had, so no statement was asked: the lease marks
    the exception and the classifier reads the mark before any list. The same error raised by a
    statement keeps its reading."""
    oracledb = pytest.importorskip("oracledb")
    from mnemiq.adapters.oracle import OracleAdapter

    # A code no list names, typed IntegrityError by the driver (ORA-00604 is typed
    # OperationalError, unavailable on its own).
    unlisted = (1, "ORA-00001: unique constraint violated")
    assert not oracle_unavailable(_ora(*unlisted))

    class _RaisingPool:
        def __init__(self, exc):
            self.exc = exc

        def acquire(self):
            raise self.exc

    adapter = OracleAdapter.over(object(), oracledb, schema="APP")
    adapter._pool = _RaisingPool(_ora(*unlisted))
    with pytest.raises(oracledb.DatabaseError) as caught:
        with adapter._lease():
            pytest.fail("no connection to lease")
    assert oracle_unavailable(caught.value)
    assert caught.value.args[0].code == 1
    # A bug in the pool, or in a stand-in for it, is not an outage (the review gate).
    adapter._pool = _RaisingPool(TypeError("acquire() takes no arguments"))
    with pytest.raises(TypeError) as bug:
        with adapter._lease():
            pytest.fail("no connection to lease")
    assert not oracle_unavailable(bug.value)


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


def test_through_the_agent_one_model_call_and_a_failure_that_says_the_source_did_not_answer(tmp_path):
    """Before, the closed connection read as EXPLAIN_FAILED: three model calls repairing a sound
    statement, and the caller told the model had failed. A failure, not a deferral: M6."""
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
    assert answer.failed and not answer.deferred, answer
    assert "did not answer" in answer.answer, answer.answer
    assert "already closed" not in answer.answer, "the driver's words stay in the server log"

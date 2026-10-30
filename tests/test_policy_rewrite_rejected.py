"""M158: when the source refuses only what the access policy added, the refusal is the operator's.

A row filter gone stale against its source -- `whom` in the snapshot, `who` in the source -- fails
the proof of every statement reading its table. That came back EXPLAIN_FAILED, which is repairable,
so each of three model calls was spent repairing a statement that was never wrong, and the caller
was told the model could not produce a valid query. Now, when the rewrite is refused and the
statement as asked is accepted, both deciders refuse POLICY_REWRITE_REJECTED: not repairable,
deferred as UNGOVERNABLE after one model call, with the source's words in the server log only.
"""

import os

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import REPAIRABLE, Approved, Refusal, RefusalCode

ENTITLED = "id IN (SELECT id FROM entitlement WHERE who = 'secret_role_42')"
STALE = "id IN (SELECT id FROM entitlement WHERE whom = 'secret_role_42')"


class _Counting:
    """The real DuckDB source, counting the proofs it is asked for."""

    def __init__(self, inner):
        self.inner, self.proofs = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def execute(self, sql):
        if sql.startswith("EXPLAIN "):
            self.proofs.append(sql)
        return self.inner.execute(sql)


def _source(tmp_path, read_only=True):
    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, amount INT);"
                "CREATE TABLE entitlement(id INT, who VARCHAR);"
                "INSERT INTO claim VALUES (1, 'west', 10), (2, 'east', 20);"
                "INSERT INTO entitlement VALUES (1, 'secret_role_42')")
    con.close()
    return _Counting(DuckDBAdapter.duckdb(path, read_only=read_only))


@pytest.fixture
def source(tmp_path):
    return _source(tmp_path)


def _snapshot(*extra):
    names = [("claim", "id"), ("claim", "region"), ("claim", "amount"),
             ("entitlement", "id"), ("entitlement", "who"), *extra]
    return Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in names])


STALE_SNAPSHOT = _snapshot(("entitlement", "whom"))


def _grants(row_filter=None, writable=()):
    return GrantSet(frozenset({"claim"}), writable=frozenset(writable),
                    row_filters={"claim": row_filter} if row_filter else {})


def _decide(sql, source, *, row_filter, snapshot=STALE_SNAPSHOT):
    grants = _grants(row_filter)
    return decide(sql, visible_schema(snapshot, grants), adapter=source, dialect="duckdb",
                  target="duckdb", policy=build_access_policy(snapshot, grants))


def test_a_stale_row_filter_is_refused_as_the_policy_s_fault(source):
    verdict = _decide("SELECT id FROM claim", source, row_filter=STALE)
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code is RefusalCode.POLICY_REWRITE_REJECTED
    assert verdict.code not in REPAIRABLE
    assert verdict.source_detail is None
    for policy_word in ("whom", "entitlement", "secret_role_42", "Binder"):
        assert policy_word not in verdict.message, policy_word


def test_a_stale_snapshot_column_on_a_filtered_table_is_the_operator_s_too(source):
    """The policy is current; the snapshot holds `claim.status`, which the source lacks. The
    policy's derived table projects every visible column, so its rewrite names `status` and the
    statement as asked does not."""
    verdict = _decide("SELECT id FROM claim", source, row_filter=ENTITLED,
                      snapshot=_snapshot(("claim", "status")))
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code is RefusalCode.POLICY_REWRITE_REJECTED


def test_a_statement_the_source_refuses_on_its_own_stays_the_model_s_to_repair(source):
    """`claim.status` is in the snapshot and not in the source: the statement as asked fails too."""
    snapshot = _snapshot(("entitlement", "whom"), ("claim", "status"))
    verdict = _decide("SELECT status FROM claim", source, row_filter=STALE, snapshot=snapshot)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.EXPLAIN_FAILED, verdict
    assert verdict.code in REPAIRABLE


def test_only_a_refused_rewrite_costs_more_proofs(source):
    assert isinstance(_decide("SELECT id FROM claim", source, row_filter=ENTITLED), Approved)
    assert len(source.proofs) == 1, "a policy the source accepts: one proof, as before"
    source.proofs.clear()
    snapshot = _snapshot(("claim", "status"))
    refused = decide("SELECT status FROM claim", visible_schema(snapshot, _grants()),
                     adapter=source, dialect="duckdb", target="duckdb")
    assert isinstance(refused, Refusal) and refused.code is RefusalCode.EXPLAIN_FAILED
    assert len(source.proofs) == 1, "nothing narrowed: no statement as asked to compare"
    source.proofs.clear()
    _decide("SELECT id FROM claim", source, row_filter=STALE)
    # The rewrite, the statement as asked, and the rewrite once more before the policy is blamed.
    assert [("whom" in proof) for proof in source.proofs] == [True, False, True], source.proofs


class _OnceUnreachable(_Counting):
    """The first proof fails as a dropped connection would; every later one reaches the source."""

    def execute(self, sql):
        if sql.startswith("EXPLAIN ") and not self.proofs:
            self.proofs.append(sql)
            raise ConnectionError("connection reset by peer")
        return super().execute(sql)


def test_a_first_proof_the_source_does_not_answer_is_nobody_s_to_repair(tmp_path):
    """Read as a refusal, it was proved twice more and approved when the source answered again,
    and blamed on the policy when it did not (Codex's review of #159). Unanswered is its own
    outcome now: not the model's, not the policy's, and no further proof to compare against."""
    source = _OnceUnreachable(_source(tmp_path).inner)
    verdict = _decide("SELECT id FROM claim", source, row_filter=ENTITLED, snapshot=_snapshot())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SOURCE_UNAVAILABLE, verdict
    assert not verdict.repairable
    assert len(source.proofs) == 1, source.proofs


def test_a_write_under_a_stale_row_filter_is_refused_the_same_way(tmp_path):
    """On the read-write attachment a write runs on: a read-only one refuses every write's proof,
    the statement as asked included, so nothing there is the policy's alone."""
    source = _source(tmp_path, read_only=False)
    grants = _grants(STALE, writable={"claim"})
    verdict = decide_write("UPDATE claim SET amount = 0 WHERE id = 1",
                           visible_schema(STALE_SNAPSHOT, grants), grants,
                           policy=build_access_policy(STALE_SNAPSHOT, grants), adapter=source,
                           dialect="duckdb", target="duckdb", writes_enabled=True)
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code is RefusalCode.POLICY_REWRITE_REJECTED


def test_through_the_agent_one_model_call_and_an_operator_s_deferral(source):
    """The door the finding was measured on: three model calls before, and the caller told the
    model had failed."""
    grants = _grants(STALE)
    generator = FakeGenerator(['{"sql": "SELECT id FROM claim", "reason": "ok"}'] * 3)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=source,
                  cache=TwoTierCache(L1Cache()))
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    answer = agent.answer(packet, STALE_SNAPSHOT, grants,
                          IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    assert len(generator.calls) == 1, generator.calls
    assert answer.deferred and answer.reason_code is DeferralReason.UNGOVERNABLE, answer
    assert "configuration fault for the operator" in answer.answer


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_the_parse_proof_tells_the_policy_s_part_apart():
    """Oracle proves by `cursor.parse`, not EXPLAIN: the same split, through the real adapter."""
    oracledb = pytest.importorskip("oracledb")
    from mnemiq.adapters.oracle import OracleAdapter

    user = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
    password = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")
    admin = oracledb.connect(user=user, password=password, dsn=DSN)
    cur = admin.cursor()
    for table in ("m158_claim", "m158_entitlement"):
        cur.execute(f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; "
                    "EXCEPTION WHEN OTHERS THEN NULL; END;")
    cur.execute("CREATE TABLE m158_claim (id NUMBER, amount NUMBER)")
    cur.execute("CREATE TABLE m158_entitlement (id NUMBER, who VARCHAR2(30))")
    adapter = OracleAdapter(dsn=DSN, user=user, password=password)
    try:
        def decided(sql, row_filter, *stale_columns):
            names = [("M158_CLAIM", "id"), ("M158_CLAIM", "amount"), ("M158_ENTITLEMENT", "id"),
                     ("M158_ENTITLEMENT", "who"), *stale_columns]
            snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                                columns=[Column(id=f"{t}.{c}", object_id=t, name=c)
                                         for t, c in names])
            grants = GrantSet(frozenset({"M158_CLAIM"}), row_filters={"M158_CLAIM": row_filter})
            return decide(sql, visible_schema(snapshot, grants), adapter=adapter,
                          dialect="oracle", target="oracle",
                          policy=build_access_policy(snapshot, grants))

        whom = ("M158_ENTITLEMENT", "whom")
        status = ("M158_CLAIM", "status")
        stale = "id IN (SELECT id FROM M158_ENTITLEMENT WHERE whom = 'r')"
        current = "id IN (SELECT id FROM M158_ENTITLEMENT WHERE who = 'r')"
        assert isinstance(decided("SELECT id FROM M158_CLAIM", current), Approved), "control"
        for policy, columns in ((stale, (whom,)), (current, (status,))):
            refused = decided("SELECT id FROM M158_CLAIM", policy, *columns)
            assert isinstance(refused, Refusal), (policy, columns, refused)
            assert refused.code is RefusalCode.POLICY_REWRITE_REJECTED, (policy, columns)
        on_its_own = decided("SELECT status FROM M158_CLAIM", stale, whom, status)
        assert isinstance(on_its_own, Refusal) and on_its_own.code is RefusalCode.EXPLAIN_FAILED
    finally:
        adapter.close()
        for table in ("m158_claim", "m158_entitlement"):
            cur.execute(f"DROP TABLE {table}")
        admin.close()

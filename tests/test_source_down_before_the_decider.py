"""M180: a source down before the decider starts is an outage, not a fault worth three model calls.

The function inventory is the first thing the decider asks the source, so a source already down
fails there, and that seam read every raised catalogue call as a fault a retry may clear (M98):
UNRESOLVABLE_CALLS, repairable. Measured on a DuckDB source whose connection was closed before the
call: three model calls through the agent, eight on the deep path, and the caller told the source
was ungovernable (the ui lane's M208, folded into M180), where the proof seam reports the same
outage after one (#159). The inventory now asks the proof seam's question, `source_did_not_answer`,
and an outage is SOURCE_UNAVAILABLE; a fault the adapter does not know keeps the retry.
"""
from __future__ import annotations

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.functions import inventory_from
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode

VISIBLE = {"CLAIM": {"id", "amount"}}


def _down(tmp_path, name="s") -> DuckDBAdapter:
    path = str(tmp_path / f"{name}.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE CLAIM(id INT, amount INT)")
    con.close()
    adapter = DuckDBAdapter.duckdb(path, read_only=False)
    adapter._con.close()  # down before the decider asks anything
    return adapter


def _outage(verdict) -> None:
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SOURCE_UNAVAILABLE, verdict
    assert not verdict.repairable
    assert "already closed" not in verdict.message, "the driver's words stay in the server log"
    assert verdict.source_detail == "ConnectionException", verdict.source_detail  # the type alone (M94)


@pytest.mark.parametrize("sql", ["SELECT id FROM CLAIM", "SELECT upper(CAST(id AS VARCHAR)) FROM CLAIM"])
def test_a_source_down_before_the_read_decider_is_unavailable(tmp_path, sql):
    _outage(decide(sql, VISIBLE, adapter=_down(tmp_path), dialect="duckdb", target="duckdb"))


def test_a_source_down_before_the_write_decider_is_unavailable(tmp_path):
    grants = GrantSet(frozenset(VISIBLE), writable=frozenset(VISIBLE))
    _outage(decide_write("UPDATE CLAIM SET amount = 0 WHERE id = 1", VISIBLE, grants, policy=AccessPolicy(),
                         adapter=_down(tmp_path), dialect="duckdb", target="duckdb", writes_enabled=True))


class _Raising:
    """A source whose catalogue call raises `fault`, classified by its adapter as `told` says,
    or by Python's own connection errors where it has no word of its own (`told` None)."""

    dialect = "duckdb"

    def __init__(self, fault: Exception, told: bool | None, by_name: bool = False):
        self.fault = fault
        if told is not None:
            self.source_unavailable = lambda exc: told
        if by_name:  # the narrow, per-name answer (M101), Oracle's shape
            self.binder_prefers_builtins = True
            self.resolve_called_functions = self._raise

    def _raise(self, *args):
        raise self.fault

    def user_functions(self):
        raise self.fault

    def execute(self, sql):
        return []


@pytest.mark.parametrize("by_name", [False, True], ids=["whole source", "by name"])
@pytest.mark.parametrize(("fault", "told", "outage"), [
    (RuntimeError("pool exhausted"), True, True),  # the adapter's word, whole
    (ConnectionError("reset"), False, False),
    (ConnectionError("reset"), None, True),  # no word of its own: Python's connection errors
    (RuntimeError("catalogue view missing"), None, False),  # unknown: the retry M98 keeps
])
def test_the_adapter_says_which_fault_is_an_outage_and_an_unknown_one_keeps_the_retry(fault, told, outage, by_name):
    adapter = _Raising(fault, told, by_name)
    inventory = inventory_from(adapter, called=frozenset({"upper"}) if by_name else None)
    assert not inventory.available and inventory.unanswered is outage
    verdict = decide("SELECT id FROM CLAIM", VISIBLE, adapter=adapter, dialect="duckdb", target="duckdb")
    if outage:
        assert verdict.code is RefusalCode.SOURCE_UNAVAILABLE and not verdict.repairable, verdict
    else:
        assert verdict.code is RefusalCode.UNRESOLVABLE_CALLS and verdict.repairable, verdict


def test_through_the_agent_one_model_call_and_a_failure_that_says_the_source_did_not_answer(tmp_path):
    grants = GrantSet(frozenset(VISIBLE))
    generator = FakeGenerator(['{"sql": "SELECT id FROM CLAIM", "reason": "ok"}'] * 3)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=_down(tmp_path),
                  cache=TwoTierCache(L1Cache()))
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="CLAIM", card="TABLE CLAIM", score=1.0)])
    snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-28T00:00:00Z",
                        columns=[Column(id=f"CLAIM.{c}", object_id="CLAIM", name=c) for c in ("id", "amount")])
    answer = agent.answer(packet, snapshot, grants, IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    assert len(generator.calls) == 1, generator.calls
    assert answer.failed and not answer.deferred, answer
    assert "did not answer" in answer.answer, answer.answer
    assert "ungovernable" not in answer.answer.lower() and "already closed" not in answer.answer, answer.answer


def _oracle_down(fault: Exception = ConnectionError("DPY-4011: the database or network closed the connection")):
    """The Oracle adapter, every dictionary read raising `fault`: its own classifier reads it."""
    from mnemiq.adapters.oracle import OracleAdapter

    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._schema = "APP"

    def rows(sql, **binds):
        raise fault

    adapter._rows = rows
    adapter.execute = rows
    return adapter


@pytest.mark.parametrize("sql", [
    "SELECT id FROM CLAIM",  # calls nothing: the per-name inventory asks about no names (Codex's review)
    "SELECT upper(id) FROM CLAIM",
])
def test_on_oracle_a_statement_calling_nothing_meets_the_outage_at_the_computed_columns(sql):
    verdict = decide(sql, VISIBLE, adapter=_oracle_down(), dialect="oracle", target="oracle")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SOURCE_UNAVAILABLE, verdict
    assert not verdict.repairable and verdict.source_detail == "ConnectionError", verdict
    grants = GrantSet(frozenset(VISIBLE), writable=frozenset(VISIBLE))
    write = decide_write("UPDATE CLAIM SET amount = 0 WHERE id = 1", VISIBLE, grants, policy=AccessPolicy(),
                         adapter=_oracle_down(), dialect="oracle", target="oracle", writes_enabled=True)
    assert isinstance(write, Refusal) and write.code is RefusalCode.SOURCE_UNAVAILABLE, write


def test_an_unclassified_fault_at_the_computed_columns_keeps_the_retry():
    verdict = decide("SELECT id FROM CLAIM", VISIBLE, adapter=_oracle_down(RuntimeError("view missing")),
                     dialect="oracle", target="oracle")
    assert verdict.code is RefusalCode.UNRESOLVABLE_CALLS and verdict.repairable, verdict


@pytest.mark.parametrize("candidates", [1, 3], ids=["single path", "deep path"])
def test_through_the_agent_one_model_call_on_either_path(tmp_path, candidates):
    grants = GrantSet(frozenset(VISIBLE))
    generator = FakeGenerator(['{"sql": "SELECT id FROM CLAIM", "reason": "ok"}'] * 8)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=_down(tmp_path),
                  cache=TwoTierCache(L1Cache()), candidates=candidates)
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="CLAIM", card="TABLE CLAIM", score=1.0)])
    snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-28T00:00:00Z",
                        columns=[Column(id=f"CLAIM.{c}", object_id="CLAIM", name=c) for c in ("id", "amount")])
    answer = agent.answer(packet, snapshot, grants, IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    assert len(generator.calls) == 1, generator.calls
    assert answer.failed and "did not answer" in answer.answer, answer

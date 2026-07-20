import pyarrow as pa

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.verify.judge import FakeJudge
from mnemiq.verify.verifier import Verifier

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


class _FakeAdapter:
    def execute_arrow(self, sql, timeout_s=None):
        return pa.table({"n": [7]})

    def execute(self, sql):  # EXPLAIN inside decide()
        return []


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="acme", created_at="t",
                    columns=[Column(id="claim.n", object_id="claim", name="n")])


def _packet() -> ContextPacket:
    return ContextPacket(question="how many claims?",
                         cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                         grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")


def _agent(verifier=None) -> Agent:
    return Agent(generator=FakeGenerator(['{"sql": "SELECT n FROM claim"}']),
                 synthesizer=FakeSynthesizer("There are 7 claims."),
                 adapter=_FakeAdapter(), cache=TwoTierCache(L1Cache()),
                 budget=Budget(), verifier=verifier)


def _answer(agent: Agent):
    return agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)


def test_verifier_none_is_byte_for_byte():
    r = _answer(_agent(verifier=None))
    assert r.answer == "There are 7 claims." and r.deferred is False


def test_low_confidence_judge_defers():
    v = Verifier(threshold=0.9, sanity=False, grounding=False, judge=FakeJudge(0.1))
    r = _answer(_agent(verifier=v))
    assert r.deferred is True
    assert r.answer == "The result may not correctly answer the question."


def test_high_confidence_judge_answers():
    v = Verifier(threshold=0.5, sanity=False, grounding=False, judge=FakeJudge(0.9))
    r = _answer(_agent(verifier=v))
    assert r.deferred is False and r.answer == "There are 7 claims."

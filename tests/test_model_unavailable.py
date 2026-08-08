"""A model-provider outage is a stated failure, not a traceback and not a deferral.

The source-outage twin of M6: something that happened TO us must not land in the deferral
rate, and must not escape the engine as an exception for the transport to turn into a 500.
"""

import pyarrow as pa
import pytest

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot
from mnemiq.llm.client import ModelUnavailable
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard


class _DeadGenerator:
    def propose(self, *args, **kwargs):
        raise ModelUnavailable("Error code: 500 - Internal Server Error.")


class _Adapter:
    dialect = "duckdb"

    def execute_arrow(self, sql, timeout_s=30):
        return pa.table({"n": [1]})


def _agent(candidates: int = 1) -> Agent:
    return Agent(
        generator=_DeadGenerator(),
        synthesizer=FakeSynthesizer(),
        adapter=_Adapter(),
        cache=TwoTierCache(L1Cache()),
        budget=Budget(wall_clock_s=5.0, max_attempts=2),
        candidates=candidates,
    )


def _ask(agent: Agent):
    packet = ContextPacket(
        question="how many claims?",
        cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="f",
        enrichment_version="v1",
    )
    snapshot = Snapshot(version="v1", source_id="acme", created_at="2026-07-13T00:00:00Z",
                        columns=[Column(id="claim.n", object_id="claim", name="n")])
    identity = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])
    return agent.answer(packet, snapshot, GrantSet(frozenset({"claim"})), identity)


def test_a_provider_outage_is_a_failure_the_caller_can_read():
    answer = _ask(_agent())

    assert answer.failed is True
    assert answer.reason_code == DeferralReason.MODEL_UNAVAILABLE
    assert "model provider" in answer.answer


def test_it_is_never_counted_as_a_deferral():
    # The whole point of M6: an outage inflating the deferral rate reported "safe: gave up
    # on an answerable question" for something that was never a decision.
    answer = _ask(_agent())

    assert answer.deferred is False
    assert answer.preview is None and answer.trace is None


def test_the_outage_does_not_escape_as_an_exception():
    # Before this, the SDK error propagated out of Runtime.ask and /v1/ask returned a 500
    # with a traceback -- no reason code, nothing a caller could branch on.
    _ask(_agent())  # would raise ModelUnavailable
    _ask(_agent(candidates=3))  # the multi-candidate path too


def test_the_reason_code_is_not_a_deferral_reason_by_accident():
    # It shares the enum with the deferral reasons, as execution_failed does. The contract
    # is `failed`, not the enum it lives in.
    assert DeferralReason.MODEL_UNAVAILABLE == "model_unavailable"


def test_a_real_bug_still_raises():
    class _Broken(_DeadGenerator):
        def propose(self, *args, **kwargs):
            raise ValueError("a genuine defect, not an outage")

    agent = _agent()
    agent.generator = _Broken()
    with pytest.raises(ValueError):
        _ask(agent)


def test_the_providers_own_words_survive_into_the_answer():
    """A results file has to say WHICH failure happened.

    This message used to assert an outage and discard the cause, so a misconfigured
    model name, a context-length refusal and a dead endpoint all read identically.
    Diagnosing a 30-case failure took twenty minutes for want of a string that was
    already in hand.
    """
    answer = _ask(_agent())

    assert answer.failed
    assert "500" in answer.answer, "the provider's status has to reach the results file"
    assert "outage, not a judgement" in answer.answer, "the plain-language framing stays"

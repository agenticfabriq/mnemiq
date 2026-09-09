"""M2 + M6 -- a deferral says why, and a failure is not a deferral.

Before this, every terminal state collapsed to `AgentAnswer(answer=<prose>, deferred=True)`.
`RefusalCode` has 17 members and `VerifyVerdict` carries a layer; none of it reached a caller.
The sharpest consequence was arithmetic: a broken policy file denies everything silently, so every
request deferred, and `aggregate` hardcoded `errors=0` -- a total authorization outage reported
`deferral_rate=1.0, errors=0`, which is the shape of the product working perfectly.
"""

import json

import pyarrow as pa

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent, DeferralReason
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import EMPTY, FileAuthzProvider, GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.observability.metrics import AnswerRecord, aggregate
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_RESULT = pa.table({"n": [7]})


class _FakeAdapter:
    def __init__(self, errors: int = 0, message: str = 'column "typo" does not exist'):
        self.errors = errors
        self.message = message
        self.queries: list[str] = []

    def execute_arrow(self, sql, timeout_s=None):
        self.queries.append(sql)
        if self.errors > 0:
            self.errors -= 1
            raise Exception(self.message)
        return _RESULT

    def execute(self, sql):
        return []


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[Column(id="claim.n", object_id="claim", name="n")],
    )


def _packet(cards=True) -> ContextPacket:
    return ContextPacket(
        question="how many claims?",
        cards=([RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)] if cards else []),
        grant_fingerprint=_GRANTS.fingerprint,
        enrichment_version="v1",
    )


def _agent(replies, adapter=None, budget=None):
    return Agent(
        generator=FakeGenerator(replies),
        synthesizer=FakeSynthesizer("There are 7 claims."),
        adapter=adapter or _FakeAdapter(),
        cache=TwoTierCache(L1Cache()),
        budget=budget or Budget(),
    )


# --------------------------------------------------------------------------------------------
# M2 -- an unreadable policy is not the same fact as an empty policy
# --------------------------------------------------------------------------------------------


def test_an_unreadable_policy_file_is_distinguishable_from_having_no_grants(tmp_path):
    missing = FileAuthzProvider(str(tmp_path / "nope.json")).grants_for(_IDENTITY)
    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json")
    unparseable = FileAuthzProvider(str(malformed)).grants_for(_IDENTITY)
    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"not_roles": {}}))
    keyless = FileAuthzProvider(str(wrong_shape)).grants_for(_IDENTITY)

    for grants in (missing, unparseable, keyless):
        assert grants.available is False, "an unreadable policy must say so"
        assert grants.objects == frozenset(), "and must still grant nothing -- it fails closed"

    readable = tmp_path / "ok.json"
    readable.write_text(json.dumps({"roles": {"analyst": {"objects": []}}}))
    real = FileAuthzProvider(str(readable)).grants_for(_IDENTITY)
    assert real.available is True, "a policy that legitimately grants nothing is not an outage"
    assert real.objects == frozenset()


def test_availability_is_not_part_of_the_access_fingerprint():
    # The fingerprint identifies the *access*. An outage must not silently repartition the cache.
    assert EMPTY.fingerprint == GrantSet(frozenset(), available=False).fingerprint


def test_an_unreadable_policy_defers_with_its_own_reason(tmp_path):
    agent = _agent(['{"sql": "SELECT n FROM claim"}'])
    unavailable = FileAuthzProvider(str(tmp_path / "nope.json")).grants_for(_IDENTITY)

    result = agent.answer(_packet(cards=False), _snapshot(), unavailable, _IDENTITY)

    assert result.deferred is True
    assert result.reason_code == DeferralReason.POLICY_UNAVAILABLE, (
        "an operator must be able to tell a broken policy file from a user who lacks grants"
    )


def test_a_retrieval_miss_under_a_working_policy_is_a_different_reason():
    agent = _agent(['{"sql": "SELECT n FROM claim"}'])

    result = agent.answer(_packet(cards=False), _snapshot(), _GRANTS, _IDENTITY)

    assert result.deferred is True
    assert result.reason_code == DeferralReason.NO_TABLES


# --------------------------------------------------------------------------------------------
# M6 -- a failure is something that happened to us, not a decision we made
# --------------------------------------------------------------------------------------------


def test_a_database_that_rejects_every_attempt_is_a_failure_not_a_deferral():
    adapter = _FakeAdapter(errors=99)
    agent = _agent(['{"sql": "SELECT n FROM claim"}'] * 9, adapter=adapter, budget=Budget(max_attempts=2))

    result = agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)

    assert result.failed is True
    assert result.deferred is False, (
        "counting a source outage as a deferral is what let an outage look like abstention"
    )
    assert result.reason_code == DeferralReason.EXECUTION_FAILED


def test_the_sources_own_error_text_does_not_reach_the_caller():
    """The answer used to carry `Last error: {failure}` -- the database's verbatim complaint.

    A source rejection is made of the caller's schema. Postgres names the relation and the
    column it refused, and a connection failure names the host and the user. An identity
    denied a table would therefore learn the table exists by being told why it could not
    read it, which is the first disclosure class SECURITY.md claims is in scope. It reaches
    `AgentAnswer.answer`, so `/v1/ask` carried it too, not only the stream's error frame.
    """
    leaky = (
        'permission denied for table hr_prod.payroll_salary; '
        'connection postgresql://svc_mnemiq@10.2.0.7:5432/hr_prod'
    )
    adapter = _FakeAdapter(errors=99, message=leaky)
    agent = _agent(['{"sql": "SELECT n FROM claim"}'] * 9, adapter=adapter,
                   budget=Budget(max_attempts=2))

    result = agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)

    for secret in ("payroll_salary", "hr_prod", "svc_mnemiq", "10.2.0.7", "postgresql://",
                   "permission denied"):
        assert secret not in result.answer, f"the answer disclosed {secret!r}"
    # The state still has to be legible, or hiding the cause would have cost the caller M6.
    assert result.failed is True
    assert result.reason_code == DeferralReason.EXECUTION_FAILED


def test_the_source_error_is_still_fed_back_to_the_planner():
    """Withholding it from the caller must not withhold it from the repair loop.

    The database's complaint is the retry's whole input -- it is what a second attempt is
    corrected *by*. The model already holds the schema this text is made of, so the boundary
    that matters is the wire, not the prompt.
    """
    adapter = _FakeAdapter(errors=1, message='column "typo" does not exist')
    agent = _agent(['{"sql": "SELECT typo FROM claim"}', '{"sql": "SELECT n FROM claim"}'],
                   adapter=adapter, budget=Budget(max_attempts=2))

    result = agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)

    assert result.failed is False
    assert any("typo" in (feedback or "") for feedback in agent.generator.calls[1:]), (
        "the second attempt was planned without being told what the first one got wrong"
    )


def test_an_outage_cannot_raise_the_deferral_rate():
    outage = [
        AnswerRecord(deferred=False, failed=True, cached=False, total_ms=1.0, mode="fast",
                     reason_code=DeferralReason.EXECUTION_FAILED)
        for _ in range(4)
    ]

    metrics = aggregate(outage)

    assert metrics.errors == 4, "errors was hardcoded to 0 before this"
    assert metrics.deferrals == 0
    assert metrics.deferral_rate == 0.0, (
        "the number the product claim rests on must not be raised by the database being down"
    )


def test_deferral_rate_is_computed_over_answers_that_were_actually_decided():
    records = [
        AnswerRecord(deferred=True, failed=False, cached=False, total_ms=1.0, mode="fast",
                     reason_code=DeferralReason.AUTHORIZATION),
        AnswerRecord(deferred=False, failed=False, cached=False, total_ms=1.0, mode="fast",
                     reason_code=None),
        AnswerRecord(deferred=False, failed=True, cached=False, total_ms=1.0, mode="fast",
                     reason_code=DeferralReason.EXECUTION_FAILED),
    ]

    metrics = aggregate(records)

    assert metrics.answers == 3
    assert metrics.errors == 1
    assert metrics.deferrals == 1
    assert metrics.deferral_rate == 0.5, "1 deferral out of the 2 requests we actually decided"

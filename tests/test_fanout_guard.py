"""The fan-out guard WIRED: decider order, the repair loop, and the setting that turns it on.

`test_fanout_check.py` owns the rule. This file owns the joins between it and everything that has
to carry its answer -- the ones a per-module suite leaves unasserted.
"""
from __future__ import annotations

import json

import pyarrow as pa

from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot
from mnemiq.generate.correct import FakeCorrector
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.plan_query import plan_query
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

FANOUT_SQL = (
    "SELECT SUM(c.paid_amount_cents) / SUM(p.earned_premium_cents) AS loss_ratio "
    "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"
)
PER_FACT_SQL = (
    "WITH paid AS (SELECT SUM(paid_amount_cents) AS v FROM fact_claim), "
    "earned AS (SELECT SUM(earned_premium_cents) AS v FROM fact_premium) "
    "SELECT paid.v / earned.v AS loss_ratio FROM paid CROSS JOIN earned"
)

VISIBLE = {
    "fact_claim": {"policy_id", "paid_amount_cents"},
    "fact_premium": {"policy_id", "earned_premium_cents"},
    "dim_policy": {"policy_id", "region"},
}
KEYS = {
    ("fact_claim", "policy_id"): False,
    ("fact_premium", "policy_id"): False,
    ("dim_policy", "policy_id"): True,
}


def _snapshot() -> Snapshot:
    profile = {
        "fact_claim": {"policy_id": (30000, 23322, 0), "paid_amount_cents": (30000, 29000, 0)},
        "fact_premium": {"policy_id": (125000, 50000, 0),
                         "earned_premium_cents": (125000, 90000, 0)},
    }
    return Snapshot(version="v1", source_id="acme", created_at="t", columns=[
        Column(id=f"{t}.{c}", object_id=t, name=c, row_count=r, distinct_count=d, null_count=n)
        for t, cols in profile.items() for c, (r, d, n) in cols.items()
    ])


def _packet() -> ContextPacket:
    return ContextPacket(
        question="what is the loss ratio?",
        cards=[RetrievedCard(object_id="fact_claim", card="TABLE fact_claim", score=1.0),
               RetrievedCard(object_id="fact_premium", card="TABLE fact_premium", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1",
    )


_GRANTS = GrantSet(frozenset({"fact_claim", "fact_premium"}))


def _reply(sql: str) -> str:
    return json.dumps({"sql": sql})


# -- the decider ------------------------------------------------------------------------------------


def test_decide_refuses_a_fan_out_only_when_key_facts_are_supplied():
    on = decide(FANOUT_SQL, VISIBLE, keys=KEYS)
    assert isinstance(on, Refusal) and on.code == RefusalCode.FAN_OUT and on.repairable
    assert isinstance(decide(FANOUT_SQL, VISIBLE), Approved), "keys=None is today's behaviour"


def test_decide_approves_the_rewrite_the_refusal_asks_for():
    assert isinstance(decide(PER_FACT_SQL, VISIBLE, keys=KEYS), Approved)


class _Index:
    """A value index holding `dim_policy.region`. Every deployment passes one, so the check has to
    be exercised with an index present, not only with `values=None`."""

    def has(self, table, column):
        return (table, column) == ("dim_policy", "region")

    def contains(self, table, column, literal):
        return literal in {"east", "west"}

    def nearest(self, table, column, literal):
        return ["east", "west"]


_REGION_SQL = (
    "SELECT SUM(c.paid_amount_cents) FROM fact_claim c "
    "JOIN fact_premium p ON c.policy_id = p.policy_id "
    "JOIN dim_policy d ON d.policy_id = c.policy_id WHERE d.region = {!r}"
)


def test_a_wrong_literal_is_reported_before_the_fan_out():
    """Both refusals are repairable; the cheap literal fix goes first, so the corrector is not
    asked to restructure a query and pick a value in the same pass."""
    verdict = decide(_REGION_SQL.format("EAST"), VISIBLE, values=_Index(), keys=KEYS)
    assert isinstance(verdict, Refusal) and verdict.code == RefusalCode.VALUE_GROUNDING


def test_the_fan_out_is_still_checked_when_every_literal_is_real():
    verdict = decide(_REGION_SQL.format("east"), VISIBLE, values=_Index(), keys=KEYS)
    assert isinstance(verdict, Refusal) and verdict.code == RefusalCode.FAN_OUT


# -- the repair loop ---------------------------------------------------------------------------------


def test_the_corrector_is_handed_the_fan_out_and_its_rewrite_is_approved():
    corrector = FakeCorrector([PER_FACT_SQL])
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, FakeGenerator([_reply(FANOUT_SQL)]),
                         target="duckdb", corrector=corrector, guard_fanout=True)
    assert isinstance(outcome, Approved) and outcome.corrected
    (sent_sql, problem), = corrector.calls
    assert "fact_premium" in problem and "CTE" in problem


def test_a_failed_correction_goes_back_to_the_model_as_feedback():
    corrector = FakeCorrector([FANOUT_SQL])  # "fixes" nothing
    generator = FakeGenerator([_reply(FANOUT_SQL), _reply(PER_FACT_SQL)])
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator, target="duckdb",
                         corrector=corrector, guard_fanout=True)
    assert isinstance(outcome, Approved)
    assert generator.calls[0] is None
    assert "fact_premium" in generator.calls[1], "the second proposal must see why"


def test_the_guard_is_off_unless_asked_for():
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, FakeGenerator([_reply(FANOUT_SQL)]),
                         target="duckdb")
    assert isinstance(outcome, Approved), "the default approves it, as it did before M109"


# -- instant mode -----------------------------------------------------------------------------------
# Instant mode caps only the OUTER loop, the one that repairs what the database rejects; the
# decider's own repair loop keeps its attempts. So a fan-out is re-planned like any repairable
# refusal. What instant mode never does is return the inflated number.


class _Adapter:
    """Records every query it is asked to run, so a test can say what executed."""

    dialect = "duckdb"

    def __init__(self):
        self.queries: list[str] = []

    def execute_arrow(self, sql, timeout_s=None):
        self.queries.append(sql)
        return pa.table({"loss_ratio": [0.25]})

    def execute(self, sql):
        return []


_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def _instant(replies: list[str], adapter: _Adapter):
    generator = FakeGenerator([_reply(sql) for sql in replies])
    agent = build_agent(
        MODES["instant"], generator=generator, synthesizer=FakeSynthesizer("ok"),
        adapter=adapter, cache=TwoTierCache(L1Cache()), corrector=None, values=None,
        selector=None, guard_fanout=True,
    )
    return agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY), generator


def test_instant_mode_re_plans_a_fan_out_with_the_refusal_as_feedback():
    adapter = _Adapter()
    answer, generator = _instant([FANOUT_SQL, PER_FACT_SQL], adapter)
    assert not answer.deferred
    assert len(generator.calls) == 2
    assert "fact_premium" in generator.calls[1], "the second proposal must see why"
    (ran,) = adapter.queries
    assert "CROSS JOIN" in ran, "only the rewrite reaches the database"


def test_instant_mode_withholds_a_fan_out_it_cannot_repair():
    """Three proposals, all inflated: the answer is a deferral, and nothing ran. That converts a
    wrong answer into a deferral -- and a false positive into one too, which is the cost the
    measurement counts."""
    adapter = _Adapter()
    answer, generator = _instant([FANOUT_SQL] * 3, adapter)
    assert answer.deferred and answer.reason_code == DeferralReason.INVALID_QUERY
    assert len(generator.calls) == 3
    assert adapter.queries == []


# -- the setting --------------------------------------------------------------------------------------


def test_one_setting_reaches_the_agent_in_both_states():
    """Settings -> build_agent -> Agent. The Agent -> plan_query hop, and every other construction
    site, is held by the call-site scan in `test_undefined_term_guard.py`, which now owes
    `guard_fanout` at the same sites as `guard_undefined_terms`."""
    assert Settings.model_fields["guard_fanout"].default is False, (
        "off until the pre-registered measurement flips it"
    )
    for wanted in (True, False):
        settings = Settings(guard_fanout=wanted,
                            llm_base_url="http://localhost:1/v1", llm_api_key="unused")
        agent = build_agent(
            MODES["thinking"], generator=None, synthesizer=None, adapter=None, cache=None,
            corrector=None, values=None, selector=None, guard_fanout=settings.guard_fanout,
        )
        assert agent.guard_fanout is wanted

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.verdict import Approved


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(id="claim.claim_identifier", object_id="claim", name="claim_identifier"),
            Column(id="claim.status", object_id="claim", name="status"),
            Column(id="person.last_name", object_id="person", name="last_name"),
        ],
    )


def _packet() -> ContextPacket:
    return ContextPacket(
        question="how many claims?",
        cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="fp",
        enrichment_version="v1",
    )


_GRANTS = GrantSet(frozenset({"claim"}))


def _plan(replies, **kw):
    generator = FakeGenerator(replies)
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator, target="duckdb", **kw)
    return outcome, generator


def test_a_good_query_is_approved_on_the_first_try():
    outcome, generator = _plan(['{"sql": "SELECT claim_identifier FROM claim"}'])
    assert isinstance(outcome, Approved)
    assert outcome.tables == ["claim"]
    assert generator.calls == [None]  # no feedback needed


def test_a_repairable_refusal_is_fed_back_and_repaired():
    outcome, generator = _plan(
        [
            '{"sql": "SELECT * FROM claim"}',  # rejected: star over a base table
            '{"sql": "SELECT claim_identifier FROM claim"}',  # repaired
        ]
    )
    assert isinstance(outcome, Approved)
    assert generator.calls[0] is None
    assert "SELECT *" in generator.calls[1]  # the refusal message came back as feedback


def test_an_unauthorized_table_is_never_retried():
    outcome, generator = _plan(
        [
            '{"sql": "SELECT last_name FROM person"}',
            '{"sql": "SELECT claim_identifier FROM claim"}',  # must never be reached
        ]
    )
    assert isinstance(outcome, Deferred)
    assert "person" in outcome.reason
    assert len(generator.calls) == 1  # a guard is not a puzzle to solve with another attempt


def test_the_loop_is_bounded():
    outcome, generator = _plan(['{"sql": "SELECT * FROM claim"}'] * 5, max_attempts=3)
    assert isinstance(outcome, Deferred)
    assert len(generator.calls) == 3


def test_a_model_deferral_is_passed_through():
    outcome, _ = _plan(['{"sql": null, "reason": "no table records payouts"}'])
    assert isinstance(outcome, Deferred)
    assert "payouts" in outcome.reason


def test_an_empty_packet_defers_without_calling_the_model():
    empty = ContextPacket(
        question="anything?", cards=[], grant_fingerprint="fp", enrichment_version="v1"
    )
    generator = FakeGenerator(['{"sql": "SELECT 1"}'])
    outcome = plan_query(empty, _snapshot(), GrantSet(frozenset()), generator, target="duckdb")

    assert isinstance(outcome, Deferred)
    assert generator.calls == []  # no tables, no grants, no reason to spend a token

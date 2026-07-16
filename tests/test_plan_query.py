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


def test_logic_lint_is_corrected_once_and_approved():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.correct import FakeCorrector
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
    from mnemiq.sql.verdict import Approved

    snapshot = Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z",
        columns=[
            Column(id="claim.name", object_id="claim", name="name"),
            Column(id="claim.score", object_id="claim", name="score"),
        ],
    )
    packet = ContextPacket(
        question="q", cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1",
    )
    grants = GrantSet(frozenset({"claim"}))
    gen = FakeGenerator(['{"sql": "SELECT name FROM claim ORDER BY score LIMIT 1"}'])
    corrector = FakeCorrector(
        ["SELECT name FROM claim WHERE score IS NOT NULL ORDER BY score LIMIT 1"]
    )

    verdict = plan_query(
        packet, snapshot, grants, gen, adapter=None, target="duckdb", corrector=corrector
    )
    assert isinstance(verdict, Approved)
    assert corrector.calls and "IS NOT NULL" not in corrector.calls[0][0]  # got the flawed SQL
    # the corrected (null-guarded) form was approved; sqlglot renders it as "NOT score IS NULL"
    assert "IS NULL" in verdict.plan_sql and "NOT" in verdict.plan_sql


def test_value_grounding_is_corrected_once_and_approved():
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.correct import FakeCorrector
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
    from mnemiq.sql.verdict import Approved

    class _FakeIndex:
        def __init__(self, data):
            self._data = data

        def has(self, t, c):
            return (t, c) in self._data

        def contains(self, t, c, v):
            return v in self._data.get((t, c), set())

        def nearest(self, t, c, v, k=8):
            return sorted(self._data.get((t, c), set()))[:k]

    snapshot = Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z",
        columns=[Column(id="gasstations.Country", object_id="gasstations", name="Country")],
    )
    packet = ContextPacket(
        question="q",
        cards=[RetrievedCard(object_id="gasstations", card="TABLE gasstations", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1",
    )
    grants = GrantSet(frozenset({"gasstations"}))
    idx = _FakeIndex({("gasstations", "Country"): {"Czech Republic", "Slovakia"}})
    gen = FakeGenerator(['{"sql": "SELECT Country FROM gasstations WHERE Country = \'CZE\'"}'])
    corrector = FakeCorrector(["SELECT Country FROM gasstations WHERE Country = 'Czech Republic'"])

    verdict = plan_query(
        packet, snapshot, grants, gen, adapter=None, target="duckdb",
        corrector=corrector, values=idx,
    )
    assert isinstance(verdict, Approved)
    assert "CZE" in corrector.calls[0][0]  # the corrector saw the flawed SQL
    assert "Czech Republic" in verdict.plan_sql  # the corrected literal was approved


def test_without_a_corrector_a_lint_falls_back_to_regenerate_then_defers():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import Deferred, plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    snapshot = Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z",
        columns=[Column(id="claim.score", object_id="claim", name="score"),
                 Column(id="claim.name", object_id="claim", name="name")],
    )
    packet = ContextPacket(question="q", cards=[RetrievedCard(object_id="claim", card="c", score=1.0)],
                           grant_fingerprint="fp", enrichment_version="v1")
    grants = GrantSet(frozenset({"claim"}))
    gen = FakeGenerator([
        '{"sql": "SELECT name FROM claim ORDER BY score LIMIT 1"}',
        '{"sql": "SELECT name FROM claim ORDER BY score LIMIT 1"}',
    ])
    out = plan_query(packet, snapshot, grants, gen, adapter=None, target="duckdb", max_attempts=2)
    assert isinstance(out, Deferred)  # lint fed back, never corrected, ran out of attempts

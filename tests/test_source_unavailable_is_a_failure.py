"""A source that does not answer its proof is a failure, not a deferral, and deep mode stops on it.

#159 gave the proof a third outcome, SOURCE_UNAVAILABLE, and `plan_query` deferred it under the
default reason, `ungovernable`. Two costs followed (Codex's third review of #159). The outage was
counted as a deferral, which is M6's conflation: MODEL_UNAVAILABLE and VERIFIER_UNAVAILABLE ride
`failed` for the same reason. And deep mode dropped the candidate and asked the next, so a source
that was down spent every candidate's model call and then the single path's, to report governance.

Measured through the agent with a real DuckDB source whose connection closes at a chosen proof.
"""

from __future__ import annotations

import duckdb

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.server.serialize import answer_payload

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t", principal_id="u", roles=["r"])
_REPLY = '{"sql": "SELECT id FROM claim", "reason": "ok"}'


class _ClosesAtProof:
    """The real DuckDB source, whose connection closes as proof number `at` is asked, so the
    fault DuckDB raises for that proof and every later one is its own."""

    def __init__(self, inner, at: int):
        self._inner, self._at, self.proofs = inner, at, 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute(self, sql):
        if sql.startswith("EXPLAIN "):
            self.proofs += 1
            if self.proofs == self._at:
                self._inner._con.close()
        return self._inner.execute(sql)


def _source(tmp_path, at: int) -> _ClosesAtProof:
    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT); INSERT INTO claim VALUES (1), (2)")
    con.close()
    return _ClosesAtProof(DuckDBAdapter.duckdb(path, read_only=False), at)


def _ask(adapter, candidates: int = 1):
    generator = FakeGenerator([_REPLY] * 12)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=adapter,
                  cache=TwoTierCache(L1Cache()), candidates=candidates)
    packet = ContextPacket(question="which claims?", grant_fingerprint=_GRANTS.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    snapshot = Snapshot(version="v1", source_id="s", created_at="2026-10-18T00:00:00Z",
                        columns=[Column(id="claim.id", object_id="claim", name="id")])
    return agent.answer(packet, snapshot, _GRANTS, _IDENTITY), generator


def _is_the_outage(answer):
    assert answer.failed is True, answer
    assert answer.deferred is False, "an outage counted as a deferral raises the deferral rate (M6)"
    assert answer.reason_code is DeferralReason.SOURCE_UNAVAILABLE, answer.reason_code
    assert "did not answer" in answer.answer, answer.answer
    assert "closed" not in answer.answer, "the driver's words stay in the server log"


def test_one_model_call_and_a_failure_that_says_the_source_did_not_answer(tmp_path):
    answer, generator = _ask(_source(tmp_path, at=1))
    assert len(generator.calls) == 1, generator.calls
    _is_the_outage(answer)


def test_the_caller_reads_failed_and_not_deferred(tmp_path):
    """The join the harness grades through: `/v1/ask` and the stream both carry this payload."""
    answer, _ = _ask(_source(tmp_path, at=1))
    out = answer_payload(answer)
    assert (out["failed"], out["deferred"], out["reason_code"]) == (True, False, "source_unavailable")


def test_deep_mode_stops_at_the_first_unanswered_proof(tmp_path):
    """Five candidates against a source that is down: one model call, no fallback.

    Before, each candidate's deferral was dropped and the next one asked, then no candidate had
    run, so the single path was tried as well: six model calls to report `ungovernable`."""
    answer, generator = _ask(_source(tmp_path, at=1), candidates=5)
    assert len(generator.calls) == 1, generator.calls
    _is_the_outage(answer)
    assert answer.candidates_executed == 0


def test_deep_mode_fails_the_turn_when_the_source_goes_down_after_a_candidate_ran(tmp_path):
    """The first candidate is proved and run, and the source closes at the second's proof.

    The turn fails rather than voting on the one table it has. A reduced set is never an answer in
    deep mode: under `min_agreement` it is DISAGREEMENT by rule, a deferral, so voting on what ran
    would put the outage back in the deferral rate. The candidate count says how far it got."""
    answer, generator = _ask(_source(tmp_path, at=2), candidates=5)
    assert len(generator.calls) == 2, generator.calls
    _is_the_outage(answer)
    assert answer.candidates_executed == 1

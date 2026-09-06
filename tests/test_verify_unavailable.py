"""A verifier that could not reach its judge must not report the answer as verified.

The defect this closes is M89's product half. `SemanticJudge.score` fails open: an unreachable
judge returns 1.0, and 1.0 is exactly what a judge returns when it APPROVES. So a verifier that was
switched off produced answers indistinguishable from confidently-checked ones -- which is what
happened for every reasoning model until the token budget was fixed, and what will happen again
whenever the endpoint is down. The answer still goes out; only the claim about it changes.
"""
import pyarrow as pa
import pytest

from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verifier import Verifier


class _Judge:
    """A SemanticJudge-shaped stub whose fail-open path is under the test's control."""

    def __init__(self, score: float, falls_open: bool = False) -> None:
        self._score, self._falls_open = score, falls_open
        self.errors = self.unparsed = 0

    @property
    def fallbacks(self) -> int:
        return self.errors + self.unparsed

    def score(self, question, schema, sql, preview) -> float:
        if self._falls_open:
            self.errors += 1
            return 1.0          # the fail-open constant, identical to an approval
        return self._score


class _NoCounters:
    """A judge with no `fallbacks` -- FakeJudge, or anyone's stub."""

    def score(self, *_a, **_k) -> float:
        return 0.9


def _verify(judge, threshold=0.5):
    packet = ContextPacket(question="q", cards=[], grant_fingerprint="", enrichment_version=None)
    approved = Approved(plan_sql="SELECT 1", target_sql="SELECT 1")
    table = pa.table({"n": [1]})
    return Verifier(threshold=threshold, sanity=False, judge=judge).verify(packet, approved, table)


def test_a_judge_that_fell_open_is_reported_as_unavailable_not_as_a_pass():
    v = _verify(_Judge(1.0, falls_open=True))
    assert v.layer == "judge_unavailable"
    assert v.defer is False, "a dead judge must not stop an answer -- that decision is unchanged"
    assert v.reason, "the reason must be sayable to whoever reads the answer"


def test_a_real_approval_scoring_the_same_1_point_0_is_still_a_pass():
    """The whole difficulty: an approval and a fail-open carry the SAME confidence. Only the
    counter separates them, so this pair is the test."""
    v = _verify(_Judge(1.0, falls_open=False))
    assert v.layer == "judge"
    assert v.defer is False
    assert v.confidence == 1.0          # identical value, different verdict


def test_a_real_low_score_still_defers():
    v = _verify(_Judge(0.1))
    assert v.layer == "judge" and v.defer is True


def test_a_judge_without_counters_is_taken_at_its_word():
    """Absence of a counter is not evidence of a fallback; a stub judge must not be reported broken."""
    v = _verify(_NoCounters())
    assert v.layer == "judge" and v.defer is False


@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9, 1.0])
def test_unavailable_never_depends_on_the_threshold(threshold):
    """The fail-open constant is 1.0, so at threshold 1.0 a naive `c < threshold` would defer it and
    look like a working verifier catching something. It is still not a judgement."""
    v = _verify(_Judge(1.0, falls_open=True), threshold=threshold)
    assert v.layer == "judge_unavailable"


# --- the wire half: engine-side visibility is not visibility ---

def _payload(layer):
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.server.serialize import answer_payload
    a = AgentAnswer(answer="x", preview=None)
    a.verify_layer = layer
    a.verify_confidence = 1.0
    return answer_payload(a)


def test_the_unavailable_state_reaches_the_wire():
    """`verify_layer` is deliberately withheld from clients, so stamping the answer engine-side
    changes nothing a reader sees. Without this, an answer from a switched-off verifier is
    byte-identical on the wire to a verified one -- which is the whole defect."""
    assert _payload("judge_unavailable")["verified"] == "unavailable"


def test_a_checked_answer_says_checked_and_leaks_no_score():
    p = _payload("judge")
    assert p["verified"] == "checked"
    assert "verify_confidence" not in p and "verify_layer" not in p, \
        "the score and the layer name stay withheld; only the derived state ships"


def test_no_verifier_configured_is_its_own_state():
    """`null` is not 'checked' and not 'unavailable'. A mode with no verifier is a third thing, and
    collapsing it into either would misreport one of them."""
    assert _payload(None)["verified"] is None


def test_every_deterministic_layer_counts_as_checked():
    for layer in ("sanity", "grounding", "pass"):
        assert _payload(layer)["verified"] == "checked", layer

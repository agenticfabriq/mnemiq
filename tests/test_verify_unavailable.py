"""A verifier that could not reach its judge must not report the answer as verified.

The defect this closes is M89's product half. `SemanticJudge.score` fails open: an unreachable
judge returns 1.0, and 1.0 is exactly what a judge returns when it APPROVES. So a verifier that was
switched off produced answers indistinguishable from confidently-checked ones -- which is what
happened for every reasoning model until the token budget was fixed, and what will happen again
whenever the endpoint is down.

The first fix changed only the claim and let the answer go out, which left `deep` -- the one mode
that wires a judge -- handing back results stamped *not checked*. The answer is withheld by
default now (issue #2), and recorded as a FAILURE rather than a deferral, because an outage in
the deferral rate is M6.
"""
import pyarrow as pa
import pytest

from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.judge import JudgeRead
from mnemiq.verify.verifier import Verifier


class _Judge:
    """A SemanticJudge-shaped stub: it answers `read`, so the fact travels with the score."""

    def __init__(self, score: float, falls_open: bool = False, why: str = "error") -> None:
        self._score, self._falls_open, self._why = score, falls_open, why
        self.errors = self.unparsed = 0

    @property
    def fallbacks(self) -> int:
        return self.errors + self.unparsed

    def read(self, question, schema, sql, preview) -> JudgeRead:
        if self._falls_open:
            self.errors += 1
            # the constant, identical in value to approval
            return JudgeRead(1.0, fell_open=True, reason=self._why)
        return JudgeRead(self._score, fell_open=False)

    def score(self, *a, **k) -> float:
        return self.read(*a, **k).score


class _NoCounters:
    """A judge that speaks only the float protocol -- FakeJudge, or anyone's stub."""

    def score(self, *_a, **_k) -> float:
        return 0.9


def _verify(judge, threshold=0.5, fail_closed=True):
    packet = ContextPacket(question="q", cards=[], grant_fingerprint="", enrichment_version=None)
    approved = Approved(plan_sql="SELECT 1", target_sql="SELECT 1")
    table = pa.table({"n": [1]})
    return Verifier(threshold=threshold, sanity=False, judge=judge,
                    fail_closed=fail_closed).verify(packet, approved, table)


def test_a_judge_that_fell_open_is_reported_as_unavailable_not_as_a_pass():
    v = _verify(_Judge(1.0, falls_open=True))
    assert v.layer == "judge_unavailable"
    assert v.reason, "the reason must be sayable to whoever reads the answer"


def test_no_judgement_carries_no_score():
    """The second half of the same defect. `layer` said `judge_unavailable` and the number beside
    it said 1.0, so a reader who took the score without the discriminator saw a confident pass --
    which is M89's shape one field over. No float is honest: 1.0 reads as a confident pass and 0.0
    as a confident failure, and what happened was neither."""
    assert _verify(_Judge(1.0, falls_open=True)).confidence is None
    assert _verify(_Judge(1.0, falls_open=False)).confidence == 1.0


@pytest.mark.parametrize("fail_closed, defers", [(True, True), (False, False)])
def test_whether_a_dead_judge_stops_an_answer_is_the_deployments_call(fail_closed, defers):
    """It used to be unconditionally False, and the comment saying so read as settled. It was not:
    the judge is wired only where a mode asks for it -- `deep` alone out of the box -- so the one
    mode a caller picks FOR assurance was handing back answers stamped "not checked" and handing
    them back anyway. Withholding is the default now (issue #2); an operator trading assurance
    for availability during a provider outage sets `MNEMIQ_VERIFY_FAIL_CLOSED=0`.

    Both positions, because pinning one leaves the switch free to be ignored. The reasons differ
    too: a deferral's text is read by whoever asked the question, and "this answer was not
    checked" describes an answer they are not getting.
    """
    v = _verify(_Judge(1.0, falls_open=True), fail_closed=fail_closed)
    assert v.layer == "judge_unavailable" and v.defer is defers
    assert ("not giving you a result" in v.reason) is defers


def test_the_default_is_the_one_the_deployment_did_not_have_to_choose():
    """A `Verifier` built without the argument must fail closed. Every other conservative default
    in this codebase is written this way -- `writes_enabled`, `covers_view_bodies`,
    `unrecognised_source` -- because forgetting has to yield the safe answer, and this one is
    reached by any caller constructing a verifier directly."""
    from mnemiq.config import Settings

    assert Verifier(judge=None).fail_closed is True
    assert Settings().verify_fail_closed is True


def test_a_real_approval_scoring_the_same_1_point_0_is_still_a_pass():
    """The whole difficulty: an approval and a fail-open carry the SAME confidence. Only the
    counter separates them, so this pair is the test."""
    v = _verify(_Judge(1.0, falls_open=False))
    assert v.layer == "judge"
    assert v.defer is False
    assert v.confidence == 1.0          # the value a fell-open read no longer carries


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


def test_a_judged_answer_says_judged_and_leaks_no_score():
    p = _payload("judge")
    assert p["verified"] == "judged"
    assert "verify_confidence" not in p and "verify_layer" not in p, \
        "the score and the layer name stay withheld; only the derived state ships"


def test_no_verifier_read_is_its_own_state():
    """`null` is neither. It also has more causes than "this mode does not verify" -- a deferral
    raised before verification and an execution failure both leave the field unset -- so a client
    must not read it as a statement about the mode."""
    assert _payload(None)["verified"] is None


def test_the_deterministic_net_does_NOT_claim_the_answer_was_judged():
    """The distinction that makes this graded rather than boolean. `instant` and `thinking` -- and
    `thinking` is the DEFAULT mode -- run only the empty/null net, which never reads whether the
    answer is right. Reporting that as the same state as a judge-approved answer would make the
    accuracy claim the withheld confidence exists to refuse, on the modes with the most traffic."""
    for layer in ("sanity", "grounding", "pass"):
        assert _payload(layer)["verified"] == "basic", layer
    assert _payload("judge")["verified"] == "judged"


def test_an_unrecognised_layer_fails_CLOSED():
    """The layer vocabulary is a comment, not a type. A future layer meaning "no judgement
    happened" -- the shape `judge_unavailable` itself had before it existed -- must not ship as a
    check. Defaulting to "checked" is how M89 stayed invisible."""
    assert _payload("some_future_layer")["verified"] == "unknown", \
        "and NOT 'unavailable' -- that state means a judge could not be reached, which an operator " \
        "may page on; an untaught layer name is not an outage"


def test_the_mcp_surface_reports_the_same_state_as_the_http_one():
    """MCP is how a governing agent reads an answer, and it hand-builds its own dict. An agent
    deciding whether to act needs the same signal; two surfaces deriving it separately is how they
    come to disagree."""
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.mcp.server import _db_read
    from mnemiq.mcp.server import verified_state as mcp_state
    from mnemiq.server.serialize import answer_payload
    from mnemiq.server.serialize import verified_state as http_state

    assert mcp_state is http_state, "MCP must import the derivation, not restate it"

    class _RT:
        def __init__(self, ans): self._ans = ans
        def ask(self, *_a, **_k): return self._ans

    # Sharing the function is not enough: `_db_read` hand-builds its dict, so the KEY is what has
    # to be exercised. An earlier version of this test asserted only the identity, and a one-line
    # change inside `_db_read` could have made a dead judge read as judged with it still green.
    for layer in (None, "judge", "judge_unavailable", "sanity", "pass", "unknown_future"):
        ans = AgentAnswer(answer="x", preview=None)
        ans.verify_layer = layer
        assert _db_read(_RT(ans), None, "q")["verified"] == answer_payload(ans)["verified"], layer


def test_a_concurrent_request_failing_does_not_mislabel_one_that_was_judged():
    """The judge is SHARED -- `runtime.py` builds one for every mode -- and FastAPI runs the sync
    endpoint in a threadpool, so two answers are verified against it at once. The first version of
    this feature read a cumulative `fallbacks` counter before and after its own call, which counts
    any other thread's failure as its own.

    MEASURED before the fix: the request below was judged 0.95 and came back `judge_unavailable`,
    with that real confidence sitting beside the claim that nothing had judged it.
    """
    import threading

    started = threading.Event()
    may_finish = threading.Event()

    class _Shared:
        def __init__(self):
            self.errors = self.unparsed = 0

        @property
        def fallbacks(self):
            return self.errors + self.unparsed

        def read(self, question, schema, sql, preview):
            if question == "judged-but-slow":
                started.set()
                may_finish.wait(2)
                return JudgeRead(0.95, fell_open=False)
            self.errors += 1                       # the OTHER request's judge fails
            return JudgeRead(1.0, fell_open=True)

        def score(self, *a, **k):
            return self.read(*a, **k).score

    judge = _Shared()
    v = Verifier(threshold=0.5, sanity=False, judge=judge)
    approved = Approved(plan_sql="SELECT 1", target_sql="SELECT 1")
    table = pa.table({"n": [1]})
    out = {}

    def run(q):
        packet = ContextPacket(question=q, cards=[], grant_fingerprint="", enrichment_version=None)
        out[q] = v.verify(packet, approved, table)

    slow = threading.Thread(target=run, args=("judged-but-slow",))
    other = threading.Thread(target=run, args=("other",))
    slow.start()
    assert started.wait(2)
    other.start()
    other.join()
    may_finish.set()
    slow.join()

    assert out["judged-but-slow"].layer == "judge", out["judged-but-slow"]
    assert out["judged-but-slow"].confidence == 0.95
    assert out["other"].layer == "judge_unavailable"


def test_a_judge_outage_is_a_failure_and_not_a_deferral():
    """M6, for the third time, and the enum marks the first two: `EXECUTION_FAILED` and
    `MODEL_UNAVAILABLE` both say *NOT a deferral* where they are defined. A fail-closed verifier
    is the third way something can happen TO us and be recorded as the engine abstaining.

    The cost of getting it wrong is not abstract. `harness` grades a deferral on an answerable
    case DEFERRED_WRONGLY, so a flaky judge would raise the measured deferral rate and lower EX
    with nothing in the totals to say why -- readable only in `verify_layer`, case by case. And
    on the wire `verification` means *the verifier declined to stand behind it*, a judgement
    about the data, which an outage is not.
    """
    from mnemiq.contract.seams import DeferralReason

    v = _verify(_Judge(1.0, falls_open=True))
    assert v.defer is True and v.failed is True
    assert v.code is DeferralReason.VERIFIER_UNAVAILABLE

    scored_low = _verify(_Judge(0.1))
    assert scored_low.defer is True and scored_low.failed is False
    assert scored_low.code is DeferralReason.VERIFICATION


def test_the_engine_records_the_outage_as_a_failure():
    """The verdict carrying `failed` proves nothing about the engine reading it. This drives the
    branch that builds the answer, because the mapping is where the two could come apart."""
    from mnemiq.agent.loop import Agent
    from mnemiq.contract.seams import DeferralReason

    def run(judge):
        agent = Agent.__new__(Agent)
        agent.verifier = Verifier(threshold=0.5, sanity=False, judge=judge, fail_closed=True)
        packet = ContextPacket(question="q", cards=[], grant_fingerprint="",
                               enrichment_version=None)
        answer, _ = agent._verified(packet=packet,
                                    approved=Approved(plan_sql="SELECT 1", target_sql="SELECT 1"),
                                    table=pa.table({"n": [1]}))
        return answer

    outage = run(_Judge(1.0, falls_open=True))
    assert outage.failed is True and outage.deferred is False
    assert outage.reason_code is DeferralReason.VERIFIER_UNAVAILABLE

    judged = run(_Judge(0.1))
    assert judged.failed is False and judged.deferred is True
    assert judged.reason_code is DeferralReason.VERIFICATION


@pytest.mark.parametrize("why, blames", [("error", "could not be reached"),
                                         ("unparsed", "a reply I could not read")])
def test_the_two_fell_open_causes_do_not_share_a_sentence(why, blames):
    """`JudgeRead.reason` splits them and the verifier flattened them back. `unparsed` is the
    endpoint ANSWERING with a reply holding no confidence -- a model or a token budget, not
    connectivity -- and telling an operator the judge was unreachable sends them to check the
    network. That is the shape `test_llm_reasoning_budget` exists for, where a starved reasoning
    model silently disabled the verifier for every answer."""
    v = _verify(_Judge(1.0, falls_open=True, why=why))
    assert blames in v.reason

    # The non-stopping branch too. Composing these from a shared clause produced "The verifier
    # it could not be reached", which no test read because only the stopping branch was checked
    # -- and that is the branch a caller sees when an operator has turned fail-closed off.
    # Nothing renders this one today -- `loop` reads `reason` only off a verdict that stops the
    # answer -- which is exactly why it went unread while being ungrammatical.
    open_v = _verify(_Judge(1.0, falls_open=True, why=why), fail_closed=False)
    assert blames in open_v.reason
    assert "verifier it" not in open_v.reason and "verifier its" not in open_v.reason


def test_the_eval_path_reads_the_same_switch():
    """Two call sites build a verifier from settings and only one had a test. Dropping
    `fail_closed=` in `eval/engine` left a sweep on the constructor default whatever the
    deployment set, and nothing noticed, because reaching that line meant standing up an
    adapter, a connection and an LLM client."""
    from mnemiq.config import Settings
    from mnemiq.eval.engine import verifier_from

    for configured in (True, False):
        s = Settings(llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
                     acme_data_dir="a", verify_fail_closed=configured)
        assert verifier_from(s, judge=None).fail_closed is configured

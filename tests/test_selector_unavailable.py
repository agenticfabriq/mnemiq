"""A selector that could not reach its judge must not report the pick as judged.

M11's second clause, and the same defect as its first one layer over. `LLMSelector.select`
fails CLOSED -- a dead endpoint, an unreadable reply or an out-of-range pick all fall back to
the majority vote -- which is right for the product and a lie in the telemetry: `loop.py`
derives `judge_engaged` from whether the clusters DISAGREED, never from whether the judge
answered, so a failed selector produces `judge_engaged=True, judge_override=False`, byte for
byte what a judge that looked and agreed with the majority produces. That pair ships on
`/v1/ask` and in the Verity trace.

The fallback stays. What changes is that the record stops claiming a judgement happened --
the same fix `judge_unavailable` is for the verifier, and the reason the two files read alike.
"""
import pytest

from mnemiq.execute.select import ClusterView, FakeSelector, LLMSelector
from tests.test_agent import _answer, _sql, _vote_agent


def _views(*sizes):
    return [ClusterView(sql=f"SELECT {i}", preview=f"n\n{i}\n(1 rows)", size=s)
            for i, s in enumerate(sizes)]


class _Client:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, system, user, max_tokens=512):
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


# --- the selector half: the pick comes back with whether it WAS a pick ---

def test_a_dead_endpoint_falls_back_and_says_so():
    got = LLMSelector(_Client(RuntimeError("endpoint down"))).read("q", _views(2, 3))
    assert got.choice == 1, "the fallback itself is unchanged -- majority, as before"
    assert got.fell_back is True
    assert got.reason == "error"


def test_an_unreadable_reply_falls_back_and_says_so():
    got = LLMSelector(_Client("no idea")).read("q", _views(2, 3))
    assert got.choice == 1 and got.fell_back is True
    assert got.reason == "unparsed"


def test_a_pick_outside_the_clusters_falls_back_and_says_so():
    """Distinct from `unparsed`: the model answered in the right SHAPE and named a cluster that
    does not exist. A model that cannot emit the format and one that miscounted the candidates
    are different problems, and only one of them is fixed by a better prompt."""
    got = LLMSelector(_Client('{"choice": 9}')).read("q", _views(2, 3))
    assert got.choice == 1 and got.fell_back is True
    assert got.reason == "out_of_range"


def test_a_judge_that_picked_the_majority_ANYWAY_is_a_judgement():
    """The pair that makes this necessary: identical `choice`, opposite fact. Without the flag
    these two are the same three integers downstream."""
    got = LLMSelector(_Client('{"choice": 1}')).read("q", _views(2, 3))
    assert got.choice == 1 and got.fell_back is False
    assert got.reason == "ok"


def test_select_still_speaks_the_int_protocol():
    """`Selector` is a Protocol and anyone's stub implements `select`. Changing that contract
    would break every caller to fix a telemetry field."""
    assert LLMSelector(_Client('{"choice": 1}')).select("q", _views(2, 3)) == 1
    assert LLMSelector(_Client(RuntimeError("down"))).select("q", _views(2, 3)) == 1


# --- the loop half: what the answer claims ---

def _agent_with(selector):
    agent = _vote_agent([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")], 3)
    agent.selector = selector
    return agent


def test_a_failed_selector_does_not_report_a_judged_pick():
    ans = _answer(_agent_with(LLMSelector(_Client(RuntimeError("endpoint down")))))
    assert ans.judge_engaged is True, "it WAS consulted -- that field keeps its meaning"
    assert ans.judge_fell_back is True
    assert ans.deferred is False, "a dead selector must not stop an answer, exactly as before"


def test_a_selector_that_answered_reports_a_judged_pick():
    ans = _answer(_agent_with(LLMSelector(_Client('{"choice": 0}'))))
    assert ans.judge_engaged is True
    assert ans.judge_fell_back is False
    assert ans.judge_override is False, "it agreed with the majority -- and that is now sayable"


def test_a_selector_never_consulted_has_nothing_to_report():
    agent = _vote_agent([_sql("count(*)")] * 3, 3)   # unanimous: selection is vacuous
    agent.selector = FakeSelector([0])
    ans = _answer(agent)
    assert ans.judge_engaged is False
    assert ans.judge_fell_back is None, "None means no judgement was attempted, not that one held"


def test_no_selector_wired_reports_nothing_either():
    ans = _answer(_vote_agent([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")], 3))
    assert ans.judge_engaged is None and ans.judge_fell_back is None


def test_a_selector_without_read_is_taken_at_its_word():
    """Absence of `read` is not evidence of a fallback. `FakeSelector` and anyone's stub speak
    only `select`; reporting them as broken would make every scripted run look degraded --
    the same rule the verifier applies to a judge with no `read`."""
    ans = _answer(_agent_with(FakeSelector([1])))
    assert ans.judge_engaged is True and ans.judge_fell_back is False
    assert ans.judge_override is True


@pytest.mark.parametrize("reply,fell_back", [('{"choice": 0}', False),
                                             (RuntimeError("down"), True),
                                             ("no idea", True)])
def test_the_fact_reaches_the_wire(reply, fell_back):
    """`judge_engaged` already ships on /v1/ask. Stamping the answer engine-side changes nothing
    a client reads, and the client is where the false claim was published."""
    from mnemiq.server.serialize import answer_payload

    p = answer_payload(_answer(_agent_with(LLMSelector(_Client(reply)))))
    assert p["judge_engaged"] is True
    assert p["judge_fell_back"] is fell_back


def test_the_fact_reaches_the_trace():
    """The audit record is the other reader, and it hand-builds its own dict beside
    `judge_engaged`. Two surfaces reading one answer must not disagree about whether a judgement
    happened -- and the audit store is the reader that cannot ask again later.

    Built by the production builder rather than asserting on the field list, because the failure
    being pinned is a key that never gets written, which a shape assertion cannot see.
    """
    import sys

    sys.path.insert(0, "tests")
    from test_verity_trace_sink import _Settings, _event

    from mnemiq.observability.trace_sink import VerityTraceSink

    ans = _answer(_agent_with(LLMSelector(_Client(RuntimeError("down")))))
    record = VerityTraceSink(_Settings())._build_trace(_event(answer=ans))
    assert record["resolved_intent"]["judge_engaged"] is True
    assert record["resolved_intent"]["judge_fell_back"] is True

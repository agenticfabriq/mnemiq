"""The disclosure reaches the caller, on every surface, without four places remembering to print it.

The four surfaces -- human CLI, HTTP, MCP, `--json` -- were each fixed in turn for the lineage
marker, and `cli.py` still carries the record of it: commit messages saying "the two surfaces",
then three, then four. A disclosure that four renderers must remember is a disclosure that will be
missing from one of them.

So the sentence is appended ONCE, where the answer is assembled, and every surface inherits it by
printing `answer`. The structured form rides on the trace for consumers that must not parse prose.
"""

from mnemiq.contract.seams import Narrowed, disclosure_sentence


def test_the_sentence_is_the_wording_that_was_chosen():
    """Product voice, decided rather than derived: honest and slightly alarming beats calm and
    vague, because the caller's next action depends on knowing the answer is partial."""
    assert disclosure_sentence([Narrowed(object="claim", rows=True, columns=False)]) == (
        "Some rows were withheld by policy.")


def test_a_mask_and_a_filter_read_differently():
    """Two channels inside one signal. Rendering a row-filtered object as "columns were masked"
    produces an identical set on both sides and passes any test that only counts disclosures."""
    mask = disclosure_sentence([Narrowed(object="claim", rows=False, columns=True)])
    both = disclosure_sentence([Narrowed(object="claim", rows=True, columns=True)])
    assert mask == "Some columns were masked by policy."
    assert both == "Some rows were withheld and some columns masked by policy."
    assert mask != both


def test_silence_when_nothing_was_narrowed_AND_when_nothing_was_evaluated():
    """A disclosure is a CLAIM. Neither state supports one, so both render nothing -- and what
    distinguishes them is `Trace.narrowed`, which is `[]` against `None` for a machine reader."""
    assert disclosure_sentence([]) == ""
    assert disclosure_sentence(None) == ""


def test_the_answer_carries_it_so_every_surface_does():
    """The load-bearing test. If this holds, no surface can omit the sentence, because they all
    print `answer` -- which is the arrangement that makes "we fixed three of the four" impossible.
    """
    import inspect

    from mnemiq.agent import loop

    src = inspect.getsource(loop)
    assert "disclosure_sentence(trace.narrowed)" in src, (
        "the answer no longer carries the disclosure; each surface would have to render it")
    i, j = src.index("disclosure_sentence(trace.narrowed)"), src.index("return AgentAnswer(answer=answer")
    assert i < j, "the disclosure must be appended BEFORE the answer is returned"


class _Holder:
    def __init__(self, narrowed=None):
        self.narrowed = narrowed


def test_the_audit_record_carries_what_the_decision_DID_not_only_a_hash_of_the_grant():
    """Spec item 4, and it is not satisfied by `Trace` carrying the field.

    The sink builds its payload key by key, so a new `Trace` field is dropped silently -- I marked
    item 4 done once while `policy_decisions` was still the empty list it had always been, beside a
    `grant_fingerprint` whose own comment calls itself "a hash of the policy, not the policy".
    """
    from mnemiq.contract.seams import Narrowed
    from mnemiq.observability.trace_sink import _policy_decisions

    got = _policy_decisions(_Holder([Narrowed(object="claim", rows=True, columns=False)]),
                            _Holder(None))
    assert got and got[0]["metadata"]["object"] == "claim"


def test_every_entry_matches_the_STORES_required_shape():
    """`policy_decisions` is `Vec<trace_schema::PolicyDecisionV1>`, not free-form JSON: `policy_id`
    and `effect` are required strings. A first version emitted `{object, rows, columns}` and the
    store would have rejected the whole trace.

    Asserted by CALLING the builder. The first version of this test regex-matched the sink's source
    and eval-ed the expression, so it tested the text and broke when the expression changed.
    """
    from mnemiq.contract.seams import Narrowed
    from mnemiq.observability.trace_sink import _policy_decisions

    got = _policy_decisions(
        _Holder([Narrowed(object="claim", rows=True, columns=False),
                 Narrowed(object="person", rows=False, columns=True),
                 Narrowed(object="both", rows=True, columns=True)]), _Holder(None))
    assert len(got) == 3
    for d in got:
        assert isinstance(d.get("policy_id"), str) and d["policy_id"]
        assert isinstance(d.get("effect"), str) and d["effect"]
        assert isinstance(d.get("metadata"), dict)
    assert {d["effect"] for d in got} == {"row_filter", "column_mask", "row_filter+column_mask"}


def test_policy_decisions_is_NEVER_null_because_the_store_cannot_deserialize_it():
    """`#[serde(default)]` covers a missing key, not an explicit null. Emitting null rejected the
    whole trace -- storing nothing where the old empty list at least stored the rest."""
    from mnemiq.observability.trace_sink import _policy_decisions

    assert _policy_decisions(_Holder(None), _Holder(None)) == []


def test_a_POST_DECISION_failure_is_not_recorded_as_ungoverned():
    """The trace is built after execution, so a verifier deferral or a synthesis failure returns an
    answer with NO trace -- and reading the trace alone recorded `access_evaluated: false` for a
    query that was governed and had already run against the source. A false audit fact, on the
    record that outlives the answer.
    """
    from mnemiq.contract.seams import Narrowed
    from mnemiq.observability.trace_sink import _decision, _policy_decisions

    answer_only = _Holder([Narrowed(object="claim", rows=True, columns=False)])
    assert _decision(None, answer_only) is not None, "post-decision failure reads as ungoverned"
    assert _policy_decisions(None, answer_only), "its effects are lost from the audit record"
    # and a request that never reached a decision still records that honestly
    assert _decision(None, _Holder(None)) is None


def test_the_VERIFIER_DEFERRAL_carries_the_decision_it_has_no_trace_for():
    """Exercised, not counted.

    An earlier version of this guard counted `narrowed=` occurrences in the module, which stayed
    green when the verifier path dropped it -- the arithmetic held because other sites carried it.
    This drives `_verified` with a verifier that defers, on an `approved` that was narrowed, and
    asserts the answer it returns still knows.

    That path is the sharpest case in the whole feature: the SQL has already executed against the
    source when the verifier defers, so the decision is a fact about a query that ran.
    """
    from types import SimpleNamespace

    from mnemiq.agent.loop import Agent
    from mnemiq.contract.seams import Narrowed

    approved = SimpleNamespace(narrowed=[Narrowed(object="claim", rows=True, columns=False)])
    deferring = SimpleNamespace(
        verify=lambda packet, approved, table: SimpleNamespace(
            defer=True, reason="not confident", confidence=0.1, layer="judge"))
    agent = Agent.__new__(Agent)
    agent.verifier = deferring

    answer, _verdict = agent._verified(packet=None, approved=approved, table=None)
    assert answer is not None and answer.deferred
    assert answer.narrowed == approved.narrowed, (
        "the verifier deferred AFTER the query ran; its answer must still carry what was narrowed")

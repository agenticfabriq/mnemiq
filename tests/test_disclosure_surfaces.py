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


def _sink_source():
    import inspect

    from mnemiq.observability import trace_sink

    return inspect.getsource(trace_sink)


def test_the_audit_record_carries_what_the_decision_DID_not_only_a_hash_of_the_grant():
    """Spec item 4, and it is not satisfied by `Trace` carrying the field.

    The sink builds its payload key by key, so a new `Trace` field is silently dropped -- I marked
    item 4 done once while `policy_decisions` was still the empty list it had always been. The
    store could distinguish two answers under different grants (`grant_fingerprint`) and could not
    say what either grant DID.
    """
    src = _sink_source()
    assert '"policy_decisions": [],' not in src, "restored the hardcoded empty decision list"
    assert 'getattr(trace, "narrowed", None)' in src, "decisions are not sourced from the trace"


def test_every_entry_matches_the_STORES_required_shape():
    """`policy_decisions` is `Vec<trace_schema::PolicyDecisionV1>`, not free-form JSON:
    `policy_id` and `effect` are required strings. A first version emitted
    `{object, rows, columns}` and the store would have rejected the whole trace.

    Asserted on the built record rather than on the source, because the shape is what travels.
    """
    from mnemiq.contract.seams import Narrowed

    class _T:
        narrowed = [Narrowed(object="claim", rows=True, columns=False),
                    Narrowed(object="person", rows=False, columns=True)]

    src = _sink_source()
    i = src.index('"policy_decisions": [')
    j = src.index("],", i)
    expr = src[i + len('"policy_decisions": '):j + 1]
    decisions = eval(expr, {"getattr": getattr}, {"trace": _T, "n": None})  # noqa: S307
    assert len(decisions) == 2
    for d in decisions:
        assert isinstance(d.get("policy_id"), str) and d["policy_id"], d
        assert isinstance(d.get("effect"), str) and d["effect"], d
        assert isinstance(d.get("metadata"), dict), d
    assert {d["effect"] for d in decisions} == {"row_filter", "column_mask"}
    assert {d["metadata"]["object"] for d in decisions} == {"claim", "person"}


def test_policy_decisions_is_NEVER_null_because_the_store_cannot_deserialize_it():
    """`#[serde(default)]` covers a missing key, not an explicit null. Emitting null rejected the
    whole trace -- storing nothing where the old empty list at least stored the rest."""
    src = _sink_source()
    i = src.index('"policy_decisions"')
    j = src.index("],", i)
    assert "else None" not in src[i:j], "a null policy_decisions rejects the trace at the store"


def test_the_not_evaluated_marker_lives_where_the_record_is_free_form():
    """A typed Vec has one empty, so `[]` cannot mean both "narrowed nothing" and "never ran".
    `collector_metadata` is `serde_json::Value` on the store side, so the marker goes there."""
    src = _sink_source()
    assert '"access_evaluated": getattr(trace, "narrowed", None) is not None' in src

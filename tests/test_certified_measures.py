"""Certified metrics and dimensions must reach the model, not just the snapshot.

`apply_certified` loaded them, validated them, appended them to `snapshot.metrics` /
`snapshot.dimensions` — and nothing read either list. Measured across the tree: `definitions` had 11
consumers and `relationships` 7; metrics and dimensions had **zero**.

On the fs payments corpus that is 5 of 28 certified records reaching nothing, and they are the five
that carry the most explicit meaning a schema cannot express: the exact SQL for settled, gross and
net volume, for counting payments by business id rather than CDC revision, and for a tip rate that
is a ratio of sums rather than a mean of ratios.

**A metric rides with its table.** Selection by term-match — the rule definitions use — would be
wrong here: a definition is looked up by the word the asker used, but a metric matters most exactly
when the asker does *not* name it ("revenue last quarter" rather than "settled payment volume").
So a metric is offered when its source table is in the retrieved context, which is the same rule
`_attach_facts` already applies to structural facts.
"""

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Dimension, MeasureExpr, Metric
from mnemiq.semantic.measures import select_dimensions, select_metrics
from mnemiq.generate.prompts import user_prompt
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, retrieve


def _settled() -> Metric:
    return Metric(
        id="settled_payment_volume",
        label="Settled Payment Volume",
        status="certified",
        owner="payments_analytics",
        grain="payment_transaction_date",
        measure=MeasureExpr(
            expr="sum(case when payment_transaction_status = 'SETTLED' then payment_amount else 0 end)",
            source="payment_transaction",
        ),
        time_dimension="payment_transaction_date",
    )


def _elsewhere() -> Metric:
    return Metric(
        id="refund_volume",
        label="Refund Volume",
        status="certified",
        owner="payments_analytics",
        grain="refund_date",
        measure=MeasureExpr(expr="sum(refund_amount)", source="refund"),
        time_dimension="refund_date",
    )


def _channel() -> Dimension:
    return Dimension(id="payment_channel", label="Payment Channel", source="payment_transaction")


def _granting(*objects: str) -> GrantSet:
    return GrantSet(objects=frozenset(objects))


def test_a_metric_is_offered_when_its_table_is_in_context():
    selected = select_metrics(["payment_transaction"], [_settled(), _elsewhere()], _granting("payment_transaction", "refund"))

    assert [m.id for m in selected] == ["settled_payment_volume"], (
        "a metric over a table that is not in context is noise; one over a table that is, is the "
        "certified way to measure it"
    )


def test_a_metric_over_an_ungranted_table_is_never_offered():
    # Same rule the glossary applies: showing it leaks that the table exists and steers the model
    # into SQL the decider must reject.
    selected = select_metrics(["payment_transaction"], [_settled()], _granting("refund"))

    assert selected == []


def test_dimensions_follow_their_table_too():
    selected = select_dimensions(["payment_transaction"], [_channel()], _granting("payment_transaction", "refund"))
    assert [d.id for d in selected] == ["payment_channel"]

    assert select_dimensions(["refund"], [_channel()], _granting("payment_transaction", "refund")) == []


def test_the_question_does_not_have_to_name_the_metric():
    """The point of the whole thing. `select_definitions` matches the asker's words; a metric that
    only appeared when named would be absent exactly when it is needed most."""
    selected = select_metrics(["payment_transaction"], [_settled()], _granting("payment_transaction", "refund"))
    assert selected, "no question text was consulted at all, and that is deliberate"


# --- the wiring: selection is worthless if the packet never carries it to the prompt -------------


def _packet(**kwargs) -> ContextPacket:
    return ContextPacket(
        question="what was revenue last quarter",
        cards=[RetrievedCard(object_id="payment_transaction", card="TABLE payment_transaction", score=1.0)],
        grant_fingerprint="fp",
        enrichment_version="v1",
        **kwargs,
    )


def test_the_prompt_carries_the_certified_measure_expression():
    """The expression IS the meaning here. A prompt naming the metric without saying how it is
    computed tells the model a phrase it already had and nothing it did not."""
    prompt = user_prompt(_packet(metrics=[_settled()]))

    assert "Settled Payment Volume" in prompt
    assert "payment_transaction_status = 'SETTLED'" in prompt, (
        "the certified SQL is the whole payload; without it the metric is a label"
    )


def test_the_prompt_says_a_certified_measure_is_authoritative():
    prompt = user_prompt(_packet(metrics=[_settled()]))
    section = prompt[prompt.index("CERTIFIED"):]
    assert "follow" in section.lower() or "authoritative" in section.lower(), (
        "a measure offered as a suggestion is one the model may improve on; the point is that it "
        "may not"
    )


def test_a_packet_with_no_measures_says_nothing_about_them():
    # Non-vacuity in the other direction: an empty section would be noise on every question that
    # has no certified metric, and would make the section meaningless where it appears.
    assert "CERTIFIED" not in user_prompt(_packet())


def test_retrieve_fills_the_packet_from_the_snapshot(tmp_path):
    """The end-to-end link, and the one a unit test of `select_metrics` cannot make.

    Every piece of this shipped correct in isolation once before — D113's citation control worked
    on one screen while the screen the question was asked about had no resolver passed to it. A
    selection function that nothing calls is the same defect one layer down.
    """
    from tests.test_retrieval import FakeEmbedder, _StaticAuthz, _con, _identity

    packet = retrieve(
        _con(tmp_path),
        "claim_identifier",
        _identity(),
        _StaticAuthz("claim", "policy"),
        FakeEmbedder(),
        metrics=[_claim_metric()],
        dimensions=[_claim_dimension()],
    )

    assert [c.object_id for c in packet.cards][:1] == ["claim"], "setup: the claim card is retrieved"
    assert [m.id for m in packet.metrics] == ["claim_volume"]
    assert [d.id for d in packet.dimensions] == ["claim_channel"]


def _claim_metric() -> Metric:
    return Metric(
        id="claim_volume", label="Claim Volume", status="certified", owner="o",
        grain="claim_date",
        measure=MeasureExpr(expr="sum(paid_amount)", source="claim"),
        time_dimension="claim_date",
    )


def _claim_dimension() -> Dimension:
    return Dimension(id="claim_channel", label="Claim Channel", source="claim")

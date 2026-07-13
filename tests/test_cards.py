from mnemiq.contract import CodedValue, Column, Relationship, Snapshot, SourceBinding
from mnemiq.semantic.cards import build_cards


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        source_bindings=[
            SourceBinding(
                id="sb:claim",
                source_id="acme",
                object_id="claim",
                source_object="claim",
                binding_type="table",
            ),
            SourceBinding(
                id="sb:party",
                source_id="acme",
                object_id="party",
                source_object="party",
                binding_type="table",
            ),
        ],
        columns=[
            Column(
                id="claim.claim_identifier",
                object_id="claim",
                name="claim_identifier",
                data_type="integer",
                semantic_type="identifier",
                description="A unique identifier for the claim.",
            ),
            Column(
                id="claim.status",
                object_id="claim",
                name="status",
                data_type="text",
                semantic_type="code",
                description="The claim's processing state.",
                coded_values=[CodedValue(code="S", meaning="suspended")],
            ),
            Column(id="party.name", object_id="party", name="name", data_type="text"),
        ],
        relationships=[
            Relationship(
                id="claim.party_identifier->party",
                from_="claim",
                to="party",
                cardinality="many_to_one",
            )
        ],
    )


def test_one_card_per_table():
    cards = build_cards(_snapshot())
    assert {c.object_id for c in cards} == {"claim", "party"}


def test_card_is_self_sufficient():
    card = next(c for c in build_cards(_snapshot()) if c.object_id == "claim")

    assert "claim" in card.text
    assert "claim_identifier" in card.text and "integer" in card.text
    assert "A unique identifier for the claim." in card.text
    # the coded vocabulary and its MEANING -- this is what makes the card searchable by intent
    assert "S = suspended" in card.text
    # joins belong on the card: the agent writes SQL from it
    assert "party" in card.text and "many_to_one" in card.text


def test_a_table_with_no_enrichment_still_gets_a_card():
    card = next(c for c in build_cards(_snapshot()) if c.object_id == "party")
    assert "party" in card.text and "name" in card.text

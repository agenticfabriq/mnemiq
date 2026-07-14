from mnemiq.contract import Definition
from mnemiq.generate.prompts import user_prompt
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard


def _packet(definitions=()) -> ContextPacket:
    return ContextPacket(
        question="what is the total premium amount?",
        cards=[RetrievedCard(object_id="premium", card="TABLE premium ...", score=1.0)],
        grant_fingerprint="abc",
        enrichment_version="v1",
        definitions=list(definitions),
    )


def test_definitions_render_as_authoritative():
    definition = Definition(
        id="def-premium", term="premium", domain="policy",
        definition="Premium amounts live in policy_amount, joined through premium.",
        bound_objects=["premium", "policy_amount"],
    )
    prompt = user_prompt(_packet([definition]))

    assert "DEFINITIONS" in prompt
    assert "premium: Premium amounts live in policy_amount" in prompt
    # definitions come before the tables: the model reads meaning before schema
    assert prompt.index("DEFINITIONS") < prompt.index("TABLES:")


def test_no_definitions_no_block():
    assert "DEFINITIONS" not in user_prompt(_packet())

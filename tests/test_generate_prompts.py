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


def test_direct_strategy_leaves_the_system_prompt_byte_identical():
    from mnemiq.generate.prompts import system_prompt

    base = system_prompt()
    assert system_prompt(strategy=None) == base
    assert system_prompt(strategy="direct") == base


def test_decompose_and_skeleton_strategies_append_their_preamble():
    from mnemiq.generate.prompts import STRATEGY_PREAMBLES, system_prompt

    base = system_prompt()
    for name in ("decompose", "skeleton"):
        prompt = system_prompt(strategy=name)
        assert prompt.startswith(base)
        assert STRATEGY_PREAMBLES[name] in prompt


def test_an_unknown_strategy_falls_back_to_direct():
    from mnemiq.generate.prompts import system_prompt

    assert system_prompt(strategy="nonsense") == system_prompt()


def test_user_prompt_renders_examples_when_present():
    from mnemiq.contract import Example
    from mnemiq.generate.prompts import user_prompt
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    packet = ContextPacket(
        question="how many claims?",
        cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1",
        examples=[Example(question="total?", sql="SELECT count(*) FROM claim",
                          tables=["claim"], object_id="claim")],
    )
    text = user_prompt(packet)
    assert "WORKED EXAMPLES" in text and "SELECT count(*) FROM claim" in text


def test_user_prompt_without_examples_is_unchanged():
    from mnemiq.generate.prompts import user_prompt
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    packet = ContextPacket(question="q", cards=[RetrievedCard(object_id="c", card="TABLE c",
                            score=1.0)], grant_fingerprint="fp", enrichment_version="v1")
    assert "WORKED EXAMPLES" not in user_prompt(packet)

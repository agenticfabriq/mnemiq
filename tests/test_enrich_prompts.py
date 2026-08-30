from mnemiq.enrichment.prompts import ColumnFacts, render_grounding_block, user_prompt
from mnemiq.enrichment.enricher import FakeEnricher

_FACTS = "TABLE: t\nCOLUMNS:\n- x (type=text)"


def test_empty_grounding_yields_empty_block():
    assert render_grounding_block([]) == ""


def test_grounding_block_has_reference_header_and_items():
    block = render_grounding_block([("Premium", "the amount paid for coverage", 0.9)])
    assert "REFERENCE" in block
    assert "Premium: the amount paid for coverage" in block


def test_user_prompt_without_grounding_is_unchanged():
    assert user_prompt(_FACTS) == f"{_FACTS}\n\nDocument these columns as JSON."


def test_user_prompt_with_grounding_prepends_reference():
    block = render_grounding_block([("Premium", "cost of coverage", 0.9)])
    out = user_prompt(_FACTS, block)
    assert out.startswith("REFERENCE")
    assert _FACTS in out and out.rstrip().endswith("Document these columns as JSON.")


def test_fake_enricher_records_grounding():
    enr = FakeEnricher(replies={"t": '{"columns": [{"name": "x"}]}'})
    enr.annotate("t", [ColumnFacts(name="x", data_type="text")], grounding="REFERENCE ...")
    assert enr.grounding_calls["t"] == "REFERENCE ..."

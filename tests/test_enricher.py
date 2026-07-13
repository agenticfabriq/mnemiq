import os

import pytest

from mnemiq.enrichment.enricher import FakeEnricher, LLMEnricher
from mnemiq.enrichment.prompts import ColumnFacts

FACTS = [ColumnFacts("fireplace", "text", codes=["yes", "no"], row_count=820, distinct_count=2)]

GOOD = """{"columns": [{"name": "fireplace", "description": "Has a fireplace.",
           "semantic_type": "boolean", "pii_level": "none",
           "code_meanings": {"yes": "has one", "no": "has none"}}]}"""


class _StubClient:
    """Stands in for LLMClient: records the prompts, replays scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_llm_enricher_sends_facts_and_screens_the_reply():
    client = _StubClient([GOOD])
    ann = LLMEnricher(client).annotate("fireclaim", FACTS)

    system, user = client.prompts[0]
    assert "CLOSED WORLD" in system
    assert "fireplace" in user and "820" in user  # the deterministic facts went out

    col = ann.columns[0]
    assert col.semantic_type == "boolean"
    assert col.code_meanings["yes"] == "has one"


def test_llm_enricher_retries_once_on_an_unusable_reply():
    client = _StubClient(["I'd rather not.", GOOD])
    ann = LLMEnricher(client).annotate("fireclaim", FACTS)
    assert len(client.prompts) == 2
    assert ann.columns[0].semantic_type == "boolean"


def test_llm_enricher_gives_up_after_the_retry():
    client = _StubClient(["nope", "still nope"])
    ann = LLMEnricher(client).annotate("fireclaim", FACTS)
    assert len(client.prompts) == 2
    assert ann.columns == []  # empty, not an exception: the caller decides what that means


def test_llm_enricher_screens_a_hallucinating_model():
    hallucination = '{"columns": [{"name": "ssn", "pii_level": "pii"}]}'
    ann = LLMEnricher(_StubClient([hallucination, hallucination])).annotate("fireclaim", FACTS)
    assert ann.columns == []  # the model cannot add a column that does not exist


def test_llm_enricher_does_not_call_the_model_for_an_empty_table():
    client = _StubClient([])
    assert LLMEnricher(client).annotate("empty", []).columns == []
    assert client.prompts == []  # no tokens spent on nothing


def test_fake_enricher_uses_the_real_validator():
    fake = FakeEnricher({"fireclaim": GOOD})
    ann = fake.annotate("fireclaim", FACTS)
    assert ann.columns[0].code_meanings == {"yes": "has one", "no": "has none"}
    assert fake.calls == ["fireclaim"]

    assert fake.annotate("unknown_table", FACTS).columns == []  # no canned reply -> empty


@pytest.mark.live_llm
@pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured")
def test_live_model_documents_a_real_column():
    from mnemiq.config import Settings
    from mnemiq.llm.client import LLMClient

    ann = LLMEnricher(LLMClient(Settings.from_env())).annotate("fireclaim", FACTS)
    col = next(c for c in ann.columns if c.name == "fireplace")
    assert col.description
    assert col.semantic_type in {"boolean", "code"}
    assert col.pii_level == "none"
    assert set(col.code_meanings) <= {"yes", "no"}

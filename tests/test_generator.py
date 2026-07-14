import os

import pytest

from mnemiq.generate.generator import FakeGenerator, LLMGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

CARD = """TABLE claim
COLUMNS:
- claim_identifier (integer, identifier) A unique identifier for the claim.
- claim_open_date (timestamp, date) When the claim was opened."""


def _packet(question="how many claims were opened in 2020?") -> ContextPacket:
    return ContextPacket(
        question=question,
        cards=[RetrievedCard(object_id="claim", card=CARD, score=1.0)],
        grant_fingerprint="abc123",
        enrichment_version="v1",
    )


class _StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_the_prompt_carries_the_cards_and_the_question():
    client = _StubClient(['{"sql": "SELECT claim_identifier FROM claim", "reason": "ok"}'])
    proposal = LLMGenerator(client).propose(_packet())

    system, user = client.prompts[0]
    assert "SELECT" in system  # the dialect and the rules
    assert "claim_open_date" in user  # the card went out
    assert "how many claims were opened in 2020?" in user

    assert proposal.sql == "SELECT claim_identifier FROM claim"


def test_a_deferral_is_a_valid_answer_not_a_failure():
    client = _StubClient(['{"sql": null, "reason": "no table records payouts"}'])
    proposal = LLMGenerator(client).propose(_packet())
    assert proposal.sql is None
    assert "payouts" in proposal.reason


def test_a_fenced_reply_still_parses():
    client = _StubClient(['```json\n{"sql": "SELECT claim_identifier FROM claim"}\n```'])
    assert LLMGenerator(client).propose(_packet()).sql == "SELECT claim_identifier FROM claim"


def test_an_unusable_reply_becomes_a_deferral_not_an_exception():
    client = _StubClient(["I'd rather not", "still no"])
    proposal = LLMGenerator(client).propose(_packet())
    assert proposal.sql is None  # the loop decides what to do; the generator does not crash


def test_feedback_is_included_on_a_repair():
    client = _StubClient(['{"sql": "SELECT claim_identifier FROM claim"}'])
    LLMGenerator(client).propose(_packet(), feedback="Do not use SELECT *.")
    _system, user = client.prompts[0]
    assert "Do not use SELECT *." in user


def test_fake_generator_replays_through_the_real_parser():
    fake = FakeGenerator(['{"sql": "SELECT claim_identifier FROM claim", "reason": "r"}'])
    assert fake.propose(_packet()).sql == "SELECT claim_identifier FROM claim"


@pytest.mark.live_llm
@pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured")
def test_the_live_model_writes_sql_for_a_real_card():
    from mnemiq.config import Settings
    from mnemiq.llm.client import LLMClient

    proposal = LLMGenerator(LLMClient(Settings.from_env())).propose(_packet())
    assert proposal.sql is not None
    assert "claim" in proposal.sql.lower()


@pytest.mark.live_llm
@pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured")
def test_the_live_model_defers_when_the_cards_cannot_answer():
    from mnemiq.config import Settings
    from mnemiq.llm.client import LLMClient

    packet = _packet("what is the average salary of our employees?")
    proposal = LLMGenerator(LLMClient(Settings.from_env())).propose(packet)
    assert proposal.sql is None, f"should have deferred, got: {proposal.sql}"

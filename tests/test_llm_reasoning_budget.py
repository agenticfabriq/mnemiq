"""A reasoning model's cap must cover the thinking, not just the reply.

The failure this prevents is silent in the worst way: the provider turns a truncated reasoning
pass into a request error, `SemanticJudge.score` catches every exception and returns its fail-open
1.0, and 1.0 is exactly what a judge that approved the answer returns. The verifier switches off
and every answer looks confidently verified.
"""
from types import SimpleNamespace

from mnemiq.config import Settings
from mnemiq.llm.client import LLMClient, reasoning_budget, token_param_name
from mnemiq.verify.judge import SemanticJudge


def test_a_reasoning_model_gets_its_reserve_on_top_of_the_request():
    # The shipped `SemanticJudge` default. Measured at 10/10 provider failures on real prompts.
    assert reasoning_budget("openai.gpt-5.5", 200) == 1224
    assert reasoning_budget("openai.gpt-5-mini", 512) == 1536


def test_every_caller_keeps_the_output_room_it_asked_for():
    """The distinction from a floor, and the reason this is not one. `synthesize` asks 1000
    because it wants 1000 tokens of prose; a floor at 1024 would hand it 24 tokens to think in
    and call the problem solved. Both of these callers must come out ABOVE what they requested."""
    assert reasoning_budget("openai.gpt-5.5", 1000) == 2024      # synthesize, correct
    assert reasoning_budget("openai.gpt-5.5", 4000) == 5024      # generator


def test_a_non_reasoning_model_is_untouched():
    # gpt-4o answered fine at 16 tokens; it spends nothing before emitting, so it needs no floor
    # and must not be charged a bigger cap.
    assert reasoning_budget("openai.gpt-4o", 200) == 200
    assert reasoning_budget("qwen2.5-coder-14b", 200) == 200


def test_the_reserve_follows_the_same_family_test_as_the_token_parameter():
    """Both answer "is this a reasoning model?". They share `_GPT5` today, so this cannot fail
    until someone gives one of them its own predicate -- which is exactly when the two would
    start disagreeing about the model in front of them, and the moment worth catching."""
    # `mygpt-5` earns its place: it is the only name here that an unanchored `gpt-5` classifies
    # differently from `_GPT5`, so it is what makes this loop a test rather than a restatement.
    for model in ("openai.gpt-5.5", "openai.gpt-5-mini", "openai.gpt-4o", "llama-3.1-70b",
                  "mygpt-5", "gpt-5", "vendor.gpt-50"):
        reserved = reasoning_budget(model, 200) != 200
        assert reserved == (token_param_name(model) == "max_completion_tokens"), model


class _CapturingOpenAI:
    """Records the kwargs `complete` actually sends, so a test can read the outgoing cap."""

    sent: dict = {}

    def __init__(self, *_, **__):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        type(self).sent = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"confidence": 0.5}'))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def _client(monkeypatch, model: str) -> LLMClient:
    import mnemiq.llm.client as module

    monkeypatch.setattr(module, "OpenAI", _CapturingOpenAI)
    return LLMClient(Settings(llm_base_url="http://x", llm_api_key="k", llm_model=model,
                              pg_dsn=None, acme_data_dir=None))


def test_the_reserve_reaches_the_provider(monkeypatch):
    """The unit tests above prove `reasoning_budget` computes the right number; this proves
    `complete` actually sends it. Without this, dropping the call and keeping the function leaves
    the whole suite green with the fix deleted."""
    _client(monkeypatch, "openai.gpt-5.5").complete("s", "u", max_tokens=200)
    assert _CapturingOpenAI.sent["max_completion_tokens"] == 1224


def test_a_non_reasoning_model_is_billed_what_the_caller_asked_for(monkeypatch):
    _client(monkeypatch, "openai.gpt-4o").complete("s", "u", max_tokens=200)
    assert _CapturingOpenAI.sent["max_tokens"] == 200


def test_the_judge_built_the_way_the_product_builds_it_gets_the_reserve(monkeypatch):
    """`runtime.py` constructs `SemanticJudge(judge_client)` with no `max_tokens`. That default is
    200 -- the value measured at 14/15 fail-open scores. The end-to-end assertion is that the
    product's own construction now sends a budget that covers the reasoning."""
    client = _client(monkeypatch, "openai.gpt-5.5")
    SemanticJudge(client).score("q", "schema", "SELECT 1", "preview")
    assert _CapturingOpenAI.sent["max_completion_tokens"] == 1224

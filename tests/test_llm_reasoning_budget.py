"""A reasoning model's cap must cover the thinking, not just the reply.

The failure this prevents is silent in the worst way: the provider turns a truncated reasoning
pass into a request error, `SemanticJudge.score` catches every exception and returns its fail-open
1.0, and 1.0 is exactly what a judge that approved the answer returns. The verifier switches off
and every answer looks confidently verified.
"""
from mnemiq.llm.client import reasoning_budget, token_param_name


def test_reasoning_model_gets_the_floor_when_the_caller_asks_for_less():
    # The shipped `SemanticJudge` default. Measured at 10/10 provider failures on real prompts.
    assert reasoning_budget("openai.gpt-5.5", 200) == 1024
    assert reasoning_budget("openai.gpt-5-mini", 512) == 1024


def test_a_caller_asking_for_more_keeps_what_it_asked_for():
    # The generator asks 4000; a floor must never become a ceiling.
    assert reasoning_budget("openai.gpt-5.5", 4000) == 4000


def test_a_non_reasoning_model_is_untouched():
    # gpt-4o answered fine at 16 tokens; it spends nothing before emitting, so it needs no floor
    # and must not be charged a bigger cap.
    assert reasoning_budget("openai.gpt-4o", 200) == 200
    assert reasoning_budget("qwen2.5-coder-14b", 200) == 200


def test_the_floor_follows_the_same_family_test_as_the_token_parameter():
    """Both answer "is this a reasoning model?"; if they ever disagree, one of them is wrong about
    the model in front of it."""
    for model in ("openai.gpt-5.5", "openai.gpt-5-mini", "openai.gpt-4o", "llama-3.1-70b"):
        floored = reasoning_budget(model, 200) != 200
        assert floored == (token_param_name(model) == "max_completion_tokens"), model

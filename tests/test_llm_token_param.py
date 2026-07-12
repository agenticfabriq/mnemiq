import pytest

from mnemiq.llm.client import token_param_name


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai.gpt-5-mini", "max_completion_tokens"),
        ("openai.gpt-5", "max_completion_tokens"),
        ("gpt-5-mini", "max_completion_tokens"),
        ("openai.gpt-4o", "max_tokens"),
        ("gpt-4o", "max_tokens"),
    ],
)
def test_token_param_name(model, expected):
    assert token_param_name(model) == expected

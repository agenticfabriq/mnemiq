from __future__ import annotations

import re

from openai import APIError, OpenAI

from mnemiq.config import Settings

_GPT5 = re.compile(r"(^|\.)gpt-5")


class ModelUnavailable(RuntimeError):
    """The model provider did not answer -- outage, timeout, rate limit, bad gateway.

    Its own type so the agent can end in a stated failure instead of a traceback, and so
    the SDK's exception classes stop at this module. This is the model-side twin of the
    source refusing to serve us: something that happened TO us, never a decision we made.
    """


def token_param_name(model: str) -> str:
    # GPT-5 family spends reasoning tokens; it rejects `max_tokens`.
    return "max_completion_tokens" if _GPT5.search(model) else "max_tokens"


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise RuntimeError("LLM base_url/api_key not configured (set MNEMIQ_LLM_* env)")
        self._model = settings.llm_model
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        # A change that buys 1% accuracy for 3x the tokens is a trade to make on purpose.
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def complete(self, system: str, user: str, max_tokens: int = 512,
                 extra_body: dict | None = None) -> str:
        kwargs = {token_param_name(self._model): max_tokens}
        if extra_body:  # e.g. vLLM guided decoding (guided_json / guided_grammar)
            kwargs["extra_body"] = extra_body
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except APIError as exc:
            raise ModelUnavailable(str(exc)) from exc
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        return resp.choices[0].message.content or ""

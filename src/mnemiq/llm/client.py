from __future__ import annotations

import re

from openai import OpenAI

from mnemiq.config import Settings

_GPT5 = re.compile(r"(^|\.)gpt-5")


def token_param_name(model: str) -> str:
    # GPT-5 family spends reasoning tokens; it rejects `max_tokens`.
    return "max_completion_tokens" if _GPT5.search(model) else "max_tokens"


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise RuntimeError("LLM base_url/api_key not configured (set MNEMIQ_LLM_* env)")
        self._model = settings.llm_model
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        kwargs = {token_param_name(self._model): max_tokens}
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

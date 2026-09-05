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


# Reasoning room ADDED to every caller's request, not a threshold a few of them fall under.
_REASONING_RESERVE = 1024


def reasoning_budget(model: str, max_tokens: int) -> int:
    """Give a reasoning model room to think on top of the room the caller asked to write in.

    Callers size `max_tokens` for their OUTPUT -- 200 for a confidence object, 4000 for a query.
    A reasoning model spends the same budget on thinking FIRST, so the number means something
    different to it than to the caller who wrote it, and a cap sized for the answer truncates the
    reasoning before the answer starts. The provider reports that as a failed request, not a short
    reply. MEASURED (openai.gpt-5.5, 10 real `SemanticJudge` prompts): 200 -> 10/10 failed,
    400 -> 5/10, 600 -> 0/10.

    A reserve, not a floor. A floor is the wrong shape: it would lift the judge's 200 to something
    workable while leaving `synthesize`'s 1000 -- a caller that genuinely wants 1000 tokens of
    prose -- with whatever the floor left over, which is not reasoning room at all. Adding instead
    means every caller keeps what it asked for and the same allowance is granted for the same
    reason.

    Cheap, and MEASURED rather than assumed: over caps of 1224, 2048, 4024 and 8192 on one prompt,
    billed completion tokens were 89, 89, 105 and 118 -- a 6.7x allowance for ~33% more spend, so
    the bill tracks the work, not the cap. It is not free, so this is a reserve and not a blank
    cheque; it is far too cheap to justify truncating a request instead.
    """
    return max_tokens + _REASONING_RESERVE if _GPT5.search(model) else max_tokens


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise RuntimeError("LLM base_url/api_key not configured (set MNEMIQ_LLM_* env)")
        self._model = settings.llm_model
        self._seed = settings.llm_seed
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
        kwargs = {token_param_name(self._model): reasoning_budget(self._model, max_tokens)}
        if self._seed is not None:
            # Forwarded, not guaranteed. MEASURED: vLLM honours it; the hosted endpoint accepts
            # it, returns no system_fingerprint, and still varies its output -- two identical
            # seeded SQL requests produced different queries. OpenAI ties seed determinism to
            # that fingerprint, and this proxy does not participate. So this buys
            # reproducibility on the local path and nothing on the frontier one; do not build
            # an experiment design that assumes it.
            #
            # Safe with multi-candidate generation and the repair loop regardless, because
            # BOTH vary the prompt -- candidates by engineered strategy, retries by appended
            # feedback -- rather than relying on sampling noise. A future strategy that
            # resamples the same prompt would need to vary this per call.
            kwargs["seed"] = self._seed
        if extra_body:  # e.g. constrained decoding (response_format json_schema)
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

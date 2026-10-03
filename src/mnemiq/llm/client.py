from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
from openai import APIError, OpenAI

from mnemiq.config import Settings

_GPT5 = re.compile(r"(^|\.)gpt-5")
logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """The model provider did not answer -- outage, timeout, rate limit, bad gateway.

    Its own type so the agent can end in a stated failure instead of a traceback, and so
    the SDK's exception classes stop at this module. This is the model-side twin of the
    source refusing to serve us: something that happened TO us, never a decision we made.
    """


_FIXED_SAMPLING = re.compile(r"(^|[./])(gpt-5|o[134](-|$))")


def accepts_temperature(model: str) -> bool:
    """Reasoning models (GPT-5, o1/o3/o4, also behind a gateway prefix such as `openai/o3`) reject
    any temperature but their default; asked for 0, the provider fails the request. Matched by
    family, so a GPT-5 chat variant that would accept one is not sent it either: it keeps its
    default sampling, which fails safe rather than failing the request."""
    return not _FIXED_SAMPLING.search(model)


# Added to MNEMIQ_LLM_SEED for every call made inside `seed_offset(n)`. A context variable rather than
# a parameter, so it reaches the client through any generator, corrector or test double without
# their signatures knowing it exists, and holds per thread under the HTTP server.
_SEED_OFFSET: ContextVar[int] = ContextVar("mnemiq_seed_offset", default=0)


@contextmanager
def seed_offset(n: int) -> Iterator[None]:
    """Calls made inside send seed + n (M124): deep mode wraps each candidate in its index, so a
    candidate repeating an earlier one's prompt does not also repeat its sample."""
    token = _SEED_OFFSET.set(n)
    try:
        yield
    finally:
        _SEED_OFFSET.reset(token)


class ContextTooLong(ModelUnavailable):
    """The server refused the request as longer than its window -- vLLM refuses a prompt plus the
    reply budget past `--max-model-len` with a 400, and OpenAI-compatible services name it
    `context_length_exceeded`. A ModelUnavailable for every caller that already handles one, and its
    own type for a caller that can shorten the prompt: the server is up, a shorter request is served.
    """


def _context_too_long(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    return (getattr(exc, "code", None) == "context_length_exceeded"
            or "maximum context length" in str(exc).lower())


class PromptCut(ModelUnavailable):
    """The server read only part of the prompt and answered anyway (M119).

    Ollama, given a prompt longer than its context window, keeps the last part of it and returns
    no error; the instructions at the start are what is lost, and the answer reads as normal. The
    one trace is `usage.prompt_tokens`, which counts what the model READ, not what was sent. A
    ModelUnavailable because the model never saw the request it was sent -- but a configuration
    failure, not an outage: retrying sends the same prompt to the same window.
    """


# Two signs that a server may have cut the prompt, because servers cut in two ways -- and neither
# sign is proof, so each only triggers a check. Only when the check cannot decide does a ratio
# decide alone, and only past _CERTAIN_CHARS_PER_TOKEN.
#
# A big cut moves the ratio. MEASURED on 612 prompts from every call site (generator, corrector,
# judge, synthesis, deep mode's selector, and enrichment's annotation, facts and examples, over BIRD
# mini-dev, KaggleDBQA and a 1,500-column enterprise schema): 1.43 to 4.26 characters a token under
# Qwen2.5's tokenizer, 2.10 to 4.50 under o200k. A recent Ollama keeps about half its window, so a
# cut prompt at least doubles its ratio. But a ratio is not proof: result rows repeating long words
# compress hard, and a synthesis prompt of 50 rows of "International Business Machines Corporation"
# reads 6.19 under o200k with nothing cut (found in review).
#
# A modest cut does not move the ratio at all. MEASURED on Ollama 0.5.4 with a 2,048-token window:
# an 8,950-character prompt (about 3,300 tokens) came back as exactly 2,048 tokens read -- 4.4
# characters a token, inside the normal range. What gives it away is the count sitting on the
# window, and windows are set in multiples of 1,024.
#
# The check: re-send the prompt with ~64 tokens of padding. A server that read the whole prompt
# reads the padding too (MEASURED on Ollama 0.5.4: 3,299 -> 3,363, from its prefix cache); one
# keeping a fixed window reads exactly as many as before (2,048 -> 2,048). Growth between the two
# proves nothing about the original: a prompt that fit just under the window gives a probe that
# does not (found in review: 2,040 fits, its probe reads 2,048), so that is undecided, not a cut. It runs on about 1.7% of calls, plus the rare compressible
# prompt. Still missed: a modest cut to a window that is not a multiple of 1,024, and densely
# tokenized scripts (CJK), whose ratio stays low even when cut.
_RAISE = ("Raise the server's context length (Ollama: OLLAMA_CONTEXT_LENGTH or num_ctx; "
          "vLLM: --max-model-len) or use a model with a larger window.")
_MAX_CHARS_PER_TOKEN = 6.0
_WINDOW_STEP = 1024
_WINDOW_SLACK = 8  # a recent Ollama reported 16,386 for a 32,768 window: half, plus two
_PROBE_PADDING = "\n" + " padding" * 64
_MIN_GROWTH = 32  # of the padding's ~64 tokens; a server that read it all shows most of them
# When the probe cannot decide (it failed, carried no count, or itself reached the window), a ratio
# this high is refused anyway:
# twice the suspicion line, and far above anything measured (4.50) or constructed (6.19). A measure,
# not a guarantee: rows built of one long repeated token could in principle pass it.
_CERTAIN_CHARS_PER_TOKEN = 12.0


def _on_a_window_edge(read: int) -> bool:
    return read >= _WINDOW_STEP and min(read % _WINDOW_STEP, -read % _WINDOW_STEP) <= _WINDOW_SLACK


def token_param_name(model: str) -> str:
    # GPT-5 family spends reasoning tokens; it rejects `max_tokens`.
    return "max_completion_tokens" if _GPT5.search(model) else "max_tokens"


# Reasoning room ADDED to every caller's request, not a threshold a few of them fall under.
_REASONING_RESERVE = 4096


def reasoning_budget(model: str, max_tokens: int) -> int:
    """Give a reasoning model room to think on top of the room the caller asked to write in.

    Callers size `max_tokens` for their OUTPUT -- 200 for a confidence object, 4000 for a query.
    A reasoning model spends the same budget on thinking FIRST, so the number means something
    different to it than to the caller who wrote it, and a cap sized for the answer truncates the
    reasoning before the answer starts. The provider reports that as a failed request, not a short
    reply. MEASURED (openai.gpt-5.5, 10 real `SemanticJudge` prompts): 200 -> 10/10 failed,
    400 -> 5/10, 600 -> 0/10.

    Sized for the HARD TAIL, not the median, because the first version was sized for the median and
    the tail is what found it. At a reserve of 1024 exactly one case in 487 still failed every
    sweep -- a harder question, on which the model reasons longer. Pooled trials on that one case,
    by the budget actually sent: 1224 -> 6/7 failed, 2224 -> 1/9, 3248 and above -> 0/24. Note 2224
    gave 1/3 in one batch and 0/6 in another: reasoning length varies per call, so there is no
    budget above which this becomes impossible, only a probability that falls steeply. That is the
    argument for headroom over the smallest number that passed once, and the reason the product
    still needs its fail-open to be visible rather than merely rare.

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
        self._check_cut = settings.llm_prompt_cut_check
        # The SDK's own _DefaultHttpxClient sets follow_redirects=True, so a base_url that
        # `assert_local_only` approved at boot could still 302 a live request off-network on
        # every call after -- the boot check validates the CONFIGURED host once, not where a
        # response's Location header points. Passing our own httpx.Client relies on ITS
        # default, follow_redirects=False, to close that: openai.OpenAI(http_client=None) (the
        # unset case) is what builds the redirect-following one, so this must be an explicit
        # client, not a kwarg tweak on the default.
        http_client = httpx.Client(follow_redirects=False) if settings.local_only else None
        self._client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, http_client=http_client
        )
        # A change that buys 1% accuracy for 3x the tokens is a trade to make on purpose.
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def complete(self, system: str, user: str, max_tokens: int = 512,
                 extra_body: dict | None = None, temperature: float | None = None) -> str:
        kwargs = self._kwargs(max_tokens)
        if temperature is not None and accepts_temperature(self._model):
            # Unset, a local server samples at the model's own default -- Qwen2.5's is 0.7 -- and
            # a caller that needs the same verdict twice has to say so (M120).
            kwargs["temperature"] = temperature
        if extra_body:  # e.g. constrained decoding (response_format json_schema)
            kwargs["extra_body"] = extra_body
        resp = self._create(system, user, kwargs)
        read = self._count(resp)
        sent = len(system) + len(user)
        # Counted first: the server did the work, and the cost report must not lose it.
        reply = resp.choices[0].message.content or ""
        if not (self._check_cut and read
                and (sent / read > _MAX_CHARS_PER_TOKEN or _on_a_window_edge(read))):
            return reply
        try:
            again = self._count(self._create(system, user + _PROBE_PADDING, self._kwargs(1)))
            why = ("the probe carried no token count" if not again else
                   "the probe itself reached the window" if 0 < again - read < _MIN_GROWTH else "")
        except ModelUnavailable as exc:
            again, why = 0, f"the probe failed: {exc}"
        if why:
            # Undecided. Failing closed would turn a passing 429 into a failed answer on every
            # suspicious count, so this answers -- unless the ratio alone is past doubt -- and
            # says so, or a server whose probe never decides is invisible.
            if sent / read > _CERTAIN_CHARS_PER_TOKEN:
                raise PromptCut(
                    f"The model server read {read:,} tokens of a {sent:,}-character prompt, "
                    f"{sent / read:.0f} characters a token, far above any whole prompt measured; the "
                    f"check that would confirm it could not run ({why}). {_RAISE}")
            logger.warning("could not check for a cut prompt (%s tokens read of %s characters): %s",
                           read, sent, why)
            return reply
        if again <= read:
            raise PromptCut(
                f"The model server read {read:,} tokens of a {sent:,}-character prompt, and {again:,} "
                "when the prompt grew by about 64 tokens: it is keeping a fixed window and dropping "
                f"the rest of the prompt. {_RAISE}")
        return reply

    def _kwargs(self, max_tokens: int) -> dict:
        kwargs = {token_param_name(self._model): reasoning_budget(self._model, max_tokens)}
        if self._seed is not None:
            # Forwarded, not guaranteed. MEASURED: vLLM honours it; the hosted endpoint accepts
            # it, returns no system_fingerprint, and still varies its output -- two identical
            # seeded SQL requests produced different queries. OpenAI ties seed determinism to
            # that fingerprint, and this proxy does not participate. So this buys
            # reproducibility on the local path and nothing on the frontier one; do not build
            # an experiment design that assumes it.
            #
            # Offset per deep-mode candidate (`seed_offset`): its five candidates cycle three
            # strategies (STRATEGIES in agent/loop.py), so the fourth and fifth repeat a prompt,
            # and one shared seed made them repeat its answer too -- inflating the agreement the
            # confidence gate counts (M124). The repair loop needs no offset: its retries append
            # feedback, so the prompt itself changes.
            kwargs["seed"] = self._seed + _SEED_OFFSET.get()
        return kwargs

    def _create(self, system: str, user: str, kwargs: dict):
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
            if _context_too_long(exc):
                raise ContextTooLong(str(exc)) from exc
            raise ModelUnavailable(str(exc)) from exc
        return resp

    def _count(self, resp) -> int:
        """Add one response to the running totals; return the prompt tokens the server read."""
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is None:
            return 0
        read = getattr(usage, "prompt_tokens", 0) or 0
        self.prompt_tokens += read
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        return read

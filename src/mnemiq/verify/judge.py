from __future__ import annotations

import re
from dataclasses import dataclass

_SYSTEM = """You are a strict SQL reviewer. Given a question, the available schema, a SQL query
that already ran, and a preview of its result, judge the PROBABILITY (0.0-1.0) that the result
correctly and completely answers the question. Check: right tables/joins, every filter the
question implies, the right aggregation and grain. A plausible-looking number can still be wrong.
Reply ONLY with JSON, no prose: {"confidence": <0.0-1.0>}"""

_CONF = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')


@dataclass(frozen=True)
class JudgeRead:
    """A score, and whether it is a JUDGEMENT or the fail-open constant.

    The two are indistinguishable by value -- an unreachable judge returns 1.0 and so does an
    approval -- which is the whole of M89. Carrying the fact beside the number is the only way a
    caller can tell them apart without consulting shared state.
    """

    score: float
    fell_open: bool
    # WHY it fell open, because the retry policy differs by cause and a caller must not have to
    # consult counters to learn it. `unparsed` is deterministic for a model that cannot emit the
    # JSON, so retrying buys nothing; `error` is worth another attempt.
    reason: str = "ok"          # "ok" | "error" | "unparsed"


def _prompt(question: str, schema: str, sql: str, preview: str) -> str:
    return f"QUESTION: {question}\n\nSCHEMA:\n{schema}\n\nSQL: {sql}\n\nRESULT:\n{preview}"


class SemanticJudge:
    """One judge call scoring correctness confidence.

    Fail-open HERE, and only here: an unreadable reply or a dead endpoint returns 1.0 with
    `fell_open` set, so the float protocol the eval wrappers speak keeps working. What the
    product then DOES about it is the `Verifier`'s call and the deployment's -- it defers by
    default (issue #2). This class reports; it does not decide."""

    def __init__(self, client, max_tokens: int = 200) -> None:
        self._client = client
        self._max_tokens = max_tokens
        # Fail-open is the reporting default here and ruinous for a MEASUREMENT: a dead endpoint scores
        # every case 1.0, which is indistinguishable BY VALUE from a judge that approved everything
        # (`min(1.0, ...)` below clamps a real reply to the same number). These counters are a
        # running tally the eval harness reads at the end of a sweep, and nothing more: everything
        # in them is derivable from `read`, whose `fell_open` sums to the total and whose `reason`
        # tallies to the same errors/unparsed split. They are kept because a sweep wants the figure
        # without holding every result, not because they are the only source. DO NOT write that
        # they are. That claim has been made here repeatedly and has been false on arrival more
        # than once -- disproved by a field already sitting in `JudgeRead` a few lines up, while
        # this very comment was being edited. The trap is not the code outrunning the comment; it
        # is asserting uniqueness without reading what is beside it. `git log -p` on this file has
        # the instances, and needs no tally here to stay accurate.
        # They change no behaviour and cost an increment.
        #
        # They are NOT the per-call signal, and were briefly used as one: a caller diffing them
        # around its own `score` sees any concurrent caller's failure as its own, because they are
        # cumulative and this judge is shared across every mode and request. `read` returns that
        # fact with the score instead. Totals here, per-call there.
        self.calls = 0
        self.errors = 0        # the endpoint raised: outage, timeout, rate limit, bad gateway
        self.unparsed = 0      # it answered, but no confidence could be read out of the reply

    @property
    def fallbacks(self) -> int:
        """Scores that are the constant rather than a judgement -- errors plus unreadable replies."""
        return self.errors + self.unparsed

    def read(self, question: str, schema: str, sql: str, preview: str) -> "JudgeRead":
        """One judged answer, carrying whether it was judged at all.

        This exists because the counters above CANNOT answer that per call. They are cumulative and
        the judge is shared -- `runtime.py` builds one and hands it to every mode's verifier, and
        FastAPI runs the sync endpoint in a threadpool -- so a caller reading `fallbacks` before and
        after its own `score` sees any CONCURRENT request's failure as its own. Measured: two
        threads, one judged 0.95 and the other failing, and the judged one came back
        `judge_unavailable` with a real confidence of 0.95 beside it.

        The counters stay: an eval sweep is single-threaded and wants the totals. What they are not
        is a per-answer signal, and the fix is to return it rather than to lock -- locking would
        serialise every judge call in the product to make a bookkeeping read safe.
        """
        self.calls += 1
        try:
            raw = self._client.complete(_SYSTEM, _prompt(question, schema, sql, preview), max_tokens=self._max_tokens)
            m = _CONF.search(raw or "")
            if not m:
                self.unparsed += 1
                return JudgeRead(1.0, fell_open=True, reason="unparsed")
            return JudgeRead(max(0.0, min(1.0, float(m.group(1)))), fell_open=False)
        except Exception:
            self.errors += 1
            return JudgeRead(1.0, fell_open=True, reason="error")

    def score(self, question: str, schema: str, sql: str, preview: str) -> float:
        """The float protocol the eval wrappers speak. Behaviour is unchanged, including fail-open."""
        return self.read(question, schema, sql, preview).score


class FakeJudge:
    """Fixed score, zero tokens -- for tests and offline replay of a scripted operating point."""

    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, *args, **kwargs) -> float:
        return self._score

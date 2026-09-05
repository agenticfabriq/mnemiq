from __future__ import annotations

import re

_SYSTEM = """You are a strict SQL reviewer. Given a question, the available schema, a SQL query
that already ran, and a preview of its result, judge the PROBABILITY (0.0-1.0) that the result
correctly and completely answers the question. Check: right tables/joins, every filter the
question implies, the right aggregation and grain. A plausible-looking number can still be wrong.
Reply ONLY with JSON, no prose: {"confidence": <0.0-1.0>}"""

_CONF = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')


def _prompt(question: str, schema: str, sql: str, preview: str) -> str:
    return f"QUESTION: {question}\n\nSCHEMA:\n{schema}\n\nSQL: {sql}\n\nRESULT:\n{preview}"


class SemanticJudge:
    """One judge call scoring correctness confidence. Fail-open: an unreadable reply or a dead
    endpoint returns 1.0, so the verifier degrades to today's answer-anyway behavior rather than
    deferring everything."""

    def __init__(self, client, max_tokens: int = 200) -> None:
        self._client = client
        self._max_tokens = max_tokens
        # Fail-open is right for the product and ruinous for a MEASUREMENT: a dead endpoint scores
        # every case 1.0, which is indistinguishable BY VALUE from a judge that approved everything
        # (`min(1.0, ...)` below clamps a real reply to the same number). These counters are the
        # only way a replay can tell the two apart, so they exist for the eval harness, not for the
        # verifier -- they change no behaviour and cost an increment.
        self.calls = 0
        self.errors = 0        # the endpoint raised: outage, timeout, rate limit, bad gateway
        self.unparsed = 0      # it answered, but no confidence could be read out of the reply

    @property
    def fallbacks(self) -> int:
        """Scores that are the constant rather than a judgement -- errors plus unreadable replies."""
        return self.errors + self.unparsed

    def score(self, question: str, schema: str, sql: str, preview: str) -> float:
        self.calls += 1
        try:
            raw = self._client.complete(_SYSTEM, _prompt(question, schema, sql, preview), max_tokens=self._max_tokens)
            m = _CONF.search(raw or "")
            if not m:
                self.unparsed += 1
                return 1.0
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            self.errors += 1
            return 1.0


class FakeJudge:
    """Fixed score, zero tokens -- for tests and offline replay of a scripted operating point."""

    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, *args, **kwargs) -> float:
        return self._score

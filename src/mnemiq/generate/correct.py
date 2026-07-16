from __future__ import annotations

import re
from typing import Protocol

_SYSTEM = (
    "You fix ONE specific problem in a SQL query and change nothing else. "
    "Return only the corrected SQL -- no prose, no explanation, no code fences."
)

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_sql(raw: str) -> str:
    s = (raw or "").strip()
    match = _FENCE.search(s)
    if match:
        s = match.group(1).strip()
    return s


class SqlCorrector(Protocol):
    def correct(self, sql: str, problem: str) -> str: ...


class LLMCorrector:
    """A constrained, surgical edit: fix exactly the stated problem, change nothing else.

    Problem-agnostic and reusable: the logic-lint passes a detector message; value grounding
    (option J) will pass a value-grounding message. The corrector does not know or care which.
    """

    def __init__(self, client, max_tokens: int = 1000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def correct(self, sql: str, problem: str) -> str:
        user = f"SQL:\n{sql}\n\nProblem to fix (change only what this requires):\n{problem}"
        return _strip_sql(self._client.complete(_SYSTEM, user, max_tokens=self._max_tokens))


class FakeCorrector:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def correct(self, sql: str, problem: str) -> str:
        self.calls.append((sql, problem))
        return self._replies.pop(0) if self._replies else sql

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.semantic.retrieval import ContextPacket

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Constrained decoding (MNEMIQ_GUIDED_SQL=1): force the reply to be a JSON object with a NON-EMPTY
# sql string, so an over-deferring local model literally cannot return {"sql": null}. Requires a
# guided-decoding backend (vLLM guided_json). No effect against endpoints that ignore extra_body.
_GUIDED_SQL_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string", "minLength": 1}, "reason": {"type": "string"}},
    "required": ["sql"],
}


def _guided_extra_body(guided_sql: bool) -> dict | None:
    if guided_sql:
        return {"guided_json": _GUIDED_SQL_SCHEMA}
    return None


@dataclass
class SqlProposal:
    sql: str | None  # None is a deferral, not a failure
    reason: str = ""


def _parse(raw: str) -> SqlProposal:
    """Read the structured channel. A reply we cannot read is a deferral, never an exception."""
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return SqlProposal(sql=None, reason="the model did not answer on the JSON channel")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return SqlProposal(sql=None, reason="the model's JSON was malformed")
    if not isinstance(payload, dict):
        return SqlProposal(sql=None, reason="the model's JSON was not an object")

    sql = payload.get("sql")
    reason = payload.get("reason")
    return SqlProposal(
        sql=sql.strip() if isinstance(sql, str) and sql.strip() else None,
        reason=reason if isinstance(reason, str) else "",
    )


class Generator(Protocol):
    def propose(self, packet: ContextPacket, feedback: str | None = None) -> SqlProposal: ...


class LLMGenerator:
    def __init__(self, client, max_tokens: int = 4000, dialect: str = "duckdb",
                 guided_sql: bool = False, assertive: bool = False) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._dialect = dialect
        self._guided_sql = guided_sql
        self._assertive = assertive

    def propose(
        self, packet: ContextPacket, feedback: str | None = None, strategy: str | None = None
    ) -> SqlProposal:
        extra = _guided_extra_body(self._guided_sql)
        raw = self._client.complete(
            system_prompt(dialect=self._dialect, strategy=strategy, assertive=self._assertive),
            user_prompt(packet, feedback),
            max_tokens=self._max_tokens,
            **({"extra_body": extra} if extra else {}),
        )
        return _parse(raw)


class StrategyGenerator:
    """Binds one strategy onto a generator, behind the plain two-arg Generator protocol --
    plan_query never learns strategies exist."""

    def __init__(self, inner, strategy: str) -> None:
        self._inner = inner
        self._strategy = strategy

    def propose(self, packet: ContextPacket, feedback: str | None = None) -> SqlProposal:
        return self._inner.propose(packet, feedback, strategy=self._strategy)


class FakeGenerator:
    """Replays scripted model replies through the real parser. Zero tokens."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str | None] = []
        self.strategies: list[str | None] = []

    def propose(
        self, packet: ContextPacket, feedback: str | None = None, strategy: str | None = None
    ) -> SqlProposal:
        self.calls.append(feedback)
        self.strategies.append(strategy)
        if not self._replies:
            return SqlProposal(sql=None, reason="no more scripted replies")
        return _parse(self._replies.pop(0))

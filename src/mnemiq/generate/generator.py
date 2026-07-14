from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.semantic.retrieval import ContextPacket

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


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
    def __init__(self, client, max_tokens: int = 4000, dialect: str = "duckdb") -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._dialect = dialect

    def propose(self, packet: ContextPacket, feedback: str | None = None) -> SqlProposal:
        raw = self._client.complete(
            system_prompt(dialect=self._dialect),
            user_prompt(packet, feedback),
            max_tokens=self._max_tokens,
        )
        return _parse(raw)


class FakeGenerator:
    """Replays scripted model replies through the real parser. Zero tokens."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str | None] = []

    def propose(self, packet: ContextPacket, feedback: str | None = None) -> SqlProposal:
        self.calls.append(feedback)
        if not self._replies:
            return SqlProposal(sql=None, reason="no more scripted replies")
        return _parse(self._replies.pop(0))

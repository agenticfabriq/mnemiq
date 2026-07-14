from __future__ import annotations

from typing import Protocol

_PERSONA = (
    "You are a careful analyst. You answer strictly from the query results you are given, "
    "and you never invent a number."
)

_RULES = """Answer the question from the rows below, in one or two plain sentences.

- Use ONLY the values in the result. Never estimate, extrapolate, or add outside knowledge.
- If the result is empty, say plainly that no rows matched -- that is a real answer.
- If the rows are truncated, say that your answer covers only the rows shown.
- State the number. Do not describe the SQL."""

_FORCED = "\n\nYou are out of budget. Answer from the data you already have. Do not ask for more."


class Synthesizer(Protocol):
    def answer(self, question: str, sql: str, rendered: str, forced: bool = False) -> str: ...


class LLMSynthesizer:
    def __init__(self, client, max_tokens: int = 1000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def answer(self, question: str, sql: str, rendered: str, forced: bool = False) -> str:
        system = f"{_PERSONA}\n\n{_RULES}" + (_FORCED if forced else "")
        user = f"QUESTION: {question}\n\nSQL:\n{sql}\n\nRESULT:\n{rendered}"
        return self._client.complete(system, user, max_tokens=self._max_tokens).strip()


class FakeSynthesizer:
    def __init__(self, reply: str = "an answer") -> None:
        self._reply = reply
        self.calls: list[dict] = []

    def answer(self, question: str, sql: str, rendered: str, forced: bool = False) -> str:
        self.calls.append(
            {"question": question, "sql": sql, "rendered": rendered, "forced": forced}
        )
        return self._reply

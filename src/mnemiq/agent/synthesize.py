from __future__ import annotations

from typing import Protocol

# Above this the answer summarises instead of enumerating. Asking the model to judge
# "more than a few rows" for itself did not work -- it read nine rows back one by one --
# so the caller counts and the instruction is unconditional.
LIST_LIMIT = 5

_PERSONA = (
    "You are a careful analyst. You answer strictly from the query results you are given, "
    "and you never invent a number."
)

_RULES = """Answer the question from the rows below, in one or two plain sentences.

- Use ONLY the values in the result. Never estimate, extrapolate, or add outside knowledge.
- If the result is empty, say plainly that no rows matched -- that is a real answer.
- If the rows are truncated, say that your answer covers only the rows shown.
- State the number. Do not describe the SQL.
- A value ending in the marker `…` was shortened to fit this prompt. Do not repeat it as
  if it were the whole value, and do not treat it as evidence about the data."""

# Only sent when the result is bigger than LIST_LIMIT.
_SUMMARISE = """

The caller is already looking at this result as a table. Do NOT list its rows back, in
any form -- not as a sentence, not as a series of "name value" pairs. Instead give one or
two sentences saying what the result covers and what stands out in it: the largest, the
smallest, a pattern, an outlier. Naming one or two rows as examples is fine. Reproducing
the table in prose is not, and it is where a long answer starts inventing detail."""

_FORCED = "\n\nYou are out of budget. Answer from the data you already have. Do not ask for more."


class Synthesizer(Protocol):
    def answer(self, question: str, sql: str, rendered: str, forced: bool = False,
               row_count: int | None = None) -> str: ...


class LLMSynthesizer:
    def __init__(self, client, max_tokens: int = 1000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def answer(self, question: str, sql: str, rendered: str, forced: bool = False,
               row_count: int | None = None) -> str:
        system = f"{_PERSONA}\n\n{_RULES}"
        if row_count is not None and row_count > LIST_LIMIT:
            system += _SUMMARISE
        if forced:
            system += _FORCED
        user = f"QUESTION: {question}\n\nSQL:\n{sql}\n\nRESULT:\n{rendered}"
        return self._client.complete(system, user, max_tokens=self._max_tokens).strip()


class FakeSynthesizer:
    def __init__(self, reply: str = "an answer") -> None:
        self._reply = reply
        self.calls: list[dict] = []

    def answer(self, question: str, sql: str, rendered: str, forced: bool = False,
               row_count: int | None = None) -> str:
        self.calls.append(
            {"question": question, "sql": sql, "rendered": rendered, "forced": forced,
             "row_count": row_count}
        )
        return self._reply

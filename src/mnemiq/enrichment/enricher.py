from __future__ import annotations

from typing import Protocol

from mnemiq.enrichment.prompts import (
    ColumnFacts,
    render_table_facts,
    system_prompt,
    user_prompt,
)
from mnemiq.enrichment.proposals import TableAnnotation, parse_annotation


def _allowed(facts: list[ColumnFacts]) -> dict[str, set[str]]:
    return {c.name: set(c.codes) for c in facts}


class Enricher(Protocol):
    def annotate(self, table: str, facts: list[ColumnFacts], grounding: str = "") -> TableAnnotation: ...


class LLMEnricher:
    """The stochastic proposer. Its output is screened before it can reach the snapshot."""

    def __init__(self, client, max_tokens: int = 4000) -> None:
        # Generous budget: reasoning models spend tokens before emitting any content, and a
        # tight cap yields an empty string rather than an error.
        self._client = client
        self._max_tokens = max_tokens

    def annotate(self, table: str, facts: list[ColumnFacts], grounding: str = "") -> TableAnnotation:
        if not facts:
            return TableAnnotation(table=table)

        system = system_prompt()
        user = user_prompt(render_table_facts(table, facts), grounding)
        allowed = _allowed(facts)

        for _attempt in range(2):  # one bounded retry; models drop the channel occasionally
            raw = self._client.complete(system, user, max_tokens=self._max_tokens)
            annotation = parse_annotation(raw, table, allowed)
            if annotation.columns:
                return annotation
        return TableAnnotation(table=table)


class FakeEnricher:
    """Canned replies through the real validator: the screening path, for free."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self._replies = replies or {}
        self.calls: list[str] = []
        self.grounding_calls: dict[str, str] = {}

    def annotate(self, table: str, facts: list[ColumnFacts], grounding: str = "") -> TableAnnotation:
        self.calls.append(table)
        self.grounding_calls[table] = grounding
        return parse_annotation(self._replies.get(table, ""), table, _allowed(facts))

from __future__ import annotations

from typing import Protocol

from mnemiq.enrichment.prompts import (
    ColumnFacts,
    render_table_facts,
    sanitize,
    system_prompt,
    user_prompt,
)
from mnemiq.enrichment.proposals import TableAnnotation, diagnose_reply, parse_annotation

# Columns described per call (M112). One call for a whole table capped the reply at the budget: a
# 34-column table's reply closed cleanly, an 84-column table's stopped mid-JSON and the table got
# nothing. 30 keeps every chunk under the size that worked.
CHUNK_COLUMNS = 30
# The rest of a wide table's column names ride with each chunk, for context only; a table of
# thousands would otherwise put them all in every call.
_CONTEXT_NAMES = 200


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
        chunks = [facts[i:i + CHUNK_COLUMNS] for i in range(0, len(facts), CHUNK_COLUMNS)]
        columns, failures = [], []
        for start, chunk in zip(range(0, len(facts), CHUNK_COLUMNS), chunks, strict=True):
            others = facts[:start] + facts[start + len(chunk):] if len(chunks) > 1 else []
            described, why = self._annotate_chunk(table, chunk, others, grounding)
            columns += described
            if why:
                span = (f"columns {start + 1}-{start + len(chunk)} of {len(facts)}: "
                        if len(chunks) > 1 else "")
                failures.append(span + why)
        return TableAnnotation(table=table, columns=columns, failures=failures)

    def _annotate_chunk(self, table: str, chunk: list[ColumnFacts], others: list[ColumnFacts],
                        grounding: str) -> tuple[list, str]:
        """The columns described in one call, and why none were if none were."""
        facts_block = render_table_facts(table, chunk)
        if others:
            names = ", ".join(sanitize(f.name) for f in others[:_CONTEXT_NAMES])
            more = f" and {len(others) - _CONTEXT_NAMES} more" if len(others) > _CONTEXT_NAMES else ""
            facts_block += ("\n\nThe table's other columns, for context only -- document only the "
                            f"columns above: {names}{more}")
        system, user = system_prompt(), user_prompt(facts_block, grounding)
        allowed = _allowed(chunk)
        why = ""
        for _attempt in range(2):  # one bounded retry; models drop the channel occasionally
            raw = self._client.complete(system, user, max_tokens=self._max_tokens)
            annotation = parse_annotation(raw, table, allowed)
            if annotation.columns:
                return annotation.columns, ""
            why = diagnose_reply(raw, allowed)
        return [], why


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

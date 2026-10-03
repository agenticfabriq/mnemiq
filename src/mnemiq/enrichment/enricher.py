from __future__ import annotations

from typing import Protocol

from mnemiq.enrichment.prompts import (
    ColumnFacts,
    render_table_facts,
    sanitize,
    system_prompt,
    user_prompt,
)
from mnemiq.enrichment.proposals import (
    CUT_OFF,
    EMPTY,
    TableAnnotation,
    diagnose_reply,
    parse_annotation,
)

# Columns described per call (M112). One call for a whole table capped the reply at the budget: a
# 34-column table's reply closed cleanly, an 84-column table's stopped mid-JSON and the table got
# nothing. 30 starts every chunk under the size that worked; one that still overflows is halved.
CHUNK_COLUMNS = 30
# The rest of a wide table's column names ride with each chunk, for context only; a table of
# thousands would otherwise put them all in every call.
_CONTEXT_NAMES = 200
# A chunk whose reply did not fit is halved until one does or it is this small. Size failures
# split; every other failure -- no JSON, unparseable, the wrong columns: a dropped channel -- gets
# one retry at any level. So a 30-column chunk that never fits costs at most 16 calls (2 for the
# chunk, then 2 + 4 + 8 for its halves down to 3- and 4-column spans). An EMPTY reply splits only
# once: a reasoning model at its cap fits in half the columns, and a server that answers empty for
# some other reason costs 6 calls (the chunk's 2, then each half asked and retried) rather than 16.
_MIN_SPLIT = 5


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
        columns, failures = [], []
        for start in range(0, len(facts), CHUNK_COLUMNS):
            described, why = self._annotate_span(table, facts, start,
                                                 min(start + CHUNK_COLUMNS, len(facts)), grounding)
            columns += described
            failures += why
        return TableAnnotation(table=table, columns=columns, failures=failures)

    def _annotate_span(self, table: str, facts: list[ColumnFacts], start: int, end: int,
                       grounding: str, root: bool = True,
                       empty_split: bool = True) -> tuple[list, list[str]]:
        """Columns start..end described, and why any span yielded none. A span whose reply did not
        fit -- cut off, or empty as a reasoning model's is at the cap -- is split in half and tried
        again, down to _MIN_SPLIT columns: a column count does not bound a reply, since code-heavy
        columns each carry their meanings."""
        chunk = facts[start:end]
        others = facts[:start] + facts[end:]
        # The root gets its usual retry up front; a split span is asked once, and retried below
        # only for a failure that is not about size.
        described, why = self._annotate_chunk(table, chunk, others, grounding, 2 if root else 1)
        too_big = why.startswith(CUT_OFF) or (why == EMPTY and empty_split)
        if not described and not root and not too_big:
            described, why = self._annotate_chunk(table, chunk, others, grounding, 1)
            too_big = why.startswith(CUT_OFF) or (why == EMPTY and empty_split)
        if described:
            return described, []
        if too_big and len(chunk) > _MIN_SPLIT:
            middle = start + len(chunk) // 2
            keep = empty_split and why != EMPTY  # an empty reply splits once, no further
            left, left_why = self._annotate_span(table, facts, start, middle, grounding, False, keep)
            right, right_why = self._annotate_span(table, facts, middle, end, grounding, False, keep)
            return left + right, left_why + right_why
        span = f"columns {start + 1}-{end} of {len(facts)}: " if len(facts) > len(chunk) else ""
        return [], [span + why]

    def _annotate_chunk(self, table: str, chunk: list[ColumnFacts], others: list[ColumnFacts],
                        grounding: str, attempts: int = 2) -> tuple[list, str]:
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
        for _attempt in range(attempts):  # one retry by default; models drop the channel occasionally
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

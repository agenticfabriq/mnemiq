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
from mnemiq.llm.client import PromptCut

# Columns described per call (M112). One call for a whole table capped the reply at the budget: a
# 34-column table's reply closed cleanly, an 84-column table's stopped mid-JSON and the table got
# nothing. 30 starts every chunk under the size that worked; one that still overflows is halved.
CHUNK_COLUMNS = 30
# The rest of a wide table's column names ride with each chunk, for context only; a table of
# thousands would otherwise put them all in every call.
_CONTEXT_NAMES = 200
# A chunk whose reply did not fit is halved until one does or it is this small. Size failures
# split; every other failure -- no JSON, unparseable, the wrong columns: a dropped channel -- gets
# one retry at any level. So a 30-column chunk whose every reply is cut off costs 16 calls (2 for
# the chunk, then 2 + 4 + 8 for its halves down to 3- and 4-column spans); mixed failures, each
# retried before splitting, can cost up to 30. An EMPTY reply splits only once, a trade: a server
# that answers empty for some other reason costs 6 calls (the chunk's 2, then each half asked and
# retried) rather than 16, and a reasoning model still over its cap at 15 columns loses them --
# reported, not silent -- where halving further might have saved them.
_MIN_SPLIT = 5
# The server's window, not the reply budget: PromptCut says this chunk's prompt was longer than the
# server keeps. A size failure like a cut-off reply -- halving shortens the prompt -- and never a
# sign the endpoint is down, since the next chunk's prompt is a different length.
_TOO_LONG = "the prompt was longer than the server's window"


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
        columns, failures, down = [], [], ""
        for start in range(0, len(facts), CHUNK_COLUMNS):
            end = min(start + CHUNK_COLUMNS, len(facts))
            if down:  # the endpoint is failing: the rest of the table is not asked
                failures.append(_span(start, end, len(facts)) + f"not asked -- {down}")
                continue
            described, why, down = self._annotate_span(table, facts, start, end, grounding)
            columns += described
            failures += why
        return TableAnnotation(table=table, columns=columns, failures=failures)

    def _annotate_span(self, table: str, facts: list[ColumnFacts], start: int, end: int,
                       grounding: str, root: bool = True,
                       empty_split: bool = True) -> tuple[list, list[str], str]:
        """Columns start..end described, why any span yielded none, and -- when every attempt
        raised -- the endpoint's failure, so the caller stops asking. A span whose reply was cut
        off is split in half and tried again, down to _MIN_SPLIT columns -- a column count does
        not bound a reply, since code-heavy columns each carry their meanings; an empty one, as a
        reasoning model's is at its cap, splits one level only (see _MIN_SPLIT)."""
        chunk = facts[start:end]
        others = facts[:start] + facts[end:]
        # The root gets its usual retry up front; a split span is asked once, and retried below
        # only for a failure that is not about size.
        described, why, down = self._annotate_chunk(table, chunk, others, grounding,
                                                    2 if root else 1)
        too_big = _size_failure(why, empty_split)
        if not described and not root and not too_big:
            # A split span gets its retry before the endpoint counts as down: one passing fault
            # on a half must not abandon the table. Down only if BOTH asks went unanswered -- a
            # first reply, however garbled, means the endpoint is up.
            answered = not down
            described, why, down = self._annotate_chunk(table, chunk, others, grounding, 1)
            down = "" if answered else down
            too_big = _size_failure(why, empty_split)
        if described:
            return described, [], ""
        if too_big and not down and len(chunk) > _MIN_SPLIT:
            middle = start + len(chunk) // 2
            keep = empty_split and why != EMPTY  # an empty reply splits once, no further
            left, left_why, down = self._annotate_span(table, facts, start, middle, grounding,
                                                       False, keep)
            if down:
                return left, [*left_why, _span(middle, end, len(facts)) + f"not asked -- {down}"], down
            right, right_why, down = self._annotate_span(table, facts, middle, end, grounding,
                                                         False, keep)
            return left + right, left_why + right_why, down
        return [], [_span(start, end, len(facts), whole=len(chunk) == len(facts)) + why], down

    def _annotate_chunk(self, table: str, chunk: list[ColumnFacts], others: list[ColumnFacts],
                        grounding: str, attempts: int = 2) -> tuple[list, str, str]:
        """The columns described in one call, why none were if none were, and the endpoint's
        failure when no attempt got a reply at all."""
        facts_block = render_table_facts(table, chunk)
        if others:
            names = ", ".join(sanitize(f.name) for f in others[:_CONTEXT_NAMES])
            more = f" and {len(others) - _CONTEXT_NAMES} more" if len(others) > _CONTEXT_NAMES else ""
            facts_block += ("\n\nThe table's other columns, for context only -- document only the "
                            f"columns above: {names}{more}")
        system, user = system_prompt(), user_prompt(facts_block, grounding)
        allowed = _allowed(chunk)
        why, replied = "", False
        for _attempt in range(attempts):  # one retry by default; models drop the channel occasionally
            try:
                raw = self._client.complete(system, user, max_tokens=self._max_tokens)
            except Exception as exc:  # noqa: BLE001 -- keep the chunks already described
                # Caught here, not by the caller: a timeout on the second chunk used to escape
                # and discard the first chunk's columns with it. The type only -- a provider's
                # exception text can carry a host or a key, and the job outlives the run. A size
                # diagnosis from an earlier attempt outranks it: that is what splitting acts on.
                if isinstance(exc, PromptCut):
                    # The server read the prompt -- part of it -- so it is up; asking again cuts
                    # again, and halving is what shortens the prompt.
                    why, replied = _TOO_LONG, True
                    break
                if not _size_failure(why, True):
                    why = f"the call failed: {type(exc).__name__}"
                continue
            replied = True
            annotation = parse_annotation(raw, table, allowed)
            if annotation.columns:
                return annotation.columns, "", ""
            why = diagnose_reply(raw, allowed)
        return [], why, ("" if replied else why)


def _size_failure(why: str, empty_split: bool) -> bool:
    """A reply that did not fit its budget, or a prompt that did not fit the window: halving acts
    on both. An empty reply counts only while `empty_split` allows (see _MIN_SPLIT)."""
    return why.startswith(CUT_OFF) or why == _TOO_LONG or (why == EMPTY and empty_split)


def _span(start: int, end: int, total: int, whole: bool = False) -> str:
    return "" if whole else f"columns {start + 1}-{end} of {total}: "


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

"""What a follow-up question is allowed to remember.

Conversation state is carried as structure -- the prior query, the tables it read, and a
few rows of its result -- rather than as prose. The result is the part that matters: a
follow-up like "and how many claims does it have?" needs an antecedent for "it", and the
previous SQL does not contain one, because `ORDER BY total DESC LIMIT 1` computes the top
region without ever naming it.

Two bounds, both deliberate:

* **Authorization.** A turn is replayed only to the authorization boundary that produced
  it. Values from a privileged turn -- a name, a masked column -- must not steer a query
  asked by an identity that cannot see them, because the new answer would disclose
  through a filter exactly what the policy withheld from the column.
* **Size.** Only the last few turns, only a few rows each. History is an antecedent, not
  a second result set, and an unbounded transcript would crowd out the schema cards that
  the query actually has to be written against.
"""

from __future__ import annotations

from mnemiq.contract import HistoryTurn

# Enough for "it"/"that one"/"those" to resolve; beyond this a follow-up is really a new
# question, and the extra turns cost prompt budget the schema needs.
MAX_TURNS = 3
MAX_ROWS = 5
MAX_COLUMNS = 8


def scope_history(
    history: list[HistoryTurn] | None, grant_fingerprint: str
) -> list[HistoryTurn]:
    """The turns this identity may be reminded of, newest last, bounded.

    A turn whose fingerprint does not match is dropped entirely rather than stripped of
    its rows: its question and SQL were themselves shaped by data this identity may not
    see, so half of it is not safer than none of it.
    """
    if not history:
        return []

    kept = [t for t in history if t.grant_fingerprint == grant_fingerprint]
    return [_bound(turn) for turn in kept[-MAX_TURNS:]]


def _bound(turn: HistoryTurn) -> HistoryTurn:
    columns = list(turn.columns[:MAX_COLUMNS])
    width = len(columns)
    return turn.model_copy(
        update={
            "columns": columns,
            "rows": [list(row[:width]) for row in turn.rows[:MAX_ROWS]],
        }
    )

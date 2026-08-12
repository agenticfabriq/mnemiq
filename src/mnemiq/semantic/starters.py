"""Questions to offer on an empty screen, composed from the source in front of the asker.

M32: the workbench shipped three hardcoded questions about claims and policyholders, so a
deployment pointed at anything else opened with an invitation to ask about a database it was
not connected to.

They are composed **here** rather than in the workbench for the reason M4 settled: the card is
prose the engine writes, and a second reader of that format in the frontend would be free to
drift from the one that writes it. The engine holds typed `Column`s, so it needs no parsing.

They are also access-scoped, and that is not decoration. A starter names a table and a column,
so an unscoped one discloses exactly what the schema channel is not allowed to -- the M4 leak,
reopened at the friendliest point in the product.
"""

from __future__ import annotations

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.contract.semantic import Column
from mnemiq.sql.policy import AccessPolicy

# One coded value groups every row into a single bucket; hundreds is an identifier wearing a
# different name. Neither demonstrates anything.
MIN_CODES, MAX_CODES = 2, 12

# Codes harvested from a timestamp or an id are an artefact of low cardinality in the sample,
# not evidence of a dimension -- `last_update` is the recurring example.
NOT_A_DIMENSION = frozenset({"timestamp", "date", "identifier"})


def _readable(column: Column, grants: GrantSet, policy: AccessPolicy) -> bool:
    key = (column.object_id, column.name)
    return (
        grants.allows(column.object_id)
        and key not in policy.denied
        # A masked column reads as NULL, so offering it would demonstrate an empty answer.
        and key not in policy.masked
    )


def _rank(column: Column) -> tuple:
    """Biggest table first -- it is usually what the source is *about* -- then a stable
    tiebreak, so the same snapshot always opens with the same questions."""
    return (-(column.row_count or 0), column.object_id, column.name)


def _first(candidates: list[Column]) -> Column | None:
    return sorted(candidates, key=_rank)[0] if candidates else None


def compose_starters(
    snapshot: Snapshot, grants: GrantSet, policy: AccessPolicy, limit: int = 3
) -> list[str]:
    """Curated questions for this source if it has any, else questions built from its columns.

    Returns fewer than `limit`, or none at all, rather than padding. Three plausible questions
    about the wrong database is worse than an empty screen: the empty screen is merely quiet,
    and the wrong three are confidently misleading.

    One shape is deliberately missing. The hardcoded set carried a question the engine ought to
    decline, because a refusal demonstrates the product as well as an answer does -- but
    composing one generically means inventing a plausible-absent column, which risks a refusal
    for a reason that has nothing to do with governance. That starter comes from a curated
    example or not at all.
    """
    out: list[str] = []

    # Curated first: `mnemiq feedback` writes these, and a question a human wrote about this
    # source beats anything composed. Scoped by the tables it reads, not by its own name.
    for example in snapshot.examples:
        question = (example.question or "").strip()
        reads = example.tables or ([example.object_id] if example.object_id else [])
        if question and reads and all(grants.allows(t) for t in reads):
            out.append(question)

    readable = [c for c in snapshot.columns if _readable(c, grants, policy)]

    measure = _first([c for c in readable if c.semantic_type == "amount"])
    if measure is not None:
        out.append(f"What is the total {measure.name} in {measure.object_id}?")

    groupable = [
        c
        for c in readable
        if MIN_CODES <= len(c.coded_values) <= MAX_CODES
        and (c.semantic_type or "") not in NOT_A_DIMENSION
    ]
    # Two questions about one table show the source once. Prefer a second table when there is
    # one, and fall back rather than drop the starter when there is not.
    elsewhere = [c for c in groupable if measure is None or c.object_id != measure.object_id]
    dimension = _first(elsewhere) or _first(groupable)
    if dimension is not None:
        out.append(
            f"How many {dimension.object_id} records are there by {dimension.name}?"
        )

    seen: set[str] = set()
    unique = [q for q in out if not (q in seen or seen.add(q))]
    return unique[:limit]

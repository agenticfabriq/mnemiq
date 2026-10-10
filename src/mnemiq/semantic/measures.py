"""Certified metrics and dimensions, selected for a question's context.

These were loaded, validated and appended to the snapshot by `apply_certified`, and read by nothing:
`definitions` had eleven consumers across the tree and `relationships` seven; metrics and dimensions
had zero. On the fs payments corpus that is five of twenty-eight certified records reaching the
model — and they are the five carrying the most explicit meaning a schema cannot express.

**The selection rule is different from the glossary's, on purpose.** `select_definitions` matches
the asker's own words, because a definition answers *what does this term mean*. A metric answers
*how is this measured*, and it is needed most precisely when the asker did **not** name it — someone
asking for "revenue last quarter" is exactly who should be shown that revenue means settled volume.
So a metric rides with its table: if the table is in the retrieved context, its certified metrics
are too. That is the rule `_attach_facts` already applies to structural facts.

Visibility is the table's. A metric over an ungranted table is never offered, for the reason the
glossary gives: it would leak that the table exists, and steer the model into SQL the decider must
then reject.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Dimension, Metric
from mnemiq.semantic.expr_columns import columns_read


def select_metrics(
    table_ids: Sequence[str], metrics: Sequence[Metric], grants: GrantSet
) -> list[Metric]:
    """The certified metrics defined over the tables in context, that this identity may see."""
    in_context = set(table_ids)
    return [
        metric
        for metric in metrics
        if metric.measure.source in in_context and grants.allows(metric.measure.source)
    ]


def select_dimensions(
    table_ids: Sequence[str], dimensions: Sequence[Dimension], grants: GrantSet, *,
    columns: Sequence[Column],
) -> list[Dimension]:
    """The certified dimensions over the tables in context, that this identity may see.

    **A personal one only where the identity reads its level raw (M135)** -- `sql.policy`'s rule
    for a column: no level, `none`, or a level in `pii_clearance`. A masked level is not enough:
    offering a dimension invites grouping by it, which is a raw read. Exact level, as clearance
    is: cleared for `pii` is not cleared for `phi`. A level outside the vocabulary clears for no
    one here -- though a certified dimension arrives with one already read as `pii` by
    `apply_certified`, as a column's is.
    """
    in_context = set(table_ids)
    by_key = {(c.object_id.lower(), c.name.lower()): c for c in columns}
    return [
        dimension
        for dimension in dimensions
        if dimension.source in in_context
        and grants.allows(dimension.source)
        and _read_raw(dimension.pii_level, grants)
        and _reads_every_column_raw(dimension, by_key, grants)
    ]


def _reads_every_column_raw(dimension: Dimension, by_key: dict, grants: GrantSet) -> bool:
    """**M136.** Every column the dimension's expression reads, read raw by this identity.

    Its own level is not the whole of it: a column requires every level the personal dimensions
    over it carry (M135), so `region` (pii) and `region_health` (phi) over one column each need
    both -- and a plain dimension over a personal column needs that column's. Offering one the
    identity cannot read invites a grouping the decider refuses. Read with `columns_read`, the
    parser that stamped the levels, and keyed as `apply_certified` keys them. A reference naming
    no column here decides nothing: the dimension's own level still does.
    """
    table = dimension.source.lower()
    for ref in _columns_read(dimension.expr or "", dimension.source):
        for name in ref:
            column = by_key.get((table, name))
            if column is not None and not all(_read_raw(level, grants)
                                               for level in column.pii_levels()):
                return False
    return True


@lru_cache(maxsize=1024)
def _columns_read(expr: str, source: str) -> frozenset[frozenset[str]]:
    # Parsed per expression, not per question: this runs on every packet.
    return frozenset(columns_read(expr, source))


def _read_raw(level: str | None, grants: GrantSet) -> bool:
    return not level or level == "none" or level in grants.pii_clearance

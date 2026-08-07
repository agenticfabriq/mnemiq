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

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Dimension, Metric


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
    table_ids: Sequence[str], dimensions: Sequence[Dimension], grants: GrantSet
) -> list[Dimension]:
    """The certified dimensions over the tables in context, that this identity may see."""
    in_context = set(table_ids)
    return [
        dimension
        for dimension in dimensions
        if dimension.source in in_context and grants.allows(dimension.source)
    ]

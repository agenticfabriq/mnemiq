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
from mnemiq.semantic.expr_columns import columns_read, expression_scope, keyword_columns


def select_metrics(
    table_ids: Sequence[str], metrics: Sequence[Metric], grants: GrantSet, *,
    columns: Sequence[Column],
) -> list[Metric]:
    """The certified metrics defined over the tables in context, that this identity may see.

    **And whose expression reads only columns the identity reads raw (M144)** -- the rule a
    dimension already meets (M136). A table grant alone let `count(secret_code)` reach the prompt
    for an identity the card had withheld `secret_code` from: the decider would refuse the SQL,
    but the column's existence and expression were already in the prompt. Raw, not merely
    visible, because the decider refuses a masked column under an aggregate too.
    """
    in_context = set(table_ids)
    by_key = _by_key(columns)
    return [
        metric
        for metric in metrics
        if metric.measure.source in in_context and grants.allows(metric.measure.source)
        and _reads_raw(metric.measure.expr, metric.measure.source, by_key, grants)
    ]


def expression_reads_raw(expr: str, source: str, columns: Sequence[Column],
                         grants: GrantSet) -> bool:
    """Does `expr` over `source` read only columns this identity reads raw -- the metric and
    dimension rule, for other selectors that hold an expression (the facts block's measures)."""
    return _reads_raw(expr, source, _by_key(columns), grants)


def read_raw(column: Column, grants: GrantSet) -> bool:
    """Does this identity read `column` raw: every level it is read under cleared (M135)? The
    `build_access_policy` rule for a column, for selectors that hold columns and grants."""
    return all(_read_raw(level, grants) for level in column.pii_levels())


def _by_key(columns: Sequence[Column]) -> dict[tuple[str, str], list[Column]]:
    # Every column under each folded name: `customer."SSN"` and `customer.ssn` are two columns in
    # Postgres, and keeping one decided the clearance by their order (Codex on #95).
    by_key: dict[tuple[str, str], list[Column]] = {}
    for column in columns:
        by_key.setdefault((column.object_id.lower(), column.name.lower()), []).append(column)
    return by_key


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
    by_key = _by_key(columns)
    return [
        dimension
        for dimension in dimensions
        if dimension.source in in_context
        and grants.allows(dimension.source)
        and _read_raw(dimension.pii_level, grants)
        and _reads_raw(dimension.expression(), dimension.source, by_key, grants)
    ]


def _reads_raw(expr: str, source: str, by_key: dict, grants: GrantSet) -> bool:
    """**M136, and M144 for metrics.** Every column the expression reads, read raw by this identity.

    Its own level is not the whole of it: a column requires every level the personal dimensions
    over it carry (M135), so `region` (pii) and `region_health` (phi) over one column each need
    both -- and a plain dimension over a personal column needs that column's. Offering one the
    identity cannot read invites a grouping the decider refuses. Read with `columns_read`, the
    parser that stamped the levels, and keyed as `apply_certified` keys them. In an expression
    that parses and reads only its source, a reference naming no column here decides nothing: the
    dimension's own level still does. One that reads another relation, or parses nowhere and
    names nothing, is withheld from every identity, since what it reads cannot be established
    (Codex's review of #121).
    """
    table = source.lower()
    scope = _expression_scope(expr, source)
    if scope == "foreign":
        return False  # reads a relation `columns_read` cannot report on (Codex's review of #121)
    refs = _columns_read(expr, source)
    if scope == "unparsed" and not refs:
        return False  # parses nowhere and its spelling names nothing of the source: unchecked
    for ref in refs:
        known = [column for name in ref for column in by_key.get((table, name), ())]
        if scope == "unparsed" and not known:
            return False  # a spelling that names no column: malformed, so unchecked, so withheld
        if not all(read_raw(column, grants) for column in known):
            return False
    # Any token of the expression's text the engine reads as a column where one is so named
    # (`true`, `current_date`, a unit the parser made a string of): held to the raw read where
    # the table has the column, nothing otherwise.
    for name in keyword_columns(expr):
        if not all(read_raw(column, grants) for column in by_key.get((table, name), ())):
            return False
    return True


@lru_cache(maxsize=1024)
def _expression_scope(expr: str, source: str) -> str:
    return expression_scope(expr, source)


@lru_cache(maxsize=1024)
def _columns_read(expr: str, source: str) -> frozenset[frozenset[str]]:
    # Parsed per expression, not per question: this runs on every packet.
    return frozenset(columns_read(expr, source))


def _read_raw(level: str | None, grants: GrantSet) -> bool:
    return not level or level == "none" or level in grants.pii_clearance

from __future__ import annotations

from sqlglot import exp

from mnemiq.sql.authz_guard import local_cte_names
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode


def _candidate_tables(column: exp.Column, aliases: dict[str, str], referenced: set[str]) -> set[str]:
    """The base table(s) a column could belong to. Qualified -> its table (if a base table);
    unqualified -> any referenced base table (fail-closed: match against all)."""
    q = column.table
    if q:
        t = aliases.get(q)
        return {t} if t else set()  # a CTE/subquery alias -> not a base-table column
    return referenced


def _is_bare_projection(column: exp.Column) -> bool:
    """A masked column is safe only as a top-level SELECT item (optionally aliased)."""
    parent = column.parent
    if isinstance(parent, exp.Select) and column in parent.expressions:
        return True
    return (
        isinstance(parent, exp.Alias)
        and isinstance(parent.parent, exp.Select)
        and parent in parent.parent.expressions
    )


def check_cls(ast: exp.Expression, policy: AccessPolicy) -> Refusal | None:
    """Refuse a query that reads a denied column, or uses a masked column anywhere but a bare
    projection (masking a filtered/aggregated column would silently change the answer)."""
    if not policy.denied and not policy.masked:
        return None
    # resolve columns against every base table the query references (not just visible ones)
    local = local_cte_names(ast)
    tables = [t for t in ast.find_all(exp.Table) if t.name not in local]
    referenced = {t.name for t in tables}
    aliases = {t.alias_or_name: t.name for t in tables}

    for column in ast.find_all(exp.Column):
        cands = _candidate_tables(column, aliases, referenced)
        name = column.name
        if any((t, name) in policy.denied for t in cands):
            return Refusal(
                code=RefusalCode.UNAUTHORIZED_COLUMN,
                message=f"You may not read the column {name!r}.",
                subject=name,
            )
        if any((t, name) in policy.masked for t in cands) and not _is_bare_projection(column):
            return Refusal(
                code=RefusalCode.MASKED_COLUMN_IN_PREDICATE,
                message=(
                    f"{name!r} is masked for you; it may only be selected, not used in a "
                    "filter, join, grouping, ordering, or aggregate."
                ),
                subject=name,
            )
    return None

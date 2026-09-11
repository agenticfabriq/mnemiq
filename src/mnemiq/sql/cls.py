from __future__ import annotations

from sqlglot import exp

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables, column_tables
from mnemiq.sql.verdict import Refusal, RefusalCode


def _candidate_tables(
    column: exp.Column, resolved: dict[int, str] | None, referenced: set[str], aliased: set[str]
) -> set[str]:
    """The base table(s) a column could belong to.

    Qualified -> the table its alias names IN ITS OWN SCOPE, which is why this takes a
    per-node map rather than a name->table dictionary: one alias can mean two tables in one
    statement, and a flat map kept whichever was seen last (Codex review, 2026-08-12).

    Unqualified -> every referenced base table, fail-closed. An alias the resolver could not
    place is treated the same way rather than as "no table": unresolvable is not permission.
    """
    if not column.table or resolved is None:
        # `resolved is None` -> the scopes could not be read, so nothing here is known to be a
        # local alias. Every referenced base table is a candidate; an unreadable statement is
        # not an argument for permission.
        return referenced
    table = resolved.get(id(column))
    if table is not None:
        return {table}
    # A qualifier naming a CTE or derived table owns no base column -- but only when we are
    # sure that is what it is. Otherwise fall back to every table.
    return set() if column.table in aliased else referenced


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


def check_cls(ast: exp.Expression, policy: AccessPolicy,
              dialect: str | None = None) -> Refusal | None:
    """Refuse a query that reads a denied column, or uses a masked column anywhere but a bare
    projection (masking a filtered/aggregated column would silently change the answer)."""
    if not policy.denied and not policy.masked:
        return None
    # resolve columns against every base table the query references (not just visible ones)
    tables = base_tables(ast, dialect)
    referenced = {object_key(t) for t in tables}
    resolved = column_tables(ast, dialect)
    local = {cte.alias_or_name for cte in ast.find_all(exp.CTE)}
    aliased = local | {s.alias_or_name for s in ast.find_all(exp.Subquery) if s.alias_or_name}

    # A JOIN KEY is not an `exp.Column`, and it reads the column all the same. Measured with
    # `denied={("claim","ssn")}`: `ON c.ssn = p.ssn` was refused and both `USING (ssn)` and
    # `NATURAL JOIN person` were APPROVED -- a denied column used as a join key, and a masked one
    # filtered through one, where whether rows match leaks the value a row at a time. The same
    # hole `check_opaque_columns` was written around; found there and then looked for here.
    for join in ast.find_all(exp.Join):
        for key in join.args.get("using") or ():
            name = key.name if hasattr(key, "name") else str(key)
            for table in referenced:
                if policy.denies(table, name):
                    return Refusal(
                        code=RefusalCode.UNAUTHORIZED_COLUMN,
                        message=f"You may not read the column {name!r}.",
                        subject=name,
                    )
                if policy.masks(table, name):
                    return Refusal(
                        code=RefusalCode.MASKED_COLUMN_IN_PREDICATE,
                        message=(
                            f"{name!r} is masked for you; joining on it is a filter, so it may "
                            "only be selected."
                        ),
                        subject=name,
                    )
        if (join.args.get("method") or "").upper() == "NATURAL":
            # It names no column: the keys are whatever the tables share, so any denied or
            # masked column on a referenced table could be one. The repair is an explicit ON.
            for table in sorted(referenced):
                for owner, name in sorted(policy.denied | policy.masked):
                    if owner == table:
                        return Refusal(
                            code=RefusalCode.UNAUTHORIZED_COLUMN,
                            message=(
                                "A NATURAL join reads whatever columns these tables share, and "
                                f"{table!r} has one this identity may not read that way. Join "
                                "with an explicit ON or USING naming the columns."
                            ),
                            subject=table,
                        )

    for column in ast.find_all(exp.Column):
        cands = _candidate_tables(column, resolved, referenced, aliased)
        name = column.name
        if any(policy.denies(t, name) for t in cands):
            return Refusal(
                code=RefusalCode.UNAUTHORIZED_COLUMN,
                message=f"You may not read the column {name!r}.",
                subject=name,
            )
        if any(policy.masks(t, name) for t in cands) and not _is_bare_projection(column):
            return Refusal(
                code=RefusalCode.MASKED_COLUMN_IN_PREDICATE,
                message=(
                    f"{name!r} is masked for you; it may only be selected, not used in a "
                    "filter, join, grouping, ordering, or aggregate."
                ),
                subject=name,
            )
    return None

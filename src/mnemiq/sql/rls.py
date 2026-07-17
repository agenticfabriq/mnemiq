from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.authz_guard import local_cte_names
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode


def _validate_filter(filt: str, cols: set[str], dialect: str) -> exp.Expression | None:
    """Parse a row filter and confirm it references only columns of the table; else None."""
    try:
        expr = sqlglot.parse_one(filt, read=dialect)
    except Exception:
        return None
    for column in expr.find_all(exp.Column):
        if column.name not in cols:
            return None
    return expr


def _derived_table(
    table: str, alias: str, cols: set[str], masked_cols: set[str], filt: exp.Expression | None
) -> exp.Subquery:
    """(SELECT <cols, masked->NULL AS c> FROM table [WHERE filt]) AS alias."""
    projections: list[exp.Expression] = []
    for c in sorted(cols):
        if c in masked_cols:
            projections.append(exp.alias_(exp.null(), c))
        else:
            projections.append(exp.column(c))
    inner = exp.select(*projections).from_(exp.to_table(table))
    if filt is not None:
        inner = inner.where(filt)
    return exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias)))


def apply_row_and_mask(
    ast: exp.Expression, policy: AccessPolicy, visible: dict[str, set[str]], dialect: str = "duckdb"
) -> exp.Expression | Refusal:
    """Wrap each base table that has a row filter or a referenced masked column in a derived
    table that applies the filter and NULLs masked columns AT THE SOURCE. Returns the rewritten
    AST, or a Refusal for a policy-invalid filter."""
    if not policy.row_filters and not policy.masked:
        return ast

    local = local_cte_names(ast)
    masked_by_table: dict[str, set[str]] = {}
    for tbl, col in policy.masked:
        masked_by_table.setdefault(tbl, set()).add(col)

    referenced_masked: set[str] = set()
    for column in ast.find_all(exp.Column):
        for tbl, cols in masked_by_table.items():
            if column.name in cols:
                referenced_masked.add(tbl)

    for table_node in list(ast.find_all(exp.Table)):
        name = table_node.name
        if name in local or name not in visible:
            continue
        needs_filter = name in policy.row_filters
        needs_mask = name in referenced_masked
        if not (needs_filter or needs_mask):
            continue
        filt: exp.Expression | None = None
        if needs_filter:
            filt = _validate_filter(policy.row_filters[name], visible[name], dialect)
            if filt is None:
                return Refusal(
                    code=RefusalCode.INVALID_ROW_FILTER,
                    message=f"The row-access policy for {name!r} is not a valid predicate.",
                    subject=name,
                )
        derived = _derived_table(
            name, table_node.alias_or_name, visible[name], masked_by_table.get(name, set()), filt
        )
        table_node.replace(derived)
    return ast

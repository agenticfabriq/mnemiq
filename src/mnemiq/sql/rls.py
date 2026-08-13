from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables
from mnemiq.sql.verdict import Refusal, RefusalCode


def _validate_filter(
    filt: str,
    cols: set[str],
    dialect: str,
    policy_schema: dict[str, set[str]] | None = None,
) -> exp.Expression | None:
    """Parse a row filter and confirm it can only speak about what it is entitled to; else None.

    Two scopes, two rules. **Outside** a subquery the predicate is a WHERE on one table and may
    name only that table's own columns -- nothing else is in scope there. **Inside** a subquery
    it may name any table in `policy_schema`, the policy author's visibility.

    M28: the single rule was the outer one, applied everywhere, which made child-table tenancy
    structurally inexpressible -- `payment` has no `store_id`, so the only way to say "payments
    belonging to this store's customers" is to reach through `customer`, and that was refused.
    The shipped pagila policy did exactly that, so the two tables the coverage warning told the
    author to filter became unqueryable instead.

    The subquery is resolved against the POLICY's visibility and never the caller's, because
    the policy is what defines the caller's boundary and resolving it through that boundary is
    circular -- and because an entitlements table, the standard shape for this, is one no
    caller is ever granted. Without a `policy_schema` a subquery cannot be checked at all, so
    it is refused: an unvalidatable filter must not become a filter that silently does nothing.
    """
    try:
        expr = sqlglot.parse_one(filt, read=dialect)
    except Exception:
        return None
    if not isinstance(expr, exp.Condition):
        return None  # a statement, not a predicate -- `DELETE FROM t` is not a row filter

    nested = list(expr.find_all(exp.Select))
    inner: set[int] = {id(node) for select in nested for node in select.walk()}

    for column in expr.find_all(exp.Column):
        if id(column) not in inner and column.name not in cols:
            return None

    if not nested:
        return expr
    if not policy_schema:
        return None

    for select in nested:
        # `object_key`, not `.name`: dropping the qualifier accepted
        # `private.entitlement` for a key of `entitlement`, and rejected the only
        # spelling a federated key (`pg.entitlement`) can be written in.
        referenced = {object_key(table) for table in select.find_all(exp.Table)}
        if not referenced or not referenced <= set(policy_schema):
            return None
        # Unqualified inside a subquery -> resolve against every table it reads, fail-closed.
        # `cols` is included because a subquery may correlate back to the row being
        # filtered. Every node under a nested SELECT was treated as inner, so the EXISTS
        # spelling of a predicate the IN spelling already allowed was refused.
        known = cols | {c for t in referenced for c in policy_schema[t]}
        for column in select.find_all(exp.Column):
            if column.name not in known:
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
    ast: exp.Expression,
    policy: AccessPolicy,
    visible: dict[str, set[str]],
    dialect: str = "duckdb",
    only: set[int] | None = None,
) -> exp.Expression | Refusal:
    """Wrap each base table that has a row filter or a referenced masked column in a derived
    table that applies the filter and NULLs masked columns AT THE SOURCE. Returns the rewritten
    AST, or a Refusal for a policy-invalid filter.

    `only` restricts the rewrite to specific table NODES rather than table names. The decider
    runs this twice -- once over the objects the caller named, once over the bases that
    inlining a view introduced -- and a caller may reference one of those bases directly as
    well. Without node identity the second pass would wrap the first pass's output again,
    nesting a filter inside an identical copy of itself.
    """
    if not policy.row_filters and not policy.masked:
        return ast

    masked_by_table: dict[str, set[str]] = {}
    for tbl, col in policy.masked:
        masked_by_table.setdefault(tbl, set()).add(col)

    referenced_masked: set[str] = set()
    for column in ast.find_all(exp.Column):
        for tbl, cols in masked_by_table.items():
            if column.name in cols:
                referenced_masked.add(tbl)

    # Resolved before the loop mutates the tree: `replace` invalidates the scope it was read from.
    for table_node in base_tables(ast):
        if only is not None and id(table_node) not in only:
            continue
        name = object_key(table_node)
        if name not in visible:
            continue
        needs_filter = name in policy.row_filters
        needs_mask = name in referenced_masked
        if not (needs_filter or needs_mask):
            continue
        filt: exp.Expression | None = None
        if needs_filter:
            filt = _validate_filter(
                policy.row_filters[name], visible[name], dialect, policy.policy_schema
            )
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

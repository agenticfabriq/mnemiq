from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables, column_tables
from mnemiq.sql.verdict import Refusal, RefusalCode


# The stand-in for the table a filter is attached to, so the predicate can be validated as the
# query it becomes. Chosen to be unspellable in SQL a policy author would write.
_SUBJECT = "__mnemiq_filtered__"


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

    # Validate the predicate as the query it will become, so the ONE scope-aware resolver
    # answers "which table does this column belong to" here as well.
    #
    # This used to build its own alias map with a descendant-wide `find_all`, so a nested
    # `FROM other e` overwrote an outer `FROM entitlement e` and an invalid column was checked
    # against the wrong table. That is the third place a flat alias map has been a bypass --
    # `check_access`, `check_cls`, and here. There is now one resolver and no local maps.
    schema = dict(policy_schema or {})
    schema[_SUBJECT] = set(cols)
    try:
        wrapped = sqlglot.parse_one(f"SELECT 1 FROM {_SUBJECT} WHERE {filt}", read=dialect)
    except Exception:
        return None
    resolved = column_tables(wrapped)
    if resolved is None:
        return None  # scopes unreadable -> the filter cannot be validated, so it is refused

    for table in base_tables(wrapped):
        key = object_key(table)
        if key != _SUBJECT and key not in schema:
            return None  # a table the policy author has not got

    for column in wrapped.find_all(exp.Column):
        owner = resolved.get(id(column))
        if owner is None:
            if column.table:
                return None  # qualified by something no scope here defines
            # Unqualified: the subquery's own tables, or a correlated reference to the row
            # being filtered. Fail-closed across both rather than resolving ambiguity.
            reachable = {c for t in base_tables(wrapped) for c in schema.get(object_key(t), ())}
            if column.name not in reachable:
                return None
        elif column.name not in schema.get(owner, set()):
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
    injected: set[int] | None = None,
) -> exp.Expression | Refusal:
    """Wrap each base table that has a row filter or a referenced masked column in a derived
    table that applies the filter and NULLs masked columns AT THE SOURCE. Returns the rewritten
    AST, or a Refusal for a policy-invalid filter.

    `injected`, when given, is filled with the id of every node in the predicates this call
    INSERTS. Those nodes belong to the policy, not to the caller: nothing downstream may inline
    a view named inside one, and nothing may filter what such a view resolves to. Without that
    record, the caller's own filter was applied inside their policy's subquery -- the exact
    inversion of the rule the subquery exists to honour (M28).

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
        if injected is not None and filt is not None:
            # Read off the DERIVED tree, not off `filt`: sqlglot may copy a node into place,
            # and an id taken before insertion can belong to nothing.
            where = derived.this.args.get("where")
            if where is not None:
                injected.update(id(node) for node in where.walk())
        table_node.replace(derived)
    return ast

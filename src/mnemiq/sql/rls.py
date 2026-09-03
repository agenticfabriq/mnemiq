from __future__ import annotations

from dataclasses import dataclass

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
    if _SUBJECT in filt:
        # The stand-in is a real identifier at execution time and is exempt from the table
        # check below, so a filter naming it would read whatever table happens to carry that
        # name. Unspellable by convention is not unspellable.
        return None
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
        if any("." in part for part in (table.text("catalog"), table.text("db"), table.name)):
            return None  # `"pg.entitlement"` is not `pg.entitlement`, and a dotted key cannot
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
    # Folded on both sides: `cols` carries the snapshot's column spelling and `masked_cols` the
    # policy's, and a mask that matches the table but not the column name nulls nothing.
    folded_masked = {m.lower() for m in masked_cols}
    for c in sorted(cols):
        if c.lower() in folded_masked:
            projections.append(exp.alias_(exp.null(), c))
        else:
            projections.append(exp.column(c))
    inner = exp.select(*projections).from_(exp.to_table(table))
    if filt is not None:
        inner = inner.where(filt)
    return exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias)))


@dataclass(frozen=True)
class Narrowing:
    """What the access policy did to ONE object, so an answer can say it was narrowed.

    TABLE-GRANULAR on purpose. Column granularity would require the loop to know which projected
    column each masked source column reaches, and it does not -- a masked column can be consumed by
    an aggregate and never appear in the output at all. Claiming per-column detail the loop cannot
    support is how a disclosure becomes a lie, so this says "columns were masked on this table" and
    stops there.

    It carries NO predicate and NO policy identity. The caller is entitled to know its answer was
    narrowed; it is not entitled to the rule that narrowed it, which would leak the shape of other
    principals' access.
    """

    object: str          # the table, in the query's own spelling
    rows: bool           # a row filter was applied to it
    columns: bool        # at least one referenced column of it was masked


def apply_row_and_mask(
    ast: exp.Expression,
    policy: AccessPolicy,
    visible: dict[str, set[str]],
    dialect: str = "duckdb",
    exclude: exp.Expression | None = None,
) -> tuple[exp.Expression | Refusal, list[Narrowing]]:
    """Wrap each base table that has a row filter or a referenced masked column in a derived
    table that applies the filter and NULLs masked columns AT THE SOURCE. Returns the rewritten
    AST, or a Refusal for a policy-invalid filter.

    `exclude` skips ONE node, by identity. It exists for the target of an UPDATE/DELETE, which is
    the single position where a derived table is not SQL -- there is nothing to mutate through
    `UPDATE (SELECT ...) AS claim`. That target is filtered by conjunction instead, in
    `apply_row_filters_to_write`, which is equivalent there because the target's rows ARE the rows
    being mutated. By identity and never by name: `UPDATE claim ... WHERE id IN (SELECT id FROM
    claim ...)` is two nodes with one name that need opposite treatment, and matching on the name
    would skip both -- the flat-name-set bypass M31 and the view rounds each paid for once.
    """
    # The list rides with the AST rather than beside it, and the return type is a tuple so a
    # caller CANNOT quietly drop it: an ignored narrowing is exactly the silence this exists to
    # end, and it would be invisible at the call site if the signature still returned one value.
    narrowed: list[Narrowing] = []
    if not policy.row_filters and not policy.masked:
        return ast, narrowed

    # Folded on BOTH sides: the table keys come from the snapshot and `name` below comes from the
    # query, and comparing them exactly is what let `CLAIM` and `claim` name different objects.
    masked_by_table: dict[str, set[str]] = {}
    for tbl, col in policy.masked:
        masked_by_table.setdefault(tbl.lower(), set()).add(col.lower())

    referenced_masked: set[str] = set()
    for column in ast.find_all(exp.Column):
        for tbl, cols in masked_by_table.items():
            if column.name.lower() in cols:
                referenced_masked.add(tbl)

    # Resolved before the loop mutates the tree: `replace` invalidates the scope it was read from.
    for table_node in base_tables(ast):
        if exclude is not None and table_node is exclude:
            continue
        name = object_key(table_node)
        if name not in visible:
            continue
        needs_filter = policy.row_filter_for(name) is not None
        needs_mask = name.lower() in referenced_masked
        if not (needs_filter or needs_mask):
            continue
        filt: exp.Expression | None = None
        if needs_filter:
            filt = _validate_filter(
                policy.row_filter_for(name), visible[name], dialect, policy.policy_schema
            )
            if filt is None:
                return Refusal(
                    code=RefusalCode.INVALID_ROW_FILTER,
                    message=f"The row-access policy for {name!r} is not a valid predicate.",
                    subject=name,
                ), []
        derived = _derived_table(
            name, table_node.alias_or_name, visible[name],
            # `.lower()`: this dict is keyed folded and `name` is the QUERY's spelling.
            # Missing it left `needs_mask` correctly true while the column set handed to
            # the projection was EMPTY, so the masked column was emitted as its real
            # value instead of NULL -- the mask silently doing nothing.
            masked_by_table.get(name.lower(), set()), filt
        )
        table_node.replace(derived)
        narrowed.append(Narrowing(object=name, rows=needs_filter, columns=needs_mask))
    return ast, narrowed


def apply_row_filters_to_write(
    ast: exp.Expression,
    policy: AccessPolicy,
    visible: dict[str, set[str]],
    target: exp.Expression | None,
    dialect: str = "duckdb",
) -> tuple[exp.Expression | Refusal, list[Narrowing]]:
    """Govern a write with the read path's RLS. Returns the rewritten AST or a Refusal, and what
    the policy narrowed.

    M7 filed "two implementations with different semantics" and M30 measured what the second one
    permitted. The measurements, on `2af9443`: the write path bound a filter only to the table
    being *written*, so every table a write *read* was unfiltered -- `INSERT INTO scratch SELECT
    id, amount FROM claim` copied governed rows into an ungoverned table, where a later plain
    SELECT returns them forever. One approved write turned a filtered table into an unfiltered
    copy. And it parsed the filter with a bare `parse_one` rather than validating it, so a policy
    naming an arbitrary table was spliced in unchecked, a non-predicate became `WHERE id = 1 AND
    DELETE FROM claim`, and a typo raised `ParseError` out of the decider instead of refusing.

    So this is not a new implementation; it is the split that lets there be only one. Reads are
    wrapped by `apply_row_and_mask` unchanged. The UPDATE/DELETE target is conjoined, because it
    cannot be wrapped -- and it is conjoined with a predicate that went through the SAME
    `_validate_filter`, which is what closes the validation half rather than patching its five
    symptoms one at a time.

    An INSERT target is exempt and stays exempt: an INSERT does not read its target, and filtering
    rows on the way IN is not what a row filter means.
    """
    rewritten, narrowed = apply_row_and_mask(ast, policy, visible, dialect=dialect, exclude=target)
    if isinstance(rewritten, Refusal):
        return rewritten, []
    ast = rewritten

    if not isinstance(ast, (exp.Update, exp.Delete)) or not isinstance(target, exp.Table):
        return ast, narrowed
    name = object_key(target)
    filt = policy.row_filter_for(name)
    if filt is None:
        return ast, narrowed
    predicate = _validate_filter(filt, visible.get(name, set()), dialect, policy.policy_schema)
    if predicate is None:
        return Refusal(
            code=RefusalCode.INVALID_ROW_FILTER,
            message=f"The row-access policy for {name!r} is not a valid predicate.",
            subject=name,
        ), []
    existing = ast.args.get("where")
    combined = exp.and_(existing.this, predicate) if existing is not None else predicate
    ast.set("where", exp.Where(this=combined))
    # The target is EXCLUDED from the loop above -- it cannot be wrapped in a derived table -- so
    # its narrowing is recorded here or nowhere. An implementer who wires up the read path and
    # stops ships a governed DELETE narrowed from 47 rows to 3 that reports nothing: this feature's
    # own silence, one decider over.
    narrowed.append(Narrowing(object=name, rows=True, columns=False))
    return ast, narrowed

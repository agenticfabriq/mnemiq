from __future__ import annotations

from collections.abc import Callable

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from mnemiq.semantic.values import shown
from mnemiq.sql.qualify import names_one_object
from mnemiq.sql.verdict import Refusal, RefusalCode


def _column_and_string(a: exp.Expression, b: exp.Expression) -> tuple[exp.Column, str] | None:
    """If one side is a column and the other a string literal, return (column, literal_text)."""
    for column, literal in ((a, b), (b, a)):
        if isinstance(column, exp.Column) and isinstance(literal, exp.Literal) and literal.is_string:
            return column, literal.this
    return None


def _scopes(ast: exp.Expression) -> dict[int, Scope]:
    """Each column node's own scope, by identity. Empty where sqlglot cannot build the scopes,
    which leaves every column to the base-table resolution below, as before M141."""
    try:
        return {id(c): scope for scope in traverse_scope(ast) for c in scope.columns}
    except (SqlglotError, RecursionError):
        return {}


def _through_scopes(
    column: exp.Column, scope: Scope, cte: set[str], depth: int = 0
) -> tuple[str, str] | None:
    """The base table and column a qualified column reads, followed through each CTE or derived table that
    projects it (M141). Only a projection that IS a column is followed, renamed or not; a
    function, a literal or a union has no single base column, so None, and the check skips."""
    source = scope.sources.get(column.table) if column.table else None
    if isinstance(source, exp.Table):
        return None if source.name in cte else (source.name, column.name)
    if not isinstance(source, Scope) or depth > 32:
        return None
    if not isinstance(source.expression, exp.Select):
        return None
    for projection in source.expression.selects:
        if projection.alias_or_name == column.name:
            inner = projection.unalias()
            if isinstance(inner, exp.Column) and inner.table:
                return _through_scopes(inner, source, cte, depth + 1)
            if isinstance(inner, exp.Column) and len(source.sources) == 1:
                (alias,) = source.sources
                return _through_scopes(exp.column(inner.name, table=alias), source, cte, depth + 1)
            return None
    return None


def check_values(
    ast: exp.Expression,
    visible: dict[str, set[str]],
    values,
    row_filtered: set[str] | None = None,
    masks: Callable[[str, str], bool] | None = None,
) -> Refusal | None:
    """Flag an equality/IN filter whose string literal does not exist in its indexed column.

    Fires only where the column resolves unambiguously to an indexed base column -- a bounded,
    non-key, non-PII string column whose full value set we hold. Anything we cannot resolve or
    have not indexed is skipped: the check never guesses, so a 'not found' is genuine.

    `row_filtered` names the tables this identity sees through a row filter. **The value index is
    built once per source with no predicate** (`semantic/values.py`), and this check runs BEFORE the
    RLS rewrite -- so listing the real values of a row-filtered table let an identity enumerate rows
    it cannot read, without executing anything (M5). For those tables the refusal still fires and
    still names the wrong literal; it just stops quoting the data, and says why.

    `masks` says whether a base column is masked for this identity, and such a column is not
    checked at all. `check_cls` refuses one used in a predicate directly, but not through a CTE
    or a derived table, which this check follows (M141) -- so without this it listed a masked
    column's raw values. Skipped rather than refused without values: "does not appear" alone
    answers whether a raw value exists, which the mask exists to withhold.
    """
    row_filtered = row_filtered or set()
    cte = {c.alias_or_name for c in ast.find_all(exp.CTE)}
    alias_to_table: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        if table.name not in cte:
            alias_to_table[table.alias_or_name] = table.name
    referenced = set(alias_to_table.values())
    scopes = _scopes(ast)

    def resolve(column: exp.Column) -> tuple[str, str] | None:
        qualifier = column.table
        if qualifier:
            scope = scopes.get(id(column))
            if scope is not None and isinstance(scope.sources.get(qualifier), Scope):
                # A CTE or a derived table: followed to the column it projects, the column
                # renamed included -- `x.doc` may be `t.region`, and read by its name it would
                # be checked against `t.doc`'s values.
                return _through_scopes(column, scope, cte)
            table = alias_to_table.get(qualifier)
            return (table, column.name) if table else None
        owners = [t for t in referenced if column.name in visible.get(t, set())]
        return (owners[0], column.name) if len(owners) == 1 else None  # ambiguous -> skip

    def violation(column: exp.Column, literal: str) -> Refusal | None:
        resolved = resolve(column)
        if resolved is None:
            return None
        table, name = resolved
        if masks is not None and masks(table, name):
            return None
        if not values.has(table, name) or values.contains(table, name, literal):
            return None
        # FOLDED, and folding is safe here for the reason it is not safe for the filter
        # lookup itself: this set decides whether to WITHHOLD, so matching more of it discloses
        # less. Measured before this, with `row_filters` keyed `CLAIM` against a snapshot keyed
        # `claim` -- a pairing the policy fold now BINDS -- `check_values` answered "the real
        # values include: west, east", naming a value from rows the caller cannot read. The
        # exact-keyed control withheld it. So the fold that made the filter reach the query left
        # this branch still believing the table was unfiltered, and turned M5 back on.
        if names_one_object(table, row_filtered):
            return Refusal(
                code=RefusalCode.VALUE_GROUNDING,
                message=(
                    f"the value {literal!r} does not appear in {table}.{name}. "
                    "The valid values are not listed because your access to this table is "
                    "row-filtered and the value index is not."
                ),
            )
        near = values.nearest(table, name, literal)
        return Refusal(
            code=RefusalCode.VALUE_GROUNDING,
            message=(
                f"the value {literal!r} does not appear in {table}.{name}; "
                f"the real values include: {', '.join(shown(v) for v in near)}. Use the intended one."
            ),
        )

    for eq in ast.find_all(exp.EQ):
        pair = _column_and_string(eq.left, eq.right)
        if pair is not None:
            found = violation(*pair)
            if found is not None:
                return found

    for member in ast.find_all(exp.In):
        column = member.this
        if not isinstance(column, exp.Column):
            continue
        for item in member.expressions:
            if isinstance(item, exp.Literal) and item.is_string:
                found = violation(column, item.this)
                if found is not None:
                    return found

    return None

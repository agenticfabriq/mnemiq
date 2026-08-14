from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables
from mnemiq.sql.verdict import Refusal, RefusalCode

# A view nested deeper than this is either a cycle the stack missed or a schema nobody should
# be governing by reading definitions. Refusing beats walking forever.
MAX_DEPTH = 8


def _body(view: ViewDefinition) -> exp.Expression | None:
    """The SELECT a view stands for, whichever way its source spells it.

    Postgres hands back a bare SELECT; DuckDB and SQLite hand back the whole
    `CREATE VIEW x AS SELECT ...`. Normalising through the parser rather than by stripping the
    prefix keeps one SQL implementation -- `str.partition` would be a second, and a worse one.
    """
    try:
        parsed = sqlglot.parse_one(view.definition, read=view.dialect)
    except Exception:
        return None
    if isinstance(parsed, exp.Create):
        parsed = parsed.expression
    return parsed if isinstance(parsed, exp.Query) else None


def check_views(
    ast: exp.Expression, views: dict[str, ViewDefinition], filtered: set[str]
) -> Refusal | None:
    """Refuse a granted view that reads a row-filtered table, and one this engine cannot read.

    M27: a view's rows are defined by SQL held in the source, so a filter on its base table
    reaches nothing -- the decider refused the base table correctly, and the repair loop used
    that refusal as a signpost to the same rows through the view.

    **This is the floor, not the fix.** The fix is to inline the body so the filter lands at
    the leaves, and it was built, merged, and taken back out. Applying a policy *through* a
    view needs column-level lineage, and a lineage model that does not understand `SELECT *`,
    `UNION`, aggregates or transitive renames fails OPEN on every shape it misses -- four
    review rounds found bypasses in ordinary view definitions, the last of them in
    `SELECT * FROM base`.

    The floor asks a question with no unmodelled shapes: *which tables does this body mention*.
    `find_all` answers that for every spelling of SQL -- a star mentions its tables, a union
    mentions both sides' -- and a body that will not parse is refused outright. There is no
    shape that yields FEWER tables than the body reads, so every way this can be wrong points
    at refusing.

    Only row filters trigger it. Column dispositions need not, because enrichment classifies a
    view's own columns independently -- `customer_list.phone` carries its own `pii_level` -- so
    CLS already governs them at the view. Including them was measured on Pagila: all seven
    views refuse for all three roles, including one with no row filters at all.
    """
    if not filtered:
        return None
    return _walk(ast, views, filtered, (), 0)


def _walk(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    filtered: set[str],
    stack: tuple[str, ...],
    depth: int,
) -> Refusal | None:
    for node in base_tables(ast):
        name = object_key(node)
        view = views.get(name)
        if view is None:
            continue
        if name in stack:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=(
                    f"The view {name!r} is defined in terms of itself "
                    f"({' -> '.join((*stack, name))}), so it cannot be resolved."
                ),
                subject=name,
            )
        if depth >= MAX_DEPTH:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=f"The view {name!r} nests deeper than this engine will resolve.",
                subject=name,
            )
        body = _body(view)
        if body is None:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=(
                    f"{name!r} is a view whose definition this engine cannot read, so it "
                    "cannot confirm the row policy on the tables behind it."
                ),
                subject=name,
            )
        reached = sorted({object_key(t) for t in base_tables(body)} & filtered)
        if reached:
            return Refusal(
                code=RefusalCode.UNGOVERNED_VIEW,
                message=(
                    f"{name!r} reads {reached[0]!r}, which is row-filtered for you, and this "
                    "engine cannot apply that filter through a view. Query the table directly."
                ),
                subject=name,
            )
        nested = _walk(body, views, filtered, (*stack, name), depth + 1)
        if nested is not None:
            return nested
    return None

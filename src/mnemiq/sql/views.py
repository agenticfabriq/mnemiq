from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.qualify import object_key
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


def _unrecognised_source(body: exp.Expression) -> str | None:
    """The name of a source shape this engine does not model, or None if all are plain.

    A **whitelist**, and that is the whole design. The previous rule enumerated dangerous
    shapes -- a table-valued function, then laterals -- and each round of review found another
    it had not thought of, because an unlisted shape PASSED. Here an unlisted shape refuses:
    every source must be a named table or a subquery over one, and anything else (LATERAL,
    UNNEST, a function call, a table literal, whatever a future dialect adds) lands in the
    refuse branch without anyone having to notice it first.

    That inversion is the same one `writes_enabled` took in M3. Forgetting must fail closed.
    """
    # Found by node type, not by args key. The first version read `args["from"]`, which this
    # sqlglot spells `from_`, so it enumerated NOTHING -- and a whitelist that finds no sources
    # approves everything. It was failing open in exactly the way it exists to prevent, and only
    # the function tests caught it. Node types do not get renamed out from under a lookup.
    sources = [node.this for node in body.find_all(exp.From)]
    sources += [join.this for join in body.find_all(exp.Join)]
    # Two conditions, and the second is the one round seven needed. A `Subquery` is allowed on
    # the assumption its inner sources reappear as From/Join nodes -- but sqlglot represents a
    # parenthesised join as `Subquery(Table-with-joins)` and a pivot as `Subquery(Pivot)`, and
    # in both the ROOT source is in neither position. `FROM (query_table('customer') JOIN film)`
    # therefore enumerated only `film`, and the real table name lives inside a string literal
    # where nothing can read it. DuckDB executes both shapes.
    #
    # A table source is checked on `.this` being an Identifier rather than on having a name: a
    # function call parses as `Table(Anonymous)`, which is what an empty name was standing in
    # for, and the type is the fact while the empty name was a symptom of it.
    for source in sources:
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier):
            continue
        if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Query):
            continue  # a real subquery: its own FROM/JOIN nodes are enumerated by this walk
        return type(source.this if isinstance(source, exp.Subquery) else source).__name__
    return None


def _mentions(body: exp.Expression) -> set[str]:
    """Every table name the body mentions, in both spellings, from the RAW tree.

    Deliberately not `base_tables`. That resolves scopes to tell a CTE reference from a base
    table, which is right for authorization and wrong here: scope resolution can only ever
    return FEWER names, and it dropped the read inside a lateral. Over-inclusion costs a
    spurious refusal on a CTE that shares a filtered table's name; under-inclusion cost the
    control. Both spellings because a body may say `public.customer` where the snapshot's
    object-id is `customer`.
    """
    tables = list(body.find_all(exp.Table))
    names = {object_key(t) for t in tables} | {t.name for t in tables if t.name}
    # Folded, because unquoted identifiers are case-insensitive in all three engines and
    # `PUBLIC.CUSTOMER` was not matching a filtered `customer`. Folding here can only ADD
    # matches, and every added match is a refusal.
    return names | {n.lower() for n in names}


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
    for node in ast.find_all(exp.Table):
        name = object_key(node)
        view = views.get(name) or views.get(node.name)
        if view is not None and name not in views:
            name = node.name  # a body may qualify a view the snapshot keys bare
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
        unrecognised = _unrecognised_source(body)
        if unrecognised is not None:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=(
                    f"{name!r} reads through a {unrecognised.lower()} rather than a named "
                    "table, so this engine cannot tell which tables it touches."
                ),
                subject=name,
            )
        # The filtered set is widened the same way and by its bare last segment, so a body
        # saying `customer` still matches a filter keyed `public.customer`. Both directions,
        # because a miss here means not refusing.
        wanted = {f for f in filtered} | {f.lower() for f in filtered} \
            | {f.rsplit(".", 1)[-1] for f in filtered} \
            | {f.rsplit(".", 1)[-1].lower() for f in filtered}
        reached = sorted(_mentions(body) & wanted)
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

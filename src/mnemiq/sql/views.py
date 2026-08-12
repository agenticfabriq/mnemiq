from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables
from mnemiq.sql.verdict import Refusal, RefusalCode

# A view whose body reaches this deep is either a cycle the stack missed or a schema nobody
# should be governing by inlining. Refusing beats emitting SQL of unbounded size.
MAX_DEPTH = 8


@dataclass(frozen=True)
class Inlined:
    """The rewritten tree, and the base tables the rewrite brought into it.

    `introduced` is the point of the type. Once a view's body is inlined, the query reads
    tables the caller never named and that are absent from their `visible` map -- and those
    tables MUST still receive the caller's row filters and column masks, or the view is exactly
    the bypass M27 describes. Returning the set explicitly is deliberate: a row filter's own
    subquery (M28) is also absent from `visible` and must NOT be filtered, and telling the two
    apart by evaluation order alone is a security property resting on the order of two
    statements. Named, it rests on a name.
    """

    ast: exp.Expression
    introduced: dict[str, set[str]]


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


def inline_views(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    schema: dict[str, set[str]],
) -> Inlined | Refusal:
    """Replace every reference to a governed view with the SQL it stands for.

    M27: a view's rows are defined by SQL held in the source, so a filter on its base table
    reaches nothing -- the decider refused the base table correctly and the repair loop used
    that refusal as a signpost to the same data through the view. Wrapping the view's *output*
    cannot fix it either: `sales_by_film_category` aggregates the tenancy column away entirely,
    so there is nothing left to filter on. The body has to be inlined and the filter has to
    land at the leaves, before the GROUP BY.

    Inlining also dissolves two problems that looked separate. A renamed column needs no
    mapping -- `customer_list` exposes `store_id` as `sid`, and the filter goes on
    `customer.store_id` inside, where it still has its own name. And the base tables come from
    parsing the body, so the dependency data needs no catalog privileges: the obvious source,
    `information_schema.view_table_usage`, returns nothing at all to a non-owner.

    Fails closed on a body that will not parse, a view nested deeper than `MAX_DEPTH`, and a
    cycle -- which is the option this replaced, kept as this one's error branch rather than as
    an alternative to it.
    """
    introduced: dict[str, set[str]] = {}
    refusal = _expand(ast, views, schema, introduced, (), 0)
    if refusal is not None:
        return refusal
    return Inlined(ast=ast, introduced=introduced)


def _expand(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    schema: dict[str, set[str]],
    introduced: dict[str, set[str]],
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
                    f"{name!r} is a view whose definition this engine cannot read, so the "
                    "access policy on the tables behind it cannot be applied."
                ),
                subject=name,
            )
        nested = _expand(body, views, schema, introduced, (*stack, name), depth + 1)
        if nested is not None:
            return nested
        for inner in base_tables(body):
            inner_name = object_key(inner)
            if inner_name not in views and inner_name in schema:
                introduced.setdefault(inner_name, set()).update(schema[inner_name])
        node.replace(
            exp.Subquery(this=body, alias=exp.TableAlias(this=exp.to_identifier(node.alias_or_name)))
        )
    return None

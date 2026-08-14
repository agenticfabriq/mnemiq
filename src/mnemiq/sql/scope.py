from __future__ import annotations

from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

from mnemiq.sql.qualify import object_key


def base_tables(ast: exp.Expression) -> list[exp.Table]:
    """Every `exp.Table` node that reads a real object in the source.

    A CTE alias is reported as a Table by `find_all`, and must not be authorized as a source
    object -- authorizing it would be a category error, and refusing it would break every
    valid WITH clause. But a name is a CTE only *where that CTE is in scope*: a reference
    inside the CTE's own body reads the base table it was named after, and so does one in a
    sibling CTE declared before it.

    M31: three guards each asked only whether the name appeared in a flat set of CTE names --
    `check_access`, `check_cls` and `apply_row_and_mask` -- and so each skipped both halves of
    `WITH t AS (SELECT ... FROM t) SELECT ... FROM t`. Table authorization, column deny/mask
    and row filters fell together, from one blind spot shared three ways. This resolver is
    deliberately the single answer to "is this node a real read", because three copies of a
    scope rule is how they came to disagree in the first place.
    """
    try:
        root = build_scope(ast)
    except Exception:
        root = None
    if root is None:
        # Unresolvable scopes mean we cannot say which names are local, so every table node
        # is treated as a real read. That over-reports -- a CTE reference will not be in
        # `visible` and the query is refused -- which is the direction to fail in: the
        # alternative, falling back to the flat name set, is exactly the M31 bypass.
        return list(ast.find_all(exp.Table))

    out: list[exp.Table] = []
    for scope in root.traverse():
        for table in scope.tables:
            if isinstance(scope.sources.get(table.alias_or_name), exp.Table):
                out.append(table)
    return out


def column_tables(ast: exp.Expression) -> dict[int, str] | None:
    """`id(column node)` -> the base table its qualifier names **in that column's own scope**,
    or **None** when the scopes could not be resolved at all.

    None and `{}` are different answers and the difference is the finding. An empty map used to
    mean both "resolved, and no column resolves to a base table" and "the resolver failed", so
    a failure was read as the former: `cls` concluded a denied column belonged to no table and
    let it through. That is the same collapse as M2's outage-versus-empty-policy and M34's
    missing-versus-unreadable baseline -- an absence and a failure wearing one value.

    A missing entry within a returned map still means only "this qualifier names a CTE or a
    derived table".

    M31 made `base_tables` scope-aware and stopped there. Both consumers went on building ONE
    alias->table dictionary across every scope, so in

        SELECT q.secret FROM inner_t q UNION ALL SELECT q.secret FROM outer_t q

    the second `q` overwrote the first and a denied column was resolved against the permitted
    table. The resolver was right; the map that consumed it was still flat.
    """
    try:
        root = build_scope(ast)
    except Exception:
        root = None
    if root is None:
        return None

    out: dict[int, str] = {}
    for scope in root.traverse():
        local = {
            name: object_key(source)
            for name, source in scope.sources.items()
            if isinstance(source, exp.Table)
        }
        for column in getattr(scope, "columns", ()):
            table = local.get(column.table)
            if column.table and table is not None:
                out[id(column)] = table
    return out

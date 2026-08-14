from __future__ import annotations

from sqlglot import exp


def object_key(table: exp.Table) -> str:
    """The object-id form a table node maps to, and the ONE place that question is answered.

    A federated deployment shows the model 'catalog.table' (parsed as db=catalog); single-source
    shows a bare name (db unset). A bare name when db is unset keeps single-source byte-for-byte.

    Every qualifier the reference carries is included. Dropping the catalog made
    `other.public.entitlement` and `public.entitlement` the same key, so a policy filter could
    read a different catalog's table; snapshot object-ids never carry three parts, so a
    three-part reference now matches nothing and is refused.

A quoted identifier containing a literal dot is ambiguous against a dotted key --
    `"pg.entitlement"` reads the same as `pg.entitlement`. It is left ambiguous here rather
    than made unmatchable: a discovered table genuinely named `sales.2026` has that string as
    its object-id, and refusing it would make a real table unqueryable to spare a hypothetical
    one. Row-filter validation, where the collision was exploitable, refuses dotted
    identifiers itself.

    This was three functions (`object_key`, `_full_key`, and a bare `.name`) that disagreed,
    and every disagreement between them was a bypass.
    """
    return ".".join(p for p in (table.text("catalog"), table.text("db"), table.name) if p)


def expand_tables(ast: exp.Expression, registry: dict[str, str]) -> None:
    """Rewrite each 'catalog.table' node to the executable 'catalog.schema.table', in place.
    registry maps a catalog alias to its source schema. No-op when registry is empty (single
    source) -- so the single-source plan is unchanged."""
    if not registry:
        return
    for table in ast.find_all(exp.Table):
        db = table.text("db")
        if db in registry and not table.args.get("catalog"):
            table.set("catalog", exp.to_identifier(db))
            table.set("db", exp.to_identifier(registry[db]))

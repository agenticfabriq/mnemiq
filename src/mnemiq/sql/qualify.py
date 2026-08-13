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

    An identifier containing a literal dot cannot be spelled unambiguously in a dotted key --
    the single quoted identifier `"pg.entitlement"` would otherwise equal `pg.entitlement`, and
    they are different tables. Those get a key that cannot match any object-id, which refuses
    them: a pathological name is not worth a matching rule that can be gamed.

    This was three functions (`object_key`, `_full_key`, and a bare `.name`) that disagreed,
    and every disagreement between them was a bypass.
    """
    parts = [p for p in (table.text("catalog"), table.text("db"), table.name) if p]
    if any("." in p for p in parts):
        # Prefixed, not merely joined: `"\x00".join(["pg.entitlement"])` is the string back
        # again, so a SINGLE quoted identifier containing a dot would have sailed through the
        # guard meant to stop it.
        return "\x00" + "\x00".join(parts)
    return ".".join(parts)


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

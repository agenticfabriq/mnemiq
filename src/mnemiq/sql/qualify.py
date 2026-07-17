from __future__ import annotations

from sqlglot import exp


def object_key(table: exp.Table) -> str:
    """The object-id form a table node maps to. A federated deployment shows the model
    'catalog.table' (parsed as db=catalog); single-source shows a bare name (db unset).
    Returning the bare name when db is unset keeps single-source byte-for-byte."""
    db = table.text("db")
    return f"{db}.{table.name}" if db else table.name


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

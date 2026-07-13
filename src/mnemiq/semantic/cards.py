from __future__ import annotations

from dataclasses import dataclass

from mnemiq.contract import Column, Snapshot


@dataclass
class SchemaCard:
    object_id: str
    text: str


def build_cards(snapshot: Snapshot) -> list[SchemaCard]:
    """One self-sufficient card per table: what it is, what it holds, how it joins.

    This is the retrieval unit -- indexed, matched, and later read by the agent to write
    SQL -- so everything needed to use the table has to be on it.
    """
    columns: dict[str, list[Column]] = {}
    for column in snapshot.columns:
        columns.setdefault(column.object_id, []).append(column)

    joins: dict[str, list[str]] = {}
    for rel in snapshot.relationships:
        keys = ", ".join(f"{k.left} = {k.right}" for k in rel.join_keys)
        line = f"- joins {rel.to} ({rel.cardinality}{': ' + keys if keys else ''})"
        joins.setdefault(rel.from_, []).append(line)

    tables = [b.object_id for b in snapshot.source_bindings] or list(columns)

    cards: list[SchemaCard] = []
    for table in tables:
        lines = [f"TABLE {table}", "COLUMNS:"]
        for column in columns.get(table, []):
            parts = [f"- {column.name} ({column.data_type or 'unknown'}"]
            if column.semantic_type:
                parts.append(f", {column.semantic_type}")
            if column.pii_level and column.pii_level != "none":
                parts.append(f", {column.pii_level}")
            parts.append(")")

            line = "".join(parts)
            if column.description:
                line += f" {column.description}"
            if column.coded_values:
                codes = "; ".join(
                    f"{cv.code} = {cv.meaning}" if cv.meaning else cv.code
                    for cv in column.coded_values
                )
                line += f" Values: {codes}"
            lines.append(line)

        if table in joins:
            lines.append("RELATIONSHIPS:")
            lines.extend(joins[table])

        cards.append(SchemaCard(object_id=table, text="\n".join(lines)))
    return cards

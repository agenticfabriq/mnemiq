from __future__ import annotations

import re
from dataclasses import dataclass

from mnemiq.contract import Column, Snapshot
from mnemiq.sql.policy import AccessPolicy


_BARE_IDENT = re.compile(r"[a-z_][a-z0-9_]*")


@dataclass
class SchemaCard:
    object_id: str
    text: str


def render_facts_block(tf) -> str:
    """The grain/gotchas/measures/time block attached to a card AFTER retrieval. Kept OUT of
    the retrieval index (build_index embeds the lean card) so facts never perturb top-k."""
    lines: list[str] = []
    if tf.grain:
        lines.append(f"GRAIN: {tf.grain}")
    if tf.gotchas:
        lines.append("GOTCHAS:")
        lines.extend(f"- {g}" for g in tf.gotchas)
    if tf.canonical_measures:
        measures = "; ".join(f"{n} = {e}" for n, e in tf.canonical_measures.items())
        lines.append(f"MEASURES: {measures}")
    if tf.default_time_column:
        lines.append(f"DEFAULT TIME COLUMN: {tf.default_time_column}")
    return "\n".join(lines)


def build_cards(snapshot: Snapshot, policy: AccessPolicy | None = None,
                style: str = "cards") -> list[SchemaCard]:
    """One self-sufficient card per table: what it is, what it holds, how it joins.

    This is the retrieval unit -- indexed, matched, and later read by the agent to write
    SQL -- so everything needed to use the table has to be on it.

    `policy` scopes the card to one identity (M4). Without it the card is what it always was: the
    enrich-time, identity-independent unit that gets embedded and indexed. **A card served to a
    caller must be rendered WITH a policy** -- a denied column is omitted outright, and a masked one
    keeps its name and type while losing everything derived from the data (description, coded
    values, code scheme, the entirely-null note).

    Redaction lives here, in the renderer, rather than in a pass over rendered text: parsing this
    format back apart would be a second implementation of the card vocabulary, free to drift from
    the one that writes it. Registers M7 and D52 are what that costs.

    `style="ddl"` re-spells the finished card as `CREATE TABLE` for a generator that was
    fine-tuned on DDL. It is deliberately NOT the default and NOT what `build_index`
    embeds: the indexed card must stay the form top-k was measured on.
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
        ddl: list[tuple[str, str, str]] = []   # (name, data_type, trailing note)
        for column in columns.get(table, []):
            if policy is not None and (table, column.name) in policy.denied:
                continue  # not merely unreadable: the identity is not told it exists
            masked = policy is not None and (table, column.name) in policy.masked
            parts = [f"- {column.name} ({column.data_type or 'unknown'}"]
            if column.semantic_type:
                parts.append(f", {column.semantic_type}")
            if column.pii_level and column.pii_level != "none":
                parts.append(f", {column.pii_level}")
            parts.append(")")

            line = "".join(parts)
            if masked:
                # Nameable, not readable. Everything below this point is derived from the values.
                lines.append(line + " MASKED for this identity: do not select or filter on it.")
                ddl.append((column.name, column.data_type or "text",
                            "MASKED for this identity: do not select or filter on it."))
                continue
            if column.description:
                line += f" {column.description}"
            if (
                column.row_count is not None
                and column.row_count > 0
                and column.null_count == column.row_count
            ):
                # The Plan 08 eval's dominant failure: a confident description of an empty
                # column reads as an invitation to use it. Outrank the description.
                line += (
                    " ENTIRELY NULL in this source: every value is missing;"
                    " do not count, filter, join or aggregate on this column."
                )
            if column.coded_values:
                codes = "; ".join(
                    f"{cv.code} = {cv.meaning}" if cv.meaning else cv.code
                    for cv in column.coded_values
                )
                line += f" Values: {codes}"
            if column.code_scheme:
                line += f" Codes from {column.code_scheme.label}."
            lines.append(line)
            # The DDL form is built from the SAME fields in the SAME redaction branch, never by
            # parsing `line` back apart. A column name may itself contain parentheses --
            # `Enrollment (K-12)` is a real mini-dev column -- so any split of the rendered
            # prose lands in the wrong place, which is the drift the docstring above warns of.
            # Include the parenthesised annotations, not just the trailing prose. Slicing
            # past `"".join(parts)` dropped semantic_type and pii_level, so a column the
            # cards form labels `(text, identifier, high)` reached a ddl-style generator
            # with no PII marker at all.
            annotations = ", ".join(
                a for a in (column.semantic_type,
                            column.pii_level if column.pii_level not in (None, "none") else None)
                if a)
            prose = line[len("".join(parts)):].strip()
            ddl.append((column.name, column.data_type or "text",
                        "; ".join(x for x in (annotations, prose) if x)))

        if table in joins:
            lines.append("RELATIONSHIPS:")
            lines.extend(joins[table])

        text = ("\n".join(lines) if style != "ddl"
                else _render_ddl(table, ddl, joins.get(table, [])))
        cards.append(SchemaCard(object_id=table, text=text))
    return cards


def _render_ddl(table: str, cols: list[tuple[str, str, str]], rels: list[str]) -> str:
    """The same card, spelled as `CREATE TABLE`, from the structured columns.

    A text-to-SQL model fine-tuned on DDL is sensitive to the SHAPE of its schema input, not
    only its content: Arctic-Text2SQL-R1 scored 50.5% on mini-dev through its own
    DDL-and-examples prompt and 35.7% on identical content through the `TABLE/COLUMNS:` form,
    with retrieval, dialect and decoding each ruled out as the cause.

    Callers pass what the redaction loop already decided -- a denied column never arrives, a
    masked one arrives carrying its mask note -- so this re-spells and decides nothing.
    """
    if not cols:
        return f"CREATE TABLE {table} (\n);"
    body = "\n".join(
        f"    {name if _BARE_IDENT.fullmatch(name) else chr(34) + name + chr(34)} {dtype}"
        + ("," if i < len(cols) - 1 else "")
        + (f"  -- {note}" if note else "")
        for i, (name, dtype, note) in enumerate(cols))
    out = f"CREATE TABLE {table} (\n{body}\n);"
    if rels:
        out += "\n-- " + "\n-- ".join(r.lstrip("- ") for r in rels)
    return out

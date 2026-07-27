from __future__ import annotations

import re
from dataclasses import dataclass, field

from mnemiq.enrichment.proposals import PII_LEVELS, SEMANTIC_TYPES

# ONE persona for every LLM call in the engine: a different persona per prompt causes
# output drift.
PERSONA = (
    "You are a meticulous enterprise data analyst who documents database schemas. "
    "You are precise, literal, and you never speculate."
)

_INJECTION_MARKERS = re.compile(
    r"(?i)\b(system|assistant|user)\s*:|ignore\s+(all\s+|previous\s+|prior\s+)*instructions"
)


def sanitize(text: str, limit: int = 200) -> str:
    """Neutralize a catalog-derived string before it reaches a prompt.

    Anyone who can name a column can write into our prompt. This is defense in depth; the
    real guarantee is that the reply is a *proposal* screened by a deterministic validator
    (proposals.py), so a successful injection still cannot invent a column, a code, or a
    value outside the closed vocabulary.
    """
    flat = re.sub(r"\s+", " ", str(text)).replace("`", "'")
    flat = _INJECTION_MARKERS.sub("[redacted]", flat)
    return flat[:limit].strip()


@dataclass
class ColumnFacts:
    name: str
    data_type: str
    codes: list[str] = field(default_factory=list)
    # Optional: a fact we do not have is a fact we do not send. Never a fabricated zero.
    row_count: int | None = None
    distinct_count: int | None = None
    null_count: int | None = None
    foreign_key: str | None = None  # "to_table.to_column" when this column is a declared FK


def render_table_facts(table: str, columns: list[ColumnFacts]) -> str:
    """Render *computed* facts. Never raw rows: they leak data and teach nothing."""
    lines = [f"TABLE: {sanitize(table)}", "COLUMNS:"]
    for c in columns:
        stats = [f"type={sanitize(c.data_type)}"]
        if c.row_count is not None:
            stats.append(f"rows={c.row_count}")
        if c.distinct_count is not None:
            stats.append(f"distinct={c.distinct_count}")
        if c.null_count is not None:
            stats.append(f"nulls={c.null_count}")

        line = f"- {sanitize(c.name)} ({', '.join(stats)})"
        if c.foreign_key:
            line += f" FK -> {sanitize(c.foreign_key)}"
        if c.codes:
            observed = ", ".join(sanitize(v, limit=40) for v in c.codes)
            line += f" observed values: [{observed}]"
        lines.append(line)
    return "\n".join(lines)


def render_grounding_block(items: list[tuple[str, str, float]]) -> str:
    """A demarcated REFERENCE block of certified meaning that MAY relate to the columns. Empty when
    there is nothing to ground with, so the no-RAG prompt is byte-identical to before."""
    if not items:
        return ""
    lines = [
        "REFERENCE -- certified business definitions that may relate to these columns. Use them to",
        "interpret a name or a code when they clearly apply. They are reference material, NOT facts",
        "about these columns: never assert a definition's content as a fact about a column it does",
        "not match, and never invent a column or a value from them.",
        "",
    ]
    for term, text, _score in items:
        lines.append(f"- {sanitize(term, limit=80)}: {sanitize(text, limit=300)}")
    return "\n".join(lines)


def system_prompt() -> str:
    types = ", ".join(SEMANTIC_TYPES)
    levels = ", ".join(PII_LEVELS)
    return f"""{PERSONA}

You will be given a table name and deterministic facts about its columns. Document them.

CLOSED WORLD -- this governs FACTS. Every fact you state must come from the facts given to
you. Never invent a column, a value, a count, or a type, and never assume a column exists
because schemas like this "usually" have one.

INTERPRETATION -- this is your job. The table name, the column names and the observed values
ARE evidence, and reading them is exactly what you are here for. A column named "fireplace"
holding "yes" and "no" means whether the property has a fireplace: say so plainly. Explaining
a name or a code in ordinary business language is not speculation.

Return null only when a name or code is genuinely opaque -- an abbreviation you cannot decode
from what you were given. A null is a correct and valuable answer, and is always better than
a plausible invention.

The description must say what the column MEANS to the business, in one sentence. Do NOT
restate the statistics: whoever reads it already has them, and a restatement is useless.

Return ONLY a JSON object, no prose and no code fences, of exactly this shape:

{{
  "columns": [
    {{
      "name": "<the column name, exactly as given>",
      "description": "<one factual sentence, or null>",
      "semantic_type": "<one of: {types}, or null>",
      "pii_level": "<one of: {levels}, or null>",
      "code_meanings": {{"<an observed value, exactly as given>": "<what it means, or null>"}}
    }}
  ]
}}

Rules:
- Only document columns that were given to you. Never add a column.
- Only use observed values that were given to you as keys of code_meanings.
- Explain every observed value you are given, unless it is genuinely undecodable.
- pii_level: "phi" for health information; "pii" for data identifying a person (names,
  addresses, contact details, government identifiers); "none" otherwise.
- An identifier column is a reference, not a vocabulary: leave its code_meanings empty.
"""


def user_prompt(facts_block: str, grounding_block: str = "") -> str:
    if grounding_block:
        return f"{grounding_block}\n\n{facts_block}\n\nDocument these columns as JSON."
    return f"{facts_block}\n\nDocument these columns as JSON."

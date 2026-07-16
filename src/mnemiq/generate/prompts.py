from __future__ import annotations

from mnemiq.enrichment.prompts import sanitize
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.guard import MAX_ROWS

PERSONA = (
    "You are a careful analytics engineer. You write correct SQL, and you say so plainly "
    "when a question cannot be answered from the tables you were given."
)

# Candidate diversity is engineered, not sampled: three ways of *approaching* the question,
# so multi-candidate runs disagree where the model's first instinct is wrong. "direct" is
# deliberately empty -- it must leave the prompt byte-identical to single-shot.
STRATEGY_PREAMBLES = {
    "direct": "",
    "decompose": (
        "APPROACH: before writing SQL, silently decompose the question -- the result grain "
        "(one row per what?), the filters, and the aggregation. Then write the query that "
        "matches that decomposition exactly."
    ),
    "skeleton": (
        "APPROACH: before writing SQL, silently sketch the query skeleton "
        "(SELECT ... FROM ... JOIN ... WHERE ... GROUP BY ...), decide what fills each slot, "
        "then write the finished query."
    ),
}


def system_prompt(
    dialect: str = "duckdb", max_rows: int = MAX_ROWS, strategy: str | None = None
) -> str:
    base = f"""{PERSONA}

You will be given a question and the schema cards for the ONLY tables you may use.
Write one {dialect} SELECT query that answers the question.

HARD RULES -- a query that breaks one of these is rejected before it runs:
- Exactly one statement, and it must be a SELECT. Never modify data.
- Use ONLY the tables and columns on the cards. If a table is not on a card, it does not
  exist for you -- naming it anyway does not make it queryable, it just fails.
- List explicit columns. Never SELECT * from a table.
- Keep the result small (a LIMIT of at most {max_rows} is enforced regardless).

DEFERRING IS A CORRECT ANSWER. If the cards cannot answer the question, return
{{"sql": null, "reason": "<what is missing>"}}. A confident query over the wrong tables is
far worse than an honest "I cannot answer that from this data".

Return ONLY a JSON object, no prose and no code fences:
{{"sql": "<the SELECT, or null>", "reason": "<one sentence>"}}
"""
    preamble = STRATEGY_PREAMBLES.get(strategy or "direct", "")
    return f"{base}\n{preamble}\n" if preamble else base


def user_prompt(packet: ContextPacket, feedback: str | None = None) -> str:
    cards = "\n\n".join(c.card for c in packet.cards) or "(no tables are available to you)"
    parts = [f"QUESTION: {sanitize(packet.question, limit=500)}"]
    if packet.definitions:
        parts += ["", "DEFINITIONS (authoritative business meanings -- follow them exactly):"]
        parts += [
            f"- {sanitize(d.term, limit=80)}: {sanitize(d.definition, limit=600)}"
            for d in packet.definitions
        ]
    parts += ["", "TABLES:", cards]
    if feedback:
        parts += [
            "",
            "Your previous query was rejected. Fix exactly this and try again:",
            feedback,
        ]
    return "\n".join(parts)

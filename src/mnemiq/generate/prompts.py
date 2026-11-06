from __future__ import annotations

from mnemiq.enrichment.prompts import sanitize
from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.guard import MAX_ROWS

# Default defer-don't-guess block (byte-for-byte with the original prompt).
_DEFER_DEFAULT = (
    'DEFERRING IS A CORRECT ANSWER. If the cards cannot answer the question, return\n'
    '{"sql": null, "reason": "<what is missing>"}. A confident query over the wrong tables is\n'
    'far worse than an honest "I cannot answer that from this data".'
)
# Assertive variant (MNEMIQ_ASSERTIVE_SQL=1): for weaker/local models that over-defer -- attempt
# when the tables are present; defer ONLY when a required table is genuinely absent.
_DEFER_ASSERTIVE = (
    "ATTEMPT EVERY QUESTION whose tables are on the cards. If the tables you need ARE present,\n"
    'you MUST write your best {dialect} SELECT -- even if the question is complex or you are\n'
    "unsure; a reasonable attempt is expected and far better than giving up.\n"
    'Return {{"sql": null, "reason": "<the missing table>"}} ONLY when a table you genuinely need\n'
    "is absent from the cards. Do NOT return null merely because the question is hard."
)

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


# M35's half of the reply, shipped only when the guard that READS it is on. Leaving it in
# unconditionally would have meant "off" was not the pre-M35 request: the model still spends
# tokens deciding which terms it had to assume, and the 66.7% baseline the withdrawal compares
# against was measured on a prompt that never asked.
_DECLARE_KEY = ''', "assumed_terms": ["<term>", ...]'''

_DECLARE = """
`assumed_terms`: business terms in the QUESTION whose meaning you had to assume because no
definition above gave it -- terms computed by a convention rather than read from a column, like
"lifetime value" or "churn rate". List the term as the question spells it. Leave it `[]` when every
term you relied on was defined above, or when the question names none. Do NOT list table or column
names: a missing column is already caught elsewhere, and listing one here would refuse a question
that is merely misspelled.
"""


# `writer_reasoning = "before_sql"`: room to reason before answering, which a JSON-only reply leaves none of. The reply
# still ends on the JSON channel, read as its last object, so declining and the M35 declaration work as before.
_REPLY_AFTER_REASONING = (
    "First think the question through in plain text, step by step: which tables and columns it needs, how they\n"
    "join, which rows it keeps, and what one row of the result stands for. Then end your reply with this JSON\n"
    "object on a line of its own, with nothing after it:"
)
_REPLY_ONLY = "Return ONLY a JSON object, no prose and no code fences:"


def writer_style(settings) -> dict:
    """What in the settings changes the writer's system prompt, as `system_prompt`'s keywords.

    One reading for every place that renders it: the writer, the prompt fitter, the window check and the profile's
    version. A setting read in one place and not another would size, warn about or hash a prompt the writer never sends.
    """
    return {"assertive": settings.assertive_sql, "declare_assumed_terms": settings.guard_undefined_terms,
            "reasoning": settings.writer_reasoning, "rules": settings.writer_rules}


def system_prompt(
    dialect: str = "duckdb", max_rows: int = MAX_ROWS, strategy: str | None = None,
    assertive: bool = False, declare_assumed_terms: bool = False,
    reasoning: str = "none", rules: str = "full",
) -> str:
    defer = (
        _DEFER_ASSERTIVE.format(dialect=dialect)
        if assertive
        else _DEFER_DEFAULT
    )
    # M35's declaration ships only when the guard that reads it is on -- see `_DECLARE_KEY`.
    declare_key = _DECLARE_KEY if declare_assumed_terms else ""
    declare = _DECLARE if declare_assumed_terms else ""
    task = f"""You will be given a question and the schema cards for the ONLY tables you may use.
Write one {dialect} SELECT query that answers the question."""
    rule_text = f"""HARD RULES -- a query that breaks one of these is rejected before it runs:
- Exactly one statement, and it must be a SELECT. Never modify data.
- Use ONLY the tables and columns on the cards. If a table is not on a card, it does not
  exist for you -- naming it anyway does not make it queryable, it just fails.
- List explicit columns. Never SELECT * from a table.
- Keep the result small (a LIMIT of at most {max_rows} is enforced regardless).

CODED VALUES: columns often store short codes. A card shows each coded column's values as either
`code = meaning` (meaning known) or a bare `code` (meaning unknown). Use values EXACTLY as stored --
filter, group, and select on the stored code. When only a bare code is given, RETURN THE RAW CODE;
never invent a human-readable label for it (e.g. do NOT write `CASE WHEN commod='ST' THEN
'Strawberries' ...`). You do not know an ungrounded code's meaning, and guessing it returns wrong
data -- the raw code is the correct answer."""
    reply = _REPLY_AFTER_REASONING if reasoning == "before_sql" else _REPLY_ONLY
    tail = f"""{defer}

{reply}
{{"sql": "<the SELECT, or null>", "reason": "<one sentence>"{declare_key}}}
{declare}"""
    # `writer_rules = "none"`: the task, the decline and the reply, without the persona or the rule sections. The decider
    # enforces the hard rules on what comes back either way; this only stops telling the model them.
    base = f"{task}\n\n{tail}" if rules == "none" else f"{PERSONA}\n\n{task}\n\n{rule_text}\n\n{tail}"
    preamble = STRATEGY_PREAMBLES.get(strategy or "direct", "")
    if preamble and reasoning == "before_sql":
        preamble = preamble.replace("silently ", "")  # the reasoning is written now, so "silently" would contradict it
    return f"{base}\n{preamble}\n" if preamble else base


def meaning_sections(packet: ContextPacket) -> list[str]:
    """The certified meaning a packet carries -- definitions, measures, dimensions, code vocabulary
    -- as prompt lines, each section led by a blank line; empty when the packet carries none.

    One rendering for every model that reads an answer, not only the one that writes it. The
    result judge was handed the cards alone, so it judged a certified expression against its own
    reading of the question's words: on the fs payments corpus it declined three of six answers
    that were the certified expression verbatim, and passed the bare arm's six wrong ones (M132).
    """
    parts: list[str] = []
    if packet.definitions:
        parts += ["", "DEFINITIONS (authoritative business meanings -- follow them exactly):"]
        parts += [
            f"- {sanitize(d.term, limit=80)}: {sanitize(d.definition, limit=600)}"
            for d in packet.definitions
        ]
    if packet.metrics:
        # The EXPRESSION is the payload. A section naming "Settled Payment Volume" without saying
        # how it is computed tells the model a phrase it already had; the certified SQL is the part
        # a schema cannot supply.
        parts += ["", "CERTIFIED MEASURES (the authoritative way to compute these -- follow the expression exactly):"]
        parts += [
            f"- {sanitize(m.label, limit=80)} ({sanitize(m.id, limit=80)}) over {sanitize(m.measure.source, limit=80)}"
            f": {sanitize(m.measure.expr, limit=600)}"
            for m in packet.metrics
        ]
    if packet.dimensions:
        parts += ["", "CERTIFIED DIMENSIONS (the agreed way to slice these tables):"]
        parts += [
            f"- {sanitize(d.label, limit=80)} = {sanitize(d.expression(), limit=200)}"
            f" on {sanitize(d.source, limit=80)}"
            for d in packet.dimensions
        ]
    if packet.concepts:
        parts += ["", "CODE VOCABULARY (candidate codes for this question -- filter on the CODE, never the label):"]
        grouped: dict[tuple[str, str], list[str]] = {}
        for concept in packet.concepts:
            grouped.setdefault((concept.column_id, concept.scheme_label), []).append(
                f"{concept.notation} = {sanitize(concept.label, limit=120)}"
            )
        for (column_id, scheme), items in grouped.items():
            parts.append(f"- {column_id} uses {scheme}. Candidates: {'; '.join(items)}")
    return parts


def history_section(packet: ContextPacket) -> list[str]:
    """The packet's earlier turns as prompt lines, led by a blank line; empty when there are none.

    Shared with the judge for the reason `meaning_sections` is: a follow-up such as "and by
    channel?" names nothing on its own, so a judge without the earlier turns grades the words and
    declined three of five right follow-ups on the fs payments corpus (M134).
    """
    if not packet.history:
        return []
    # Prior turns, oldest first. The rows are what a pronoun resolves against: the
    # previous SQL computed "the top region" without ever naming it.
    parts = [
        "",
        "EARLIER IN THIS CONVERSATION (resolve references like \"it\" against these "
        "results; the current QUESTION is still the one to answer):",
    ]
    for turn in packet.history:
        parts.append(f"Q: {sanitize(turn.question, limit=300)}")
        if turn.sql:
            parts.append(f"SQL: {turn.sql}")
        if turn.columns and turn.rows:
            parts.append("RESULT: " + " | ".join(sanitize(c, limit=60) for c in turn.columns))
            for row in turn.rows:
                parts.append(
                    "        " + " | ".join(sanitize(str(v), limit=60) for v in row)
                )
    return parts


def _context_parts(packet: ContextPacket) -> list[str]:
    """What the question is read against: the sections and the tables, each section led by a blank line."""
    cards = "\n\n".join(c.card for c in packet.cards) or "(no tables are available to you)"
    parts = meaning_sections(packet)
    if packet.examples:
        parts += ["", "WORKED EXAMPLES (verified queries over these tables -- adapt, don't copy blindly):"]
        for ex in packet.examples:
            parts += [f"Q: {sanitize(ex.question, limit=300)}", f"SQL: {ex.sql}"]
    parts += history_section(packet)
    parts += ["", "TABLES:", cards]
    return parts


def _feedback_parts(feedback: str | None) -> list[str]:
    if not feedback:
        return []
    return ["", "Your previous query was rejected. Fix exactly this and try again:", feedback]


def question_last_prompt(system: str, packet: ContextPacket, feedback: str | None = None) -> tuple[str, str]:
    """(system, user) with the context in the system message and the question alone in the user message."""
    context = "\n".join(_context_parts(packet)).strip("\n")
    return f"{system}\n\n{context}", "\n".join([f"QUESTION: {packet.question}", *_feedback_parts(feedback)])


def user_prompt(packet: ContextPacket, feedback: str | None = None) -> str:
    # As asked, the way the judge, the selector and the synthesizer read it. A 500-character cut here answered long
    # questions as cut ones (M147), and `sanitize` turned backticks into quotes and folded line breaks for the
    # generator alone (M148). `sanitize` guards catalog text, which anyone who can name a column writes; the question is
    # the asker's own. Its length is bounded before this runs: both doors decline a question over
    # MNEMIQ_MAX_QUESTION_CHARS before retrieval, and `Agent.answer` declines it again.
    return "\n".join([f"QUESTION: {packet.question}", *_context_parts(packet), *_feedback_parts(feedback)])

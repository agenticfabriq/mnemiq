"""Fit a packet's cards to the model server's context window (M127).

Describing every column (M112) took a design partner's largest generation prompt to 40,293 tokens,
past the 28,672-token window a 32B gets on one 24 GB card -- so every question that reached their
wide tables was refused by the server. Fewer tables did not rescue it: one 965-column table's card,
described, nearly filled the window alone.

So the cards keep every column's name and type -- a column the model cannot see, it cannot select
-- and spend descriptions where the question is: first the columns the question, a definition or a
measure names, then the join keys, then the columns whose name and description share the
question's words. As many as the window has room for, after the reply budget and an allowance for
the corrector's feedback.

**Nothing changes while the prompt fits.** The packet comes back as the same object, cards byte
for byte as retrieval rendered them, without a call to the server when a conservative estimate
already fits; only past that does a server count decide. A schema that fits today (every BIRD
database, the demo stores) never reaches the trimming at all.

Both doors call `fit`: `Runtime.ask` and eval's `build_engine`, after the packet is complete and
before the agent sees it, so the generator, the corrector and the judge read the same cards.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace

from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, _attach_facts

logger = logging.getLogger(__name__)

# The corrector's retry re-sends the prompt with the database's error and the failed SQL, and the
# judge adds the SQL and a result preview: room for those, so a first call that just fits does not
# become a retry the server refuses.
FEEDBACK_ALLOWANCE = 1000
# Estimates from a server-calibrated ratio stay this far under the window: the trimmed text is not
# the text that was counted.
MARGIN = 1.03
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be by for from in into is it its of on or per show the
this that to was were what which who with all any each list me our your how many much""".split())


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def rank_columns(packet: ContextPacket, snapshot, policy) -> list[tuple[str, str]]:
    """The packet's described columns, the ones a description helps most first.

    Named (the question, a definition or a measure spells the column), then join keys, then by
    shared words -- a word of the question in the column's name counts twice, in its description
    once. Ties keep retrieval's table order and the snapshot's column order. Denied and masked
    columns are left out: their descriptions are never rendered.
    """
    order = {c.object_id: i for i, c in enumerate(packet.cards)}
    named_text = "\n".join(
        [packet.question]
        + [f"{d.term} {d.definition}" for d in packet.definitions]
        + [f"{m.label} {m.measure.expr}" for m in packet.metrics]
        + [f"{d.label} {d.expr or ''}" for d in packet.dimensions])
    question = _words(packet.question)
    joins: set[tuple[str, str]] = set()
    for rel in snapshot.relationships:
        for key in rel.join_keys:
            joins.add((rel.from_, key.left))
            joins.add((rel.to, key.right))
    ranked = []
    for index, column in enumerate(snapshot.columns):
        table = column.object_id
        if table not in order or not column.description:
            continue
        key = (table, column.name)
        if key in policy.denied or key in policy.masked:
            continue
        named = re.search(rf"(?<![A-Za-z0-9_]){re.escape(column.name)}(?![A-Za-z0-9_])",
                          named_text, re.IGNORECASE) is not None
        tier = 0 if named else 1 if key in joins else 2
        score = (2 * len(question & _words(column.name.replace("_", " ")))
                 + len(question & _words(column.description)))
        ranked.append((tier, -score, order[table], index, key))
    ranked.sort()
    return [entry[-1] for entry in ranked]


@dataclass
class PromptFitter:
    """What `fit` needs to know about the model and the prompt it will be sent.

    `window` is the declared one (MNEMIQ_LLM_CONTEXT_WINDOW); without it, the window the server
    reports the first time a prompt is counted is kept for the rest of the process. `count` is
    `(system, user) -> (tokens, window) | None`, the server's own count, None where it has none.
    """

    dialect: str
    card_style: str
    reply_tokens: int
    assertive: bool = False
    declare_assumed_terms: bool = False
    window: int | None = None
    count: object = None
    _server_window: int | None = field(default=None, init=False, repr=False)

    def system(self) -> str:
        """The longest system prompt a call can carry: deep mode's strategies differ in length."""
        from mnemiq.agent.loop import STRATEGIES
        from mnemiq.generate.prompts import system_prompt

        return max((system_prompt(dialect=self.dialect, strategy=s, assertive=self.assertive,
                                  declare_assumed_terms=self.declare_assumed_terms)
                    for s in (None, *STRATEGIES)), key=len)

    def fit(self, packet: ContextPacket, snapshot, grants) -> ContextPacket:
        """`packet` itself when it fits (or nothing says how big the window is); otherwise a copy
        whose cards carry as many descriptions as fit. Never raises: a question is not stopped by
        the step that exists to let it through."""
        try:
            return self._fit(packet, snapshot, grants)
        except Exception:  # noqa: BLE001
            logger.warning("prompt window: fitting the cards failed; sending them as retrieved",
                           exc_info=True)
            return packet

    def _fit(self, packet: ContextPacket, snapshot, grants) -> ContextPacket:
        from mnemiq.generate.prompts import user_prompt
        from mnemiq.llm.window import CHARS_PER_TOKEN_FLOOR

        if snapshot is None or not packet.cards:
            return packet
        system = self.system()
        reserve = self.reply_tokens + FEEDBACK_ALLOWANCE
        full_chars = len(system) + len(user_prompt(packet))
        window = self.window or self._server_window
        # A floor of characters per token makes this a ceiling on tokens: when it fits, the prompt
        # fits, and no request leaves the process.
        if window is not None and math.ceil(full_chars / CHARS_PER_TOKEN_FLOOR) + reserve <= window:
            return packet
        counted = self.count(system, user_prompt(packet)) if callable(self.count) else None
        if counted is not None:
            tokens, reported = counted
            self._server_window = self._server_window or reported
            window = window or reported
        if window is None:
            return packet  # nothing to fit against: as before M127
        if counted is not None:
            if tokens + reserve <= window:
                return packet
            per_char = tokens / full_chars
        else:
            if math.ceil(full_chars / CHARS_PER_TOKEN_FLOOR) + reserve <= window:
                return packet
            per_char = 1 / CHARS_PER_TOKEN_FLOOR
        return self._trim(packet, snapshot, grants, system, window - reserve, per_char)

    def _trim(self, packet, snapshot, grants, system, budget, per_char) -> ContextPacket:
        from mnemiq.generate.prompts import user_prompt
        from mnemiq.semantic.cards import build_cards
        from mnemiq.sql.policy import build_access_policy

        policy = build_access_policy(snapshot, grants)
        ranked = rank_columns(packet, snapshot, policy)

        def render(n: int) -> ContextPacket:
            keep = set(ranked[:n])
            texts = {c.object_id: c.text for c in build_cards(
                snapshot, policy=policy, style=self.card_style,
                describe=lambda table, column: (table, column) in keep)}
            fresh = [RetrievedCard(object_id=c.object_id, card=texts[c.object_id], score=c.score)
                     for c in packet.cards if c.object_id in texts]
            _attach_facts(fresh, snapshot.table_facts, self.card_style)
            by_id = {c.object_id: c for c in fresh}
            # A card retrieval served from the stored text (no snapshot binding) is kept as served.
            cards = [by_id.get(c.object_id, c) for c in packet.cards]
            return replace(packet, cards=cards, descriptions=(n, len(ranked)))

        def tokens(candidate: ContextPacket) -> int:
            return math.ceil((len(system) + len(user_prompt(candidate))) * per_char * MARGIN)

        low, high = 0, len(ranked)
        best = render(0)
        if tokens(best) > budget:
            logger.warning("prompt window: even with no column descriptions the prompt needs about "
                           "%d tokens against %d after the reply budget; the server may refuse it",
                           tokens(best), budget)
            return best
        while low < high:  # the most descriptions that fit, in rank order
            mid = (low + high + 1) // 2
            candidate = render(mid)
            if tokens(candidate) <= budget:
                low, best = mid, candidate
            else:
                high = mid - 1
        logger.info("prompt window: %d of %d column descriptions sent, to fit %d tokens",
                    low, len(ranked), budget)
        return best


def fitter_for(settings, dialect: str) -> PromptFitter:
    """The fitter both doors use, from the same settings the generator is built from."""
    from mnemiq.generate.generator import GENERATOR_MAX_TOKENS
    from mnemiq.llm.client import reasoning_budget
    from mnemiq.llm.window import _within_deadline, count_on_server

    def count(system: str, user: str):
        if not settings.llm_base_url:
            return None
        try:
            return _within_deadline(count_on_server, settings.llm_base_url, settings.llm_model,
                                    system, user, api_key=settings.llm_api_key)
        except Exception:  # noqa: BLE001 -- on the question path, a failed count is no count
            logger.debug("prompt window: the server count failed", exc_info=True)
            return None

    return PromptFitter(
        dialect=dialect, card_style=settings.card_style,
        reply_tokens=reasoning_budget(settings.llm_model, GENERATOR_MAX_TOKENS),
        assertive=settings.assertive_sql, declare_assumed_terms=settings.guard_undefined_terms,
        window=settings.llm_context_window, count=count)

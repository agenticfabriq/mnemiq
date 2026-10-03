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
for byte as retrieval rendered them. Without a call to the server when the prompt's UTF-8 bytes
already fit -- a byte-level tokenizer spends at most one token a byte, so that is a bound, where a
characters-per-token ratio is only an estimate (digits are a token each; a CJK character is
three bytes) -- and on the server's count otherwise. A schema that fits today never reaches the
trimming at all.

Both doors call `fit`: `Runtime.ask` and eval's `build_engine`, after the packet is complete and
before the agent sees it, so the generator, the corrector and the judge read the same cards.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field, replace

# Defined with the boot advisory, so the boot line and the question path agree on what fits.
from mnemiq.llm.window import FEEDBACK_ALLOWANCE
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, _attach_facts

logger = logging.getLogger(__name__)

# What the server's chat template wraps around the two messages (role markers, separators), in
# tokens: added to the byte bound below, which counts only the messages' own text.
TEMPLATE_ALLOWANCE = 64
# A server that gave no count is asked again after this long, not on every question.
NO_COUNT_RETRY_S = 600.0
# How many times the chosen prompt is re-counted and shrunk before it is sent as it stands.
RECOUNTS = 3
# Estimates from a server-calibrated ratio stay this far under the window: the trimmed text is not
# the text that was counted.
MARGIN = 1.03
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be by for from in into is it its of on or per show the
this that to was were what which who with all any each list me our your how many much""".split())


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def _byte_bound(system: str, user: str) -> int:
    """An upper bound on the prompt's tokens: a byte-level tokenizer spends at most one a byte."""
    return len(system.encode()) + len(user.encode()) + TEMPLATE_ALLOWANCE


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
    _no_count_until: float | None = field(default=None, init=False, repr=False)

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

    def _counted(self, system: str, user: str) -> int | None:
        """The server's count of this prompt, or None. A server that gave none (no /tokenize, or not
        answering) is not asked again for NO_COUNT_RETRY_S: without a declared window every question
        would otherwise pay a round trip, up to the count's deadline, for nothing."""
        if not callable(self.count) or (
                self._no_count_until is not None and time.monotonic() < self._no_count_until):
            return None
        counted = self.count(system, user)
        if counted is None:
            self._no_count_until = time.monotonic() + NO_COUNT_RETRY_S
            return None
        self._no_count_until = None
        tokens, reported = counted
        self._server_window = self._server_window or reported
        return tokens

    def _fit(self, packet: ContextPacket, snapshot, grants) -> ContextPacket:
        from mnemiq.generate.prompts import user_prompt

        if snapshot is None or not packet.cards:
            return packet
        system = self.system()
        reserve = self.reply_tokens + FEEDBACK_ALLOWANCE
        user = user_prompt(packet)
        ceiling = _byte_bound(system, user)
        window = self.window or self._server_window
        # At most a token a byte: when the bytes fit, the prompt fits, and no request leaves the
        # process.
        if window is not None and ceiling + reserve <= window:
            return packet
        tokens = self._counted(system, user)
        window = self.window or self._server_window
        if window is None:
            return packet  # nothing to fit against: as before M127
        if (tokens if tokens is not None else ceiling) + reserve <= window:
            return packet
        return self._trim(packet, snapshot, grants, system, window - reserve, tokens)

    def _trim(self, packet, snapshot, grants, system, budget, full_tokens) -> ContextPacket:
        """The most descriptions, in rank order, whose prompt fits `budget`.

        Estimated from the server's counts at both ends -- the full prompt and the one with no
        descriptions -- because one average ratio is wrong for the middle: descriptions are prose
        and column names are not (measured on the partner shape: 3.96 characters a token with
        every description, 3.16 with none), so a ratio taken from the full prompt undercounts a
        trimmed one by a quarter, which the server then refused. The chosen prompt is counted
        once more and shrunk while it is still over. Without counts, the byte bound is used
        throughout: it trims more than needed, never too little."""
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

        def chars(candidate: ContextPacket) -> int:
            return len(system) + len(user_prompt(candidate))

        floor = render(0)
        floor_tokens = self._counted(system, user_prompt(floor)) if full_tokens is not None else None
        if floor_tokens is None:
            def estimate(candidate: ContextPacket) -> int:
                return _byte_bound(system, user_prompt(candidate))
        else:
            spread = max(chars(render(len(ranked))) - chars(floor), 1)
            per_char = max(full_tokens - floor_tokens, 0) / spread

            def estimate(candidate: ContextPacket) -> int:
                return floor_tokens + math.ceil((chars(candidate) - chars(floor)) * per_char * MARGIN)

        if estimate(floor) > budget:
            logger.warning("prompt window: even with no column descriptions the prompt needs about "
                           "%d tokens against %d after the reply budget; the server may refuse it",
                           estimate(floor), budget)
            return floor
        low, high, best = 0, len(ranked), floor
        while low < high:  # the most descriptions that fit, in rank order
            mid = (low + high + 1) // 2
            candidate = render(mid)
            if estimate(candidate) <= budget:
                low, best = mid, candidate
            else:
                high = mid - 1
        if floor_tokens is not None and low:
            per_description = max((full_tokens - floor_tokens) / len(ranked), 1.0)
            for _ in range(RECOUNTS):  # the server has the last word on the chosen prompt
                counted = self._counted(system, user_prompt(best))
                if counted is None or counted <= budget:
                    break
                low = max(0, low - math.ceil((counted - budget) / per_description) - 1)
                best = render(low)
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

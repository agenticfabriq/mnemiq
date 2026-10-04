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

**With a server count, nothing that fits is changed.** The packet comes back as the same object,
cards byte for byte as retrieval rendered them: without a request when the prompt's UTF-8 bytes
already fit (`mnemiq.llm.window.byte_bound`: at most a token a byte, plus room for the chat
template), and on the server's /tokenize count otherwise -- one round trip for each question whose
bytes do not fit, and one a minute however small the prompt, to keep the server's window current
(milliseconds on vLLM). **Without a count** (a server with no /tokenize and a
declared MNEMIQ_LLM_CONTEXT_WINDOW) the byte bound decides alone, and it errs long, about 2.5
times for English: such a deployment trims descriptions it has room for. That is the trade --
never too little, often too much -- and the boot check, which uses the same bound, says so.

Both doors call `fit`: `Runtime.ask` and eval's `build_engine`, after the packet is complete and
before the agent sees it, so the generator, the corrector and the judge read the same cards.
"""

from __future__ import annotations

import logging
import math
import re
import time
from itertools import zip_longest
from dataclasses import dataclass, field, replace

# Defined with the boot advisory, so the boot line and the question path agree on what fits.
from mnemiq.llm.window import FEEDBACK_ALLOWANCE, byte_bound
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, _attach_facts

logger = logging.getLogger(__name__)

# A server that gave no count is asked again after this long, not on every question: long for one
# that never counted (no /tokenize, most likely), short for one that has (a passing failure).
NO_COUNT_RETRY_S = 600.0
TRANSIENT_RETRY_S = 30.0
# The server's window is re-learned after this long: a server restarted with a smaller window must
# not keep receiving prompts sized for the old one. One /tokenize a minute, at most.
WINDOW_TTL_S = 60.0
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


def rank_columns(packet: ContextPacket, snapshot, policy) -> list[tuple[str, str]]:
    """The packet's described columns, the ones a description helps most first.

    Named (the question, a definition or a measure spells the column), then join keys, then the
    rest **in turns across tables**: each table's best remaining column, table by table, then each
    table's next. Within a table the best is the one sharing most of the question's words (in its
    name counting twice, in its description once), ties in the snapshot's column order. Denied and
    masked columns are left out: their descriptions are never rendered.

    In turns because one ranking over every column let a single wide table take the budget: on a
    design partner's shape, a 965-column table whose descriptions all mention "ingot" and "batch"
    got most of the descriptions sent, the batch table holding the definition's column came last
    with few, and the model answered that the batch table was missing (M127's follow-up).
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
    first = [entry[-1] for entry in ranked if entry[0] < 2]
    by_table: dict[str, list[tuple[str, str]]] = {}
    for entry in ranked:  # already in score order within each table
        if entry[0] == 2:
            by_table.setdefault(entry[-1][0], []).append(entry[-1])
    turns = [by_table[t] for t in sorted(by_table, key=order.__getitem__)]
    shared = [column for round_ in zip_longest(*turns) for column in round_ if column is not None]
    return first + shared


@dataclass
class PromptFitter:
    """What `fit` needs to know about the model and the prompt it will be sent.

    `window` is the declared one (MNEMIQ_LLM_CONTEXT_WINDOW), the fallback: the window the server
    reports outranks it, refreshed by every count and re-learned once WINDOW_TTL_S has passed.
    `count` is `(system, user) -> (tokens, window) | None`, the server's own count, None where it
    has none.
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
    _window_seen_at: float = field(default=0.0, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)

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
        """The server's count of this prompt, or None. A server that gave none is not asked again for
        a while -- TRANSIENT_RETRY_S if it has counted before and has failed fewer than three times in
        a row, NO_COUNT_RETRY_S otherwise (no /tokenize, most likely) -- since every question would
        otherwise pay a round trip, up to the count's deadline, for nothing. A failure also retires
        an expired server window in favour of the declared one; a success refreshes it."""
        if not callable(self.count) or (
                self._no_count_until is not None and time.monotonic() < self._no_count_until):
            return None
        counted = self.count(system, user)
        if counted is None:
            now = time.monotonic()
            self._failures += 1
            passing = self._window_seen_at and self._failures < 3
            self._no_count_until = now + (TRANSIENT_RETRY_S if passing else NO_COUNT_RETRY_S)
            if (self._server_window is not None and self.window is not None
                    and now - self._window_seen_at > WINDOW_TTL_S):
                # An expired window the server would not confirm yields to the declared one -- a
                # server restarted smaller must not keep its old limit through a failed count. With
                # no declared window, the last one the server gave is still the best word there is.
                self._server_window = None
            return None
        self._no_count_until, self._failures = None, 0
        tokens, reported = counted
        # Every count refreshes it: the latest word on the window is the server's latest answer.
        self._server_window, self._window_seen_at = reported, time.monotonic()
        return tokens

    def _fit(self, packet: ContextPacket, snapshot, grants) -> ContextPacket:
        from mnemiq.generate.prompts import user_prompt

        if snapshot is None or not packet.cards:
            return packet
        system = self.system()
        reserve = self.reply_tokens + FEEDBACK_ALLOWANCE
        user = user_prompt(packet)
        ceiling = byte_bound(system, user)
        tokens = None
        if (self._server_window is None
                or time.monotonic() - self._window_seen_at > WINDOW_TTL_S):
            # The server's window outranks a declared one (MNEMIQ_LLM_CONTEXT_WINDOW is the fallback
            # for a server that reports none), so it is learned before anything is waved through:
            # a count on the first question, and again once WINDOW_TTL_S has passed.
            tokens = self._counted(system, user)
        window = self._server_window or self.window
        if window is None:
            return packet  # nothing to fit against: as before M127
        # At most a token a byte: when the bytes fit, the prompt fits, and no request leaves the
        # process.
        if tokens is None and ceiling + reserve <= window:
            return packet
        if tokens is None:
            tokens = self._counted(system, user)
            window = self._server_window or self.window  # that count may have moved it
        if (tokens if tokens is not None else ceiling) + reserve <= window:
            return packet
        return self._trim(packet, snapshot, grants, system, reserve, tokens)

    def _trim(self, packet, snapshot, grants, system, reserve, full_tokens) -> ContextPacket:
        """The most descriptions, in rank order, whose prompt fits the window less `reserve`.

        Estimated from the server's counts at both ends -- the full prompt and the one with no
        descriptions -- because one average ratio is wrong for the middle: descriptions are prose
        and column names are not (measured on the partner shape: 3.96 characters a token with
        every description, 3.16 with none), so a ratio taken from the full prompt undercounts a
        trimmed one by a quarter, which the server then refused. Only a prompt the server has
        counted inside the budget is returned: the chosen one is counted and shrunk while it is
        still over, and if no count confirms it, the prompt with no descriptions -- counted at the
        start -- goes instead. Without counts, the byte bound is used throughout: it trims more
        than needed, never too little."""
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

        def budget() -> int:
            # Re-read at every decision: each count can report a new window (a server restarted
            # mid-fit), and a candidate is verified against the latest, not the one it began with.
            return (self._server_window or self.window) - reserve

        floor = render(0)
        floor_tokens = self._counted(system, user_prompt(floor)) if full_tokens is not None else None
        if floor_tokens is None:
            def estimate(candidate: ContextPacket) -> int:
                return byte_bound(system, user_prompt(candidate))
        else:
            spread = max(chars(render(len(ranked))) - chars(floor), 1)
            per_char = max(full_tokens - floor_tokens, 0) / spread

            def estimate(candidate: ContextPacket) -> int:
                return floor_tokens + math.ceil((chars(candidate) - chars(floor)) * per_char * MARGIN)

        if estimate(floor) > budget():
            logger.warning("prompt window: even with no column descriptions the prompt needs about "
                           "%d tokens against %d after the reply budget; the server may refuse it",
                           estimate(floor), budget())
            return floor
        low, high, best = 0, len(ranked), floor
        while low < high:  # the most descriptions that fit, in rank order
            mid = (low + high + 1) // 2
            candidate = render(mid)
            if estimate(candidate) <= budget():
                low, best = mid, candidate
            else:
                high = mid - 1
        if floor_tokens is not None and low:
            # The server has the last word: return a prompt it counted inside the budget, or the
            # floor, which it already did. An estimate is not enough -- descriptions differ in
            # length and density, and a shrink by their average can miss three times running.
            per_description = max((full_tokens - floor_tokens) / len(ranked), 1.0)
            verified = False
            for _ in range(RECOUNTS):
                counted = self._counted(system, user_prompt(best))
                if counted is None:
                    break
                if counted <= budget():
                    verified = True
                    break
                low = max(0, low - math.ceil((counted - budget()) / per_description) - 1)
                best = render(low)
                if low == 0:
                    break
            if not verified:
                low, best = 0, floor
                if floor_tokens > budget():  # the window shrank below even the floor mid-fit
                    logger.warning("prompt window: even with no column descriptions the prompt "
                                   "needs %d tokens against %d after the reply budget; the "
                                   "server may refuse it", floor_tokens, budget())
        logger.info("prompt window: %d of %d column descriptions sent, to fit %d tokens",
                    low, len(ranked), budget())
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

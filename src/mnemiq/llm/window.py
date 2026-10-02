"""Will the largest generation prompt fit the server's context window? (M119, part 3)

The per-call cut check refuses an answer built on a prompt the server cut, and vLLM refuses a
prompt plus reply budget longer than `--max-model-len` outright. Both are right, and both surface
one question at a time, after the work. This asks once, at boot: the largest prompt this store can
produce, plus the room the generator is given to reply in, against the window.

"Largest" is built with the real formatter from the parts that do not depend on the question: the
`retrieval_k` tables that bring the most into a prompt -- card in the generator's card style,
unscoped (a policy only removes columns), facts, and the definitions and certified measures bound
to it -- with every table visible, a definition shared by two of them counted with each, and a
section they leave empty given one item from a table that fills it; the `retrieval_k` worked
examples that add the most once rendered; the longest strategy's system prompt. The question is an
allowance of 500 tokens, not text: `sanitize` keeps 500 characters, and a byte-level tokenizer
spends at most one token on an ASCII character. Left out, because they depend on the question: the
code vocabulary for its columns, glossary terms it happens to name, and the conversation history.

Longest by length, then counted in tokens -- not longest in tokens. A set of tables shorter in
characters but denser in tokens (long identifiers, non-Latin text) can count more than the one
measured, as can a non-ASCII question against its allowance, and counting every candidate would
cost a server round trip per table at boot. So a warning means the measured parts do not fit, and a
"fits" close to the window is not a promise: leave headroom.

Counted by the server where it can count: vLLM's `/tokenize` returns the count, chat template
included, and `max_model_len` in one call. Otherwise the window is `MNEMIQ_LLM_CONTEXT_WINDOW`, as
declared, and the count an estimate at 2.5 characters a token -- below the fewest measured for a
generation prompt (2.56, over 220 of them), so the estimate errs toward warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import httpx

from mnemiq.authz.grants import GrantSet
from mnemiq.generate.generator import GENERATOR_MAX_TOKENS
from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.llm.client import reasoning_budget

CHARS_PER_TOKEN_FLOOR = 2.5
# `sanitize` keeps 500 characters of a question, and a byte-level tokenizer spends at most one token
# on an ASCII character. Added as tokens: as text, any filler tokenizes its own way (500 x's are a few).
# ASCII only: a byte-level tokenizer can spend a token per UTF-8 byte, so a 500-character question in
# Chinese or heavy with accents can cost more -- headroom again.
QUESTION_ALLOWANCE = 500


@dataclass(frozen=True)
class WindowReport:
    prompt_tokens: int
    reply_tokens: int
    window: int | None
    counted_by_server: bool
    cards: int

    @property
    def needed(self) -> int:
        return self.prompt_tokens + self.reply_tokens

    @property
    def fits(self) -> bool | None:
        """None when no window is known: nothing to compare against."""
        return None if self.window is None else self.needed <= self.window

    def sentence(self) -> str:
        count = (f"{self.prompt_tokens:,} tokens, counted by the server" if self.counted_by_server
                 else f"about {self.prompt_tokens:,} tokens, estimated from its length")
        window = ("unknown: the server gave no count (it has no /tokenize, or it did not answer) "
                  "and MNEMIQ_LLM_CONTEXT_WINDOW is unset"
                  if self.window is None else
                  f"{self.window:,} tokens ({'reported by the server' if self.counted_by_server else 'MNEMIQ_LLM_CONTEXT_WINDOW'})")
        tables = "the table" if self.cards == 1 else f"the {self.cards} tables"
        return (f"the longest generation prompt this store can produce ({tables} "
                f"bringing the most into it, with their definitions, measures and examples, and "
                f"{QUESTION_ALLOWANCE} tokens for the question) is "
                f"{count}; with the "
                f"generator's {self.reply_tokens:,}-token reply budget it needs {self.needed:,}, and "
                f"the window is {window}")


def largest_prompt(con, snapshot, settings, dialect: str) -> tuple[str, str, int]:
    """(system, user, cards) for the largest prompt the question-independent parts can make."""
    from mnemiq.agent.loop import STRATEGIES
    from mnemiq.semantic.cards import build_cards
    from mnemiq.semantic.glossary import select_definitions
    from mnemiq.semantic.measures import select_dimensions, select_metrics
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, _attach_facts

    k = settings.retrieval_k
    rendered = build_cards(snapshot, policy=None, style=settings.card_style)
    # Every table visible: an identity granted all of them sees the most, and a definition bound to
    # a table outside the k still rides in with the one inside it.
    shown = GrantSet(frozenset(c.object_id for c in rendered))

    def packet_for(cards: list) -> ContextPacket:
        _attach_facts(cards, snapshot.table_facts, settings.card_style)
        ids = [c.object_id for c in cards]
        return ContextPacket(
            question="", cards=cards, grant_fingerprint="", enrichment_version=None,
            # Bound definitions ride with their tables whatever the question says; "" names no term.
            definitions=select_definitions("", snapshot.definitions, shown, ids),
            metrics=select_metrics(ids, snapshot.metrics, shown),
            dimensions=select_dimensions(ids, snapshot.dimensions, shown),
        )

    def card(c) -> RetrievedCard:
        return RetrievedCard(object_id=c.object_id, card=c.text, score=0.0)

    singles = {c.object_id: packet_for([card(c)]) for c in rendered}

    def brings(c) -> int:
        """What a table adds to a prompt that already has every section's header: each section it
        fills, rendered twice minus once, which is that section's content without its header."""
        one = singles[c.object_id]
        base = len(user_prompt(one))
        return sum(len(user_prompt(replace(one, **{part: getattr(one, part) * 2}))) - base
                   for part in ("cards", "definitions", "metrics", "dimensions"))

    # Ranked by what each table brings into the prompt, not by its card alone: a narrow table with
    # fifty bound definitions outweighs a wide one with none. Headers are left out of the rank, since
    # the prompt prints each once however many tables fill its section.
    largest = sorted(rendered, key=lambda c: (-brings(c), c.object_id))[:k]
    per_table = [singles[c.object_id] for c in largest]
    # Then an upper bound rather than one real packet. Each chosen table's bound definitions and
    # measures are listed with it, so one bound to two of them appears twice where a real packet
    # lists it once: a real packet of k tables is at most the headers plus the sum of what its
    # tables bring, and these k bring the largest sum. And a section these k leave empty gets one
    # item from a table that fills it, header and all, so a real packet cannot carry a header this
    # one lacks. Exact when nothing is shared and no section is left empty.
    sections = {part: [x for p in per_table for x in getattr(p, part)]
                for part in ("definitions", "metrics", "dimensions")}
    for part, items in sections.items():
        if not items:
            filled = next((getattr(p, part) for p in singles.values() if getattr(p, part)), [])
            items.extend(filled[:1])
    packet = ContextPacket(
        question="", cards=[p.cards[0] for p in per_table], grant_fingerprint="",
        enrichment_version=None, **sections,
    )
    packet.examples = _largest_examples(con, k)
    system = max((system_prompt(dialect=dialect, strategy=s, assertive=settings.assertive_sql,
                                declare_assumed_terms=settings.guard_undefined_terms)
                  for s in (None, *STRATEGIES)), key=len)
    return system, user_prompt(packet), len(packet.cards)


def _largest_examples(con, k: int) -> list:
    """The k examples that add the most once rendered -- `user_prompt` cuts a question to 300
    characters, so a long question with short SQL can add less than it looks. Every stored example
    is read and rendered twice, once per boot: linear in the example store, which is a few per
    table today."""
    from mnemiq.contract import Example
    from mnemiq.semantic.retrieval import ContextPacket

    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'example' "
        "AND table_schema = current_schema() AND table_catalog = current_database()"
    ).fetchone()
    if exists is None:
        return []
    examples = [Example(question=q, sql=s, object_id=o)
                for q, s, o in con.execute("SELECT question, sql, object_id FROM example").fetchall()]
    empty = ContextPacket(question="", cards=[], grant_fingerprint="", enrichment_version=None)

    def adds(example) -> int:
        once = len(user_prompt(replace(empty, examples=[example])))
        return len(user_prompt(replace(empty, examples=[example, example]))) - once

    return sorted(examples, key=lambda e: (-adds(e), e.question, e.sql))[:k]


def count_on_server(base_url: str, model: str, system: str, user: str,
                    http: httpx.Client | None = None, timeout: float = 10.0) -> tuple[int, int] | None:
    """(prompt tokens, window) from vLLM's `/tokenize`, or None where the server has no such door.

    `/tokenize` sits at the server root, beside `/v1`, and takes the chat messages, so the count
    includes the template the server wraps them in.
    """
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]}
    client = http or httpx.Client(follow_redirects=False, timeout=timeout)
    try:
        resp = client.post(f"{root}/tokenize", json=body)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        # Down, slow, or answering with something other than JSON (a proxy's catch-all page):
        # none of it is a count, and a declared window can still be checked without one -- an
        # Ollama that is not up yet at boot is exactly who MNEMIQ_LLM_CONTEXT_WINDOW is for.
        return None
    finally:
        if http is None:
            client.close()
    if not isinstance(data, dict):
        return None
    count, window = data.get("count"), data.get("max_model_len")
    if isinstance(count, int) and isinstance(window, int):
        return count, window
    return None


def check_window(con, snapshot, settings, dialect: str,
                 http: httpx.Client | None = None) -> WindowReport:
    system, user, cards = largest_prompt(con, snapshot, settings, dialect)
    reply = reasoning_budget(settings.llm_model, GENERATOR_MAX_TOKENS)
    counted = count_on_server(settings.llm_base_url, settings.llm_model, system, user, http=http)
    if counted is not None:
        prompt, window = counted
        return WindowReport(prompt + QUESTION_ALLOWANCE, reply, window, counted_by_server=True,
                            cards=cards)
    estimate = math.ceil((len(system) + len(user)) / CHARS_PER_TOKEN_FLOOR) + QUESTION_ALLOWANCE
    return WindowReport(estimate, reply, settings.llm_context_window, counted_by_server=False,
                        cards=cards)

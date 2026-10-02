"""Will the largest generation prompt fit the server's context window? (M119, part 3)

The per-call cut check refuses an answer built on a prompt the server cut, and vLLM refuses a
prompt plus reply budget longer than `--max-model-len` outright. Both are right, and both surface
one question at a time, after the work. This asks once, at boot: the largest prompt this store can
produce, plus the room the generator is given to reply in, against the window.

"Largest" is built with the real formatter, for one identity at a time, from the parts that do not
depend on the question: the `retrieval_k` granted tables that bring the most into a prompt -- card
rendered under that identity's access policy (denied columns gone, masked ones carrying their mask
note), facts, and the definitions and certified measures bound to it -- with a definition shared by
two of them counted with each, and a section they leave empty given one item from a granted table
that fills it; the `retrieval_k` permitted worked examples that add the most once rendered; the
longest strategy's system prompt. The question is an allowance of 500 tokens, not text: `sanitize`
keeps 500 characters, and a byte-level tokenizer spends at most one token on an ASCII character.
Left out, because they depend on the question: the code vocabulary for its columns, glossary terms
it happens to name, and the conversation history.

Per identity, because the prompt is sent to the model server to be counted, and the server must see
nothing it would not see from that identity's own questions: a worst case over metadata no one is
shown -- a PII column's description, a denied table -- would be a disclosure the boot check made by
itself. Each role the access policy declares is measured locally (the configured identity, where
the provider cannot list its roles), and only the heaviest is counted on the server.

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

import json
import math
import threading
import time
from dataclasses import dataclass, replace

import httpx

from mnemiq.authz.grants import GrantSet
from mnemiq.generate.generator import GENERATOR_MAX_TOKENS
from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.llm.client import reasoning_budget

CHARS_PER_TOKEN_FLOOR = 2.5
# The `/tokenize` call's limits (`count_on_server`). The reply lists every token id, about 7 bytes
# each: some 700 KB for a 100,000-token prompt.
DEADLINE_S = 15.0
MAX_REPLY_BYTES = 8 * 1024 * 1024
_PHASE_TIMEOUT_S = 5.0
# `sanitize` keeps 500 characters of a question, and a byte-level tokenizer spends at most one token
# on an ASCII character. Added as tokens: as text, any filler tokenizes its own way (500 x's are a few).
# ASCII only: a byte-level tokenizer can spend a token per UTF-8 byte, so a 500-character question in
# Chinese or heavy with accents can cost more -- headroom again.
QUESTION_ALLOWANCE = 500


@dataclass(frozen=True)
class WindowReport:
    """`prompt_tokens` is the upper bound; `real_tokens` the deduplicated packet of the same tables,
    one a retrieval can build (None when it could not be counted) -- with the largest examples, the
    longest strategy and a long question, so a question meeting those tables with less still fits.
    Past the window, the bound says a question MAY fail; the real packet says one CAN."""

    prompt_tokens: int
    reply_tokens: int
    window: int | None
    counted_by_server: bool
    cards: int
    real_tokens: int | None = None
    who: str = ""

    @property
    def needed(self) -> int:
        return self.prompt_tokens + self.reply_tokens

    @property
    def fits(self) -> bool | None:
        """None when no window is known: nothing to compare against."""
        return None if self.window is None else self.needed <= self.window

    @property
    def certain(self) -> bool:
        """A real packet, counted by the server, past the window: a question can fail."""
        return (self.counted_by_server and self.window is not None
                and self.real_tokens is not None
                and self.real_tokens + self.reply_tokens > self.window)

    def sentence(self) -> str:
        count = (f"{self.prompt_tokens:,} tokens, counted by the server" if self.counted_by_server
                 else f"about {self.prompt_tokens:,} tokens, estimated from its length")
        if self.real_tokens is not None and self.real_tokens != self.prompt_tokens:
            count += (f" (an upper bound; a real packet of those tables, each shared definition "
                      f"once, is {self.real_tokens:,})")
        window = ("unknown: the server gave no count (it has no /tokenize, or it did not answer) "
                  "and MNEMIQ_LLM_CONTEXT_WINDOW is unset"
                  if self.window is None else
                  f"{self.window:,} tokens ({'reported by the server' if self.counted_by_server else 'MNEMIQ_LLM_CONTEXT_WINDOW'})")
        tables = "the table" if self.cards == 1 else f"the {self.cards} tables"
        return (f"{'for ' + self.who + ', ' if self.who else ''}"
                f"the longest generation prompt this store can produce ({tables} "
                f"bringing the most into it, with their definitions, measures and examples, and "
                f"{QUESTION_ALLOWANCE} tokens for the question) is "
                f"{count}; with the "
                f"generator's {self.reply_tokens:,}-token reply budget it needs {self.needed:,}, and "
                f"the window is {window}")


def everything(snapshot) -> GrantSet:
    """Every table, every PII level cleared: the widest view, for tests and a single-identity store."""
    levels = frozenset(c.pii_level for c in snapshot.columns if c.pii_level)
    tables = {b.object_id for b in snapshot.source_bindings} | {c.object_id for c in snapshot.columns}
    return GrantSet(frozenset(tables), pii_clearance=levels)


def largest_prompt(con, snapshot, settings, dialect: str,
                   grants: GrantSet | None = None) -> tuple[str, str, int]:
    """(system, user, cards) for the largest prompt the question-independent parts can make."""
    system, bound, _real, cards = largest_prompts(con, snapshot, settings, dialect,
                                                  grants or everything(snapshot))
    return system, bound, cards


def largest_prompts(con, snapshot, settings, dialect: str,
                    grants: GrantSet) -> tuple[str, str, str, int]:
    """(system, upper bound, real packet of the same tables, cards), as `grants` may see them."""
    from mnemiq.agent.loop import STRATEGIES
    from mnemiq.semantic.cards import build_cards
    from mnemiq.semantic.glossary import select_definitions
    from mnemiq.semantic.measures import select_dimensions, select_metrics
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, _attach_facts

    from mnemiq.sql.policy import build_access_policy

    k = settings.retrieval_k
    # The cards retrieval renders for this identity: its tables only, under its column policy.
    rendered = [c for c in build_cards(snapshot, policy=build_access_policy(snapshot, grants),
                                       style=settings.card_style)
                if c.object_id in grants.objects]
    # Every granted table visible, so a definition bound to one inside the k and one outside it
    # still rides in.
    shown = grants

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
    examples = _largest_examples(con, k, grants.objects)
    packet.examples = examples
    # The same tables as one retrieval can bring them: each shared definition once, nothing
    # borrowed for an empty section -- though still with the largest examples, the longest
    # strategy and a long question. Past the window, a question can fail.
    real = packet_for([card(c) for c in largest])
    real.examples = examples
    system = max((system_prompt(dialect=dialect, strategy=s, assertive=settings.assertive_sql,
                                declare_assumed_terms=settings.guard_undefined_terms)
                  for s in (None, *STRATEGIES)), key=len)
    return system, user_prompt(packet), user_prompt(real), len(packet.cards)


def _largest_examples(con, k: int, allowed) -> list:
    """The k examples that add the most once rendered -- `user_prompt` cuts a question to 300
    characters, so a long question with short SQL can add less than it looks. Every stored example
    is read and rendered twice on each check, which runs at boot and again on every ask that sees a
    snapshot swap: linear in the size of the example store."""
    from mnemiq.contract import Example
    from mnemiq.semantic.retrieval import ContextPacket

    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'example' "
        "AND table_schema = current_schema() AND table_catalog = current_database()"
    ).fetchone()
    if exists is None:
        return []
    # Only examples whose tables the identity may see, as retrieval filters them.
    examples = [Example(question=q, sql=s, object_id=o)
                for q, s, t, o in con.execute(
                    "SELECT question, sql, tables, object_id FROM example").fetchall()
                if set(json.loads(t)) <= set(allowed)]
    empty = ContextPacket(question="", cards=[], grant_fingerprint="", enrichment_version=None)

    def adds(example) -> int:
        once = len(user_prompt(replace(empty, examples=[example])))
        return len(user_prompt(replace(empty, examples=[example, example]))) - once

    return sorted(examples, key=lambda e: (-adds(e), e.question, e.sql))[:k]


def count_on_server(base_url: str, model: str, system: str, user: str,
                    http: httpx.Client | None = None,
                    api_key: str | None = None) -> tuple[int, int] | None:
    """(prompt tokens, window) from vLLM's `/tokenize`, or None where the server has no such door.

    `/tokenize` sits at the server root, beside `/v1`, and takes the chat messages, so the count
    includes the template the server wraps them in. It carries the chat calls' own key, to the host
    they already send it to (redirects stay off): a server behind an authenticating proxy answers
    a keyless call with 401, which would read as no count.

    Bounded overall, not only per phase: httpx's timeout limits each silence, so a proxy dripping a
    byte every few seconds would hold the boot -- or the ask that saw a snapshot swap -- for as long
    as it liked. The reply is read in chunks against a deadline and a size cap, and either one
    reached is no count. What comes before the body -- the DNS lookup, which has no timeout, each
    address tried, the headers -- is bounded by `check_window`, which stops waiting at the deadline.
    """
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]}
    client = http or httpx.Client(follow_redirects=False, timeout=_PHASE_TIMEOUT_S)
    started = time.monotonic()
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with client.stream("POST", f"{root}/tokenize", json=body, headers=headers) as resp:
            if resp.status_code != 200:
                return None
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_REPLY_BYTES or time.monotonic() - started > DEADLINE_S:
                    return None
                chunks.append(chunk)
        data = json.loads(b"".join(chunks))
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


def _within_deadline(fn, *args, deadline: float | None = None, **kwargs):
    """`fn`'s result, or None once `deadline` (default DEADLINE_S) has passed -- a wall-clock
    bound on the caller.

    A daemon thread, because nothing in the socket stack can be interrupted from outside: on the
    deadline the caller stops waiting and the thread is left to its own timeouts. It holds nothing
    the caller needs when the caller passed no client of its own, as the runtime does not. Only the
    deadline means "no count": an exception `fn` raised is raised here, so a base URL httpx cannot
    parse still reaches the advisory's "measurement failed" line instead of reading as a server
    with no `/tokenize`.
    """
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["result"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- handed to the caller, raised there
            outcome["error"] = exc

    worker = threading.Thread(target=run, name="mnemiq-window-count", daemon=True)
    worker.start()
    worker.join(DEADLINE_S if deadline is None else max(deadline, 0.0))
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def check_window(con, snapshot, settings, dialect: str, grants: GrantSet | None = None,
                 http: httpx.Client | None = None, who: str = "") -> WindowReport:
    prompts = largest_prompts(con, snapshot, settings, dialect, grants or everything(snapshot))
    return _count(prompts, settings, http, who)


def grant_sets(authz, settings) -> list[tuple[str, GrantSet]]:
    """(label, grants) for each distinct view to measure: every role the policy declares, and the
    configured identity, whose roles merge into one grant that may be wider than any of them. Views that see no
    table are dropped, and identical grants are measured once. Other principals holding several
    roles are not: their combinations are not listed anywhere to measure."""
    from mnemiq.config import identity_from_settings
    from mnemiq.contract import IdentityContext

    roles = getattr(authz, "policy_roles", None)
    views = [(f"role {role}", authz.grants_for(
        IdentityContext(tenant_id="boot", principal_id="boot", roles=[role])))
        for role in (roles() if roles is not None else [])]
    identity = identity_from_settings(settings)
    views.append((f"identity {identity.principal_id}", authz.grants_for(identity)))
    measured: list[tuple[str, GrantSet]] = []
    seen: set[str] = set()
    for label, grants in views:
        if grants.objects and grants.fingerprint not in seen:
            seen.add(grants.fingerprint)
            measured.append((label, grants))
    return measured


def worst_window(con, snapshot, settings, dialect: str, authz,
                 http: httpx.Client | None = None) -> WindowReport | None:
    """The report for the view whose largest prompt is longest, measured locally and then counted
    on the server; None when no view sees a table."""
    candidates = [(label, largest_prompts(con, snapshot, settings, dialect, grants))
                  for label, grants in grant_sets(authz, settings)]
    if not candidates:
        return None
    label, prompts = max(candidates, key=lambda c: (len(c[1][0]) + len(c[1][1]), c[0]))
    return _count(prompts, settings, http, label)


def _count(prompts: tuple[str, str, str, int], settings, http: httpx.Client | None,
           who: str) -> WindowReport:
    system, bound, real, cards = prompts
    reply = reasoning_budget(settings.llm_model, GENERATOR_MAX_TOKENS)

    def count(user: str):
        return count_on_server(settings.llm_base_url, settings.llm_model, system, user, http=http,
                               api_key=settings.llm_api_key)

    # One deadline for both counts, the bound first: whatever the second does, the first is kept,
    # and a real packet that could not be counted only weakens CAN to MAY.
    started = time.monotonic()
    first = _within_deadline(count, bound)
    if first is not None:
        prompt, window = first
        second = first if real == bound else _within_deadline(
            count, real, deadline=DEADLINE_S - (time.monotonic() - started))
        return WindowReport(prompt + QUESTION_ALLOWANCE, reply, window, counted_by_server=True,
                            cards=cards,
                            real_tokens=None if second is None else second[0] + QUESTION_ALLOWANCE,
                            who=who)

    def estimate(user: str) -> int:
        return math.ceil((len(system) + len(user)) / CHARS_PER_TOKEN_FLOOR) + QUESTION_ALLOWANCE

    return WindowReport(estimate(bound), reply, settings.llm_context_window,
                        counted_by_server=False, cards=cards, real_tokens=estimate(real), who=who)

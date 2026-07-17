from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from mnemiq.contract import Example, Snapshot
from mnemiq.enrichment.pipeline import content_version
from mnemiq.enrichment.prompts import PERSONA
from mnemiq.enrichment.proposals import _clean_text, _extract_json
from mnemiq.execute.runner import ExecutionError, run
from mnemiq.semantic.cards import build_cards
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved

_MAX_QUESTION = 300


@dataclass
class ExampleProposal:
    question: str
    sql: str


def parse_examples(raw: str) -> list[ExampleProposal]:
    """Read a list of {question, sql}. Never raises; drops malformed items."""
    payload = _extract_json(raw)
    items = payload.get("examples") if isinstance(payload, dict) else None
    out: list[ExampleProposal] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        q = _clean_text(item.get("question"), _MAX_QUESTION)
        sql = item.get("sql")
        if q and isinstance(sql, str) and sql.strip():
            out.append(ExampleProposal(question=q, sql=sql.strip()))
    return out


def _example_system_prompt(dialect: str) -> str:
    return f"""{PERSONA}

You are given one table's card (columns, structural facts, relationships). Propose realistic
analyst questions and the {dialect} SELECT that answers each, using ONLY the columns on the
card (you may join to the related tables it lists). Prefer questions that exercise the grain,
the canonical measures, and the joins.

Return ONLY a JSON object, no prose:
{{"examples": [{{"question": "<natural language>", "sql": "<a {dialect} SELECT>"}}]}}"""


class ExampleGenerator(Protocol):
    def propose(self, table: str, card_text: str, dialect: str) -> list[ExampleProposal]: ...


class LLMExampleGenerator:
    def __init__(self, client, n: int = 5, max_tokens: int = 3000) -> None:
        self._client = client
        self._n = n
        self._max_tokens = max_tokens

    def propose(self, table: str, card_text: str, dialect: str) -> list[ExampleProposal]:
        # The card is engine-built; pass it through with structure intact (do NOT sanitize-flatten
        # it -- that would collapse newlines). The decider screens the OUTPUT SQL (closed-world +
        # execute + rows>0), so a tenant-name injection in the card cannot produce a surviving
        # example -- the same "the screen is the guarantee" model as every other phase.
        user = f"{card_text}\n\nPropose up to {self._n} (question, SQL) pairs as JSON."
        raw = self._client.complete(_example_system_prompt(dialect), user, max_tokens=self._max_tokens)
        return parse_examples(raw)


class FakeExampleGenerator:
    def __init__(self, replies: dict[str, list[dict]] | None = None) -> None:
        self._replies = replies or {}
        self.calls: list[str] = []

    def propose(self, table: str, card_text: str, dialect: str) -> list[ExampleProposal]:
        self.calls.append(table)
        return parse_examples(json.dumps({"examples": self._replies.get(table, [])}))


def _visible_for(snapshot: Snapshot, table: str) -> dict[str, set[str]]:
    by_table: dict[str, set[str]] = {}
    for c in snapshot.columns:
        by_table.setdefault(c.object_id, set()).add(c.name)
    related = {rel.to for rel in snapshot.relationships if rel.from_ == table}
    related |= {rel.from_ for rel in snapshot.relationships if rel.to == table}
    visible = {table: by_table.get(table, set())}
    for r in related:
        if r in by_table:
            visible[r] = by_table[r]
    return visible


def enrich_examples(
    snapshot: Snapshot, generator: ExampleGenerator, adapter, dialect: str = "duckdb",
    per_table: int = 3,
) -> Snapshot:
    """Third LLM phase: verified worked examples. Keeps only decider-approved, executed,
    rows>0 pairs. Fail-soft per table; re-versions."""
    cards = {c.object_id: c.text for c in build_cards(snapshot)}

    kept: list[Example] = []
    for table in cards:
        visible = _visible_for(snapshot, table)
        try:
            proposals = generator.propose(table, cards[table], dialect)
        except Exception:
            continue  # fail-soft
        n = 0
        for p in proposals:
            if n >= per_table:
                break
            verdict = decide(p.sql, visible, adapter=adapter, dialect=dialect, target=dialect)
            if not isinstance(verdict, Approved):
                continue
            try:
                result = run(adapter, verdict.target_sql)
            except ExecutionError:
                continue
            if result.table.num_rows <= 0:
                continue
            kept.append(Example(question=p.question, sql=verdict.plan_sql,
                                tables=verdict.tables, object_id=table))
            n += 1

    enriched = snapshot.model_copy(update={"examples": kept}, deep=True)
    enriched.version = content_version(enriched)
    return enriched

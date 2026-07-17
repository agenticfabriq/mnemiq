from __future__ import annotations

from typing import Protocol

import sqlglot
from sqlglot import exp

from mnemiq.contract import Column, Job, Snapshot, TableFacts
from mnemiq.enrichment.pipeline import content_version
from mnemiq.enrichment.prompts import PERSONA, sanitize
from mnemiq.enrichment.proposals import _clean_text, _extract_json

_MAX_GRAIN = 240
_MAX_GOTCHA = 200
_MAX_GOTCHAS = 6
_MAX_MEASURE_NAME = 80
_MAX_MEASURE_EXPR = 200


def _expr_columns_ok(expr: str, allowed: set[str]) -> bool:
    """A measure expression may reference only real columns (parsed, closed-world)."""
    try:
        tree = sqlglot.parse_one(expr)
    except Exception:
        return False
    return all(c.name in allowed for c in tree.find_all(exp.Column))


def parse_facts(raw: str, table: str, columns: set[str], related: set[str]) -> TableFacts:
    """Screen a facts proposal. Never raises; drops what it cannot verify."""
    payload = _extract_json(raw)
    allowed = columns | related

    gotchas: list[str] = []
    raw_gotchas = payload.get("gotchas")
    if isinstance(raw_gotchas, list):
        for g in raw_gotchas:
            text = _clean_text(g, _MAX_GOTCHA)
            if text is not None:
                gotchas.append(text)
            if len(gotchas) >= _MAX_GOTCHAS:
                break

    measures: dict[str, str] = {}
    raw_measures = payload.get("canonical_measures")
    if isinstance(raw_measures, dict):
        for name, expr in raw_measures.items():
            clean_name = _clean_text(name, _MAX_MEASURE_NAME)
            clean_expr = _clean_text(expr, _MAX_MEASURE_EXPR)
            if clean_name and clean_expr and _expr_columns_ok(clean_expr, allowed):
                measures[clean_name] = clean_expr

    tc = payload.get("default_time_column")
    return TableFacts(
        object_id=table,
        grain=_clean_text(payload.get("grain"), _MAX_GRAIN),
        gotchas=gotchas,
        canonical_measures=measures,
        default_time_column=tc if isinstance(tc, str) and tc in columns else None,
    )


def _facts_system_prompt() -> str:
    return f"""{PERSONA}

You are given a table, its columns (with descriptions), and its relationships. State the
table's structural facts for an analyst who will write queries against it.

CLOSED WORLD -- every column you name in a measure or as the time column MUST be one of the
columns given to you (or a column of a related table for a measure). Never invent a column.

Return ONLY a JSON object, no prose, of exactly this shape:
{{
  "grain": "<one sentence: 'one row per X within Y', or null>",
  "gotchas": ["<a query trap: text-typed numerics, fan-out/pre-aggregate, role-specific keys, PII>", "..."],
  "canonical_measures": {{"<business name>": "<expression over real columns, e.g. sum(amount)>"}},
  "default_time_column": "<the column analysts filter time by, or null>"
}}

Gotchas are the highest-value output: encode traps a schema alone never reveals. A null grain
or time column is better than a guess."""


def _render_context(table: str, columns: list[Column], relationships: list[str]) -> str:
    lines = [f"TABLE: {sanitize(table)}", "COLUMNS:"]
    for c in columns:
        line = f"- {sanitize(c.name)} ({sanitize(c.data_type or 'unknown')})"
        if c.description:
            line += f" {sanitize(c.description, limit=300)}"
        lines.append(line)
    if relationships:
        lines.append("RELATIONSHIPS:")
        lines.extend(relationships)
    return "\n".join(lines)


class FactsEnricher(Protocol):
    def facts(
        self, table: str, context: str, columns: set[str], related: set[str]
    ) -> TableFacts: ...


class LLMFactsEnricher:
    def __init__(self, client, max_tokens: int = 2000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def facts(
        self, table: str, context: str, columns: set[str], related: set[str]
    ) -> TableFacts:
        raw = self._client.complete(_facts_system_prompt(), context, max_tokens=self._max_tokens)
        return parse_facts(raw, table, columns, related)


class FakeFactsEnricher:
    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self._replies = replies or {}
        self.calls: list[str] = []

    def facts(
        self, table: str, context: str, columns: set[str], related: set[str]
    ) -> TableFacts:
        self.calls.append(table)
        return parse_facts(self._replies.get(table, ""), table, columns, related)


def enrich_table_facts(snapshot: Snapshot, enricher: FactsEnricher) -> Snapshot:
    """Second LLM phase: table-level structural facts. Fail-soft per table; re-versions."""
    by_table: dict[str, list[Column]] = {}
    for column in snapshot.columns:
        by_table.setdefault(column.object_id, []).append(column)

    rel_lines: dict[str, list[str]] = {}
    related: dict[str, set[str]] = {}
    for rel in snapshot.relationships:
        keys = ", ".join(f"{k.left} = {k.right}" for k in rel.join_keys)
        rel_lines.setdefault(rel.from_, []).append(
            f"- joins {rel.to} ({rel.cardinality}{': ' + keys if keys else ''})"
        )
        related.setdefault(rel.from_, set()).add(rel.to)
        related.setdefault(rel.to, set()).add(rel.from_)

    facts: list[TableFacts] = []
    jobs: list[Job] = []
    for table, columns in by_table.items():
        col_names = {c.name for c in columns}
        related_cols = {
            c.name for c in snapshot.columns if c.object_id in related.get(table, set())
        }
        try:
            context = _render_context(table, columns, rel_lines.get(table, []))
            tf = enricher.facts(table, context, col_names, related_cols)
        except Exception:
            jobs.append(Job(id=f"facts:{table}", source_id=snapshot.source_id,
                            kind="facts", status="failed"))
            continue
        if tf.grain or tf.gotchas or tf.canonical_measures or tf.default_time_column:
            facts.append(tf)
        jobs.append(Job(id=f"facts:{table}", source_id=snapshot.source_id,
                        kind="facts", status="done"))

    enriched = snapshot.model_copy(
        update={"table_facts": facts, "jobs": [*snapshot.jobs, *jobs]}, deep=True
    )
    enriched.version = content_version(enriched)
    return enriched

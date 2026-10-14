from __future__ import annotations

import re
from collections.abc import Mapping
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




def _fold(name: str) -> str:
    """A name or a fact's text as both sides are compared: case folded, whitespace runs to one."""
    return " ".join(name.split()).lower()


def _vocabulary(table: str, columns: set[str],
                related_tables: Mapping[str, set[str]]) -> dict[str, set[str]]:
    """Each name a fact may use, folded, to the dependencies it may mean: an own column to
    `table.column`, a related table to itself, a related column to `related.column` -- under
    its bare name too, so a bare `plan_id` shared by two tables means both. Meaning more only
    withholds more."""
    vocab: dict[str, set[str]] = {}

    def add(name: str, dep: str) -> None:
        vocab.setdefault(_fold(name), set()).add(dep)

    for column in columns:
        add(column, f"{table}.{column}")
        add(f"{table}.{column}", f"{table}.{column}")
    for other, other_columns in related_tables.items():
        bare = other.rsplit(".", 1)[-1]
        add(other, other)
        add(bare, other)
        for column in other_columns:
            for spelled in (column, f"{other}.{column}", f"{bare}.{column}"):
                add(spelled, f"{other}.{column}")
    return vocab


def _named(text: str, vocab: dict[str, set[str]]) -> set[str]:
    """Every dependency the text names: each known name matched WHOLE, case folded.

    Not by splitting the text into identifier tokens, which read `secret code` as two words and
    matched neither -- a quoted identifier with a space, or with punctuation such as
    `Enrollment (K-12)`, was never found, and the fact went out with an empty list to every
    identity (Codex's review of #130). A name counts where no word character touches either end,
    so `id` is not found in `idle`; a dot before it does not stop it, so a qualified mention still
    names the bare column. Over-matching only withholds more. The boundaries are ASCII, as the old
    tokenizer's were -- a digit before a name or a non-ASCII letter after it does not hide it
    (`2ssn`, `ssné`) -- and whitespace is folded on both sides, as `_clean_text` folds the fact.
    """
    low = _fold(text)
    return {dep for name, deps in vocab.items()
            if name and re.search(r"(?<![A-Za-z_$])" + re.escape(name) + r"(?![A-Za-z0-9_$])", low)
            for dep in deps}


def _declared(names, vocab: dict[str, set[str]]) -> set[str] | None:
    """The model's own list, resolved; None when it is missing or names anything it was not
    given -- an invented name cannot be checked against a grant, so the fact goes unrecorded."""
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return None
    out: set[str] = set()
    for name in names:
        deps = vocab.get(_fold(name))
        if not deps:
            return None
        out |= deps
    return out


def parse_facts(raw: str, table: str, columns: set[str], related: set[str], *,
                related_tables: Mapping[str, set[str]] | None = None) -> TableFacts:
    """Screen a facts proposal. Never raises; drops what it cannot verify.

    `related_tables` is `{related table: its columns}`. With it, each prose fact records what
    it depends on (M143): the columns the model lists for it, checked closed-world, together
    with every column or table its text names. A fact the model gave no valid list for has none
    recorded, and is shown only to an identity that sees its table whole. Without it nothing is
    recorded, since a related column cannot be qualified.
    """
    payload = _extract_json(raw)
    allowed = columns | related
    vocab = _vocabulary(table, columns, related_tables) if related_tables is not None else None
    reads: dict[str, set[str]] = {}
    unrecorded: set[str] = set()

    # Keyed by text, so a grain and a gotcha with the same words share one entry: their lists
    # are unioned, and one that has none recorded leaves the text with none (review gate).
    def record(text: str, names) -> None:
        if vocab is None:
            return
        declared = _declared(names, vocab)
        if declared is None:
            unrecorded.add(text)
        else:
            reads.setdefault(text, set()).update(declared | _named(text, vocab))

    gotchas: list[str] = []
    raw_gotchas = payload.get("gotchas")
    if isinstance(raw_gotchas, list):
        for g in raw_gotchas:
            names = g.get("reads") if isinstance(g, dict) else None
            text = _clean_text(g.get("text") if isinstance(g, dict) else g, _MAX_GOTCHA)
            if text is not None:
                gotchas.append(text)
                record(text, names)
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

    grain = _clean_text(payload.get("grain"), _MAX_GRAIN)
    if grain is not None:
        record(grain, payload.get("grain_reads"))
    tc = payload.get("default_time_column")
    return TableFacts(
        object_id=table,
        grain=grain,
        gotchas=gotchas,
        canonical_measures=measures,
        default_time_column=tc if isinstance(tc, str) and tc in columns else None,
        reads={text: sorted(deps) for text, deps in reads.items() if text not in unrecorded},
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
  "grain_reads": ["<each column or table the grain refers to>"],
  "gotchas": [{{"text": "<a query trap: text-typed numerics, fan-out/pre-aggregate, role-specific keys, PII>",
               "reads": ["<each column or table this trap refers to>"]}}],
  "canonical_measures": {{"<business name>": "<expression over real columns, e.g. sum(amount)>"}},
  "default_time_column": "<the column analysts filter time by, or null>"
}}

Gotchas are the highest-value output: encode traps a schema alone never reveals. A null grain
or time column is better than a guess.

Every "reads" list names, exactly as given to you, each column and related table the sentence
is about -- including one it describes in other words ("the diagnosis code" is about the column
holding it). An empty list means the sentence is about no particular column."""


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
        self, table: str, context: str, columns: set[str], related: set[str], *,
        related_tables: Mapping[str, set[str]] | None = None,
    ) -> TableFacts: ...


class LLMFactsEnricher:
    def __init__(self, client, max_tokens: int = 2000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def facts(
        self, table: str, context: str, columns: set[str], related: set[str], *,
        related_tables: Mapping[str, set[str]] | None = None,
    ) -> TableFacts:
        raw = self._client.complete(_facts_system_prompt(), context, max_tokens=self._max_tokens)
        return parse_facts(raw, table, columns, related, related_tables=related_tables)


class FakeFactsEnricher:
    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self._replies = replies or {}
        self.calls: list[str] = []

    def facts(
        self, table: str, context: str, columns: set[str], related: set[str], *,
        related_tables: Mapping[str, set[str]] | None = None,
    ) -> TableFacts:
        self.calls.append(table)
        return parse_facts(self._replies.get(table, ""), table, columns, related,
                           related_tables=related_tables)


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
        related_tables = {other: {c.name for c in by_table.get(other, [])}
                          for other in related.get(table, set())}
        try:
            context = _render_context(table, columns, rel_lines.get(table, []))
            tf = enricher.facts(table, context, col_names, related_cols,
                                related_tables=related_tables)
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

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from mnemiq.authz.grants import AuthzProvider
from mnemiq.contract import Column, Definition, Example, IdentityContext, TableFacts
from mnemiq.llm.embeddings import Embedder
from mnemiq.semantic.cards import render_facts_block
from mnemiq.semantic.glossary import select_definitions

_RRF_K = 60  # the standard RRF constant (reciprocal-rank blending)


@dataclass
class RetrievedCard:
    object_id: str
    card: str
    score: float


@dataclass(frozen=True)
class ResolvedConcept:
    """A candidate code for the asked question, on a scheme-bound column."""
    column_id: str
    scheme_label: str
    notation: str
    label: str


@dataclass
class ContextPacket:
    question: str
    cards: list[RetrievedCard]
    grant_fingerprint: str
    enrichment_version: str | None
    definitions: list[Definition] = field(default_factory=list)
    concepts: list[ResolvedConcept] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)  # Plan 08 seam; filled by retrieve()


def _rank(rows: list[tuple[str, float]]) -> dict[str, int]:
    return {object_id: rank for rank, (object_id, _score) in enumerate(rows, start=1)}


def _attach_facts(cards: list[RetrievedCard], table_facts: Sequence[TableFacts]) -> None:
    """Insert each retrieved table's structural-facts block right after its TABLE line. Facts
    live here, NOT in the indexed card, so they never perturb retrieval top-k."""
    by_id = {tf.object_id: tf for tf in table_facts}
    for c in cards:
        tf = by_id.get(c.object_id)
        block = render_facts_block(tf) if tf else ""
        if not block:
            continue
        head, _, rest = c.card.partition("\n")
        c.card = f"{head}\n{block}\n{rest}" if rest else f"{head}\n{block}"


def _retrieve_examples(con, embedding, allowed: set[str], k: int = 5) -> list[Example]:
    """Top-k validated examples by QUESTION similarity to the asked question -- on-target
    few-shot, not per-table. Never surfaces an example touching a table outside the grants."""
    placeholders = ", ".join("?" for _ in allowed)
    try:
        rows = con.execute(
            "SELECT question, sql, tables, object_id, "
            f"array_cosine_similarity(embedding, ?::FLOAT[{len(embedding)}]) AS s "
            f"FROM example WHERE object_id IN ({placeholders}) ORDER BY s DESC LIMIT ?",
            [embedding, *allowed, k * 3],
        ).fetchall()
    except Exception:
        return []  # no example index built for this store
    out: list[Example] = []
    for question, sql, tables_json, object_id, _s in rows:
        tables = json.loads(tables_json)
        if set(tables) <= allowed:  # an example must not reveal a forbidden table
            out.append(Example(question=question, sql=sql, tables=tables, object_id=object_id))
        if len(out) >= k:
            break
    return out


_CONCEPT_CAP = 15  # total across all columns: a prompt block, not a data dump


def _resolve_concepts(question, columns, shown: set[str], index) -> list[ResolvedConcept]:
    """Candidate codes for the question, for scheme-bound columns on RETRIEVED tables only.

    Grant safety is inherited, not re-implemented: `shown` comes from cards that were already
    scoped to the identity's grants before anything was ranked.
    """
    out: list[ResolvedConcept] = []
    for column in columns:
        if column.object_id not in shown or column.code_scheme is None:
            continue
        for notation, label in index.nearest(column.code_scheme.id, question):
            out.append(ResolvedConcept(column.id, column.code_scheme.label, notation, label))
            if len(out) >= _CONCEPT_CAP:
                return out
    return out


def retrieve(
    con: duckdb.DuckDBPyConnection,
    question: str,
    identity: IdentityContext,
    authz: AuthzProvider,
    embedder: Embedder,
    k: int = 5,
    definitions: Sequence[Definition] = (),
    table_facts: Sequence[TableFacts] = (),
    columns: Sequence[Column] = (),
    ontology_index=None,
) -> ContextPacket:
    """Hybrid retrieval, scoped to the identity's grants *before* anything is ranked.

    An object the identity may not access is never scored, never ranked, never counted and
    never returned: the model cannot be tempted by a table it was never shown, and no packet
    can disclose that the table exists. Metadata is itself confidential -- a table named
    `layoff_plans` discloses by its mere existence.
    """
    grants = authz.grants_for(identity)
    packet = ContextPacket(
        question=question,
        cards=[],
        grant_fingerprint=grants.fingerprint,
        enrichment_version=None,
    )
    if not grants.objects:
        return packet  # nothing is visible: do not even touch the index

    packet.definitions = select_definitions(question, definitions, grants)

    allowed = list(grants.objects)
    placeholders = ", ".join("?" for _ in allowed)

    lexical = con.execute(
        f"""
        SELECT object_id, score FROM (
          SELECT object_id,
                 fts_main_semantic_object.match_bm25(object_id, ?) AS score
          FROM semantic_object
          WHERE object_id IN ({placeholders})
        ) WHERE score IS NOT NULL
        ORDER BY score DESC LIMIT ?
        """,
        [question, *allowed, k],
    ).fetchall()

    (embedding,) = embedder.embed([question])
    semantic = con.execute(
        f"""
        SELECT object_id, array_cosine_similarity(embedding, ?::FLOAT[{len(embedding)}]) AS score
        FROM semantic_object
        WHERE object_id IN ({placeholders})
        ORDER BY score DESC LIMIT ?
        """,
        [embedding, *allowed, k],
    ).fetchall()

    lexical_rank, semantic_rank = _rank(lexical), _rank(semantic)
    fused: dict[str, float] = {}
    for object_id in {*lexical_rank, *semantic_rank}:
        score = 0.0
        if object_id in lexical_rank:
            score += 1.0 / (_RRF_K + lexical_rank[object_id])
        if object_id in semantic_rank:
            score += 1.0 / (_RRF_K + semantic_rank[object_id])
        fused[object_id] = score

    top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    if not top:
        return packet

    rows = con.execute(
        "SELECT object_id, card, version FROM semantic_object WHERE object_id IN "
        f"({', '.join('?' for _ in top)})",
        [object_id for object_id, _ in top],
    ).fetchall()
    cards = {object_id: (card, version) for object_id, card, version in rows}

    packet.cards = [
        RetrievedCard(object_id=object_id, card=cards[object_id][0], score=score)
        for object_id, score in top
        if object_id in cards
    ]
    packet.enrichment_version = next(iter(cards.values()))[1] if cards else None
    _attach_facts(packet.cards, table_facts)
    packet.examples = _retrieve_examples(con, embedding, set(grants.objects), k=k)
    if ontology_index is not None and packet.cards:
        packet.concepts = _resolve_concepts(
            question, columns, {c.object_id for c in packet.cards}, ontology_index
        )
    return packet

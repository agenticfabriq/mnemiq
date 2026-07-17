from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from mnemiq.authz.grants import AuthzProvider
from mnemiq.contract import Definition, Example, IdentityContext
from mnemiq.llm.embeddings import Embedder
from mnemiq.semantic.glossary import select_definitions

_RRF_K = 60  # the standard RRF constant (reciprocal-rank blending)


@dataclass
class RetrievedCard:
    object_id: str
    card: str
    score: float


@dataclass
class ContextPacket:
    question: str
    cards: list[RetrievedCard]
    grant_fingerprint: str
    enrichment_version: str | None
    definitions: list[Definition] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)  # Plan 08 seam; filled by retrieve()


def _rank(rows: list[tuple[str, float]]) -> dict[str, int]:
    return {object_id: rank for rank, (object_id, _score) in enumerate(rows, start=1)}


def _attach_examples(packet: ContextPacket, examples, cap: int = 5) -> None:
    """Attach validated examples for the tables actually retrieved (grounds few-shot on the
    tables in play). Stable order: retrieval rank, then question."""
    if not examples:
        return
    ranked = {c.object_id: i for i, c in enumerate(packet.cards)}
    hits = [e for e in examples if e.object_id in ranked]
    hits.sort(key=lambda e: (ranked[e.object_id], e.question))
    packet.examples = hits[:cap]


def retrieve(
    con: duckdb.DuckDBPyConnection,
    question: str,
    identity: IdentityContext,
    authz: AuthzProvider,
    embedder: Embedder,
    k: int = 5,
    definitions: Sequence[Definition] = (),
    examples: Sequence[Example] = (),
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
    _attach_examples(packet, examples)
    return packet

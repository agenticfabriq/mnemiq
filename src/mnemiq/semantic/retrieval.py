from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from mnemiq.authz.grants import AuthzProvider
from mnemiq.contract import (
    Column, Definition, Dimension, Example, HistoryTurn, IdentityContext, Metric,
    TableFacts,
)
from mnemiq.llm.embeddings import Embedder
from mnemiq.semantic.cards import render_facts_block
from mnemiq.semantic.glossary import select_definitions
from mnemiq.semantic.measures import select_dimensions, select_metrics

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
    # Certified metrics and dimensions over the tables in `cards`. They rode in the snapshot and
    # were read by nothing until this; see `semantic.measures` for why they are selected by table
    # rather than by the words of the question.
    metrics: list[Metric] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    concepts: list[ResolvedConcept] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)  # Plan 08 seam; filled by retrieve()
    # Prior turns, already scoped to this identity's authorization boundary by the caller.
    history: list[HistoryTurn] = field(default_factory=list)


def _rank(rows: list[tuple[str, float]]) -> dict[str, int]:
    return {object_id: rank for rank, (object_id, _score) in enumerate(rows, start=1)}


def _attach_facts(cards: list[RetrievedCard], table_facts: Sequence[TableFacts],
                  card_style: str = "cards") -> None:
    """Attach each retrieved table's structural-facts block. Facts live here, NOT in the
    indexed card, so they never perturb retrieval top-k.

    Where the block goes depends on the card form. Splicing after the first line is right
    for `TABLE x` and WRONG for `CREATE TABLE x (` -- it drops `GRAIN:`/`GOTCHAS:` between
    the header and the first column and hands the generator malformed DDL, which is the one
    thing the ddl form exists to avoid. In ddl style the block follows the closing paren,
    commented, so it is legal SQL either way.
    """
    by_id = {tf.object_id: tf for tf in table_facts}
    for c in cards:
        tf = by_id.get(c.object_id)
        block = render_facts_block(tf) if tf else ""
        if not block:
            continue
        if card_style == "ddl":
            commented = "\n".join(f"-- {line}" if line else "--" for line in block.split("\n"))
            c.card = f"{c.card}\n{commented}"
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
    metrics: Sequence[Metric] = (),
    dimensions: Sequence[Dimension] = (),
    table_facts: Sequence[TableFacts] = (),
    columns: Sequence[Column] = (),
    ontology_index=None,
    snapshot=None,
    card_style: str = "cards",
) -> ContextPacket:
    """Hybrid retrieval, scoped to the identity's grants *before* anything is ranked.

    An object the identity may not access is never scored, never ranked, never counted and
    never returned: the model cannot be tempted by a table it was never shown, and no packet
    can disclose that the table exists. Metadata is itself confidential -- a table named
    `layoff_plans` discloses by its mere existence.

    **That was true of tables and false of columns until M4.** The stored card is rendered once at
    enrich time from the whole snapshot, so a granted table's card carried every column, its
    pii_level and its harvested coded values. Given `snapshot`, the cards are re-rendered against
    the identity's column policy before they go in the packet. Ranking still runs on the stored,
    identity-independent card -- an index per grant set is combinatorial, and ranking is not a
    channel: nothing about a hidden column reaches the caller.
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

    scoped: dict[str, str] = {}
    if snapshot is not None:
        from mnemiq.semantic.cards import build_cards
        from mnemiq.sql.policy import build_access_policy

        # `card_style` reaches only THIS re-render, the identity-scoped copy handed to the
        # generator. `build_index` keeps the default, so the embedded card stays the form
        # top-k was measured on and retrieval is untouched by the generator's preference.
        scoped = {
            card.object_id: card.text
            for card in build_cards(
                snapshot, policy=build_access_policy(snapshot, grants), style=card_style)
        }
    packet.cards = [
        RetrievedCard(
            object_id=object_id,
            card=scoped.get(object_id, cards[object_id][0]),
            score=score,
        )
        for object_id, score in top
        if object_id in cards
    ]
    packet.enrichment_version = next(iter(cards.values()))[1] if cards else None
    _attach_facts(packet.cards, table_facts, card_style)
    # After the cards are chosen, because a certified measure rides with its table -- see
    # `semantic.measures` for why that rule differs from the glossary's word-matching one.
    table_ids = [c.object_id for c in packet.cards]
    # After the cards, not before: a bound definition rides with its table, so which tables were
    # retrieved has to be known first. It used to be selected before anything was ranked.
    packet.definitions = select_definitions(question, definitions, grants, table_ids)
    packet.metrics = select_metrics(table_ids, metrics, grants)
    packet.dimensions = select_dimensions(table_ids, dimensions, grants)
    packet.examples = _retrieve_examples(con, embedding, set(grants.objects), k=k)
    if ontology_index is not None and packet.cards:
        packet.concepts = _resolve_concepts(
            question, columns, {c.object_id for c in packet.cards}, ontology_index
        )
    return packet

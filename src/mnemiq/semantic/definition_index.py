from __future__ import annotations

import logging

import duckdb

from mnemiq.contract import Snapshot
from mnemiq.llm.embeddings import EMBED_DIM
from mnemiq.ontology.records import OntologyRecords

logger = logging.getLogger(__name__)

DEFINITION_INDEX_MAX_CONCEPTS = 500  # a scheme above this contributes only its scheme-level text

_DDL = f"""
CREATE TABLE IF NOT EXISTS definition_concept (
  source_id TEXT,
  object_id TEXT,
  term      TEXT,
  text      TEXT,
  embedding FLOAT[{EMBED_DIM}]
)
"""


def _corpus(records: OntologyRecords, snapshot: Snapshot,
            max_concepts: int) -> list[tuple[str, str, str]]:
    """(object_id, term, text) rows: the certified definitions plus per-concept text for in-scale
    schemes. An oversized scheme is skipped so a huge standard code system cannot trigger tens of
    thousands of embedding calls."""
    rows: list[tuple[str, str, str]] = []
    for d in snapshot.definitions:
        rows.append((d.id, d.term, f"{d.term}: {d.definition}"))
    for scheme in records.schemes:
        if len(scheme.concepts) > max_concepts:
            continue
        for c in scheme.concepts:
            text = f"{c.pref_label}: {c.definition}" if c.definition else c.pref_label
            rows.append((c.id, c.pref_label, text))
    return rows


def build_definition_index(records: OntologyRecords, snapshot: Snapshot,
                           con: duckdb.DuckDBPyConnection, embedder,
                           max_concepts: int = DEFINITION_INDEX_MAX_CONCEPTS) -> int:
    """Embed the local meaning corpus into `con` for enrich-time grounding. Delete-then-insert per
    source. Fail-soft: no embedder, empty corpus, or an embed error writes nothing and returns 0."""
    con.execute(_DDL)
    con.execute("DELETE FROM definition_concept WHERE source_id = ?", [snapshot.source_id])
    rows = _corpus(records, snapshot, max_concepts)
    if not rows or embedder is None:
        return 0
    try:
        vectors = embedder.embed([text for _oid, _term, text in rows])
    except Exception as exc:  # degrade-to-local: grounding is optional, never fatal
        logger.warning("definition index embedding failed; grounding skipped: %s", exc)
        return 0
    con.executemany(
        "INSERT INTO definition_concept (source_id, object_id, term, text, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        [[snapshot.source_id, oid, term, text, vec]
         for (oid, term, text), vec in zip(rows, vectors)],
    )
    return len(rows)


class DefinitionIndex:
    """Read side. Ensures its table on construction so a store built before SP5b answers
    'nothing indexed' rather than raising."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con
        con.execute(_DDL)

    def nearest(self, query_text: str, embedder, k: int = 5,
                floor: float = 0.0) -> list[tuple[str, str, float]]:
        """Top-k (term, text, score) by cosine of the query embedding to each indexed row.
        Fail-soft: no embedder or an embed error returns []."""
        if embedder is None:
            return []
        try:
            (embedding,) = embedder.embed([query_text])
            rows = self._con.execute(
                f"SELECT term, text, "
                f"array_cosine_similarity(embedding, ?::FLOAT[{len(embedding)}]) AS score "
                f"FROM definition_concept ORDER BY score DESC LIMIT ?",
                [embedding, k],
            ).fetchall()
        except Exception as exc:  # embed error or an unavailable array function -> no grounding
            logger.warning("definition index query failed; grounding skipped: %s", exc)
            return []
        return [(term, text, float(score)) for term, text, score in rows if score is not None
                and float(score) >= floor]


class DefinitionRetriever:
    """Enrich-time grounding: turn a table + its columns into a REFERENCE block of nearby certified
    meaning. The query is deterministic structure (table name + column names)."""

    def __init__(self, index: DefinitionIndex, embedder, k: int = 5, floor: float = -1.0) -> None:
        # floor=-1.0 admits everything: grounding surfaces the top-k nearest reference (bounded by
        # k, advisory "use when they clearly apply"), rather than dropping all of it when the
        # nearest match sits below an arbitrary positive threshold. A tuned floor is a later knob.
        self._index = index
        self._embedder = embedder
        self._k = k
        self._floor = floor

    def grounding_for(self, table: str, columns) -> str:
        from mnemiq.enrichment.prompts import render_grounding_block

        query = " ".join([table, *(c.name for c in columns)])
        items = self._index.nearest(query, self._embedder, k=self._k, floor=self._floor)
        return render_grounding_block(items)

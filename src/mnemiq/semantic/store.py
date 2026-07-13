from __future__ import annotations

import duckdb

from mnemiq.contract import Snapshot
from mnemiq.llm.embeddings import EMBED_DIM, Embedder
from mnemiq.semantic.cards import build_cards

_DDL = f"""
CREATE TABLE IF NOT EXISTS semantic_object (
  object_id TEXT,
  source_id TEXT,
  version   TEXT,
  card      TEXT,
  embedding FLOAT[{EMBED_DIM}]
)
"""


def build_index(con: duckdb.DuckDBPyConnection, snapshot: Snapshot, embedder: Embedder) -> int:
    """Render, embed and index the snapshot's tables. One live index per source."""
    cards = build_cards(snapshot)
    con.execute(_DDL)

    # One live index per source: a stale card is worse than a missing one, because at
    # retrieval time it is indistinguishable from a fresh one.
    con.execute("DELETE FROM semantic_object WHERE source_id = ?", [snapshot.source_id])
    if not cards:
        return 0

    vectors = embedder.embed([c.text for c in cards])
    con.executemany(
        "INSERT INTO semantic_object (object_id, source_id, version, card, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            [c.object_id, snapshot.source_id, snapshot.version, c.text, v]
            for c, v in zip(cards, vectors, strict=True)
        ],
    )

    # FTS indexes a point-in-time copy of the table: build it *after* the rows land, and
    # overwrite the previous one.
    con.execute("PRAGMA create_fts_index('semantic_object', 'object_id', 'card', overwrite=1)")

    # HNSW on a persistent database is gated behind an experimental flag; without it the
    # CREATE INDEX below raises.
    con.execute("SET hnsw_enable_experimental_persistence = true")
    con.execute("DROP INDEX IF EXISTS semantic_object_hnsw")
    con.execute(
        "CREATE INDEX semantic_object_hnsw ON semantic_object "
        "USING HNSW (embedding) WITH (metric = 'cosine')"
    )
    return len(cards)


def indexed_version(con: duckdb.DuckDBPyConnection, source_id: str) -> str | None:
    con.execute(_DDL)
    row = con.execute(
        "SELECT DISTINCT version FROM semantic_object WHERE source_id = ?", [source_id]
    ).fetchone()
    return row[0] if row else None

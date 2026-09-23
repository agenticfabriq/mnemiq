from __future__ import annotations

import json

import duckdb

from mnemiq.contract import Snapshot
from mnemiq.llm.embeddings import Embedder
from mnemiq.semantic.cards import build_cards
from mnemiq.semantic.embedding_width import refuse_if_mismatched


def _ddl(dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS semantic_object (
  object_id TEXT,
  source_id TEXT,
  version   TEXT,
  card      TEXT,
  embedding FLOAT[{dim}]
)
"""


def _example_ddl(dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS example (
  question   TEXT,
  sql        TEXT,
  tables     TEXT,
  object_id  TEXT,
  source_id  TEXT,
  embedding  FLOAT[{dim}]
)
"""


def build_index(con: duckdb.DuckDBPyConnection, snapshot: Snapshot, embedder: Embedder) -> int:
    """Render, embed and index the snapshot's tables. One live index per source: this source's
    existing rows are cleared whenever the function runs, even when the new build is empty, so a
    stale card never outlives the build that was meant to refresh it -- at retrieval time a stale
    card is indistinguishable from a fresh one.

    Checked for empty input before anything else touches `embedder`: on LLMEmbedder, `.dim`
    itself makes an embedding call to probe the endpoint, and a snapshot with nothing new to
    write should never cost a network round trip for that -- the case that matters most in
    exactly the air-gapped deployment this width-from-the-embedder design is for. The clearing
    DELETE below needs no such width, so it still runs on this path.

    Sizes the embedding column from embedder.dim, but only on first create: CREATE TABLE IF NOT
    EXISTS is a no-op against a store already built at a different width, so a mismatch here is
    refused (see refuse_if_mismatched) before the clearing DELETE, rather than letting the DELETE
    run and then crashing on the INSERT.
    """
    def _clear_stale() -> None:
        try:
            con.execute("DELETE FROM semantic_object WHERE source_id = ?", [snapshot.source_id])
        except duckdb.CatalogException:
            pass  # nothing built yet for any source -- nothing to clear

    cards = build_cards(snapshot)
    if not cards:
        _clear_stale()
        return 0

    dim = embedder.dim
    refuse_if_mismatched(con, "semantic_object", dim)
    _clear_stale()
    con.execute(_ddl(dim))

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


def build_example_index(
    con: duckdb.DuckDBPyConnection, snapshot: Snapshot, embedder: Embedder
) -> int:
    """Embed each validated example's QUESTION into its own index, so retrieval pulls the
    examples most similar to the asked question -- not whatever tables happened to rank.

    Kept out of semantic_object so example text never perturbs table retrieval. Same clearing,
    empty-check-first, and width-mismatch-refused-before-clearing contract as build_index, above
    -- see its docstring for the reasoning.
    """
    def _clear_stale() -> None:
        try:
            con.execute("DELETE FROM example WHERE source_id = ?", [snapshot.source_id])
        except duckdb.CatalogException:
            pass  # nothing built yet for any source -- nothing to clear

    if not snapshot.examples:
        _clear_stale()
        return 0

    dim = embedder.dim
    refuse_if_mismatched(con, "example", dim)
    _clear_stale()
    con.execute(_example_ddl(dim))

    vectors = embedder.embed([e.question for e in snapshot.examples])
    con.executemany(
        "INSERT INTO example (question, sql, tables, object_id, source_id, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            [e.question, e.sql, json.dumps(e.tables), e.object_id, snapshot.source_id, v]
            for e, v in zip(snapshot.examples, vectors, strict=True)
        ],
    )
    return len(snapshot.examples)


def indexed_version(con: duckdb.DuckDBPyConnection, source_id: str) -> str | None:
    # No embedder here to size a column with, and none needed -- this only reads. Guessing a
    # width to stand the table up on would risk fixing it at the wrong size before build_index
    # ever runs with the real embedder, so a store that hasn't been built yet is just "nothing
    # indexed" rather than a table created on a guess.
    try:
        row = con.execute(
            "SELECT DISTINCT version FROM semantic_object WHERE source_id = ?", [source_id]
        ).fetchone()
    except duckdb.CatalogException:
        return None
    return row[0] if row else None

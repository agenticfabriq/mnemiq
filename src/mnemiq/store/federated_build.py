from __future__ import annotations

import duckdb

from mnemiq.config import Settings, SourceSpec
from mnemiq.contract import Snapshot
from mnemiq.llm.embeddings import Embedder
from mnemiq.semantic.cards import build_cards
from mnemiq.semantic.embedding_width import refuse_if_mismatched
from mnemiq.semantic.federation import merge_snapshots
from mnemiq.semantic.store import build_example_index, build_index
from mnemiq.store.snapshot_store import current_version, load_snapshot


def build_federated_snapshot(
    con: duckdb.DuckDBPyConnection,
    pairs: list[tuple[SourceSpec, Snapshot]],
    embedder: Embedder,
) -> tuple[int, int]:
    """Merge N per-source snapshots into one qualified snapshot and index it once. Also re-key
    value_index.object_id to the qualified form so value grounding still resolves."""
    fed = merge_snapshots(pairs)
    # Refuse a width mismatch BEFORE the per-source DELETE below, not after: that DELETE runs
    # and autocommits on its own, one statement per table, so a refusal raised only once
    # build_index/build_example_index reach their OWN width check would already have emptied
    # every per-source card and example. That is precisely the failure this branch removed from
    # the single-source path (see build_index) -- it was only ever fixed one call site lower,
    # never here, where the federated rebuild has its own DELETE ahead of both of them.
    #
    # Gated on there being something to embed, matching build_index/build_example_index's own
    # rule that embedder.dim (a network probe on LLMEmbedder) is never touched for empty input.
    # `build_cards(fed)`, not `fed.columns`: build_cards makes one card per source_bindings entry
    # and falls back to columns only when there are no bindings, so gating on columns alone left
    # a federated snapshot with bindings but no columns skipping this check while build_index
    # still produced cards for it -- reopening the exact data-loss window this fix closes, just
    # one binding shape narrower.
    if build_cards(fed):
        refuse_if_mismatched(con, "semantic_object", embedder.dim)
    if fed.examples:
        refuse_if_mismatched(con, "example", embedder.dim)
    # Clear any stale per-source rows so the federated index is the only live one -- and so the
    # FTS rebuild inside build_index never sees two sources' bare (colliding) object_ids. The
    # tables may not exist yet on a fresh store, in which case there is nothing stale to clear.
    for tbl in ("semantic_object", "example"):
        try:
            con.execute(f"DELETE FROM {tbl} WHERE source_id <> ?", [fed.source_id])
        except duckdb.CatalogException:
            pass
    n = build_index(con, fed, embedder)
    n_ex = build_example_index(con, fed, embedder)
    _requalify_value_index(con, pairs)
    return n, n_ex


def _requalify_value_index(con, pairs: list[tuple[SourceSpec, Snapshot]]) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS value_index ("
        "  source_id TEXT, object_id TEXT, column_name TEXT, value_text TEXT)"
    )
    for spec, snap in pairs:
        con.execute(
            "UPDATE value_index SET object_id = ? || '.' || object_id, source_id = 'federated' "
            "WHERE source_id = ? AND object_id NOT LIKE ?",
            [spec.catalog, snap.source_id, f"{spec.catalog}.%"],
        )


def build_federated(settings: Settings, con, embedder: Embedder) -> tuple[int, int]:
    pairs: list[tuple[SourceSpec, Snapshot]] = []
    for spec in settings.source_specs():
        version = current_version(con, spec.id)
        if version is None:
            raise RuntimeError(f"no snapshot for source {spec.id!r} -- run `mnemiq enrich` first")
        pairs.append((spec, load_snapshot(con, version)))
    return build_federated_snapshot(con, pairs, embedder)

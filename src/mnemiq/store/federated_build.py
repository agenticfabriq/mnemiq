from __future__ import annotations

import duckdb

from mnemiq.config import Settings, SourceSpec
from mnemiq.contract import Snapshot
from mnemiq.llm.embeddings import Embedder
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

from __future__ import annotations

import hashlib

from mnemiq.config import SourceSpec
from mnemiq.contract import Snapshot


def qualify_object_id(catalog: str, object_id: str) -> str:
    return f"{catalog}.{object_id}"


class FederatedSnapshot(Snapshot):
    """A merged Snapshot whose object_ids are catalog-qualified, carrying the catalog->schema
    registry the decider needs. Internal to the engine -- NEVER serialized to mnemiq-contract."""

    registry: dict[str, str] = {}


def _composite_version(pairs: list[tuple[SourceSpec, Snapshot]]) -> str:
    parts = sorted(f"{spec.catalog}:{snap.source_id}:{snap.version}" for spec, snap in pairs)
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return f"fed-{digest}"


def merge_snapshots(pairs: list[tuple[SourceSpec, Snapshot]]) -> FederatedSnapshot:
    """Union N per-source snapshots into one qualified FederatedSnapshot. object_ids (and the
    table refs inside examples) are prefixed with each source's catalog so nothing collides in
    a single shared index."""
    columns, table_facts, examples, bindings, definitions = [], [], [], [], []
    for spec, snap in pairs:
        cat = spec.catalog
        for c in snap.columns:
            columns.append(c.model_copy(update={"object_id": qualify_object_id(cat, c.object_id)}))
        for tf in snap.table_facts:
            table_facts.append(
                tf.model_copy(update={"object_id": qualify_object_id(cat, tf.object_id)})
            )
        for e in snap.examples:
            examples.append(e.model_copy(update={
                "object_id": qualify_object_id(cat, e.object_id),
                "tables": [qualify_object_id(cat, t) for t in e.tables],
            }))
        for sb in snap.source_bindings:
            bindings.append(
                sb.model_copy(update={"object_id": qualify_object_id(cat, sb.object_id)})
            )
        for d in snap.definitions:
            definitions.append(d.model_copy(update={
                "bound_objects": [qualify_object_id(cat, o) for o in d.bound_objects]}))
    return FederatedSnapshot(
        version=_composite_version(pairs),
        source_id="federated",
        created_at=max((snap.created_at for _, snap in pairs), default="t"),
        columns=columns, table_facts=table_facts, examples=examples,
        source_bindings=bindings, definitions=definitions,
        registry={spec.catalog: spec.schema for spec, _ in pairs},
    )

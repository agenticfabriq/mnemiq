from __future__ import annotations

from mnemiq.contract import CodedValue, Column, Job, Snapshot
from mnemiq.enrichment.enricher import Enricher
from mnemiq.enrichment.pipeline import content_version
from mnemiq.enrichment.prompts import ColumnFacts
from mnemiq.enrichment.proposals import ColumnAnnotation

_SENSITIVE = {"pii", "phi"}
# A measurement is not a controlled vocabulary. A small sample makes one look like the other,
# and "0.7 means the value 0.7" is noise that would compete with real signal in the index.
_MEASURES = {"ratio", "amount", "count"}


def _facts(
    columns: list[Column], fk_map: dict[tuple[str, str], str] | None = None
) -> list[ColumnFacts]:
    # Only what the snapshot actually knows; absent counts stay absent -- a fabricated
    # zero would poison a closed-world prompt.
    fk_map = fk_map or {}
    return [
        ColumnFacts(
            name=c.name,
            data_type=c.data_type or "unknown",
            codes=[cv.code for cv in c.coded_values],
            row_count=c.row_count,
            distinct_count=c.distinct_count,
            null_count=c.null_count,
            foreign_key=fk_map.get((c.object_id, c.name)),
        )
        for c in columns
    ]


def _annotated(column: Column, annotation: ColumnAnnotation) -> Column:
    """A new Column carrying the annotation. Grounded coded_values win; the LLM only adds
    description/semantic_type/pii_level -- it no longer proposes code meanings (those are
    grounded from data or the operator dictionary in ground_codes, or left bare)."""
    if annotation.pii_level in _SENSITIVE or annotation.semantic_type in _MEASURES:
        # pii/phi: the model recognized personal data in a column whose name did not give it
        #   away. Its observed values ARE that personal data -- drop them rather than carry
        #   real people in the snapshot. Profiling is the first line of defense (catalog.py).
        # measures: the observed values are samples, not a vocabulary.
        coded_values: list[CodedValue] = []
    else:
        # Preserve grounded meanings + provenance; annotation.code_meanings is ignored.
        coded_values = column.coded_values

    return column.model_copy(
        update={
            "description": annotation.description,
            "semantic_type": annotation.semantic_type,
            "pii_level": annotation.pii_level,
            "coded_values": coded_values,
        }
    )


def enrich_semantic(
    snapshot: Snapshot, enricher: Enricher, protected: frozenset[str] = frozenset(),
    retriever=None,
) -> Snapshot:
    """The only LLM phase: give the structural facts their business meaning.

    The enricher proposes; this function decides. Annotations are merged onto a *copy* of the
    snapshot -- structural facts (types, codes, relationships) are annotated, never
    overwritten -- and the result is re-versioned.
    """
    by_table: dict[str, list[Column]] = {}
    for column in snapshot.columns:
        by_table.setdefault(column.object_id, []).append(column)

    # (from_table, from_col) -> "to_table.to_col", so the enricher describes FK columns right
    fk_map: dict[tuple[str, str], str] = {}
    for rel in snapshot.relationships:
        for jk in rel.join_keys:
            fk_map[(rel.from_, jk.left)] = f"{rel.to}.{jk.right}"

    annotated: dict[str, Column] = {}
    jobs: list[Job] = []
    for table, columns in by_table.items():
        grounding = ""
        if retriever is not None:
            try:
                grounding = retriever.grounding_for(table, columns)
            except Exception:
                grounding = ""  # degrade-to-local: grounding is optional, never fatal
        try:
            annotation = enricher.annotate(table, _facts(columns, fk_map), grounding)
        except Exception:
            annotation = None  # fail-soft: a rate-limit costs us a table, not the run

        if annotation is None or not annotation.columns:
            jobs.append(
                Job(
                    id=f"semantic:{table}",
                    source_id=snapshot.source_id,
                    kind="semantic",
                    status="failed",
                )
            )
            continue

        by_name = {a.name: a for a in annotation.columns}
        for column in columns:
            if column.id in protected:
                continue  # certified meaning is authoritative; the LLM never clobbers it
            if column.name in by_name:
                annotated[column.id] = _annotated(column, by_name[column.name])
        jobs.append(
            Job(
                id=f"semantic:{table}",
                source_id=snapshot.source_id,
                kind="semantic",
                status="done",
            )
        )

    enriched = snapshot.model_copy(
        update={
            "columns": [annotated.get(c.id, c) for c in snapshot.columns],
            "jobs": [*snapshot.jobs, *jobs],
        },
        deep=True,
    )
    enriched.version = content_version(enriched)
    return enriched

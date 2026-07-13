from __future__ import annotations

from mnemiq.contract import CodedValue, Column, Job, Snapshot
from mnemiq.enrichment.enricher import Enricher
from mnemiq.enrichment.pipeline import content_version
from mnemiq.enrichment.prompts import ColumnFacts
from mnemiq.enrichment.proposals import ColumnAnnotation

_SENSITIVE = {"pii", "phi"}


def _facts(columns: list[Column]) -> list[ColumnFacts]:
    # Only what the snapshot actually knows. The counts live in the source, not here, and a
    # fabricated zero would poison a closed-world prompt.
    return [
        ColumnFacts(
            name=c.name,
            data_type=c.data_type or "unknown",
            codes=[cv.code for cv in c.coded_values],
        )
        for c in columns
    ]


def _annotated(column: Column, annotation: ColumnAnnotation) -> Column:
    """A new Column carrying the annotation. Structural facts win; the LLM only fills blanks."""
    if annotation.pii_level in _SENSITIVE:
        # The model recognized personal data in a column whose name did not give it away.
        # Its observed values ARE that personal data: drop them rather than carry real people
        # in the snapshot. The second line of defense -- profiling is the first (catalog.py).
        coded_values: list[CodedValue] = []
    else:
        meanings = annotation.code_meanings
        coded_values = [
            CodedValue(code=cv.code, meaning=meanings.get(cv.code)) for cv in column.coded_values
        ]

    return column.model_copy(
        update={
            "description": annotation.description,
            "semantic_type": annotation.semantic_type,
            "pii_level": annotation.pii_level,
            "coded_values": coded_values,
        }
    )


def enrich_semantic(snapshot: Snapshot, enricher: Enricher) -> Snapshot:
    """The only LLM phase: give the structural facts their business meaning.

    The enricher proposes; this function decides. Annotations are merged onto a *copy* of the
    snapshot -- structural facts (types, codes, relationships) are annotated, never
    overwritten -- and the result is re-versioned.
    """
    by_table: dict[str, list[Column]] = {}
    for column in snapshot.columns:
        by_table.setdefault(column.object_id, []).append(column)

    annotated: dict[str, Column] = {}
    jobs: list[Job] = []
    for table, columns in by_table.items():
        try:
            annotation = enricher.annotate(table, _facts(columns))
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

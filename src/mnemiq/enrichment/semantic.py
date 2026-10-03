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
            failed_call = ""
        except Exception as exc:
            # Fail-soft: a rate-limit costs us a table, not the run. The type only: a provider's
            # exception text can carry hosts and keys, and the job outlives the run.
            annotation, failed_call = None, f"the call failed: {type(exc).__name__}"

        if annotation is None or not annotation.columns:
            # The cause on the job (M112): an undescribed table used to be a bare `failed` that
            # nothing read, so a run that described none of the wide tables read as a success.
            jobs.append(
                Job(
                    id=f"semantic:{table}",
                    source_id=snapshot.source_id,
                    kind="semantic",
                    status="failed",
                    detail=failed_call or "; ".join(annotation.failures if annotation else [])
                    or "no column was described",
                )
            )
            continue

        by_name = {a.name: a for a in annotation.columns}
        for column in columns:
            if column.id in protected:
                continue  # certified meaning is authoritative; the LLM never clobbers it
            if column.name in by_name:
                annotated[column.id] = _annotated(column, by_name[column.name])
        # Done, but possibly only partly: a chunk can fail, or the model can skip columns it was
        # asked about. Counted from what LANDED -- a description on an unprotected column -- so a
        # proposal with a blank description, or one for a certified column, is not counted. The
        # job says how much and why, as `detail` on a done job.
        askable = [c for c in columns if c.id not in protected]
        landed = sum(1 for c in askable if c.id in annotated and annotated[c.id].description)
        partial = None
        if landed < len(askable):
            partial = "; ".join([f"{landed} of {len(askable)} columns described",
                                 *annotation.failures])
        jobs.append(
            Job(
                id=f"semantic:{table}",
                source_id=snapshot.source_id,
                kind="semantic",
                status="done",
                detail=partial,
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


def semantic_warnings(snapshot: Snapshot) -> list[str]:
    """What `mnemiq enrich` says about semantic enrichment that fell short (M112): tables left
    with no description, and tables only partly described, each with its cause."""
    def name(job: Job) -> str:
        return job.id.removeprefix("semantic:")

    failed = [f"{name(j)} ({j.detail})" if j.detail else name(j)
              for j in snapshot.jobs if j.kind == "semantic" and j.status == "failed"]
    partial = [f"{name(j)} ({j.detail})"
               for j in snapshot.jobs if j.kind == "semantic" and j.status == "done" and j.detail]
    lines = []
    if failed:
        lines.append(f"WARNING: {len(failed)} table(s) got NO column descriptions -- semantic "
                     f"enrichment failed: {', '.join(sorted(failed))}")
    if partial:
        lines.append(f"WARNING: {len(partial)} table(s) were only PARTLY described: "
                     f"{', '.join(sorted(partial))}")
    return lines

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

from mnemiq.catalog import introspect
from mnemiq.contract import CodedValue, Column, Job, Snapshot, SourceBinding
from mnemiq.enrichment.joins import build_relationships
from mnemiq.enrichment.profiling import profile_table


def content_version(snapshot: Snapshot) -> str:
    """Hash what the snapshot asserts, not when it was built.

    Covers the observed codes -- and, after semantic enrichment, the descriptions and code
    meanings -- so a shifted vocabulary invalidates downstream caches. Excludes created_at
    and jobs: those are run bookkeeping, not content.

    The ontology vocabulary lives outside the columns (a column carries only its scheme id/label
    via `code_scheme`, never the concepts), so `ontology_version` -- the merged local+certified
    OntologyRecords version -- is folded in too, when present. That closes the gap where a changed
    governed scheme (an edited concept label, an added code) that does not rebind any column would
    otherwise reuse a stale enrichment. Included only when set, so non-ontology snapshots keep the
    exact same version they had before this field existed.
    """
    body: dict = {
        "source_id": snapshot.source_id,
        "columns": [c.model_dump(by_alias=True) for c in snapshot.columns],
        "relationships": [r.model_dump(by_alias=True) for r in snapshot.relationships],
        "source_bindings": [b.model_dump(by_alias=True) for b in snapshot.source_bindings],
    }
    if snapshot.ontology_version:
        body["ontology_version"] = snapshot.ontology_version
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def enrich_structural(adapter, source_id: str) -> Snapshot:
    """The no-LLM enrichment pass: discover, profile, infer joins.

    Produces a Snapshot stating what the source *contains*. What any of it *means* is
    left to semantic enrichment, which fills in the blanks this pass leaves behind.
    """
    catalog = introspect(adapter)
    columns: list[Column] = []
    source_bindings: list[SourceBinding] = []
    jobs: list[Job] = []

    # Declared-FK child columns are keys, not coded vocabularies -- even when their names
    # (CDSCode, ID) don't match the naming gate. Fail-soft: no catalog FKs -> the gate stands.
    fk_children: dict[str, set[str]] = {}
    try:
        for from_table, from_col, _to_table, _to_col, _cid in adapter.foreign_keys():
            fk_children.setdefault(from_table, set()).add(from_col)
    except Exception:
        fk_children = {}

    for table in catalog:
        try:
            stats = {
                s.column: s
                for s in profile_table(adapter, table, key_columns=fk_children.get(table.name))
            }
            table_columns = [
                Column(
                    id=f"{table.name}.{col.name}",
                    object_id=table.name,
                    name=col.name,
                    data_type=col.data_type,
                    # codes now, meanings later
                    coded_values=[
                        CodedValue(code=str(value))
                        for value, _count in (stats[col.name].top_k if col.name in stats else [])
                    ],
                    row_count=stats[col.name].row_count if col.name in stats else None,
                    distinct_count=stats[col.name].distinct_count if col.name in stats else None,
                    null_count=stats[col.name].null_count if col.name in stats else None,
                )
                for col in table.columns
            ]
            status = "done"
        except Exception as exc:
            # fail-soft: one bad table never sinks the run. It contributes nothing --
            # a half-profiled table would silently poison the version hash. But do NOT swallow
            # it silently: log loudly so a systematic failure (e.g. a missing driver dep that
            # sinks EVERY table of a given column type) is visible, not a quiet 93% data loss.
            logger.warning("profile failed for table %r; EXCLUDED from the model: %s",
                           table.name, exc)
            table_columns, status = [], "failed"

        if status == "done":
            columns.extend(table_columns)
            source_bindings.append(
                SourceBinding(
                    id=f"sb:{table.name}",
                    source_id=source_id,
                    object_id=table.name,
                    source_object=table.name,
                    binding_type="table",
                )
            )
        jobs.append(
            Job(id=f"profile:{table.name}", source_id=source_id, kind="profile", status=status)
        )

    try:
        relationships = build_relationships(adapter, catalog)
        status = "done"
    except Exception:
        relationships, status = [], "failed"
    jobs.append(Job(id="infer:relationships", source_id=source_id, kind="join", status=status))

    snapshot = Snapshot(
        version="",
        source_id=source_id,
        created_at=datetime.now(UTC).isoformat(),
        columns=columns,
        source_bindings=source_bindings,
        relationships=relationships,
        jobs=jobs,
    )
    snapshot.version = content_version(snapshot)
    return snapshot

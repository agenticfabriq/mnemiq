"""`mnemiq enrich --add`: merge a run over some tables into the source's latest snapshot instead of replacing it.

A run is otherwise the whole truth: its snapshot replaces the last, so enriching a large schema a subset at a time kept
only the last subset. With `--add` the run's tables replace their earlier entries and every other table keeps its own,
descriptions included, which are the expensive part to redo.
"""
from __future__ import annotations

from mnemiq.contract import Snapshot
from mnemiq.contract.semantic import Job
from mnemiq.sql.fanout_check import PARTITIONS_JOB

# Every Snapshot field and how a merge treats it. A field added to Snapshot later fails
# `test_every_snapshot_field_has_a_merge_rule` until it is given a rule here, so --add never keeps or drops it silently.
#   per_table      the kept entry for a kept table, the run's for the run's tables (by `object_id`)
#   examples       a kept example survives only if it reads no table the run enriched again
#   jobs           a per-table job goes with its table; source-level jobs are the run's, the partitions job is merged
#   run            the run's, since it applied the current records, certified set and ontology
#   relationships  re-derived over the union by the caller (`union_relationships`); here, a consistent interim
#   version        recomputed by the caller (`content_version`)
MERGE_FIELDS: dict[str, str] = {
    "version": "version",
    "source_id": "run",
    "created_at": "run",
    "ontology_version": "run",
    "columns": "per_table",
    "views": "per_table",
    "dimensions": "run",
    "metrics": "run",
    "definitions": "run",
    "relationships": "relationships",
    "source_bindings": "per_table",
    "qualities": "per_table",
    "skills": "run",
    "jobs": "jobs",
    "compatibility_profiles": "run",
    "table_facts": "per_table",
    "examples": "examples",
    "certified_refs": "run",
}
_PER_TABLE = tuple(field for field, rule in MERGE_FIELDS.items() if rule == "per_table")


def _job_table(job: Job, known: set[str]) -> str | None:
    """The table a per-table job is about, by the ids `enrich_structural` writes: `profile:<t>` (kind `profile`),
    `profile:<t>.<column>` (`profile:column`) and `profile:<t>:exact` (`profile:exact`). None for a source-level job.
    Matched against the known tables rather than split on the first dot, because a table name can hold one."""
    if job.id == PARTITIONS_JOB or not job.id.startswith("profile:"):
        return None
    rest = job.id[len("profile:"):]
    if job.kind == "profile":
        return rest
    if job.kind == "profile:exact" and rest.endswith(":exact"):
        return rest[: -len(":exact")]
    if job.kind == "profile:column":
        matches = [t for t in known if rest.startswith(f"{t}.")]
        return max(matches, key=len) if matches else None
    return None


def _failure_entries(detail: str | None, known: set[str]) -> list[tuple[str | None, str]]:
    """The partitions job's detail as (table, entry) pairs. `enrich_structural` joins `<table>.<column>: <why>` entries
    with "; ", and a reason can itself hold "; ", so a piece that starts with no known table continues the one before."""
    entries: list[tuple[str | None, str]] = []
    for piece in (detail or "").split("; "):
        if not piece:
            continue
        owners = [t for t in known if piece.startswith(f"{t}.")]
        if owners or not entries:
            entries.append((max(owners, key=len) if owners else None, piece))
        else:
            table, text = entries[-1]
            entries[-1] = (table, f"{text}; {piece}")
    return entries


def _merged_partitions(kept: Job | None, run: Job | None, run_tables: set[str], known: set[str]) -> Job | None:
    """One partitions job for the merged snapshot, the weaker of the two runs': a kept table's recorded failure goes once
    this run profiled the table again, any failure left makes it `partial` (the warning `enrich` prints names them),
    then `estimated` if either run had no partition facts, else `done`. The fan-out guard's auto setting counts all
    three as profiled (`partitions_profiled`)."""
    if kept is None and run is None:
        return None
    base = run if run is not None else kept
    kept_entries = [text for table, text in _failure_entries(kept.detail if kept else None, known)
                    if table not in run_tables]
    run_entries = [text for _table, text in _failure_entries(run.detail if run else None, known)]
    failures = kept_entries + run_entries
    statuses = {job.status for job in (kept, run) if job is not None}
    if failures:
        status = "partial"
    elif "estimated" in statuses:
        status = "estimated"
    else:
        status = "done"
    return base.model_copy(update={"status": status, "detail": "; ".join(failures)[:2000] or None})


def merge_snapshots(kept: Snapshot, run: Snapshot) -> tuple[Snapshot, list[str]]:
    """`run`, a run over some of the source's tables, merged into `kept`, the source's latest snapshot. Returns the
    merged snapshot and the tables whose profile failed in this run but which `kept` still holds: those keep their
    earlier entries and job, so the caller warns about them rather than reporting them excluded."""
    run_tables = {b.object_id for b in run.source_bindings}
    kept_tables = {b.object_id for b in kept.source_bindings}
    run_profiled = {j.id[len("profile:"):] for j in run.jobs if j.kind == "profile" and j.id.startswith("profile:")}
    stay = kept_tables - run_tables
    kept_failed = sorted((run_profiled - run_tables) & kept_tables)
    known = kept_tables | run_tables | run_profiled

    update: dict[str, object] = {}
    for field in _PER_TABLE:
        update[field] = ([entry for entry in getattr(kept, field) if entry.object_id in stay]
                         + [entry for entry in getattr(run, field) if entry.object_id not in stay])

    def reads(example) -> set[str]:
        return set(example.tables or [example.object_id])

    update["examples"] = [e for e in kept.examples if not reads(e) & run_tables] + list(run.examples)

    jobs: list[Job] = []
    for job in kept.jobs:
        table = _job_table(job, known)
        if table is not None and table in stay:
            jobs.append(job)
    for job in run.jobs:
        table = _job_table(job, known)
        if job.id == PARTITIONS_JOB or (table is not None and table in stay):
            continue
        jobs.append(job)
    partitions = _merged_partitions(next((j for j in kept.jobs if j.id == PARTITIONS_JOB), None),
                                    next((j for j in run.jobs if j.id == PARTITIONS_JOB), None), run_tables, known)
    if partitions is not None:
        jobs.append(partitions)
    update["jobs"] = jobs

    # An interim the caller replaces with joins re-derived over the union: the kept joins among tables that stay, and
    # the run's own.
    update["relationships"] = ([r for r in kept.relationships if r.from_ in stay and r.to in stay]
                               + list(run.relationships))
    return run.model_copy(update=update, deep=True), kept_failed

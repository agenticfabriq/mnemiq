"""`mnemiq enrich --add`: merge a run over some tables into the source's latest snapshot instead of replacing it.

A run is otherwise the whole truth: its snapshot replaces the last, so enriching a large schema a subset at a time kept
only the last subset. With `--add` the run's tables replace their earlier entries and every other table keeps its own,
descriptions included, which are the expensive part to redo.
"""
from __future__ import annotations

import dataclasses
import glob
import logging

from mnemiq.config import SourceSpec
from mnemiq.contract import Relationship, Snapshot
from mnemiq.contract.semantic import Job
from mnemiq.sql.fanout_check import PARTITIONS_JOB

logger = logging.getLogger(__name__)

# Every Snapshot field and how a merge treats it. A field added to Snapshot later fails
# `test_every_snapshot_field_has_a_merge_rule` until it is given a rule here, so --add never keeps or drops it silently.
#   per_table      the run's entry for each object it holds or table it bound, the kept one for every other (by
#                  `object_id`; a view need not be bound as a table)
#   union          views: in the merge, as per_table with discovery done only if both runs' was; then discovered again
#                  over the union by the caller (`union_views`), since a run discovers views only within its subset
#   examples       a kept example survives only if it reads no table the run enriched again
#   jobs           a per-table job (`profile:*`, `facts:<t>`) goes with its table; source-level jobs are the run's,
#                  except the partitions and view-discovery jobs, which keep the weaker of the two runs'
#   run            the run's, since it applied the current records, certified set and ontology
#   relationships  re-derived over the union by the caller (`union_relationships`); here, a consistent interim
#   version        recomputed by the caller (`content_version`)
MERGE_FIELDS: dict[str, str] = {
    "version": "version",
    "source_id": "run",
    "created_at": "run",
    "ontology_version": "run",
    "columns": "per_table",
    "views": "union",
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
_PER_TABLE = tuple(field for field, rule in MERGE_FIELDS.items() if rule in ("per_table", "union"))
VIEWS_JOB = "discover:views"


# Jobs about the whole source: the run's, except the partitions and view-discovery jobs, which merge_run reconciles.
# Every other job whose id names a table after its kind (`profile:<t>`, `profile:<t>.<column>`, `profile:<t>:exact`,
# `facts:<t>`, `semantic:<t>`, and any kind added later) goes with that table, so a kept table keeps its record of what
# failed, which the closing report warns from (Codex's fourth review found `semantic:<t>` dropped).
_SOURCE_JOBS = frozenset({"infer:relationships", VIEWS_JOB, PARTITIONS_JOB, "superset:source", "enrich:time-limit"})


def _job_table(job: Job, known: set[str]) -> str | None:
    """The table a per-table job is about, or None for a job about the whole source. Matched against the known tables
    rather than split on a dot, because a table name can hold one: the longest table the id names after its kind."""
    if job.id in _SOURCE_JOBS or job.id.startswith("certified:") or ":" not in job.id:
        return None
    rest = job.id.split(":", 1)[1]
    matches = [t for t in known if rest == t or rest.startswith(f"{t}.") or rest.startswith(f"{t}:")]
    return max(matches, key=len) if matches else None


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


def merge_run(kept: Snapshot, run: Snapshot) -> tuple[Snapshot, list[str]]:
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
        replaced = run_tables | {entry.object_id for entry in getattr(run, field)}
        update[field] = ([entry for entry in getattr(kept, field) if entry.object_id not in replaced]
                         + list(getattr(run, field)))

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
        if job.id in (PARTITIONS_JOB, VIEWS_JOB) or (table is not None and table in stay):
            continue
        jobs.append(job)
    # View discovery is the run's over its subset only, so its `done` cannot vouch for a kept run whose discovery did
    # not complete: the view floor reads this job (`inventory_for`), and a `done` over a partial inventory lets a
    # row-filtered view through. Done only when both were; the caller then discovers again over the union.
    discoveries = [next((j for j in snap.jobs if j.id == VIEWS_JOB), None) for snap in (kept, run)]
    if any(j is not None for j in discoveries):
        both = all(j is not None and j.status == "done" for j in discoveries)
        base = next(j for j in reversed(discoveries) if j is not None)
        jobs.append(base.model_copy(update={"status": "done" if both else "failed",
                                            "detail": None if both else "a run's view discovery did not complete"}))
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


def estimated_tables(snapshot: Snapshot) -> frozenset[str]:
    """The tables profiled from the source's statistics, each as its own run recorded it (`profile:<t>` done with detail
    "estimated"), never by the current run's `exact_tables`: a table kept from an exact run stays measured."""
    return frozenset(j.id.removeprefix("profile:") for j in snapshot.jobs
                     if j.kind == "profile" and j.status == "done" and j.detail == "estimated")


def union_spec(spec: SourceSpec, snapshot: Snapshot) -> SourceSpec:
    """`spec` scoped to every table `snapshot` binds. Each name is escaped, because the manifest's `tables` are matched
    as patterns: a table named `a[1]` would otherwise also match `a1`."""
    return dataclasses.replace(spec, tables=tuple(glob.escape(t) for t in
                                                  sorted(b.object_id for b in snapshot.source_bindings)))


def union_relationships(union_adapter, snapshot: Snapshot, *,
                        harvest_values: bool) -> tuple[list[Relationship], Job]:
    """The joins of the merged snapshot's tables, read again through `union_adapter`, which the caller scopes to them
    (`union_spec`): declared keys from the source, and inference among the tables profiled exactly, as
    `enrich_structural` reads them for one run. Returns the relationships and the `infer:relationships` job."""
    from mnemiq.catalog import introspect
    from mnemiq.enrichment.joins import build_relationships
    from mnemiq.enrichment.pipeline import _listed
    from mnemiq.enrichment.profiling import unharvested

    profiled = {j.id.removeprefix("profile:") for j in snapshot.jobs if j.kind == "profile" and j.status == "done"}
    failures: list[str] = []
    try:
        catalog = [t for t in introspect(union_adapter) if t.name in profiled]
        relationships = build_relationships(union_adapter, catalog, **unharvested(harvest_values), failures=failures,
                                            estimated=estimated_tables(snapshot))
        status = "partial" if failures else "done"
    except Exception as exc:  # noqa: BLE001 -- as the run's own join pass: a failure costs the joins, not the snapshot
        logger.warning("the joins across the merged tables could not be read (%s); none are kept",
                       type(exc).__name__)
        relationships, status = [], "failed"
    return relationships, Job(id="infer:relationships", source_id=snapshot.source_id, kind="join", status=status,
                              detail=_listed(failures))


def views_spec(spec: SourceSpec, kept: Snapshot, snapshot: Snapshot) -> SourceSpec:
    """`spec` scoped for discovering views again under --add: every table `snapshot` binds, every view either snapshot
    already knows (a view need not be bound as a table), and the current manifest's own patterns, so the views this run
    would have discovered alone are discovered too. A manifest with no `tables` means the whole source, and stays so:
    narrowed to known names, a view no earlier run saw would drop out of a discovery then recorded done."""
    if not spec.tables:
        return spec
    known_views = sorted({v.object_id for v in kept.views} | {v.object_id for v in snapshot.views})
    names = [*union_spec(spec, snapshot).tables, *(glob.escape(v) for v in known_views), *spec.tables]
    return dataclasses.replace(spec, tables=tuple(dict.fromkeys(names)))


def union_views(views_adapter, kept: Snapshot, snapshot: Snapshot) -> tuple[list, Job]:
    """The views of the merged snapshot, discovered again through `views_adapter` (`views_spec`), and the
    `discover:views` job. Failed when the source does not answer, as `enrich_structural` records it, and failed too while
    the kept snapshot's own discovery never completed: that run's scope is not recorded, so a fresh discovery over the
    union cannot know it covered those views, and the view floor must keep refusing until a run without --add does."""
    from mnemiq.contract.semantic import ViewDefinition

    try:
        views = [ViewDefinition(object_id=name, definition=body, dialect=dialect)
                 for name, body, dialect in views_adapter.view_definitions()]
        status, detail = "done", None
    except Exception as exc:  # noqa: BLE001 -- as enrich_structural: no answer leaves the inventory unavailable
        logger.warning("the views across the merged tables could not be read (%s)", type(exc).__name__)
        views, status = [], "failed"
        detail = f"the source did not answer when its views were read ({type(exc).__name__}): enrich again"
    kept_job = next((j for j in kept.jobs if j.id == VIEWS_JOB), None)
    if status == "done" and (kept_job is None or kept_job.status != "done"):
        status, detail = "failed", ("an earlier run's view discovery did not complete and its scope is not recorded: "
                                    "enrich without --add to discover the views again")
    return views, Job(id=VIEWS_JOB, source_id=snapshot.source_id, kind="discover", status=status, detail=detail)


def _laid_by(entry, payload) -> bool:
    """Whether `entry` is what laying `payload` put in a list: the same id where the type has one (a dimension whose
    level was unrecognised is laid as a copy with the strictest level, so equality would miss it), else equal."""
    if hasattr(payload, "id") and hasattr(entry, "id"):
        return entry.id == payload.id
    return entry == payload


def reapply_overlays(snapshot: Snapshot, certified, dictionary) -> Snapshot:
    """--add: the certified set and the operator's dictionary laid again over the merged snapshot. A run lays them over
    its own subset only (`apply_certified_set`, `apply_dictionary`), so a column kept from an earlier run would miss a
    classification certified since (a personal dimension over it, which gates value harvesting), and the join pass,
    which reads joins again over the union, has dropped the certified joins. What the run's laying appended (the
    certified standalone records and `certified:*` jobs) is taken out first, since laying appends rather than replaces.
    Precedence stays as one run's: certified, then the dictionary, the operator's final override."""
    from mnemiq.enrichment.certified import _STANDALONE, apply_certified_set
    from mnemiq.enrichment.grounding import apply_dictionary

    payloads: dict[str, list] = {}
    for record in certified.records:
        payloads.setdefault(record.envelope.object_type, []).append(record.payload)
    update: dict[str, object] = {
        attr: [entry for entry in getattr(snapshot, attr)
               if not any(_laid_by(entry, p) for p in payloads.get(object_type, []))]
        for object_type, attr in _STANDALONE.items() if payloads.get(object_type)
    }
    update["jobs"] = [j for j in snapshot.jobs if not j.id.startswith("certified:")]
    laid, _protected = apply_certified_set(snapshot.model_copy(update=update, deep=True), certified)
    return apply_dictionary(laid, dictionary) if dictionary else laid

"""`mnemiq enrich --add`: a run over a subset of tables merges into the source's latest snapshot instead of replacing it.

Without the merge, enriching a 600-table schema in batches loses every batch but the last; and the joins between
batches, which no single run sees, are re-derived over the union (`union_relationships`)."""
import pytest

from mnemiq.contract import Column, Snapshot, SourceBinding
from mnemiq.contract.semantic import Example, Job, Quality, TableFacts, ViewDefinition
from mnemiq.enrichment.accumulate import MERGE_FIELDS, merge_snapshots
from mnemiq.sql.fanout_check import PARTITIONS_JOB


def _snap(tables: dict[str, str], *, jobs=(), facts=(), examples=(), partitions=None, version="v",
          definitions=()) -> Snapshot:
    """One binding, one column `id` described as `tables[t]`, and a done profile job per table."""
    all_jobs = [Job(id=f"profile:{t}", source_id="src", kind="profile", status="done") for t in tables]
    all_jobs += list(jobs)
    if partitions is not None:
        all_jobs.append(Job(id=PARTITIONS_JOB, source_id="src", kind="profile:partitions", status=partitions[0],
                            detail=partitions[1]))
    return Snapshot(
        version=version, source_id="src", created_at="2026-11-23T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="src", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id=f"{t}.id", object_id=t, name="id", description=d) for t, d in tables.items()],
        jobs=all_jobs, table_facts=list(facts), examples=list(examples), definitions=list(definitions),
    )


def _described(snap: Snapshot, table: str) -> str | None:
    return next(c.description for c in snap.columns if c.object_id == table)


def test_every_snapshot_field_has_a_merge_rule():
    # A field added to Snapshot later must be given a rule here, or --add would silently keep or drop it.
    assert set(MERGE_FIELDS) == set(Snapshot.model_fields)


def test_kept_tables_keep_their_descriptions():
    merged, kept_failed = merge_snapshots(_snap({"a": "kept-a"}), _snap({"b": "run-b"}))
    assert {b.object_id for b in merged.source_bindings} == {"a", "b"}
    assert _described(merged, "a") == "kept-a"
    assert _described(merged, "b") == "run-b"
    assert kept_failed == []


def test_re_adding_a_table_replaces_every_per_table_entry():
    kept = _snap({"a": "old-a", "c": "kept-c"},
                 facts=[TableFacts(object_id="a", grain="old grain"), TableFacts(object_id="c", grain="c grain")],
                 examples=[Example(question="old a?", sql="SELECT 1 FROM a", tables=["a"], object_id="a"),
                           Example(question="c?", sql="SELECT 1 FROM c", tables=["c"], object_id="c")],
                 jobs=[Job(id="profile:a.id", source_id="src", kind="profile:column", status="failed")])
    kept = kept.model_copy(update={
        "views": [ViewDefinition(object_id="a", definition="old", dialect="duckdb")],
        "qualities": [Quality(id="q:a", object_id="a", name="n", status="ok", threshold="1", observed_value="1")]})
    run = _snap({"a": "new-a"}, facts=[TableFacts(object_id="a", grain="new grain")],
                examples=[Example(question="new a?", sql="SELECT 2 FROM a", tables=["a"], object_id="a")])
    merged, _ = merge_snapshots(kept, run)

    assert [c.description for c in merged.columns if c.object_id == "a"] == ["new-a"]
    assert [f.grain for f in merged.table_facts if f.object_id == "a"] == ["new grain"]
    assert sorted(e.question for e in merged.examples) == ["c?", "new a?"]
    assert [j.status for j in merged.jobs if j.id == "profile:a"] == ["done"]
    assert not [j for j in merged.jobs if j.id == "profile:a.id"], "the kept column job of a re-added table went"
    assert not merged.views and not merged.qualities, "a re-added table takes the run's views and qualities (none)"
    assert _described(merged, "c") == "kept-c"


def test_an_example_reading_a_re_added_table_is_dropped_even_when_it_belongs_to_another():
    kept = _snap({"a": "a", "c": "c"},
                 examples=[Example(question="a and c?", sql="SELECT 1 FROM c JOIN a", tables=["c", "a"],
                                   object_id="c")])
    merged, _ = merge_snapshots(kept, _snap({"a": "a2"}))
    assert merged.examples == []


def test_a_re_added_table_whose_profile_failed_keeps_its_earlier_enrichment():
    kept = _snap({"a": "kept-a"})
    run = _snap({}, jobs=[Job(id="profile:a", source_id="src", kind="profile", status="failed", detail="boom")])
    merged, kept_failed = merge_snapshots(kept, run)

    assert _described(merged, "a") == "kept-a"
    assert [j.status for j in merged.jobs if j.id == "profile:a"] == ["done"]
    assert kept_failed == ["a"]


def test_a_new_table_whose_profile_failed_keeps_the_runs_failed_job():
    run = _snap({}, jobs=[Job(id="profile:b", source_id="src", kind="profile", status="failed", detail="boom")])
    merged, kept_failed = merge_snapshots(_snap({"a": "kept-a"}), run)
    assert [j.status for j in merged.jobs if j.id == "profile:b"] == ["failed"]
    assert kept_failed == []


@pytest.mark.parametrize(("kept", "run", "expected"), [
    (("done", None), ("estimated", None), "estimated"),
    (("estimated", None), ("done", None), "estimated"),
    (("done", None), ("done", None), "done"),
    (("partial", "c.flag: boom"), ("done", None), "partial"),
    (("done", None), ("partial", "b.flag: boom"), "partial"),
])
def test_the_partitions_job_reports_the_weaker_of_the_two_runs(kept, run, expected):
    merged, _ = merge_snapshots(_snap({"c": "c"}, partitions=kept), _snap({"b": "b"}, partitions=run))
    (job,) = [j for j in merged.jobs if j.id == PARTITIONS_JOB]
    assert job.status == expected


def test_a_kept_partition_failure_goes_once_its_table_is_profiled_again():
    # The warning names the tables whose current-row joins the guard will refuse; a table this run profiled again
    # no longer belongs in it.
    kept = _snap({"a": "a", "c": "c"}, partitions=("partial", "a.flag: boom; c.flag: bang"))
    merged, _ = merge_snapshots(kept, _snap({"a": "a2"}, partitions=("done", None)))
    (job,) = [j for j in merged.jobs if j.id == PARTITIONS_JOB]
    assert (job.status, job.detail) == ("partial", "c.flag: bang")

    merged, _ = merge_snapshots(_snap({"a": "a"}, partitions=("partial", "a.flag: boom")),
                                _snap({"a": "a2"}, partitions=("done", None)))
    (job,) = [j for j in merged.jobs if j.id == PARTITIONS_JOB]
    assert (job.status, job.detail) == ("done", None)


def test_the_runs_global_entries_win():
    from mnemiq.contract import Definition

    kept = _snap({"a": "a"}, definitions=[Definition(id="d:old", term="old", domain="x", definition="old")])
    run = _snap({"b": "b"}, definitions=[Definition(id="d:new", term="new", domain="x", definition="new")])
    merged, _ = merge_snapshots(kept, run)
    assert [d.id for d in merged.definitions] == ["d:new"]
    assert merged.created_at == run.created_at

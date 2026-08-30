"""M59: the profile jobs were recorded honestly and nothing read them.

`enrich_structural` is fail-soft per table -- one that will not profile is logged and excluded, so
a single bad table never sinks a run. It records `Job(id="profile:<t>", status="failed")` for each.
When EVERY table failed, the run still returned a snapshot and exited 0, so an engine that had
connected to a database and could read none of it was indistinguishable from one connected to an
empty database. The signal existed; the consumer did not.
"""

from __future__ import annotations

import pytest

from mnemiq.contract import Job, Snapshot
from mnemiq.enrichment.pipeline import profile_outcome


def _snap(*statuses: str, extra: list[Job] | None = None) -> Snapshot:
    jobs = [Job(id=f"profile:t{i}", source_id="s", kind="profile", status=st)
            for i, st in enumerate(statuses)]
    return Snapshot(version="v", source_id="s", created_at="t", jobs=jobs + (extra or []))


def test_every_table_failing_is_unread_not_empty():
    """The finding itself. This must not be reported as success, and must not read as 'no tables'."""
    verdict, detail = profile_outcome(_snap("failed", "failed"))
    assert verdict == "unread"
    assert "NOT evidence that the source is empty" in detail
    assert "2 table(s)" in detail


def test_a_source_with_no_tables_is_empty_and_distinct_from_unread():
    """A real state, and the one a threshold would collapse `unread` into: both have zero columns."""
    assert profile_outcome(_snap())[0] == "empty"


def test_all_profiled_is_complete():
    assert profile_outcome(_snap("done", "done"))[0] == "complete"


def test_some_failing_is_partial_and_names_the_tables():
    """Fail-soft is preserved -- the run proceeds -- but the caller is told which tables are gone,
    because the engine will answer as though they do not exist."""
    verdict, detail = profile_outcome(_snap("done", "failed", "done"))
    assert verdict == "partial"
    assert "1 of 3" in detail and "t1" in detail


def test_only_profile_jobs_decide_it():
    """`infer:relationships` and `discover:views` fail for their own reasons and have their own
    handling; a failed view discovery must not make a fully-profiled model look unread."""
    views_failed = [Job(id="discover:views", source_id="s", kind="discover", status="failed"),
                    Job(id="infer:relationships", source_id="s", kind="join", status="failed")]
    assert profile_outcome(_snap("done", extra=views_failed))[0] == "complete"
    assert profile_outcome(_snap(extra=views_failed))[0] == "empty"


@pytest.mark.parametrize("statuses, expected", [
    (("failed",), "unread"),
    (("done",), "complete"),
])
def test_a_single_table_still_distinguishes_the_two(statuses, expected):
    """One table is the smallest case where 'could not read' and 'nothing to read' diverge, and a
    count-based check is likeliest to get it wrong."""
    assert profile_outcome(_snap(*statuses))[0] == expected


def test_enrich_refuses_to_save_a_snapshot_that_describes_nothing(monkeypatch, tmp_path, capsys):
    """The wiring, which is the half that was missing.

    Classifying the outcome is worth nothing if `_cmd_enrich` proceeds anyway: the failure mode was
    never that the truth was unknown, it was that nothing acted on it. So this asserts the two
    things that make it a fix -- a non-zero exit, and NOTHING written to the store. Saving an empty
    snapshot and returning 1 would leave the next `build` and `ask` reading a model that describes
    a database nobody could read.
    """
    from mnemiq.cli import _cmd_enrich
    from mnemiq.config import Settings

    saved: list = []
    monkeypatch.setattr("mnemiq.enrichment.pipeline.enrich_structural",
                        lambda adapter, source_id: _snap("failed", "failed"))
    monkeypatch.setattr("mnemiq.store.snapshot_store.save_snapshot",
                        lambda *a, **k: saved.append(a))
    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: object())

    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, acme_data_dir=None,
                        pg_dsn="postgresql://h/db", store_path=str(tmp_path / "s.duckdb"))
    assert _cmd_enrich(settings) == 1
    assert saved == [], "an unreadable source must not leave a snapshot behind"
    err = capsys.readouterr().err
    assert "enrich failed" in err
    assert "NOT evidence that the source is empty" in err

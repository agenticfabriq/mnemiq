"""M59: the profile jobs were read and reported, and nothing ACTED on them.

`enrich_structural` is fail-soft per table -- one that will not profile is logged and excluded, so
a single bad table never sinks a run. It records `Job(id="profile:<t>", status="failed")` for each.

The finding as first filed said nothing read those statuses. That was wrong: `_cmd_enrich` already
ended with a WARNING naming the excluded tables, and had before any of this work. What nobody did
was act on them -- the run exited 0 and SAVED the snapshot whatever they said. So a run in which
every table failed printed "the semantic model is INCOMPLETE", which is a wild understatement for
"describes nothing", and persisted it for the next `build` and `ask` to plan against.
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
    """Distinct from `unread` in the DIAGNOSIS -- could not read, versus nothing was there -- but
    both are refused, because neither produces a snapshot worth keeping. A schema name matching
    nothing introspects to zero tables and so never reaches a profile at all, which is how it
    reaches this state without a single failure recorded."""
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


@pytest.mark.parametrize("statuses, why", [
    ((), "no tables at all -- empty source or a schema matching nothing"),
    (("failed",), "one table and it could not be read"),
])
def test_enrich_refuses_and_saves_nothing_for_every_describes_nothing_outcome(
    monkeypatch, tmp_path, capsys, statuses, why
):
    """Both refusing outcomes are wired, not just the one the finding was filed about.

    A review pointed out that only `unread` had an end-to-end test, so a wiring mistake in the
    other branch would not have been caught -- and `empty` was, at that moment, still exiting 0.
    """
    from mnemiq.cli import _cmd_enrich
    from mnemiq.config import Settings

    saved: list = []
    monkeypatch.setattr("mnemiq.enrichment.pipeline.enrich_structural",
                        lambda adapter, source_id: _snap(*statuses))
    monkeypatch.setattr("mnemiq.store.snapshot_store.save_snapshot",
                        lambda *a, **k: saved.append(a))
    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: object())

    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, acme_data_dir=None,
                        pg_dsn="postgresql://h/db", store_path=str(tmp_path / "s.duckdb"))
    assert _cmd_enrich(settings) == 1, why
    assert saved == [], "a snapshot that describes nothing must not be persisted"
    assert "enrich failed" in capsys.readouterr().err


def test_a_partial_run_is_not_refused_by_the_early_branch(monkeypatch, tmp_path, capsys):
    """`_cmd_enrich` already reports excluded tables at its tail, so the early check must not.

    **This test previously claimed more than it could show, and the way it failed is the point.**
    It asserted `err.count("FAILED to profile") <= 1` to prove the tail warning fires exactly once
    -- but with no LLM configured the run raises at `LLMClient(settings)` long before that tail,
    the `except Exception: pass` swallowed it, and the count was 0, so the assertion was `0 <= 1`
    and could not fail. It verified nothing while reading as though it verified the duplication.

    So it asserts only what this call can actually establish: a `partial` run is NOT refused by
    the early branch. It does NOT show the tail warning fires exactly once -- a second attempt at
    that, `assert "enrich incomplete" not in err`, was dead too, since that string had already been
    deleted from `_cmd_enrich` and so could never fail. Duplicate-suppression is not testable from
    here, and a passing assertion that implies it is worse than its absence.
    """
    from mnemiq.cli import _cmd_enrich
    from mnemiq.config import Settings

    monkeypatch.setattr("mnemiq.enrichment.pipeline.enrich_structural",
                        lambda adapter, source_id: _snap("done", "failed"))
    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: object())
    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, acme_data_dir=None,
                        pg_dsn="postgresql://h/db", store_path=str(tmp_path / "s.duckdb"))
    with pytest.raises(Exception):  # noqa: B017 - the run goes on to work this test does not set up
        _cmd_enrich(settings)
    err = capsys.readouterr().err
    assert "enrich failed" not in err, "a partial run must not be refused"


# -- M73: a column that FAILED to measure, vs one whose type cannot be measured ------------------
#
# Both persist as `distinct_count=None, null_count=None`. One is a permanent, expected property of
# the schema and needs no action; the other is a measurement that did not complete, for a cause
# with nothing to do with the column -- temp space, a privilege, a driver fault. Before this, the
# failure existed only in a log line, the table's job said `done`, and the run reported `complete`.


def _col_job(name: str) -> Job:
    return Job(id=f"profile:{name}", source_id="s", kind="profile:column", status="failed")


def test_a_column_that_failed_to_measure_is_not_a_complete_profile():
    verdict, detail = profile_outcome(_snap("done", "done", extra=[_col_job("t0.AMOUNT")]))
    assert verdict == "unmeasured", "every table profiled, so this is not `partial` -- but it is"
    assert "t0.AMOUNT" in detail, "the operator has to know WHICH column"
    assert "type cannot be counted" in detail, (
        "the detail must say what it is NOT, because that is the state it is confusable with")


def test_a_column_job_is_not_counted_as_a_table():
    """`kind` is deliberately not "profile": `profile_outcome` and `_cmd_enrich` both filter on
    that exact string to count TABLES, so a column job landing in that count would report a table
    that does not exist -- and would make a one-table source claim two."""
    verdict, detail = profile_outcome(_snap("done", extra=[_col_job("t0.AMOUNT")]))
    assert "all 1 table(s) profiled" in detail
    assert verdict == "unmeasured"


def test_lost_tables_and_unmeasured_columns_are_both_reported():
    """A run can lose whole tables AND fail to measure columns of the tables it kept. Naming only
    the larger problem hides the smaller one in exactly the run where it matters most."""
    verdict, detail = profile_outcome(
        _snap("failed", "done", extra=[_col_job("t1.AMOUNT")]))
    assert verdict == "partial", "a missing table is the bigger fact and still decides the verdict"
    assert "t0" in detail and "EXCLUDED" in detail
    assert "t1.AMOUNT" in detail and "A further 1 column(s)" in detail


def test_a_clean_run_is_still_complete():
    """The control. `unmeasured` must not fire on a run with nothing wrong with it, or it becomes
    the always-on warning this codebase has already had to remove twice."""
    assert profile_outcome(_snap("done", "done"))[0] == "complete"


def test_a_partially_measured_table_KEEPS_its_columns():
    """The regression this fix had to avoid, pinned so it cannot be reintroduced.

    `enrich_structural` uses `status == "done"` as the allowlist deciding whether a table's columns
    are kept -- `views.py` documents that `Job.status` is a free string for exactly this reason. So
    recording a per-column failure as a THIRD table status would have silently dropped every column
    of a partially measured table: a fix that loses more of the model than the finding it closes.
    The failure is a separate job and the table stays `done`.
    """
    from dataclasses import dataclass

    from mnemiq.enrichment.pipeline import enrich_structural

    class _Err(Exception):
        def __init__(self, code, msg):
            e = type("E", (), {})()
            e.code, e.full_code = code, f"ORA-{code:05d}"
            super().__init__(e, msg)
            self.args = (e, msg)

        def __str__(self):
            return f"{self.args[0].full_code}: {self.args[1]}"

    @dataclass
    class _A:
        dialect: str = "oracle"
        DatabaseError = _Err

        def introspect(self):
            return ["T"]

        def list_columns(self):
            return [("T", "GOOD", "NUMBER"), ("T", "BAD", "NUMBER")]

        def foreign_keys(self):
            return []

        def view_definitions(self):
            return []

        def execute(self, sql):
            u = sql.upper()
            if "COUNT(DISTINCT" in u and '"BAD"' in sql:
                raise _Err(1652, "unable to extend temp segment")
            if u.startswith("SELECT COUNT(*) FROM"):
                return [(100,)]
            if u.startswith("SELECT COUNT(DISTINCT"):
                return [(5, 100)]
            if "COUNT(DISTINCT" in u and "GROUP BY" not in u:
                return [tuple([100] + [5, 100] * sql.count("count(DISTINCT"))]
            return []

    snap = enrich_structural(_A(), "s")

    assert {c.name for c in snap.columns} == {"GOOD", "BAD"}, (
        "the partially measured table lost columns -- the fix would cost more than the finding"
    )
    assert [j.status for j in snap.jobs if j.id == "profile:T"] == ["done"], (
        "the table WAS profiled; only one column's measurement failed"
    )
    col_job = next(j for j in snap.jobs if j.kind == "profile:column")
    assert col_job.id == "profile:T.BAD"
    # The CAUSE, carried from `ColumnStats.failure` to the job by the pipeline. Asserted HERE,
    # through `enrich_structural`, because the round-trip test builds its job by hand: dropping
    # `detail=st.failure` from the pipeline left that test green, and the mutation control is what
    # showed the gap.
    assert col_job.detail and "ORA-01652" in col_job.detail
    verdict, why = profile_outcome(snap)
    assert verdict == "unmeasured" and "ORA-01652" in why
    # ... and the surviving column still carries its real statistics.
    good = next(c for c in snap.columns if c.name == "GOOD")
    assert good.distinct_count == 5 and good.row_count == 100


def test_the_CAUSE_survives_the_run_that_produced_it():
    """The half of M73 a first pass left in logging, and the reason it matters.

    A snapshot outlives its run. Recording only WHICH column was not measured tells a reader
    months later that something is missing and nothing about whether it is worth retrying -- temp
    space clears, a revoked privilege is granted back, a driver fault is fixed, and an
    unaggregatable type never changes. The cause was in `ColumnStats.failure`, whose only reader
    was a test: it reached no persisted record at all, and the closing note claiming otherwise was
    caught by a review.
    """
    import json

    from mnemiq.contract import Snapshot

    snap = _snap("done", extra=[Job(id="profile:t0.AMOUNT", source_id="s", kind="profile:column",
                                    status="failed", detail="ORA-01652: unable to extend temp")])

    # Through a serialisation round trip, because "the snapshot outlives the run" is the claim.
    reloaded = Snapshot.model_validate(json.loads(snap.model_dump_json()))
    job = next(j for j in reloaded.jobs if j.kind == "profile:column")
    assert job.detail == "ORA-01652: unable to extend temp"

    verdict, detail = profile_outcome(reloaded)
    assert verdict == "unmeasured"
    assert "ORA-01652" in detail, "the operator is told WHY, not just which"


def test_a_snapshot_written_before_the_cause_existed_still_loads():
    """`detail` is optional and defaulted, so every snapshot persisted before this field parses
    unchanged -- and reports the column without a cause rather than failing to load at all."""
    import json

    from mnemiq.contract import Snapshot

    raw = json.loads(_snap("done").model_dump_json())
    raw["jobs"].append({"id": "profile:t0.AMOUNT", "source_id": "s", "kind": "profile:column",
                        "status": "failed"})  # no `detail` key at all
    snap = Snapshot.model_validate(raw)
    assert next(j for j in snap.jobs if j.kind == "profile:column").detail is None
    verdict, detail = profile_outcome(snap)
    assert verdict == "unmeasured" and "t0.AMOUNT" in detail

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

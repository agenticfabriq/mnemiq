"""M27 slice 1: the source reports its views, and the snapshot carries them.

A view is the one object a governed engine cannot reason about from its columns: the rows it
returns are defined by SQL the source holds. Until that SQL is in the snapshot, a granted view
over a filtered table returns unfiltered rows, and nothing in the model even marks it as a view.
"""

from __future__ import annotations

import sqlite3

import pytest

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.contract import Snapshot, ViewDefinition
from mnemiq.enrichment.pipeline import content_version, enrich_structural


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "src.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE claim (claim_id INTEGER, region TEXT, amount REAL)")
    con.execute("INSERT INTO claim VALUES (1, 'west', 10.0), (2, 'east', 20.0)")
    con.execute("CREATE VIEW claim_totals AS SELECT region, SUM(amount) AS total "
                "FROM claim GROUP BY region")
    con.commit()
    con.close()
    return str(path)


def test_the_adapter_reports_a_view_with_its_body_and_the_sources_dialect(source):
    views = SQLiteAdapter(source).view_definitions()
    assert len(views) == 1
    name, body, dialect = views[0]
    assert name == "claim_totals"
    assert "SUM(amount)" in body.replace("sum(", "SUM(")
    assert dialect == "sqlite", "the body's dialect is the source's, not the executor's"


def test_a_source_with_no_views_reports_none_rather_than_failing(tmp_path):
    path = tmp_path / "bare.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (a INTEGER)")
    con.commit()
    con.close()
    assert SQLiteAdapter(str(path)).view_definitions() == []


def test_the_snapshot_carries_the_view(source):
    snap = enrich_structural(SQLiteAdapter(source), source_id="s")
    assert [v.object_id for v in snap.views] == ["claim_totals"]
    assert snap.views[0].dialect == "sqlite"


def test_discovery_is_recorded_as_a_job_so_a_silent_failure_is_visible(source):
    snap = enrich_structural(SQLiteAdapter(source), source_id="s")
    job = next(j for j in snap.jobs if j.id == "discover:views")
    assert job.status == "done"


def test_a_source_that_will_not_answer_fails_soft_and_says_so(source):
    class _Refuses(SQLiteAdapter):
        def view_definitions(self):
            raise RuntimeError("no catalog access")

    snap = enrich_structural(_Refuses(source), source_id="s")
    assert snap.views == []
    job = next(j for j in snap.jobs if j.id == "discover:views")
    assert job.status == "failed", (
        "an empty view list must be distinguishable from a source that refused to answer"
    )


# --- the version ----------------------------------------------------------------


def _snap(**kw) -> Snapshot:
    return Snapshot(version="", source_id="s", created_at="2026-01-01T00:00:00Z", **kw)


def test_changing_a_view_body_changes_the_content_version():
    """Change the body and the rows change, so a cached answer built on the old one is stale."""
    before = _snap(views=[ViewDefinition(object_id="v", definition="SELECT 1", dialect="duckdb")])
    after = _snap(views=[ViewDefinition(object_id="v", definition="SELECT 2", dialect="duckdb")])
    assert content_version(before) != content_version(after)


def test_a_snapshot_with_no_views_keeps_the_version_it_had_before_views_existed():
    """The `ontology_version` rule: an absent field must not re-version every old snapshot."""
    assert content_version(_snap()) == content_version(_snap(views=[]))

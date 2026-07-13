import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot, save_snapshot

pytestmark = pytest.mark.integration

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


def _adapter():
    return DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))


@pytest.fixture(scope="module")
def snap():
    return enrich_structural(_adapter(), "acme")


def test_pipeline_covers_the_whole_source(snap):
    assert len(snap.source_bindings) == 29
    assert len(snap.columns) > 29
    assert snap.relationships, "ACME has foreign keys"
    assert all(j.status in {"done", "failed"} for j in snap.jobs)


def test_pipeline_harvests_codes_without_meanings(snap):
    coded = [c for c in snap.columns if c.coded_values]
    assert coded, "structural pass should harvest observed code sets"
    # the structural pass says what values exist -- never what they mean
    assert all(cv.meaning is None for c in coded for cv in c.coded_values)
    assert all(isinstance(cv.code, str) for c in coded for cv in c.coded_values)

    fireplace = next(c for c in snap.columns if c.id == "fireclaim.fireplace")
    assert {cv.code for cv in fireplace.coded_values} == {"yes", "no"}


def test_pipeline_persists_and_reloads(snap, tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, snap)
    assert load_snapshot(con, snap.version) == snap
    assert current_version(con, "acme") == snap.version


def test_pipeline_is_idempotent(snap):
    assert enrich_structural(_adapter(), "acme").version == snap.version

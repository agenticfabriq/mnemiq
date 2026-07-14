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


def test_the_snapshot_carries_profile_counts(snap):
    by_id = {c.id: c for c in snap.columns}
    claimnumber = by_id["fireclaim.claimnumber"]

    # the trap column: 820 rows, every one NULL -- the snapshot must say so
    assert claimnumber.row_count == 820
    assert claimnumber.null_count == 820
    assert claimnumber.distinct_count == 0


def test_declared_fks_become_relationships(tmp_path):
    import sqlite3

    from mnemiq.adapters.sqlite import SQLiteAdapter
    from mnemiq.enrichment.pipeline import enrich_structural

    # `dist` is a differently-named, non-_id key -> naming inference cannot produce it,
    # so a relationship here proves the DECLARED path populated it.
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE district (district_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE client (id INTEGER PRIMARY KEY, dist INTEGER REFERENCES district(district_id));
        INSERT INTO district VALUES (1,'NY'),(2,'LA');
        INSERT INTO client VALUES (1,1),(2,2);
        """
    )
    con.commit()
    con.close()

    snap = enrich_structural(SQLiteAdapter(str(path)), "shop")
    rel = next(r for r in snap.relationships if r.from_ == "client")
    assert rel.to == "district"
    assert (rel.join_keys[0].left, rel.join_keys[0].right) == ("dist", "district_id")

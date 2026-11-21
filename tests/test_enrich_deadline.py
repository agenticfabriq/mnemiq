"""Enrich sends every statement with the statement time limit, and names what the limit cut.

`EnrichDeadline` is the adapter enrich reads through (`_enrichment_source`): each `execute` carries
`enrich_statement_timeout_s` (600 by default, 0 for none). A statement the limit cuts still raises, as it did; first it
is recorded by the tables it reads, and enrich turns the record into a job and a warning naming them.
"""
from __future__ import annotations

import sqlite3

import duckdb
import pytest

from mnemiq.adapters.base import QueryTimedOut
from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.deadline import EnrichDeadline, time_limit_job, time_limit_warning
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.semantic.values import build_value_index


def _db(tmp_path):
    path = tmp_path / "s.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, region TEXT);"
        "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);"
        + "".join(f"INSERT INTO orders VALUES ({i}, '{('OPEN', 'SHUT')[i % 2]}', '{('N', 'S', 'E')[i % 3]}');"
                  for i in range(60))
        + "".join(f"INSERT INTO notes VALUES ({i}, 'note {i}');" for i in range(10)))
    con.close()
    return str(path)


class _Recording(SQLiteAdapter):
    """A SQLite source that records the time limit each statement arrives with, and can time out a table's reads."""

    def __init__(self, path, cut: str | None = None):
        super().__init__(path)
        self.limits: list = []
        self._cut = cut

    def execute(self, sql, timeout_s=None):
        self.limits.append(timeout_s)
        if self._cut and self._cut in sql.lower():
            raise QueryTimedOut(f"query timed out after {timeout_s}s")
        return super().execute(sql, timeout_s=timeout_s)


def test_enrich_statements_carry_the_deadline(tmp_path):
    source = _Recording(_db(tmp_path))
    adapter = EnrichDeadline(source, 600)
    snap = enrich_structural(adapter, "src")
    build_value_index(adapter, snap, duckdb.connect())
    assert source.limits, "enrich read nothing"
    assert set(source.limits) == {600}


def test_zero_means_no_deadline(tmp_path):
    source = _Recording(_db(tmp_path))
    enrich_structural(EnrichDeadline(source, 0), "src")
    assert source.limits and set(source.limits) == {None}


def test_a_source_whose_execute_takes_no_deadline_is_read_unbounded(tmp_path):
    class Old(SQLiteAdapter):
        def execute(self, sql):
            return super().execute(sql)

    path = _db(tmp_path)
    adapter = EnrichDeadline(Old(path), 600)
    assert adapter.unbounded
    assert enrich_structural(adapter, "src").columns
    assert not EnrichDeadline(_Recording(path), 600).unbounded
    assert not EnrichDeadline(Old(path), 0).unbounded, "no limit asked, so none is missing"


def test_a_timed_out_statement_is_recorded_by_its_tables_and_raised(tmp_path):
    adapter = EnrichDeadline(_Recording(_db(tmp_path), cut="notes"), 600)
    with pytest.raises(QueryTimedOut):
        adapter.execute('SELECT count(*) FROM "notes"')
    assert adapter.timed_out == [("notes",)]


def test_a_timed_out_statement_is_named_in_a_job_and_a_warning(tmp_path):
    adapter = EnrichDeadline(_Recording(_db(tmp_path), cut="notes"), 600)
    snap = enrich_structural(adapter, "src")
    job = time_limit_job(adapter, "src")
    assert job is not None and job.id == "enrich:time-limit" and job.status == "partial"
    assert "notes" in job.detail and "600" in job.detail
    warning = time_limit_warning(adapter, "src")
    assert warning.startswith("WARNING:") and "notes" in warning and "600 s" in warning
    assert "orders" in {c.object_id for c in snap.columns}, "the table the limit did not cut is profiled"


def test_nothing_cut_means_no_job_and_no_warning(tmp_path):
    adapter = EnrichDeadline(_Recording(_db(tmp_path)), 600)
    enrich_structural(adapter, "src")
    assert time_limit_job(adapter, "src") is None
    assert time_limit_warning(adapter, "src") is None


def test_enrich_reads_its_source_through_the_deadline(monkeypatch, tmp_path):
    from mnemiq.cli import _enrichment_source
    from mnemiq.config import Settings

    source = _Recording(_db(tmp_path))
    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: source)
    manifest = tmp_path / "sources.json"
    manifest.write_text('[{"id": "src", "kind": "sqlite", "target": "x.sqlite", "catalog": "src", "schema": "main"}]')
    settings = Settings(sources_path=str(manifest), store_path=str(tmp_path / "store.duckdb"),
                        enrich_statement_timeout_s=120)
    _, adapter = _enrichment_source(settings)
    assert isinstance(adapter, EnrichDeadline) and adapter.timeout_s == 120
    assert Settings(sources_path=str(manifest)).enrich_statement_timeout_s == 600

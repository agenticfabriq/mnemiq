"""Enrich sends every statement with the statement time limit, and names what the limit cut.

`EnrichDeadline` is the adapter enrich reads through (`_enrichment_source`): each `execute` carries
`enrich_statement_timeout_s` (600 by default, 0 for none). A statement the limit cuts still raises, as it did; first it
is recorded by the tables it reads, and enrich turns the record into a job and a warning naming them.
"""
from __future__ import annotations

import re
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
        if self._cut and (self._cut(sql) if callable(self._cut) else self._cut in sql.lower()):
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


def test_exact_mode_excludes_a_table_whose_profile_timed_out(tmp_path):
    adapter = EnrichDeadline(_Recording(_db(tmp_path), cut="notes"), 600)
    snap = enrich_structural(adapter, "src")
    jobs = {j.id: j for j in snap.jobs}
    assert jobs["profile:notes"].status == "failed"
    assert "time limit" in jobs["profile:notes"].detail and "600" in jobs["profile:notes"].detail
    assert "notes" not in {c.object_id for c in snap.columns}
    assert jobs["profile:orders"].status == "done"


def test_a_cut_grounding_statement_skips_its_item(tmp_path):
    from mnemiq.enrichment.grounding import ground_codes

    path = tmp_path / "g.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, status_name TEXT);"
                      + "".join(f"INSERT INTO orders VALUES ({i}, '{'AB'[i % 2]}', '{('Active', 'Blocked')[i % 2]}');"
                                for i in range(40)))
    con.close()
    # Grounding's functional-dependency check alone: `SELECT "c", count(DISTINCT "d") ...` (a partition read starts
    # `SELECT "c", count(*)`, and ends the table's profile when cut).
    functional_check = re.compile(r'^SELECT "\w+", count\(DISTINCT "\w+"\) FROM').match
    adapter = EnrichDeadline(_Recording(str(path), cut=functional_check), 600)
    snap = ground_codes(adapter, enrich_structural(adapter, "src"), dictionary=None, ontology=None)
    status = next(c for c in snap.columns if c.name == "status")
    assert status.coded_values and not any(v.meaning for v in status.coded_values), "the cut check grounded nothing"
    assert {j.id: j for j in snap.jobs}["profile:orders"].status == "done", "the table stays profiled"
    assert "orders" in time_limit_job(adapter, "src").detail


def test_a_timeout_ends_the_tables_profile_without_retrying_column_by_column(tmp_path):
    # The batched profile falls back to one statement per column when it fails. A timeout says the table is too slow
    # to read, not that one column cannot be counted, so each retry would reach the limit again: 184 statements of 600 s
    # on a 183-column table. The timeout ends the profile instead.
    from mnemiq.catalog import introspect
    from mnemiq.enrichment.profiling import profile_table

    source = _Recording(_db(tmp_path), cut="notes")
    notes = next(t for t in introspect(source) if t.name == "notes")
    with pytest.raises(QueryTimedOut):
        profile_table(EnrichDeadline(source, 600), notes)
    assert len(source.limits) == 1, "nothing after the cut statement was sent"


def test_a_timed_out_partition_read_ends_the_tables_profile(tmp_path):
    from mnemiq.catalog import introspect
    from mnemiq.enrichment.profiling import profile_table

    path = tmp_path / "p.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE facts (id INTEGER PRIMARY KEY, current INTEGER, region TEXT, item TEXT);"
                      + "".join(f"INSERT INTO facts VALUES ({i}, {i % 2}, '{'NS'[i % 2]}', 'i{i % 7}');"
                                for i in range(60)))
    con.close()
    source = _Recording(str(path), cut="is not null group by")
    facts = next(t for t in introspect(source) if t.name == "facts")
    with pytest.raises(QueryTimedOut):
        profile_table(EnrichDeadline(source, 600), facts)

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
        self.cut_sent = 0

    def execute(self, sql, timeout_s=None):
        self.limits.append(timeout_s)
        if self._cut and (self._cut(sql) if callable(self._cut) else self._cut in sql.lower()):
            self.cut_sent += 1
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


def _shop(tmp_path):
    """A parent named by two children's key-like columns, so join inference checks the parent's key once per child."""
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE customer (customer_id INTEGER, name TEXT);"
        "CREATE TABLE orders (order_id INTEGER, customer_id INTEGER);"
        "CREATE TABLE invoice (invoice_id INTEGER, customer_id INTEGER);"
        + "".join(f"INSERT INTO customer VALUES ({i}, 'c{i}');" for i in range(20))
        + "".join(f"INSERT INTO orders VALUES ({i}, {i % 20});" for i in range(40))
        + "".join(f"INSERT INTO invoice VALUES ({i}, {i % 20});" for i in range(30)))
    con.close()
    return str(path)


def test_a_table_whose_profile_timed_out_is_not_read_by_join_inference(tmp_path):
    # Excluded from the model, the table has no join to offer, and each inference check of it would reach the limit
    # again (Codex's review of #276: the same timed-out count sent three times).
    source = _Recording(_shop(tmp_path), cut=lambda sql: 'FROM "customer"' in sql)
    snap = enrich_structural(EnrichDeadline(source, 600), "src")
    assert {j.id: j for j in snap.jobs}["profile:customer"].status == "failed"
    assert source.cut_sent == 1, "only the profile's cut statement read the excluded table"


def test_join_inference_does_not_repeat_a_timed_out_count(tmp_path):
    # The parent profiles, but inference's own count of its key reaches the limit: asked for by both children, it is
    # sent once.
    key_count = 'SELECT count(*), count(DISTINCT "customer_id"), count("customer_id") FROM "customer"'
    source = _Recording(_shop(tmp_path), cut=lambda sql: sql == key_count)
    snap = enrich_structural(EnrichDeadline(source, 600), "src")
    assert {j.id: j for j in snap.jobs}["profile:customer"].status == "done"
    assert source.cut_sent == 1


def test_enrich_bounds_a_postgres_source_on_the_server(monkeypatch, tmp_path):
    # A Postgres source is read through DuckDB's attach, whose interrupt does not stop the server's scan; the bound that
    # does is the connection's statement timeout, which adapter_for sets from `statement_timeout_s`.
    from mnemiq.cli import _enrichment_source
    from mnemiq.config import Settings

    seen = {}
    source = _Recording(_db(tmp_path))

    def adapter_for(spec, settings=None, **kwargs):
        seen.update(kwargs)
        return source

    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", adapter_for)
    manifest = tmp_path / "sources.json"
    manifest.write_text('[{"id": "src", "kind": "postgres", "target": "postgresql://h/db", "catalog": "src", '
                        '"schema": "public"}]')
    _enrichment_source(Settings(sources_path=str(manifest), store_path=str(tmp_path / "s.duckdb"),
                                enrich_statement_timeout_s=120))
    assert seen.get("statement_timeout_s") == 120
    seen.clear()
    _enrichment_source(Settings(sources_path=str(manifest), store_path=str(tmp_path / "s.duckdb"),
                                enrich_statement_timeout_s=0))
    assert seen.get("statement_timeout_s") is None


def test_a_bounded_arrow_read_is_capped_by_the_limit_and_named(tmp_path):
    # The examples phase reads through `execute_arrow` with its own 30 s (Codex's second review of #276): the enrich
    # limit caps it, and a cut there is named like any other.
    class Arrow(_Recording):
        def __init__(self, path):
            super().__init__(path)
            self.arrow_limits: list = []

        def execute_arrow(self, sql, timeout_s=None):
            self.arrow_limits.append(timeout_s)
            if "notes" in sql:
                raise QueryTimedOut(f"query timed out after {timeout_s}s")
            return super().execute_arrow(sql, timeout_s=timeout_s)

    source = Arrow(_db(tmp_path))
    adapter = EnrichDeadline(source, 5)
    adapter.execute_arrow('SELECT count(*) FROM "orders"', timeout_s=30)
    adapter.execute_arrow('SELECT count(*) FROM "orders"', timeout_s=2)
    adapter.execute_arrow('SELECT count(*) FROM "orders"')
    assert source.arrow_limits == [5, 2, 5], "the tighter of the caller's bound and the enrich limit"
    with pytest.raises(QueryTimedOut):
        adapter.execute_arrow('SELECT count(*) FROM "notes"', timeout_s=30)
    assert adapter.timed_out == [("notes",)]
    (tmp_path / "b").mkdir()
    unbounded = EnrichDeadline(Arrow(_db(tmp_path / "b")), 0)
    unbounded.execute_arrow('SELECT count(*) FROM "orders"', timeout_s=30)
    assert unbounded._source.arrow_limits == [30], "no enrich limit: the caller's bound stands"

"""A DuckDB file's own table and column comments, read the way Oracle's are.

A store built from a DuckDB copy of an Oracle schema lost every comment the schema carried: only the Oracle adapter
read any, so the copy route traded a person's documentation for speed. The copy writes them as DuckDB `COMMENT ON`
statements, and the DuckDB adapter reads them back from DuckDB's own catalogue for a DuckDB-file source. Every other
kind the adapter attaches (SQLite, Postgres) reports none here.
"""
from __future__ import annotations

import sqlite3

import duckdb

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.enrichment.pipeline import enrich_structural


def _shop(tmp_path) -> str:
    path = str(tmp_path / "shop.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE orders (id INTEGER, status VARCHAR)")
    con.execute("CREATE TABLE customer (id INTEGER, name VARCHAR)")
    con.execute("INSERT INTO orders VALUES (1, 'O'), (2, 'C')")
    con.execute("INSERT INTO customer VALUES (1, 'Ann')")
    con.execute("COMMENT ON TABLE orders IS 'One row per order'")
    con.execute("COMMENT ON COLUMN orders.status IS 'O open, C closed'")
    con.close()
    return path


def test_a_duckdb_file_reports_its_comments(tmp_path):
    adapter = DuckDBAdapter.duckdb(_shop(tmp_path))
    assert adapter.comments(["orders", "customer"]) == (
        {"orders": "One row per order"}, {("orders", "status"): "O open, C closed"})


def test_only_the_tables_asked_for_are_reported(tmp_path):
    adapter = DuckDBAdapter.duckdb(_shop(tmp_path))
    assert adapter.comments(["customer"]) == ({}, {})


def test_enrich_carries_a_duckdb_files_comments_to_the_snapshot(tmp_path):
    snap = enrich_structural(DuckDBAdapter.duckdb(_shop(tmp_path)), "src")
    bindings = {b.object_id: b for b in snap.source_bindings}
    columns = {(c.object_id, c.name): c for c in snap.columns}
    assert bindings["orders"].comment == "One row per order"
    assert columns[("orders", "status")].comment == "O open, C closed"
    assert bindings["customer"].comment is None
    assert columns[("orders", "id")].comment is None


def test_an_attached_sqlite_source_reports_no_comments(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE orders (id INTEGER, status TEXT); INSERT INTO orders VALUES (1, 'O');")
    con.close()
    assert DuckDBAdapter.sqlite(str(path)).comments(["orders"]) == ({}, {})

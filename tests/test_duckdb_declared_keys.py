"""A DuckDB file's declared foreign keys, from the `mnemiq.declared_keys` table the Oracle copy writes.

A DuckDB file reported no foreign keys: the adapter read keys from Postgres only, so a store built on a DuckDB copy of an
Oracle schema had only the joins inference could find. The copy writes Oracle's declared keys into
`mnemiq.declared_keys` (outside `main`, so no reader of the source's tables sees it as one), and the adapter's
`foreign_keys` reads that table when the file holds it. Nothing is enforced by it, so an orphan row cannot stop a copy.
"""
from __future__ import annotations

import duckdb

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.scoped import TableScopedAdapter
from mnemiq.enrichment.pipeline import enrich_structural

_KEYS_DDL = ("CREATE TABLE mnemiq.declared_keys (from_table VARCHAR, from_column VARCHAR, to_table VARCHAR, "
             "to_column VARCHAR, constraint_name VARCHAR)")


def _shop(tmp_path, keys: list[tuple] | None) -> str:
    path = str(tmp_path / "shop.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER)")
    con.execute("CREATE TABLE customer (id INTEGER, name VARCHAR)")
    con.execute("INSERT INTO customer VALUES (1, 'Ann'), (2, 'Bo')")
    con.execute("INSERT INTO orders VALUES (10, 1), (11, 2), (12, 1)")
    if keys is not None:
        con.execute("CREATE SCHEMA mnemiq")
        con.execute(_KEYS_DDL)
        con.executemany("INSERT INTO mnemiq.declared_keys VALUES (?, ?, ?, ?, ?)", keys)
    con.close()
    return path


_KEY = ("orders", "customer_id", "customer", "id", "FK_ORD_CUST")


def test_a_duckdb_file_reports_the_keys_its_keys_table_declares(tmp_path):
    assert DuckDBAdapter.duckdb(_shop(tmp_path, [_KEY])).foreign_keys() == [_KEY]


def test_a_composite_key_keeps_its_column_pairs(tmp_path):
    # Two keys of one table, one composite: ordered by table, constraint, column, each pair kept whole.
    keys = [("orders", "region", "store", "region", "FK_B"), ("orders", "customer_id", "customer", "id", "FK_A"),
            ("orders", "store_no", "store", "no", "FK_B")]
    assert DuckDBAdapter.duckdb(_shop(tmp_path, keys)).foreign_keys() == [
        ("orders", "customer_id", "customer", "id", "FK_A"),
        ("orders", "region", "store", "region", "FK_B"),
        ("orders", "store_no", "store", "no", "FK_B"),
    ]


def test_enrich_records_a_declared_key_from_a_duckdb_file(tmp_path):
    snap = enrich_structural(DuckDBAdapter.duckdb(_shop(tmp_path, [_KEY])), "src")
    assert "orders.customer_id->customer" in {r.id for r in snap.relationships}


def test_the_keys_table_is_not_enriched_as_a_table(tmp_path):
    adapter = DuckDBAdapter.duckdb(_shop(tmp_path, [_KEY]))
    assert "declared_keys" not in adapter.introspect()
    assert not [c for c in adapter.list_columns() if c[0] == "declared_keys"]
    snap = enrich_structural(adapter, "src")
    assert "declared_keys" not in {b.object_id for b in snap.source_bindings}


def test_a_duckdb_file_without_a_keys_table_reports_none(tmp_path):
    assert DuckDBAdapter.duckdb(_shop(tmp_path, None)).foreign_keys() == []


def test_a_scoped_source_keeps_only_keys_with_both_ends_in_scope(tmp_path):
    source = DuckDBAdapter.duckdb(_shop(tmp_path, [_KEY]))
    assert TableScopedAdapter(source, ("orders",), "src").foreign_keys() == []
    assert TableScopedAdapter(source, ("orders", "customer"), "src").foreign_keys() == [_KEY]

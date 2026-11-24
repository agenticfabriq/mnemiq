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
from mnemiq.catalog import introspect
from mnemiq.enrichment.joins import relationships_from_foreign_keys
from mnemiq.enrichment.pipeline import enrich_structural

_KEYS_DDL = ("CREATE TABLE mnemiq.declared_keys (from_table VARCHAR, from_column VARCHAR, to_table VARCHAR, "
             "to_column VARCHAR, constraint_name VARCHAR, position INTEGER)")


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
        con.executemany("INSERT INTO mnemiq.declared_keys VALUES (?, ?, ?, ?, ?, ?)", keys)
    con.close()
    return path


_KEY = ("orders", "customer_id", "customer", "id", "FK_ORD_CUST")
_ROW = (*_KEY, 1)  # as stored: the pair's position in its key


def test_a_duckdb_file_reports_the_keys_its_keys_table_declares(tmp_path):
    assert DuckDBAdapter.duckdb(_shop(tmp_path, [_ROW])).foreign_keys() == [_KEY]


def test_a_composite_key_keeps_its_column_pairs(tmp_path):
    # Two keys of one table, one composite: ordered by table, constraint, position, each pair kept whole.
    keys = [("orders", "store_no", "store", "no", "FK_B", 2), ("orders", "customer_id", "customer", "id", "FK_A", 1),
            ("orders", "region", "store", "region", "FK_B", 1)]
    assert DuckDBAdapter.duckdb(_shop(tmp_path, keys)).foreign_keys() == [
        ("orders", "customer_id", "customer", "id", "FK_A"),
        ("orders", "region", "store", "region", "FK_B"),
        ("orders", "store_no", "store", "no", "FK_B"),
    ]


class _OracleOrder:
    """Declared keys as `OracleAdapter.foreign_keys` gives them: by table, constraint and `ALL_CONS_COLUMNS.position`."""

    def __init__(self, rows):
        self._rows = rows

    def foreign_keys(self):
        return list(self._rows)


def test_a_composite_key_keeps_its_declared_column_order(tmp_path):
    # Declared on (zone, area), against the alphabet. Sorted by column name the pairs came back (area, zone), so the
    # relationship's id and its join keys' order differed from a run on Oracle, which reads them by position.
    declared = [("orders", "zone", "store", "zone", "FK_STORE", 1), ("orders", "area", "store", "area", "FK_STORE", 2)]
    path = _shop(tmp_path, [declared[1], declared[0]])  # stored out of order: only the position decides
    con = duckdb.connect(path)
    con.execute("CREATE TABLE store (zone VARCHAR, area VARCHAR)")
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    assert adapter.foreign_keys() == [row[:5] for row in declared]
    catalog = introspect(adapter)
    copied = relationships_from_foreign_keys(adapter, catalog)
    on_oracle = relationships_from_foreign_keys(_OracleOrder(row[:5] for row in declared), catalog)
    assert [r.id for r in copied] == [r.id for r in on_oracle] == ["orders.zone+area->store"]
    assert [(k.left, k.right) for k in copied[0].join_keys] == [("zone", "zone"), ("area", "area")]


def test_enrich_records_a_declared_key_from_a_duckdb_file(tmp_path):
    snap = enrich_structural(DuckDBAdapter.duckdb(_shop(tmp_path, [_ROW])), "src")
    assert "orders.customer_id->customer" in {r.id for r in snap.relationships}


def test_the_keys_table_is_not_enriched_as_a_table(tmp_path):
    adapter = DuckDBAdapter.duckdb(_shop(tmp_path, [_ROW]))
    assert "declared_keys" not in adapter.introspect()
    assert not [c for c in adapter.list_columns() if c[0] == "declared_keys"]
    snap = enrich_structural(adapter, "src")
    assert "declared_keys" not in {b.object_id for b in snap.source_bindings}


def test_a_duckdb_file_without_a_keys_table_reports_none(tmp_path):
    assert DuckDBAdapter.duckdb(_shop(tmp_path, None)).foreign_keys() == []


def test_a_scoped_source_keeps_only_keys_with_both_ends_in_scope(tmp_path):
    source = DuckDBAdapter.duckdb(_shop(tmp_path, [_ROW]))
    assert TableScopedAdapter(source, ("orders",), "src").foreign_keys() == []
    assert TableScopedAdapter(source, ("orders", "customer"), "src").foreign_keys() == [_KEY]

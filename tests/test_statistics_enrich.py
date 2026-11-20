"""`enrich_structural` in statistics mode: each table profiled from the dictionary unless the manifest's "exact_tables"
names it, and no data read from a table it estimates.

The source is SQLite with the statistics profile's dictionary reads added (`_Recording`), so exact tables are profiled by
the real queries, and every statement sent for data is recorded.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from mnemiq.adapters.dictionary import ColumnStatistics, HistogramEndpoint, TableSize
from mnemiq.adapters.scoped import TablesNotFound
from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.contract.semantic import ColumnEstimate
from mnemiq.enrichment.pipeline import enrich_structural

SHOP = """
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, region VARCHAR2(10));
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
                     status VARCHAR2(10), status_name VARCHAR2(10));
CREATE TABLE product (product_id INTEGER PRIMARY KEY, label VARCHAR2(10));
CREATE TABLE order_lines (line_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, sku VARCHAR2(10));
INSERT INTO customers VALUES (1, 'N'), (2, 'S'), (3, 'N');
INSERT INTO product VALUES (1, 'p1'), (2, 'p2');
INSERT INTO orders VALUES (1, 1, 'A', 'Active'), (2, 2, 'A', 'Active'), (3, 3, 'B', 'Blocked'), (4, 1, 'A', 'Active');
INSERT INTO order_lines VALUES (1, 1, 1, 'x'), (2, 1, 2, 'y'), (3, 2, 1, 'x'), (4, 3, 2, 'z'), (5, 4, 1, 'x');
"""


def _stats(column, distinct, histogram="NONE", rows=4, low=None, high=None):
    return ColumnStatistics(column=column, distinct=distinct, nulls=0, histogram=histogram,
                            buckets=distinct if histogram == "FREQUENCY" else None, sample_size=rows,
                            analyzed="2026-11-01", low=low, high=high)


# What a dictionary would say about SHOP: every table analysed, `orders.status` with a frequency histogram.
DICTIONARY = {
    "customers": (TableSize(3, 20, "2026-11-01", False),
                  [_stats("customer_id", 3, rows=3), _stats("region", 2, rows=3)], {}, frozenset({"customer_id"})),
    "orders": (TableSize(4, 40, "2026-11-01", False),
               [_stats("id", 4), _stats("customer_id", 3), _stats("status", 2, "FREQUENCY", low=b"A", high=b"B"),
                _stats("status_name", 2)],
               {"status": [HistogramEndpoint(3, None, "A", b"A"), HistogramEndpoint(4, None, "B", b"B")]},
               frozenset({"id"})),
    "product": (TableSize(2, 10, "2026-11-01", False),
                [_stats("product_id", 2, rows=2), _stats("label", 2, rows=2)], {}, frozenset({"product_id"})),
    "order_lines": (TableSize(5, 30, "2026-11-01", False),
                    [_stats("line_id", 5, rows=5), _stats("order_id", 4, rows=5), _stats("product_id", 2, rows=5),
                     _stats("sku", 3, rows=5)], {}, frozenset({"line_id"})),
}


class _Recording:
    """SQLite plus the four dictionary reads, recording every statement sent for data."""

    def __init__(self, inner, dictionary=None):
        self._inner = inner
        self.dictionary = DICTIONARY if dictionary is None else dictionary
        self.sql: list[str] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute(self, sql, *args, **kwargs):
        self.sql.append(sql)
        return self._inner.execute(sql, *args, **kwargs)

    def execute_arrow(self, sql, *args, **kwargs):
        self.sql.append(sql)
        return self._inner.execute_arrow(sql, *args, **kwargs)

    def table_size(self, table):
        return self.dictionary[table][0]

    def column_statistics(self, table):
        return self.dictionary[table][1]

    def histogram(self, table, column):
        return self.dictionary[table][2].get(column, [])

    def unique_columns(self, table):
        return self.dictionary[table][3]


@pytest.fixture
def shop(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SHOP)
    con.commit()
    con.close()
    return _Recording(SQLiteAdapter(str(path)))


def _columns(snap, table):
    return {c.name: c for c in snap.columns if c.object_id == table}


def _touches(sql: str, table: str) -> bool:
    return f'"{table}"' in sql or f" {table} " in f" {sql} "


def test_statistics_mode_reads_no_table_data(shop):
    snap = enrich_structural(shop, "shop", statistics=())
    assert shop.sql == []
    assert {c.object_id for c in snap.columns} == {"customers", "orders", "product", "order_lines"}


def test_estimated_columns_have_no_exact_counts(shop):
    status = _columns(enrich_structural(shop, "shop", statistics=()), "orders")["status"]
    assert (status.row_count, status.distinct_count, status.null_count, status.unique_within) == (None,) * 4
    assert status.estimate == ColumnEstimate(rows=4, distinct=2, nulls=0, analyzed="2026-11-01", sampled=False,
                                             stale=False)
    assert [cv.code for cv in status.coded_values] == ["A", "B"]


def test_declared_keys_are_recorded_on_estimated_columns(shop):
    orders = _columns(enrich_structural(shop, "shop", statistics=()), "orders")
    assert (orders["id"].declared_unique, orders["status"].declared_unique) == (True, False)


def test_estimated_tables_are_recorded_as_estimated(shop):
    snap = enrich_structural(shop, "shop", statistics=("orders",))
    jobs = {j.id: j for j in snap.jobs}
    assert (jobs["profile:customers"].status, jobs["profile:customers"].detail) == ("done", "estimated")
    assert (jobs["profile:orders"].status, jobs["profile:orders"].detail) == ("done", None)


def test_exact_tables_are_profiled_exactly(shop):
    snap = enrich_structural(shop, "shop", statistics=("orders",))
    orders, customers = _columns(snap, "orders"), _columns(snap, "customers")
    assert orders["status"].distinct_count == 2 and orders["status"].estimate is None
    assert orders["status"].declared_unique is None
    assert customers["region"].distinct_count is None and customers["region"].estimate.distinct == 2
    assert shop.sql and all(_touches(sql, "orders") for sql in shop.sql), shop.sql
    assert not any(_touches(sql, t) for sql in shop.sql for t in ("customers", "product", "order_lines"))


def test_exact_tables_match_without_regard_to_case(shop):
    snap = enrich_structural(shop, "shop", statistics=("ORDE?S",))
    assert _columns(snap, "orders")["status"].distinct_count == 2
    assert _columns(snap, "order_lines")["sku"].distinct_count is None


def test_exact_tables_matching_nothing_is_refused(shop):
    """A typo must not quietly estimate the table the operator meant to count exactly."""
    with pytest.raises(TablesNotFound, match=r'"exact_tables".*\bordrs\b'):
        enrich_structural(shop, "shop", statistics=("orders", "ordrs"))
    assert shop.sql == []


def test_estimated_tables_keep_declared_foreign_keys_and_infer_none(shop):
    """Exactly profiled, the join from order_lines to product is inferred from the data; estimated, it is not looked
    for, and the key orders declares to customers is kept either way."""
    exact = enrich_structural(shop, "shop")
    assert any(r.from_ == "order_lines" and r.to == "product" for r in exact.relationships)
    shop.sql.clear()
    snap = enrich_structural(shop, "shop", statistics=())
    assert [(r.from_, r.to) for r in snap.relationships] == [("orders", "customers")]
    assert shop.sql == []


def test_inference_still_runs_between_exact_tables(shop):
    snap = enrich_structural(shop, "shop", statistics=("product", "order_lines"))
    assert any(r.from_ == "order_lines" and r.to == "product" for r in snap.relationships)
    half = enrich_structural(shop, "shop", statistics=("order_lines",))
    assert not any(r.from_ == "order_lines" for r in half.relationships), "the parent is estimated: not looked for"


def test_no_partition_profile_is_recorded_estimated(shop):
    """Decision B: with no table profiled exactly, the partitions job says `estimated`, so the guard's auto setting can
    run on the estimates (`partitions_profiled`)."""
    snap = enrich_structural(shop, "shop", statistics=())
    assert next(j for j in snap.jobs if j.id == "profile:partitions").status == "estimated"
    mixed = enrich_structural(shop, "shop", statistics=("orders",))
    assert next(j for j in mixed.jobs if j.id == "profile:partitions").status == "done"


def test_correlated_grounding_skips_estimated_columns(shop):
    """Codes from a histogram are not grounded from a label column: that would read the table's rows."""
    from mnemiq.enrichment.grounding import ground_from_correlated

    exact = enrich_structural(shop, "shop")
    assert ground_from_correlated(shop, exact)["orders.status"] == {"A": "Active", "B": "Blocked"}
    snap = enrich_structural(shop, "shop", statistics=())
    shop.sql.clear()
    assert ground_from_correlated(shop, snap) == {}
    assert shop.sql == []


def test_value_index_skips_estimated_columns(shop):
    import duckdb

    from mnemiq.semantic.values import build_value_index

    snap = enrich_structural(shop, "shop", statistics=())
    build_value_index(shop, snap, duckdb.connect())
    assert shop.sql == []


def test_estimated_columns_are_never_harvested_without_values(shop):
    snap = enrich_structural(shop, "shop", harvest_values=False, statistics=())
    assert all(not c.coded_values for c in snap.columns)


# The statements `enrich_structural(shop, "shop")` sends with no setting, and the snapshot's version, pinned before
# statistics mode was added (this branch at 206eeb96, whose `enrich_structural` sends what main 5f0ca71a sends).
TODAYS_STATEMENTS = ("de846434b63c2801", 11)
TODAYS_VERSION = "bae721c4ab36"


def test_absent_setting_sends_todays_queries(shop):
    snap = enrich_structural(shop, "shop")
    digest = hashlib.sha256("\n".join(shop.sql).encode()).hexdigest()[:16]
    assert (digest, len(shop.sql)) == TODAYS_STATEMENTS, "\n".join(shop.sql)
    assert snap.version == TODAYS_VERSION


# --- after the structural pass: nothing enrich runs next reads an estimated table -----------------

STATUS_SHOP = """
CREATE TABLE order_status (status_id INTEGER PRIMARY KEY, status_name VARCHAR2(10));
CREATE TABLE orders (id INTEGER PRIMARY KEY, status_id INTEGER REFERENCES order_status(status_id), note VARCHAR2(10));
INSERT INTO order_status VALUES (1, 'open'), (2, 'shipped');
INSERT INTO orders VALUES (1, 1, 'x'), (2, 2, 'y'), (3, 1, 'x');
"""
STATUS_DICTIONARY = {
    "order_status": (TableSize(2, 10, "2026-11-01", False),
                     [_stats("status_id", 2, rows=2), _stats("status_name", 2, rows=2)], {},
                     frozenset({"status_id"})),
    "orders": (TableSize(3, 10, "2026-11-01", False),
               [_stats("id", 3, rows=3), _stats("status_id", 2, rows=3), _stats("note", 2, rows=3)], {},
               frozenset({"id"})),
}


def test_the_steps_after_the_structural_pass_read_no_estimated_table(tmp_path):
    """Grounding and the value index run on every enrich. Lookup grounding joined an exactly profiled child to its
    parent's label column, and so read the rows of a parent profiled from statistics."""
    import duckdb

    from mnemiq.enrichment.grounding import ground_codes
    from mnemiq.semantic.values import build_value_index

    path = tmp_path / "status.sqlite"
    con = sqlite3.connect(path)
    con.executescript(STATUS_SHOP)
    con.commit()
    con.close()
    source = _Recording(SQLiteAdapter(str(path)), STATUS_DICTIONARY)

    exact = ground_codes(source, enrich_structural(source, "shop"))
    assert any(_touches(sql, "order_status") and _touches(sql, "orders") for sql in source.sql), \
        "control: exactly profiled, the label is looked up across the join"
    assert {cv.meaning for c in exact.columns if c.id == "orders.status_id" for cv in c.coded_values} >= {"open"}

    source.sql.clear()
    snap = ground_codes(source, enrich_structural(source, "shop", statistics=("orders",)))
    build_value_index(source, snap, duckdb.connect())
    assert any(_touches(sql, "orders") for sql in source.sql), "control: the exact table is still read"
    assert not any(_touches(sql, "order_status") for sql in source.sql), source.sql

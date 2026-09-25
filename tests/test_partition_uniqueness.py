"""Key uniqueness per partition (register M109): the data, not the syntax, tells a current-row flag
from an ordinary binary filter.

Among the rows where `is_current = 1`, a type-2 dimension's `customer_id` is unique; among the rows
where `is_returned = 0`, a line item's `product_id` still repeats. Syntax cannot tell those filters
apart -- two syntactic exemptions let an ordinary filter switch the fan-out guard off for an
inflated sum -- but one GROUP BY per two-valued (or nullable date) column at profiling time can.
"""
from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.catalog import introspect
from mnemiq.contract import Column, Snapshot
from mnemiq.enrichment.profiling import profile_table
from mnemiq.sql.fanout_check import check_fanout, key_facts, partition_label
from mnemiq.sql.schema import schema_map


def _db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE dim_customer (customer_id INTEGER, is_current INTEGER, valid_to DATE,
                                   tier TEXT);
        INSERT INTO dim_customer VALUES (1, 0, '2019-01-01', 'gold'), (1, 0, '2020-01-01', 'silver'),
                                        (1, 1, NULL, 'gold'), (2, 1, NULL, 'bronze');
        CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, amount INTEGER);
        INSERT INTO orders VALUES (10, 1, 5), (11, 1, 7), (12, 2, 9);
        CREATE TABLE products (product_id INTEGER, unit_price INTEGER);
        INSERT INTO products VALUES (1, 100), (2, 200);
        CREATE TABLE order_items (order_id INTEGER, product_id INTEGER, is_returned INTEGER);
        INSERT INTO order_items VALUES (10, 1, 0), (11, 1, 0), (12, 2, 0), (12, 2, 1);
    """)
    con.commit()
    con.close()
    return SQLiteAdapter(str(path))


def _stats(adapter, name):
    table = next(t for t in introspect(adapter) if t.name == name)
    return {s.column: s for s in profile_table(adapter, table)}


# -- profiling -------------------------------------------------------------------------------------


def test_a_current_row_flag_records_the_key_it_makes_unique(tmp_path):
    stats = _stats(_db(tmp_path), "dim_customer")
    assert "customer_id" in stats["is_current"].unique_within["=1"]
    assert "customer_id" not in stats["is_current"].unique_within.get("=0", []), (
        "customer 1 has two historical rows, so the key repeats among is_current = 0"
    )


def test_an_open_validity_end_records_the_key_it_makes_unique(tmp_path):
    stats = _stats(_db(tmp_path), "dim_customer")
    assert "customer_id" in stats["valid_to"].unique_within["IS NULL"]


def test_an_ordinary_binary_filter_records_nothing_it_does_not_narrow(tmp_path):
    stats = _stats(_db(tmp_path), "order_items")
    within = stats["is_returned"].unique_within or {}
    assert "product_id" not in within.get("=0", []), "kept items still repeat each product"
    assert within["=1"] == [], "a one-row partition makes everything unique by default; not pinned"


def test_a_partition_query_that_fails_costs_only_its_partition(tmp_path):
    adapter = _db(tmp_path)

    class _Failing:
        dialect = adapter.dialect

        def __getattr__(self, name):
            return getattr(adapter, name)

        def execute(self, sql, *a, **kw):
            if "IS NOT NULL GROUP BY" in sql or sql.rstrip().endswith("IS NULL"):
                raise RuntimeError("temp space")  # the partition queries, nothing else
            return adapter.execute(sql, *a, **kw)

    table = next(t for t in introspect(adapter) if t.name == "dim_customer")
    stats = {s.column: s for s in profile_table(_Failing(), table)}
    assert stats["is_current"].unique_within is None
    assert stats["customer_id"].distinct_count == 2, "the table's own counts are unaffected"


# -- key facts and the check -----------------------------------------------------------------------


def test_partition_labels_match_between_profile_values_and_sql_literals():
    assert partition_label(True) == "=true" and partition_label(1) == "=1"
    assert partition_label("Y") == "='Y'" and partition_label(None) == "IS NULL"
    assert partition_label(1.0) == "=1"


def _snapshot():
    def col(t, c, r, d, n, within=None):
        return Column(id=f"{t}.{c}", object_id=t, name=c, row_count=r, distinct_count=d,
                      null_count=n, unique_within=within)
    return Snapshot(version="v", source_id="shop", created_at="t", columns=[
        col("orders", "customer_id", 1000, 200, 0), col("orders", "amount", 1000, 900, 0),
        col("dim_customer", "customer_id", 600, 200, 0),
        col("dim_customer", "is_current", 600, 2, 0, {"=1": ["customer_id"]}),
        col("dim_customer", "valid_to", 600, 300, 200, {"IS NULL": ["customer_id"]}),
        col("dim_customer", "tier", 600, 3, 0),
        col("dim_customer", "band", 600, 2, 0, {"=1": ["customer_id"], "=2": []}),
        col("regions", "k", 50, 50, 0),
    ])


def _check(sql):
    snap = _snapshot()
    return check_fanout(sqlglot.parse_one(sql, read="duckdb"), schema_map(snap), key_facts(snap))


_JOIN = ("SELECT d.tier, SUM(o.amount) FROM orders o JOIN dim_customer d "
         "ON d.customer_id = o.customer_id")


def test_key_facts_carry_the_partitions():
    facts = key_facts(_snapshot())
    assert facts.partitions[("dim_customer", "is_current", "=1")] == frozenset({"customer_id"})
    assert facts[("dim_customer", "customer_id")] is False, "table-wide it still repeats"


@pytest.mark.parametrize("current_row", [
    " AND d.is_current = 1", " AND d.is_current", " AND d.is_current IS TRUE",
    " AND d.valid_to IS NULL", " WHERE d.is_current = 1",
])
def test_a_filter_the_profile_shows_as_one_row_per_key_is_answered(current_row):
    assert _check(_JOIN + current_row + " GROUP BY d.tier") is None


def test_a_filter_on_an_outer_joins_preserved_side_pins_nothing():
    """`LEFT JOIN regions r ON ... AND d.is_current = 1` removes no row of `d`: every history row
    still joins its orders, and pinning `d` from it approved an inflated sum."""
    assert _check("SELECT d.tier, SUM(o.amount) FROM orders o JOIN dim_customer d "
                  "ON d.customer_id = o.customer_id LEFT JOIN regions r "
                  "ON r.k = o.customer_id AND d.is_current = 1 GROUP BY d.tier") is not None


def test_a_filter_on_an_outer_joins_null_supplying_side_pins_it():
    assert _check("SELECT d.tier, SUM(o.amount) FROM orders o LEFT JOIN dim_customer d "
                  "ON d.customer_id = o.customer_id AND d.is_current = 1 GROUP BY d.tier") is None


def test_a_bare_filter_on_a_flag_that_is_not_0_1_pins_nothing():
    """On values {1, 2} a bare `d.band` keeps both partitions; only `= 1` pins the key."""
    assert _check(_JOIN + " AND d.band GROUP BY d.tier") is not None
    assert _check(_JOIN + " AND d.band = 1 GROUP BY d.tier") is None


def test_a_filter_that_keeps_the_history_still_refuses():
    assert _check(_JOIN + " AND d.is_current = 0 GROUP BY d.tier") is not None


def test_without_partition_facts_a_current_row_filter_still_refuses():
    """An older snapshot has no partitions: today's behaviour, a visible refusal."""
    snap = _snapshot()
    bare = {k: v for k, v in key_facts(snap).items()}
    ast = sqlglot.parse_one(_JOIN + " AND d.is_current = 1 GROUP BY d.tier", read="duckdb")
    assert check_fanout(ast, schema_map(snap), bare) is not None


# -- end to end --------------------------------------------------------------------------------------


def test_a_current_row_join_plans_after_structural_enrichment(tmp_path):
    from mnemiq.authz.grants import GrantSet
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import Deferred, plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
    from mnemiq.sql.verdict import Approved

    snapshot = enrich_structural(_db(tmp_path), "shop")
    packet = ContextPacket(
        question="revenue by customer tier",
        cards=[RetrievedCard(object_id="orders", card="TABLE orders", score=1.0),
               RetrievedCard(object_id="dim_customer", card="TABLE dim_customer", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1")
    grants = GrantSet(frozenset({"orders", "dim_customer"}))

    def plan(sql):
        return plan_query(packet, snapshot, grants, FakeGenerator([f'{{"sql": "{sql}"}}']),
                          target="sqlite", dialect="sqlite", max_attempts=1, guard_fanout=True)

    current = plan(_JOIN + " AND d.is_current = 1 GROUP BY d.tier")
    history = plan(_JOIN + " AND d.is_current = 0 GROUP BY d.tier")
    assert isinstance(current, Approved)
    assert isinstance(history, Deferred)

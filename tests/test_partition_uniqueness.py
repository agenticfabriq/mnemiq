"""Key uniqueness per partition (register M109): the data, not the syntax, tells a current-row flag
from an ordinary binary filter.

Among the rows where `is_current = 1`, a type-2 dimension's `customer_id` is unique; among the rows
where `is_returned = 0`, a line item's `product_id` still repeats. Syntax cannot tell those filters
apart -- two syntactic exemptions let an ordinary filter switch the fan-out guard off for an
inflated sum -- but one GROUP BY per two-valued (or nullable date) column at profiling time can.
"""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal

import pytest
import sqlglot

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.catalog import introspect
from mnemiq.contract import Column, Snapshot
from mnemiq.enrichment.profiling import profile_table
from mnemiq.sql.fanout_check import (PARTITIONS_JOB, check_fanout, key_facts, partition_label,
                                     partitions_profiled)
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


def test_two_values_spelled_alike_keep_only_the_facts_both_hold(tmp_path):
    """A driver can hand back 1 and 1.0 as separate groups. One label then names both, and it must
    not carry the facts of whichever group came last."""
    adapter = _db(tmp_path)

    class _Alike:
        dialect = adapter.dialect

        def __getattr__(self, name):
            return getattr(adapter, name)

        def execute(self, sql, *a, **kw):
            rows = adapter.execute(sql, *a, **kw)
            if '"is_current" IS NOT NULL GROUP BY' in sql:  # history (0) first, current (1) last
                return [(1.0 if row[0] == 0 else 1, *row[1:]) for row in rows]
            return rows

    table = next(t for t in introspect(adapter) if t.name == "dim_customer")
    stats = {s.column: s for s in profile_table(_Alike(), table)}
    assert stats["is_current"].unique_within == {"=1": ["tier"]}, (
        "customer_id repeats among the history rows; tier is unique in both groups")


# -- key facts and the check -----------------------------------------------------------------------


def test_partition_labels_match_between_profile_values_and_sql_literals():
    assert partition_label(True) == "=true" and partition_label(1) == "=1"
    assert partition_label("Y") == "='Y'" and partition_label(None) == "IS NULL"
    assert partition_label(1.0) == "=1"


def test_partition_labels_are_exact():
    """A float round trip spelled 2**53 and 2**53 + 1 alike."""
    assert partition_label(2 ** 53) != partition_label(2 ** 53 + 1)
    assert partition_label(2 ** 53 + 1) == "=9007199254740993"
    assert partition_label(Decimal("1.50")) == partition_label(1.5) == "=1.5"
    assert partition_label(Decimal("10")) == partition_label(1e1) == "=10"


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


@pytest.mark.parametrize("one", ["1.0", "1.00", "1e0", "10e-1"])
def test_a_literal_meets_the_profiled_value_however_it_is_spelled(one):
    assert _check(_JOIN + f" AND d.is_current = {one} GROUP BY d.tier") is None


@pytest.mark.parametrize("huge", ["1e5000", "1e99999999", "-1e-99999999"])
def test_an_extreme_literal_is_spelled_without_building_it(huge):
    """`int()` on `1e5000` raised out of the decider; on `1e99999999` it ran for as long as a
    hundred-million-digit integer takes. The literal picks no partition, so the join refuses."""
    assert _check(_JOIN + f" AND d.is_current = {huge} GROUP BY d.tier") is not None
    assert partition_label(Decimal(huge)).startswith("=")


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


_CURRENT = _JOIN + " AND d.is_current = 1 GROUP BY d.tier"
_HISTORY = _JOIN + " AND d.is_current = 0 GROUP BY d.tier"


def _plan(snapshot, sql, guard):
    from mnemiq.authz.grants import GrantSet
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    packet = ContextPacket(
        question="revenue by customer tier",
        cards=[RetrievedCard(object_id="orders", card="TABLE orders", score=1.0),
               RetrievedCard(object_id="dim_customer", card="TABLE dim_customer", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1")
    return plan_query(packet, snapshot, GrantSet(frozenset({"orders", "dim_customer"})),
                      FakeGenerator([f'{{"sql": "{sql}"}}']), target="sqlite", dialect="sqlite",
                      max_attempts=1, guard_fanout=guard)


def _legacy(snapshot):
    """The same snapshot as one persisted before per-partition profiling: no partition facts and
    no job saying they were profiled, round-tripped through JSON as a stored snapshot is."""
    raw = json.loads(snapshot.model_dump_json())
    for column in raw["columns"]:
        column.pop("unique_within", None)
    raw["jobs"] = [j for j in raw["jobs"] if j["id"] != PARTITIONS_JOB]
    return Snapshot.model_validate(raw)


def test_a_current_row_join_plans_after_structural_enrichment(tmp_path):
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.generate.plan_query import Deferred
    from mnemiq.sql.verdict import Approved

    snapshot = enrich_structural(_db(tmp_path), "shop")
    assert isinstance(_plan(snapshot, _CURRENT, True), Approved)
    assert isinstance(_plan(snapshot, _HISTORY, True), Deferred)


# -- rollout: auto is on exactly where the snapshot can support it ----------------------------------


def test_enrichment_records_that_partitions_were_profiled(tmp_path):
    from mnemiq.enrichment.pipeline import enrich_structural

    snapshot = enrich_structural(_db(tmp_path), "shop")
    job = next(j for j in snapshot.jobs if j.id == PARTITIONS_JOB)
    assert job.status == "done" and job.kind != "profile", "kind 'profile' counts tables"
    assert partitions_profiled(snapshot) and not partitions_profiled(_legacy(snapshot))


def test_auto_on_a_freshly_enriched_snapshot_is_on(tmp_path):
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.generate.plan_query import Deferred
    from mnemiq.sql.verdict import Approved

    snapshot = enrich_structural(_db(tmp_path), "shop")
    assert isinstance(_plan(snapshot, _CURRENT, None), Approved)
    assert isinstance(_plan(snapshot, _HISTORY, None), Deferred), "the guard is on"


def test_auto_on_a_snapshot_enriched_before_partitions_changes_nothing(tmp_path, caplog,
                                                                      monkeypatch):
    """The upgrade: a stored snapshot has no partition facts, so with the guard on a correct
    current-row join is refused until re-enrichment. Auto leaves such a snapshot as it was and
    says so, once; an explicit setting still forces the guard."""
    import logging

    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.generate.plan_query import Deferred
    from mnemiq.sql.verdict import Approved

    from mnemiq.sql import fanout_check

    monkeypatch.setattr(fanout_check, "_warned", set())  # once per process: start this one clean
    legacy = _legacy(enrich_structural(_db(tmp_path), "shop"))
    with caplog.at_level(logging.WARNING, logger="mnemiq.sql.fanout_check"):
        assert isinstance(_plan(legacy, _CURRENT, None), Approved)
        assert isinstance(_plan(legacy, _HISTORY, None), Approved), "off: as before the upgrade"
    warnings = [r for r in caplog.records if "enriched before per-partition" in r.getMessage()]
    assert len(warnings) == 1, "said once per snapshot, not once per question"
    assert isinstance(_plan(legacy, _CURRENT, True), Deferred), "explicit 1 forces it"


def test_flag_values_past_float_precision_keep_their_own_facts(tmp_path):
    """2**53 and 2**53 + 1 are one float. Spelled through one, the partition where the key repeats
    took the facts of the one where it does not, and an inflated sum was approved."""
    from mnemiq.enrichment.pipeline import enrich_structural

    big = 2 ** 53
    path = tmp_path / "big.sqlite"
    con = sqlite3.connect(path)
    con.executescript(f"""
        CREATE TABLE dim (customer_id INTEGER, batch INTEGER, tier TEXT);
        INSERT INTO dim VALUES (1, {big}, 'a'), (1, {big}, 'b'), (2, {big + 1}, 'a'),
                               (3, {big + 1}, 'b');
        CREATE TABLE orders (customer_id INTEGER, amount INTEGER);
        INSERT INTO orders VALUES (1, 5), (1, 7), (2, 9);
    """)
    con.commit()
    con.close()
    snap = enrich_structural(SQLiteAdapter(str(path)), "big")

    def check(batch):
        sql = ("SELECT d.tier, SUM(o.amount) FROM orders o JOIN dim d "
               f"ON d.customer_id = o.customer_id AND d.batch = {batch} GROUP BY d.tier")
        return check_fanout(sqlglot.parse_one(sql, read="sqlite"), schema_map(snap), key_facts(snap))

    assert check(big) is not None, "customer 1 has two rows in this batch"
    assert check(big + 1) is None
    assert check(f"{big + 1}.0") is None, "the literal side spells it exactly too"


def test_a_failed_partition_query_costs_its_table_not_the_snapshot(tmp_path):
    """Recorded, never claimed as profiled: the job reads `partial` and names the column. The
    guard stays on -- only that table's current-row join refuses, visibly -- rather than leaving
    every other table's sums unguarded."""
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.generate.plan_query import Deferred

    adapter = _db(tmp_path)

    class _Failing:
        dialect = adapter.dialect

        def __getattr__(self, name):
            return getattr(adapter, name)

        def execute(self, sql, *a, **kw):
            if '"is_current" IS NOT NULL GROUP BY' in sql:
                raise RuntimeError("temp space")
            return adapter.execute(sql, *a, **kw)

    snapshot = enrich_structural(_Failing(), "shop")
    job = next(j for j in snapshot.jobs if j.id == PARTITIONS_JOB)
    assert job.status == "partial" and "dim_customer.is_current: temp space" in job.detail
    assert partitions_profiled(snapshot)
    assert isinstance(_plan(snapshot, _CURRENT, None), Deferred), "on: that join refuses"


def test_one_profiled_source_does_not_vouch_for_a_federated_legacy_one(tmp_path):
    from mnemiq.config import SourceSpec
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.semantic.federation import merge_snapshots

    fresh = enrich_structural(_db(tmp_path), "shop")

    def spec(catalog):
        return SourceSpec(id=catalog, kind="sqlite", target="x", catalog=catalog, schema="main")

    assert partitions_profiled(merge_snapshots([(spec("a"), fresh), (spec("b"), fresh)]))
    assert not partitions_profiled(merge_snapshots([(spec("a"), fresh), (spec("b"), _legacy(fresh))]))


# -- the marker moves the version: it decides whether the guard runs ----------------------------------


def test_re_enriching_a_source_with_no_partitions_moves_the_version(tmp_path):
    """The join `reload_if_stale` and every version-keyed cache depend on. A source with no
    two-valued column re-enriches to identical columns, so only the job tells the new snapshot
    from the one it replaces -- and under auto the job is what turns the guard on."""
    from mnemiq.enrichment.pipeline import content_version, enrich_structural

    path = tmp_path / "flat.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, amount INTEGER);
        INSERT INTO orders VALUES (10, 1, 5), (11, 1, 7), (12, 2, 9), (13, 3, 4);
    """)
    con.commit()
    con.close()
    fresh = enrich_structural(SQLiteAdapter(str(path)), "flat")
    assert not any(c.unique_within for c in fresh.columns), "no partitions: columns are identical"
    assert content_version(_legacy(fresh)) != fresh.version


def test_a_snapshot_predating_the_job_keeps_its_version():
    """Only when present, as for `discover:views`: a legacy store must not churn on upgrade."""
    from mnemiq.contract import Job
    from mnemiq.enrichment.pipeline import content_version

    snap = _snapshot()
    other = snap.model_copy(update={"jobs": [
        Job(id="profile:t", source_id="shop", kind="profile", status="done")]})
    assert content_version(snap) == content_version(other), "an unrelated job still does not"
    marked = snap.model_copy(update={"jobs": [
        Job(id=PARTITIONS_JOB, source_id="shop", kind="profile:partitions", status="done")]})
    assert content_version(marked) != content_version(snap)

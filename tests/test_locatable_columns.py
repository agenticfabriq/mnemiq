"""`locatable_columns`: the columns a repair hint may name, or a source probe may ask, for one caller.

A hint that says "it does appear in t.c" tells the caller something about t.c's values, and a probe asks the source
about them. So a column is locatable only where the caller could have read those values itself, plainly: a granted
base table with no row filter (a view can read a filtered table, M27), a column the policy neither denies nor masks,
not computed, a string, and not personal data, the value index's own rule. Each test below leaves one column out for
one reason and keeps an eligible column beside it, so a rule that returned nothing would fail too.
"""

from __future__ import annotations

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.contract.semantic import ViewDefinition
from mnemiq.sql.locatable import locatable_columns
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.views import inventory_for


def _col(table: str, name: str, data_type: str = "VARCHAR", pii: str | None = None) -> Column:
    return Column(id=f"{table}.{name}", object_id=table, name=name, data_type=data_type, pii_level=pii)


def _locatable(columns, grants: GrantSet, views=()):
    snapshot = Snapshot(version="v1", source_id="src", created_at="2026-11-17T00:00:00Z", columns=columns,
                        views=list(views))
    return set(locatable_columns(snapshot, visible_schema(snapshot, grants), build_access_policy(snapshot, grants),
                                 inventory_for(snapshot)))


def test_a_denied_column_with_a_neutral_name_on_a_granted_table_is_left_out():
    """A `direct`-level column the caller holds no clearance for is denied whatever its name: neither the name rule nor
    the personal-data levels the index skips would catch it, so the policy's own denial must (Codex's review)."""
    found = _locatable([_col("orders", "batch_ref"), _col("orders", "ref_code", pii="direct")],
                       GrantSet(frozenset({"orders"})))
    assert ("orders", "batch_ref") in found and ("orders", "ref_code") not in found


def test_a_granted_view_over_a_row_filtered_table_is_left_out():
    view = ViewDefinition(object_id="orders_v", definition="SELECT batch_ref FROM orders", dialect="duckdb")
    found = _locatable([_col("orders", "batch_ref"), _col("orders_v", "batch_ref"), _col("shipments", "batch_ref")],
                       GrantSet(frozenset({"orders", "orders_v", "shipments"}), row_filters={"orders": "id > 0"}),
                       views=[view])
    assert ("shipments", "batch_ref") in found
    assert ("orders_v", "batch_ref") not in found and ("orders", "batch_ref") not in found


def test_masked_row_filtered_personal_and_non_string_columns_are_left_out():
    columns = [_col("orders", "batch_ref"),
               _col("orders", "card_hint", pii="direct"),         # masked for this caller, and no pii/phi level
               _col("orders", "diagnosis", pii="phi"),            # personal data, cleared or not
               _col("orders", "customer_name"),                   # a personal name, by the index's name rule
               _col("orders", "qty", data_type="INTEGER"),        # not a string
               _col("returns", "batch_ref")]                      # a row-filtered table
    grants = GrantSet(frozenset({"orders", "returns"}), row_filters={"returns": "id > 0"},
                      pii_mask=frozenset({"direct"}), pii_clearance=frozenset({"phi"}))
    assert _locatable(columns, grants) == {("orders", "batch_ref")}


def test_a_table_outside_the_callers_grants_is_left_out():
    found = _locatable([_col("orders", "batch_ref"), _col("ledger", "batch_ref")], GrantSet(frozenset({"orders"})))
    assert found == {("orders", "batch_ref")}


# --- the decider drops computed columns before check_values sees them --------------------------------------------

class _Index:
    def __init__(self, data):
        self._data = data

    def has(self, table, column):
        return (table, column) in self._data

    def contains(self, table, column, literal):
        return literal in self._data.get((table, column), set())

    def nearest(self, table, column, literal, k=8):
        return sorted(self._data.get((table, column), set()))[:k]


class _Source:
    """A source whose function catalogue cannot be listed, so every virtual column it reports is computed."""
    dialect = "duckdb"

    def __init__(self, virtual):
        self.virtual = virtual

    def execute(self, sql, *args, **kwargs):
        return []

    def virtual_columns(self):
        return self.virtual


def test_decide_drops_computed_columns_before_check_values_sees_them():
    from mnemiq.sql.decide import decide

    index = _Index({("orders", "status"): {"open"}, ("orders", "region"): {"west"}, ("orders", "lane"): {"west"}})
    visible = {"orders": {"id", "status", "region", "lane"}}
    locatable = (("orders", "lane"), ("orders", "region"), ("orders", "status"))
    sql = "SELECT id FROM orders WHERE status = 'west'"
    computed = decide(sql, visible, adapter=_Source([("orders", "region", "my_fn(status)")]), dialect="duckdb",
                      target="duckdb", values=index, repair_hints=True, locatable=locatable)
    assert computed.message.endswith(" It does appear in orders.lane: if that column is the one meant, filter on it "
                                     "instead.")
    plain = decide(sql, visible, adapter=_Source([]), dialect="duckdb", target="duckdb", values=index,
                   repair_hints=True, locatable=locatable)
    assert " It does appear in orders.lane and orders.region: " in plain.message


def test_a_catalogue_that_cannot_say_which_columns_are_computed_locates_nothing():
    from mnemiq.sql.decide import decide

    class _Down(_Source):
        def virtual_columns(self):
            raise RuntimeError("catalogue unavailable")

    index = _Index({("orders", "status"): {"open"}, ("orders", "lane"): {"west"}})
    verdict = decide("SELECT id FROM orders WHERE status = 'west'", {"orders": {"id", "status", "lane"}},
                     adapter=_Down([]), dialect="duckdb", target="duckdb", values=index, repair_hints=True,
                     locatable=(("orders", "lane"), ("orders", "status")))
    assert "It does appear" not in getattr(verdict, "message", "")


def test_plan_query_names_only_what_the_callers_grants_allow():
    """The join: `plan_query` builds the allow-list from the caller's grants and the refusal it feeds back names only
    from it. A masked column and a row-filtered table both hold the value; the one plain column beside them is named."""
    import json

    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    snapshot = Snapshot(version="v1", source_id="src", created_at="2026-11-17T00:00:00Z", columns=[
        _col("orders", "id", "INTEGER"), _col("orders", "status"), _col("orders", "region"),
        _col("orders", "card_hint", pii="direct"), _col("returns", "state")])
    grants = GrantSet(frozenset({"orders", "returns"}), row_filters={"returns": "id > 0"},
                      pii_mask=frozenset({"direct"}))
    index = _Index({("orders", "status"): {"open"}, ("orders", "region"): {"west"},
                    ("orders", "card_hint"): {"west"}, ("returns", "state"): {"west"}})
    packet = ContextPacket(question="which orders?", cards=[RetrievedCard(object_id="orders", card="TABLE orders",
                                                                          score=1.0)],
                           grant_fingerprint="fp", enrichment_version="v1")
    replies = [json.dumps({"sql": "SELECT id FROM orders WHERE status = 'west'"}),
               json.dumps({"sql": "SELECT id FROM orders WHERE region = 'west'"})]
    on = FakeGenerator(list(replies))
    plan_query(packet, snapshot, grants, on, dialect="duckdb", target="duckdb", values=index, repair_hints=True)
    assert on.calls[1].endswith(" It does appear in orders.region: if that column is the one meant, filter on it "
                                "instead."), on.calls[1]
    off = FakeGenerator(list(replies))
    plan_query(packet, snapshot, grants, off, dialect="duckdb", target="duckdb", values=index)
    assert "It does appear" not in off.calls[1]

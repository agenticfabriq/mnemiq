"""M29: the row-filter coverage warning names a granted view that reads across the boundary.

The warning walked child -> parent foreign keys from each granted TABLE. A granted view has no
foreign keys, so a view over `payment` -- a child of the filtered `customer` -- was a door the
warning never named: measured, `SELECT sum(amount) FROM payments_v` was approved and read every
store's payments, while the warning listed nothing. A view whose bases include a filtered table
is not a hole: the deciders refuse every read of it (UNGOVERNED_VIEW, M27's floor). Their other
view refusals are not modelled, so the warning can name a view they refuse -- the safe direction.
"""

from __future__ import annotations

import logging

from mnemiq.authz.coverage import unfiltered_dependents
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, JoinKey, Relationship, Snapshot, ViewDefinition


def _rel(child: str, parent: str) -> Relationship:
    return Relationship(id=f"{child}->{parent}", **{"from": child}, to=parent,
                        cardinality="many_to_one", join_keys=[JoinKey(left="k", right="k")])


def _views(**bodies: str) -> dict[str, ViewDefinition]:
    return {name: ViewDefinition(object_id=name, dialect="duckdb", definition=body)
            for name, body in bodies.items()}


RELS = [_rel("payment", "customer")]
FILTER = {"customer": "store_id = 1"}


def test_a_granted_view_over_an_unfiltered_dependent_is_named():
    grants = GrantSet(frozenset({"customer", "payments_v"}), row_filters=FILTER)
    found = unfiltered_dependents(grants, RELS,
                                  views=_views(payments_v="SELECT amount FROM payment"))
    assert found == [("payments_v", "customer")]


def test_a_view_reading_a_filtered_table_is_not_a_hole():
    """Refused on every read, so naming it as readable across the boundary would be false."""
    grants = GrantSet(frozenset({"customer", "v"}), row_filters=FILTER)
    body = "SELECT p.amount FROM payment p JOIN customer c USING (k)"
    assert unfiltered_dependents(grants, RELS, views=_views(v=body)) == []


def test_a_nested_view_is_followed_to_its_base():
    grants = GrantSet(frozenset({"customer", "outer_v"}), row_filters=FILTER)
    views = _views(outer_v="SELECT amount FROM inner_v", inner_v="SELECT amount FROM payment")
    assert unfiltered_dependents(grants, RELS, views=views) == [("outer_v", "customer")]


def test_a_federated_view_body_is_read_in_its_source_naming():
    """Federation keys `pg.payments_v` and `pg.payment` while the body says `payment`."""
    grants = GrantSet(frozenset({"pg.customer", "pg.payments_v"}),
                      row_filters={"pg.customer": "store_id = 1"})
    views = {"pg.payments_v": ViewDefinition(object_id="pg.payments_v", dialect="duckdb",
                                             definition="SELECT amount FROM payment")}
    found = unfiltered_dependents(grants, [_rel("pg.payment", "pg.customer")], views=views)
    assert found == [("pg.payments_v", "pg.customer")]


def test_a_view_over_shared_reference_data_is_not_named():
    """The direction rule holds through a view: an address is not a store's property."""
    grants = GrantSet(frozenset({"customer", "addresses_v"}), row_filters=FILTER)
    found = unfiltered_dependents(grants, [_rel("customer", "address")],
                                  views=_views(addresses_v="SELECT city FROM address"))
    assert found == []


def test_the_boot_advisory_hands_the_snapshot_views_to_the_warning(caplog):
    """The join: `_warn_policy_advisories` is what boot and a snapshot swap call."""
    from mnemiq.runtime import _warn_policy_advisories

    class _Authz:
        def policy_roles(self):
            return ["manager"]

        def grants_for(self, _identity):
            return GrantSet(frozenset({"customer", "payments_v"}), row_filters=FILTER)

    snapshot = Snapshot(version="v1", source_id="s", created_at="t",
                        columns=[Column(id="customer.store_id", object_id="customer",
                                        name="store_id"),
                                 Column(id="payments_v.amount", object_id="payments_v",
                                        name="amount")],
                        relationships=RELS,
                        views=list(_views(payments_v="SELECT amount FROM payment").values()))
    with caplog.at_level(logging.WARNING, logger="mnemiq.authz.coverage"):
        _warn_policy_advisories(_Authz(), snapshot)
    assert any("payments_v (reaches filtered customer)" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_a_same_named_view_in_another_catalog_does_not_hide_the_hole():
    """Review gate: resolved by the bare segment, `inner_v` in `pg.outer_v`'s body could follow
    `my.inner_v`, which reads the filtered `my.customer`, and the hole went unnamed."""
    grants = GrantSet(frozenset({"pg.customer", "pg.outer_v"}),
                      row_filters={"pg.customer": "store_id = 1", "my.customer": "region = 'w'"})
    views = {key: ViewDefinition(object_id=key, dialect="duckdb", definition=body)
             for key, body in {"pg.outer_v": "SELECT amount FROM inner_v",
                               "pg.inner_v": "SELECT amount FROM payment",
                               "my.inner_v": "SELECT amount FROM customer"}.items()}
    rels = [_rel("pg.payment", "pg.customer"), _rel("my.payment", "my.customer")]
    for order in (views, dict(reversed(list(views.items())))):
        assert unfiltered_dependents(grants, rels, views=order) == [("pg.outer_v", "pg.customer")]


def test_a_schema_qualified_body_reaches_a_table_the_graph_keys_bare():
    """Neither the name as written nor the view's catalog finds `public.payment`; only the
    bare-segment fallback does -- in either case, as the deciders read an unquoted name."""
    grants = GrantSet(frozenset({"customer", "v"}), row_filters=FILTER)
    for body in ("SELECT amount FROM public.payment", "SELECT amount FROM PUBLIC.PAYMENT"):
        found = unfiltered_dependents(grants, RELS, views=_views(v=body))
        assert found == [("v", "customer")], body


def test_a_bare_body_name_reaches_a_table_the_graph_keys_by_schema():
    """The other direction: a body on a search path says `payment` where the graph keys
    `sales.payment`, and the view's own key carries no catalog to qualify it with."""
    grants = GrantSet(frozenset({"sales.customer", "v"}),
                      row_filters={"sales.customer": "store_id = 1"})
    found = unfiltered_dependents(grants, [_rel("sales.payment", "sales.customer")],
                                  views=_views(v="SELECT amount FROM payment"))
    assert found == [("v", "sales.customer")]

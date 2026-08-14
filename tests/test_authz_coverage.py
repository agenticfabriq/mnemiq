"""A row filter secures one table. The tables hanging off it are the author's problem.

Found against Pagila: a policy filtering customer/staff/inventory/store still let a store
manager run `SELECT SUM(amount) FROM payment` over the whole business -- 67,416.51 against
their own 33,689.74, so the other store's revenue by subtraction, without naming it or
touching a filtered table. The engine was correct; the policy was incomplete and nothing said so.
"""

import logging

from mnemiq.authz.coverage import (
    unfiltered_dependents,
    unknown_pii_levels,
    warn_unknown_pii_levels,
)
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import JoinKey, Relationship


def rel(child, parent, cardinality="many_to_one"):
    return Relationship(id=f"{child}->{parent}", **{"from": child}, to=parent,
                        cardinality=cardinality, join_keys=[JoinKey(left="k", right="k")])


def grants(objects, filters):
    return GrantSet(frozenset(objects), row_filters=filters)


def test_a_child_of_a_filtered_table_is_reported():
    g = grants({"customer", "payment"}, {"customer": "store_id = 1"})

    assert unfiltered_dependents(g, [rel("payment", "customer")]) == [("payment", "customer")]


def test_a_grandchild_leaks_just_as_well():
    g = grants({"customer", "rental", "payment"}, {"customer": "store_id = 1"})

    found = unfiltered_dependents(g, [rel("rental", "customer"), rel("payment", "rental")])

    assert ("payment", "customer") in found, "transitive: a grandchild is as readable as a child"


def test_shared_reference_data_is_not_reported():
    """The direction is the whole of the precision.

    `customer.address_id -> address` does not make addresses the property of a store.
    Warning here would push an author to filter shared lookups and hide rows people need.
    """
    g = grants({"customer", "address"}, {"customer": "store_id = 1"})

    assert unfiltered_dependents(g, [rel("customer", "address")]) == []


def test_a_filtered_child_is_not_reported():
    g = grants({"customer", "payment"}, {"customer": "store_id = 1", "payment": "x = 1"})

    assert unfiltered_dependents(g, [rel("payment", "customer")]) == []


def test_no_filters_means_nothing_to_be_inconsistent_with():
    # An unrestricted role is not a policy hole -- it is a policy.
    g = grants({"customer", "payment"}, {})

    assert unfiltered_dependents(g, [rel("payment", "customer")]) == []


def test_a_table_that_is_not_granted_is_not_reported():
    g = grants({"customer"}, {"customer": "store_id = 1"})

    assert unfiltered_dependents(g, [rel("payment", "customer")]) == []


def test_many_to_many_does_not_imply_ownership():
    g = grants({"customer", "payment"}, {"customer": "store_id = 1"})

    assert unfiltered_dependents(g, [rel("payment", "customer", "many_to_many")]) == []


# --- PII-clearance vocabulary (M40) -----------------------------------------
# A column's pii_level is only ever none/pii/phi. A `pii_clearance` or `pii_mask` value
# outside that set matches no column and silently grants nothing -- so an author who reaches
# for a low/medium/high scale gets a clearance that clears nothing, with no feedback. The
# engine is right to deny (fail-closed), but the misconfiguration must be visible.


def cls_grants(clearance=(), mask=()):
    return GrantSet(
        frozenset({"customer"}),
        pii_clearance=frozenset(clearance),
        pii_mask=frozenset(mask),
    )


def test_a_clearance_value_outside_the_pii_vocabulary_is_reported():
    g = cls_grants(clearance={"low", "medium", "high"})

    assert unknown_pii_levels(g) == ["high", "low", "medium"]


def test_the_real_pii_levels_are_not_reported():
    g = cls_grants(clearance={"pii", "phi"}, mask={"pii"})

    assert unknown_pii_levels(g) == []


def test_none_in_a_clearance_is_a_no_op_not_an_error():
    # 'none' is a valid level -- clearing it is pointless but not a misconfiguration.
    g = cls_grants(clearance={"none", "pii"})

    assert unknown_pii_levels(g) == []


def test_an_unknown_mask_value_is_reported_too():
    g = cls_grants(mask={"secret"})

    assert unknown_pii_levels(g) == ["secret"]


def test_warn_names_the_role_and_the_unknown_values(caplog):
    g = cls_grants(clearance={"high"})

    with caplog.at_level(logging.WARNING):
        warn_unknown_pii_levels(g, role="hq_analyst")

    assert "hq_analyst" in caplog.text and "high" in caplog.text


def test_warn_is_silent_for_a_valid_policy(caplog):
    g = cls_grants(clearance={"pii", "phi"}, mask={"pii"})

    with caplog.at_level(logging.WARNING):
        warn_unknown_pii_levels(g, role="hq_analyst")

    assert caplog.records == []

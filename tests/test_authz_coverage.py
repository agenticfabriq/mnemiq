"""A row filter secures one table. The tables hanging off it are the author's problem.

Found against Pagila: a policy filtering customer/staff/inventory/store still let a store
manager run `SELECT SUM(amount) FROM payment` over the whole business -- 67,416.51 against
their own 33,689.74, so the other store's revenue by subtraction, without naming it or
touching a filtered table. The engine was correct; the policy was incomplete and nothing said so.
"""

from mnemiq.authz.coverage import unfiltered_dependents
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

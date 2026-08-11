"""M31: a CTE named after a source object must not disarm the access controls.

`WITH t AS (SELECT ... FROM t) SELECT ... FROM t` is legal SQL in which the inner
reference reads the **base table** and the outer one reads the CTE. Three guards
used to skip both, because each asked only whether the name appeared in a flat set
of CTE names -- so table authorization, column deny/mask and row filters all fell
to the same blind spot at once.

Each test carries the equivalent un-shadowed query as a control: the point is not
that the shadowed form is refused, but that it is treated exactly like the query
it is a disguise for.
"""

from __future__ import annotations

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, RefusalCode

DIALECT = "postgres"

VISIBLE = {
    "customer": {"customer_id", "store_id", "first_name", "email"},
    "rental": {"rental_id", "customer_id"},
}
FILTER = AccessPolicy(row_filters={"customer": "store_id = 1"})
MASK = AccessPolicy(masked={("customer", "email")})
DENY = AccessPolicy(denied={("customer", "email")})


def plan(sql: str, visible=VISIBLE, policy: AccessPolicy | None = None):
    return decide(
        sql, visible, dialect=DIALECT, target=DIALECT, policy=policy or AccessPolicy()
    )


# --- table authorization -------------------------------------------------------


def test_a_cte_cannot_launder_a_table_the_identity_has_no_grant_on():
    """The strongest form: `staff` is not in `visible` at all."""
    only_rental = {"rental": {"rental_id", "customer_id"}}

    control = plan("SELECT staff_id FROM staff", visible=only_rental)
    assert control.code is RefusalCode.UNAUTHORIZED_TABLE

    shadowed = plan(
        "WITH staff AS (SELECT staff_id FROM staff) SELECT staff_id FROM staff",
        visible=only_rental,
    )
    assert shadowed.code is RefusalCode.UNAUTHORIZED_TABLE, (
        "a CTE named after an ungranted table read it in full"
    )


# --- column-level policy -------------------------------------------------------


def test_a_cte_cannot_launder_a_denied_column():
    control = plan("SELECT email FROM customer", policy=DENY)
    assert control.code is RefusalCode.UNAUTHORIZED_COLUMN

    shadowed = plan(
        "WITH customer AS (SELECT customer_id, email FROM customer) SELECT email FROM customer",
        policy=DENY,
    )
    assert shadowed.code is RefusalCode.UNAUTHORIZED_COLUMN


def test_a_cte_cannot_launder_a_masked_column():
    control = plan("SELECT email FROM customer", policy=MASK)
    assert isinstance(control, Approved) and "NULL AS email" in control.plan_sql

    shadowed = plan(
        "WITH customer AS (SELECT customer_id, email FROM customer) SELECT email FROM customer",
        policy=MASK,
    )
    assert isinstance(shadowed, Approved)
    assert "NULL AS email" in shadowed.plan_sql, "the mask was dropped inside the CTE body"


# --- row filters ---------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # the plain shadow
        "WITH customer AS (SELECT customer_id, store_id FROM customer) "
        "SELECT customer_id FROM customer",
        # RECURSIVE: the parser puts the self-reference in the same place
        "WITH RECURSIVE customer AS (SELECT customer_id, store_id FROM customer) "
        "SELECT customer_id FROM customer",
        # the shadow is the SECOND cte, reading the real table through the first
        "WITH c1 AS (SELECT customer_id, store_id FROM customer), "
        "customer AS (SELECT customer_id, store_id FROM c1) SELECT customer_id FROM customer",
        # the outer query filters for a store the identity may not see
        "WITH customer AS (SELECT customer_id, store_id FROM customer) "
        "SELECT customer_id FROM customer WHERE store_id = 2",
    ],
    ids=["plain", "recursive", "nested", "reads-another-store"],
)
def test_a_cte_cannot_launder_a_row_filter(sql):
    control = plan("SELECT customer_id FROM customer", policy=FILTER)
    assert isinstance(control, Approved) and "store_id = 1" in control.plan_sql

    shadowed = plan(sql, policy=FILTER)
    assert isinstance(shadowed, Approved), f"unexpected refusal: {shadowed}"
    assert "store_id = 1" in shadowed.plan_sql, "the row filter was dropped inside the CTE body"


# --- what must keep working ----------------------------------------------------


def test_an_ordinary_cte_is_still_not_treated_as_a_source_object():
    """The reason the flat skip existed: a CTE name is not an object to authorize."""
    v = plan(
        "WITH recent AS (SELECT customer_id FROM rental) SELECT customer_id FROM recent",
        policy=FILTER,
    )
    assert isinstance(v, Approved), f"an ordinary CTE was refused: {v}"


def test_a_cte_selecting_from_a_filtered_table_is_filtered_at_the_base():
    v = plan(
        "WITH c AS (SELECT customer_id, store_id FROM customer) SELECT customer_id FROM c",
        policy=FILTER,
    )
    assert isinstance(v, Approved) and "store_id = 1" in v.plan_sql


def test_a_derived_table_sharing_the_name_was_already_correct():
    """Not a regression guard for the fix -- a guard that the fix did not break it."""
    v = plan(
        "SELECT customer_id FROM (SELECT customer_id, store_id FROM customer) AS customer",
        policy=FILTER,
    )
    assert isinstance(v, Approved) and "store_id = 1" in v.plan_sql

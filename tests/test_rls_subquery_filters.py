"""M28: a row filter may reference another table.

Child-table tenancy is not expressible otherwise. `payment` has no `store_id`; the only way
to say "payments belonging to this store's customers" is to reach through `customer`. Until
this, `_validate_filter` required every column in the filter to belong to the filtered table,
so the shipped pagila policy made `payment` and `rental` **refuse** rather than filter -- the
fix for one finding (M29's coverage gap) made inexpressible by another.

The rule the subquery runs under is the whole finding. It is evaluated with the **policy
author's** reach, never the caller's: the policy is what defines the caller's boundary, so
resolving it through that boundary is circular. Snowflake settled on the same rule for row
access policies with mapping tables.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.verdict import Refusal

DIALECT = "postgres"

VISIBLE = {
    "payment": {"payment_id", "customer_id", "amount"},
    "customer": {"customer_id", "store_id", "email"},
}
# What the POLICY AUTHOR can see -- a superset of any one caller's scope, and the schema a
# filter's subquery is resolved against.
POLICY_SCHEMA = {
    **VISIBLE,
    "entitlement": {"principal", "customer_id"},  # no caller is granted this
}

CHILD = "customer_id IN (SELECT customer_id FROM customer WHERE store_id = 1)"


def rewrite(sql: str, policy: AccessPolicy, visible=None):
    ast = sqlglot.parse_one(sql, read=DIALECT)
    return apply_row_and_mask(ast, policy, visible or VISIBLE, dialect=DIALECT)[0]


def policy(**kw) -> AccessPolicy:
    kw.setdefault("policy_schema", POLICY_SCHEMA)
    return AccessPolicy(**kw)


def test_a_filter_may_reach_through_another_table():
    out = rewrite("SELECT payment_id FROM payment", policy(row_filters={"payment": CHILD}))
    assert not isinstance(out, Refusal), out
    assert "IN (SELECT customer_id FROM customer WHERE store_id = 1)" in out.sql(dialect=DIALECT)


def test_the_subquery_is_not_re_filtered_by_the_callers_own_policy():
    """The load-bearing property, and it is easy to break by accident: the rewriter must not
    re-scan the tree it just rewrote. If it did, the filter's own `customer` would be wrapped
    in the caller's `customer` filter -- an intersection the policy author never wrote."""
    out = rewrite(
        "SELECT payment_id FROM payment",
        policy(row_filters={"payment": CHILD, "customer": "store_id = 1"}),
    )
    rendered = out.sql(dialect=DIALECT)
    assert rendered.count("FROM customer") == 1, (
        "the filter's own read of `customer` was rewritten as if the caller had asked for it"
    )
    # ...and it is the bare table, not a derived one.
    assert "IN (SELECT customer_id FROM customer WHERE store_id = 1)" in rendered


def test_the_callers_own_reference_to_that_table_is_still_filtered():
    """Two visibilities in one statement: the policy's read is raw, the caller's is scoped."""
    out = rewrite(
        "SELECT p.payment_id FROM payment p JOIN customer c ON p.customer_id = c.customer_id",
        policy(row_filters={"payment": CHILD, "customer": "store_id = 1"}),
    )
    rendered = out.sql(dialect=DIALECT)
    assert "FROM customer WHERE store_id = 1) AS c" in rendered.replace("\n", " ")


def test_a_filter_may_name_a_table_the_caller_cannot_see():
    """An entitlements table is the standard shape, and the caller never has a grant on it.
    Resolving the filter through the caller's scope would make the policy unwritable."""
    out = rewrite(
        "SELECT payment_id FROM payment",
        policy(row_filters={
            "payment": "customer_id IN (SELECT customer_id FROM entitlement WHERE principal = 'a')"
        }),
    )
    assert not isinstance(out, Refusal), out
    assert "FROM entitlement" in out.sql(dialect=DIALECT)


# --- still fails closed ---------------------------------------------------------


def test_an_outer_column_that_is_not_the_tables_own_is_still_refused():
    """The original rule, kept where it applies: outside a subquery, the predicate can only
    speak about the table it filters."""
    out = rewrite("SELECT payment_id FROM payment",
                  policy(row_filters={"payment": "store_id = 1"}))
    assert isinstance(out, Refusal), out


def test_a_subquery_naming_a_table_no_one_has_is_refused():
    """A typo in a policy must not become a filter that silently does nothing."""
    out = rewrite(
        "SELECT payment_id FROM payment",
        policy(row_filters={
            "payment": "customer_id IN (SELECT customer_id FROM custmoer WHERE store_id = 1)"
        }),
    )
    assert isinstance(out, Refusal), out


def test_a_subquery_naming_a_column_that_table_does_not_have_is_refused():
    out = rewrite(
        "SELECT payment_id FROM payment",
        policy(row_filters={
            "payment": "customer_id IN (SELECT customer_id FROM customer WHERE stroe_id = 1)"
        }),
    )
    assert isinstance(out, Refusal), out


def test_unparseable_is_still_refused():
    out = rewrite("SELECT payment_id FROM payment",
                  policy(row_filters={"payment": "customer_id IN (SELECT"}))
    assert isinstance(out, Refusal), out


def test_a_filter_that_is_a_statement_rather_than_a_predicate_is_refused():
    out = rewrite("SELECT payment_id FROM payment",
                  policy(row_filters={"payment": "DELETE FROM customer"}))
    assert isinstance(out, Refusal), out


def test_without_a_policy_schema_a_subquery_filter_fails_closed():
    """An AccessPolicy built by hand (tests, older callers) carries no policy visibility. A
    subquery cannot be validated against nothing, and guessing would be the whole finding
    again in the other direction."""
    out = rewrite("SELECT payment_id FROM payment",
                  AccessPolicy(row_filters={"payment": CHILD}))
    assert isinstance(out, Refusal), out


def test_a_simple_filter_still_works_without_a_policy_schema():
    """The common case must not acquire a new prerequisite."""
    out = rewrite("SELECT customer_id FROM customer",
                  AccessPolicy(row_filters={"customer": "store_id = 1"}))
    assert not isinstance(out, Refusal), out
    assert "store_id = 1" in out.sql(dialect=DIALECT)


def test_the_policy_schema_is_built_from_the_whole_snapshot_not_the_grants():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract.semantic import Column, Snapshot
    from mnemiq.sql.policy import build_access_policy

    snap = Snapshot(
        version="v1", source_id="s", created_at="2026-01-01T00:00:00Z",
        columns=[
            Column(id="payment.customer_id", object_id="payment", name="customer_id"),
            Column(id="entitlement.principal", object_id="entitlement", name="principal"),
        ],
    )
    built = build_access_policy(snap, GrantSet(objects=frozenset({"payment"})))
    assert "entitlement" in built.policy_schema, (
        "a filter's subquery must be resolvable against tables the caller cannot see"
    )
    assert built.policy_schema["entitlement"] == {"principal"}


def test_an_injected_filter_never_widens_what_the_query_reads_of_its_own_table():
    """Sanity: the derived table still projects only the caller's visible columns."""
    out = rewrite("SELECT payment_id FROM payment", policy(row_filters={"payment": CHILD}))
    derived = next(iter(out.find_all(exp.Subquery)))
    projected = {e.alias_or_name for e in derived.this.expressions}
    assert projected == VISIBLE["payment"]


# --- what the caller is told they read ------------------------------------------


def test_the_policys_own_reads_are_not_reported_as_the_callers_lineage():
    """M28: a filter that reaches through another table makes the engine read it. Reporting
    that would tell the caller which tables their policy consults -- an entitlements table
    being the standard shape, and one no caller is granted."""
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import Approved

    verdict = decide(
        "SELECT payment_id FROM payment",
        VISIBLE,
        dialect=DIALECT,
        target=DIALECT,
        policy=policy(row_filters={
            "payment": "customer_id IN (SELECT customer_id FROM entitlement WHERE principal = 'a')"
        }),
    )
    assert isinstance(verdict, Approved)
    assert verdict.tables == ["payment"], "the policy's own read leaked into the caller's lineage"
    # ...while the SQL that runs still reads it.
    assert "FROM entitlement" in verdict.target_sql


def test_columns_report_what_the_query_asked_for_not_what_the_rewrite_projects():
    """The derived table projects every visible column of a filtered table, so reading
    provenance after the rewrite named columns the caller never mentioned."""
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import Approved

    verdict = decide(
        "SELECT payment_id FROM payment",
        VISIBLE,
        dialect=DIALECT,
        target=DIALECT,
        policy=policy(row_filters={"payment": CHILD}),
    )
    assert isinstance(verdict, Approved)
    assert verdict.columns == ["payment_id"]

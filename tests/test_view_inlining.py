"""M27 slice 2: a granted view no longer sees around the filter on its base tables.

The decider refused the base table correctly; the repair loop took that refusal as a signpost
and reached the same rows through a view nobody had marked as one. Wrapping the view's OUTPUT
cannot fix it -- an aggregating view has grouped the tenancy column away, so there is nothing
left to filter. The body is inlined and the filter lands at the leaves.
"""

from __future__ import annotations

from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

DIALECT = "postgres"

SCHEMA = {
    "customer": {"customer_id", "store_id", "email"},
    "payment": {"payment_id", "customer_id", "amount"},
    "customer_list": {"id", "sid"},
    "sales_by_store": {"store", "total"},
    "broken_view": {"x"},
}
GRANTED = {k: SCHEMA[k] for k in ("customer_list", "sales_by_store", "broken_view")}

VIEWS = {
    # renames store_id -> sid, the case that looked like it needed column mapping
    "customer_list": ViewDefinition(
        object_id="customer_list", dialect=DIALECT,
        definition="SELECT cu.customer_id AS id, cu.store_id AS sid FROM customer cu",
    ),
    # aggregates the tenancy column away entirely -- output filtering is impossible here
    "sales_by_store": ViewDefinition(
        object_id="sales_by_store", dialect=DIALECT,
        definition="SELECT c.email AS store, SUM(p.amount) AS total FROM payment p "
                   "JOIN customer c ON p.customer_id = c.customer_id GROUP BY c.email",
    ),
    "broken_view": ViewDefinition(
        object_id="broken_view", dialect=DIALECT, definition="SELECT FROM WHERE ((",
    ),
}

FILTERED = AccessPolicy(row_filters={"customer": "store_id = 1"}, policy_schema=SCHEMA)


def plan(sql: str, policy: AccessPolicy = FILTERED, visible=None, views=VIEWS):
    return decide(sql, visible or GRANTED, dialect=DIALECT, target=DIALECT,
                  policy=policy, views=views)


def test_a_granted_view_no_longer_sees_around_the_base_tables_filter():
    verdict = plan("SELECT id, sid FROM customer_list")
    assert isinstance(verdict, Approved), verdict
    assert "store_id = 1" in verdict.plan_sql, "the view returned unfiltered rows"


def test_the_filter_lands_inside_the_view_not_on_its_output():
    """The renamed column needs no mapping: the filter goes on `customer.store_id` where it
    still has its own name, and `sid` is only the outer projection."""
    verdict = plan("SELECT id, sid FROM customer_list")
    assert "FROM customer WHERE store_id = 1" in verdict.plan_sql.replace("\n", " ")
    assert "sid = 1" not in verdict.plan_sql


def test_an_aggregating_view_is_filtered_before_the_grouping():
    """`sales_by_store` groups the tenancy column away, so there is no output column to
    filter on -- the case that makes output-wrapping not a weaker fix but an impossible one."""
    verdict = plan("SELECT store, total FROM sales_by_store")
    assert isinstance(verdict, Approved), verdict
    rendered = verdict.plan_sql.replace("\n", " ")
    assert "store_id = 1" in rendered
    assert rendered.index("store_id = 1") < rendered.index("GROUP BY")


def test_a_view_over_an_unfiltered_table_is_left_alone_in_substance():
    unfiltered = AccessPolicy(row_filters={"payment": "amount > 0"}, policy_schema=SCHEMA)
    verdict = plan("SELECT store, total FROM sales_by_store", policy=unfiltered)
    assert isinstance(verdict, Approved) and "amount > 0" in verdict.plan_sql


# --- fails closed ---------------------------------------------------------------


def test_a_view_whose_body_will_not_parse_is_refused():
    """The floor this replaced, kept as this one's error branch rather than its alternative."""
    verdict = plan("SELECT x FROM broken_view")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_a_view_defined_in_terms_of_itself_is_refused():
    cyclic = {
        "a": ViewDefinition(object_id="a", dialect=DIALECT, definition="SELECT x FROM b"),
        "b": ViewDefinition(object_id="b", dialect=DIALECT, definition="SELECT x FROM a"),
    }
    verdict = decide("SELECT x FROM a", {"a": {"x"}}, dialect=DIALECT, target=DIALECT,
                     policy=AccessPolicy(row_filters={"z": "1 = 1"}, policy_schema={"z": {"q"}}),
                     views=cyclic)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_a_view_over_a_view_resolves_to_the_base():
    nested = {
        "store_roster": ViewDefinition(object_id="store_roster", dialect=DIALECT,
                                       definition="SELECT id, sid FROM customer_list"),
        **VIEWS,
    }
    verdict = decide("SELECT id FROM store_roster", {"store_roster": {"id", "sid"}}, dialect=DIALECT,
                     target=DIALECT, policy=FILTERED, views=nested)
    assert isinstance(verdict, Approved), verdict
    assert "store_id = 1" in verdict.plan_sql


# --- what the caller is told, and what stays untouched ---------------------------


def test_the_caller_is_told_they_read_the_view_not_its_bases():
    """M28's precedent: what the engine reads on the policy's behalf is not the caller's
    lineage, and a view's internals are the same kind of fact."""
    verdict = plan("SELECT id, sid FROM customer_list")
    assert verdict.tables == ["customer_list"]


def test_an_ungoverned_deployment_is_not_rewritten_at_all():
    verdict = plan("SELECT id, sid FROM customer_list", policy=AccessPolicy())
    assert isinstance(verdict, Approved)
    assert "FROM customer_list" in verdict.plan_sql, "an empty policy paid the rewrite's risk"


def test_a_row_filters_own_subquery_is_still_not_filtered_while_a_views_bases_are():
    """The two cases that look identical -- a table absent from `visible` -- and must be
    treated oppositely. M28's subquery reads with the policy's reach; M27's bases read with
    the caller's."""
    both = AccessPolicy(
        row_filters={
            "customer": "customer_id IN (SELECT customer_id FROM payment WHERE amount > 0)",
        },
        policy_schema=SCHEMA,
    )
    verdict = plan("SELECT id, sid FROM customer_list", policy=both)
    assert isinstance(verdict, Approved), verdict
    rendered = verdict.plan_sql.replace("\n", " ")
    # the view's base got the caller's filter...
    assert "FROM customer WHERE customer_id IN" in rendered
    # ...and the filter's own read of `payment` was left verbatim, not wrapped.
    assert "IN (SELECT customer_id FROM payment WHERE amount > 0)" in rendered

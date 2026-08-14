"""M27, floored: a granted view that reads a row-filtered table is declined, not answered past.

The fix -- inlining the body so the filter lands at the leaves -- was built, merged, and taken
back out. Applying a policy *through* a view needs column-level lineage, and a lineage model
that does not understand `SELECT *`, `UNION`, aggregates or transitive renames fails OPEN on
each shape it misses. Four review rounds found bypasses in ordinary view definitions.

So these tests pin a floor, and the property that makes it a floor: every way it can be wrong
points at refusing. The shapes that defeated the lineage model are here as tests -- not because
the engine now answers them correctly, but because it must refuse every one.
"""

from __future__ import annotations

import pytest

from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

D = "postgres"
SCHEMA = {"customer": {"customer_id", "store_id", "email"},
          "film": {"film_id", "title"},
          "v": {"a", "b"}}
GRANTED = {"v": {"a", "b"}}


def view(body: str) -> dict[str, ViewDefinition]:
    return {"v": ViewDefinition(object_id="v", dialect=D, definition=body)}


def plan(sql: str, views, policy):
    return decide(sql, GRANTED, dialect=D, target=D, policy=policy, views=views)


FILTERED = AccessPolicy(row_filters={"customer": "store_id = 1"}, policy_schema=SCHEMA)
UNFILTERED = AccessPolicy(row_filters={"film": "film_id > 0"}, policy_schema=SCHEMA)


@pytest.mark.parametrize(
    "body",
    [
        "SELECT customer_id AS a, store_id AS b FROM customer",
        # every shape that defeated the lineage model, each of which must refuse
        "SELECT * FROM customer",
        "SELECT customer_id AS a FROM customer UNION ALL SELECT film_id AS a FROM film",
        "SELECT COUNT(email) FROM customer",
        "SELECT c.customer_id AS a FROM customer c JOIN film f ON c.customer_id = f.film_id",
        "SELECT a FROM (SELECT customer_id AS a FROM customer) inner_q",
    ],
    ids=["plain", "star", "union", "unaliased-aggregate", "join", "derived-table"],
)
def test_every_view_shape_that_reads_a_filtered_table_is_refused(body):
    verdict = plan("SELECT a FROM v", view(body), FILTERED)
    assert isinstance(verdict, Refusal), f"answered past the filter: {getattr(verdict,'plan_sql','')}"
    assert verdict.code is RefusalCode.UNGOVERNED_VIEW


def test_a_view_that_reads_nothing_filtered_still_answers():
    """The cost is bounded: only views touching a filtered table are declined."""
    verdict = plan("SELECT a FROM v", view("SELECT film_id AS a FROM film"), FILTERED)
    assert isinstance(verdict, Approved), verdict


def test_an_ungoverned_deployment_is_unaffected():
    verdict = plan("SELECT a FROM v", view("SELECT * FROM customer"), AccessPolicy())
    assert isinstance(verdict, Approved), verdict


def test_a_view_reached_through_another_view_is_refused():
    views = {
        "v": ViewDefinition(object_id="v", dialect=D, definition="SELECT a FROM inner_v"),
        "inner_v": ViewDefinition(object_id="inner_v", dialect=D,
                                  definition="SELECT customer_id AS a FROM customer"),
    }
    verdict = plan("SELECT a FROM v", views, FILTERED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW


def test_a_body_this_engine_cannot_read_is_refused():
    verdict = plan("SELECT a FROM v", view("SELECT FROM WHERE (("), FILTERED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_a_view_defined_in_terms_of_itself_is_refused():
    views = {"v": ViewDefinition(object_id="v", dialect=D, definition="SELECT a FROM w"),
             "w": ViewDefinition(object_id="w", dialect=D, definition="SELECT a FROM v")}
    verdict = plan("SELECT a FROM v", views, FILTERED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_a_qualified_base_is_recognised_as_the_filtered_table():
    """`object_key`, the same spelling authorization and the policy builder use."""
    policy = AccessPolicy(row_filters={"pg.customer": "store_id = 1"},
                          policy_schema={"pg.customer": {"customer_id", "store_id"}})
    verdict = plan("SELECT a FROM v", view("SELECT customer_id AS a FROM pg.customer"), policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW


def test_the_refusal_names_the_table_and_says_what_to_do_instead():
    verdict = plan("SELECT a FROM v", view("SELECT * FROM customer"), FILTERED)
    assert "customer" in verdict.message and "directly" in verdict.message


def test_the_caller_is_told_they_read_the_view():
    verdict = plan("SELECT a FROM v", view("SELECT film_id AS a FROM film"), FILTERED)
    assert verdict.tables == ["v"]


# --- the structural claim, attacked ----------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "SELECT * FROM query_table('customer')",
        "SELECT * FROM read_csv('customer.csv')",
        "SELECT * FROM generate_series(1, 10)",
    ],
    ids=["query_table", "read_csv", "generate_series"],
)
def test_a_body_reading_through_a_function_is_refused(body):
    """The one shape that defeats "which tables does this body mention": a table-valued
    function parses to a table node with an EMPTY name, so it matches nothing and the body
    appears to read nothing at all. `query_table('customer')` reads a filtered table by a name
    only the database resolves. A source this engine cannot bind to an object is refused while
    a policy is active -- the unbound-source rule, which the floor needs to be a floor."""
    verdict = plan("SELECT a FROM v", view(body), FILTERED)
    assert isinstance(verdict, Refusal), f"a function source escaped the floor: {verdict}"
    assert verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_a_schema_qualified_base_is_still_recognised_as_the_filtered_table():
    """Snapshot object-ids are bare names for a single source, but a body may spell the same
    table `public.customer`. Deriving one key and comparing it to the other dropped the filter
    from the policy entirely, so the floor had nothing left to match on."""
    policy = AccessPolicy(row_filters={"customer": "store_id = 1"}, policy_schema=SCHEMA)
    verdict = plan("SELECT a FROM v", view("SELECT customer_id AS a FROM public.customer"), policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW

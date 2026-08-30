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


# --- the inversion: an unrecognised shape refuses without being listed -----------


@pytest.mark.parametrize(
    "dialect,body",
    [
        ("postgres", "SELECT * FROM film f, LATERAL (VALUES ((SELECT max(store_id) FROM customer))) AS v(x)"),
        ("duckdb", "SELECT * FROM film f, LATERAL (SELECT * FROM query_table('customer')) t"),
        ("postgres", "SELECT * FROM ROWS FROM (generate_series(1,2))"),
        ("postgres", "SELECT * FROM XMLTABLE('/a' PASSING x COLUMNS c text)"),
        ("duckdb", "SELECT * FROM read_parquet('customer.parquet')"),
        ("sqlite", "SELECT * FROM json_each('[]')"),
    ],
    ids=["lateral-values", "lateral-function", "rows-from", "xmltable", "read_parquet", "json_each"],
)
def test_a_source_shape_this_engine_does_not_model_is_refused(dialect, body):
    """None of these is enumerated anywhere in the engine. They refuse because the rule is a
    whitelist -- every source must be a named table or a subquery over one -- so a shape nobody
    thought of lands in the refuse branch instead of the allow branch. The two lateral cases
    are the ones that defeated the previous rule by hiding a filtered read from scope
    resolution."""
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition=body)}
    verdict = decide("SELECT a FROM v", GRANTED, dialect=dialect, target=dialect,
                     policy=FILTERED, views=views)
    assert isinstance(verdict, Refusal), f"an unmodelled source was allowed: {verdict}"


def test_the_ordinary_shapes_still_pass_the_whitelist():
    """The whitelist has to admit real views or it is just an outage."""
    for body in ("SELECT customer_id AS a FROM public.film",
                 "SELECT f.film_id AS a FROM film f JOIN film g ON f.film_id = g.film_id",
                 "SELECT a FROM (SELECT film_id AS a FROM film) inner_q",
                 "SELECT film_id AS a FROM film UNION ALL SELECT film_id AS a FROM film"):
        verdict = plan("SELECT a FROM v", view(body), FILTERED)
        assert isinstance(verdict, Approved), f"{body!r} was refused: {verdict}"


def test_a_view_qualified_in_the_body_is_still_recognised_as_a_view():
    """Nested-view lookup was keyed on `object_key` only, so a body saying `public.inner_v`
    never recursed and the inner view's filtered base went unseen."""
    views = {
        "v": ViewDefinition(object_id="v", dialect=D, definition="SELECT a FROM public.inner_v"),
        "inner_v": ViewDefinition(object_id="inner_v", dialect=D,
                                  definition="SELECT customer_id AS a FROM customer"),
    }
    verdict = plan("SELECT a FROM v", views, FILTERED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW


# --- round seven: a subquery's own root is a source too ---------------------------


@pytest.mark.parametrize(
    "body",
    [
        # sqlglot renders a parenthesised join as Subquery(Table-with-joins) and a pivot as
        # Subquery(Pivot). In both the ROOT source sits in neither a From nor a Join, so it was
        # never enumerated -- and the real table name lives inside a string literal where
        # nothing can read it. DuckDB executes both.
        "SELECT 1 AS a FROM (query_table('customer') JOIN film ON true) q",
        "SELECT * FROM (PIVOT query_table('customer') ON c USING sum(x)) p",
    ],
    ids=["parenthesised-join-over-function", "pivot-over-function"],
)
def test_a_subquerys_own_root_source_is_checked_too(body):
    views = {"v": ViewDefinition(object_id="v", dialect="duckdb", definition=body)}
    verdict = decide("SELECT a FROM v", GRANTED, dialect="duckdb", target="duckdb",
                     policy=FILTERED, views=views)
    assert isinstance(verdict, Refusal), f"a hidden dynamic source was allowed: {verdict}"


def test_a_function_source_is_recognised_by_node_type_not_by_an_empty_name():
    """`query_table(...)` parses as `Table(Anonymous)`. The empty name was a symptom; the node
    type is the fact, and checking the fact does not depend on the symptom holding."""
    from mnemiq.sql.views import unrecognised_source
    import sqlglot

    body = sqlglot.parse_one("SELECT * FROM query_table('customer')", read="duckdb")
    assert unrecognised_source(body) is not None


def test_a_case_varying_reference_still_matches_a_filtered_table():
    """Unquoted identifiers fold in all three engines, so `PUBLIC.CUSTOMER` is `customer`."""
    verdict = plan("SELECT a FROM v",
                   view("SELECT customer_id AS a FROM PUBLIC.CUSTOMER"), FILTERED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW


def test_a_bare_reference_matches_a_filter_keyed_with_a_qualifier():
    """The other direction of the two-spelling match: filter keyed `public.customer`, body
    saying `customer`. A miss here means not refusing."""
    policy = AccessPolicy(row_filters={"public.customer": "store_id = 1"},
                          policy_schema={"public.customer": {"customer_id", "store_id"}})
    verdict = plan("SELECT a FROM v", view("SELECT customer_id AS a FROM customer"), policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW


@pytest.mark.parametrize(
    "body",
    [
        "SELECT * FROM ((query_table('customer'))) q",
        "SELECT * FROM (((( query_table('customer') )))) q",
        "SELECT * FROM (SELECT * FROM ((query_table('customer'))) i) o",
        "WITH c AS (SELECT * FROM ((query_table('customer')))) SELECT * FROM c",
    ],
    ids=["double-paren", "quadruple-paren", "nested-in-derived", "nested-in-cte"],
)
def test_a_function_source_cannot_hide_behind_nested_containers(body):
    """Checking source POSITIONS caught `Subquery(Table)` and `Subquery(Pivot)` and still
    missed a Subquery inside a Subquery -- that satisfies the Query test while its root sits in
    no From or Join slot. Containers nest arbitrarily; the node type does not move, so the
    check is on the node type and not on where it sits.

    Found by the Verity lane asking whether anything else in the model treats a CONTAINER as
    not-a-source -- the same class, a different node type."""
    views = {"v": ViewDefinition(object_id="v", dialect="duckdb", definition=body)}
    verdict = decide("SELECT a FROM v", GRANTED, dialect="duckdb", target="duckdb",
                     policy=FILTERED, views=views)
    assert isinstance(verdict, Refusal), f"a function hid behind parentheses: {verdict}"


def test_a_body_may_only_read_objects_the_snapshot_knows():
    """A caller writing `SELECT a FROM 'customer.csv'` is already refused UNAUTHORIZED_TABLE --
    the name is not in `visible`. A view body was never held to that, so a view could reach a
    file the caller could not. That is M27's shape again: the view reaching past the caller's
    own boundary."""
    verdict = decide("SELECT a FROM v", GRANTED, dialect="duckdb", target="duckdb",
                     policy=FILTERED,
                     views={"v": ViewDefinition(object_id="v", dialect="duckdb",
                                                definition="SELECT a FROM 'customer.csv'")})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_the_known_object_rule_closes_every_unknown_spelling_not_just_files():
    verdict = decide("SELECT a FROM v", GRANTED, dialect=D, target=D, policy=FILTERED,
                     views={"v": ViewDefinition(object_id="v", dialect=D,
                                                definition="SELECT a FROM nowhere_at_all")})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


@pytest.mark.parametrize(
    "body",
    [
        "SELECT 1 AS a FROM (film JOIN film g ON film.film_id = g.film_id) q",
        "SELECT * FROM (PIVOT film ON title USING count(film_id)) p",
        "SELECT * FROM ((VALUES (hidden()))) v(x)",
    ],
    ids=["parenthesised-named-join", "pivot-over-named-table", "values-over-function"],
)
def test_a_subquery_must_hold_a_query_even_though_that_refuses_working_sql(body):
    """The deliberate trade, and I made it the wrong way round once.

    Allowing a Subquery to hold anything was measured as removing the floor's material
    over-refusal -- parenthesised joins and pivots over named tables, which the source runs
    happily. It also opened an EXECUTABLE escape: `((VALUES (hidden())))` produces no
    `exp.Table` at all, so the node-type scan has nothing to scan and the raw walk has nothing
    to collect. A scalar macro reading a governed table was built as a view and returned its
    rows, in DuckDB and in SQLite both.

    So the first two shapes refuse for the third one's sake. That is the cost, it is
    availability rather than safety, and it is written down here rather than discovered again:
    reducing over-refusal is not worth a hole, and a cost measurement is not a safety argument.
    """
    views = {"v": ViewDefinition(object_id="v", dialect="duckdb", definition=body)}
    verdict = decide("SELECT a FROM v", GRANTED, dialect="duckdb", target="duckdb",
                     policy=FILTERED, views=views)
    assert isinstance(verdict, Refusal), verdict


def test_an_ordinary_cte_view_is_not_refused_for_naming_something_the_source_lacks():
    """A CTE alias is a name the BODY defines. Counting it as an unknown object refused every
    ordinary CTE view -- five of eight new refusals in the review's matrix were legitimate SQL."""
    verdict = plan("SELECT a FROM v",
                   view("WITH recent AS (SELECT film_id AS a FROM film) SELECT a FROM recent"),
                   FILTERED)
    assert isinstance(verdict, Approved), verdict


def test_but_a_cte_named_after_a_filtered_table_still_refuses():
    """Exempting CTE aliases from the KNOWN check weakens nothing: `_mentions` still collects
    the alias, so the filtered check still trips."""
    verdict = plan("SELECT a FROM v",
                   view("WITH customer AS (SELECT film_id AS a FROM film) SELECT a FROM customer"),
                   FILTERED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW

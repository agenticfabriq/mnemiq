"""M176: an Oracle view reading a filtered table under a spelling Oracle folds onto it is refused.

#148 (M154) made the known-object check fold a view body's names as the view's engine folds, so
on Oracle `orderſ` (long s) is recognised as ORDERS. The restricting comparison, the body's
mentions against the filtered set, still folded with `str.lower`, which keeps ſ apart from s, so a
view whose body read `orderſ` was approved where `orders` was refused, and read the filtered
table unfiltered. Both sides of the restricting comparison now fold the engine's way as well as
the wide one: a match refuses, so the union is never narrower than either. Measured on main
`2b28f1eb` through `decide` as oracle.
"""
from __future__ import annotations

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode
from mnemiq.sql.views import ViewDefinition

SCHEMA = {"orders": {"id", "store_id"}, "client": {"id"}, "ΚΟΣΜΟΣ": {"id"}, "stores": {"id"}, "v": {"id"}}


def _read_view(body: str, filtered: str, dialect: str = "oracle"):
    policy = AccessPolicy(row_filters={filtered: "1 = 1"}, policy_schema=SCHEMA)
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition=body)}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect=dialect, target=dialect, policy=policy, views=views)


@pytest.mark.parametrize("filtered", ["orders", "ORDERS"])
@pytest.mark.parametrize("body", [
    "SELECT id FROM orderſ",
    "SELECT id FROM ORDERſ",
    "SELECT id FROM app.orderſ",
    "SELECT id FROM (SELECT id FROM orderſ) x",
    "SELECT id FROM stores WHERE id IN (SELECT id FROM orderſ)",
])
def test_a_long_s_spelling_of_a_filtered_table_is_refused_as_the_plain_one_is(body, filtered):
    verdict = _read_view(body, filtered)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (body, verdict)
    plain = _read_view(body.replace("ſ", "s").replace("ORDERs", "ORDERS"), filtered)
    assert isinstance(plain, Refusal) and plain.code is RefusalCode.UNGOVERNED_VIEW, plain  # the control


@pytest.mark.parametrize(("body", "filtered"), [
    ("SELECT id FROM clıent", "client"),  # dotless i: Oracle folds it to I
    ("SELECT id FROM κοσμος", "ΚΟΣΜΟΣ"),  # final sigma: Oracle folds it to Σ
])
def test_the_dotless_i_and_the_final_sigma_are_refused_too(body, filtered):
    verdict = _read_view(body, filtered)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (body, verdict)


def test_a_long_s_spelling_of_an_unfiltered_table_is_still_read():
    # The fold adds refusals only where a filtered table is reached: `ſtores` is Oracle's STORES.
    verdict = _read_view("SELECT id FROM ſtores", "orders")
    assert isinstance(verdict, Approved), verdict


def _read_through_a_nested_view(spelled: str, dialect: str = "oracle"):
    schema = {"orders": {"id"}, "sales": {"id"}, "v": {"id"}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    views = {"sales": ViewDefinition(object_id="sales", dialect=dialect, definition="SELECT id FROM orders"),
             "v": ViewDefinition(object_id="v", dialect=dialect, definition=f"SELECT id FROM {spelled}")}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect=dialect, target=dialect, policy=policy, views=views)


@pytest.mark.parametrize("spelled", ["ſales", "app.ſales", "SALEſ", "(SELECT id FROM ſales) x"])
def test_a_view_read_under_a_spelling_oracle_folds_onto_it_is_walked(spelled):
    # The known-object check vouched for `ſales` as SALES, and the nested lookup folded with
    # `str.lower` alone, so it found no view, walked nothing, and `v` was approved over the
    # filtered ORDERS that SALES reads (Codex's review of #192).
    verdict = _read_through_a_nested_view(spelled)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (spelled, verdict)
    plain = _read_through_a_nested_view(spelled.replace("ſ", "s").replace("SALEs", "SALES"))
    assert isinstance(plain, Refusal) and plain.code is RefusalCode.UNGOVERNED_VIEW, plain  # the control


@pytest.mark.parametrize("body", ['SELECT id FROM app."ſALES"', "SELECT id FROM app.ſALES"])
def test_every_view_a_fold_matches_is_walked_not_the_first(body):
    # Two views a wider fold cannot tell apart: APP.SALES reads nothing filtered and sorts first.
    # Walking only the first match approved `app."ſALES"`, which Oracle reads as the filtered
    # APP.ſALES (Codex's second review of #192). The unquoted spelling is Oracle's APP.SALES, and
    # is refused too: the lookup does not resolve quoting, so it walks every candidate.
    schema = {"orders": {"id"}, "safe_table": {"id"}, "APP.ſALES": {"id"}, "APP.SALES": {"id"}, "v": {"id"}}
    views = {"APP.ſALES": ViewDefinition(object_id="APP.ſALES", dialect="oracle", definition="SELECT id FROM orders"),
             "APP.SALES": ViewDefinition(object_id="APP.SALES", dialect="oracle", definition="SELECT id FROM safe_table"),
             "v": ViewDefinition(object_id="v", dialect="oracle", definition=body)}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    verdict = decide("SELECT id FROM v", {"v": {"id"}}, dialect="oracle", target="oracle", policy=policy, views=views)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (body, verdict)


def _over_twins(body: str, dialect: str, views: dict[str, str], visible=("v",)):
    defs = {k: ViewDefinition(object_id=k, dialect=dialect, definition=d) for k, d in views.items()}
    defs.setdefault("v", ViewDefinition(object_id="v", dialect=dialect, definition=body))
    schema = {"orders": {"id"}, "safe_table": {"id"}, **{k: {"id"} for k in defs}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    return decide("SELECT id FROM v", {k: {"id"} for k in visible}, dialect=dialect, target=dialect,
                  policy=policy, views=defs)


@pytest.mark.parametrize(("dialect", "body", "harmless", "filtered"), [
    ("oracle", "SELECT id FROM ſALES", "ſALES", "SALES"),  # unquoted: Oracle reads SALES
    ("postgres", "SELECT id FROM SALES", "SALES", "sales"),  # unquoted: Postgres reads sales
])
def test_an_exact_key_does_not_hide_the_view_the_engine_reads(dialect, body, harmless, filtered):
    # The body's raw spelling is a key, of a view over nothing filtered; the engine folds the same
    # spelling onto another key, a view over the filtered ORDERS. Trusting the exact key walked
    # only the harmless view and approved (Codex's third review of #192).
    verdict = _over_twins(body, dialect, {harmless: "SELECT id FROM safe_table", filtered: "SELECT id FROM orders"})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict


def test_a_view_with_no_twin_is_still_read():
    verdict = _over_twins("SELECT id FROM safe_v", "duckdb",
                          {"safe_v": "SELECT id FROM safe_table", "sales": "SELECT id FROM orders"})
    assert isinstance(verdict, Approved), verdict


def test_a_refusal_through_a_twin_names_only_the_view_the_query_read():
    # The query reads `v`, which the caller holds; `V` folds onto it and reads the filtered table.
    # The refusal is about `v`, and the twin's name stays in the log.
    verdict = _over_twins("SELECT id FROM safe_table", "oracle", {"V": "SELECT id FROM orders"})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict
    assert verdict.subject == "v" and "'V'" not in verdict.message, verdict


def test_on_duckdb_a_long_s_view_name_is_still_unknown():
    verdict = _read_through_a_nested_view("ſales", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def test_on_duckdb_the_long_s_names_another_object_and_is_refused_as_unknown():
    # DuckDB folds ASCII alone (M154), so `orderſ` is not ORDERS there: unknown, not ungoverned.
    verdict = _read_view("SELECT id FROM orderſ", "orders", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict

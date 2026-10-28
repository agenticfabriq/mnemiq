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


def test_on_duckdb_the_long_s_names_another_object_and_is_refused_as_unknown():
    # DuckDB folds ASCII alone (M154), so `orderſ` is not ORDERS there: unknown, not ungoverned.
    verdict = _read_view("SELECT id FROM orderſ", "orders", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict

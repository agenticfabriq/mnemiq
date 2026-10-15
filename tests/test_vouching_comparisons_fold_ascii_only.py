"""M154: a name comparison that GRANTS or VOUCHES folds case no wider than the engine.

DuckDB, SQLite and Postgres fold ASCII letters only (measured in `test_identifiers_fold_ascii_only`:
a table `ä` behind a CTE `Ä` reads the table on all three); Oracle folds `ä` to `Ä`. Python's
`str.lower` folds every letter, and two comparisons that vouch used it:

* `AccessPolicy.names`, which decides whether a card may name a table: a grant on `Ä` showed the
  `orders` card's join to an ungranted `ä`, its key column included.
* The view check's known-object test: a known `Ä` vouched for a view body reading an unknown `ä`,
  and the view was approved (Codex's review of #147).

Comparisons that RESTRICT -- a filter, a mask, a denial, what a body mentions -- keep the wide fold,
which can only apply them to more.
"""

from __future__ import annotations

import pytest

from mnemiq.contract import Column, JoinKey, Relationship, Snapshot, ViewDefinition
from mnemiq.semantic.cards import build_cards
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode


def _orders_card(visible, related):
    snap = Snapshot(version="v1", source_id="s", created_at="t",
                    columns=[Column(id="orders.id", object_id="orders", name="id"),
                             Column(id="orders.rel_id", object_id="orders", name="rel_id")],
                    relationships=[Relationship(id="r", **{"from": "orders"}, to=related,
                                                cardinality="many_to_one",
                                                join_keys=[JoinKey(left="rel_id", right="id")])])
    cards = build_cards(snap, AccessPolicy(visible=visible))
    return next(c for c in cards if c.object_id == "orders").text


def test_a_grant_on_one_case_of_a_non_ascii_letter_does_not_show_a_join_to_the_other():
    assert "joins ä" not in _orders_card({"orders", "Ä"}, "ä")


@pytest.mark.parametrize("granted,related", [("ä", "ä"), ("CLAIM", "claim")])
def test_a_join_to_a_granted_table_is_still_shown(granted, related):
    """The controls: the same name, and an ASCII case variant, which every engine folds."""
    assert f"joins {related}" in _orders_card({"orders", granted}, related)


INVENTORY = {"orders": {"id", "store_id"}, "Ä": {"id"}}


def _decide_view(body, dialect, inventory=INVENTORY):
    policy = AccessPolicy(row_filters={"orders": "store_id = 1"}, policy_schema=inventory)
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition=body)}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect="duckdb", target="duckdb",
                  policy=policy, views=views)


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb", "postgres"])
@pytest.mark.parametrize("body", ["SELECT id FROM ä", "WITH Ä AS (SELECT 1 AS id) SELECT id FROM ä"])
def test_a_known_table_does_not_vouch_for_an_unknown_one_in_its_other_non_ascii_case(dialect, body):
    verdict = _decide_view(body, dialect)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


@pytest.mark.parametrize("dialect,body", [
    ("duckdb", "SELECT id FROM ORDERS"),   # ASCII case: one object everywhere
    ("oracle", "SELECT id FROM ä"),        # Oracle folds `ä` to the known `Ä`
])
def test_a_spelling_the_engine_folds_to_a_known_table_is_still_known(dialect, body):
    verdict = _decide_view(body, dialect)
    assert not (isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW), \
        verdict

"""M154: a name comparison that GRANTS or VOUCHES folds case no wider than the engine.

DuckDB, SQLite and Postgres fold ASCII letters only (measured in `test_identifiers_fold_ascii_only`:
a table `ä` behind a CTE `Ä` reads the table on all three); Oracle folds `ä` to `Ä`. Python's
`str.lower` folds every letter, and three comparisons that vouch used it, each reproduced:

* `AccessPolicy.names`, which decides whether a card may name a table: a grant on `Ä` showed the
  `orders` card's join to an ungranted `ä`, its key column included.
* The view check's known-object test: a known `Ä` vouched for a view body reading an unknown `ä`,
  and the view was approved (Codex's review of #147).
* The facts grant check (`retrieval._scoped_facts`): a grant on `Ä` showed a fact depending on an
  ungranted `ä` (the review gate).

Oracle folds each letter to its ONE-character uppercase, and keeps `ß` (measured: `straße` is
stored `STRAßE`), where Python's `str.upper` merges `straße` with `strasse`.

Comparisons that RESTRICT -- a filter, a mask, a denial, what a body mentions -- keep the wide fold,
which can only apply them to more.
"""

from __future__ import annotations

import os

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


def _fact_shown(objects):
    import json

    from mnemiq.authz.grants import GrantSet
    from mnemiq.enrichment.facts import parse_facts
    from mnemiq.semantic.retrieval import RetrievedCard, _attach_facts

    columns = [Column(id="orders.id", object_id="orders", name="id"),
               Column(id="orders.rel_id", object_id="orders", name="rel_id"),
               Column(id="ä.id", object_id="ä", name="id")]
    rels = [Relationship(id="r", **{"from": "orders"}, to="ä", cardinality="many_to_one",
                         join_keys=[JoinKey(left="rel_id", right="id")])]
    text = "join ä for the secret plan"
    tf = parse_facts(json.dumps({"gotchas": [{"text": text, "reads": []}]}), "orders",
                     {"id", "rel_id"}, {"ä"}, related_tables={"ä": {"id"}})
    assert tf.reads[text] == ["ä"]
    card = RetrievedCard(object_id="orders", card="TABLE orders", score=1.0)
    _attach_facts([card], [tf], grants=GrantSet(frozenset(objects)), columns=columns,
                  relationships=rels)
    return "secret plan" in card.card


def test_a_grant_on_one_case_of_a_non_ascii_letter_does_not_show_a_fact_about_the_other():
    assert not _fact_shown({"orders", "Ä"})
    assert _fact_shown({"orders", "ä"})  # the control


def test_oracle_folds_each_letter_by_its_one_character_uppercase():
    from mnemiq.sql.identifiers import resolve

    def oracle(name):
        return resolve(name, quoted=False, dialect="oracle")

    assert oracle("ä") == oracle("Ä") and oracle("orderſ") == oracle("ORDERS")
    assert oracle("straße") != oracle("strasse")


@pytest.mark.parametrize("known,body,refused", [
    ("STRASSE", "SELECT id FROM straße", True),   # Oracle keeps ß: an unknown object
    ("ORDERS", "SELECT id FROM orderſ", False),   # Oracle folds ſ to S: the known one
])
def test_an_oracle_view_s_known_objects_fold_as_oracle_does(known, body, refused):
    verdict = _decide_view(body, "oracle", {"orders": {"id", "store_id"}, known: {"id"}})
    unresolvable = isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW
    assert unresolvable is refused, verdict


@pytest.mark.skipif(not os.environ.get("MNEMIQ_ORACLE_TEST_DSN"),
                    reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_live_oracle_folds_identifiers_as_the_rule_says():
    """Live: Oracle's own folding of each letter, read back as the name of an unquoted alias, so
    no table is created. `ſ`, `ä` and dotless `ı` fold; `ß` is kept; a Greek letter with a
    subscript iota folds where this rule keeps it, the one measured direction it differs in."""
    from mnemiq.sql.identifiers import resolve

    oracledb = pytest.importorskip("oracledb")
    letters = ["ſ", "ä", "ı", "ß", "\u1f80"]
    with oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                          password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", ""),
                          dsn=os.environ["MNEMIQ_ORACLE_TEST_DSN"]) as con:
        cur = con.cursor()
        cur.execute("SELECT " + ", ".join(f"1 AS x{c}" for c in letters) + " FROM dual")
        oracle = {c: d[0][1:] for c, d in zip(letters, cur.description)}
    assert oracle == {"ſ": "S", "ä": "Ä", "ı": "I", "ß": "ß", "\u1f80": "\u1f88"}
    ours = {c: resolve(c, quoted=False, dialect="oracle") for c in letters}
    assert ours == {**oracle, "\u1f80": "\u1f80"}  # the one difference: ours keeps, Oracle folds

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


@pytest.mark.parametrize("kept", ["\u2170", "\u24d0", "\u0345"])  # ⅰ, ⓐ, combining ypogegrammeni
def test_oracle_keeps_a_non_letter_whose_uppercase_is_one_character(kept):
    """No measurement says Oracle folds a small Roman numeral, a circled letter or a combining
    mark, so the rule keeps each -- the narrowing direction -- though Python uppercases them."""
    from mnemiq.sql.identifiers import resolve

    assert len(kept.upper()) == 1 and kept.upper() != kept
    assert resolve(kept, quoted=False, dialect="oracle") == kept


@pytest.mark.parametrize("known,body,refused", [
    ("STRASSE", "SELECT id FROM straße", True),   # Oracle keeps ß: an unknown object
    ("ORDERS", "SELECT id FROM orderſ", False),   # Oracle folds ſ to S: the known one
])
def test_an_oracle_view_s_known_objects_fold_as_oracle_does(known, body, refused):
    verdict = _decide_view(body, "oracle", {"orders": {"id", "store_id"}, known: {"id"}})
    unresolvable = isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW
    assert unresolvable is refused, verdict


_ORACLE = pytest.mark.skipif(not os.environ.get("MNEMIQ_ORACLE_TEST_DSN"),
                             reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")


def _oracle_connection():
    oracledb = pytest.importorskip("oracledb")
    return oracledb, oracledb.connect(
        user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
        password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", ""),
        dsn=os.environ["MNEMIQ_ORACLE_TEST_DSN"])


def _alias_folds(cur, oracledb, letters):
    """Oracle's folding of each letter, read back as the name of an unquoted alias `x<letter>`;
    None for a letter Oracle refuses unquoted. Halves a batch Oracle rejects."""
    try:
        cur.execute("SELECT " + ", ".join(f"1 AS x{c}" for c in letters) + " FROM dual")
        return {c: d[0][1:] for c, d in zip(letters, cur.description)}
    except oracledb.DatabaseError:
        if len(letters) == 1:
            return {letters[0]: None}
        mid = len(letters) // 2
        return {**_alias_folds(cur, oracledb, letters[:mid]),
                **_alias_folds(cur, oracledb, letters[mid:])}


@_ORACLE
def test_live_oracle_folds_table_names_and_aliases_alike():
    """Live: a TABLE name folds as an alias does, for the letters the rule turns on -- the
    sweep reads aliases, and what the checks compare is object names."""
    oracledb, con = _oracle_connection()
    with con:
        cur = con.cursor()
        for table in ("fold_probe_s", "fold_probe_ß"):
            try:  # a run killed before its own cleanup left the table behind
                cur.execute(f"DROP TABLE {table} PURGE")
            except oracledb.DatabaseError:
                pass
        try:
            cur.execute("CREATE TABLE fold_probe_s (id NUMBER)")
            cur.execute("INSERT INTO fold_probe_s VALUES (40)")
            cur.execute("SELECT id FROM fold_probe_ſ")
            assert cur.fetchall() == [(40,)]
            cur.execute("CREATE TABLE fold_probe_ß (id NUMBER)")
            cur.execute("SELECT table_name FROM user_tables WHERE table_name LIKE 'FOLD_PROBE_%'")
            assert sorted(r[0] for r in cur.fetchall()) == ["FOLD_PROBE_S", "FOLD_PROBE_ß"]
            assert _alias_folds(cur, oracledb, ["ſ", "ß"]) == {"ſ": "S", "ß": "ß"}
        finally:
            for table in ("fold_probe_s", "fold_probe_ß"):
                try:
                    cur.execute(f"DROP TABLE {table} PURGE")
                except oracledb.DatabaseError:
                    pass


@_ORACLE
def test_live_oracle_sweep_never_folds_a_letter_oracle_keeps():
    """Live, the sweep the Oracle rule rests on: every letter below U+3000 and every cased letter
    above it. None may be folded by the rule where Oracle keeps it -- the direction that vouches.
    The one measured difference the other way is Greek with a subscript iota (U+1F80 on), which
    Oracle folds and the rule keeps."""
    from mnemiq.sql.identifiers import resolve

    letters = [chr(cp) for cp in range(0x80, 0x110000)
               if not 0xD800 <= cp <= 0xDFFF and chr(cp).isalpha()
               and (cp < 0x3000 or chr(cp).upper() != chr(cp) or chr(cp).lower() != chr(cp))]
    oracledb, con = _oracle_connection()
    with con:
        cur = con.cursor()
        folded = {}
        for i in range(0, len(letters), 200):
            folded.update(_alias_folds(cur, oracledb, letters[i:i + 200]))
    ours = {c: resolve(c, quoted=False, dialect="oracle") for c, v in folded.items() if v}
    # A floor, or a session that refuses everything -- a non-AL32UTF8 charset, a lost connection --
    # compares nothing and passes. 6,679 answered on Oracle Free 23 (AL32UTF8).
    assert len(ours) >= 6679 and {"ſ", "ä", "ı"} <= ours.keys(), len(ours)
    wider = [c for c in ours if folded[c] == c and ours[c] != c]
    differ = {c for c in ours if ours[c] != folded[c]}
    assert not wider, [f"U+{ord(c):04X}" for c in wider]
    two_char_greek = {chr(cp) for cp in range(0x1F80, 0x2000) if len(chr(cp).upper()) > 1}
    assert differ <= two_char_greek, sorted(f"U+{ord(c):04X}" for c in differ - two_char_greek)

"""Codex's review of #130 (M143): the name scan reads a known name whole, spaces and punctuation in it.

The scan split the text into identifier tokens, so a column named `secret code` or a related
table named `layoff plans` -- legal quoted identifiers, and common in BIRD-style schemas
(`Enrollment (K-12)`) -- matched nothing. With the model listing nothing either, the fact
recorded an empty list and reached every identity, the PHI column's included. Each known name is
now matched as a whole phrase, case and whitespace folded on both sides.

Not addressed here, by the owner's decision (2026-10-14): a fact that PARAPHRASES a column the
model then omits from its list is still recorded without it. That is the residual of the
model-plus-scan design, kept.
"""

from __future__ import annotations

import json

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, JoinKey, Relationship
from mnemiq.enrichment.facts import parse_facts
from mnemiq.semantic.retrieval import RetrievedCard, _attach_facts

OWN = {"id", "secret code", "Enrollment (K-12)"}
RELATED = {"layoff plans": {"plan_id"}}


def _reads(text: str) -> list[str]:
    tf = parse_facts(json.dumps({"gotchas": [{"text": text, "reads": []}]}), "orders", OWN,
                     {"plan_id"}, related_tables=RELATED)
    return tf.reads[text]


def test_a_column_name_with_a_space_is_found_whatever_its_case():
    assert _reads("Secret Code contains diagnoses") == ["orders.secret code"]


def test_a_related_table_name_with_a_space_is_found():
    assert "layoff plans" in _reads("join layoff plans before counting")


def test_a_name_with_punctuation_is_found():
    assert "orders.Enrollment (K-12)" in _reads("Enrollment (K-12) is stored as a float")


def test_a_short_name_inside_a_longer_word_is_not_a_mention():
    assert _reads("idle rows are archived") == []


def test_the_withheld_column_never_reaches_an_identity_that_cannot_read_it():
    """The render: the recorded dependency withholds the line, where the empty list showed it."""
    text = "secret code contains diagnoses"
    tf = parse_facts(json.dumps({"gotchas": [{"text": text, "reads": []}]}), "orders",
                     {"id", "secret code"}, {"plan_id"}, related_tables=RELATED)
    columns = [Column(id="orders.id", object_id="orders", name="id"),
               Column(id="orders.secret code", object_id="orders", name="secret code",
                      pii_level="phi")]
    rels = [Relationship(id="r", **{"from": "orders"}, to="layoff plans",
                         cardinality="many_to_one", join_keys=[JoinKey(left="id", right="plan_id")])]

    def block(grants):
        card = RetrievedCard(object_id="orders", card="TABLE orders", score=1.0)
        _attach_facts([card], [tf], grants=grants, columns=columns, relationships=rels)
        return card.card

    assert "diagnoses" not in block(GrantSet(frozenset({"orders", "layoff plans"})))
    assert "diagnoses" in block(GrantSet(frozenset({"orders", "layoff plans"}),
                                         pii_clearance=frozenset({"phi"})))


def test_the_old_tokenizers_reach_is_kept():
    """Review gate: Unicode word boundaries counted a digit before a name, or an accented letter
    after it, as part of the word, and missed what the old ASCII tokenizer found."""
    vocab_columns = {"ssn"}
    for text in ("2ssn values are padded", "$ssn is a bind name", "ssn\u00e9 is masked"):
        tf = parse_facts(json.dumps({"gotchas": [{"text": text, "reads": []}]}), "orders",
                         vocab_columns, set(), related_tables={})
        assert tf.reads[text] == ["orders.ssn"], text


def test_a_name_with_a_run_of_whitespace_is_found():
    """The fact's text has its whitespace folded (`_clean_text`); the name is folded the same."""
    tf = parse_facts(json.dumps({"gotchas": [{"text": "secret code holds x", "reads": []}]}),
                     "orders", {"secret  code"}, set(), related_tables={})
    assert tf.reads["secret code holds x"] == ["orders.secret  code"]

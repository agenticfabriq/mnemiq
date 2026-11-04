"""The frontier profile is today's prompt, byte for byte.

Work on a smaller model's profile must not move a frontier model's results, and nothing else in the suite
would notice if it did: a changed rule or card line reads as an improvement until a benchmark says otherwise.
A hash fails on any change at all, so changing this prompt is a decision. If it is meant, re-measure the
frontier baseline first, then update the hash here.
"""

from __future__ import annotations

import hashlib

from mnemiq.contract import CodedValue, Column, JoinKey, Relationship, Snapshot, SourceBinding
from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.semantic.cards import build_cards
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

SYSTEM_SHA = "2223e5947946df88cb31f6135078a9e7dbacf79e3bdbf231661bacd8f4c868e3"
USER_SHA = "b23fc19802c5c414fcb34411c2748a30b8fd4cf2a65e4b6f9250349c2507ee1d"


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[
            Column(id="orders.id", object_id="orders", name="id", data_type="INTEGER", semantic_type="identifier",
                   description="The order."),
            Column(id="orders.state", object_id="orders", name="state", data_type="VARCHAR",
                   semantic_type="category", description="Where the order is.", row_count=9, null_count=0,
                   coded_values=[CodedValue(code="O", meaning="open"), CodedValue(code="C", meaning="closed")]),
            Column(id="orders.gone", object_id="orders", name="gone", data_type="VARCHAR", row_count=9, null_count=9),
            Column(id="orders.account_id", object_id="orders", name="account_id", data_type="INTEGER"),
            Column(id="account.id", object_id="account", name="id", data_type="INTEGER"),
        ],
        relationships=[Relationship(id="r", **{"from": "orders"}, to="account", cardinality="many_to_one",
                                    join_keys=[JoinKey(left="account_id", right="id")])],
        source_bindings=[SourceBinding(id=t, source_id="s", object_id=t, source_object=t, binding_type="table")
                         for t in ("orders", "account")],
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def rendered() -> tuple[str, str]:
    cards = [RetrievedCard(object_id=c.object_id, card=c.text, score=1.0) for c in build_cards(_snapshot())]
    packet = ContextPacket(question="How many open orders?", cards=cards, grant_fingerprint="fp",
                           enrichment_version=None)
    return system_prompt(dialect="duckdb"), user_prompt(packet)


def test_the_frontier_prompt_has_not_moved():
    system, user = rendered()
    assert _sha(system) == SYSTEM_SHA
    assert _sha(user) == USER_SHA

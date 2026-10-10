"""A card served to an identity names no table that identity is not granted, not even as a join.

`retrieve` promises that no packet discloses a table the identity may not access, since a table
named `layoff_plans` discloses by its mere existence. The card's columns were scoped (M4), but its
RELATIONSHIPS section was rendered from the whole snapshot, so a granted `orders` card said
`joins layoff_plans (many-to-one: orders.plan_id = layoff_plans.id)` to an identity granted
`orders` alone. Found by Codex reviewing the eval's per-question grants.
"""
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, IdentityContext, Relationship, Snapshot, SourceBinding
from mnemiq.contract.values import JoinKey
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store

SNAP = Snapshot(
    version="v", source_id="s", created_at="t",
    columns=[Column(id="orders.id", object_id="orders", name="id"),
             Column(id="orders.plan_id", object_id="orders", name="plan_id"),
             Column(id="layoff_plans.id", object_id="layoff_plans", name="id"),
             Column(id="layoff_plans.cut_date", object_id="layoff_plans", name="cut_date")],
    source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                   binding_type="table") for t in ("orders", "layoff_plans")],
    relationships=[Relationship(id="r1", from_="orders", to="layoff_plans", cardinality="many-to-one",
                                join_keys=[JoinKey(left="orders.plan_id", right="layoff_plans.id")])],
)


class _Grants:
    def __init__(self, *objects):
        self._grants = GrantSet(frozenset(objects))

    def grants_for(self, identity):
        return self._grants


def _orders_card(tmp_path, granted, style="cards"):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, SNAP, FakeEmbedder())
    packet = retrieve(con, "orders plan", IdentityContext(tenant_id="t", principal_id="p", roles=[]),
                      _Grants(*granted), FakeEmbedder(), snapshot=SNAP, card_style=style)
    return next(c.card for c in packet.cards if c.object_id == "orders")


@pytest.mark.parametrize("style", ["cards", "ddl"])
def test_a_granted_card_does_not_name_a_table_the_identity_is_not_granted(tmp_path, style):
    card = _orders_card(tmp_path, {"orders"}, style)
    assert "layoff_plans" not in card
    assert "plan_id" in card, "the column itself is the identity's to see"


@pytest.mark.parametrize("style", ["cards", "ddl"])
def test_the_join_is_shown_when_both_tables_are_granted(tmp_path, style):
    assert "orders.plan_id = layoff_plans.id" in _orders_card(tmp_path, {"orders", "layoff_plans"}, style)


def test_a_grant_spelled_in_another_case_still_shows_the_join(tmp_path):
    # Unquoted identifiers are case-insensitive in every engine mnemiq targets, so a grant on
    # `LAYOFF_PLANS` is a grant on `layoff_plans`, as the policy reads it elsewhere.
    assert "layoff_plans" in _orders_card(tmp_path, {"orders", "LAYOFF_PLANS"})

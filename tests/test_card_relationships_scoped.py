"""A card served to an identity names no table it is not granted, nor a key column it is denied, as a join.

`retrieve` promises that no packet discloses a table the identity may not access, since a table
named `layoff_plans` discloses by its mere existence. The card's columns were scoped (M4), but its
RELATIONSHIPS section was rendered from the whole snapshot, so a granted `orders` card said
`joins layoff_plans (many-to-one: plan_id = id)` to an identity granted `orders` alone (M142).
Found by Codex reviewing the eval's per-question grants.

Scope of the guarantee: the card as `build_cards` renders it for a bound snapshot, which is how both
doors call `retrieve`. Without a snapshot the stored, unscoped card is served (M4's fallback), and
the facts block attached after rendering is free text no grant can filter.
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
                                join_keys=[JoinKey(left="plan_id", right="id")])],
)


class _Grants:
    def __init__(self, *objects, cleared=frozenset()):
        self._grants = GrantSet(frozenset(objects), pii_clearance=frozenset(cleared))

    def grants_for(self, identity):
        return self._grants


def _orders_card(tmp_path, granted, style="cards", snap=SNAP, cleared=frozenset()):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, snap, FakeEmbedder())
    packet = retrieve(con, "orders plan", IdentityContext(tenant_id="t", principal_id="p", roles=[]),
                      _Grants(*granted, cleared=cleared), FakeEmbedder(), snapshot=snap,
                      card_style=style)
    return next(c.card for c in packet.cards if c.object_id == "orders")


@pytest.mark.parametrize("style", ["cards", "ddl"])
def test_a_granted_card_does_not_name_a_table_the_identity_is_not_granted(tmp_path, style):
    card = _orders_card(tmp_path, {"orders"}, style)
    assert "layoff_plans" not in card
    assert "plan_id" in card, "the column itself is the identity's to see"


@pytest.mark.parametrize("style", ["cards", "ddl"])
def test_the_join_is_shown_when_both_tables_are_granted(tmp_path, style):
    assert "joins layoff_plans" in _orders_card(tmp_path, {"orders", "layoff_plans"}, style)


def test_a_grant_spelled_in_another_case_still_shows_the_join(tmp_path):
    # Unquoted identifiers are case-insensitive in every engine mnemiq targets, so a grant on
    # `LAYOFF_PLANS` is a grant on `layoff_plans`, as the policy reads it elsewhere.
    assert "layoff_plans" in _orders_card(tmp_path, {"orders", "LAYOFF_PLANS"})


@pytest.mark.parametrize("style", ["cards", "ddl"])
def test_a_join_on_a_denied_key_column_is_not_shown(tmp_path, style):
    # The card leaves a denied column out "so the identity is not told it exists"; a join line
    # naming it as the key would tell it anyway.
    personal = SNAP.model_copy(update={"columns": [
        c.model_copy(update={"pii_level": "high"}) if c.id == "orders.plan_id" else c
        for c in SNAP.columns]})
    both = {"orders", "layoff_plans"}
    assert "plan_id" not in _orders_card(tmp_path, both, style, snap=personal)
    assert "joins layoff_plans" in _orders_card(tmp_path, both, style, snap=personal, cleared={"high"})


def test_a_denied_key_whose_name_holds_a_dot_is_checked_as_written(tmp_path):
    # A quoted column can be named `plan.id`; cutting it at the dot would check `id` and pass it.
    dotted = SNAP.model_copy(update={
        "columns": [c.model_copy(update={"name": "plan.id", "id": "orders.plan.id", "pii_level": "high"})
                    if c.id == "orders.plan_id" else c for c in SNAP.columns],
        "relationships": [SNAP.relationships[0].model_copy(update={"join_keys": [JoinKey(left="plan.id", right="id")]})],
    })
    assert "plan.id" not in _orders_card(tmp_path, {"orders", "layoff_plans"}, snap=dotted)

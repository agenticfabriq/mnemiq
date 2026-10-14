"""M13: one question is answered under one grant set.

`Runtime.ask` resolved the identity's grants twice -- inside `retrieve`, which scoped the cards,
and again for the history, the fit, the agent and the record -- and `FileAuthzProvider` re-reads
its policy file on every call. A policy edit landing between the two meant retrieval scoped the
cards against one file version and the agent decided against another, within one request; the
packet's `grant_fingerprint`, which would have shown it, had no consumer. Grants are now resolved
once per question and handed to retrieval, so the packet's fingerprint is the agent's.
"""

from __future__ import annotations

from mnemiq.agent.loop import AgentAnswer
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.runtime import Runtime
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store

IDENTITY = IdentityContext(tenant_id="t", principal_id="p", roles=["analyst"])
SNAPSHOT = Snapshot(version="v1", source_id="s", created_at="t", columns=[
    Column(id="orders.id", object_id="orders", name="id"),
    Column(id="layoff_plans.id", object_id="layoff_plans", name="id")])


class _EditedMidRequest:
    """A policy file edited between two reads: the first read grants `orders`, every later one
    `orders` and `layoff_plans`."""

    def __init__(self):
        self.calls = 0

    def grants_for(self, _identity):
        self.calls += 1
        objects = {"orders"} if self.calls == 1 else {"orders", "layoff_plans"}
        return GrantSet(frozenset(objects))


class _Agent:
    def decline_if_too_long(self, question):
        return None

    def answer(self, packet, snapshot, grants, identity, emit=None):
        self.packet, self.grants = packet, grants
        return AgentAnswer(answer="ok")


def _runtime(tmp_path, authz, agent):
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, SNAPSHOT, embedder)
    return Runtime(con=con, snapshot=SNAPSHOT, adapter=None, agent=agent, embedder=embedder,
                   authz=authz, settings=None)


def test_the_cards_and_the_agent_are_scoped_by_the_same_grants(tmp_path):
    authz, agent = _EditedMidRequest(), _Agent()
    answer = _runtime(tmp_path, authz, agent).ask("how many orders", IDENTITY)
    assert authz.calls == 1, "one question, one read of the policy"
    assert agent.packet.grant_fingerprint == agent.grants.fingerprint == answer.grant_fingerprint
    # And the cards are the agent's grants' cards: nothing from a second read slipped in.
    assert {c.object_id for c in agent.packet.cards} <= set(agent.grants.objects)


def test_retrieve_still_resolves_grants_itself_when_none_are_handed_to_it(tmp_path):
    """A caller handing `retrieve` no grants still has them resolved by its provider, once. Both
    doors in this repo now pass grants, so this is the fallback for any other caller."""
    from mnemiq.semantic.retrieval import retrieve

    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, SNAPSHOT, embedder)
    authz = _EditedMidRequest()
    packet = retrieve(con, "how many orders", IDENTITY, authz, embedder, snapshot=SNAPSHOT)
    assert authz.calls == 1 and packet.grant_fingerprint == GrantSet(frozenset({"orders"})).fingerprint

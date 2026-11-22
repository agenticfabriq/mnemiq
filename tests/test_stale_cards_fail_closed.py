"""M274: a card the current snapshot does not hold is never served unscoped.

The stored card is rendered once at enrich time from the whole snapshot (M4): it names every column, denied ones
included, with their PII levels and coded values. Retrieval and the schema endpoint re-render each granted card under the
caller's policy, and used to fall back to the stored text for an id the re-render lacked. That happens between a
re-enrich that drops or renames a table and the next build: the index still holds the old card, the snapshot no longer
holds the table, and the old card reached a restricted identity whole. Both now omit such a card, and the schema
endpoint serves none without a snapshot to scope against.
"""
from __future__ import annotations

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.runtime import Runtime
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store

IDENTITY = IdentityContext(tenant_id="t", principal_id="p", roles=["analyst"])

# Indexed at enrich time: two tables, one with a sensitive column.
INDEXED = Snapshot(version="v1", source_id="s", created_at="t", columns=[
    Column(id="orders.id", object_id="orders", name="id"),
    Column(id="orders.amount", object_id="orders", name="amount"),
    Column(id="payroll.id", object_id="payroll", name="id"),
    Column(id="payroll.salary_band", object_id="payroll", name="salary_band")])
# The current snapshot, re-enriched without the payroll table, before the index was rebuilt.
CURRENT = Snapshot(version="v2", source_id="s", created_at="t", columns=[
    Column(id="orders.id", object_id="orders", name="id"),
    Column(id="orders.amount", object_id="orders", name="amount")])


class _Authz:
    def grants_for(self, _identity):
        return GrantSet(frozenset({"orders", "payroll"}))


def _store(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, INDEXED, FakeEmbedder())
    return con


def test_retrieval_omits_a_card_the_current_snapshot_does_not_hold(tmp_path):
    con = _store(tmp_path)
    grants = _Authz().grants_for(IDENTITY)
    packet = retrieve(con, "payroll salary band", IDENTITY, _Authz(), FakeEmbedder(), k=12,
                      snapshot=CURRENT, grants=grants)
    served = {c.object_id for c in packet.cards}
    assert "payroll" not in served, "the old card was served unscoped"
    assert not any("salary_band" in c.card for c in packet.cards)
    assert served <= {"orders"}


def test_retrieval_still_serves_the_tables_the_snapshot_holds(tmp_path):
    con = _store(tmp_path)
    packet = retrieve(con, "orders amount", IDENTITY, _Authz(), FakeEmbedder(), k=12,
                      snapshot=CURRENT, grants=_Authz().grants_for(IDENTITY))
    assert [c.object_id for c in packet.cards] == ["orders"]


def test_the_schema_endpoint_omits_a_card_the_current_snapshot_does_not_hold(tmp_path):
    rt = Runtime(con=_store(tmp_path), snapshot=CURRENT, adapter=None, agent=None, embedder=None,
                 authz=_Authz(), settings=None)
    got = rt.schema(IDENTITY)
    assert [c["object_id"] for c in got] == ["orders"]
    assert not any("salary_band" in c["card"] for c in got)


def test_the_schema_endpoint_serves_no_card_without_a_snapshot(tmp_path):
    rt = Runtime(con=_store(tmp_path), snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=_Authz(), settings=None)
    assert rt.schema(IDENTITY) == []


def _store_with_examples(tmp_path):
    from mnemiq.contract import Example
    from mnemiq.semantic.store import build_example_index

    con = _store(tmp_path)
    indexed = INDEXED.model_copy(update={"examples": [
        Example(question="payroll salary band", sql="SELECT salary_band FROM payroll", tables=["payroll"],
                object_id="payroll"),
        Example(question="orders amount", sql="SELECT amount FROM orders", tables=["orders"], object_id="orders")]})
    build_example_index(con, indexed, FakeEmbedder())
    return con


def test_an_example_on_a_table_the_current_snapshot_does_not_hold_is_withheld(tmp_path):
    # Codex's review of #288: the cards were left out, but example retrieval still took every granted table, and the
    # current snapshot holds no columns of the dropped table to deny, so its stored example SQL reached the prompt.
    con = _store_with_examples(tmp_path)
    packet = retrieve(con, "payroll salary band", IDENTITY, _Authz(), FakeEmbedder(), k=12,
                      snapshot=CURRENT, grants=_Authz().grants_for(IDENTITY))
    assert not any("payroll" in e.tables or "salary_band" in e.sql for e in packet.examples)
    assert any(e.tables == ["orders"] for e in packet.examples), "an example on a held table is still served"


def test_no_eligible_table_means_no_example_and_no_failure(tmp_path, caplog):
    # Granted only the dropped table: nothing is eligible, which is an empty answer, not a failed query.
    import logging

    class OnlyPayroll:
        def grants_for(self, _identity):
            return GrantSet(frozenset({"payroll"}))

    con = _store_with_examples(tmp_path)
    with caplog.at_level(logging.WARNING):
        packet = retrieve(con, "payroll salary band", IDENTITY, OnlyPayroll(), FakeEmbedder(), k=12,
                          snapshot=CURRENT, grants=OnlyPayroll().grants_for(IDENTITY))
    assert packet.examples == [] and packet.cards == []
    assert "example retrieval failed" not in caplog.text

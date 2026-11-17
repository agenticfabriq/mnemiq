"""`repair_hints`, a mode setting, reaches every function whose words it changes: the join, not the ends.

Each fix it gates is tested where it lives; this file proves the mode's value arrives there through the real
`build_agent` -> `Agent` -> `plan_query` -> `decide` -> `check_values` path, on and off. Every spy must have been
called, or "every one received False" would hold over no calls at all.
"""

from __future__ import annotations

import json
from dataclasses import replace

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.generate import plan_query as plan_query_module
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.semantic.values import ValueIndex, build_value_index
from mnemiq.sql import decide as decide_module
from mnemiq.store.bootstrap import init_store

_GRANTS = GrantSet(frozenset({"orders"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_CARD = "TABLE orders: id INTEGER, status VARCHAR (open|closed), amount DOUBLE"
_SQL = "SELECT id, amount FROM orders WHERE status = 'open'"


@pytest.fixture
def source(tmp_path):
    path = str(tmp_path / "src.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE orders (id INTEGER, status VARCHAR, amount DOUBLE)")
    con.executemany("INSERT INTO orders VALUES (?, ?, ?)",
                    [(i, "open" if i % 3 else "closed", float(i)) for i in range(200)])
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    snap = enrich_structural(adapter, "src")
    store = init_store(str(tmp_path / "store.duckdb"))
    build_value_index(adapter, snap, store)
    values = ValueIndex(store)
    assert values.has("orders", "status")  # the premise: check_values has a column to check
    return adapter, snap, values


def _spies(monkeypatch):
    seen = {"decide": [], "check_values": []}
    real_decide, real_check_values = plan_query_module.decide, decide_module.check_values

    def decide(*args, **kwargs):
        seen["decide"].append(kwargs.get("repair_hints", "absent"))
        return real_decide(*args, **kwargs)

    def check_values(*args, **kwargs):
        seen["check_values"].append(kwargs.get("hints", "absent"))
        return real_check_values(*args, **kwargs)

    monkeypatch.setattr(plan_query_module, "decide", decide)
    monkeypatch.setattr(decide_module, "check_values", check_values)
    return seen


def _answer(source, mode):
    adapter, snap, values = source
    agent = build_agent(mode, generator=FakeGenerator([json.dumps({"sql": _SQL})] * 3),
                        synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                        cache=TwoTierCache(L1Cache()), corrector=None, values=values, selector=None,
                        verifier=None)
    packet = ContextPacket(question="Which orders are open?",
                           cards=[RetrievedCard(object_id="orders", card=_CARD, score=1.0)],
                           grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")
    return agent.answer(packet, snap, _GRANTS, _IDENTITY)


def test_the_mode_setting_reaches_decide(source, monkeypatch):
    seen = _spies(monkeypatch)
    ans = _answer(source, replace(MODES["instant"], repair_hints=True))
    assert ans.deferred is False, ans.answer
    assert seen["decide"] and set(seen["decide"]) == {True}
    assert seen["check_values"] and set(seen["check_values"]) == {True}


def test_off_reaches_decide_as_false(source, monkeypatch):
    seen = _spies(monkeypatch)
    ans = _answer(source, MODES["instant"])
    assert ans.deferred is False, ans.answer
    assert seen["decide"] and set(seen["decide"]) == {False}
    assert seen["check_values"] and set(seen["check_values"]) == {False}

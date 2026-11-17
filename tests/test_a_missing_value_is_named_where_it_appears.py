"""An empty result whose filter value is not in its column names, under `repair_hints`, the columns that do hold it.

Measured on a local model: asked about a batch by its ID, it looked the ID up in a column of another kind of ID, got
no rows, and the turn was declined with "does not appear in" that column. The value was one column over. When the
source has said a value is missing from its column, the engine now asks, within the turn's probe budget, the other
columns this caller may read plainly (`sql.locatable`), and the decline, and a retry's feedback, says where it is.

Driven through the real Agent and the product's instant verifier over a DuckDB source the real structural enrichment
profiles. Only the model calls are scripted. Every exclusion is tested beside a column that is probed, so a rule that
probed nothing would fail too.
"""

from __future__ import annotations

import json
from dataclasses import replace

import duckdb
import pytest

import mnemiq.agent.loop as loop_mod
from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.contract.semantic import Job, SourceBinding
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.generate.generator import FakeGenerator
from mnemiq.runtime import _build_mode_verifiers
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.semantic.values import ValueIndex, build_value_index
from mnemiq.store.bootstrap import init_store

_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_GRANTS = GrantSet(frozenset({"orders", "shipments", "returns", "returns_v", "LOTS"}),
                   row_filters={"returns": "id > 0"}, pii_mask=frozenset({"direct"}))
_CARDS = ("orders", "shipments", "returns", "returns_v", "LOTS")
MISSING_SQL = "SELECT id FROM orders WHERE customer_ref = 'B-0017'"
NOWHERE_SQL = "SELECT id FROM orders WHERE customer_ref = 'Z-0001'"
FOUND_SQL = "SELECT id FROM orders WHERE batch_ref = 'B-0017'"
TODAY = ("No rows matched, and 'B-0017' does not appear in orders.customer_ref, so I can't tell a true \"none\" from a "
         "misspelled value.")
NAMED = ("No rows matched, and 'B-0017' does not appear in orders.customer_ref, though it does appear in "
         "orders.batch_ref, shipments.batch_ref and LOTS.batch_ref, so I can't tell a true \"none\" from a misspelled "
         "value.")


@pytest.fixture
def world(tmp_path, monkeypatch):
    for var in ("MNEMIQ_VERIFY", "MNEMIQ_VERIFY_SANITY", "MNEMIQ_VERIFY_BASE_URL", "MNEMIQ_VERIFY_MODEL",
                "MNEMIQ_VERIFY_API_KEY", "MNEMIQ_VERIFY_GROUNDING", "MNEMIQ_VERIFY_THRESHOLD",
                "MNEMIQ_VERIFY_FAIL_CLOSED"):
        monkeypatch.delenv(var, raising=False)
    path = str(tmp_path / "src.duckdb")
    con = duckdb.connect(path)
    n = range(400)
    con.execute("CREATE TABLE orders (id INTEGER, customer_ref VARCHAR, batch_ref VARCHAR, card_hint VARCHAR, "
                "audit_ref VARCHAR, contact_name VARCHAR, x1 VARCHAR, x2 VARCHAR, status VARCHAR)")
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(i, f"C-{i:04d}", f"B-{i:04d}", f"B-{i:04d}", f"B-{i:04d}", f"B-{i:04d}", f"X-{i:04d}",
                      f"Y-{i:04d}", "open" if i % 2 else "closed") for i in n])
    for table in ("shipments", "returns", "ledger"):
        con.execute(f"CREATE TABLE {table} (id INTEGER, batch_ref VARCHAR)")
        con.executemany(f"INSERT INTO {table} VALUES (?, ?)", [(i, f"B-{i:04d}") for i in n])
    con.execute('CREATE TABLE "LOTS" (id INTEGER, batch_ref VARCHAR)')
    con.executemany('INSERT INTO "LOTS" VALUES (?, ?)', [(i, f"B-{i:04d}") for i in n])
    con.execute("CREATE VIEW returns_v AS SELECT id, batch_ref FROM returns")
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    snap = enrich_structural(adapter, "src")
    # The policy's levels: a masked column and a denied one, neither named like personal data.
    levels = {("orders", "card_hint"): "direct", ("orders", "audit_ref"): "restricted"}
    snap = snap.model_copy(update={"columns": [
        c.model_copy(update={"pii_level": levels.get((c.object_id, c.name), c.pii_level)}) for c in snap.columns]})
    store = init_store(str(tmp_path / "store.duckdb"))
    build_value_index(adapter, snap, store)
    values = ValueIndex(store)
    assert values.has("orders", "status") and not values.has("orders", "customer_ref")  # the premises
    assert not values.has("orders", "batch_ref")
    return adapter, snap, values


class _Recording:
    """The adapter, recording each statement it runs; `fail` names the probes to fail, `tick` a fake clock to move."""

    def __init__(self, inner, fail=lambda sql: False, tick=None, virtual=None):
        self.inner, self.fail, self.tick, self.virtual, self.statements = inner, fail, tick, virtual, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def virtual_columns(self):
        if self.virtual is None:
            return []
        return self.virtual()

    def execute_arrow(self, sql, timeout_s=None):
        self.statements.append(sql)
        if sql.startswith("SELECT 1 FROM "):
            if self.tick:
                self.tick()
            if self.fail(sql):
                raise TimeoutError("the probe ran past its bound")
        return self.inner.execute_arrow(sql, timeout_s=timeout_s)

    def probes(self):
        return [s for s in self.statements if s.startswith("SELECT 1 FROM ")]

    def located(self):
        return [s for s in self.probes() if '"customer_ref"' not in s]


def _ask(adapter, snap, values, sqls, grants=_GRANTS, cards=_CARDS, **mode):
    verifiers, _ = _build_mode_verifiers(Settings(), client=None)
    generator = FakeGenerator([json.dumps({"sql": s}) for s in sqls])
    agent = build_agent(replace(MODES["instant"], **mode), generator=generator,
                        synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter, cache=TwoTierCache(L1Cache()),
                        corrector=None, values=values, selector=None, verifier=verifiers["instant"])
    packet = ContextPacket(question="Which orders are in batch B-0017?",
                           cards=[RetrievedCard(object_id=t, card=f"TABLE {t}", score=1.0) for t in cards],
                           grant_fingerprint=grants.fingerprint, enrichment_version="v1")
    return agent.answer(packet, snap, grants, _IDENTITY), generator


def test_a_missing_value_found_in_other_columns_is_named(world):
    adapter, snap, values = world
    recording = _Recording(adapter)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    assert ans.deferred is True and ans.answer == NAMED, ans.answer
    # `status` is indexed: answered from the index, never asked of the source.
    assert recording.located() and not any('"status"' in s for s in recording.located())


def test_the_retry_feedback_carries_the_columns(world):
    adapter, snap, values = world
    ans, generator = _ask(_Recording(adapter), snap, values, [MISSING_SQL, FOUND_SQL], repair_hints=True,
                          empty_retries=1)
    assert "though it does appear in orders.batch_ref" in generator.calls[1], generator.calls[1]
    assert ans.deferred is False, ans.answer


def test_no_column_the_caller_may_not_read_plainly_is_probed_or_named(world):
    """Masked (`card_hint`), denied (`audit_ref`), a personal name (`contact_name`), a row-filtered table (`returns`), a
    view over it (`returns_v`) and an ungranted table (`ledger`) all hold the value; none is asked or named, while the
    plain columns beside them are."""
    adapter, snap, values = world
    recording = _Recording(adapter)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    asked = " ".join(recording.located())
    assert '"batch_ref"' in asked  # the positive control: locating ran
    for name in ('"card_hint"', '"audit_ref"', '"contact_name"', '"returns"', '"returns_v"', '"ledger"'):
        assert name not in asked, name
    for name in ("card_hint", "audit_ref", "contact_name", "returns", "ledger"):
        assert name not in ans.answer, name


def test_a_locating_probe_that_fails_leaves_todays_decline(world):
    adapter, snap, values = world
    recording = _Recording(adapter, fail=lambda sql: '"customer_ref"' not in sql)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    assert recording.located() and ans.answer == TODAY, ans.answer


def test_locating_stops_at_its_limit(world, monkeypatch):
    adapter, snap, values = world
    monkeypatch.setattr(loop_mod, "ELSEWHERE_LIMIT", 2)
    recording = _Recording(adapter)
    _ask(recording, snap, values, [NOWHERE_SQL], repair_hints=True)
    assert len(recording.located()) == 2


def test_no_locating_probe_once_the_turns_budget_is_spent(world, monkeypatch):
    adapter, snap, values = world
    monkeypatch.setattr(loop_mod, "PROBE_BUDGET_S", 0.0)
    recording = _Recording(adapter)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    assert recording.probes() == [] and "though it does appear" not in ans.answer


def test_a_slow_source_overruns_the_budget_by_one_probe_at_most(world, monkeypatch):
    """No absolute deadline (the plan's ruling 9): the budget is read between probes, so one probe can overrun it, and
    no probe starts once it is spent. A fake clock the probes move, four seconds each, against the ten-second budget."""
    adapter, snap, values = world
    clock = [0.0]

    class _Time:
        @staticmethod
        def monotonic():
            return clock[0]

    monkeypatch.setattr(loop_mod, "time", _Time)
    recording = _Recording(adapter, tick=lambda: clock.__setitem__(0, clock[0] + 4.0))
    _ask(recording, snap, values, [NOWHERE_SQL], repair_hints=True)
    assert len(recording.probes()) == 3  # at 0 s (its own column), 4 s and 8 s; none at 12 s
    assert clock[0] <= loop_mod.PROBE_BUDGET_S + 4.0


def test_a_source_bound_table_is_decided_by_its_snapshot_name_and_probed_by_its_source_name(world, monkeypatch):
    adapter, snap, values = world
    renamed = [c.model_copy(update={"object_id": "shipped", "id": f"shipped.{c.name}"})
               if c.object_id == "shipments" else c for c in snap.columns]
    binding = SourceBinding(id="b1", source_id="src", object_id="shipped", source_object="shipments",
                            binding_type="table")
    bound = snap.model_copy(update={"columns": renamed, "source_bindings": [*snap.source_bindings, binding]})
    decided = []
    real = loop_mod.decide

    def spy(sql, *args, **kwargs):
        decided.append(sql)
        return real(sql, *args, **kwargs)

    monkeypatch.setattr(loop_mod, "decide", spy)
    grants = GrantSet(frozenset({"orders", "shipped"}))
    recording = _Recording(adapter)
    ans, _ = _ask(recording, bound, values, [MISSING_SQL], grants=grants, cards=("orders", "shipped"),
                  repair_hints=True)
    assert any('"shipped"' in s for s in decided) and not any('"shipped"' in s for s in recording.probes())
    assert any('FROM "shipments" WHERE "batch_ref"' in s for s in recording.probes())
    assert "shipped.batch_ref" in ans.answer, ans.answer


def _forcing(monkeypatch, extra):
    """`locatable_columns` returning `extra` beside what it returns, as a gap in it would."""
    real = loop_mod.locatable_columns
    monkeypatch.setattr(loop_mod, "locatable_columns", lambda *a, **k: (*real(*a, **k), *extra))


def test_the_probe_decide_refuses_a_view_over_a_filtered_table_on_its_own(world, monkeypatch):
    adapter, snap, values = world
    for forced in ("returns_v", "returns"):  # each reached as a candidate before the third column is found
        with monkeypatch.context() as m:
            _forcing(m, [(forced, "batch_ref")])
            recording = _Recording(adapter)
            _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
            asked = " ".join(recording.located())
            assert '"orders" WHERE "batch_ref"' in asked and f'"{forced}"' not in asked, forced


def test_a_probe_the_decider_refuses_never_runs(world, monkeypatch):
    adapter, snap, values = world
    _forcing(monkeypatch, [("ledger", "batch_ref")])  # not granted: the decider has no such table
    recording = _Recording(adapter)
    _ask(recording, snap, values, [MISSING_SQL], cards=("orders", "ledger"), repair_hints=True)
    assert recording.located() and not any('"ledger"' in s for s in recording.located())


def test_the_probe_decide_folds_names_as_the_source_does(world):
    """An upper-case stored name, `LOTS`, decided under the source's own target: approved and probed."""
    adapter, snap, values = world
    recording = _Recording(adapter)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    assert any('FROM "LOTS" WHERE "batch_ref"' in s for s in recording.located())
    assert "LOTS.batch_ref" in ans.answer


def test_a_catalogue_that_cannot_say_which_columns_are_computed_stops_locating(world):
    """The decider asks the catalogue first, and a catalogue down then refuses the statement itself; here it answers the
    decider and is down by the time the turn looks for the value (`Reach` is built only when a value is missing)."""
    adapter, snap, values = world
    calls = []

    def down():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("catalogue unavailable")
        return []

    recording = _Recording(adapter, virtual=down)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    assert len(calls) == 2 and recording.located() == [] and ans.answer == TODAY, ans.answer


def test_a_computed_column_is_never_probed(world):
    adapter, snap, values = world
    recording = _Recording(adapter, virtual=lambda: [("orders", "batch_ref", "upper(customer_ref)")])
    ans, _ = _ask(recording, snap, values, [MISSING_SQL], repair_hints=True)
    assert not any('"orders" WHERE "batch_ref"' in s for s in recording.located())
    assert "orders.batch_ref" not in ans.answer and "shipments.batch_ref" in ans.answer


def test_a_snapshot_whose_view_discovery_failed_runs_no_locating_probe(world):
    """The gate's deferred advisory: the probes' views are `inventory_for(snapshot)`, so an unavailable inventory
    still stops them, as it stops a statement over a filtered table."""
    adapter, snap, values = world
    failed = snap.model_copy(update={"jobs": [*snap.jobs, Job(id="discover:views", source_id="src",
                                                                kind="discover", status="failed")]})
    recording = _Recording(adapter)
    _ask(recording, failed, values, [MISSING_SQL], repair_hints=True)
    assert recording.located() == []


def test_off_asks_the_source_nothing_more(world):
    adapter, snap, values = world
    recording = _Recording(adapter)
    ans, _ = _ask(recording, snap, values, [MISSING_SQL])
    assert recording.probes() == ['SELECT 1 FROM "orders" WHERE "customer_ref" = \'B-0017\' LIMIT 1']
    assert ans.answer == TODAY, ans.answer

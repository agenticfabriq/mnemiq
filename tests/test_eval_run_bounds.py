"""The ACME eval bounds the engine's and the gold's queries on the server, as a served deployment does since M152, and not
the profiling.

`run_acme` profiles the source, runs the engine, and executes the gold, all over Postgres through DuckDB. `run_case` asks
the candidate and the gold for 30 s each, so a server bound at DEFAULT_TIMEOUT_S enforces what it already asks. Profiling
has no such limit and keeps an unbounded adapter, and so does the value index `build_engine` builds: its per-column
`SELECT DISTINCT` carries no limit, and a cancelled one is skipped silently (fail-soft), losing that column's value check.
The engine and the gold must share ONE adapter: `run_case` probes the candidate on the gold's adapter whenever the two
are distinct objects, re-running every candidate.
"""
import inspect
from types import SimpleNamespace

import pytest

import mnemiq.enrichment.certified as certified
import mnemiq.eval.engine as engine
import mnemiq.enrichment.grounding as grounding
import mnemiq.eval.run as run
import mnemiq.eval.trend as trend
from mnemiq.agent.loop import Agent
from mnemiq.execute.runner import DEFAULT_TIMEOUT_S


class Adapter:
    def __init__(self, dsn, *, local_only=False, statement_timeout_s=None):
        self.dsn, self.statement_timeout_s = dsn, statement_timeout_s


def test_the_engine_and_gold_share_one_server_bounded_adapter_and_the_profiling_keeps_an_unbounded_one(monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "DuckDBPostgresAdapter", Adapter)
    monkeypatch.setattr(run, "enrich_structural", lambda adapter, source: seen.setdefault("profile", adapter) and "snap")
    monkeypatch.setattr(grounding, "ground_codes", lambda adapter, snap, **kw: seen.setdefault("codes", adapter) and snap)
    monkeypatch.setattr(certified, "fetch_certified_records", lambda settings: None)
    monkeypatch.setattr(certified, "require_certified", lambda records, settings: None)
    monkeypatch.setattr(certified, "apply_certified_set", lambda snap, records: (snap, set()))
    monkeypatch.setattr(run, "LLMClient", lambda settings: None)
    monkeypatch.setattr(run, "LLMEnricher", lambda client: None)
    monkeypatch.setattr(run, "enrich_semantic", lambda snap, enricher, protected: snap)
    client = SimpleNamespace(total_tokens=0, calls=0)
    def build_engine(snap, adapter, settings, index_adapter=None):
        seen["engine"], seen["index"] = adapter, index_adapter if index_adapter is not None else adapter
        return adapter, client
    monkeypatch.setattr(run, "build_engine", build_engine)
    monkeypatch.setattr(run, "load_cases", lambda path: ["case"])

    def run_case(case, engine, adapter, gold_adapter=None):
        seen["executes"], seen["gold"] = adapter, gold_adapter if gold_adapter is not None else adapter
    monkeypatch.setattr(run, "run_case", run_case)
    monkeypatch.setattr(run, "summarize", lambda results, **kw: SimpleNamespace(render=lambda: ""))
    monkeypatch.setattr(trend, "last_run", lambda *a, **kw: None)
    settings = SimpleNamespace(pg_dsn="postgresql://x/y", local_only=False, source_id="acme", dictionary_path=None,
                               ontology_records_path=None, control_dsn=None)

    assert run.run_acme(settings) == 0
    # One object for the engine, its re-execution and the gold, so run_case's cross-engine probe stays off.
    assert seen["engine"] is seen["executes"] is seen["gold"]
    assert seen["engine"].statement_timeout_s == DEFAULT_TIMEOUT_S
    assert seen["profile"] is seen["codes"] is seen["index"] and seen["profile"] is not seen["engine"]
    assert seen["profile"].statement_timeout_s is None


class Built(Exception):
    """Stops build_engine once the value index has been built: the rest needs a model."""


def test_build_engine_builds_the_value_index_on_the_index_adapter_and_on_its_own_adapter_by_default(monkeypatch):
    built = []
    monkeypatch.setattr(engine, "LLMEmbedder", lambda settings: None)
    monkeypatch.setattr(engine, "init_store", lambda path, local_only=False: None)
    monkeypatch.setattr(engine, "build_index", lambda con, snapshot, embedder: None)
    monkeypatch.setattr(engine, "build_example_index", lambda con, snapshot, embedder: None)

    def build_value_index(adapter, snapshot, con):
        built.append(adapter)
        raise Built
    monkeypatch.setattr(engine, "build_value_index", build_value_index)
    settings = SimpleNamespace(local_only=False)
    bounded, unbounded = Adapter("pg", statement_timeout_s=30.0), Adapter("pg")

    with pytest.raises(Built):
        engine.build_engine(None, bounded, settings, index_adapter=unbounded)
    with pytest.raises(Built):
        engine.build_engine(None, bounded, settings)
    assert built == [unbounded, bounded]


def test_the_agent_s_execution_timeout_is_the_runner_s_constant_not_a_copy_of_its_value():
    # The server bound (M152) is DEFAULT_TIMEOUT_S; the agent's default read a 30.0 literal that agreed with it only by
    # coincidence. Identity, not equality: equal values pass today and would not catch the two drifting apart.
    assert inspect.signature(Agent.__init__).parameters["timeout_s"].default is DEFAULT_TIMEOUT_S

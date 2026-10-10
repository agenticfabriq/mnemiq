"""The eval's `ask` takes grants per question, the way `Runtime.ask` resolves them per identity.

An engine is built once per database, so an arm that narrows each question to a few tables has to
hand the narrowing to `ask`. It must reach all three places the product door hands its grants --
retrieval, the prompt fit and the agent -- or the arm measures a mix of two scopes.
"""
from types import SimpleNamespace

import mnemiq.eval.engine as engine
from mnemiq.agent.loop import AgentAnswer
from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import Snapshot


def _engine(monkeypatch, tmp_path, grants):
    seen: dict[str, list] = {"retrieve": [], "fit": [], "answer": []}
    monkeypatch.setattr(engine, "LLMEmbedder", lambda settings: None)
    # The engine reads its default table list from the index it built.
    monkeypatch.setattr(engine, "build_index", lambda con, *a, **k: con.execute(
        "CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT)"))
    monkeypatch.setattr(engine, "build_example_index", lambda *a, **k: None)
    monkeypatch.setattr(engine, "build_value_index", lambda *a, **k: 0)

    def retrieve(con, question, identity, authz, embedder, **kw):
        seen["retrieve"].append(authz.grants_for(identity).objects)
        return SimpleNamespace()
    monkeypatch.setattr(engine, "retrieve", retrieve)

    class Fitter:
        def fit(self, packet, snapshot, grants):
            seen["fit"].append(grants.objects)
            return packet

    monkeypatch.setattr(engine, "build_components", lambda settings, adapter, con: SimpleNamespace(
        fitter=Fitter(), client=None, generator=None, synthesizer=None, corrector=None,
        values=None, selector=None))

    class Agent:
        def __init__(self, **kw):
            pass

        def answer(self, packet, snapshot, grants, identity):
            seen["answer"].append(grants.objects)
            return AgentAnswer(answer="A")
    monkeypatch.setattr(engine, "Agent", Agent)

    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x",
                        acme_data_dir=None, store_path=str(tmp_path / "s.duckdb"))
    snap = Snapshot(version="v", source_id="s", created_at="t")
    ask, _ = engine.build_engine(snap, adapter=None, settings=settings, grants=grants)
    return ask, seen


def test_a_question_asked_with_grants_is_scoped_to_them_everywhere(monkeypatch, tmp_path):
    ask, seen = _engine(monkeypatch, tmp_path, GrantSet(frozenset({"a", "b", "c"})))
    ask("q", grants=GrantSet(frozenset({"a"})))
    assert seen == {"retrieve": [{"a"}], "fit": [{"a"}], "answer": [{"a"}]}


def test_without_grants_a_question_keeps_the_engines_and_a_narrowing_does_not_stick(monkeypatch, tmp_path):
    ask, seen = _engine(monkeypatch, tmp_path, GrantSet(frozenset({"a", "b"})))
    ask("q")
    ask("q", grants=GrantSet(frozenset({"b"})))
    ask("q")
    assert seen["retrieve"] == seen["fit"] == seen["answer"] == [{"a", "b"}, {"b"}, {"a", "b"}]

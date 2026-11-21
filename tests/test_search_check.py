"""`mnemiq check-search`: does table search find the tables each question needs, and does the prompt fit?

At a few tables every table reaches the prompt; at hundreds the search picks `retrieval_k` of them, and a question whose
tables it misses cannot be answered whatever the model does. The check asks each question as `ask` would up to the
model call (`Runtime.search`, the same retrieval and fit), takes the tables the question's gold SQL reads (the
engine's own reading), and reports where each was found and whether the prompt fits the window. No model is asked. Its
output holds counts and question ids only, unless names are asked for, so a deployment can share it.
"""
from __future__ import annotations

import json
from types import SimpleNamespace


from mnemiq.eval.search_check import check, gold_tables, render
from mnemiq.semantic.fit import PromptFit


def test_gold_tables_are_the_engines_reading():
    assert gold_tables('SELECT o.id FROM orders o JOIN "Lines" l ON l.order_id = o.id', "duckdb") == {"orders", "Lines"}
    assert gold_tables("WITH x AS (SELECT * FROM orders) SELECT * FROM x", "duckdb") == {"orders"}
    assert gold_tables("SELEC nothing FROM", "duckdb") is None


class _Runtime:
    """Answers `search` with a fixed ranking, as Runtime.search returns one."""

    def __init__(self, tables, ranked, fits=True):
        self.snapshot = SimpleNamespace(source_bindings=[SimpleNamespace(object_id=t) for t in tables])
        self._ranked, self._fits = ranked, fits
        self.adapter = SimpleNamespace(dialect="duckdb")

    def search(self, question, identity):
        return SimpleNamespace(tables=self._ranked[question],
                               fit=PromptFit(tokens=900, budget=1000 if self._fits else 800, reserve=100,
                                             counted_by="server", fits=self._fits))


_QUESTIONS = [
    {"id": "q1", "question": "orders by region", "gold_sql": "SELECT region, count(*) FROM orders GROUP BY region"},
    {"id": "q2", "question": "lines per order",
     "gold_sql": "SELECT o.id, count(*) FROM orders o JOIN lines l ON l.order_id = o.id GROUP BY o.id"},
    {"id": "q3", "question": "broken", "gold_sql": "SELEC FROM"},
    {"id": "q4", "question": "a view", "gold_sql": "SELECT * FROM monthly_summary"},
]


def test_the_check_counts_gold_tables_found_and_their_ranks():
    rt = _Runtime(["orders", "lines", "customers"],
                  {"orders by region": ["customers", "ORDERS"], "lines per order": ["orders"],
                   "broken": [], "a view": ["orders"]})
    rows, summary = check(rt, _QUESTIONS, identity=None)
    by = {r["id"]: r for r in rows}
    assert by["q1"]["found"] == 1 and by["q1"]["gold"] == 1 and by["q1"]["ranks"] == [2]
    assert by["q2"]["found"] == 1 and by["q2"]["gold"] == 2 and by["q2"]["ranks"] == [1, None]
    assert summary["questions"] == 4 and summary["checked"] == 2 and summary["all_found"] == 1


def test_unreadable_gold_is_reported_apart():
    rt = _Runtime(["orders", "lines"], {q["question"]: ["orders"] for q in _QUESTIONS})
    rows, summary = check(rt, _QUESTIONS, identity=None)
    by = {r["id"]: r for r in rows}
    assert by["q3"]["unreadable"] == "the gold SQL does not parse"
    assert by["q4"]["unreadable"] == "the gold SQL reads 1 table(s) outside this deployment"
    assert summary["unreadable"] == 2 and summary["checked"] == 2, "neither counts as a search miss"


def test_each_question_says_whether_its_prompt_fits():
    rt = _Runtime(["orders", "lines"], {q["question"]: ["orders", "lines"] for q in _QUESTIONS}, fits=False)
    rows, summary = check(rt, _QUESTIONS, identity=None)
    assert all(r["fits"] is False for r in rows if not r.get("unreadable"))
    assert summary["fits"] == 0 and summary["over_window"] == 2


def test_output_holds_no_name_without_names():
    rt = _Runtime(["orders", "lines"], {q["question"]: ["orders"] for q in _QUESTIONS})
    rows, summary = check(rt, _QUESTIONS, identity=None)
    text = render(rows, summary, names=False)
    assert "orders" not in text and "lines" not in text and "region" not in text
    assert "q1" in text and "found 1 of 1" in text
    assert "orders" in render(rows, summary, names=True)


def test_ask_and_search_build_the_same_packet(monkeypatch, tmp_path):
    # The two doors: the check scores the packet answering would read, or it measures a prompt `ask` cannot send.
    from mnemiq import runtime as runtime_module
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, IdentityContext, Snapshot
    from mnemiq.llm.embeddings import FakeEmbedder
    from mnemiq.runtime import Runtime
    from mnemiq.semantic.store import build_index
    from mnemiq.store.bootstrap import init_store

    snapshot = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="orders.id", object_id="orders", name="id"), Column(id="lines.id", object_id="lines", name="id")])

    class Authz:
        def grants_for(self, _identity):
            return GrantSet(frozenset({"orders", "lines"}))

    class Agent:
        def decline_if_too_long(self, question):
            return None

        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer="ok")

    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, snapshot, embedder)
    rt = Runtime(con=con, snapshot=snapshot, adapter=None, agent=Agent(), embedder=embedder, authz=Authz(),
                 settings=None)
    calls = []
    real = runtime_module.retrieve

    def recording(*args, **kwargs):
        packet = real(*args, **kwargs)
        calls.append(([c.object_id for c in packet.cards], args[1:3], sorted(kwargs)))
        return packet

    monkeypatch.setattr(runtime_module, "retrieve", recording)
    identity = IdentityContext(tenant_id="t", principal_id="p", roles=["analyst"])
    found = rt.search("how many orders", identity)
    rt.ask("how many orders", identity)
    assert len(calls) == 2 and calls[0] == calls[1], "search retrieves exactly as ask does"
    assert found.tables == calls[0][0]


def test_the_command_prints_counts(monkeypatch, tmp_path, capsys):
    from mnemiq.cli import main

    questions = tmp_path / "q.json"
    questions.write_text(json.dumps(_QUESTIONS))
    rt = _Runtime(["orders", "lines"], {q["question"]: ["orders", "lines"] for q in _QUESTIONS})
    monkeypatch.setattr("mnemiq.cli.build_runtime", lambda settings: rt)
    assert main(["check-search", "--questions", str(questions)]) == 0
    out = capsys.readouterr().out
    assert "4 questions" in out and "orders" not in out


def test_a_qualified_gold_table_is_matched_to_the_id_the_deployment_holds():
    # Gold SQL may qualify a table by its schema (`SHOP.ORDERS` against a snapshot holding `ORDERS`) or by a federated
    # catalog (`erp.orders` against `erp.orders`); either is the deployment's table, not one outside it.
    questions = [{"id": "q1", "question": "a", "gold_sql": "SELECT * FROM SHOP.ORDERS"},
                 {"id": "q2", "question": "b", "gold_sql": "SELECT * FROM erp.lines"}]
    rt = _Runtime(["ORDERS", "erp.lines"], {"a": ["ORDERS"], "b": ["erp.lines"]})
    rows, summary = check(rt, questions, identity=None)
    assert [r.get("unreadable") for r in rows] == [None, None]
    assert summary["all_found"] == 2

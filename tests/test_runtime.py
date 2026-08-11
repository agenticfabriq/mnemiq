import pytest

from mnemiq.authz.grants import DenyAll, GrantSet
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.runtime import Runtime, SnapshotMissing, build_runtime


def _identity():
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def test_build_runtime_without_a_snapshot_is_a_clear_error(tmp_path):
    # empty store, no snapshot -> actionable error, not a crash (and no LLM client built)
    s = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
        store_path=str(tmp_path / "empty.duckdb"),
    )
    with pytest.raises(SnapshotMissing):
        build_runtime(s)


class _StaticAuthz:
    def __init__(self, *objects):
        self._g = GrantSet(frozenset(objects))

    def grants_for(self, _identity):
        return self._g


class _Con:
    """Minimal store stand-in: schema() reads semantic_object."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        self._last = (sql, params)
        return self

    def fetchall(self):
        return self._rows


def test_schema_returns_only_granted_objects():
    con = _Con([("claim", "TABLE claim ..."), ("party", "TABLE party ...")])
    rt = Runtime(con=con, snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=_StaticAuthz("claim"), settings=None)
    got = rt.schema(_identity())
    assert got == [{"object_id": "claim", "card": "TABLE claim ..."}]


def test_schema_is_empty_under_denyall():
    con = _Con([("claim", "c")])
    rt = Runtime(con=con, snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=DenyAll(), settings=None)
    assert rt.schema(_identity()) == []


def test_ask_retrieves_scoped_and_delegates_to_the_agent(monkeypatch):
    # ask wires retrieve -> agent.answer; verify with fakes, no store/LLM
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.contract import Snapshot
    from mnemiq.semantic.retrieval import ContextPacket

    calls = {}

    def fake_retrieve(con, question, identity, authz, embedder, k=5, table_facts=(), **kwargs):
        calls["retrieve"] = (question, k, list(table_facts))
        return ContextPacket(question=question, cards=[], grant_fingerprint="fp",
                             enrichment_version="v1")

    class _Agent:
        def answer(self, packet, snapshot, grants, identity, emit=None):
            calls["answer"] = (packet, snapshot, grants.objects)
            return AgentAnswer(answer="ANSWER")

    snap = Snapshot(version="v1", source_id="acme", created_at="t")
    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)
    rt = Runtime(con=None, snapshot=snap, adapter=None, agent=_Agent(), embedder=None,
                 authz=_StaticAuthz("claim"), settings=None)
    got = rt.ask("how many claims?", _identity())
    assert got.answer == "ANSWER"
    assert got.mode == "thinking"  # the resolved default, stamped by Runtime
    assert calls["retrieve"] == ("how many claims?", 24, [])  # k=24 default; facts threaded in
    assert calls["answer"][0].question == "how many claims?"
    assert calls["answer"][1] is snap


def test_ask_dispatches_to_the_mode_agent_and_stamps_the_mode(monkeypatch):
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer

    from mnemiq.semantic.retrieval import ContextPacket

    monkeypatch.setattr(
        rt_mod, "retrieve",
        lambda *a, **k: ContextPacket(question="q", cards=[], grant_fingerprint="fp",
                                      enrichment_version="v1"),
    )

    class _A:
        def __init__(self, tag):
            self.tag = tag

        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer=self.tag)

    agents = {"instant": _A("i"), "thinking": _A("t"), "deep": _A("d")}
    rt = Runtime(con=None, snapshot=None, adapter=None, agent=agents["thinking"],
                 embedder=None, authz=_StaticAuthz("claim"), settings=None, agents=agents)
    deep = rt.ask("q", _identity(), mode="deep")
    assert deep.answer == "d" and deep.mode == "deep"
    default = rt.ask("q", _identity())
    assert default.answer == "t" and default.mode == "thinking"


def test_ask_with_an_unknown_mode_fails_closed_before_any_work(monkeypatch):
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.route import UnknownMode

    def _no_retrieve(*a, **k):
        raise AssertionError("retrieval must not run for an unknown mode")

    monkeypatch.setattr(rt_mod, "retrieve", _no_retrieve)
    rt = Runtime(con=None, snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=DenyAll(), settings=None)
    with pytest.raises(UnknownMode):
        rt.ask("q", _identity(), mode="fastest")


def test_build_runtime_rejects_an_unknown_default_mode(tmp_path):
    from mnemiq.agent.route import UnknownMode

    s = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
        store_path=str(tmp_path / "empty.duckdb"), default_mode="fastest",
    )
    with pytest.raises(UnknownMode):  # validated BEFORE the snapshot check -- boot fails fast
        build_runtime(s)


class _WriteAuthz:
    def __init__(self, *tables):
        self._g = GrantSet(frozenset(tables), writable=frozenset(tables))

    def grants_for(self, _identity):
        return self._g


def test_write_refuses_under_denyall():
    from mnemiq.contract import Snapshot
    from mnemiq.runtime import Runtime

    snap = Snapshot(version="v1", source_id="acme", created_at="t")
    rt = Runtime(con=None, snapshot=snap, adapter=None, agent=None, embedder=None,
                 authz=DenyAll(), settings=None)
    res = rt.write("INSERT INTO claim (id) VALUES (1)", _identity())
    assert res.approved is False and res.refusal


def test_write_executes_on_approval():
    from mnemiq.contract import Column, Snapshot
    from mnemiq.runtime import Runtime

    snap = Snapshot(version="v1", source_id="acme", created_at="t",
                    columns=[Column(id="claim.id", object_id="claim", name="id")])

    class _RWAdapter:
        dialect = "duckdb"

        def __init__(self):
            self.ran = []

        def execute(self, sql):
            self.ran.append(sql)
            return [] if sql.startswith("EXPLAIN") else [(1,)]

    adapter = _RWAdapter()

    class _WritesEnabled:
        # M3: the deployment switch now reaches the decider and defaults CLOSED, so a test that
        # exercises the execution path has to say which deployment it is testing. `settings=None`
        # used to mean "unconfigured", which quietly meant "writes allowed".
        write_enabled = True
        source_id = "acme"

    rt = Runtime(con=None, snapshot=snap, adapter=adapter, agent=None, embedder=None,
                 authz=_WriteAuthz("claim"), settings=_WritesEnabled())  # write grant on claim
    res = rt.write("INSERT INTO claim (id) VALUES (1)", _identity())
    assert res.approved is True and res.target == "claim" and res.rows_affected == 1
    assert any(not s.startswith("EXPLAIN") for s in adapter.ran)  # the write actually ran


def test_ask_threads_ontology_index_columns_and_definitions(monkeypatch):
    """Regression: question-time code resolution and the glossary reached eval's build_engine
    but NOT the product path, so `mnemiq ask` and the MCP server saw neither. The absence of a
    test over Runtime.ask is exactly why that gap survived review."""
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.contract import (
        CodeScheme, Column, Definition, Dimension, MeasureExpr, Metric, Snapshot,
    )

    seen = {}

    def fake_retrieve(con, question, identity, authz, embedder, k=5, table_facts=(),
                      definitions=(), metrics=(), dimensions=(), columns=(), ontology_index=None,
                      snapshot=None):
        seen["definitions"] = list(definitions)
        # Certified metrics and dimensions were the next pair to reach the snapshot and stop
        # there -- five of the fs corpus's records, carrying the certified SQL for settled, gross
        # and net volume, appended by `apply_certified` and read by nothing. Same gap this test
        # was written for, one generation later.
        seen["metrics"] = [m.id for m in metrics]
        seen["dimensions"] = [d.id for d in dimensions]
        seen["columns"] = [c.id for c in columns]
        seen["ontology_index"] = ontology_index
        # M4 added a fourth thing that has to reach the product path: without the snapshot,
        # retrieve cannot re-render a card against the caller's column policy and silently
        # serves the unscoped one. That is the same gap this test was written for.
        seen["snapshot"] = snapshot
        from mnemiq.semantic.retrieval import ContextPacket

        return ContextPacket(question=question, cards=[], grant_fingerprint="f",
                             enrichment_version=None)

    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)

    class _Agent:
        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer="ANSWER")

    snap = Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[Column(id="patient.icd10_cd", object_id="patient", name="icd10_cd",
                        code_scheme=CodeScheme(id="urn:icd10", label="ICD-10-CM"))],
        definitions=[Definition(id="d1", term="ICD-10-CM", domain="ontology",
                                definition="A diagnosis coding system.")],
        metrics=[Metric(id="patient_count", label="Patient Count", status="certified", owner="o",
                        grain="day",
                        measure=MeasureExpr(expr="count(distinct patient_id)", source="patient"),
                        time_dimension="day")],
        dimensions=[Dimension(id="patient.icd10_cd", label="Diagnosis", source="patient")],
    )
    sentinel = object()
    rt = Runtime(con=None, snapshot=snap, adapter=None, agent=_Agent(), embedder=None,
                 authz=_StaticAuthz("patient"), settings=None, ontology=sentinel)
    rt.ask("how many with type 2 diabetes", _identity())

    assert seen["ontology_index"] is sentinel        # the index reaches retrieval
    assert seen["snapshot"] is snap                 # ...and so does the snapshot (M4)
    assert seen["columns"] == ["patient.icd10_cd"]   # bound columns are visible to it
    assert [d.term for d in seen["definitions"]] == ["ICD-10-CM"]  # glossary seam fed
    assert seen["metrics"] == ["patient_count"]      # ...and the certified measures
    assert seen["dimensions"] == ["patient.icd10_cd"]

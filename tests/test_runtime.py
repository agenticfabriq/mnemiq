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

    calls = {}

    def fake_retrieve(con, question, identity, authz, embedder, k=5):
        calls["retrieve"] = (question, k)
        return "PACKET"

    class _Agent:
        def answer(self, packet, snapshot, grants, identity):
            calls["answer"] = (packet, snapshot, grants.objects)
            return AgentAnswer(answer="ANSWER")

    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)
    rt = Runtime(con=None, snapshot="SNAP", adapter=None, agent=_Agent(), embedder=None,
                 authz=_StaticAuthz("claim"), settings=None)
    got = rt.ask("how many claims?", _identity())
    assert got.answer == "ANSWER"
    assert got.mode == "thinking"  # the resolved default, stamped by Runtime
    assert calls["retrieve"] == ("how many claims?", 6)
    assert calls["answer"][0] == "PACKET" and calls["answer"][1] == "SNAP"


def test_ask_dispatches_to_the_mode_agent_and_stamps_the_mode(monkeypatch):
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer

    monkeypatch.setattr(rt_mod, "retrieve", lambda *a, **k: "PACKET")

    class _A:
        def __init__(self, tag):
            self.tag = tag

        def answer(self, packet, snapshot, grants, identity):
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

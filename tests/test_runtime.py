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

    calls = {}

    def fake_retrieve(con, question, identity, authz, embedder, k=5):
        calls["retrieve"] = (question, k)
        return "PACKET"

    class _Agent:
        def answer(self, packet, snapshot, grants, identity):
            calls["answer"] = (packet, snapshot, grants.objects)
            return "ANSWER"

    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)
    rt = Runtime(con=None, snapshot="SNAP", adapter=None, agent=_Agent(), embedder=None,
                 authz=_StaticAuthz("claim"), settings=None)
    assert rt.ask("how many claims?", _identity()) == "ANSWER"
    assert calls["retrieve"] == ("how many claims?", 6)
    assert calls["answer"][0] == "PACKET" and calls["answer"][1] == "SNAP"

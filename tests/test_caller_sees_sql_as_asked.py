"""M51: a caller is shown the SQL as asked; the policy's rewrite of it stays in the audit trace.

Under an entitlements-style row filter -- `id IN (SELECT id FROM entitlement WHERE who = 'u')`, the
standard shape (M28) -- the executed statement names the entitlement table, its columns and the
predicate's literals: exactly what the narrowed `tables` list was built to keep from the caller.
It reached every surface: the HTTP payload, MCP and the CLI. The owner's decision (2026-10-13):
callers get the statement as asked; `narrowed` already says whether the policy changed anything;
the executed statement stays on `Trace.target_sql`, for the audit record and the in-process graders
that re-run it.
"""

from __future__ import annotations

from mnemiq.agent.loop import AgentAnswer
from mnemiq.agent.trace import build_trace
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.mcp.server import _db_read, _db_write
from mnemiq.runtime import Runtime
from mnemiq.server.serialize import answer_payload
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import build_access_policy

_ENTITLED = "id IN (SELECT id FROM entitlement WHERE who = 'u')"
_IDENTITY = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id=f"{o}.{n}", object_id=o, name=n)
        for o, n in [("claim", "id"), ("claim", "region"), ("scratch", "id"),
                     ("entitlement", "id"), ("entitlement", "who")]])


def _grants(filtered: bool = True) -> GrantSet:
    return GrantSet(frozenset({"claim", "scratch"}), writable=frozenset({"scratch"}),
                    row_filters={"claim": _ENTITLED} if filtered else {})


def _read(filtered: bool = True):
    return decide("SELECT id FROM claim", {"claim": {"id", "region"}},
                  policy=build_access_policy(_snapshot(), _grants(filtered)), target="duckdb")


def test_the_read_keeps_the_statement_as_asked_beside_the_one_executed():
    verdict = _read()
    assert "entitlement" in verdict.target_sql  # the control: the policy really applied
    assert "entitlement" not in verdict.asked_sql
    assert verdict.asked_sql == "SELECT id FROM claim LIMIT 1000"
    assert verdict.narrowed  # and the answer says so, by table


def test_with_nothing_to_rewrite_the_two_are_one_statement():
    verdict = _read(filtered=False)
    assert verdict.asked_sql == verdict.target_sql
    assert verdict.narrowed == []


def test_the_write_keeps_its_statement_as_asked_too():
    grants = _grants()
    verdict = decide_write("INSERT INTO scratch (id) SELECT id FROM claim",
                           {"claim": {"id", "region"}, "scratch": {"id"}}, grants,
                           policy=build_access_policy(_snapshot(), grants), dialect="duckdb",
                           writes_enabled=True)
    assert "entitlement" in verdict.target_sql
    assert "entitlement" not in verdict.asked_sql


def _answer() -> AgentAnswer:
    trace = build_trace("how many claims", _read(), _IDENTITY, "v1", {}, "1x1", [])
    return AgentAnswer(answer="2.", trace=trace)


def test_the_http_payload_shows_the_statement_as_asked():
    """The join: decider -> trace builder -> serializer, as `/v1/ask` runs them."""
    payload = answer_payload(_answer())
    assert payload["sql"] == "SELECT id FROM claim LIMIT 1000"
    assert payload["narrowed"]


def test_the_mcp_read_shows_the_statement_as_asked():
    class _Runtime:
        def ask(self, question, identity, mode=None):
            return _answer()

    assert "entitlement" not in _db_read(_Runtime(), _IDENTITY, "how many claims")["sql"]


def test_a_governed_write_shows_the_statement_as_asked_through_runtime_and_mcp():
    class _Authz:
        def grants_for(self, _identity):
            return _grants()

    class _Adapter:
        dialect = "duckdb"

        def execute(self, sql):
            return [] if sql.startswith("EXPLAIN") else [(1,)]

    class _WritesEnabled:
        write_enabled = True
        source_id = "s"

    runtime = Runtime(con=None, snapshot=_snapshot(), adapter=_Adapter(), agent=None,
                      embedder=None, authz=_Authz(), settings=_WritesEnabled())
    result = runtime.write("INSERT INTO scratch (id) SELECT id FROM claim", _IDENTITY)
    assert result.approved, result.refusal
    assert "entitlement" in result.target_sql
    assert "entitlement" not in result.asked_sql
    sql = _db_write(runtime, _IDENTITY, "INSERT INTO scratch (id) SELECT id FROM claim")["sql"]
    assert sql and "entitlement" not in sql


def test_the_cli_write_json_shows_the_statement_as_asked(monkeypatch, capsys):
    """Codex's review of #122: `mnemiq write --json` dumped `vars(res)`, so `plan_sql` and
    `target_sql` -- the write as executed, carrying the entitlement subquery -- reached the
    caller beside the safe field. Driven through the real Runtime.write, for an approved write and
    for one the source rejects (which also carries the executed statement)."""
    import json

    import mnemiq.cli as cli
    from mnemiq.config import Settings

    class _Authz:
        def grants_for(self, _identity):
            return _grants()

    def runtime(rejects: bool):
        class _Adapter:
            dialect = "duckdb"

            def execute(self, sql):
                if sql.startswith("EXPLAIN"):
                    return []
                if rejects:
                    raise RuntimeError("permission denied")
                return [(1,)]

        class _WritesEnabled:
            write_enabled = True
            source_id = "s"

        return Runtime(con=None, snapshot=_snapshot(), adapter=_Adapter(), agent=None,
                       embedder=None, authz=_Authz(), settings=_WritesEnabled())

    settings = Settings(llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d")
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    for rejects in (False, True):
        monkeypatch.setattr(cli, "build_runtime", lambda s, r=rejects: runtime(r))
        assert cli.main(["write", "INSERT INTO scratch (id) SELECT id FROM claim", "--json"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "entitlement" not in out, (rejects, out)
        assert payload["sql"] and payload["approved"] is not rejects
        # `narrowed` structured, which the old dump existed to fix, and still the flag.
        assert payload["narrowed"] == [{"object": "claim", "rows": True, "columns": False}]

"""Superset mode at the runtime: no source connection of its own at boot, no result cache, no `deep` mode, and each
request's statements run through an adapter built for that request from the token it carries.

The deployment is the scripted one (`tests/scripted_deployment.py`): a real store, a stub model answering the writer
with fixed SQL, and retrieval returning the one table. Superset is `FakeSuperset`, a real MCP server on loopback, so
every statement crosses the wire, and the bearer each one carried is the one the fake recorded.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

import mnemiq.adapters.resolve as resolve_module
import mnemiq.runtime as runtime_module
from mnemiq.agent.loop import Agent
from mnemiq.agent.route import UnknownMode
from mnemiq.cache.store import NullCache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot
from mnemiq.contract.semantic import Job, SourceBinding
from mnemiq.runtime import SnapshotMissing, build_runtime
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot
from mnemiq.superset.client import Bearer, DelegatedToken, SupersetMcp
from mnemiq.superset.executor import NoSourceConnection, SupersetCredentialMissing, superset_executor
from mnemiq.superset.settings import SUPERSET_DIALECT, SUPERSET_SOURCE_JOB
from tests.scripted_deployment import IDENTITY, SQL, deployment, retrieve_the_one_table
from tests.superset_fake import FakeSuperset
from tests.sweep.stub_model import ScriptedCase, StubModel

DATABASE_ID = 7
SCHEMA = "public"
SOURCE_ID = "only"
TOKEN = "tok-1"
# What `execute_sql` answers a dry run it accepts.
DRY_RUN_OK = {"success": True, "rows": None, "columns": None, "row_count": None, "affected_rows": None, "error": None,
              "error_type": None, "statements": []}
# The proof seam's probe: the statement as a subquery that returns no rows (`SupersetSqlAdapter.validate`).
_PROBE = re.compile(r"AS mnemiq_probe WHERE 1 = 0$")
# The answer statement's alias, which names the person asking in the concurrent test.
_ALIAS = re.compile(r"COUNT\(\*\) AS (\w+) FROM wide", re.IGNORECASE)

PROFILES = """
default = "a"

[profiles.a]
llm_model = "model-a"

[profiles.b]
llm_model = "model-b"
roles = ["*"]
"""


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


def _answer_rows(request: dict, authorization: str | None) -> dict:
    """`execute_sql` as Superset answers it for a person who may read `wide`: a dry run accepted, an empty catalogue
    (every catalogue statement sends `force_refresh`), the probe with no rows, and the answer statement's one row."""
    if request.get("dry_run"):
        return DRY_RUN_OK
    if request.get("force_refresh"):
        return {"success": True, "columns": [], "rows": []}
    alias = (match.group(1) if (match := _ALIAS.search(request["sql"])) else "n")
    columns = [{"name": alias, "type": "int64"}]
    rows = [] if _PROBE.search(request["sql"]) else [{alias: 1}]
    return {"success": True, "columns": columns, "rows": rows}


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        served.reply("execute_sql", _answer_rows)
        yield served


@pytest.fixture
def model():
    """The scripted model, answering the writer with SQL over the one table."""
    with StubModel(sql=SQL) as stub:
        yield stub
    assert not stub.all_errors()


def _source_job(*, status: str = "done", detail: str | None = None, **fields) -> Job:
    """The job a snapshot enriched through Superset carries, naming the database and schema it was read from."""
    values = {"database_id": DATABASE_ID, "schema": SCHEMA, "backend": "postgresql", "values": "not harvested"}
    values.update(fields)
    return Job(id=SUPERSET_SOURCE_JOB, source_id=SOURCE_ID, kind="superset", status=status,
               detail=json.dumps(values) if detail is None else detail)


def _snapshot(jobs: list[Job], version: str = "v1") -> Snapshot:
    return Snapshot(version=version, source_id=SOURCE_ID, created_at="t",
                    columns=[Column(id="wide.col_0", object_id="wide", name="col_0", data_type="VARCHAR")],
                    source_bindings=[SourceBinding(id="sb:wide", source_id=SOURCE_ID, object_id="wide",
                                                   source_object="wide", binding_type="table")],
                    jobs=jobs)


def superset_deployment(tmp_path, model_url: str, superset_url: str, *, jobs: list[Job] | None = None,
                        snapshot: bool = True, profiles: str | None = None, **settings) -> Settings:
    """Settings for a Superset-mode deployment over a store whose snapshot carries `jobs` (by default the Superset
    source job, as enrichment through Superset writes it), with `profiles` as its profiles file when given."""
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    if snapshot:
        save_snapshot(con, _snapshot([_source_job()] if jobs is None else jobs))
    con.close()
    if profiles is not None:
        (tmp_path / "profiles.toml").write_text(profiles)
        settings["profiles_path"] = str(tmp_path / "profiles.toml")
    return Settings(acme_data_dir=None, llm_base_url=model_url, llm_api_key="stub-key", llm_model="stub-model",
                    store_path=str(store), source_id=SOURCE_ID, answer_log_path="off",
                    llm_window_check=settings.pop("llm_window_check", False), superset_mcp_url=superset_url,
                    superset_database_id=DATABASE_ID, superset_schema=SCHEMA, **settings)


@pytest.fixture
def no_direct_source(monkeypatch):
    """`source_adapter` and `adapter_for` fail the test if reached."""
    def reached(*args, **kwargs):
        raise AssertionError("Superset mode reached a direct source connection")

    monkeypatch.setattr(runtime_module, "source_adapter", reached)
    monkeypatch.setattr(resolve_module, "adapter_for", reached)


@pytest.fixture
def runtime(tmp_path, model, fake, monkeypatch, no_direct_source):
    retrieve_the_one_table(monkeypatch)
    return build_runtime(superset_deployment(tmp_path, model.url, fake.url))


def _answer_statements(fake: FakeSuperset) -> list:
    """The calls that ran an answer's statement itself: not a dry run, a catalogue read or the proof's probe."""
    return [call for call in fake.calls if call.tool == "execute_sql" and not call.request.get("dry_run")
            and not call.request.get("force_refresh") and not _PROBE.search(call.request["sql"])]


# --- boot -------------------------------------------------------------------------------------------------------------


def test_boot_opens_no_source(tmp_path, model, fake, monkeypatch, no_direct_source, caplog):
    monkeypatch.setenv("MNEMIQ_PG_DSN", "postgresql://reader@127.0.0.1:5432/warehouse")
    settings = superset_deployment(tmp_path, model.url, fake.url)
    assert settings.pg_dsn  # the direct source is configured, and still never connected to
    with caplog.at_level(logging.INFO, logger="mnemiq.runtime"):
        runtime = build_runtime(settings)
    assert isinstance(runtime.adapter, NoSourceConnection) and runtime.adapter.dialect == SUPERSET_DIALECT
    assert fake.calls == []  # booting asks Superset nothing either
    messages = [record.getMessage() for record in caplog.records]
    assert any("Superset mode reads the function inventory per request" in m for m in messages)
    # The advisories that read the source are skipped, not run against the sentinel.
    assert not any("function inventory" in m and "UNAVAILABLE" in m for m in messages)
    assert not any(m.startswith("could not assess") for m in messages)


def test_the_sentinel_refuses_every_source_method():
    sentinel = NoSourceConnection(SUPERSET_DIALECT)
    calls = {"introspect": (), "list_columns": (), "foreign_keys": (), "view_definitions": (), "user_functions": (),
             "builtin_functions": (), "reachable_user_functions": (), "virtual_columns": (),
             "execute": ("SELECT 1",), "execute_arrow": ("SELECT 1",)}
    for method, args in calls.items():
        with pytest.raises(RuntimeError, match="^Superset mode opens no source connection of its own$"):
            getattr(sentinel, method)(*args)


def test_the_prompt_window_check_runs_on_the_sentinel_and_executes_nothing(tmp_path, model, fake, monkeypatch,
                                                                         no_direct_source, caplog):
    """The window check runs at boot as it does in direct mode. The sentinel raises on any statement and the check
    turns any exception into "the measurement failed", so a check that ran to its verdict executed nothing."""
    authz = tmp_path / "authz.json"
    authz.write_text(json.dumps({"roles": {"analyst": ["wide"]}}))
    settings = superset_deployment(tmp_path, model.url, fake.url, authz_path=str(authz), llm_window_check=True)
    with caplog.at_level(logging.INFO, logger="mnemiq.runtime"):
        build_runtime(settings)
    messages = [record.getMessage() for record in caplog.records]
    assert model.tokenize_calls > 0  # the prompt was counted on the model server: the check ran
    assert any(m.startswith("prompt window fits") for m in messages), messages
    assert not any("measurement failed" in m for m in messages)


@pytest.mark.parametrize("jobs, says", [
    ([], "was not enriched through Superset"),
    ([_source_job(status="failed")], "did not finish"),
    ([_source_job(detail="not json")], "cannot be read"),
    ([_source_job(detail=json.dumps(["database_id", DATABASE_ID]))], "cannot be read"),
    ([_source_job(values="harvested")], "cannot be read"),
    ([_source_job(backend="mysql")], "cannot be read"),
    ([_source_job(database_id=True)], "cannot be read"),
    ([_source_job(extra="field")], "cannot be read"),
    ([_source_job(database_id=DATABASE_ID + 1)], "MNEMIQ_SUPERSET_DATABASE_ID"),
    ([_source_job(schema="sales")], "MNEMIQ_SUPERSET_SCHEMA"),
    ([_source_job(), _source_job(schema="sales")], "MNEMIQ_SUPERSET_SCHEMA"),
], ids=["missing", "not-done", "not-json", "not-an-object", "values-harvested", "another-backend", "id-not-int",
        "another-shape", "another-database", "another-schema", "one-of-two-differs"])
def test_boot_refuses_a_snapshot_enriched_elsewhere(tmp_path, model, fake, no_direct_source, jobs, says):
    with pytest.raises(SnapshotMissing) as refused:
        build_runtime(superset_deployment(tmp_path, model.url, fake.url, jobs=jobs))
    assert says in str(refused.value)
    assert fake.calls == []


def test_boot_without_a_snapshot_refuses_before_any_source(tmp_path, model, fake, no_direct_source):
    with pytest.raises(SnapshotMissing, match="no snapshot for source 'only'"):
        build_runtime(superset_deployment(tmp_path, model.url, fake.url, snapshot=False))


def test_a_swap_to_a_snapshot_enriched_elsewhere_is_refused(runtime, monkeypatch):
    """A newer published version is held to the rule boot holds the first one to: one enriched elsewhere is never
    served, and the snapshot that was serving stays."""
    serving = runtime.snapshot
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    monkeypatch.setattr(runtime_module, "load_current_snapshot",
                        lambda settings, con: (_snapshot([], version="v2"), {SOURCE_ID: "v2"}))
    with pytest.raises(SnapshotMissing, match="was not enriched through Superset"):
        runtime.reload_if_stale()
    assert runtime.snapshot is serving


def test_a_swap_to_a_snapshot_enriched_through_superset_is_served(runtime, monkeypatch):
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    newer = _snapshot([_source_job()], version="v2")
    monkeypatch.setattr(runtime_module, "load_current_snapshot", lambda settings, con: (newer, {SOURCE_ID: "v2"}))
    runtime.reload_if_stale()
    assert runtime.snapshot is newer


# --- per-request execution ----------------------------------------------------------------------------------------------


def test_ask_without_a_delegation_is_refused_before_any_call(runtime, fake, model):
    with pytest.raises(SupersetCredentialMissing):
        runtime.ask("q", IDENTITY)
    assert fake.calls == []
    assert model.calls == 0


def test_a_question_too_long_to_read_is_declined_before_superset_is_asked(tmp_path, model, fake, monkeypatch,
                                                                          no_direct_source):
    retrieve_the_one_table(monkeypatch)
    runtime = build_runtime(superset_deployment(tmp_path, model.url, fake.url, max_question_chars=10))
    answer = runtime.ask("x" * 11, IDENTITY, delegation=DelegatedToken(TOKEN))
    assert answer.deferred and "the limit is 10" in answer.answer
    assert fake.calls == [] and model.calls == 0


def test_the_executor_takes_only_a_delegated_token(tmp_path, model, fake):
    settings = superset_deployment(tmp_path, model.url, fake.url)
    for credential in ("tok-1", Bearer("tok-1"), None):
        with pytest.raises(TypeError):
            superset_executor(settings, credential)
    adapter = superset_executor(settings, DelegatedToken(TOKEN))
    assert adapter.client.credential == DelegatedToken(TOKEN)
    assert (adapter.database_id, adapter.schema, adapter.dialect) == (DATABASE_ID, SCHEMA, SUPERSET_DIALECT)
    assert fake.calls == []  # building the adapter calls nothing


def test_every_call_carries_the_requests_token(runtime, fake):
    answer = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    assert not answer.deferred and not answer.failed, answer.answer
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}
    # Each kind of statement went through Superset: the grant's dry run, the catalogue and the answer itself.
    assert any(call.request.get("dry_run") for call in fake.calls)
    assert any(call.request.get("force_refresh") for call in fake.calls)
    assert [call.request["sql"] for call in _answer_statements(fake)] == [answer.trace.target_sql]
    assert answer.trace.target_sql.startswith(SQL)


def test_retrieval_is_scoped_by_what_superset_grants(runtime, fake, monkeypatch):
    """Retrieval is handed Superset's grants for this request, and an access provider that answers with those same
    grants for anyone, never the deployment's access policy."""
    seen = {}
    scripted = runtime_module.retrieve

    def recording_retrieve(con, question, identity, authz, embedder, **kwargs):
        seen.update(authz=authz, grants=kwargs["grants"])
        return scripted(con, question, identity, authz, embedder, **kwargs)

    monkeypatch.setattr(runtime_module, "retrieve", recording_retrieve)
    runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    assert seen["grants"].objects == frozenset({"wide"}) and seen["grants"].available
    assert seen["authz"] is not runtime.authz and seen["authz"].grants_for(IDENTITY) is seen["grants"]

    fake.reply("execute_sql", lambda request, authorization: {
        "success": False, "rows": None, "columns": None, "error": "You need access to the following tables: `wide`",
        "error_type": "TABLE_SECURITY_ACCESS_ERROR"})
    fake.reply("list_datasets", lambda request, authorization: {"datasets": []})
    runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    assert seen["grants"].objects == frozenset()


def test_concurrent_requests_never_share_a_token(tmp_path, fake, monkeypatch, no_direct_source):
    """Two people ask at once, five times each. Every call each person's request makes carries their own token, at
    the client and on the wire, and the answer statement, which names who asked it, arrives with that person's token."""
    retrieve_the_one_table(monkeypatch)
    cases = [ScriptedCase(id=who, question=f"count for {who}", gold_sql=f"SELECT COUNT(*) AS n_{who} FROM wide")
             for who in ("a", "b")]
    asker = threading.local()
    held: list[tuple[str, str]] = []  # (whose request, the token its client carried), one per call
    real_call = SupersetMcp.call

    def recording_call(self, tool, request, **kwargs):
        held.append((asker.who, self.credential.value))
        return real_call(self, tool, request, **kwargs)

    monkeypatch.setattr(SupersetMcp, "call", recording_call)
    with StubModel(gold=True) as model:
        model.learn(cases)
        runtime = build_runtime(superset_deployment(tmp_path, model.url, fake.url))
        start = threading.Barrier(2)

        def asks(who: str) -> list:
            asker.who = who
            start.wait()
            return [runtime.ask(f"count for {who}", IDENTITY, mode="instant", delegation=DelegatedToken(who))
                    for _ in range(5)]

        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = [answer for future in [pool.submit(asks, who) for who in ("a", "b")]
                       for answer in future.result()]
        assert not model.all_errors()
    assert all(not answer.deferred and not answer.failed for answer in answers), [a.answer for a in answers]
    assert all(who == token for who, token in held)
    assert {who for who, _ in held} == {"a", "b"}
    assert Counter(call.authorization for call in fake.calls) == Counter(f"Bearer {token}" for _, token in held)
    statements = [(_ALIAS.search(call.request["sql"]).group(1), call.authorization)
                  for call in _answer_statements(fake)]
    assert sorted(statements) == [("n_a", "Bearer a")] * 5 + [("n_b", "Bearer b")] * 5


def test_the_same_question_twice_runs_twice(runtime, fake):
    """No result cache: the second ask runs the statement again, through an adapter of its own that reads the
    function inventory again."""
    first = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    second = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    assert not first.cached and not second.cached
    assert [call.request["sql"] for call in _answer_statements(fake)] == [first.trace.target_sql] * 2
    user_functions = [call for call in fake.calls if call.request.get("force_refresh")
                      and "NOT IN ('pg_catalog', 'information_schema')" in call.request["sql"]]
    assert len(user_functions) == 2


def test_every_agent_has_no_result_cache(tmp_path, model, fake, monkeypatch, no_direct_source):
    retrieve_the_one_table(monkeypatch)
    runtime = build_runtime(superset_deployment(tmp_path, model.url, fake.url, profiles=PROFILES))
    agents = [agent for loaded in runtime.profiles.values() for agent in loaded.agents.values()]
    assert len(runtime.profiles) == 2 and len(agents) == 4  # two profiles, two modes each
    assert all(isinstance(agent.cache, NullCache) for agent in [*agents, runtime.agent])


def test_deep_is_not_offered(tmp_path, model, fake, monkeypatch, no_direct_source):
    retrieve_the_one_table(monkeypatch)
    runtime = build_runtime(superset_deployment(tmp_path, model.url, fake.url, profiles=PROFILES))
    for loaded in runtime.profiles.values():
        assert "deep" not in loaded.modes and "deep" not in loaded.agents
    with pytest.raises(UnknownMode) as refused:
        runtime.ask("q", IDENTITY, mode="deep", delegation=DelegatedToken(TOKEN))
    assert str(refused.value) == "unknown mode 'deep'; valid modes: ['instant', 'thinking']"
    assert fake.calls == [] and model.calls == 0


def test_a_deployment_whose_default_mode_is_deep_does_not_start(tmp_path, model, fake, no_direct_source):
    with pytest.raises(UnknownMode, match="MNEMIQ_MODE='deep'"):
        build_runtime(superset_deployment(tmp_path, model.url, fake.url, default_mode="deep"))


def test_write_is_refused(runtime, fake):
    with pytest.raises(SupersetCredentialMissing):
        runtime.write("UPDATE wide SET col_0 = 'two'", IDENTITY)
    assert fake.calls == []


def test_scope_and_schema_need_the_requests_token(runtime, fake):
    with pytest.raises(SupersetCredentialMissing):
        runtime.scope(IDENTITY)
    with pytest.raises(SupersetCredentialMissing):
        runtime.schema(IDENTITY)
    assert fake.calls == []


def test_scope_and_schema_show_what_superset_grants_the_person(runtime, fake):
    """What a person is shown is what Superset says they may read, asked with their token: `wide` while Superset
    accepts a dry run of it, nothing once it refuses it and lists no datasets."""
    runtime.con.execute("CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT, card TEXT)")
    runtime.con.execute("INSERT INTO semantic_object (object_id, card) VALUES ('wide', 'wide (col_0 TEXT)')")
    shown = runtime.scope(IDENTITY, delegation=DelegatedToken(TOKEN))
    assert [table["object_id"] for table in shown["tables"]] == ["wide"]
    assert [table["object_id"] for table in runtime.schema(IDENTITY, delegation=DelegatedToken(TOKEN))] == ["wide"]
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}

    fake.reply("execute_sql", lambda request, authorization: {
        "success": False, "rows": None, "columns": None, "error": "You need access to the following tables: `wide`",
        "error_type": "TABLE_SECURITY_ACCESS_ERROR"})
    fake.reply("list_datasets", lambda request, authorization: {"datasets": []})
    assert runtime.scope(IDENTITY, delegation=DelegatedToken(TOKEN))["tables"] == []
    assert runtime.schema(IDENTITY, delegation=DelegatedToken(TOKEN)) == []


def test_history_reaches_the_prompt_without_the_rows_the_client_sent(runtime, model):
    """Rows a client sends back with a turn may be another person's or stale, so none reach the prompt: the turn's
    question and SQL do, as in direct mode."""
    from mnemiq.contract import HistoryTurn

    first = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    turn = HistoryTurn(question="which region sold most", sql="SELECT 'east' AS region", columns=["region"],
                       rows=[["east-secret-row"]], grant_fingerprint=first.grant_fingerprint)
    runtime.ask("and how many?", IDENTITY, history=[turn], delegation=DelegatedToken(TOKEN))
    prompts = [json.dumps(request) for request in model.requests]
    assert any("which region sold most" in prompt for prompt in prompts)  # the turn reached the prompt
    assert not any("east-secret-row" in prompt for prompt in prompts)


# --- the agent's copy -------------------------------------------------------------------------------------------------


def _agent(adapter) -> Agent:
    return Agent(generator=None, synthesizer=None, adapter=adapter, cache=NullCache())


def test_with_adapter_refuses_another_dialect():
    agent = _agent(NoSourceConnection(SUPERSET_DIALECT))
    with pytest.raises(ValueError, match="duckdb"):
        agent.with_adapter(NoSourceConnection("duckdb"))


def test_with_adapter_replaces_only_the_adapter():
    sentinel = NoSourceConnection(SUPERSET_DIALECT)
    agent = _agent(sentinel)
    per_request = NoSourceConnection(SUPERSET_DIALECT)
    twin = agent.with_adapter(per_request)
    assert twin is not agent and twin.adapter is per_request and agent.adapter is sentinel
    assert twin.cache is agent.cache and twin.dialect == agent.dialect and twin.budget is agent.budget


# --- direct mode ------------------------------------------------------------------------------------------------------


def test_direct_mode_runtime_is_unchanged(tmp_path, model, monkeypatch):
    """The scripted deployment answers in direct mode as it did before Superset mode: through its own source, with its
    result cache, and with `deep` offered."""
    retrieve_the_one_table(monkeypatch)
    runtime = build_runtime(deployment(tmp_path, model.url))
    assert not isinstance(runtime.adapter, NoSourceConnection)
    assert all(isinstance(agent.cache, TwoTierCache) for agent in runtime.agents.values())
    assert set(runtime.agents) == {"instant", "thinking", "deep"} and runtime.router.route("q", "deep") == "deep"
    first = runtime.ask("q", IDENTITY)
    assert not first.deferred and not first.failed and not first.cached, first.answer
    assert first.answer == "The rows shown answer the question."
    assert first.trace.target_sql == f"{SQL} LIMIT 1000" and first.mode == "thinking"
    second = runtime.ask("q", IDENTITY)
    assert second.cached  # the result cache answered the same statement under the same grants
    assert runtime.ask("q", IDENTITY, mode="deep").mode == "deep"
    assert runtime.write("SELECT 1", IDENTITY).approved is False  # the governed write path, still answering itself

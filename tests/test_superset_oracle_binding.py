"""On Oracle, nothing of a person's request reaches Superset until Superset's session for them is shown to run their
statements as the schema the snapshot was read from: `Runtime.bound` asks the extension's `session_binding`, as the
person, whether the session Superset opens for them logs in as the schema the snapshot's `superset:oracle` job records,
keeps it as its current schema and reached its service. The person is served only on `{"bound": true}`. The answer
holds for the request it was asked in, and the next request asks again, so a connection re-pointed while the server
runs serves no one it was shown for before; a refusal is never kept either.

The adapter a request runs on (`superset_executor`) is held on Oracle: its client calls nothing but `session_binding`
until the runtime marks it bound, so no statement and no other tool (a chart, a dataset, a dashboard written) can
reach Superset first, on any route.

Superset is `FakeSuperset`, a real MCP server on loopback, which answers `session_binding` for the person whose token
the call carries.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import mnemiq
import mnemiq.runtime as runtime_module
from mnemiq.contract import IdentityContext
from mnemiq.runtime import build_runtime
from mnemiq.server.superset_build import WRITE_SCOPES, confirm_proposed_build
from mnemiq.server.superset_refusals import Refused, superset_refusal
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.binding import NOT_BOUND, ConnectionNotBound
from mnemiq.superset.build_store import PROPOSED, StoredBuild
from mnemiq.superset.client import DelegatedToken, SupersetRefusedCaller, SupersetUnavailable
from mnemiq.superset.executor import superset_executor
from mnemiq.superset.extension import TOOL_PREFIX
from tests.superset_extension_path import use_extension_source
from tests.superset_fake import (
    SUPERSET_TOOLS,
    FakeSuperset,
    RecordingClient,
    as_superset_runs,
    dry_run_as_sent,
    no_table_rule,
    tables_read_as_mnemiq,
)
from tests.superset_shared import (
    DATABASE_ID,
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_SETTING,
    ORACLE_SERVICE,
    _oracle_job,
    _snapshot,
    _source_job,
    superset_deployment,
)
from tests.sweep.stub_model import StubModel

BINDING = TOOL_PREFIX + "session_binding"
# What mnemiq asks it: the database Superset mode answers from, and the service and schema the snapshot's job records,
# as Oracle stores them.
ASKED = {"database_id": DATABASE_ID, "service": ORACLE_SERVICE, "schema": ORACLE_SCHEMA}

OWNER = IdentityContext(tenant_id="t", principal_id="owner")
OTHER = IdentityContext(tenant_id="t", principal_id="other")
OWNER_TOKEN, OTHER_TOKEN = "tok-owner", "tok-other"
PEOPLE = {OWNER_TOKEN: OWNER, OTHER_TOKEN: OTHER}
# The tools a build, a change or an answer calls on the adapter's client, Superset's own and the extension's.
EVERY_TOOL = (*SUPERSET_TOOLS, *(TOOL_PREFIX + base for base in (
    "create_dashboard", "create_dataset", "create_chart", "patch_chart", "add_saved_metric", "delete_dataset",
    "set_layout", "dataset_metrics", "editors_of", "get_layout", "row_rules", "table_rules", "tables_read")))


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


class OracleSuperset:
    """What `FakeSuperset` answers as a Superset whose database is Oracle's: `session_binding` for each person by their
    token (`sessions`: True or False is answered `{"bound": ...}`, a dict as it is, an exception raised), every dry run
    as sent, every other statement with no rows, the extension's `tables_read` and `table_rules` as Superset would for a
    statement it reads as mnemiq does, with no row rule, and a dashboard's layout."""

    def __init__(self, fake: FakeSuperset, sessions: dict) -> None:
        self.fake = fake
        self.sessions = dict(sessions)
        fake.reply(BINDING, self._binding)
        fake.reply("execute_sql", self._execute)
        fake.reply(TOOL_PREFIX + "tables_read", lambda request, authorization: tables_read_as_mnemiq(request))
        fake.reply(TOOL_PREFIX + "table_rules", lambda request, authorization: no_table_rule(request))
        fake.reply(TOOL_PREFIX + "get_layout", lambda request, authorization: {
            "dashboard_id": request["dashboard_id"], "position_json": "{}", "changed_on": "2026-11-27T00:00:00"})

    def _binding(self, request, authorization):
        session = self.sessions[authorization.removeprefix("Bearer ")]
        if isinstance(session, Exception):
            raise session
        return {"bound": session} if isinstance(session, bool) else session

    @staticmethod
    def _execute(request, authorization):
        if request.get("dry_run"):
            return dry_run_as_sent(request)
        return {"success": True, "columns": [{"name": "1", "type": "int64"}], "rows": [], "error": None,
                "error_type": None}


@pytest.fixture
def served():
    with FakeSuperset() as fake, StubModel(sql="SELECT 1") as model:
        yield fake, model


@pytest.fixture
def runtime(tmp_path, served):
    """A runtime in Superset mode serving a snapshot enriched directly from Oracle, as the `superset:oracle` job
    records it: service `freepdb1`, schema `SHOP`, configured as Superset shows it (`shop`)."""
    fake, model = served
    return _indexed(build_runtime(superset_deployment(tmp_path, model.url, fake.url, jobs=[_oracle_job()],
                                                      superset_schema=ORACLE_SCHEMA_SETTING)))


def _indexed(runtime):
    """`runtime` with its store's index holding the snapshot's one table, as `mnemiq build` writes it."""
    runtime.con.execute("CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT, card TEXT)")
    runtime.con.execute("INSERT INTO semantic_object (object_id, card) VALUES ('wide', 'wide (col_0 TEXT)')")
    return runtime


def _schema(runtime, token: str):
    return runtime.schema(PEOPLE[token], delegation=DelegatedToken(token))


def _tools(fake, token: str | None = None) -> list[str]:
    """Each tool called, in order, for the person whose token is `token` (everyone's with None)."""
    return [call.tool for call in fake.calls if token is None or call.authorization == f"Bearer {token}"]


# --- the check --------------------------------------------------------------------------------------------------------


def test_the_check_is_the_first_call_of_a_request_and_asks_as_the_person(runtime, served):
    """The first call of a person's request is `session_binding`, asked with their own token for the configured
    database and the service and schema the snapshot's job records, as Oracle stores them; no statement reads the
    session and no `get_database_info` is asked. Only after it is anything else sent."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: True})
    assert [card["object_id"] for card in _schema(runtime, OWNER_TOKEN)] == ["wide"]
    assert {call.authorization for call in fake.calls} == {f"Bearer {OWNER_TOKEN}"}
    assert fake.calls[0].tool == BINDING and fake.calls[0].request == ASKED
    assert _tools(fake).count(BINDING) == 1 and len(fake.calls) > 1
    assert "get_database_info" not in _tools(fake)
    assert not any("SYS_CONTEXT" in call.request.get("sql", "") for call in fake.calls)


def test_a_person_not_bound_is_refused_and_nothing_else_is_sent(runtime, served):
    fake, model = served
    OracleSuperset(fake, {OWNER_TOKEN: False})
    with pytest.raises(ConnectionNotBound) as refused:
        _schema(runtime, OWNER_TOKEN)
    assert str(refused.value) == refused.value.message == NOT_BOUND
    assert "does not log in as" in refused.value.detail
    assert _tools(fake) == [BINDING] and model.calls == 0


@pytest.mark.parametrize("answer", [
    {"refused": "backend"}, {"not_found": "database"}, {"unread": True}, {"refused": "request", "field": "schema"},
    {"bound": "true"}, {"bound": True, "more": True},
    RuntimeError("Internal error in extensions.agenticfabriq.mnemiq.session_binding: An unexpected error occurred."),
], ids=["not-oracle", "database-not-shown", "session-unread", "request-refused", "a-text", "saying-more",
        "older-extension"])
def test_an_answer_that_is_not_bound_or_not_is_a_refusal(runtime, served, answer):
    """None from the extension's reading, an older extension's failure included, is not bound: the person is refused,
    and nothing else is sent."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: answer})
    with pytest.raises(ConnectionNotBound) as refused:
        _schema(runtime, OWNER_TOKEN)
    assert "did not say" in refused.value.detail
    assert _tools(fake) == [BINDING]


def test_superset_not_answering_is_superset_not_answering(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: True})
    fake.fail_http(502)
    with pytest.raises(SupersetUnavailable):
        _schema(runtime, OWNER_TOKEN)


def test_a_refused_token_ends_the_request_as_that_refusal(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: True})
    fake.refuse_token(OWNER_TOKEN)
    with pytest.raises(SupersetRefusedCaller):
        _schema(runtime, OWNER_TOKEN)


# --- once per request -------------------------------------------------------------------------------------------------


def test_the_check_is_asked_once_per_request_and_again_on_the_next(runtime, served):
    """Within one request the answer holds: its adapter, once shown bound, asks nothing more. The next request, on an
    adapter of its own, asks again."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: True})
    adapter = superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN))
    assert runtime.bound(adapter) is None and runtime.bound(adapter) is None
    runtime.builder_grants(adapter)
    assert _tools(fake).count(BINDING) == 1
    _schema(runtime, OWNER_TOKEN)
    runtime.scope(OWNER, delegation=DelegatedToken(OWNER_TOKEN))
    assert _tools(fake).count(BINDING) == 3


def test_a_connection_re_pointed_while_the_server_runs_serves_no_one_it_was_shown_for(runtime, served):
    """A person served, then Superset's connection changes under the running server: their next request is refused."""
    fake, _ = served
    superset = OracleSuperset(fake, {OWNER_TOKEN: True})
    assert _schema(runtime, OWNER_TOKEN)
    superset.sessions[OWNER_TOKEN] = False
    with pytest.raises(ConnectionNotBound):
        _schema(runtime, OWNER_TOKEN)


def test_a_refusal_is_not_kept_so_a_connection_put_right_is_served_without_a_restart(runtime, served):
    fake, _ = served
    superset = OracleSuperset(fake, {OWNER_TOKEN: False})
    with pytest.raises(ConnectionNotBound):
        _schema(runtime, OWNER_TOKEN)
    superset.sessions[OWNER_TOKEN] = True
    assert _schema(runtime, OWNER_TOKEN)


def test_one_person_is_refused_while_another_is_served(runtime, served):
    """Superset's connection logs `other` in as another user (impersonation, say): they are refused, each time they
    ask, and nothing but the check is sent for them, while `owner`, shown bound, is served before and after."""
    fake, model = served
    OracleSuperset(fake, {OWNER_TOKEN: True, OTHER_TOKEN: False})
    assert _schema(runtime, OWNER_TOKEN)
    for _ in range(2):
        with pytest.raises(ConnectionNotBound, match="nothing is answered for you"):
            _schema(runtime, OTHER_TOKEN)
    assert _schema(runtime, OWNER_TOKEN)
    assert _tools(fake, OTHER_TOKEN) == [BINDING, BINDING]
    assert model.calls == 0


def test_the_check_asks_for_the_snapshot_being_served(runtime, served, monkeypatch):
    """A newer snapshot read from another service is asked for with that service."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: False})
    newer = _snapshot([_oracle_job(service="otherpdb")], version="v2")
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    monkeypatch.setattr(runtime_module, "load_current_snapshot", lambda settings, con: (newer, {newer.source_id: "v2"}))
    with pytest.raises(ConnectionNotBound):
        _schema(runtime, OWNER_TOKEN)
    assert fake.calls[0].request == {**ASKED, "service": "otherpdb"}


# --- the order: nothing before the check, on every door ---------------------------------------------------------------


def test_an_ask_of_a_person_not_bound_sends_nothing_but_the_check(runtime, served):
    fake, model = served
    OracleSuperset(fake, {OTHER_TOKEN: False})
    with pytest.raises(ConnectionNotBound):
        runtime.ask("how many?", OTHER, delegation=DelegatedToken(OTHER_TOKEN))
    assert _tools(fake) == [BINDING] and model.calls == 0


def test_a_builders_grants_are_read_only_once_the_builder_is_bound(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OTHER_TOKEN: False})
    with pytest.raises(ConnectionNotBound):
        runtime.builder_grants(superset_executor(runtime.settings, DelegatedToken(OTHER_TOKEN)))
    assert _tools(fake) == [BINDING]


class _Builds:
    """The build store as the confirm reads it: one proposed build onto dashboard 5."""

    def __init__(self) -> None:
        self.build = StoredBuild(tenant_id="t", build_id="b1", principal_id="owner", dashboard_id=5, goal="g", plan={},
                                 steps=(), status=PROPOSED, created_at="t", updated_at="t",
                                 planned_against="2026-11-27T00:00:00")
        self.confirmed = False

    def load_build(self, tenant, build_id, principal):
        return self.build

    def confirm_build(self, tenant, build_id, principal):
        self.confirmed = True
        return self.build


def _asking(token: str):
    return SimpleNamespace(identity=PEOPLE[token], handed={"delegation": DelegatedToken(token)},
                           scopes=set(WRITE_SCOPES))


def test_a_confirm_reads_the_dashboard_only_once_the_builder_is_bound(runtime, served):
    """The confirm of a proposed build onto a dashboard reads the dashboard's `changed_on` (`get_layout`) before it
    starts the build: only after the binding is shown, and never for a builder not shown bound."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: True, OTHER_TOKEN: False})
    builds = _Builds()
    confirm_proposed_build(_asking(OWNER_TOKEN), builds, "b1", runtime=runtime)
    assert _tools(fake) == [BINDING, TOOL_PREFIX + "get_layout"] and builds.confirmed
    builds = _Builds()
    with pytest.raises(ConnectionNotBound):
        confirm_proposed_build(_asking(OTHER_TOKEN), builds, "b1", runtime=runtime)
    assert _tools(fake, OTHER_TOKEN) == [BINDING] and not builds.confirmed


@pytest.mark.parametrize("tool", EVERY_TOOL)
def test_a_held_adapters_client_calls_nothing_but_the_check_until_its_person_is_bound(superset_oracle, tool):
    """Every tool a route calls on the adapter's client, a statement or a chart, dataset or dashboard written, is
    refused before the call while the adapter is held, as a refusal the request ends as; the check itself is called.
    Once marked bound, the same call goes."""
    inner = RecordingClient(as_superset_runs({"success": True, "columns": [], "rows": []}))
    adapter = SupersetSqlAdapter(inner, database_id=DATABASE_ID, schema=ORACLE_SCHEMA_SETTING, held=True)
    with pytest.raises(ConnectionNotBound):
        adapter.client.call(tool, {})
    assert inner.calls == [] and isinstance(adapter.terminal_refusal, ConnectionNotBound)
    adapter.client.call(BINDING, dict(ASKED))
    assert [sent.tool for sent in inner.calls] == [BINDING]
    adapter.mark_bound()
    adapter.client.call(tool, {})
    assert [sent.tool for sent in inner.calls] == [BINDING, tool]


def test_a_held_adapter_sends_no_statement_until_its_person_is_bound(superset_oracle):
    inner = RecordingClient(as_superset_runs({"success": True, "columns": [], "rows": []}))
    adapter = SupersetSqlAdapter(inner, database_id=DATABASE_ID, schema=ORACLE_SCHEMA_SETTING, held=True)
    for send in (lambda: adapter.execute("SELECT 1 FROM DUAL"), lambda: adapter.dry_run("SELECT 1 FROM DUAL"),
                 lambda: adapter.validate("SELECT 1 FROM DUAL"), lambda: adapter.confirm_sent('SELECT 1 FROM dual "DUAL"')):
        with pytest.raises(ConnectionNotBound):
            send()
    assert inner.calls == []
    adapter.mark_bound()
    adapter.execute("SELECT 1 FROM DUAL")
    assert [sent.tool for sent in inner.calls][-1] == "execute_sql"


def test_the_ask_paths_adapter_is_held_on_oracle(runtime, served):
    fake, _ = served
    adapter = superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN))
    with pytest.raises(ConnectionNotBound):
        adapter.client.call(TOOL_PREFIX + "create_chart", {})
    assert fake.calls == []


def test_a_refused_persons_adapter_stays_held(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OTHER_TOKEN: False})
    adapter = superset_executor(runtime.settings, DelegatedToken(OTHER_TOKEN))
    assert "does not log in as" in runtime.bound(adapter)
    with pytest.raises(ConnectionNotBound):
        adapter.client.call(TOOL_PREFIX + "create_dataset", {})
    assert _tools(fake) == [BINDING]


def test_only_the_runtime_marks_an_adapter_bound():
    """`mark_bound` releases an adapter's calls, so it is called where the binding was checked and nowhere else."""
    source = Path(mnemiq.__file__).parent
    calls = sorted(path.relative_to(source).as_posix() for path in source.rglob("*.py")
                   for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.Attribute) and node.attr == "mark_bound" and not isinstance(node.ctx, ast.Store))
    assert set(calls) == {"runtime.py"}


# --- PostgreSQL -------------------------------------------------------------------------------------------------------


def test_postgresql_asks_nothing_more(tmp_path, served):
    """On PostgreSQL there is no binding to check: no `session_binding`, and a held adapter holds nothing."""
    fake, model = served
    fake.reply("execute_sql", lambda request, authorization: {"success": True, "rows": None, "columns": None,
                                                              "error": None, "error_type": None, "statements": []})
    runtime = _indexed(build_runtime(superset_deployment(tmp_path, model.url, fake.url, jobs=[_source_job()])))
    adapter = superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN))
    assert runtime.bound(adapter) is None and fake.calls == []
    assert [card["object_id"] for card in _schema(runtime, OWNER_TOKEN)] == ["wide"]
    assert _tools(fake) == ["execute_sql"]  # the grants' one dry run, as before


# --- what the person is told ------------------------------------------------------------------------------------------


def test_the_refusal_is_the_persons_sentence_and_the_reason_stays_in_the_log():
    assert superset_refusal(ConnectionNotBound("logs in as 'REPORTS'")) == Refused(403, NOT_BOUND)
    assert "REPORTS" not in NOT_BOUND


# --- live: the Oracle container ---------------------------------------------------------------------------------------

LIVE_DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
LIVE_USER = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
LIVE_PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")


@pytest.mark.integration
def test_live_the_tools_statement_reads_the_session_the_job_records():
    """The statement the extension's `session_binding` runs, word for word, runs on Oracle, and its comparison reads
    the session as the enrichment's own read records it: the container's login is bound to a job naming its own
    service and schema, and not bound for a job naming another schema, nor once the session's current schema is moved
    away from its login. (Run through Superset on an Oracle database is the live Oracle run's: the live Superset stack
    has no Oracle database.)"""
    if not LIVE_DSN:
        pytest.skip("set MNEMIQ_ORACLE_TEST_DSN to run the live Oracle binding test")
    oracledb = pytest.importorskip("oracledb", reason="the Oracle adapter needs mnemiq[oracle]")
    use_extension_source()
    from agenticfabriq.mnemiq import reads

    con = oracledb.connect(user=LIVE_USER, password=LIVE_PASSWORD, dsn=LIVE_DSN)
    try:
        cur = con.cursor()
        cur.execute("SELECT SYS_CONTEXT('USERENV', 'SERVICE_NAME'), SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL")
        service, schema = cur.fetchone()  # what enrichment records in the job
        cur.execute(reads.SESSION_SQL)
        [session] = cur.fetchall()
        assert reads.session_bound(session, service=service, schema=schema) is True, session
        assert reads.session_bound(session, service=service, schema=schema + "X") is False
        assert reads.session_bound(session, service=service.upper(), schema=schema) is (service == service.upper())
        cur.execute("ALTER SESSION SET CURRENT_SCHEMA = SYS")
        cur.execute(reads.SESSION_SQL)
        [moved] = cur.fetchall()
        assert moved[2] == "SYS" and reads.session_bound(moved, service=service, schema=schema) is False
    finally:
        con.close()

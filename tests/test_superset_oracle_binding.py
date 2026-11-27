"""On Oracle, a person's statements are sent only once Superset's connection for them is shown to run them as the schema
the snapshot was read from: `Runtime.bound` asks Superset, as the person, which backend the database Superset mode answers
from is (`get_database_info`), and, with the binding statement sent as every Oracle statement is, which user the
session logs in as, the service it reached and its current schema. The person is served only when the backend is
Oracle, the service is the one the snapshot's `superset:oracle` job records, and the login user and the current schema
are both the job's schema. A bound answer is kept for the person for as long as the process runs; a refusal and a
failed read are not, so a connection an operator puts right is served on the next request.

Superset is `FakeSuperset`, a real MCP server on loopback, which answers the binding statement with the session of the
person whose token the call carries.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

import mnemiq
import mnemiq.runtime as runtime_module
from mnemiq.contract import IdentityContext
from mnemiq.runtime import build_runtime
from mnemiq.server.superset_refusals import Refused, superset_refusal
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.binding import (
    BINDING_SQL,
    BINDING_UNREAD,
    NOT_BOUND,
    BindingUnread,
    ConnectionNotBound,
    session_refusal,
)
from mnemiq.superset.client import DelegatedToken, SupersetRefusedCaller, SupersetUnavailable
from mnemiq.superset.executor import superset_executor
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.names import for_superset
from tests.superset_fake import (
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

# The binding statement as mnemiq writes it, and as it is sent: each table named as Superset names it
# (`names.for_superset`), so DUAL is sent as Superset's `dual` under its own name as an alias.
WRITTEN = "SELECT USER, SYS_CONTEXT('USERENV','SERVICE_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL"
SENT = "SELECT USER, SYS_CONTEXT('USERENV','SERVICE_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM dual \"DUAL\""

OWNER = IdentityContext(tenant_id="t", principal_id="owner")
OTHER = IdentityContext(tenant_id="t", principal_id="other")
OWNER_TOKEN, OTHER_TOKEN = "tok-owner", "tok-other"
PEOPLE = {OWNER_TOKEN: OWNER, OTHER_TOKEN: OTHER}

# A session as the binding statement reads it: the login user, the service and the current schema, as Oracle stores
# each. BOUND is the one the snapshot's job records.
BOUND = (ORACLE_SCHEMA, ORACLE_SERVICE, ORACLE_SCHEMA)
ORACLE_DATABASE = {"id": DATABASE_ID, "database_name": "warehouse", "backend": "oracle"}


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


def _session_payload(session) -> dict:
    """`execute_sql`'s answer to the binding statement: one row of the session's three values, under the names Oracle
    gives the statement's columns."""
    names = ["USER", "SYS_CONTEXT('USERENV','SERVICE_NAME')", "SYS_CONTEXT('USERENV','CURRENT_SCHEMA')"]
    rows = session if isinstance(session, list) else [session]
    return {"success": True, "columns": [{"name": name, "type": "object"} for name in names],
            "rows": [dict(zip(names, row)) for row in rows], "row_count": len(rows), "error": None, "error_type": None}


class OracleSuperset:
    """What `FakeSuperset` answers as a Superset whose database is `database` (Oracle's by default): the session of
    each person by their token (`sessions`; a dict is answered as `execute_sql`'s payload, an exception raised), every
    dry run as sent, every other statement with no rows, and the extension's `tables_read` and `table_rules` as
    Superset would for a statement it reads as mnemiq does, with no row rule."""

    def __init__(self, fake: FakeSuperset, sessions: dict, *, database: dict | Exception | None = None) -> None:
        self.fake = fake
        self.sessions = dict(sessions)
        self.database = ORACLE_DATABASE if database is None else database
        self.dry_run = dry_run_as_sent
        fake.reply("get_database_info", self._database)
        fake.reply("execute_sql", self._execute)
        fake.reply(TOOL_PREFIX + "tables_read", lambda request, authorization: tables_read_as_mnemiq(request))
        fake.reply(TOOL_PREFIX + "table_rules", lambda request, authorization: no_table_rule(request))

    def _database(self, request, authorization):
        if isinstance(self.database, Exception):
            raise self.database
        return self.database

    def _execute(self, request, authorization):
        if request.get("dry_run"):
            return self.dry_run(request)
        if request["sql"] == SENT:
            session = self.sessions[authorization.removeprefix("Bearer ")]
            if isinstance(session, Exception):
                raise session
            return session if isinstance(session, dict) else _session_payload(session)
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


def _statements(fake, token: str | None = None) -> list[tuple[str, bool]]:
    """Each `execute_sql` call, as its statement and whether it was a dry run, for the person whose token is `token`
    (everyone's with None)."""
    return [(call.request["sql"], bool(call.request.get("dry_run"))) for call in fake.calls
            if call.tool == "execute_sql" and (token is None or call.authorization == f"Bearer {token}")]


def _binding_calls(fake, token: str) -> list[str]:
    """The tools the binding check of `token`'s person called, in order: up to the binding statement's own call."""
    tools = []
    for call in fake.calls:
        if call.authorization != f"Bearer {token}":
            continue
        tools.append(call.tool.removeprefix(TOOL_PREFIX))
        if call.tool == "execute_sql" and call.request["sql"] == SENT and not call.request.get("dry_run"):
            return tools
    return tools


# --- the statement ----------------------------------------------------------------------------------------------------


def test_the_binding_statement_is_written_and_sent_as_named(superset_oracle):
    assert BINDING_SQL == WRITTEN
    assert for_superset(BINDING_SQL) == SENT


def test_the_check_asks_as_the_person_and_sends_the_statement_through_every_oracle_check(runtime, served):
    """The first call of a person's first request is `get_database_info` for the configured database; then the binding
    statement, sent as every Oracle statement is: Superset's dry run of it, the extension's `tables_read` and
    `table_rules`, and only then the statement itself, on the configured database and schema, asked to run afresh. Each
    call carries the person's own token, and only after them is anything else sent."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    _schema(runtime, OWNER_TOKEN)
    assert {call.authorization for call in fake.calls} == {f"Bearer {OWNER_TOKEN}"}
    assert _binding_calls(fake, OWNER_TOKEN) == ["get_database_info", "execute_sql", "tables_read", "table_rules",
                                                 "execute_sql"]
    assert fake.calls[0].request == {"identifier": DATABASE_ID}
    dry, real = fake.calls[1].request, fake.calls[4].request
    assert real == {"database_id": DATABASE_ID, "sql": SENT, "schema": ORACLE_SCHEMA_SETTING, "timeout": 30,
                    "force_refresh": True}
    assert dry == {**real, "dry_run": True}
    statements = _statements(fake)
    assert statements[:2] == [(SENT, True), (SENT, False)]
    assert len(statements) > 2 and SENT not in {sql for sql, _ in statements[2:]}  # the grants' probe came after


def test_a_bound_person_is_served(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    assert runtime.bound(OWNER, superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN))) is None
    assert [card["object_id"] for card in _schema(runtime, OWNER_TOKEN)] == ["wide"]


# --- each condition ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("session, says", [
    ((ORACLE_SCHEMA, "otherpdb", ORACLE_SCHEMA), "service"),
    (("REPORTS", ORACLE_SERVICE, ORACLE_SCHEMA), "logs in as 'REPORTS'"),
    ((ORACLE_SCHEMA, ORACLE_SERVICE, "REPORTS"), "current schema is 'REPORTS'"),
    # Superset's spelling of the schema is not the name Oracle stores: compared as stored, it is another name.
    ((ORACLE_SCHEMA_SETTING, ORACLE_SERVICE, ORACLE_SCHEMA_SETTING), "logs in as 'shop'"),
], ids=["another-service", "another-login", "another-current-schema", "superset-spelling"])
def test_a_session_other_than_the_snapshots_is_refused_and_nothing_else_is_sent(runtime, served, session, says):
    fake, model = served
    OracleSuperset(fake, {OWNER_TOKEN: session})
    with pytest.raises(ConnectionNotBound) as refused:
        _schema(runtime, OWNER_TOKEN)
    assert str(refused.value) == refused.value.message == NOT_BOUND
    assert says in refused.value.detail
    assert _statements(fake) == [(SENT, True), (SENT, False)]  # the check's statement, and no grant probe
    assert model.calls == 0


@pytest.mark.parametrize("database, says", [
    ({**ORACLE_DATABASE, "backend": "postgresql"}, "backend 'postgresql'"),
    ({"error": "Database with ID 7 not found", "error_type": "not_found"}, "did not describe database 7"),
], ids=["postgresql", "not-shown"])
def test_a_database_superset_does_not_describe_as_oracle_is_refused_before_any_statement(runtime, served, database,
                                                                                        says):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND}, database=database)
    with pytest.raises(ConnectionNotBound) as refused:
        _schema(runtime, OWNER_TOKEN)
    assert says in refused.value.detail
    assert [call.tool for call in fake.calls] == ["get_database_info"]


@pytest.mark.parametrize("user, service, current, says", [
    (ORACLE_SCHEMA, ORACLE_SERVICE, ORACLE_SCHEMA, None),
    (ORACLE_SCHEMA, "FREEPDB1", ORACLE_SCHEMA, "service is 'FREEPDB1'"),
    ("APP", ORACLE_SERVICE, ORACLE_SCHEMA, "logs in as 'APP'"),
    (ORACLE_SCHEMA, ORACLE_SERVICE, "APP", "current schema is 'APP'"),
])
def test_the_session_is_compared_with_the_job_as_oracle_stores_each(user, service, current, says):
    """No name is mapped: the job records the service and the schema as the enrichment's session read them, and the
    binding statement reads the person's the same way, so each is compared exactly."""
    refusal = session_refusal((user, service, current), service=ORACLE_SERVICE, schema=ORACLE_SCHEMA)
    assert refusal is None if says is None else says in refusal


# --- once per person --------------------------------------------------------------------------------------------------


def test_a_bound_person_is_checked_once_and_every_door_reads_the_answer(runtime, served):
    """A second request of a person shown bound asks nothing about the connection: the schema, the opening screen and a
    builder's grants alike."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    _schema(runtime, OWNER_TOKEN)
    runtime.scope(OWNER, delegation=DelegatedToken(OWNER_TOKEN))
    runtime.builder_grants(superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN)), OWNER)
    assert [call.tool for call in fake.calls].count("get_database_info") == 1
    assert [sql for sql, _ in _statements(fake)].count(SENT) == 2  # its dry run and its call, once


def test_one_person_is_refused_while_another_is_served(runtime, served):
    """Superset's connection logs `other` in as another user (impersonation, say): they are refused, each time they
    ask, and nothing but the check is sent for them, while `owner`, shown bound, is served before and after."""
    fake, model = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND, OTHER_TOKEN: ("OTHER", ORACLE_SERVICE, "OTHER")})
    assert _schema(runtime, OWNER_TOKEN)
    for _ in range(2):
        with pytest.raises(ConnectionNotBound, match="nothing is answered for you"):
            _schema(runtime, OTHER_TOKEN)
    assert _schema(runtime, OWNER_TOKEN)
    assert _statements(fake, OTHER_TOKEN) == [(SENT, True), (SENT, False)] * 2
    assert [sql for sql, _ in _statements(fake, OWNER_TOKEN)].count(SENT) == 2
    assert model.calls == 0


def test_a_person_is_their_tenant_and_their_principal(runtime, served):
    """The answer kept is for the person as the door verified them: a principal of the same id in another tenant is
    another person, and is checked for themselves."""
    fake, _ = served
    elsewhere = IdentityContext(tenant_id="t2", principal_id=OWNER.principal_id)
    OracleSuperset(fake, {OWNER_TOKEN: BOUND, OTHER_TOKEN: ("OTHER", ORACLE_SERVICE, "OTHER")})
    assert _schema(runtime, OWNER_TOKEN)
    with pytest.raises(ConnectionNotBound):
        runtime.schema(elsewhere, delegation=DelegatedToken(OTHER_TOKEN))


def test_a_refused_persons_adapter_stays_held(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OTHER_TOKEN: ("OTHER", ORACLE_SERVICE, "OTHER")})
    adapter = superset_executor(runtime.settings, DelegatedToken(OTHER_TOKEN))
    assert "logs in as 'OTHER'" in runtime.bound(OTHER, adapter)
    calls = len(fake.calls)
    with pytest.raises(ConnectionNotBound):
        adapter.execute("SELECT 1 FROM DUAL")
    assert len(fake.calls) == calls


def test_a_refusal_is_not_kept_so_a_connection_put_right_is_served_without_a_restart(runtime, served):
    fake, _ = served
    superset = OracleSuperset(fake, {OWNER_TOKEN: ("REPORTS", ORACLE_SERVICE, "REPORTS")})
    with pytest.raises(ConnectionNotBound):
        _schema(runtime, OWNER_TOKEN)
    superset.sessions[OWNER_TOKEN] = BOUND
    assert _schema(runtime, OWNER_TOKEN)
    assert [call.tool for call in fake.calls].count("get_database_info") == 2


# --- a failed read ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("session", [
    {"success": False, "rows": None, "columns": None, "error": "ORA-03113: end-of-file on communication channel",
     "error_type": "GENERIC_DB_ENGINE_ERROR"},
    {"success": False, "rows": None, "columns": None, "error": "timed out", "error_type": "timed_out"},
    [BOUND, BOUND],
    [],
    (None, ORACLE_SERVICE, ORACLE_SCHEMA),
    (ORACLE_SCHEMA, "", ORACLE_SCHEMA),
], ids=["database-error", "time-limit", "two-rows", "no-row", "a-null", "an-empty-value"])
def test_a_failed_read_refuses_that_request_alone_and_is_asked_again(runtime, served, session):
    fake, _ = served
    superset = OracleSuperset(fake, {OWNER_TOKEN: session})
    with pytest.raises(BindingUnread) as unread:
        _schema(runtime, OWNER_TOKEN)
    assert str(unread.value) == BINDING_UNREAD
    assert _statements(fake) == [(SENT, True), (SENT, False)]
    superset.sessions[OWNER_TOKEN] = BOUND
    assert _schema(runtime, OWNER_TOKEN)  # the next request asks again, and is served
    assert [call.tool for call in fake.calls].count("get_database_info") == 2


@pytest.mark.parametrize("database", [{"id": DATABASE_ID}, {"id": 8, "backend": "oracle"}, {"id": "7", "backend": "oracle"}],
                         ids=["no-backend", "another-id", "id-as-text"])
def test_a_description_that_cannot_be_read_is_a_failed_read(runtime, served, database):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND}, database=database)
    with pytest.raises(BindingUnread):
        _schema(runtime, OWNER_TOKEN)
    assert [call.tool for call in fake.calls] == ["get_database_info"]


def test_a_check_mnemiq_would_not_send_is_a_failed_read_and_never_sent(runtime, served):
    """The binding statement is held to every check an Oracle statement is: here Superset's dry run says it would run
    another statement, so it is not sent, and the request is refused as a read that failed."""
    fake, _ = served
    superset = OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    superset.dry_run = lambda request: {**dry_run_as_sent(request), "statements": [
        {"original_sql": request["sql"], "executed_sql": request["sql"].replace("USER", "'SHOP'"), "row_count": 0,
         "data": None}]}
    with pytest.raises(BindingUnread):
        _schema(runtime, OWNER_TOKEN)
    assert _statements(fake) == [(SENT, True)]  # its dry run, and no call


def test_superset_not_answering_is_superset_not_answering(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND}, database=RuntimeError("boom"))
    fake.fail_http(502)
    with pytest.raises(SupersetUnavailable):
        _schema(runtime, OWNER_TOKEN)


def test_a_refused_token_ends_the_request_as_that_refusal(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    fake.refuse_token(OWNER_TOKEN)
    with pytest.raises(SupersetRefusedCaller):
        _schema(runtime, OWNER_TOKEN)


def test_superset_refusing_the_statements_table_refuses_the_person(runtime, served):
    """Superset's own access check refuses the binding statement where the person may not read DUAL as Superset
    reads it: the person is refused, as not shown bound, with Superset's refusal in the log only."""
    fake, _ = served
    superset = OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    superset.dry_run = lambda request: {"success": False, "rows": None, "columns": None,
                                        "error": "You need access to the following tables: `shop.dual`",
                                        "error_type": "TABLE_SECURITY_ACCESS_ERROR"}
    with pytest.raises(ConnectionNotBound) as refused:
        _schema(runtime, OWNER_TOKEN)
    assert "refused" in refused.value.detail and "dual" not in str(refused.value)
    assert _statements(fake) == [(SENT, True)]


# --- the order: nothing before the check, on every door ---------------------------------------------------------------


def test_an_ask_of_a_person_not_bound_sends_nothing_but_the_check(runtime, served, monkeypatch):
    fake, model = served
    OracleSuperset(fake, {OTHER_TOKEN: ("OTHER", ORACLE_SERVICE, "OTHER")})
    with pytest.raises(ConnectionNotBound):
        runtime.ask("how many?", OTHER, delegation=DelegatedToken(OTHER_TOKEN))
    assert _statements(fake) == [(SENT, True), (SENT, False)]
    assert model.calls == 0


def test_a_builders_grants_are_read_only_once_the_builder_is_bound(runtime, served):
    fake, _ = served
    OracleSuperset(fake, {OTHER_TOKEN: ("OTHER", ORACLE_SERVICE, "OTHER")})
    with pytest.raises(ConnectionNotBound):
        runtime.builder_grants(superset_executor(runtime.settings, DelegatedToken(OTHER_TOKEN)), OTHER)
    assert _statements(fake) == [(SENT, True), (SENT, False)]


def test_an_adapter_holds_every_statement_but_the_check_until_its_person_is_bound(superset_oracle):
    """The adapter `superset_executor` builds on Oracle sends nothing before the runtime marks its person bound, the
    binding statement apart: a statement, a dry run, the proof's probe and a dataset's SQL are each refused before any
    call, as a refusal the request ends as."""
    client = RecordingClient(as_superset_runs(_session_payload(BOUND)))
    adapter = SupersetSqlAdapter(client, database_id=DATABASE_ID, schema=ORACLE_SCHEMA_SETTING, held=True)
    for send in (lambda: adapter.execute("SELECT 1 FROM DUAL"), lambda: adapter.dry_run("SELECT 1 FROM DUAL"),
                 lambda: adapter.validate("SELECT 1 FROM DUAL"), lambda: adapter.execute_arrow_afresh("SELECT 1 FROM DUAL"),
                 lambda: adapter.confirm_sent("SELECT 1 FROM dual \"DUAL\"")):
        with pytest.raises(ConnectionNotBound):
            send()
    assert client.calls == []
    assert isinstance(adapter.terminal_refusal, ConnectionNotBound)
    assert adapter.execute_arrow_afresh(BINDING_SQL).num_rows == 1  # the check's own statement is sent
    adapter.mark_bound()
    adapter.execute("SELECT 1 FROM DUAL")
    assert [call.request["sql"] for call in client.calls if call.tool == "execute_sql"][-1] == "SELECT 1 FROM dual \"DUAL\""


def test_the_ask_paths_adapter_is_held_on_oracle(runtime, superset_oracle):
    adapter = superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN))
    with pytest.raises(ConnectionNotBound):
        adapter.execute("SELECT 1 FROM DUAL")  # refused before any call: nothing listens at the client's URL


def test_only_the_runtime_marks_an_adapter_bound():
    """`mark_bound` releases an adapter's statements, so it is called where the binding was checked and nowhere else."""
    source = Path(mnemiq.__file__).parent
    calls = sorted(path.relative_to(source).as_posix() for path in source.rglob("*.py")
                   for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.Attribute) and node.attr == "mark_bound" and not isinstance(node.ctx, ast.Store))
    assert set(calls) == {"runtime.py"}


# --- the job the check compares with ----------------------------------------------------------------------------------


def test_a_snapshot_read_from_another_service_is_checked_again(runtime, served, monkeypatch):
    """The answer kept is for the service and schema the check compared with: a newer snapshot read from another
    service is checked again, and a person whose connection does not reach it is refused."""
    fake, _ = served
    OracleSuperset(fake, {OWNER_TOKEN: BOUND})
    _schema(runtime, OWNER_TOKEN)
    newer = _snapshot([_oracle_job(service="otherpdb")], version="v2")
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    monkeypatch.setattr(runtime_module, "load_current_snapshot", lambda settings, con: (newer, {newer.source_id: "v2"}))
    with pytest.raises(ConnectionNotBound):
        _schema(runtime, OWNER_TOKEN)
    assert [call.tool for call in fake.calls].count("get_database_info") == 2


# --- PostgreSQL -------------------------------------------------------------------------------------------------------


def test_postgresql_asks_nothing_more(tmp_path, served):
    """On PostgreSQL there is no binding to check: no `get_database_info`, no binding statement, and an adapter that
    holds nothing."""
    fake, model = served
    fake.reply("execute_sql", lambda request, authorization: {"success": True, "rows": None, "columns": None,
                                                              "error": None, "error_type": None, "statements": []})
    runtime = _indexed(build_runtime(superset_deployment(tmp_path, model.url, fake.url, jobs=[_source_job()])))
    adapter = superset_executor(runtime.settings, DelegatedToken(OWNER_TOKEN))
    assert runtime.bound(OWNER, adapter) is None and fake.calls == []
    assert [card["object_id"] for card in _schema(runtime, OWNER_TOKEN)] == ["wide"]
    assert [call.tool for call in fake.calls] == ["execute_sql"]  # the grants' one dry run, as before


# --- what the person is told ------------------------------------------------------------------------------------------


def test_each_refusal_is_the_persons_sentence_and_the_reason_stays_in_the_log():
    assert superset_refusal(ConnectionNotBound("logs in as 'REPORTS'")) == Refused(403, NOT_BOUND)
    assert superset_refusal(BindingUnread("ORA-03113")) == Refused(503, BINDING_UNREAD)
    assert "REPORTS" not in NOT_BOUND and "ORA" not in BINDING_UNREAD


# --- live: the Oracle container ---------------------------------------------------------------------------------------

LIVE_DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
LIVE_USER = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
LIVE_PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")


@pytest.mark.integration
def test_live_the_statement_as_sent_reads_the_session_the_job_records(superset_oracle):
    """The binding statement as it is sent runs on Oracle and reads the session as the enrichment's own read records
    it: the container's login is bound to a job naming its own service and schema, and refused for a job naming
    another schema, and once the session's current schema is moved away from its login."""
    if not LIVE_DSN:
        pytest.skip("set MNEMIQ_ORACLE_TEST_DSN to run the live Oracle binding test")
    oracledb = pytest.importorskip("oracledb", reason="the Oracle adapter needs mnemiq[oracle]")
    con = oracledb.connect(user=LIVE_USER, password=LIVE_PASSWORD, dsn=LIVE_DSN)
    try:
        cur = con.cursor()
        cur.execute("SELECT SYS_CONTEXT('USERENV', 'SERVICE_NAME'), SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL")
        service, schema = cur.fetchone()  # what enrichment records in the job
        cur.execute(for_superset(BINDING_SQL))
        [session] = cur.fetchall()
        assert session_refusal(session, service=service, schema=schema) is None, session
        assert "logs in as" in session_refusal(session, service=service, schema=schema + "X")
        cur.execute("ALTER SESSION SET CURRENT_SCHEMA = SYS")
        cur.execute(for_superset(BINDING_SQL))
        [moved] = cur.fetchall()
        assert "current schema is 'SYS'" in session_refusal(moved, service=service, schema=schema)
    finally:
        con.close()

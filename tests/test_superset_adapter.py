"""`SupersetSqlAdapter`: mnemiq's statements run through Superset's `execute_sql` tool, and every answer read the way
the query runner and the proof seam need it.

Most tests drive the adapter over a recording client, which stands in for `SupersetMcp` without a wire, and the
query runner's `run` over the adapter, so each reading is checked where it lands: repaired, retried, or the end of the
turn. One test runs the adapter over the real client against the fake Superset, so the two are checked joined.
"""

from __future__ import annotations

import logging
import time

import pytest

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.execute.runner import ExecutionError, QueryCancelledError, SourceUnavailableError, run
from mnemiq.sql.prove import Proof, prove_outcome
from mnemiq.superset.adapter import JINJA_DELIMITERS, TRANSPORT_MARGIN_S, SupersetSqlAdapter
from mnemiq.superset.client import (
    DelegatedToken,
    SupersetDenied,
    SupersetMcp,
    SupersetPermissionDenied,
    SupersetRefusedCaller,
    SupersetResponseTooLarge,
    SupersetToolError,
    SupersetUnavailable,
)
from mnemiq.superset.payload import ResultCut, SupersetAccessRefused, SupersetQueryFailed
from mnemiq.superset.settings import SUPERSET_DIALECT
from tests.superset_fake import FakeSuperset, RecordingClient

SQL = "SELECT region, count(*) AS policy_count FROM policies GROUP BY region"
REGION_COUNTS = {"success": True, "rows": [{"region": "west", "policy_count": 16666}],
                 "columns": [{"name": "region", "type": "object"}, {"name": "policy_count", "type": "int64"}],
                 "row_count": 1, "affected_rows": None, "execution_time": 0.012, "error": None, "error_type": None}
# What Superset answers a dry run: the statements it would run, and no rows or columns.
DRY_RUN = {"success": True, "rows": None, "columns": None, "row_count": None, "affected_rows": None,
           "error": None, "error_type": None,
           "statements": [{"original_sql": SQL, "executed_sql": SQL, "row_count": 0, "data": None}]}
# What Superset answers a statement that reads a table the person asking cannot see.
HIDDEN_TABLE_REFUSAL = {"success": False, "rows": None, "columns": None,
                        "error": "You need access to the following tables: `sales.secret_table`",
                        "error_type": "TABLE_SECURITY_ACCESS_ERROR"}


def _failed(error_type: str, error: str = "Superset's own words") -> dict:
    return {"success": False, "rows": None, "columns": None, "error": error, "error_type": error_type}


def _adapter(answer: dict | Exception = REGION_COUNTS, **kwargs) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(RecordingClient(answer), database_id=7, schema="sales", **kwargs)


def test_request_names_database_schema_and_timeout():
    adapter = _adapter()
    table = adapter.execute_arrow(SQL)
    assert [call.tool for call in adapter.client.calls] == ["execute_sql"]
    # No `limit`, which would replace the statement's own LIMIT, and no `template_params`.
    assert adapter.client.calls[0].request == {"database_id": 7, "sql": SQL, "schema": "sales", "timeout": 30}
    assert table.to_pylist() == [{"region": "west", "policy_count": 16666}]
    assert adapter.dialect == SUPERSET_DIALECT == "postgres"


def test_data_statements_never_force_refresh():
    """Only the catalogue asks Superset to skip its result cache. A data statement may be answered from it: Superset
    keys a cached result on the statement after its row rules are applied, and checks access before it looks. Asked on
    an adapter that has already read its catalogue, so a catalogue read cannot change what a data statement sends."""

    def answer(request: dict) -> dict:
        if "pg_catalog" in request["sql"]:
            return {"success": True, "rows": [], "columns": [{"name": "function_name", "type": "object"}]}
        return DRY_RUN if request.get("dry_run") else REGION_COUNTS

    adapter = _adapter(answer)
    adapter.user_functions()
    assert adapter.client.calls[0].request["force_refresh"] is True
    adapter.execute_arrow(SQL)
    adapter.execute(SQL)
    adapter.validate(SQL)
    adapter.dry_run(SQL)
    data = [call.request for call in adapter.client.calls[1:]]
    assert len(data) == 4
    for request in data:
        assert "force_refresh" not in request
        assert set(request) - {"dry_run"} == {"database_id", "sql", "schema", "timeout"}


def test_execute_returns_rows_as_tuples_in_column_order():
    assert _adapter().execute(SQL) == [("west", 16666)]


@pytest.mark.parametrize(("asked", "sent"), [(0.2, 1), (29.5, 30), (900, 300), (None, 30), (60, 60)])
def test_timeout_is_clamped_to_superset_range(asked, sent):
    """Superset takes whole seconds from 1 to 300 (`ExecuteSqlRequest.timeout`); a part second rounds up, so a query
    is never given less time than it asked for."""
    adapter = _adapter()
    adapter.execute_arrow(SQL, timeout_s=asked)
    timeout = adapter.client.calls[0].request["timeout"]
    assert timeout == sent
    assert type(timeout) is int


def test_the_adapter_default_timeout_is_its_own():
    adapter = _adapter(timeout_s=120.0)
    adapter.execute_arrow(SQL)
    assert adapter.client.calls[0].request["timeout"] == 120


@pytest.mark.parametrize("method", ["execute_arrow", "validate", "dry_run"])
@pytest.mark.parametrize("asked", [0.2, 29.5, 900, None])
def test_the_transport_waits_out_supersets_own_timeout(method, asked):
    """Superset's own `timed_out` answer has to arrive before the transport gives up, or a slow query reads as an
    outage instead of a timeout."""
    adapter = _adapter(DRY_RUN if method == "dry_run" else REGION_COUNTS)
    if method == "execute_arrow":
        adapter.execute_arrow(SQL, timeout_s=asked)
    else:
        getattr(adapter, method)(SQL)
    call = adapter.client.calls[0]
    assert TRANSPORT_MARGIN_S == 15.0
    assert call.timeout_s == call.request["timeout"] + TRANSPORT_MARGIN_S


@pytest.mark.parametrize("delimiter", JINJA_DELIMITERS)
def test_jinja_delimiters_are_never_sent(delimiter):
    """Superset renders a statement as a Jinja template before it runs it, so a statement holding a template marker
    could run as something other than what was proved."""
    assert JINJA_DELIMITERS == ("{{", "{%", "{#")
    sql = f"SELECT '{delimiter} x' AS label FROM policies"
    for method in ("execute_arrow", "execute", "validate", "dry_run"):
        adapter = _adapter()
        with pytest.raises(SupersetQueryFailed) as caught:
            getattr(adapter, method)(sql)
        assert adapter.client.calls == [], method
        assert delimiter in str(caught.value)


def test_validate_runs_the_statement_as_an_empty_subquery():
    adapter = _adapter(_answer_with_no_rows())
    assert adapter.validate(SQL) is None
    assert adapter.client.calls[0].request == {
        "database_id": 7, "sql": f"SELECT * FROM ({SQL}) AS mnemiq_probe WHERE 1 = 0", "schema": "sales",
        "timeout": 30}


def test_validate_drops_a_trailing_semicolon():
    adapter = _adapter(_answer_with_no_rows())
    adapter.validate(f"{SQL};\n")
    assert adapter.client.calls[0].request["sql"] == f"SELECT * FROM ({SQL}) AS mnemiq_probe WHERE 1 = 0"


def test_validate_raises_what_superset_refuses():
    with pytest.raises(SupersetQueryFailed) as caught:
        _adapter(_failed("failed", 'column "regoin" does not exist')).validate(SQL)
    assert str(caught.value) == 'column "regoin" does not exist'


def test_dry_run_sends_dry_run():
    adapter = _adapter(DRY_RUN)
    assert adapter.dry_run(SQL) is None
    assert adapter.client.calls[0].request == {"database_id": 7, "sql": SQL, "schema": "sales", "timeout": 30,
                                               "dry_run": True}


def test_dry_run_reads_a_failure_as_a_run_does():
    """A dry run that Superset refuses raises what the same refusal raises from a run; whatever a successful dry run
    carries, it returns nothing."""
    with pytest.raises(SupersetAccessRefused):
        _adapter(HIDDEN_TABLE_REFUSAL).dry_run(SQL)
    with pytest.raises(QueryTimedOut):
        _adapter(_failed("timed_out")).dry_run(SQL)
    doubled = {**REGION_COUNTS, "columns": [{"name": "id", "type": "int64"}] * 2, "_response_truncated": True}
    assert _adapter(doubled).dry_run(SQL) is None


def test_source_unavailable_is_true_for_an_outage_and_every_refusal_of_the_caller():
    adapter = _adapter()
    for unanswered in (SupersetUnavailable("down"), SupersetRefusedCaller("401"), SupersetPermissionDenied("role"),
                       SupersetDenied("no"), QueryTimedOut("slow"), QueryCancelled("stopped")):
        assert adapter.source_unavailable(unanswered) is True, unanswered
    for answered in (SupersetToolError("tool"), SupersetQueryFailed("bad column"), SupersetAccessRefused("refused"),
                     ResultCut("cut"), RuntimeError("other")):
        assert adapter.source_unavailable(answered) is False, answered


def test_the_proof_reads_an_unanswered_probe_as_unavailable():
    assert prove_outcome(_adapter(SupersetRefusedCaller("401")), SQL).proof is Proof.UNAVAILABLE
    assert prove_outcome(_adapter(_failed("timed_out")), SQL).proof is Proof.UNAVAILABLE
    assert prove_outcome(_adapter(_failed("failed", "syntax error")), SQL).proof is Proof.REJECTED
    assert prove_outcome(_adapter(_answer_with_no_rows()), SQL).proof is Proof.ACCEPTED


def test_cut_result_is_repaired_not_an_outage():
    cut = {**REGION_COUNTS, "_response_truncated": True, "_truncation_notes": ["Result truncated: 1 of 90000 rows"]}
    with pytest.raises(ExecutionError) as caught:
        run(_adapter(cut), SQL)
    assert type(caught.value) is ExecutionError
    assert "aggregate, or select fewer rows or columns" in caught.value.repair_text


TOO_LARGE = SupersetResponseTooLarge("Superset's answer to execute_sql was larger than its MCP service will send; ask "
                                     "for less", detail="Error: Response too large: 61,234 bytes (limit: 50,000)")


@pytest.mark.parametrize("method", ["execute_arrow", "execute"])
def test_a_result_too_large_to_send_reads_as_cut(method):
    """Superset's MCP service refuses a result too large to send when dropping rows does not bring it under its size.
    That reads as a cut result does: one exception for a result too large, the statement's to make smaller."""
    with pytest.raises(ResultCut) as caught:
        getattr(_adapter(TOO_LARGE), method)(SQL)
    assert caught.value.__cause__ is TOO_LARGE
    assert "aggregate, or select fewer rows or columns" in str(caught.value)
    assert "61,234" not in str(caught.value)


def test_a_result_too_large_to_send_is_repaired_not_an_outage():
    with pytest.raises(ExecutionError) as caught:
        run(_adapter(TOO_LARGE), SQL)
    assert type(caught.value) is ExecutionError
    assert "aggregate, or select fewer rows or columns" in caught.value.repair_text


def test_a_dry_run_too_large_to_send_raises_as_itself():
    """A dry run returns no rows to make fewer of, and grants read its failures as Superset's tool errors, so it
    raises as itself."""
    with pytest.raises(SupersetResponseTooLarge):
        _adapter(TOO_LARGE).dry_run(SQL)


def test_timeout_reads_as_timeout():
    with pytest.raises(ExecutionError) as caught:
        run(_adapter(_failed("timed_out", "Query exceeded the timeout limit")), SQL, timeout_s=30)
    assert type(caught.value) is ExecutionError
    assert caught.value.message.startswith("The query timed out after 30")


def test_a_statement_still_running_at_the_transport_deadline_reads_as_timed_out(monkeypatch):
    """Superset's own time limit ends no statement: it fires once the database call returns (measured on 7.0.0rc2), so
    a statement running past its limit and the transport's margin is still running when the transport gives up. Over
    the real client that reads as the statement's time limit (`QueryTimedOut`), which the runner gives the model to
    make cheaper, never as Superset being unavailable, which would tell the person to try again."""
    import mnemiq.superset.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "TRANSPORT_MARGIN_S", 0.5)

    def slow(request, authorization):
        time.sleep(3)
        return REGION_COUNTS

    with FakeSuperset() as fake:
        fake.reply("execute_sql", slow)
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=7, schema="sales")
        with pytest.raises(QueryTimedOut):
            adapter.execute_arrow(SQL, timeout_s=1)
        with pytest.raises(ExecutionError) as caught:
            run(adapter, SQL, timeout_s=1)
    assert type(caught.value) is ExecutionError
    assert caught.value.message.startswith("The query timed out after 1")
    assert adapter.terminal_refusal is None


@pytest.mark.parametrize("cut", [False, True], ids=["stream ended", "connection cut"])
def test_a_response_that_ends_before_its_result_reads_as_an_outage(monkeypatch, cut):
    """Superset's response to the statement ends without its result (a worker restarted, a proxy cut the connection):
    that is Superset not answering, which the runner reads as the source unavailable, never as the statement's time
    limit. mcp 2.x reports the closed response at once; 1.x waits out the transport's deadline, and that deadline must
    not read as a statement still running, which would have the model make a statement cheaper that never ran long."""
    import mnemiq.superset.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "TRANSPORT_MARGIN_S", 0.5)
    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: REGION_COUNTS)
        fake.close_stream_before_result(cut=cut)
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=7, schema="sales")
        with pytest.raises(SupersetUnavailable):
            adapter.execute_arrow(SQL, timeout_s=1)
        with pytest.raises(SourceUnavailableError):
            run(adapter, SQL, timeout_s=1)
    assert adapter.terminal_refusal is None


def test_stopped_reads_as_cancelled():
    with pytest.raises(QueryCancelledError):
        run(_adapter(_failed("stopped")), SQL)


@pytest.mark.parametrize("answer", [
    SupersetUnavailable("Superset's MCP service could not be reached", detail="ConnectError: refused"),
    _failed("OAUTH2_REDIRECT", "You don't have permission to access the data. Please authenticate."),
])
def test_unavailable_reads_as_outage(answer):
    with pytest.raises(SourceUnavailableError):
        run(_adapter(answer), SQL)


@pytest.mark.parametrize("refusal", [
    SupersetRefusedCaller("Superset did not accept the credential this request carries (HTTP 401)"),
    SupersetPermissionDenied("Superset's roles for this user do not permit execute_sql"),
    SupersetDenied("Superset refused execute_sql and did not say why"),
])
def test_refused_caller_ends_the_turn(refusal):
    """A token Superset stops accepting mid-ask is the end of the turn, never a statement for the model to repair:
    every repair would carry the same token and be refused again."""
    with pytest.raises(SourceUnavailableError) as caught:
        run(_adapter(refusal), SQL)
    assert caught.value.source_detail == str(refusal)


def test_access_refusal_text_never_leaves_the_log(caplog):
    """Superset's text can name a table the person asking cannot see; it goes to the log and nowhere else."""
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"), pytest.raises(ExecutionError) as caught:
        run(_adapter(HIDDEN_TABLE_REFUSAL), SQL)
    assert "secret_table" in caplog.text
    exc = caught.value
    assert not isinstance(exc, SourceUnavailableError)
    assert exc.source_detail == "Superset refused access to something this query reads."
    for shown in (exc.message, exc.source_detail, exc.repair_text, str(exc)):
        assert "secret_table" not in shown


def test_runs_through_the_real_client():
    """The adapter over `SupersetMcp`, against the fake Superset: the request reaches Superset as the adapter built it,
    carrying the caller's bearer, and its payload reads as the table."""
    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: REGION_COUNTS)
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=7, schema="sales")
        assert adapter.execute(SQL) == [("west", 16666)]
    assert fake.calls[0].request == {"database_id": 7, "sql": SQL, "schema": "sales", "timeout": 30}
    assert fake.calls[0].authorization == "Bearer tok"


@pytest.mark.parametrize("text", [
    "Permission denied: can_execute_sql_query on SQLLab for user west (tool: execute_sql)",
    "Permission denied for execute_sql: You don't have access to this resource.",
], ids=["role denial", "generic denial"])
def test_a_refusal_from_the_wire_ends_the_turn(text):
    """A refusal as Superset sends it, after the `Error: ` it puts before every error, reaches the runner over the real
    client as the end of the turn, never as a statement for the model to repair."""
    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: PermissionError(text))
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=7, schema="sales")
        with pytest.raises(SourceUnavailableError):
            run(adapter, SQL)


def _answer_with_no_rows() -> dict:
    return {**REGION_COUNTS, "rows": [], "row_count": 0}


TERMINAL = {
    "refused-caller": lambda: SupersetRefusedCaller("Superset did not accept the credential this request carries "
                                                    "(HTTP 401)"),
    "permission-denied": lambda: SupersetPermissionDenied("Superset's roles for this user do not permit execute_sql"),
    "denied": lambda: SupersetDenied("Superset refused execute_sql and did not say why"),
}


@pytest.mark.parametrize("make", TERMINAL.values(), ids=TERMINAL.keys())
def test_the_adapter_remembers_the_first_terminal_refusal(make):
    """The runner reads a refusal of the caller, or of the caller's roles, or one Superset does not explain, as a
    source that did not answer, and the function inventory as one it could not ask; the refusal's kind survives on
    the adapter, so the request can end as that refusal. Every call is watched, a dry run, a run and the catalogue
    included, and the first refusal is the one kept."""
    first = make()
    adapter = _adapter(first)
    assert adapter.terminal_refusal is None
    with pytest.raises(type(first)):
        adapter.dry_run(SQL)
    adapter.client.answer = make()
    with pytest.raises(SourceUnavailableError):
        run(adapter, SQL)
    with pytest.raises(type(first)):
        adapter.user_functions()
    assert adapter.terminal_refusal is first


@pytest.mark.parametrize("before", ["permission-denied", "denied"])
def test_a_refused_caller_outranks_an_earlier_terminal_refusal(before):
    """Signing in again is the one thing the person can do that might help, so a refusal of the caller met at any
    point is the one kept, over another terminal refusal met before it; nothing met after it replaces it."""
    adapter = _adapter(TERMINAL[before]())
    with pytest.raises(Exception):
        adapter.execute(SQL)
    caller = TERMINAL["refused-caller"]()
    adapter.client.answer = caller
    with pytest.raises(SupersetRefusedCaller):
        adapter.dry_run(SQL)
    adapter.client.answer = TERMINAL[before]()
    with pytest.raises(Exception):
        adapter.validate(SQL)
    assert adapter.terminal_refusal is caller


@pytest.mark.parametrize("answer", [
    SupersetUnavailable("Superset's MCP service could not be reached"),
    SupersetToolError("Superset's execute_sql tool failed"),
    SupersetResponseTooLarge("Superset's answer to execute_sql was larger than its MCP service will send"),
    HIDDEN_TABLE_REFUSAL,
    _failed("timed_out"),
    _failed("stopped"),
    _failed("GENERIC_DB_ENGINE_ERROR"),
], ids=["unavailable", "tool-error", "too-large", "access-refused", "timed-out", "stopped", "statement-failed"])
def test_no_other_failure_is_remembered(answer):
    """A table Superset refuses stays the model's to repair, and an outage, a tool error, a cut result, a time limit
    and a cancel keep the readings they have: none of them is a refusal of the caller."""
    adapter = _adapter(answer)
    for call in (adapter.execute, adapter.dry_run, adapter.validate):
        with pytest.raises(Exception):
            call(SQL)
    with pytest.raises(Exception):
        adapter.user_functions()
    assert adapter.terminal_refusal is None

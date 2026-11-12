"""Superset mode's history: no row a client sends back with a turn reaches the prompt, and the newest turn's SQL is
decided again under the person's current grants and run again as the person.

A follow-up such as "and how many claims does it have?" needs the rows the earlier turn returned: its SQL computes "it"
without naming it. The rows a client sends back may be another person's, or older than what this person may read now,
so none is kept. The newest turn with SQL is decided as a new question's SQL is decided, run through the request's own
adapter, and the rows it returns now are the ones the prompt carries. Every way that fails leaves the turn without
rows, and what Superset said goes to the log, never into the history.

The tests of `superset_history` itself run it against `FakeSuperset`, a real MCP server on loopback. The runtime tests
run the scripted Superset-mode deployment `tests/test_superset_runtime.py` builds.
"""

from __future__ import annotations

import json
import logging
import os
import re

import pytest

import mnemiq.generate.plan_query as plan_query_module
import mnemiq.runtime as runtime_module
import mnemiq.superset.history as history_module
from mnemiq.authz.grants import UNAVAILABLE, GrantSet
from mnemiq.contract import Column, HistoryTurn, Snapshot
from mnemiq.runtime import build_runtime
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetRefusedCaller
from mnemiq.superset.history import superset_history
from tests.scripted_deployment import IDENTITY, SQL, retrieve_the_one_table
from tests.superset_fake import FakeSuperset
from tests.sweep.stub_model import StubModel
from tests.test_superset_runtime import superset_deployment

TOKEN = "tok-1"
DATABASE_ID = 7
SCHEMA = "public"
# The person may read `policy`, and not `claim`.
GRANTS = GrantSet(frozenset({"policy"}))
SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="policy.region", object_id="policy", name="region", data_type="VARCHAR"),
    Column(id="policy.premium", object_id="policy", name="premium", data_type="INTEGER"),
    Column(id="claim.region", object_id="claim", name="region", data_type="VARCHAR"),
    Column(id="claim.amount", object_id="claim", name="amount", data_type="INTEGER"),
])
TOP_REGION = "SELECT region, SUM(premium) AS total FROM policy GROUP BY region ORDER BY total DESC NULLS LAST LIMIT 1"
# Text only Superset writes in these tests, so where it ends up can be read.
SAID = "superset-said-this"
# What `execute_sql` answers a catalogue statement and the proof's probe here: no rows.
EMPTY = {"success": True, "columns": [], "rows": []}
# The proof seam's probe: the statement as a subquery that returns no rows (`SupersetSqlAdapter.validate`).
_PROBE = re.compile(r"AS mnemiq_probe WHERE 1 = 0$")
# The keys a data statement on the ask path sends: no `limit`, no `template_params`, no `force_refresh`.
_ASK_PATH_KEYS = {"database_id", "sql", "schema", "timeout"}


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        yield served


def _result(*rows, columns=(("region", "object"), ("total", "int64"))) -> dict:
    """A successful `execute_sql` payload holding `rows`, each a tuple in the order of `columns`."""
    names = [name for name, _ in columns]
    return {"success": True, "columns": [{"name": name, "type": dtype} for name, dtype in columns],
            "rows": [dict(zip(names, row)) for row in rows]}


def _serve(fake: FakeSuperset, rerun) -> None:
    """`execute_sql` answering the decision's own statements as Superset does: an empty catalogue (every catalogue
    statement sends `force_refresh`) and the proof's probe with no rows. Any other statement is the turn run again,
    answered by `rerun`: a payload, an exception, or a function of the request returning either."""
    def answer(request: dict, authorization: str | None):
        if request.get("force_refresh") or _PROBE.search(request["sql"]):
            return EMPTY
        return rerun(request) if callable(rerun) else rerun

    fake.reply("execute_sql", answer)


def _reruns(fake: FakeSuperset) -> list:
    """The calls that ran a statement itself: not a dry run, a catalogue read or the proof's probe."""
    return [call for call in fake.calls if call.tool == "execute_sql" and not call.request.get("dry_run")
            and not call.request.get("force_refresh") and not _PROBE.search(call.request["sql"])]


def _adapter(fake: FakeSuperset) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken(TOKEN)), database_id=DATABASE_ID, schema=SCHEMA)


def _turn(sql: str = TOP_REGION, rows=(("east", 16667),), *, question: str = "which region sold most",
          columns=("region", "total"), fingerprint: str = GRANTS.fingerprint) -> HistoryTurn:
    return HistoryTurn(question=question, sql=sql, tables_used=["policy"], columns=list(columns),
                       rows=[list(row) for row in rows], grant_fingerprint=fingerprint)


def _history(turns, adapter, *, grants: GrantSet = GRANTS, timeout_s: float = 30.0) -> list[HistoryTurn]:
    return superset_history(turns, grants=grants, snapshot=SNAPSHOT, adapter=adapter, timeout_s=timeout_s,
                            values=None, guard_fanout=False)


# --- the newest turn, run again -----------------------------------------------------------------------------------------


def test_the_newest_turn_is_run_again_as_the_person(fake):
    """The turn keeps the rows Superset returns now, under the column names the result has, and none of the client's.
    The statement run is the one the decider approved, sent as an ask-path statement with the request's token."""
    _serve(fake, _result(("west", 16666), columns=(("top_region", "object"), ("total", "int64"))))
    [turn] = _history([_turn()], _adapter(fake))
    assert turn.rows == [["west", 16666]] and turn.columns == ["top_region", "total"]
    assert (turn.question, turn.sql) == ("which region sold most", TOP_REGION)
    [rerun] = _reruns(fake)
    assert set(rerun.request) == _ASK_PATH_KEYS
    assert (rerun.request["database_id"], rerun.request["schema"], rerun.request["timeout"]) == (DATABASE_ID, SCHEMA,
                                                                                                 30)
    assert "SUM(premium)" in rerun.request["sql"]
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}


def test_the_decided_statement_is_what_runs(fake):
    """The client's SQL is never sent as it arrived: what runs is the statement the decider approved, which here carries
    the row cap the decider adds to a statement without one."""
    _serve(fake, _result(("west",), columns=(("region", "object"),)))
    [turn] = _history([_turn("SELECT region FROM policy", rows=(("east",),), columns=("region",))], _adapter(fake))
    assert turn.rows == [["west"]]
    [rerun] = _reruns(fake)
    assert rerun.request["sql"] != "SELECT region FROM policy" and "LIMIT" in rerun.request["sql"]


def test_the_rerun_runs_within_the_timeout_it_is_given(fake):
    _serve(fake, _result(("west", 16666)))
    _history([_turn()], _adapter(fake), timeout_s=12.5)
    [rerun] = _reruns(fake)
    assert rerun.request["timeout"] == 13  # whole seconds, rounded up, as the adapter sends every limit


def test_the_rerun_is_bounded_as_history_is(fake):
    """The first five rows of the first eight columns, the columns named as the result names them."""
    columns = tuple((f"c{i}", "int64") for i in range(10))
    _serve(fake, _result(*[tuple(range(r * 10, r * 10 + 10)) for r in range(7)], columns=columns))
    [turn] = _history([_turn()], _adapter(fake))
    assert turn.columns == [f"c{i}" for i in range(8)]
    assert turn.rows == [list(range(r * 10, r * 10 + 8)) for r in range(5)]


def test_older_turns_keep_no_rows(fake):
    _serve(fake, _result(("west", 16666)))
    older = _turn("SELECT region FROM policy", rows=(("north",),), question="which regions are there",
                  columns=("region",))
    kept = _history([older, _turn()], _adapter(fake))
    assert [(turn.question, turn.rows, turn.columns) for turn in kept] == [
        ("which regions are there", [], []),
        ("which region sold most", [["west", 16666]], ["region", "total"]),
    ]
    [rerun] = _reruns(fake)  # only the newest turn ran
    assert "SUM(premium)" in rerun.request["sql"]


def test_the_newest_turn_with_sql_is_the_one_run(fake):
    """A newer turn with no SQL (one the engine declined) is not the one "it" points back to; the newest turn with SQL
    is, and the newer one keeps no rows of its own."""
    _serve(fake, _result(("west", 16666)))
    declined = _turn("", rows=(("client-row",),), question="what is the weather", columns=("x",))
    kept = _history([_turn(), declined], _adapter(fake))
    assert [(turn.rows, turn.columns) for turn in kept] == [([["west", 16666]], ["region", "total"]), ([], [])]
    assert len(_reruns(fake)) == 1


def test_a_turn_from_another_boundary_is_dropped_before_anything_runs(fake):
    """Scoped first, as direct mode scopes history: a turn answered under other grants is dropped whole, so even as the
    newest turn with SQL it is never run."""
    _serve(fake, _result(("west", 16666)))
    other = _turn("SELECT region FROM policy", rows=(("north",),), question="another boundary's turn",
                  columns=("region",), fingerprint="another-boundary")
    kept = _history([_turn(), other], _adapter(fake))
    assert [turn.question for turn in kept] == ["which region sold most"]
    [rerun] = _reruns(fake)
    assert "SUM(premium)" in rerun.request["sql"]


def test_no_history_asks_superset_nothing(fake):
    _serve(fake, _result(("west", 16666)))
    assert _history(None, _adapter(fake)) == []
    assert _history([], _adapter(fake)) == []
    assert _history([_turn("", question="no sql")], _adapter(fake))[0].rows == []
    assert fake.calls == []


# --- what is not run ----------------------------------------------------------------------------------------------------


def test_a_turn_outside_current_grants_is_not_run(fake):
    """Its SQL names `claim`, which this person may not read: the decider refuses it before any call."""
    _serve(fake, _result(("west", 16666)))
    [turn] = _history([_turn("SELECT region, SUM(amount) AS total FROM claim GROUP BY region")], _adapter(fake))
    assert turn.rows == [] and turn.columns == []
    assert fake.calls == []


def test_a_write_in_history_is_never_run(fake):
    _serve(fake, _result(("west", 16666)))
    [turn] = _history([_turn("DELETE FROM policy")], _adapter(fake))
    assert turn.rows == []
    assert fake.calls == []


def test_a_template_marker_in_history_is_never_sent(fake):
    """Superset renders every statement as a template before it checks access, so a turn's SQL holding a marker is
    refused at the proof without being sent, and the turn keeps no rows."""
    _serve(fake, _result(("west",), columns=(("region", "object"),)))
    [turn] = _history([_turn("SELECT region FROM policy WHERE region = '{{ current_username() }}'",
                             columns=("region",))], _adapter(fake))
    assert turn.rows == []
    assert not any("{{" in call.request["sql"] for call in fake.calls)
    assert _reruns(fake) == []


def test_nothing_runs_when_nothing_is_granted(fake):
    """With no table granted, a new question is deferred before any statement is decided, and a turn is not run
    either: not even a statement that reads no table."""
    _serve(fake, _result(("east",), columns=(("region", "object"),)))
    turn = _turn("SELECT 'east' AS region", columns=("region",), fingerprint=UNAVAILABLE.fingerprint)
    [kept] = _history([turn], _adapter(fake), grants=UNAVAILABLE)
    assert kept.rows == []
    assert fake.calls == []


# --- failing closed -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rerun", [
    {**_result(("west", 16666)), "_response_truncated": True},
    RuntimeError(f"Response too large: the answer is larger than the response size allows ({SAID})"),
], ids=["cut-to-size", "too-large-to-send"])
def test_a_cut_rerun_leaves_no_rows(fake, rerun):
    """Part of a result reads as all of it, so a result Superset cut, or would not send at all, gives none."""
    _serve(fake, rerun)
    [turn] = _history([_turn()], _adapter(fake))
    assert turn.rows == [] and turn.columns == []
    assert SAID not in repr(turn)
    assert len(_reruns(fake)) == 1


@pytest.mark.parametrize("rerun, logged", [
    ({"success": False, "rows": None, "columns": None, "error": SAID, "error_type": "timed_out"}, False),
    ({"success": False, "rows": None, "columns": None, "error": SAID, "error_type": "stopped"}, False),
    ({"success": False, "rows": None, "columns": None, "error": f"You need access to `policy` ({SAID})",
      "error_type": "TABLE_SECURITY_ACCESS_ERROR"}, True),
    ({"success": False, "rows": None, "columns": None, "error": f"relation does not exist ({SAID})",
      "error_type": "GENERIC_DB_ENGINE_ERROR"}, True),
    ({"success": False, "rows": None, "columns": None, "error": SAID, "error_type": "OAUTH2_REDIRECT"}, True),
    (RuntimeError(f"Permission denied: can_execute_sql on SQLLab ({SAID})"), True),
    (RuntimeError(f"Permission denied for execute_sql: {SAID}"), True),
    (RuntimeError(f"something went wrong ({SAID})"), True),
], ids=["timed-out", "stopped", "access-refused", "statement-failed", "unavailable", "roles-deny", "denied",
        "tool-error"])
def test_every_failed_rerun_leaves_no_rows(fake, caplog, rerun, logged):
    """Each failure the runner reports, a source that did not answer and every refusal but the caller's included,
    leaves the turn without rows and the request going on. Superset's words reach the log where it sent any, and the
    history never."""
    _serve(fake, rerun)
    with caplog.at_level(logging.WARNING):
        [turn] = _history([_turn()], _adapter(fake))
    assert turn.rows == [] and turn.columns == []
    assert SAID not in repr(turn)
    assert (SAID in caplog.text) is logged
    assert len(_reruns(fake)) == 1


def test_a_refused_caller_at_the_rerun_ends_the_request(fake):
    """A credential Superset does not accept stays refused for every later call of the request, so the history does not
    swallow it into a turn without rows: it ends the request, as it does when the grants meet it."""
    _serve(fake, RuntimeError("User 'u' not found in Superset database"))
    with pytest.raises(SupersetRefusedCaller):
        _history([_turn()], _adapter(fake))
    assert len(_reruns(fake)) == 1


def test_a_refused_caller_while_deciding_ends_the_request(fake):
    """The decider reads a refused credential as a source that could not be asked, and refuses: the refusal of the
    caller still ends the request, and nothing is run."""
    _serve(fake, _result(("west", 16666)))
    fake.refuse_token(TOKEN)
    with pytest.raises(SupersetRefusedCaller):
        _history([_turn()], _adapter(fake))
    assert _reruns(fake) == []


# --- at the runtime -----------------------------------------------------------------------------------------------------

# The runtime's history statement, over the scripted deployment's one table.
WIDE_TOP = "SELECT col_0 AS region, COUNT(*) AS total FROM wide GROUP BY col_0 ORDER BY total DESC NULLS LAST LIMIT 1"


@pytest.fixture
def model():
    """The scripted model, answering the writer with SQL over the one table."""
    with StubModel(sql=SQL) as stub:
        yield stub
    assert not stub.all_errors()


@pytest.fixture
def runtime(tmp_path, model, fake, monkeypatch):
    retrieve_the_one_table(monkeypatch)
    return build_runtime(superset_deployment(tmp_path, model.url, fake.url))


def _serve_runtime(fake: FakeSuperset, history) -> None:
    """`execute_sql` as Superset answers it for a person who may read `wide`: a dry run accepted, an empty catalogue,
    the probe with no rows, the answer statement's one row, and the history statement answered by `history`, a
    function of how many times it has run."""
    runs = []

    def answer(request: dict, authorization: str | None):
        if request.get("dry_run"):
            return {"success": True, "rows": None, "columns": None, "error": None, "error_type": None}
        if request.get("force_refresh") or _PROBE.search(request["sql"]):
            return EMPTY
        if "AS region" in request["sql"]:
            runs.append(request)
            return history(len(runs))
        return _result((1,), columns=(("n", "int64"),))

    fake.reply("execute_sql", answer)


def _packets(monkeypatch) -> list:
    """Each packet the runtime's retrieval returns, which it then gives the history and hands to the agent."""
    seen = []
    scripted = runtime_module.retrieve

    def recording(*args, **kwargs):
        packet = scripted(*args, **kwargs)
        seen.append(packet)
        return packet

    monkeypatch.setattr(runtime_module, "retrieve", recording)
    return seen


def test_client_rows_never_reach_the_prompt(runtime, fake, model, monkeypatch):
    _serve_runtime(fake, lambda run: _result(("west", 16666)))
    first = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    packets = _packets(monkeypatch)
    turn = HistoryTurn(question="which region sold most", sql=WIDE_TOP, columns=["region", "total"],
                       rows=[["east", 16667]], grant_fingerprint=first.grant_fingerprint)
    answer = runtime.ask("and how many claims does it have?", IDENTITY, history=[turn],
                         delegation=DelegatedToken(TOKEN))
    assert not answer.failed and not answer.deferred, answer.answer
    [packet] = packets
    assert [(t.question, t.columns, t.rows) for t in packet.history] == [
        ("which region sold most", ["region", "total"], [["west", 16666]])]
    prompts = [json.dumps(request) for request in model.requests]
    assert any("16666" in prompt for prompt in prompts)
    assert not any("16667" in prompt for prompt in prompts)
    # Run as the person, as an ask-path statement, within the time limit the answer statement itself runs under.
    history_run, answer_run = _reruns(fake)[-2:]
    assert "AS region" in history_run.request["sql"] and "AS region" not in answer_run.request["sql"]
    assert set(history_run.request) == set(answer_run.request) == _ASK_PATH_KEYS
    assert history_run.request["timeout"] == answer_run.request["timeout"]
    assert history_run.authorization == answer_run.authorization == f"Bearer {TOKEN}"


def test_narrowed_access_shows_on_the_next_follow_up(runtime, fake, monkeypatch):
    """Superset's row rule narrows between two follow-ups: the first is reminded of two rows, the second of the one the
    person may still read, though the client sent the same turn both times."""
    _serve_runtime(fake, lambda run: _result(("west", 16666), ("east", 16667)) if run == 1
                   else _result(("west", 16666)))
    first = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    packets = _packets(monkeypatch)
    turn = HistoryTurn(question="which regions sold", sql=WIDE_TOP, columns=["region", "total"],
                       rows=[["west", 16666], ["east", 16667]], grant_fingerprint=first.grant_fingerprint)
    for _ in range(2):
        runtime.ask("and how many claims does each have?", IDENTITY, history=[turn], delegation=DelegatedToken(TOKEN))
    assert [packet.history[0].rows for packet in packets] == [[["west", 16666], ["east", 16667]], [["west", 16666]]]


def test_a_refused_caller_during_history_ends_the_request_before_the_model_is_asked(runtime, fake, model):
    """The token Superset accepted for the grants is refused when the turn runs again. The request ends there, as it
    does when the grants meet the same refusal; the model is not asked to answer without the rows."""
    _serve_runtime(fake, lambda run: _result(("west", 16666)))
    first = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
    asked = model.calls
    _serve_runtime(fake, lambda run: RuntimeError("User 'u' not found in Superset database"))
    turn = HistoryTurn(question="which region sold most", sql=WIDE_TOP, columns=["region", "total"],
                       rows=[["east", 16667]], grant_fingerprint=first.grant_fingerprint)
    with pytest.raises(SupersetRefusedCaller):
        runtime.ask("and how many claims does it have?", IDENTITY, history=[turn], delegation=DelegatedToken(TOKEN))
    assert model.calls == asked


def test_history_is_decided_with_what_a_new_question_is_decided_with(tmp_path, fake, monkeypatch):
    """The newest turn's SQL is decided with every argument the answer path decides the writer's SQL with: the same
    schema, access policy, views, key facts, types, value index, dialects, and the request's own adapter. With the
    fan-out guard on, a history that left the key facts out would be looser than a new question, and this shows it."""
    retrieve_the_one_table(monkeypatch)
    _serve_runtime(fake, lambda run: _result(("west", 16666)))
    seen: dict[str, list] = {"answer": [], "history": []}

    def recording(name: str, real):
        def decide(sql, visible, **kwargs):
            seen[name].append((sql, visible, kwargs))
            return real(sql, visible, **kwargs)
        return decide

    monkeypatch.setattr(plan_query_module, "decide", recording("answer", plan_query_module.decide))
    monkeypatch.setattr(history_module, "decide", recording("history", history_module.decide))
    with StubModel(sql=SQL) as model:
        runtime = build_runtime(superset_deployment(tmp_path, model.url, fake.url, guard_fanout=True))
        first = runtime.ask("q", IDENTITY, delegation=DelegatedToken(TOKEN))
        seen["answer"].clear()
        turn = HistoryTurn(question="how many", sql=SQL, grant_fingerprint=first.grant_fingerprint)
        runtime.ask("and again?", IDENTITY, history=[turn], delegation=DelegatedToken(TOKEN))
        assert not model.all_errors()
    (answer_sql, answer_visible, answer_kwargs), = seen["answer"]
    (history_sql, history_visible, history_kwargs), = seen["history"]
    assert history_sql == answer_sql == SQL and history_visible == answer_visible
    assert set(history_kwargs) == set(answer_kwargs)
    # The request's own adapter, which carries the request's token: read through, never another.
    assert history_kwargs.pop("adapter").client is answer_kwargs.pop("adapter").client
    assert history_kwargs["values"] is answer_kwargs["values"]
    assert answer_kwargs["keys"] is not None  # the guard is on, so both decisions read key facts
    assert history_kwargs["views"].available == answer_kwargs["views"].available
    assert history_kwargs == answer_kwargs

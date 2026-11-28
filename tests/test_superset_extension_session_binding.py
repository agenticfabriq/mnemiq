"""The extension's `session_binding` (`agenticfabriq.mnemiq.reads`), with Superset's lookups and the database's
connection stood in, and mnemiq's reading of its answer (`mnemiq.superset.extension.session_binding`).

The tool answers, as the caller, whether the session Superset opens on a database for them, as `execute_sql` opens it,
logs in as a schema, keeps it as its current schema and reached a service: one fixed statement on that connection, and
an answer that is only whether, never a value read. What it does inside Superset is the live suite's
(`tests/superset_live/test_extension_tools.py`).
"""

from __future__ import annotations

import ast
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemiq.superset.client import SupersetPermissionDenied, SupersetToolError, SupersetUnavailable
from mnemiq.superset.extension import EXTENSION_TOOLS, TOOL_PREFIX, session_binding
from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import reads, writes  # noqa: E402

# The statement the tool runs, word for word.
SESSION_SQL = "SELECT USER, SYS_CONTEXT('USERENV','SERVICE_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL"
REQUEST = {"database_id": 7, "service": "freepdb1", "schema": "SHOP"}
BOUND = ("SHOP", "freepdb1", "SHOP")


def refusal(request):
    with pytest.raises(writes.Refused) as caught:
        reads.read("session_binding", request)
    return caught.value.answer


class Cursor:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.closed = False

    def execute(self, sql) -> None:
        self.connection.asked.append(("execute", sql))
        if isinstance(self.connection.rows, Exception):
            raise self.connection.rows

    def fetchall(self):
        return list(self.connection.rows)

    def close(self) -> None:
        self.closed = True


class Database:
    """A database as rc2's `Database` model offers what the tool reads: its `backend`, its dialect's `normalize_name`
    (SQLAlchemy's Oracle reading: a name stored in upper case is shown in lower case), its default catalog, its engine
    spec's `execute`, and `get_raw_connection`, the connection the executor opens. `rows` is what the statement
    answers (an exception is raised by it), `connect` an exception raised in place of a connection."""

    def __init__(self, asked: list, *, backend="oracle", rows=(BOUND,), connect: Exception | None = None,
                 catalog=None) -> None:
        self.id = 7
        self.backend = backend
        self.asked = asked
        self.rows = rows
        self.connect = connect
        self.catalog = catalog
        self.cursors: list[Cursor] = []
        self.db_engine_spec = SimpleNamespace(execute=self._execute)

    def _execute(self, cursor, query, database) -> None:
        assert database is self
        cursor.execute(query)

    def get_dialect(self):
        return SimpleNamespace(normalize_name=lambda name: name.lower() if name.isupper() else name)

    def get_default_catalog(self):
        return self.catalog

    @contextmanager
    def get_raw_connection(self, catalog=None, schema=None, nullpool=True, source=None):
        self.asked.append(("connect", catalog, schema))
        if self.connect is not None:
            raise self.connect
        connection = SimpleNamespace(cursor=lambda: self._cursor())
        yield connection
        self.asked.append(("closed",))

    def _cursor(self) -> Cursor:
        cursor = Cursor(self)
        self.cursors.append(cursor)
        return cursor


def superset_binding(monkeypatch, *, databases: dict) -> list:
    """Superset's lookup `session_binding` makes, stood in: `DatabaseDAO.find_by_id` answering the databases the caller
    may see (`databases`, by id; each a `Database` sharing the returned record of what was asked)."""
    asked: list = []

    def find_by_id(database_id, *args, **options):
        asked.append(("database", database_id, *args, *options.items()))
        found = databases.get(database_id)
        return found(asked) if callable(found) else found

    modules = {"superset": {}, "superset.daos": {}, "superset.daos.database": {"DatabaseDAO": SimpleNamespace(
        find_by_id=find_by_id)}}
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return asked


# --- the request ------------------------------------------------------------------------------------------------------


def test_the_tool_is_registered_and_named_as_mnemiq_calls_it():
    """Registered as the other read tools are, under the name mnemiq calls, gated as `table_rules` is (`can_read` on
    Dataset), and handing its request to `reads.session_binding`. `tools.py` imports Superset's decorator, which is not
    importable here, so its source is read."""
    assert "session_binding" in EXTENSION_TOOLS and "session_binding" in reads.REQUESTS
    source = (Path(reads.__file__).parent / "tools.py").read_text()
    registered = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            [decorator] = node.decorator_list
            registered[node.name] = ({keyword.arg: keyword.value.value for keyword in decorator.keywords
                                      if keyword.arg != "description"}, ast.unparse(node.body[-1]))
    gate = {key: value for key, value in registered["table_rules"][0].items() if key != "name"}
    assert registered["session_binding"] == ({"name": "session_binding", **gate},
                                             "return reads.session_binding(request)")


def test_the_request_is_a_database_a_service_and_a_schema_and_no_sql():
    assert reads.read("session_binding", REQUEST) == REQUEST
    for bad in ("", "  ", 3, None, "x" * 256):
        assert refusal({**REQUEST, "schema": bad}) == {"refused": "request", "field": "schema"}
        assert refusal({**REQUEST, "service": bad}) == {"refused": "request", "field": "service"}
    for bad in (0, "7", True, None):
        assert refusal({**REQUEST, "database_id": bad}) == {"refused": "request", "field": "database_id"}
    for field in REQUEST:
        assert refusal({key: value for key, value in REQUEST.items() if key != field}) == {"refused": "request",
                                                                                        "field": field}
    assert refusal({**REQUEST, "sql": "SELECT 1 FROM DUAL"}) == {"refused": "request", "unknown": ["sql"]}
    assert refusal({}) == {"refused": "request", "field": "database_id"}


def test_a_bad_request_is_answered_not_raised():
    assert reads.session_binding({**REQUEST, "schema": ""}) == {"refused": "request", "field": "schema"}


# --- the answer -------------------------------------------------------------------------------------------------------


def test_a_session_that_logs_in_as_the_schema_keeps_it_and_reached_the_service_is_bound(monkeypatch):
    """The one statement, word for word, on the connection the executor opens for the caller: the database's default
    catalog, and the schema as Superset shows it (`normalize_name`), as `execute_sql` opens a connection for a
    statement sent with that schema. The cursor is closed, and so is the connection."""
    made: list = []
    asked = superset_binding(monkeypatch, databases={7: lambda record: made.append(Database(record)) or made[-1]})
    assert reads.session_binding(REQUEST) == {"bound": True}
    assert asked == [("database", 7), ("connect", None, "shop"), ("execute", SESSION_SQL), ("closed",)]
    assert [cursor.closed for cursor in made[0].cursors] == [True]
    assert reads.SESSION_SQL == SESSION_SQL


@pytest.mark.parametrize("row", [
    ("APP", "freepdb1", "SHOP"), ("SHOP", "otherpdb", "SHOP"), ("SHOP", "freepdb1", "APP"),
    ("shop", "freepdb1", "shop"), ("SHOP", "FREEPDB1", "SHOP"), (None, "freepdb1", "SHOP"),
], ids=["another-login", "another-service", "another-current-schema", "the-shown-spelling", "the-service-cased",
        "no-login"])
def test_each_mismatch_is_not_bound_and_no_value_read_is_answered(monkeypatch, row):
    """Each value is compared exactly, as Oracle stores it, with the one asked; the answer says only whether."""
    superset_binding(monkeypatch, databases={7: lambda record: Database(record, rows=[row])})
    answer = reads.session_binding(REQUEST)
    assert answer == {"bound": False}
    for value in row:
        assert value is None or json.dumps(value) not in json.dumps(answer)


def test_a_database_that_is_not_oracle_is_refused_and_nothing_is_run(monkeypatch):
    asked = superset_binding(monkeypatch, databases={7: lambda record: Database(record, backend="postgresql")})
    assert reads.session_binding(REQUEST) == {"refused": "backend"}
    assert asked == [("database", 7)]


def test_a_database_the_caller_cannot_see_is_not_found_and_nothing_is_run(monkeypatch):
    """A database the caller may not see (Superset's own `DatabaseDAO` filter, as `table_rules` finds one) and one that
    does not exist are answered alike."""
    asked = superset_binding(monkeypatch, databases={})
    assert reads.session_binding(REQUEST) == {"not_found": "database"}
    assert asked == [("database", 7)]


@pytest.mark.parametrize("database", [
    {"connect": RuntimeError("ORA-01017: invalid credential or not authorized; logon denied")},
    {"rows": RuntimeError("ORA-00942: table or view does not exist")},
    {"rows": []}, {"rows": [BOUND, BOUND]}, {"rows": [("SHOP", "freepdb1")]},
], ids=["no-connection", "statement-failed", "no-row", "two-rows", "two-values"])
def test_a_session_that_cannot_be_read_answers_a_flag_and_no_message(monkeypatch, database):
    superset_binding(monkeypatch, databases={7: lambda record: Database(record, **database)})
    answer = reads.session_binding(REQUEST)
    assert answer == {"unread": True}


def test_the_comparison_is_the_three_values_exactly():
    assert reads.session_bound(BOUND, service="freepdb1", schema="SHOP") is True
    assert reads.session_bound(("SHOP", "freepdb1", "SHOP"), service="freepdb1", schema="shop") is False


# --- mnemiq's reading -------------------------------------------------------------------------------------------------


class Reading:
    """mnemiq's Superset client, stood in for `session_binding`: each call answered by `answer` (a function of the
    request, or an answer or exception), every call recorded."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool, request, *, timeout_s=None):
        self.calls.append((tool, json.loads(json.dumps(request))))
        found = self.answer(request) if callable(self.answer) else self.answer
        if isinstance(found, Exception):
            raise found
        return json.loads(json.dumps(found))


def read_by(client):
    return session_binding(client, database_id=7, service="freepdb1", schema="SHOP")


def test_mnemiqs_client_reads_what_the_tool_answers(monkeypatch):
    """The join of the two ends: the extension's own tool, over Superset's lookups stood in, answering mnemiq's own
    request, read by mnemiq's client."""
    superset_binding(monkeypatch, databases={7: lambda record: Database(record)})
    client = Reading(reads.session_binding)
    assert read_by(client) is True
    assert client.calls == [(TOOL_PREFIX + "session_binding", REQUEST)]
    superset_binding(monkeypatch, databases={7: lambda record: Database(record, rows=[("APP", "freepdb1", "SHOP")])})
    assert read_by(client) is False


@pytest.mark.parametrize("answer", [
    SupersetToolError("Internal error in extensions.agenticfabriq.mnemiq.session_binding: An unexpected error "
                      "occurred."),
    {"refused": "backend"}, {"not_found": "database"}, {"unread": True}, {"refused": "request", "field": "schema"},
    {"bound": "true"}, {"bound": 1}, {"bound": True, "more": True}, {"bound": True, "_response_truncated": True}, [],
], ids=["not offered", "not oracle", "database not found", "unread", "refused", "a text", "a number", "saying more",
        "cut", "not an object"])
def test_any_answer_but_bound_or_not_is_none(answer):
    """Read strictly: exactly `{"bound": <a boolean>}`. Anything else, or a call that fails (an extension older than
    the tool among them), is None, which mnemiq reads as not bound."""
    assert read_by(Reading(answer)) is None


def test_superset_not_answering_and_a_refusal_of_the_caller_are_raised():
    with pytest.raises(SupersetUnavailable):
        read_by(Reading(SupersetUnavailable("Superset did not answer")))
    with pytest.raises(SupersetPermissionDenied):
        read_by(Reading(SupersetPermissionDenied("no")))

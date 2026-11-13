"""The statement a direct Postgres source reads its declared foreign keys with is the one the Superset catalogue reads
them with, rendered for the adapter's own schema, and a read the direct adapter cannot make returns `[]` with one
WARNING naming the exception's type. No database: the adapter is a bare instance over a connection that records what it
is sent. `tests/test_pg_declared_keys.py` runs the statement against a real Postgres.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import duckdb
import pytest

from mnemiq.adapters import duckdb as duckdb_adapter
from mnemiq.adapters import pg_keys
from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.pg_keys import DECLARED_KEYS
from mnemiq.superset.pg_catalog import CATALOGUE_SQL, PAGE_ROWS, PostgresCatalogue

LOGGER = "mnemiq.adapters.duckdb"
TAG = "$q$"


class _Recording:
    """A DuckDB connection that records each statement it is sent and answers with no rows, or raises `error`."""

    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[str] = []
        self.error = error

    def execute(self, sql: str, *args):
        self.sent.append(sql)
        if self.error is not None:
            raise self.error
        return self

    def fetchall(self) -> list[tuple]:
        return []


def _postgres_adapter(schema: str, con: _Recording) -> DuckDBAdapter:
    """A direct Postgres adapter reading `schema`, as the resolver builds one, over `con`."""
    adapter = DuckDBAdapter.__new__(DuckDBAdapter)
    adapter._con = con
    adapter._catalog = "src"
    adapter._table_schema = schema
    adapter._fk_via_postgres = True
    return adapter


def _warnings(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == LOGGER and record.levelno >= logging.WARNING]


def test_the_superset_catalogue_reads_keys_with_the_shared_statement():
    assert CATALOGUE_SQL["foreign_keys"] is DECLARED_KEYS


@pytest.mark.parametrize("schema", ["public", "sales", "Sales", "it's", "x:schema"])
def test_the_direct_adapter_sends_the_statement_the_superset_catalogue_sends(schema):
    """The direct adapter sends Postgres, in its own schema, the statement the Superset catalogue sends for that schema,
    which the catalogue sends with a page size after it."""
    con = _Recording()
    assert _postgres_adapter(schema, con).foreign_keys() == []
    (sent,) = con.sent
    assert sent.count(TAG) == 2, sent
    direct = sent.split(TAG)[1]

    superset: list[str] = []
    PostgresCatalogue(lambda sql: superset.append(sql) or [], schema).foreign_keys()
    assert direct == PostgresCatalogue(lambda sql: [], schema).statement("foreign_keys")
    assert superset == [f"{direct} LIMIT {PAGE_ROWS}"]
    assert f"n.nspname = {pg_keys.quoted(schema)}" in direct


def test_a_failed_read_logs_one_warning_naming_the_type_and_returns_nothing(caplog):
    """The exception's text can carry the DSN, password and all, so the WARNING names only its type."""
    dsn = "user=reader password=hunter2 host=db.internal port=5432 dbname=warehouse"
    con = _Recording(duckdb.IOException(f'IO Error: Unable to connect to Postgres at "{dsn}"'))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("public", con).foreign_keys() == []
    (record,) = _warnings(caplog)
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "(IOException)" in message, message
    for part in ("hunter2", "db.internal", "warehouse", "Unable to connect"):
        assert part not in message, message


def test_a_schema_the_literal_cannot_carry_is_a_read_not_made(caplog):
    """A backslash in the schema is refused by the literal it is written as, so nothing is sent: `[]`, and the WARNING
    any failed read logs."""
    con = _Recording()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("sa\\les' OR true --", con).foreign_keys() == []
    assert con.sent == []
    (record,) = _warnings(caplog)
    assert "(ValueError)" in record.getMessage()


def test_a_statement_holding_the_quote_tag_is_not_sent(caplog):
    """The statement goes to `postgres_query` in a dollar-quoted string, which the quote's tag would end early. A schema
    holding the tag makes such a statement, and it is refused rather than sent. The control: a schema holding a dollar
    sign, without the tag, is sent as the statement itself."""
    con = _Recording()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter(f"a{TAG}) --", con).foreign_keys() == []
    assert con.sent == []
    (record,) = _warnings(caplog)
    assert "(ValueError)" in record.getMessage()

    caplog.clear()
    control = _Recording()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("a$b", control).foreign_keys() == []
    (sent,) = control.sent
    assert sent.split(TAG)[1] == DECLARED_KEYS.whole("a$b")
    assert _warnings(caplog) == []


def test_a_duckdb_file_reports_no_declared_keys_and_logs_nothing(tmp_path, caplog):
    """A DuckDB file's declared keys are not read, as before: `[]`, with nothing logged, since nothing failed."""
    path = tmp_path / "keys.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE b (id INTEGER, a_id INTEGER REFERENCES a (id))")
    con.close()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert DuckDBAdapter.duckdb(str(path)).foreign_keys() == []
    assert _warnings(caplog) == []


@pytest.mark.parametrize("module", [pg_keys, duckdb_adapter], ids=lambda module: module.__name__)
def test_the_direct_path_never_imports_superset(module):
    """The direct adapter and the module holding the shared statement read keys without `mnemiq.superset`, at import
    or later: no import anywhere in either names it."""
    tree = ast.parse(Path(module.__file__).read_text())
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imported, "the walk found no imports at all"
    assert [name for name in imported if name == "mnemiq.superset" or name.startswith("mnemiq.superset.")] == []

"""The statement a direct Postgres source reads its declared foreign keys with is the one the Superset catalogue reads
them with, rendered for the adapter's own schema by the name Postgres stores it under, and a read the direct adapter
cannot make returns `[]` with one WARNING: naming the exception's type, or why the schema could not be resolved. No
database: the adapter is a bare instance over a connection that records what it is sent.
`tests/test_pg_declared_keys.py` runs the statements against a real Postgres.
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
    """A DuckDB connection that records each statement it is sent, with its parameters. The read of the stored schema
    names matching the configured one is answered with `stored`, by default the configured name itself, as though a
    schema of exactly that name were stored; the key statement with no rows, or by raising `key_error`. Or every
    statement raises `error`."""

    def __init__(self, stored: tuple[str, ...] | None = None, error: Exception | None = None,
                 key_error: Exception | None = None) -> None:
        self.sent: list[str] = []
        self.params: list[list | None] = []
        self.stored = stored
        self.error = error
        self.key_error = key_error
        self._rows: list[tuple] = []

    def execute(self, sql: str, params: list | None = None):
        self.sent.append(sql)
        self.params.append(params)
        if self.error is not None:
            raise self.error
        if "pg_constraint" in sql:
            if self.key_error is not None:
                raise self.key_error
            self._rows = []
        else:
            self._rows = [(name,) for name in (self.stored if self.stored is not None else params)]
        return self

    def fetchall(self) -> list[tuple]:
        return self._rows

    def key_statements(self) -> list[str]:
        """The statement sent to Postgres inside each key read sent, which is the only statement reading
        `pg_constraint`."""
        return [sql.split(TAG)[1] for sql in self.sent if "pg_constraint" in sql]


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
    (sent,) = [sql for sql in con.sent if "pg_constraint" in sql]
    assert sent.count(TAG) == 2, sent
    (direct,) = con.key_statements()

    superset: list[str] = []
    PostgresCatalogue(lambda sql: superset.append(sql) or [], schema).foreign_keys()
    assert direct == PostgresCatalogue(lambda sql: [], schema).statement("foreign_keys")
    assert superset == [f"{direct} LIMIT {PAGE_ROWS}"]
    assert f"n.nspname = {pg_keys.quoted(schema)}" in direct


def test_the_direct_adapter_orders_the_keys_it_reads_by_the_statements_key():
    """Postgres orders the rows by the statement's key; DuckDB orders them again by the same columns, so the order a
    composite key's columns come in, and the relationships made from them, rest on neither side keeping the other's.
    The key's position is ordered by without being returned."""
    con = _Recording()
    _postgres_adapter("public", con).foreign_keys()
    (sent,) = [sql for sql in con.sent if "pg_constraint" in sql]
    head, tail = sent.rsplit(TAG, 1)
    assert head.startswith("SELECT from_table, from_column, to_table, to_column, constraint_name FROM "), sent
    assert tail == ") ORDER BY from_table, key_position, constraint_name", sent
    assert DECLARED_KEYS.key == ("src.relname", "k.key_position", "con.conname")
    assert "src.relname AS from_table" in DECLARED_KEYS.select
    assert "k.key_position AS key_position" in DECLARED_KEYS.select
    assert "con.conname AS constraint_name" in DECLARED_KEYS.select


def test_the_direct_adapter_reads_the_schema_by_its_stored_name():
    """The configured schema is matched to a stored one as tables are, without regard to case and on DuckDB's side, the
    configured name a parameter there; the key statement names the schema as Postgres stores it, as the Superset
    catalogue's statement does for that stored name."""
    con = _Recording(stored=("sales",))
    assert _postgres_adapter("SALES", con).foreign_keys() == []
    resolve, _ = con.sent
    assert "lower(nspname) = lower(?)" in resolve and con.params[0] == ["SALES"]
    assert "SALES" not in resolve
    assert con.key_statements() == [PostgresCatalogue(lambda sql: [], "sales").statement("foreign_keys")]


@pytest.mark.parametrize(("stored", "reason"), [
    ((), "no schema is stored under that name, in any letter case"),
    (("Sales", "sales"), "schemas 'Sales' and 'sales' both match it without regard to case"),
    (("SALES", "Sales", "sales"), "schemas 'SALES', 'Sales' and 'sales' all match it without regard to case"),
])
def test_a_schema_resolving_to_no_stored_schema_or_to_two_reads_nothing(stored, reason, caplog):
    """No key statement is sent: `[]`, and one WARNING saying which, naming no more than schemas."""
    con = _Recording(stored=stored)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("SALES", con).foreign_keys() == []
    assert con.key_statements() == []
    (record,) = _warnings(caplog)
    assert f"({reason})" in record.getMessage(), record.getMessage()


def test_a_failed_read_logs_one_warning_naming_the_type_and_returns_nothing(caplog):
    """The exception's text can carry the DSN, password and all, so the WARNING names only its type."""
    dsn = "user=reader password=hunter2 host=db.internal port=5432 dbname=warehouse"
    con = _Recording(error=duckdb.IOException(f'IO Error: Unable to connect to Postgres at "{dsn}"'))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("public", con).foreign_keys() == []
    (record,) = _warnings(caplog)
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "(IOException)" in message, message
    for part in ("hunter2", "db.internal", "warehouse", "Unable to connect"):
        assert part not in message, message


def test_a_failed_key_read_after_the_schema_is_found_logs_its_type_and_returns_nothing(caplog):
    """The schema resolves and the key statement is sent, and fails: `[]`, and one WARNING naming the exception's type
    only, as for a failure of the schema's own read."""
    dsn = "user=reader password=hunter2 host=db.internal port=5432 dbname=warehouse"
    con = _Recording(stored=("public",),
                     key_error=duckdb.IOException(f'IO Error: Unable to connect to Postgres at "{dsn}"'))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("public", con).foreign_keys() == []
    assert len(con.key_statements()) == 1
    (record,) = _warnings(caplog)
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "(IOException)" in message, message
    for part in ("hunter2", "db.internal", "warehouse", "Unable to connect"):
        assert part not in message, message


def test_a_schema_the_literal_cannot_carry_is_a_read_not_made(caplog):
    """A stored schema name holding a backslash is refused by the literal it is written as, so no key statement is
    sent: `[]`, and the WARNING any failed read logs."""
    con = _Recording(stored=("sa\\les' OR true --",))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("SA\\LES' OR TRUE --", con).foreign_keys() == []
    assert con.key_statements() == []
    (record,) = _warnings(caplog)
    assert "(ValueError)" in record.getMessage()


def test_a_statement_holding_the_quote_tag_is_not_sent(caplog):
    """The statement goes to `postgres_query` in a dollar-quoted string, which the quote's tag would end early. A stored
    schema name holding the tag makes such a statement, and it is refused rather than sent. The control: a schema
    holding a dollar sign, without the tag, is sent as the statement itself."""
    con = _Recording()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter(f"a{TAG}) --", con).foreign_keys() == []
    assert con.key_statements() == []
    (record,) = _warnings(caplog)
    assert "(ValueError)" in record.getMessage()

    caplog.clear()
    control = _Recording()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _postgres_adapter("a$b", control).foreign_keys() == []
    assert control.key_statements() == [DECLARED_KEYS.whole("a$b")]
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

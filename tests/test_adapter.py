import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter

pytestmark = pytest.mark.integration


def _dsn():
    return os.getenv("MNEMIQ_PG_DSN", "postgresql://mnemiq:mnemiq@localhost:5432/acme")


def test_introspect_sees_acme_tables():
    a = DuckDBPostgresAdapter(_dsn())
    tables = a.introspect()
    assert "party" in [t.lower() for t in tables]


def test_execute_counts_party():
    a = DuckDBPostgresAdapter(_dsn())
    rows = a.execute("SELECT count(*) FROM party")
    assert rows[0][0] == 30


def test_read_only_default_still_attaches_read_only():
    import pytest
    a = DuckDBPostgresAdapter(_dsn())  # default read_only=True
    with pytest.raises(Exception):  # a write against a read-only attach is rejected
        a.execute("CREATE TABLE src.public._probe_ro (x INT)")


def test_duckdb_adapter_can_attach_a_duckdb_file(tmp_path):
    """The class docstring calls DuckDB "the universal executor", and it could attach Postgres and
    SQLite and not DuckDB. A customer whose warehouse IS DuckDB had no adapter, and neither did the
    fs payments corpus."""
    import duckdb as _duckdb

    from mnemiq.adapters.duckdb import DuckDBAdapter

    path = tmp_path / "warehouse.duckdb"
    con = _duckdb.connect(str(path))
    con.execute("create table payment (id integer, amount decimal(10,2))")
    con.execute("insert into payment values (1, 10.50), (2, 20.00)")
    con.close()

    adapter = DuckDBAdapter.duckdb(str(path))

    assert "payment" in adapter.introspect()
    assert float(adapter.execute("select sum(amount) from payment")[0][0]) == 30.50

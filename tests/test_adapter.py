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

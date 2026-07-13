import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.catalog import introspect

pytestmark = pytest.mark.integration

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


def _dsn():
    return os.getenv("MNEMIQ_PG_DSN", _DSN)


def test_introspect_returns_tables_with_columns():
    tables = introspect(DuckDBPostgresAdapter(_dsn()))
    assert len(tables) == 29
    party = next(t for t in tables if t.name == "party")
    colnames = {c.name for c in party.columns}
    assert "party_type_code" in colnames  # from Party.csv header
    assert all(c.data_type for c in party.columns)

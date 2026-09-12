import os

import pytest

from acme_dsn import (
    acme_dsn,
    assert_acme_seeded,
    requires_acme,
    source_table_count,
)

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.catalog import introspect

pytestmark = [pytest.mark.integration, requires_acme]

_DSN = acme_dsn()


def _dsn():
    return os.getenv("MNEMIQ_PG_DSN", _DSN)


def test_introspect_returns_tables_with_columns():
    adapter = DuckDBPostgresAdapter(_dsn())
    tables = introspect(adapter)
    # Against the SOURCE, not a frozen number. `== 29` broke when the catalogue reached 32,
    # and a count that has to be edited whenever the schema changes fails on the one event
    # that is not a defect. This still catches introspection dropping a table.
    assert len(tables) == source_table_count(adapter)
    # And the fixture is really seeded -- the equality above would pass on a half-seeded ACME
    # agreeing with itself.
    assert_acme_seeded(adapter)
    party = next(t for t in tables if t.name == "party")
    colnames = {c.name for c in party.columns}
    assert "party_type_code" in colnames  # from Party.csv header
    assert all(c.data_type for c in party.columns)

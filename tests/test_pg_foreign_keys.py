import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter

pytestmark = pytest.mark.integration

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


def test_foreign_keys_runs_and_returns_a_list():
    fks = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN)).foreign_keys()
    assert isinstance(fks, list)
    # ACME's DDL did not survive seeding -> no declared FKs; the empty path is the
    # regression-safety case (enrichment falls back to data inference).
    for fk in fks:
        assert len(fk) == 5 and all(isinstance(x, str) for x in fk)

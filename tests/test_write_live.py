"""End-to-end write against ACME: a read-write DuckDBPostgres adapter + Runtime.write on a
scratch table. Integration-gated (needs the ACME Postgres); no LLM, so not live_llm."""

import os

import pytest

from mnemiq.adapters.duckdb import DuckDBPostgresAdapter
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.runtime import Runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no ACME Postgres configured"),
]

_TABLE = "scratch_write_test"


class _WriteAuthz:
    def __init__(self, *tables):
        self._g = GrantSet(frozenset(tables), writable=frozenset(tables))

    def grants_for(self, _identity):
        return self._g


def test_write_inserts_a_row_end_to_end():
    dsn = os.environ["MNEMIQ_PG_DSN"]
    setup = DuckDBPostgresAdapter(dsn, read_only=False)  # out-of-band DDL for the scratch table
    setup.execute(f"CREATE TABLE IF NOT EXISTS src.public.{_TABLE} (id INTEGER, note VARCHAR)")
    setup.execute(f"DELETE FROM {_TABLE}")
    try:
        snap = Snapshot(version="v1", source_id="acme", created_at="t", columns=[
            Column(id=f"{_TABLE}.id", object_id=_TABLE, name="id"),
            Column(id=f"{_TABLE}.note", object_id=_TABLE, name="note"),
        ])
        rt = Runtime(con=None, snapshot=snap, adapter=DuckDBPostgresAdapter(dsn, read_only=False),
                     agent=None, embedder=None, authz=_WriteAuthz(_TABLE), settings=None)
        identity = IdentityContext(tenant_id="t", principal_id="u", roles=["writer"])

        res = rt.write(f"INSERT INTO {_TABLE} (id, note) VALUES (1, 'hello')", identity)
        assert res.approved is True and res.target == _TABLE

        rows = setup.execute(f"SELECT count(*) FROM {_TABLE}")
        assert rows[0][0] == 1
    finally:
        setup.execute(f"DROP TABLE IF EXISTS src.public.{_TABLE}")

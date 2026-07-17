"""Live cross-source federation against ACME (Postgres) + a companion SQLite file: a
cross-catalog join executes, and DuckDB pushes the predicate into the Postgres scanner.
Integration-gated (needs the ACME Postgres); no LLM."""

import os

import pytest

from mnemiq.adapters.federated import FederatedAdapter
from mnemiq.config import SourceSpec
from tests.fixtures.build_sqlite_fixture import build

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no ACME Postgres configured"),
]


def _adapter(tmp_path):
    sqlite_path = str(tmp_path / "ext.db")
    build(sqlite_path, pids=[1, 2, 3])
    specs = [
        SourceSpec(id="acme", kind="postgres", target=os.environ["MNEMIQ_PG_DSN"],
                   catalog="pg", schema="public"),
        SourceSpec(id="ext", kind="sqlite", target=sqlite_path, catalog="ext", schema="main"),
    ]
    return FederatedAdapter(specs)


def test_cross_source_join_executes(tmp_path):
    adapter = _adapter(tmp_path)
    # person is an ACME (Postgres) table; ext_orders is the SQLite companion. One DuckDB
    # connection joins across both catalogs. person_identifier is VARCHAR, so cast the join key.
    rows = adapter.execute(
        "SELECT count(*) FROM pg.public.person p "
        "JOIN ext.main.ext_orders o ON CAST(o.pid AS VARCHAR) = CAST(p.person_identifier AS VARCHAR)"
    )
    assert rows[0][0] >= 0  # executes across both catalogs without error


def test_predicate_pushed_into_source_scanner(tmp_path):
    adapter = _adapter(tmp_path)
    plan = "\n".join(
        str(cell) for row in adapter.execute(
            "EXPLAIN SELECT * FROM pg.public.person WHERE last_name = 'Smith'"
        ) for cell in row
    )
    # DuckDB's postgres extension reports the pushed predicate inside the source scan node.
    assert "POSTGRES_SCAN" in plan
    assert "Filters" in plan and "last_name" in plan

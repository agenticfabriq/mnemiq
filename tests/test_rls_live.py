"""Live RLS/CLS against ACME: a scoped identity sees only its row-filtered set and a NULL-masked
PII column, enforced at the source. Integration-gated (needs the ACME Postgres); no LLM."""

import os

import pytest

from mnemiq.adapters.duckdb import DuckDBPostgresAdapter
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Approved

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no ACME Postgres configured"),
]


def _snapshot():
    return Snapshot(version="v1", source_id="acme", created_at="t", columns=[
        Column(id="person.pid", object_id="person", name="person_identifier", pii_level="none"),
        Column(id="person.last", object_id="person", name="last_name", pii_level="pii"),
    ])


def test_scoped_identity_gets_filtered_rows_and_masked_pii():
    adapter = DuckDBPostgresAdapter(os.environ["MNEMIQ_PG_DSN"])  # read-only
    snap = _snapshot()
    grants = GrantSet(frozenset({"person"}),
                      row_filters={"person": "last_name < 'M'"}, pii_mask=frozenset({"pii"}))
    visible = visible_schema(snap, grants)
    policy = build_access_policy(snap, grants)

    v = decide("SELECT person_identifier, last_name FROM person", visible, adapter=adapter,
               dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(v, Approved), v
    low = v.target_sql.lower()
    assert "last_name < 'm'" in low and "null as last_name" in low  # filter + mask at source

    rows = adapter.execute(v.target_sql)
    assert all(r[1] is None for r in rows)  # CLS mask: last_name NULLed
    raw = adapter.execute("SELECT count(*) FROM person WHERE last_name < 'M'")[0][0]
    assert len(rows) == raw  # RLS: exactly the filtered rows, no more


def test_unscoped_identity_is_unaffected():
    adapter = DuckDBPostgresAdapter(os.environ["MNEMIQ_PG_DSN"])
    snap = _snapshot()
    grants = GrantSet(frozenset({"person"}), pii_clearance=frozenset({"pii"}))  # sees raw, no filter
    visible = visible_schema(snap, grants)
    policy = build_access_policy(snap, grants)

    v = decide("SELECT person_identifier, last_name FROM person", visible, adapter=adapter,
               dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(v, Approved)
    assert "null as last_name" not in v.target_sql.lower()  # cleared -> raw, no mask
    total = adapter.execute("SELECT count(*) FROM person")[0][0]
    assert len(adapter.execute(v.target_sql)) == total  # no row filter -> all rows

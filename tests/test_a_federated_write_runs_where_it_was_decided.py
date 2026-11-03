"""A federated write runs against the schema its grant names (M229).

A federation keys a table `catalog.table` and its registry maps the catalog to the schema the
source was registered with; a read is expanded to `catalog.schema.table` before it runs. A write was
not: `DELETE FROM pg.claim` was approved and ran as written, and DuckDB read `pg.claim` in the
catalog's default schema, so with the registered schema `allowed` beside a default `main` holding a
table of the same name, the write changed `pg.main.claim` and left `pg.allowed.claim` alone
(measured; Codex's review of the M226 branch). Each test below holds a same-named table in both
schemas and checks which one changed.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite

REGISTRY = {"pg": "allowed"}
SCHEMA = {"pg.claim": {"id", "region"}, "pg.sink": {"id", "region"}, "pg.ent": {"id"}}


def _federation():
    """A catalog `pg` with the registered schema `allowed` and a default `main`, each holding
    claim, sink and ent: claim rows 1 and 2, sink empty, ent row 1 in `allowed` and row 2 in `main`."""
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS pg")
    for schema in ("main", "allowed"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS pg.{schema}")
        con.execute(f"CREATE TABLE pg.{schema}.claim (id INT, region VARCHAR)")
        con.execute(f"INSERT INTO pg.{schema}.claim VALUES (1, '{schema}'), (2, '{schema}')")
        con.execute(f"CREATE TABLE pg.{schema}.sink (id INT, region VARCHAR)")
        con.execute(f"CREATE TABLE pg.{schema}.ent (id INT)")
    con.execute("INSERT INTO pg.allowed.ent VALUES (1)")
    con.execute("INSERT INTO pg.main.ent VALUES (2)")
    return con


def _write(sql, policy=None):
    grants = GrantSet(frozenset(SCHEMA), writable=frozenset(SCHEMA),
                      row_filters=dict(policy.row_filters) if policy else {})
    return decide_write(sql, SCHEMA, grants, dialect="duckdb", target="duckdb",
                        policy=policy or AccessPolicy(policy_schema=SCHEMA), writes_enabled=True,
                        registry=REGISTRY)


def _rows(con, table):
    return {schema: con.execute(f"SELECT id, region FROM pg.{schema}.{table} ORDER BY id").fetchall()
            for schema in ("main", "allowed")}


def test_the_read_runs_in_the_registered_schema():
    # The control: the read path already expands, and this is where the write must run too.
    verdict = decide("SELECT region FROM pg.claim", SCHEMA, dialect="duckdb", target="duckdb", registry=REGISTRY)
    assert isinstance(verdict, Approved), verdict
    assert _federation().execute(verdict.target_sql).fetchall() == [("allowed",), ("allowed",)]


@pytest.mark.parametrize("sql, table, after", [
    ("DELETE FROM pg.claim WHERE id = 1", "claim", [(2, "allowed")]),
    ("UPDATE pg.claim SET region = 'x' WHERE id = 1", "claim", [(1, "x"), (2, "allowed")]),
    ("INSERT INTO pg.sink (id, region) VALUES (9, 'new')", "sink", [(9, "new")]),
    ("INSERT INTO pg.sink (id, region) SELECT id, region FROM pg.claim", "sink", [(1, "allowed"), (2, "allowed")]),
])
def test_a_federated_write_changes_only_the_registered_schema(sql, table, after):
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite), verdict
    # No policy rewrote it, so the write as asked, which the source proves, is the one that runs.
    assert verdict.asked_sql == verdict.target_sql, (verdict.asked_sql, verdict.target_sql)
    con = _federation()
    before = _rows(con, table)
    con.execute(verdict.target_sql)
    rows = _rows(con, table)
    assert rows["allowed"] == after, rows
    assert rows["main"] == before["main"], rows


def test_a_row_filter_on_a_federated_write_reads_the_registered_schema():
    # The filter's own text names `pg.ent`, printed as written (M222): it is expanded too, so the
    # write keeps to the row the registered `ent` allows (1), not the default schema's (2).
    policy = AccessPolicy(policy_schema=SCHEMA, row_filters={"pg.claim": "id IN (SELECT id FROM pg.ent)"})
    verdict = _write("DELETE FROM pg.claim WHERE id > 0", policy)
    assert isinstance(verdict, ApprovedWrite), verdict
    con = _federation()
    con.execute(verdict.target_sql)
    rows = _rows(con, "claim")
    assert rows["allowed"] == [(2, "allowed")], rows
    assert rows["main"] == [(1, "main"), (2, "main")], rows


def test_the_approved_write_keeps_the_qualified_ids_it_was_decided_on():
    verdict = _write("INSERT INTO pg.sink (id, region) SELECT id, region FROM pg.claim")
    assert isinstance(verdict, ApprovedWrite), verdict
    assert verdict.target == "pg.sink" and verdict.tables == ["pg.claim", "pg.sink"], verdict

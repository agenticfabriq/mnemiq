"""Codex's review of #138 (M45): a view body's CTE names are bound the way its own source binds them.

#138 exempted a view-body table from the unknown-object check only where a CTE in scope binds it,
by DuckDB's rule: earlier siblings, never a later one. The view's SOURCE runs the body, and SQLite
binds a later sibling -- measured on SQLite 3.50 with a table `b` holding 40, `WITH a AS (SELECT id
FROM b), b AS (SELECT 1 AS id) SELECT id FROM a` read 1, where DuckDB read 40 -- so a legitimate
SQLite view with a forward reference was refused as reading an unknown `b`. Postgres binds a later
sibling only under RECURSIVE, and its view bodies come from `pg_get_viewdef`, which writes a
RECURSIVE WITH in dependency order, so DuckDB's rule already reads them right. The deciders keep
DuckDB's rule: DuckDB executes model SQL, whatever dialect it was written in.
"""

from __future__ import annotations

import os

import pytest
import sqlglot

from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.scope import cte_references
from mnemiq.sql.verdict import Refusal, RefusalCode

INVENTORY = {"customer": {"id"}, "orders": {"id", "store_id"}}
FORWARD = "WITH a AS (SELECT id FROM b), b AS (SELECT 1 AS id) SELECT id FROM a"


def _decide(body, dialect):
    policy = AccessPolicy(row_filters={"orders": "store_id = 1"}, policy_schema=INVENTORY)
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition=body)}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect="duckdb", target="duckdb",
                  policy=policy, views=views)


@pytest.mark.parametrize("body", [
    FORWARD,
    "WITH a AS (SELECT id FROM (SELECT id FROM b) x), b AS (SELECT 1 AS id) SELECT id FROM a",
])
def test_a_sqlite_view_reading_a_later_cte_is_approved(body):
    verdict = _decide(body, "sqlite")
    assert not isinstance(verdict, Refusal), verdict


@pytest.mark.parametrize("dialect,body", [
    ("duckdb", FORWARD),  # DuckDB binds the table: measured, 40
    ("sqlite", "WITH a AS (SELECT id FROM main.b), b AS (SELECT 1 AS id) SELECT id FROM a"),
    # An inner WITH's name is not in scope outside it: SQLite read the table, measured, 40.
    ("sqlite", "WITH a AS (SELECT id FROM b) SELECT id FROM (WITH b AS (SELECT 1 AS id) "
               "SELECT id FROM a) x"),
])
def test_a_reference_the_engine_binds_to_a_table_is_still_refused(dialect, body):
    verdict = _decide(body, dialect)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def test_a_later_cte_in_the_other_case_of_a_non_ascii_letter_does_not_bind_the_read():
    """Codex's review of #144: SQLite folds ASCII letters only, so a later CTE `Ä` does not bind
    a read of `ä` -- measured on SQLite 3.50, the table's 40. Reviewed on top of the resolver as it
    was, folding every letter, the widened binding exempted it; #147 narrowed the fold first."""
    verdict = _decide("WITH a AS (SELECT id FROM ä), Ä AS (SELECT 1 AS id) SELECT id FROM a",
                      "sqlite")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def test_the_deciders_binding_is_not_the_dialect_s():
    """Model SQL written in SQLite's dialect still executes on DuckDB, which reads `b` as the
    table: only an explicit engine widens the binding, never the parse dialect."""
    ast = sqlglot.parse_one(FORWARD, read="sqlite")
    assert [t.name for t in cte_references(ast, "sqlite")] == ["a"]
    assert sorted(t.name for t in cte_references(ast, "sqlite", engine="sqlite")) == ["a", "b"]


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_a_live_postgres_recursive_forward_view_is_stored_in_dependency_order():
    """Live: what keeps Postgres on DuckDB's rule. A RECURSIVE view written with a forward
    reference comes back from `pg_get_viewdef` -- the body this engine checks -- with `b` first."""
    import psycopg

    with psycopg.connect(os.environ["MNEMIQ_PG_DSN"]) as con:
        con.execute("CREATE TEMP VIEW fwd AS WITH RECURSIVE a AS (SELECT id FROM b), "
                    "b AS (SELECT 1 AS id) SELECT id FROM a")
        (body,) = con.execute("SELECT pg_get_viewdef('fwd'::regclass, true)").fetchone()
        con.rollback()
    verdict = _decide(body.strip().rstrip(";"), "postgres")
    assert not isinstance(verdict, Refusal), (verdict, body)

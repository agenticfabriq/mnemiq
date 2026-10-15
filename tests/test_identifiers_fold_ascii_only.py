"""Codex's review of #144: the shared resolver folds case as far as the engine does, and no further.

`identifiers.resolve` folded DuckDB, SQLite and Postgres names with Python's `str.lower`, which
folds every letter -- `Ä` to `ä`, the Kelvin sign to `k`. Those engines fold ASCII letters only, so
a CTE `Ä` and a table `ä` are two objects to them and were one to the resolver: a view body
`WITH Ä AS (...) SELECT id FROM ä` reading a table the snapshot does not know was approved, the
CTE vouching for the table (on SQLite and DuckDB views alike, before #144 widened SQLite's binding
to later siblings). Oracle folds `ä` to `Ä`, so its `str.upper` stays.
"""

from __future__ import annotations

import os
import sqlite3

import duckdb
import pytest

from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.identifiers import resolve
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode

KELVIN = "K"  # folds to an ASCII `k` under str.lower


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite", "postgres", "snowflake"])
@pytest.mark.parametrize("upper,lower", [("Ä", "ä"), (KELVIN, "k")])
def test_a_non_ascii_letter_is_not_folded(dialect, upper, lower):
    assert resolve(upper, quoted=False, dialect=dialect) != resolve(lower, quoted=False,
                                                                    dialect=dialect)


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite", "postgres", "snowflake"])
def test_ascii_letters_still_fold(dialect):
    assert resolve("Claim", quoted=False, dialect=dialect) == "claim"


def test_oracle_still_folds_a_non_ascii_letter_up():
    """Measured on Oracle Free: `CREATE TABLE ä` is stored `Ä`, and a CTE `Ä` binds `ä`."""
    assert resolve("ä", quoted=False, dialect="oracle") == resolve("Ä", quoted=False,
                                                                   dialect="oracle")


@pytest.mark.parametrize("engine", ["sqlite", "duckdb"])
def test_the_engines_read_the_table_behind_a_same_letter_cte(engine):
    """The rule's evidence, kept: if an engine ever folds further, this says so first."""
    con = sqlite3.connect(":memory:") if engine == "sqlite" else duckdb.connect()
    con.execute("CREATE TABLE ä (id INTEGER)")
    con.execute("INSERT INTO ä VALUES (40)")
    assert con.execute("WITH Ä AS (SELECT 1 AS id) SELECT id FROM ä").fetchall() == [(40,)]


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_live_postgres_reads_the_table_behind_a_same_letter_cte():
    import psycopg

    with psycopg.connect(os.environ["MNEMIQ_PG_DSN"]) as con:
        con.execute("CREATE TEMP TABLE ä (id int)")
        con.execute("INSERT INTO ä VALUES (40)")
        rows = con.execute("WITH Ä AS (SELECT 1 AS id) SELECT id FROM ä").fetchall()
        con.rollback()
    assert rows == [(40,)]


INVENTORY = {"customer": {"id"}, "orders": {"id", "store_id"}}


def _decide_view(body, dialect):
    policy = AccessPolicy(row_filters={"orders": "store_id = 1"}, policy_schema=INVENTORY)
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition=body)}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect="duckdb", target="duckdb",
                  policy=policy, views=views)


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
@pytest.mark.parametrize("body", [
    "WITH Ä AS (SELECT 1 AS id) SELECT id FROM ä",
    "WITH Ä AS (SELECT 1 AS id), a AS (SELECT id FROM ä) SELECT id FROM a",
    f"WITH {KELVIN} AS (SELECT 1 AS id) SELECT id FROM k",
])
def test_a_view_reading_an_unknown_table_behind_a_same_letter_cte_is_refused(dialect, body):
    verdict = _decide_view(body, dialect)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_an_ascii_case_variant_is_still_the_cte(dialect):
    """The control: `B` and `b` are one object to both engines, so the CTE binds the read."""
    verdict = _decide_view("WITH B AS (SELECT 1 AS id) SELECT id FROM b", dialect)
    assert not isinstance(verdict, Refusal), verdict

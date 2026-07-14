import sqlite3

import pytest

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE sale (id INTEGER, customer_id INTEGER, amount REAL, ts TEXT);
        INSERT INTO customer VALUES (1,'Ada','NYC'),(2,'Bo','LA'),(3,'Cy','NYC');
        INSERT INTO sale VALUES (1,1,10.0,'2021-01-01'),(2,1,5.0,'2021-02-01'),(3,2,7.0,'2022-01-01');
        """
    )
    con.commit()
    con.close()
    return str(path)


def test_structural_enrichment_profiles_a_sqlite_source(db):
    snapshot = enrich_structural(SQLiteAdapter(db), "shop")

    by_id = {c.id: c for c in snapshot.columns}
    assert by_id["customer.id"].row_count == 3
    assert by_id["customer.city"].distinct_count == 2  # NYC, LA
    assert {b.object_id for b in snapshot.source_bindings} == {"customer", "sale"}


def test_a_generated_query_transpiles_to_sqlite_and_executes(db):
    adapter = SQLiteAdapter(db)
    visible = {"customer": {"id", "name", "city"}, "sale": {"id", "customer_id", "amount", "ts"}}

    # duckdb-dialect input, as the generator emits; target=sqlite, as BIRD requires
    verdict = decide(
        "SELECT city, count(*) AS n FROM customer GROUP BY city",
        visible,
        adapter=adapter,
        dialect="duckdb",
        target="sqlite",
    )
    assert isinstance(verdict, Approved)

    result = adapter.execute_arrow(verdict.target_sql)
    got = {row["city"]: row["n"] for row in result.to_pylist()}
    assert got == {"NYC": 2, "LA": 1}


def test_the_decider_rejects_an_untranspilable_query_via_explain(db):
    # if a duckdb construct has no SQLite form, EXPLAIN on the source must catch it --
    # the safety net that turns a dialect gap into a deferral, never a wrong answer
    adapter = SQLiteAdapter(db)
    visible = {"sale": {"id", "amount", "ts"}}
    verdict = decide(
        "SELECT date_trunc('year', CAST(ts AS TIMESTAMP)) AS y FROM sale",
        visible,
        adapter=adapter,
        dialect="duckdb",
        target="sqlite",
    )
    # Either transpiled to something SQLite runs (Approved) or refused by EXPLAIN -- but
    # NEVER an approval of SQL the source cannot execute.
    if isinstance(verdict, Approved):
        adapter.execute_arrow(verdict.target_sql)  # must not raise

"""M162: the row cap reads the bound a statement wrote, in every spelling, before it imposes its own.

`_with_limit` read a LIMIT's count from `expression`. sqlglot keeps a FETCH's in `count`, so an
Oracle `FETCH FIRST 5 ROWS ONLY` counted as unreadable and ran as `FETCH FIRST 1000 ROWS ONLY`: a
top-5 that returns up to a thousand rows. A bound that states no plain count -- a percentage, WITH
TIES, an expression -- is refused instead of replaced: replacing it changed the answer, and leaving
it uncapped leaves the result unbounded (DuckDB's `LIMIT 10%` read as ten rows and ran uncapped).

The live half runs the kept bounds on Oracle; it skips without `MNEMIQ_ORACLE_TEST_DSN`.
"""

from __future__ import annotations

import os

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id", "amount"}}


def _decide(sql, engine):
    return decide(sql, VISIBLE, dialect=engine, target=engine)


@pytest.mark.parametrize("engine,tail,runs", [
    ("oracle", "FETCH FIRST 5 ROWS ONLY", "FETCH FIRST 5 ROWS ONLY"),
    ("oracle", "OFFSET 2 ROWS FETCH NEXT 5 ROWS ONLY", "OFFSET 2 ROWS FETCH NEXT 5 ROWS ONLY"),
    ("oracle", "LIMIT 5", "FETCH FIRST 5 ROWS ONLY"),
    ("duckdb", "LIMIT 5", "LIMIT 5"),
    ("duckdb", "LIMIT 5 OFFSET 2", "LIMIT 5 OFFSET 2"),
])
def test_a_bound_the_statement_wrote_within_the_cap_is_kept(engine, tail, runs):
    verdict = _decide(f"SELECT id FROM claim ORDER BY amount DESC {tail}", engine)
    assert isinstance(verdict, Approved), verdict
    assert verdict.target_sql.endswith(runs), verdict.target_sql
    assert verdict.row_cap is None  # the statement's own bound: no cut to disclose


def test_fetch_first_row_only_is_one_row_not_the_cap():
    """No count is written, and the clause means one row."""
    shaped = check_shape("SELECT id FROM claim ORDER BY amount DESC FETCH FIRST ROW ONLY",
                         dialect="oracle", executes_as="oracle")
    assert not isinstance(shaped, Refusal), shaped
    assert shaped.meta.get("row_cap") is None
    assert "1000" not in shaped.sql(dialect="oracle")


@pytest.mark.parametrize("engine,tail", [("oracle", "FETCH FIRST 5000 ROWS ONLY"),
                                         ("duckdb", "LIMIT 5000")])
def test_a_bound_over_the_cap_is_clamped_and_marked(engine, tail):
    verdict = _decide(f"SELECT id FROM claim ORDER BY amount DESC {tail}", engine)
    assert isinstance(verdict, Approved), verdict
    assert verdict.row_cap == MAX_ROWS and str(MAX_ROWS) in verdict.target_sql


@pytest.mark.parametrize("engine,tail", [
    ("oracle", "FETCH FIRST 5 ROWS WITH TIES"),
    ("oracle", "FETCH FIRST 10 PERCENT ROWS ONLY"),
    ("duckdb", "LIMIT 10%"),
    ("duckdb", "LIMIT 1 + 1"),
])
def test_a_bound_with_no_plain_count_is_refused_not_replaced(engine, tail):
    verdict = _decide(f"SELECT id FROM claim ORDER BY amount DESC {tail}", engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNREADABLE_LIMIT, verdict
    assert verdict.repairable  # the same question can always state its bound as a count
    assert f"at most {MAX_ROWS:,}" in verdict.message


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
@pytest.mark.parametrize("tail,rows", [("FETCH FIRST 5 ROWS ONLY", 5),
                                       ("FETCH FIRST ROW ONLY", 1),
                                       ("OFFSET 2 ROWS FETCH NEXT 5 ROWS ONLY", 5)])
def test_on_oracle_the_kept_bound_returns_its_own_count(tail, rows):
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    try:
        cur = con.cursor()
        # 50 rows from dual, so a bound replaced by the cap returns all 50, not the count.
        # NULLS LAST: an unguarded top-N is the logic lint's to refuse (M161), not this test's.
        verdict = decide(f"SELECT id FROM claim ORDER BY id DESC NULLS LAST {tail}", {"claim": {"id"}},
                         dialect="oracle", target="oracle")
        assert isinstance(verdict, Approved), verdict
        sql = verdict.target_sql.replace(
            "FROM claim", "FROM (SELECT level AS id FROM dual CONNECT BY level <= 50) claim")
        assert len(cur.execute(sql).fetchall()) == rows, sql
    finally:
        con.close()

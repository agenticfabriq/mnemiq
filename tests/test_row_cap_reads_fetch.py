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
import re

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


@pytest.mark.parametrize("dialect,tail,runs", [
    ("postgres", "LIMIT NULL", f"LIMIT {MAX_ROWS}"),
    ("sqlite", "LIMIT -1", f"LIMIT {MAX_ROWS}"),
    ("sqlite", "LIMIT -1 OFFSET 5", f"LIMIT {MAX_ROWS} OFFSET 5"),
    ("postgres", "LIMIT ALL", f"LIMIT {MAX_ROWS}"),
])
def test_a_spelling_of_no_bound_takes_the_cap_as_if_none_were_written(dialect, tail, runs):
    """On main SQLite's `LIMIT -1` was kept and ran uncapped (the ui lane's review of #156)."""
    shaped = check_shape(f"SELECT id FROM claim ORDER BY amount DESC {tail}", dialect=dialect,
                         executes_as=dialect)
    assert not isinstance(shaped, Refusal), shaped
    assert shaped.meta.get("row_cap") == MAX_ROWS
    assert shaped.sql(dialect=dialect).endswith(runs), shaped.sql(dialect=dialect)


@pytest.mark.parametrize("tail", ["LIMIT -(-5)", "LIMIT -0", "LIMIT -(5 + 1)"])
def test_a_minus_over_anything_but_a_positive_number_is_refused_in_sqlite(tail):
    """`-(-5)` is five rows and `-0` none: only a negative plain number is SQLite's no bound."""
    shaped = check_shape(f"SELECT id FROM claim ORDER BY amount DESC {tail}", dialect="sqlite",
                         executes_as="sqlite")
    assert isinstance(shaped, Refusal) and shaped.code is RefusalCode.UNREADABLE_LIMIT, shaped


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "oracle"])
def test_a_negative_limit_outside_sqlite_is_refused_not_capped(dialect):
    """Only SQLite reads a negative LIMIT as no bound; elsewhere it is an error the cap must not
    turn into a thousand rows."""
    shaped = check_shape("SELECT id FROM claim ORDER BY amount DESC LIMIT -1", dialect=dialect,
                         executes_as=dialect)
    assert isinstance(shaped, Refusal) and shaped.code is RefusalCode.UNREADABLE_LIMIT, shaped


# --- the rewrites the refusal suggests, run as written ---------------------------------------------

def _advice(sql, engine):
    """The ranking and share filters the refusal of `sql` suggests, read from its message."""
    refused = _decide(sql, engine)
    assert isinstance(refused, Refusal) and refused.code is RefusalCode.UNREADABLE_LIMIT, refused
    ties = re.search(r"`(RANK\(\) OVER \(ORDER BY \.\.\.\)) <= n`", refused.message)
    share = re.search(r"filter on `(rn <= [^`]+)`", refused.message)
    # The pieces `_share_rewrite` writes itself, so the advice cannot drift from what runs here.
    for piece in ("ROW_NUMBER() OVER (ORDER BY ...) AS rn", "COUNT(*) OVER () AS total",
                  "RANK() in place of ROW_NUMBER() to keep ties"):
        assert piece in refused.message, (piece, refused.message)
    assert ties and share, refused.message
    return ties.group(1), share.group(1)


def _share_rewrite(share, p, *, rank=False):
    number = "RANK()" if rank else "ROW_NUMBER()"
    keep = re.sub(r"\bp\b", str(p), share)
    return (f"SELECT id FROM (SELECT id, {number} OVER (ORDER BY amount DESC) AS rn, "
            f"COUNT(*) OVER () AS total FROM claim) t WHERE {keep}")


def _ties_rewrite(ties, n):
    return (f"SELECT id FROM (SELECT id, {ties.replace('...', 'amount DESC')} AS rn FROM claim) t "
            f"WHERE rn <= {n}")


DUCK_ROWS = {"tied": "SELECT i AS id, 1 AS amount FROM range(10) r(i)",
             "seven": "SELECT i AS id, i AS amount FROM range(7) r(i)",
             "hundred": "SELECT i AS id, i AS amount FROM range(100) r(i)"}


@pytest.mark.parametrize("rows,p,kept", [
    ("tied", 50, 5),      # ten rows on one value: the CUME_DIST advice kept none (Codex, #156)
    ("seven", 50, 4),     # rounded up, as Oracle's FETCH ... PERCENT is
    ("seven", 10, 1),     # a positive share keeps a row; DuckDB's own `LIMIT 10%` returns none
    ("hundred", 7, 7),    # divided before multiplied, DuckDB's doubles round 7% of 100 up to 8
])
def test_the_suggested_share_keeps_its_count_of_rows_on_duckdb(rows, p, kept):
    import duckdb

    _, share = _advice(f"SELECT id FROM claim ORDER BY amount DESC LIMIT {p}%", "duckdb")
    verdict = _decide(_share_rewrite(share, p), "duckdb")
    assert isinstance(verdict, Approved), verdict
    con = duckdb.connect()
    con.execute(f"CREATE TABLE claim AS {DUCK_ROWS[rows]}")
    assert len(con.execute(verdict.target_sql).fetchall()) == kept, verdict.target_sql


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


ORA_ROWS = {"tied": "SELECT level AS id, 1 AS amount FROM dual CONNECT BY level <= 10",
            "seven": "SELECT level AS id, level AS amount FROM dual CONNECT BY level <= 7",
            "hundred": "SELECT level AS id, level AS amount FROM dual CONNECT BY level <= 100",
            # 9, 9, 9, then 8 down to 4: a peer group of three at the top
            "mixed": "SELECT level AS id, CASE WHEN level <= 3 THEN 9 ELSE level END AS amount "
                     "FROM dual CONNECT BY level <= 8"}


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
@pytest.mark.parametrize("rows,tail", [
    ("tied", "FETCH FIRST 50 PERCENT ROWS ONLY"),
    ("seven", "FETCH FIRST 50 PERCENT ROWS ONLY"),
    ("seven", "FETCH FIRST 10 PERCENT ROWS ONLY"),
    ("hundred", "FETCH FIRST 7 PERCENT ROWS ONLY"),
    ("mixed", "FETCH FIRST 1 ROWS WITH TIES"),
    ("mixed", "FETCH FIRST 4 ROWS WITH TIES"),
    ("mixed", "FETCH FIRST 30 PERCENT ROWS WITH TIES"),
    ("mixed", "FETCH FIRST 50 PERCENT ROWS WITH TIES"),
])
def test_on_oracle_the_suggested_rewrite_returns_what_the_refused_bound_would(rows, tail):
    oracledb = pytest.importorskip("oracledb")
    count = int(re.search(r"FIRST (\d+)", tail).group(1))
    ties, share = _advice(f"SELECT id FROM claim ORDER BY amount DESC {tail}", "oracle")
    if "PERCENT" in tail:
        rewrite = _share_rewrite(share, count, rank="WITH TIES" in tail)
    else:
        rewrite = _ties_rewrite(ties, count)
    verdict = _decide(rewrite, "oracle")
    assert isinstance(verdict, Approved), verdict
    data = f"FROM ({ORA_ROWS[rows]}) claim"
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    try:
        cur = con.cursor()
        written = len(cur.execute(f"SELECT id {data} ORDER BY amount DESC {tail}").fetchall())
        suggested = len(cur.execute(verdict.target_sql.replace("FROM claim", data)).fetchall())
        assert suggested == written, (tail, verdict.target_sql)
    finally:
        con.close()

"""M161: the null-before-limit lint fires where the engine puts NULLs first, on a limit the statement
wrote, and repairs by reordering.

It fired on the row cap the shape check appends -- 13 of a pilot customer's 18 gold statements and
40 of BEAVER's scored 200, none with a LIMIT of its own -- on a premise (ascending puts NULLs first)
that neither engine production runs holds, DuckDB nor Oracle, and its repair (`WHERE col IS NOT NULL`) turned 4 of those 40 correct
statements into wrong ones by dropping rows. The premise is measured here, per engine, so an engine
that changes its order fails this file rather than the answers.
"""

from __future__ import annotations

import os
import sqlite3

import duckdb
import pytest
import sqlglot

from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id", "name"}}
THREE = "SELECT 1 AS x UNION ALL SELECT NULL UNION ALL SELECT 2"


def _verdict(tail, engine):
    return decide(f"SELECT id FROM claim {tail}", VISIBLE, dialect=engine, target=engine)


def _linted(verdict):
    return isinstance(verdict, Refusal) and verdict.code is RefusalCode.LOGIC_LINT


# --- the premise: where each engine puts NULLs, and that sqlglot says the same ---------------------

def test_duckdb_puts_nulls_last_both_ways_and_sqlglot_agrees():
    con = duckdb.connect()
    for direction, first in (("ASC", 1), ("DESC", 2)):
        got = con.execute(f"SELECT x FROM ({THREE}) t ORDER BY x {direction} LIMIT 1").fetchall()
        assert got == [(first,)], direction
        parsed = sqlglot.parse_one(f"SELECT x FROM t ORDER BY x {direction}", read="duckdb")
        assert not parsed.args["order"].expressions[0].args.get("nulls_first"), direction


def test_sqlite_puts_nulls_first_ascending_and_sqlglot_agrees():
    con = sqlite3.connect(":memory:")
    for direction, first in (("ASC", None), ("DESC", 2)):
        got = con.execute(f"SELECT x FROM ({THREE}) ORDER BY x {direction} LIMIT 1").fetchall()
        assert got == [(first,)], direction
        parsed = sqlglot.parse_one(f"SELECT x FROM t ORDER BY x {direction}", read="sqlite")
        nulls_first = bool(parsed.args["order"].expressions[0].args.get("nulls_first"))
        assert nulls_first is (first is None), direction


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_oracle_puts_nulls_first_only_descending_and_sqlglot_agrees():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    try:
        three = ("SELECT 1 x FROM dual UNION ALL SELECT NULL FROM dual "
                 "UNION ALL SELECT 2 FROM dual")
        for direction, first in (("ASC", 1), ("DESC", None)):
            got = con.cursor().execute(
                f"SELECT x FROM ({three}) ORDER BY x {direction} FETCH FIRST 1 ROWS ONLY").fetchall()
            assert got == [(first,)], direction
            parsed = sqlglot.parse_one(f"SELECT x FROM t ORDER BY x {direction}", read="oracle")
            nulls_first = bool(parsed.args["order"].expressions[0].args.get("nulls_first"))
            assert nulls_first is (first is None), direction
    finally:
        con.close()


# --- the rule ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["duckdb", "oracle", "sqlite"])
def test_the_row_cap_is_not_a_limit_the_statement_wrote(engine):
    """No LIMIT written: the cap's own is not the statement's, whatever the order."""
    for tail in ("ORDER BY name", "ORDER BY name DESC"):
        assert isinstance(_verdict(tail, engine), Approved), (engine, tail)


@pytest.mark.parametrize("engine,tail", [
    ("oracle", "ORDER BY name DESC FETCH FIRST 5 ROWS ONLY"),
    ("oracle", "ORDER BY name DESC FETCH FIRST 5000 ROWS ONLY"),  # written, then clamped: still its own
    ("sqlite", "ORDER BY name LIMIT 5"),
])
def test_a_written_limit_whose_first_key_puts_nulls_first_is_linted(engine, tail):
    verdict = _verdict(tail, engine)
    assert _linted(verdict), verdict
    assert "NULLS LAST" in verdict.message and "IS NOT NULL" not in verdict.message


@pytest.mark.parametrize("engine,tail", [
    ("duckdb", "ORDER BY name LIMIT 5"),
    ("duckdb", "ORDER BY name DESC LIMIT 5"),
    ("oracle", "ORDER BY name FETCH FIRST 5 ROWS ONLY"),
    ("oracle", "ORDER BY name DESC NULLS LAST FETCH FIRST 5 ROWS ONLY"),
    ("oracle", "ORDER BY name DESC NULLS FIRST FETCH FIRST 5 ROWS ONLY"),  # written: the author's
    ("sqlite", "ORDER BY name DESC LIMIT 5"),
])
def test_a_limit_whose_order_puts_nulls_last_or_says_where_they_go_is_not_linted(engine, tail):
    assert isinstance(_verdict(tail, engine), Approved), (engine, tail)


def test_a_not_null_guard_still_clears_it():
    verdict = _verdict("WHERE name IS NOT NULL ORDER BY name DESC FETCH FIRST 5 ROWS ONLY", "oracle")
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("sql", [
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id NULLS LAST) AS r FROM claim "
    "ORDER BY name DESC FETCH FIRST 5 ROWS ONLY",
    "SELECT id FROM claim WHERE id IN (SELECT id FROM claim ORDER BY id NULLS LAST "
    "FETCH FIRST 3 ROWS ONLY) ORDER BY name DESC FETCH FIRST 5 ROWS ONLY",
    "SELECT id FROM claim ORDER BY name DESC, id NULLS LAST FETCH FIRST 5 ROWS ONLY",
])
def test_a_null_order_written_anywhere_but_the_first_key_exempts_nothing(sql):
    """A window's, a subquery's or a later key's NULLS clause says nothing about the top-N."""
    assert _linted(decide(sql, VISIBLE, dialect="oracle", target="oracle"))

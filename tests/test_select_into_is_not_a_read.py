"""A SELECT ... INTO is a write, not a read (M234).

`SELECT id INTO flags FROM kw` has a `Select` root, so it cleared the read decider's root check,
which promises "this engine never modifies data". DuckDB and SQLite render it as
`CREATE TABLE flags AS ...` and Postgres as `SELECT ... INTO`, each creating or overwriting the
target table; `INTO TEMP` plants a connection-local table that can shadow a real one. The read
decider now refuses any statement holding an INTO clause.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.guard import check_shape
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

V = {"kw": {"id"}, "flags": {"id"}}


@pytest.mark.parametrize("sql", [
    "SELECT id INTO flags FROM kw",
    "SELECT id INTO TEMP flags FROM kw",
    "SELECT id INTO TEMPORARY flags FROM kw",
    "WITH c AS (SELECT id FROM kw) SELECT id INTO flags FROM c",
])
@pytest.mark.parametrize("target", ["duckdb", "sqlite", "postgres"])
def test_a_select_into_is_refused_as_not_a_read(sql, target):
    verdict = decide(sql, V, dialect="duckdb", target=target, policy=AccessPolicy(policy_schema=V))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.NOT_SELECT_ONLY, verdict


def test_check_shape_refuses_into_directly():
    shaped = check_shape("SELECT id INTO flags FROM kw", dialect="duckdb")
    assert isinstance(shaped, Refusal) and shaped.code is RefusalCode.NOT_SELECT_ONLY, shaped


def test_an_ordinary_select_still_passes():
    verdict = decide("SELECT id FROM kw", V, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=V))
    assert isinstance(verdict, Approved), verdict
    con = duckdb.connect()
    con.execute("CREATE TABLE kw (id INT); INSERT INTO kw VALUES (1)")
    assert con.execute(verdict.target_sql).fetchall() == [(1,)]


def test_a_column_or_alias_named_into_is_not_mistaken_for_the_clause():
    # `into` as an identifier is a column, not an INTO clause: still a read.
    con = duckdb.connect()
    con.execute('CREATE TABLE kw (id INT, "into" INT)')
    con.execute("INSERT INTO kw VALUES (1, 9)")
    verdict = decide('SELECT "into" FROM kw', {"kw": {"id", "into"}}, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema={"kw": {"id", "into"}}))
    assert isinstance(verdict, Approved), verdict
    assert con.execute(verdict.target_sql).fetchall() == [(9,)]

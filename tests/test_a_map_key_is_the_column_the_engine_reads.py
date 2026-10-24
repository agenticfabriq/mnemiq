"""A MAP literal's bare key is the column DuckDB reads, not the struct name sqlglot parses, at
every parse the decider makes: the plan's, a view body's, and the round trip that renders what
runs (Codex's forty-third review of #160, found on a row filter and traced to the shared seam).
`map_keys(MAP {secret: 1})[1]` returns the column's value on DuckDB; the grant check walked
Columns and saw a struct's name, and the plan's round trip to the target wrote `MAP {'secret':
1}`, a string, so the statement run was not the statement checked, and `MAP {id: 1}[id]`
answered for the string where the author asked for the column."""

from __future__ import annotations

import duckdb
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode
from mnemiq.sql.views import body_of

VISIBLE = {"claim": {"id", "region"}, "sink": {"i", "s"}}


class _OkAdapter:
    def execute(self, sql):
        return []


def _table():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, secret VARCHAR); INSERT INTO claim VALUES (1, 'north', 's3cret'); "
                "CREATE TABLE sink(i INT, s VARCHAR); INSERT INTO sink VALUES (1, 'a')")
    return con


def _decide(sql):
    return decide(sql, VISIBLE, dialect="duckdb", target="duckdb")


def test_a_hidden_column_named_as_a_map_key_is_a_read():
    con = _table()
    sql = "SELECT map_keys(MAP {secret: 1})[1] FROM claim"
    assert con.execute(sql).fetchall() == [("s3cret",)]
    for spelling in (sql, 'SELECT map_keys(MAP {"secret": 1})[1] FROM claim', "SELECT (MAP {secret: 1})[1] FROM claim",
                     "SELECT MAP {secret: 1}[1] FROM claim"):
        verdict = _decide(spelling)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (spelling, verdict)
    # a struct literal's key is a name to DuckDB, no read
    assert con.execute("SELECT {secret: 1}.secret FROM claim").fetchall() == [(1,)]
    assert isinstance(_decide("SELECT {secret: 1}.secret FROM claim"), Approved)


def test_a_granted_map_key_runs_as_written():
    con = _table()
    for sql, written in (("SELECT MAP {id: 1}[id] FROM claim", "MAP {id: 1}"),
                         ("SELECT map_keys(MAP {region: 1})[1] FROM claim", "MAP {region: 1}")):
        rows = con.execute(sql).fetchall()
        verdict = _decide(sql)
        assert isinstance(verdict, Approved), verdict
        assert written in verdict.plan_sql and written in verdict.target_sql, verdict.target_sql
        assert con.execute(verdict.target_sql).fetchall() == rows, verdict.target_sql
    # the round trip under the row cap (a LIMIT sqlglot cannot parse after GROUPING SETS) reads the same
    sql = "SELECT MAP {id: 1}[id] AS k FROM claim GROUP BY GROUPING SETS ((id))"
    rows = con.execute(sql).fetchall()
    verdict = _decide(sql)
    assert isinstance(verdict, Approved) and "MAP {id: 1}" in verdict.target_sql, verdict
    assert con.execute(verdict.target_sql).fetchall() == rows


def test_a_write_renders_its_map_key_as_written_and_reads_it():
    grants = GrantSet(frozenset(VISIBLE), writable=frozenset({"sink"}))
    sql = "UPDATE sink SET s = 'b' WHERE MAP {i: 1}[i] = 1"
    verdict = decide_write(sql, VISIBLE, grants, adapter=_OkAdapter(), dialect="duckdb", target="duckdb", writes_enabled=True,
                           policy=AccessPolicy())
    assert isinstance(verdict, ApprovedWrite) and "MAP {i: 1}" in verdict.target_sql, verdict
    con = _table()
    con.execute(verdict.target_sql)
    assert con.execute("SELECT s FROM sink").fetchall() == [("b",)]
    verdict = decide_write("UPDATE sink SET s = 'b' WHERE MAP {secret: 1}[1] IS NULL", VISIBLE, grants, adapter=_OkAdapter(),
                           dialect="duckdb", target="duckdb", writes_enabled=True, policy=AccessPolicy())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


def test_a_view_bodys_map_key_is_a_column():
    body = body_of(ViewDefinition(object_id="leak", dialect="duckdb",
                                  definition="SELECT map_keys(MAP {secret: 1})[1] AS k FROM claim"))
    assert body is not None and {c.name for c in body.find_all(exp.Column)} == {"secret"}

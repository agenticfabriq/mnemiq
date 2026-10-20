"""A join key reads the column it names, and a key hidden from the grant was read by nothing
(M200, the ui lane's replay of #169 and Codex's seventh review of it): `USING (secret)` and a
NATURAL join over a table whose `secret` is absent from the grant were approved, and whether
rows match says whether the hidden value is the guess. M103 closed the denied and masked key in
`check_cls`; this closes the hidden one in `check_access`, through a derived star too.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

SCHEMA = {"claim": {"id", "secret"}, "pair": {"id", "secret"}, "one": {"secret"}, "sink": {"i", "s"}}
HIDDEN = {"claim": {"id"}, "pair": {"id"}, "one": set(), "sink": {"i", "s"}}
COMPLETE = AccessPolicy(policy_schema=SCHEMA)


@pytest.fixture
def worlds():
    """Two engines differing only in the hidden values: an oracle answers differently on them."""
    made = []
    for secret in ("Z", "A"):
        con = duckdb.connect()
        con.execute(f"CREATE TABLE claim (id INT, secret VARCHAR); INSERT INTO claim VALUES (1, '{secret}'); "
                    f"CREATE TABLE pair (id INT, secret VARCHAR); INSERT INTO pair VALUES (1, '{secret}'), (2, 'Z'); "
                    f"CREATE TABLE one (secret VARCHAR); INSERT INTO one VALUES ('{secret}'); "
                    "CREATE TABLE sink (i INT, s VARCHAR); INSERT INTO sink VALUES (1, 'x')")
        made.append(con)
    return made


def _decide(sql, visible, policy=COMPLETE):
    return decide(sql, visible, dialect="duckdb", target="duckdb", policy=policy)


class _OkAdapter:
    def execute(self, sql):
        return []


def _decide_write(sql, visible, policy=COMPLETE):
    grants = GrantSet(frozenset(visible), writable=frozenset({"sink"}))
    return decide_write(sql, visible, grants, adapter=_OkAdapter(), dialect="duckdb", target="duckdb",
                        writes_enabled=True, policy=policy)


@pytest.mark.parametrize("sql", [
    "SELECT count(*) AS n FROM claim JOIN (SELECT 1 AS id, 'Z' AS secret) x USING (id, secret)",
    "SELECT count(*) AS n FROM claim NATURAL JOIN (SELECT 1 AS id, 'Z' AS secret) x",
    "SELECT count(*) AS n FROM one JOIN (SELECT 'Z' AS secret) x USING (secret)",
    "SELECT d.id FROM (SELECT * FROM pair) d JOIN (SELECT 'Z' AS secret) e USING (secret)",
    "SELECT count(*) AS n FROM (SELECT * FROM pair) d JOIN (SELECT * FROM pair) e USING (secret)",
    "WITH w AS (SELECT * FROM pair) SELECT count(*) AS n FROM w NATURAL JOIN (SELECT 1 AS id, 'Z' AS secret) x",
    "WITH w AS (SELECT * FROM pair) SELECT count(*) AS n FROM w JOIN (SELECT 'Z' AS secret) e USING (secret)",
    "WITH w AS (SELECT * FROM pair) SELECT n FROM (SELECT count(*) AS n FROM w JOIN (SELECT 'Z' AS secret) e USING (secret)) q",
    "WITH w AS (SELECT * FROM pair) SELECT (SELECT count(*) FROM w JOIN (SELECT 'Z' AS secret) e USING (secret)) AS n FROM claim",
    "SELECT id FROM claim WHERE EXISTS (SELECT 1 FROM one JOIN (SELECT 'Z' AS secret) x USING (secret))",
    # A CTE name defined in two scopes is the join's own scope's (the review gate).
    "WITH w AS (SELECT * FROM pair) SELECT q.n FROM (SELECT count(*) AS n FROM w JOIN (SELECT 'Z' AS secret) e USING (secret)) q CROSS JOIN (WITH w AS (SELECT i FROM sink) SELECT i FROM w) r",
    "WITH w AS (SELECT i FROM sink) SELECT q.n FROM (WITH w AS (SELECT * FROM pair) SELECT count(*) AS n FROM w JOIN (SELECT 'Z' AS secret) e USING (secret)) q",
    # A CTE named after the table it reads, and a sibling cycle (the review gate's third round).
    "WITH pair AS (SELECT * FROM pair) SELECT count(*) AS n FROM pair JOIN (SELECT 'Z' AS secret) e USING (secret)",
    "WITH one AS (SELECT * FROM one) SELECT count(*) AS n FROM one NATURAL JOIN (SELECT 'Z' AS secret) e",
    "WITH w AS (SELECT * FROM pair), pair AS (SELECT * FROM w) SELECT count(*) AS n FROM pair JOIN (SELECT 'Z' AS secret) e USING (secret)",
    # A later sibling named after a base table is not visible inside an earlier body.
    "WITH a AS (SELECT * FROM pair), pair AS (SELECT i FROM sink) SELECT count(*) AS n FROM a JOIN (SELECT 'Z' AS secret) e USING (secret)",
    # An alias reused in another scope (Codex's second review): the alias map keeps one table
    # per alias, the outer one, so a join inside the subquery lost its table from the set.
    "SELECT count(*) AS n FROM sink t WHERE EXISTS (SELECT 1 FROM claim t JOIN (SELECT 'Z' AS secret) e USING (secret))",
    "SELECT count(*) AS n FROM sink t WHERE EXISTS (SELECT 1 FROM claim t NATURAL JOIN (SELECT 1 AS id, 'Z' AS secret) e)",
    "SELECT count(*) AS n FROM claim t JOIN (SELECT 'Z' AS secret) e USING (secret) WHERE EXISTS (SELECT 1 FROM sink t)",
    "SELECT count(x.id) AS n FROM claim NATURAL LEFT JOIN (SELECT 1 AS id, 'Z' AS secret) x",
])
def test_a_hidden_join_key_is_refused(worlds, sql):
    first, second = (con.execute(sql).fetchall() for con in worlds)
    assert first != second, (sql, first, second)  # an oracle on the engine
    hidden = _decide(sql, HIDDEN)
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, (sql, hidden)
    # Without the complete schema the hiding cannot be ruled out: refused the same, fail-closed.
    grant_only = _decide(sql, HIDDEN, policy=AccessPolicy())
    assert isinstance(grant_only, Refusal) and grant_only.code is RefusalCode.UNKNOWN_COLUMN, (sql, grant_only)
    for policy in (AccessPolicy(denied={("claim", "secret"), ("pair", "secret"), ("one", "secret")}, policy_schema=SCHEMA),
                   AccessPolicy(masked={("claim", "secret"), ("pair", "secret"), ("one", "secret")}, policy_schema=SCHEMA)):
        assert isinstance(_decide(sql, SCHEMA, policy=policy), Refusal), (sql, policy)  # M103, as before


@pytest.mark.parametrize("sql", [
    "INSERT INTO sink (i) SELECT count(*) FROM claim JOIN (SELECT 1 AS id, 'Z' AS secret) x USING (id, secret)",
    "INSERT INTO sink (i) SELECT count(*) FROM claim NATURAL JOIN (SELECT 1 AS id, 'Z' AS secret) x",
    "DELETE FROM sink WHERE i IN (SELECT x.id FROM claim JOIN (SELECT 1 AS id, 'Z' AS secret) x USING (id, secret))",
    "INSERT INTO sink (i) SELECT count(*) FROM sink t WHERE EXISTS (SELECT 1 FROM claim t JOIN (SELECT 'Z' AS secret) e USING (secret))",
])
def test_a_hidden_join_key_in_a_write_is_refused(worlds, sql):
    for con in worlds:
        con.execute(sql)
    first, second = (con.execute("SELECT * FROM sink").fetchall() for con in worlds)
    assert first != second, (sql, first, second)
    verdict = _decide_write(sql, HIDDEN)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_granted_join_key_is_read_as_before(worlds):
    for sql in ("SELECT count(*) AS n FROM claim JOIN (SELECT 1 AS id) x USING (id)",
                "SELECT count(*) AS n FROM claim JOIN pair USING (id)",
                "SELECT d.id FROM (SELECT * FROM pair) d JOIN (SELECT 1 AS id) e USING (id)"):
        worlds[0].execute(sql)
        assert isinstance(_decide(sql, HIDDEN), Approved), sql
        assert isinstance(_decide(sql, SCHEMA), Approved), sql
    # A NATURAL join over tables listed in full with nothing hidden is read as before; over a
    # table with a hidden column, or not listed in full, it is refused.
    natural = "SELECT count(*) AS n FROM claim NATURAL JOIN (SELECT 1 AS id) x"
    assert isinstance(_decide(natural, SCHEMA), Approved)
    assert isinstance(_decide(natural, HIDDEN), Refusal)
    assert isinstance(_decide(natural, SCHEMA, policy=AccessPolicy()), Refusal)
    write = "INSERT INTO sink (i) SELECT count(*) FROM claim JOIN pair USING (id)"
    assert isinstance(_decide_write(write, SCHEMA), ApprovedWrite)
    # Only the join's own inputs are read for the key, not every table its SELECT mentions
    # (the review gate): a WHERE's subquery over a table without the key is no reason.
    aside = "SELECT count(*) AS n FROM claim JOIN pair USING (id) WHERE id IN (SELECT i FROM sink)"
    worlds[0].execute(aside)
    assert isinstance(_decide(aside, SCHEMA, policy=AccessPolicy()), Approved)
    assert isinstance(_decide(aside, SCHEMA), Approved)


def test_a_whole_row_as_an_order_by_key_over_a_derived_star_stays_refused(worlds):
    # The ui lane's replay of #169: sqlglot calls `t` and `t.*` bare columns, DuckDB reads the
    # whole row, withheld column included; another check refuses both, pinned here.
    for sql in ("SELECT d.id FROM (SELECT * FROM pair t ORDER BY t LIMIT 1) d",
                "SELECT d.id FROM (SELECT * FROM pair t ORDER BY t.* LIMIT 1) d"):
        verdict = _decide(sql, HIDDEN)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)

"""A CTE named like a granted table in another case was invisible to the column check, which
read the granted base table while the engine read the CTE (M202, the ui lane's replay of #170):
with `w(id, secret)` granted and `claim.secret` withheld, `WITH W AS (SELECT * FROM claim) SELECT
w.secret FROM w` returned claim's secret with no policy, with the schema, denied and masked.
sqlglot keys a CTE's scope by its spelling as written and matches a reference by that spelling;
DuckDB and Oracle fold both. M31 compared CTE names as written in the scope and column path;
#170 taught the join check the shared resolver's folding, and this teaches the rest.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

SCHEMA = {"claim": {"id", "secret"}, "w": {"id", "secret"}}
HIDDEN = {"claim": {"id"}, "w": {"id", "secret"}}
DENIED = AccessPolicy(denied={("claim", "secret")}, policy_schema=SCHEMA)
MASKED = AccessPolicy(masked={("claim", "secret")}, policy_schema=SCHEMA)


@pytest.fixture
def worlds():
    """Two engines differing only in claim's secret: an oracle answers differently on them."""
    made = []
    for secret in ("Z", "A"):
        con = duckdb.connect()
        con.execute(f"CREATE TABLE claim (id INT, secret VARCHAR); INSERT INTO claim VALUES (1, '{secret}'); "
                    "CREATE TABLE w (id INT, secret VARCHAR); INSERT INTO w VALUES (1, 'w-own')")
        made.append(con)
    return made


def _decide(sql, visible, policy=None, dialect="duckdb"):
    return decide(sql, visible, dialect=dialect, target=dialect, policy=policy)


_FOLDED = [
    "WITH W AS (SELECT * FROM claim) SELECT w.secret FROM w",
    "WITH W AS (SELECT * FROM claim) SELECT x.secret FROM w x",
    "WITH W AS (SELECT * FROM claim) SELECT secret FROM w",
    "WITH W AS (SELECT * FROM claim) SELECT id FROM w WHERE secret LIKE 'Z%'",
    "WITH W AS (SELECT id, secret FROM claim) SELECT w.secret FROM w",
    'WITH "W" AS (SELECT * FROM claim) SELECT w.secret FROM w',
    "WITH w AS (SELECT * FROM claim) SELECT W.secret FROM W",
    "WITH W AS (SELECT * FROM claim) SELECT (SELECT w.secret FROM w) AS s FROM claim",
    # a star over the folded reference expanded from the granted table's columns (the gate)
    "WITH W AS (SELECT * FROM claim) SELECT d.secret FROM (SELECT * FROM w) d",
    "WITH W AS (SELECT * FROM claim), v AS (SELECT * FROM w) SELECT secret FROM v",
    "WITH W AS (SELECT * FROM claim) SELECT secret FROM (SELECT w.* FROM w) d",
    # a nearer WITH declaring the name in another case shadows the outer CTE sqlglot matched
    # by spelling (Codex's first review of #174): the engine binds the inner one
    "WITH v AS (SELECT * FROM w) SELECT d.secret FROM (WITH V AS (SELECT * FROM claim) SELECT * FROM v) d",
    "WITH v AS (SELECT * FROM w) SELECT d.secret FROM (WITH V AS (SELECT * FROM claim) SELECT v.secret FROM v) d",
    "WITH v AS (SELECT * FROM w) SELECT secret FROM (WITH V AS (SELECT * FROM claim) SELECT secret FROM v) d",
    "WITH v AS (SELECT * FROM w), u AS (WITH V AS (SELECT * FROM claim) SELECT * FROM v) SELECT secret FROM u",
    "WITH V AS (SELECT * FROM w) SELECT d.secret FROM (WITH v AS (SELECT * FROM claim) SELECT * FROM V) d",
]


@pytest.mark.parametrize("sql", _FOLDED)
def test_a_cte_the_engine_binds_by_folding_stands_in_front_of_the_granted_table(worlds, sql):
    first, second = (con.execute(sql).fetchall() for con in worlds)
    assert first != second, (sql, first, second)  # the statement is an oracle on claim's secret
    for visible, policy in ((HIDDEN, None), (HIDDEN, AccessPolicy(policy_schema=SCHEMA))):
        verdict = _decide(sql, visible, policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    verdict = _decide(sql, SCHEMA, DENIED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNAUTHORIZED_COLUMN, (sql, verdict)
    verdict = _decide(sql, SCHEMA, MASKED)
    if isinstance(verdict, Approved):  # masked: the worlds answer alike; in a predicate, refused
        answers = [con.execute(verdict.target_sql).fetchall() for con in worlds]
        assert answers[0] == answers[1], (sql, verdict.target_sql, answers)
    else:
        assert verdict.code is RefusalCode.MASKED_COLUMN_IN_PREDICATE, (sql, verdict)
    assert isinstance(_decide(sql, SCHEMA, DENIED, dialect="oracle"), Refusal), sql  # Oracle folds both to W


def test_the_spelling_matched_as_written_still_binds(worlds):
    # The exact-case shapes M31 handled, unchanged: a control that the fold adds and takes nothing.
    sql = "WITH w AS (SELECT * FROM claim) SELECT w.secret FROM w"
    assert isinstance(_decide(sql, HIDDEN), Refusal)
    assert isinstance(_decide(sql, SCHEMA, DENIED), Refusal)
    masked = _decide(sql, SCHEMA, MASKED)
    assert isinstance(masked, Approved)
    assert [con.execute(masked.target_sql).fetchall() for con in worlds] == [[(None,)], [(None,)]]


def test_without_a_cte_the_granted_table_is_read(worlds):
    sql = "SELECT w.secret FROM w"
    verdict = _decide(sql, SCHEMA, DENIED)
    assert isinstance(verdict, Approved), verdict
    assert [con.execute(verdict.target_sql).fetchall() for con in worlds] == [[("w-own",)], [("w-own",)]]


def test_a_recursive_ctes_own_reference_binds_to_its_anchor_as_sqlglot_binds_it(monkeypatch):
    # The gate on the second commit: re-binding every reference bound the recursive branch's
    # `r` to the scope over the whole body, which holds that branch, a loop in the scope
    # graph; sqlglot binds it to the anchor's scope, and so does the fold, for `R` the same.
    import sqlglot
    from sqlglot.optimizer.scope import Scope
    from mnemiq.sql.scope import _bound_root, _root_scope
    # sqlglot binds it one step down, to the body's left operand: the anchor of a two-branch
    # body, `SELECT 1 AS id UNION ALL SELECT 2 AS id` of a three-branch one (the gate's third
    # round, where the leftmost branch matched nothing nearer and the whole body was taken).
    for spelling in ("r", "R"):
        for anchor in ("SELECT 1 AS id", "SELECT 1 AS id UNION ALL SELECT 2 AS id"):
            sql = f"WITH RECURSIVE r AS ({anchor} UNION ALL SELECT id + 1 FROM {spelling} WHERE id < 3) SELECT * FROM r"
            ast = sqlglot.parse_one(sql, read="duckdb")
            root = _bound_root(ast, "duckdb")
            recursive = next(s for s in root.traverse() if s.expression.sql().startswith("SELECT id + 1"))
            bound = recursive.sources[spelling]
            assert isinstance(bound, Scope) and bound.expression.sql() == anchor, (spelling, bound.expression.sql())
            # not a scope over an expression that holds the branch itself
            assert not any(node is recursive.expression for node in bound.expression.walk()), spelling
            if spelling == "r":  # the same-case reference is left exactly as sqlglot bound it
                plain = {id(s.expression): {n: id(getattr(v, "expression", v)) for n, v in s.sources.items()} for s in _root_scope(ast).traverse()}
                rebound = {id(s.expression): {n: id(getattr(v, "expression", v)) for n, v in s.sources.items()} for s in _bound_root(ast, "duckdb").traverse()}
                assert plain == rebound, anchor
    # A parenthesised anchor (Codex's second review): sqlglot's scope is over the SELECT inside
    # the parentheses, the wrapper matched nothing, the whole body was taken and
    # `fields_as_members` recursed without end. No scope holds the reference it binds.
    from mnemiq.sql.scope import fields_as_members
    for spelling in ("r", "R"):
        sql = f"WITH RECURSIVE r AS ((SELECT 1 AS id) UNION ALL SELECT id + 1 FROM {spelling} WHERE id < 3) SELECT (SELECT o.id FROM r) FROM (SELECT 1 AS id) o"
        ast = sqlglot.parse_one(sql, read="duckdb")
        assert fields_as_members(ast, "duckdb", {}) is None, spelling
        root = _bound_root(ast, "duckdb")
        for scope in root.traverse():
            for table in scope.tables:
                source = scope.sources.get(table.alias_or_name)
                if isinstance(source, Scope):
                    assert not any(node is table for node in source.expression.walk()), (spelling, table.sql())
        recursive = next(s for s in root.traverse() if s.expression.sql().startswith("SELECT id + 1"))
        assert recursive.sources[spelling].expression.sql() == "SELECT 1 AS id", spelling
    # The guard itself: were the scope over the whole body answered for the reference, the
    # binding is declined and the reference stays as sqlglot left it.
    import mnemiq.sql.scope as scope_module
    from sqlglot import exp
    ast = sqlglot.parse_one("WITH RECURSIVE r AS (SELECT 1 AS id UNION ALL SELECT id + 1 FROM R WHERE id < 3) SELECT * FROM r", read="duckdb")
    root = _root_scope(ast)
    whole = next(s for s in root.traverse() if isinstance(s.expression, exp.Union))
    monkeypatch.setattr(scope_module, "_cte_scope", lambda scope, cte: whole)
    scope_module._bind_folded_ctes(root, "duckdb")
    recursive = next(s for s in root.traverse() if s.expression.sql().startswith("SELECT id + 1"))
    assert isinstance(recursive.sources["R"], exp.Table)


def test_the_fold_is_the_engines_own():
    # Postgres keeps a quoted name's case and folds the unquoted reference to `w`, so `"W"` and
    # `w` are different objects there and `FROM w` reads the granted base table.
    sql = 'WITH "W" AS (SELECT * FROM claim) SELECT w.secret FROM w'
    assert isinstance(_decide(sql, HIDDEN, dialect="postgres"), Approved)
    assert isinstance(_decide(sql, HIDDEN, dialect="duckdb"), Refusal)

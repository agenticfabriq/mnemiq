"""`scope.cte_bindings` binds every reference in one pass down the tree as `_cte_binding` binds one.

The view walk asks whether a CTE binds a reference before looking it up as a view, and asked node by
node, each climbing to every WITH above it, the asking was quadratic in CTEs and in UNION branches,
which sqlglot nests one deep per branch (Codex's fourteenth to sixteenth reviews of #192). The pass
must answer exactly as the climb from each reference does, which the deciders' other checks still use.
"""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.sql.scope import _cte_binding, cte_bindings

SHAPES = [
    "WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a) SELECT x FROM b",
    # A later sibling: the base table on DuckDB's rule, the CTE on SQLite's.
    "WITH a AS (SELECT x FROM b), b AS (SELECT 1 AS x) SELECT x FROM a",
    "WITH t AS (SELECT x FROM t) SELECT x FROM t",
    # Without RECURSIVE a UNION body has no recursive part: its own name is the base table there.
    "WITH t AS (SELECT 1 AS x UNION ALL SELECT x FROM t) SELECT x FROM t",
    "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT n FROM r",
    # Only the last UNION ALL branch is the recursive part; INTERSECT has none.
    "WITH RECURSIVE r AS (SELECT n FROM r UNION ALL SELECT n FROM r UNION ALL SELECT n FROM r) SELECT n FROM r",
    "WITH RECURSIVE r AS (SELECT n FROM r INTERSECT SELECT n FROM r) SELECT n FROM r",
    "WITH RECURSIVE r AS ((SELECT 1 AS n) UNION ALL (SELECT n FROM r)) SELECT n FROM r",
    "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n FROM r UNION ALL (SELECT n FROM r)) SELECT n FROM r",
    "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n FROM (SELECT n FROM r UNION ALL SELECT n FROM r) AS z) "
    "SELECT n FROM r",
    "SELECT * FROM (WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n FROM (WITH q AS (SELECT n FROM r) "
    "SELECT n FROM q) AS z) SELECT n FROM r) AS w",
    # An earlier sibling of the same name is in scope before the CTE itself.
    "WITH RECURSIVE a AS (SELECT 1 AS x), a AS (SELECT 1 AS x UNION ALL SELECT x FROM a) SELECT x FROM a",
    "WITH RECURSIVE a AS (SELECT 1 AS n UNION ALL SELECT n FROM b), b AS (SELECT n FROM a) SELECT n FROM b",
    "WITH a AS (SELECT 1 AS x) SELECT x FROM (WITH a AS (SELECT 2 AS x) SELECT x FROM a) AS s, a",
    "WITH a AS (SELECT 1 AS x), b AS (WITH a AS (SELECT 2 AS x) SELECT x FROM a) SELECT x FROM b JOIN a USING (x)",
    "WITH a AS (SELECT 1 AS x) SELECT (SELECT x FROM a) AS y FROM a WHERE EXISTS (SELECT 1 FROM a)",
    "WITH a AS (SELECT 1 AS x) SELECT x FROM a UNION ALL SELECT x FROM a UNION ALL SELECT x FROM b",
    "WITH a AS (SELECT 1 AS x) SELECT x FROM main.a",
    "WITH a AS (SELECT 1 AS x), a AS (SELECT 2 AS x) SELECT x FROM a",
    'WITH "A" AS (SELECT 1 AS x) SELECT x FROM a, A, "A"',
    "WITH ſales AS (SELECT 1 AS x) SELECT x FROM SALES, sales",
    # M122: the scalar subquery's CTE is not in scope at the outer FROM.
    "UPDATE t SET a = (WITH claim AS (SELECT 1 AS a) SELECT a FROM claim) FROM claim",
    "WITH x AS (SELECT a FROM claim) INSERT INTO t SELECT a FROM x",
    "INSERT INTO t WITH x AS (SELECT a FROM claim) SELECT a FROM x",
    "WITH x AS (SELECT id FROM claim) DELETE FROM t WHERE id IN (SELECT id FROM x)",
    "MERGE INTO t USING (WITH s AS (SELECT 1 AS id) SELECT id FROM s) AS src ON t.id = src.id WHEN MATCHED THEN DELETE",
    "SELECT x FROM a",
]


@pytest.mark.parametrize("engine", [None, "sqlite"])
@pytest.mark.parametrize("dialect", ["duckdb", "oracle", "postgres", "sqlite"])
def test_the_one_pass_binds_every_reference_as_the_climb_from_it_does(dialect, engine):
    compared = 0
    for sql in SHAPES:
        try:
            ast = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            continue
        bound = cte_bindings(ast, dialect, engine=engine)
        for table in ast.find_all(exp.Table):
            assert bound.get(id(table)) is _cte_binding(table, dialect, engine), (sql, table.sql(dialect=dialect))
            compared += 1
    assert compared > 60, compared


def test_a_tree_deeper_than_the_recursion_limit_is_bound():
    # sqlglot nests a UNION of N branches N deep, and the pass runs without recursing.
    branches = " UNION ALL ".join(["SELECT x FROM a"] * 3000)
    ast = sqlglot.parse_one(f"WITH a AS (SELECT 1 AS x) {branches}", read="duckdb")
    cte = ast.find(exp.CTE)
    bound = cte_bindings(ast, "duckdb")
    assert len(bound) == 3000 and all(found is cte for found in bound.values())

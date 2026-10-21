"""M174 (Codex's review of #128): the value check misread three CTE shapes. A CTE column list
renames by position, and `WITH x(region, doc) AS (SELECT doc, region FROM t)` matched `x.doc`
by name to `t.doc` where it is `t.region`; three chained CTEs counted every earlier CTE among
a scope's sources and skipped the walk, so a wrong literal passed to a zero count; a CTE named
after its own base table was discarded by a statement-wide CTE-name set, the same. Each now
follows the scope's own answer.
"""

from __future__ import annotations

import duckdb
import sqlglot

from mnemiq.sql.values_check import check_values
from mnemiq.sql.verdict import Refusal, RefusalCode

VISIBLE = {"t": {"doc", "region"}}


class _Index:
    data = {("t", "doc"): {"a", "b"}}

    def has(self, table, column):
        return (table, column) in self.data

    def contains(self, table, column, literal):
        return literal in self.data.get((table, column), set())

    def nearest(self, table, column, literal, k=8):
        return sorted(self.data.get((table, column), set()))[:k]


def _check(sql: str):
    return check_values(sqlglot.parse_one(sql, read="duckdb"), VISIBLE, _Index())


def _refused_on_doc(sql: str) -> None:
    verdict = _check(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING, (sql, verdict)
    assert "t.doc" in verdict.message, verdict.message


def _runs_to_zero(sql: str) -> None:
    """The premise: the wrong literal executes and counts nothing, which a skipped check accepts."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    assert con.execute(sql).fetchall() == [(0,)], sql


def test_a_column_list_renames_by_position():
    # `x.doc` is the second projection, `t.region`, not indexed: checked against nothing.
    assert _check("WITH x(region, doc) AS (SELECT doc, region FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'") is None
    # and `x.region` is `t.doc`, whose values the literal is not among
    _refused_on_doc("WITH x(region, doc) AS (SELECT doc, region FROM t) SELECT count(*) FROM x WHERE x.region = 'west'")
    _refused_on_doc("SELECT count(*) FROM (SELECT doc, region FROM t) AS d(region, doc) WHERE d.region = 'west'")


def test_a_partial_column_list_keeps_the_unlisted_projections_by_their_own_names():
    # The gate's fourth round: `x(k)` names the first projection alone, and DuckDB keeps the
    # second as `doc`, which is `t.doc`.
    for sql in ("WITH x(k) AS (SELECT region, doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'",
                "SELECT count(*) FROM (SELECT region, doc FROM t) AS d(k) WHERE d.doc = 'west'"):
        _runs_to_zero(sql)
        _refused_on_doc(sql)
    # and the listed name is positional still: `x.k` is `t.region`, unindexed
    assert _check("WITH x(k) AS (SELECT region, doc FROM t) SELECT count(*) FROM x WHERE x.k = 'west'") is None


def test_a_star_among_the_listed_positions_leaves_the_positions_unknown():
    # The gate's fifth round: a star yields a column per source column, so `x(a, b)` over
    # `SELECT *, doc AS z FROM t` makes `b` the second star column, not `doc AS z`; the walk
    # gives up rather than map by a position it cannot know. A star past the list keeps the
    # name walk: `x(k)` over `SELECT region, * FROM t` still reaches `t.doc` through the star.
    assert _check("WITH x(a, b) AS (SELECT *, doc AS z FROM t) SELECT count(*) FROM x WHERE x.b = 'west'") is None
    assert _check("WITH x(a, b) AS (SELECT *, region AS z FROM t) SELECT count(*) FROM x WHERE x.b = 'west'") is None
    _refused_on_doc("WITH x(k) AS (SELECT region, * FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'")


def test_a_star_with_a_modifier_is_not_passed_through():
    # The same round: a REPLACE, a RENAME or an EXCLUDE changes what a name reaches, so the
    # walk gives up rather than read the raw column (`upper(doc)` is not among `t.doc`'s raw
    # values, a false refusal; `region AS doc` is `t.region`).
    assert _check("WITH x AS (SELECT * REPLACE (upper(doc) AS doc) FROM t) SELECT count(*) FROM x WHERE x.doc = 'A'") is None
    assert _check("WITH x AS (SELECT * RENAME (region AS doc) FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'") is None
    assert _check("WITH x AS (SELECT * EXCLUDE (doc) FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'") is None
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west')")
    assert con.execute("WITH x AS (SELECT * REPLACE (upper(doc) AS doc) FROM t) SELECT count(*) FROM x WHERE x.doc = 'A'").fetchall() == [(1,)]


def test_three_chained_ctes_are_followed_to_the_base_column():
    sql = "WITH a AS (SELECT 1 AS k), b AS (SELECT doc FROM t), c AS (SELECT doc FROM b) SELECT count(*) FROM c WHERE c.doc = 'west'"
    _runs_to_zero(sql)
    _refused_on_doc(sql)
    _refused_on_doc("WITH a AS (SELECT 1 AS k), b AS (SELECT doc FROM t), c AS (SELECT doc FROM b) SELECT count(*) FROM c WHERE doc = 'west'")


def test_a_cte_named_after_its_own_base_table_is_the_scopes_answer():
    sql = "WITH t AS (SELECT doc FROM t) SELECT count(*) FROM t WHERE t.doc = 'west'"
    _runs_to_zero(sql)
    _refused_on_doc(sql)
    _refused_on_doc("WITH t AS (SELECT doc FROM t) SELECT count(*) FROM t WHERE doc = 'west'")


def test_a_name_through_a_star_comes_from_the_scopes_one_selected_source():
    # The gate on the first draft: an unqualified name through `SELECT *` matched no projection
    # and lost its owner once the statement-wide lookup became the fallback.
    for sql in ("WITH x AS (SELECT * FROM t) SELECT count(*) FROM x WHERE doc = 'west'",
                "SELECT count(*) FROM (SELECT * FROM t) d WHERE doc = 'west'",
                "WITH x AS (SELECT doc FROM t), y AS (SELECT * FROM x) SELECT count(*) FROM y WHERE doc = 'west'",
                "WITH x AS (SELECT * FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'",
                "SELECT count(*) FROM (SELECT t.* FROM t) d WHERE d.doc = 'west'"):
        _runs_to_zero(sql)
        _refused_on_doc(sql)


def test_the_shapes_only_the_statement_wide_reading_answers_still_refuse():
    # The gate's second round: a star over a join, a set operation, a recursive CTE and a
    # correlated name resolve to no owner in their scope alone, and the statement-wide
    # reading, kept as the fallback, refuses each as before.
    for sql in ("WITH x AS (SELECT * FROM t JOIN u ON t.region = u.name) SELECT count(*) FROM x WHERE doc = 'west'",
                "WITH x AS (SELECT t.* FROM t JOIN u ON t.region = u.name) SELECT count(*) FROM x WHERE doc = 'west'",
                "WITH x AS (SELECT doc FROM t UNION ALL SELECT doc FROM t) SELECT count(*) FROM x WHERE doc = 'west'",
                "WITH RECURSIVE r AS (SELECT doc FROM t UNION ALL SELECT doc FROM r WHERE false) SELECT count(*) FROM r WHERE doc = 'west'",
                "SELECT count(*) FROM u WHERE EXISTS (SELECT 1 FROM t WHERE t.region = u.name AND doc = 'west')"):
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), {**VISIBLE, "u": {"name"}}, _Index())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)


def test_a_subquerys_own_column_is_read_in_its_own_scope():
    # The gate's third round: sqlglot lists an inner subquery's unqualified `doc` in the outer
    # scope too, and read there it was `t.doc`, refusing the right literal against the wrong
    # table; DuckDB binds it to `u.doc`, unindexed here, so nothing is checked.
    visible = {**VISIBLE, "u": {"name", "doc"}}
    for sql in ("SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM u WHERE doc = 'z')",
                "SELECT count(*) FROM t WHERE doc IN (SELECT doc FROM u WHERE doc = 'z')",
                "SELECT count(*) FROM t WHERE region IN (SELECT name FROM u WHERE doc = 'z')"):
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
        assert verdict is None or "u.doc" in verdict.message, (sql, verdict)
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'); "
                "CREATE TABLE u (name VARCHAR, doc VARCHAR); INSERT INTO u VALUES ('west', 'z')")
    assert con.execute("SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM u WHERE doc = 'z')").fetchall() == [(1,)]


def test_the_one_cte_control_and_a_present_literal_still_behave():
    _refused_on_doc("WITH x AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'")
    assert _check("WITH x AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'a'") is None
    assert _check("WITH x(region, doc) AS (SELECT doc, region FROM t) SELECT count(*) FROM x WHERE x.region = 'a'") is None

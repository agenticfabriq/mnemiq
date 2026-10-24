"""M174 (Codex's review of #128): the value check misread three CTE shapes. A CTE column list
renames by position, and `WITH x(region, doc) AS (SELECT doc, region FROM t)` matched `x.doc`
by name to `t.doc` where it is `t.region`; three chained CTEs counted every earlier CTE among
a scope's sources and skipped the walk, so a wrong literal passed to a zero count; a CTE named
after its own base table was discarded by a statement-wide CTE-name set, the same. Each now
follows the scope's own answer.
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest
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


def _check(sql: str, dialect: str = "duckdb"):
    return check_values(sqlglot.parse_one(sql, read=dialect), VISIBLE, _Index(), dialect=dialect)


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


def test_a_star_before_a_projection_of_the_same_name_yields_the_name_first():
    # Codex's first review of #178: `SELECT *, region AS doc FROM t` names `doc` twice and the
    # engine keeps the star's, renaming the later one (`doc_1` on DuckDB), so `t.doc` is
    # `t.doc`; the walk matched the alias by name and checked `t.region`, refusing 'a' and
    # passing 'west'. A star after the projection leaves the projection first.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    assert con.execute("WITH t AS (SELECT *, region AS doc FROM t) SELECT t.doc FROM t WHERE t.doc = 'a'").fetchall() == [("a",)]
    assert con.execute("WITH t AS (SELECT region AS doc, * FROM t) SELECT t.doc FROM t WHERE t.doc = 'west'").fetchall() == [("west",)]
    assert _check("WITH t AS (SELECT *, region AS doc FROM t) SELECT count(*) FROM t WHERE t.doc = 'a'") is None
    _refused_on_doc("WITH t AS (SELECT *, region AS doc FROM t) SELECT count(*) FROM t WHERE t.doc = 'west'")
    assert _check("WITH t AS (SELECT region AS doc, * FROM t) SELECT count(*) FROM t WHERE t.doc = 'west'") is None
    # a star whose source lacks the name leaves it to the later projection: `t.k` is `t.doc`
    _refused_on_doc("WITH t AS (SELECT *, doc AS k FROM t) SELECT count(*) FROM t WHERE t.k = 'west'")
    assert _check("WITH t AS (SELECT *, doc AS k FROM t) SELECT count(*) FROM t WHERE t.k = 'a'") is None
    # a star over a CTE projecting the name from a column outside the grant yields it (the
    # gate on the second round): the later projection is renamed, and nothing is checked
    assert _check("WITH a AS (SELECT hidden AS k, doc AS d FROM t), b AS (SELECT *, d AS k FROM a) SELECT count(*) FROM b WHERE b.k = 'west'") is None
    # and where the star's source is the base table itself, its columns decide
    _refused_on_doc("WITH b AS (SELECT *, doc AS k FROM t) SELECT count(*) FROM b WHERE b.k = 'west'")
    # a star over two sources before the name: not known either way, nothing checked
    assert _check("WITH t AS (SELECT *, u.region AS doc FROM t JOIN t u USING (doc)) SELECT count(*) FROM t WHERE t.doc = 'zzz'") is None


def test_a_base_table_alias_column_list_renames_by_position_and_is_not_placed():
    # The same review: `FROM t q(region, doc)` makes `q.doc` the second column, `t.region`;
    # the walk answered `t.doc` and refused 'west', which the engine returns.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    assert con.execute("WITH t AS (SELECT q.doc FROM t q(region, doc)) SELECT t.doc FROM t WHERE t.doc = 'west'").fetchall() == [("west",)]
    assert _check("WITH t AS (SELECT q.doc FROM t q(region, doc)) SELECT count(*) FROM t WHERE t.doc = 'west'") is None
    assert _check("SELECT count(*) FROM t q(region, doc) WHERE q.doc = 'west'") is None
    assert _check("SELECT count(*) FROM t q(region, doc) WHERE doc = 'west'") is None
    # the base table read elsewhere in the statement too (the gate on the second round): the
    # statement-wide reading would place the unqualified name by it, and does not
    assert con.execute("SELECT count(*) FROM t q(region, doc) WHERE doc = 'west' AND EXISTS (SELECT 1 FROM t)").fetchall() == [(1,)]
    assert _check("SELECT count(*) FROM t q(region, doc) WHERE doc = 'west' AND EXISTS (SELECT 1 FROM t)") is None
    assert _check("SELECT count(*) FROM t q(region, doc) WHERE doc = 'west' UNION ALL SELECT count(*) FROM t") is None


def test_names_meet_as_the_engine_folds_them():
    # The ui lane's probe of #178: Postgres reads `x.DOC` as `doc`, and the walk's exact
    # comparison answered nothing, a skip; the column list the same.
    for dialect in ("duckdb", "postgres"):
        for sql in ("WITH x AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.DOC = 'west'",
                    "WITH x(DOC) AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'",
                    "WITH x AS (SELECT * FROM t) SELECT count(*) FROM x WHERE x.Doc = 'west'"):
            verdict = _check(sql, dialect)
            assert isinstance(verdict, Refusal) and "t.doc" in verdict.message, (dialect, sql, verdict)


def test_a_quoted_name_meets_as_the_engine_resolves_it():
    # The ui lane's probe of #178 at `20753d6`, nine rows: Postgres keeps a quoted name's case,
    # so `"DOC"` beside `doc` is another column there, and the fold by `.lower()` read the
    # wrong projection (a present literal refused, an absent one approved, the wrong column
    # named); DuckDB folds a quoted name too and reads the first projection. The shared
    # resolver answers for each. Postgres's reading is the ui lane's measurement on its engine;
    # DuckDB's is run here.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    shapes = [
        # Postgres compares t.doc; DuckDB the first projection, t.region
        ('WITH x AS (SELECT region AS "DOC", doc FROM t) SELECT count(*) FROM x WHERE x.doc = \'west\'', 1),
        ('WITH x("DOC", doc) AS (SELECT region, doc FROM t) SELECT count(*) FROM x WHERE x.doc = \'west\'', 1),
        ('WITH x AS (SELECT region AS doc, doc AS "DOC" FROM t) SELECT count(*) FROM x WHERE x."DOC" = \'west\'', 1),
    ]
    for sql, duck_rows in shapes:
        assert con.execute(sql).fetchall() == [(duck_rows,)], sql
        assert _check(sql, "duckdb") is None, sql                      # t.region holds 'west'
        verdict = _check(sql, "postgres")
        assert isinstance(verdict, Refusal) and "t.doc" in verdict.message, (sql, verdict)   # t.doc does not
    # through a star, Postgres has no column `"DOC"` (the engine errors) and the check answers
    # nothing; DuckDB reads `doc`
    assert _check('WITH x AS (SELECT * FROM t) SELECT count(*) FROM x WHERE x."DOC" = \'west\'', "postgres") is None
    _refused_on_doc('WITH x AS (SELECT * FROM t) SELECT count(*) FROM x WHERE x."DOC" = \'west\'')


def test_a_catalog_name_meets_a_reference_as_created_and_as_the_engine_folds_it():
    # A snapshot spells a column as its author did (`Country` in a BIRD world decided for
    # SQLite, `region` for Oracle), so a stored name meets a reference as created and as the
    # engine folds an unquoted one, the grant check's reading; a quoted reference keeps its
    # case where the engine does (`"DOC"` names nothing on Postgres), and folds where it does not.
    class _Index:
        data = {("t", "Doc"): {"a", "b"}}

        def has(self, table, column):
            return (table, column) in self.data

        def contains(self, table, column, literal):
            return literal in self.data.get((table, column), set())

        def nearest(self, table, column, literal, k=8):
            return sorted(self.data.get((table, column), set()))[:k]

    visible = {"t": {"Doc", "region"}}
    for sql in ("SELECT count(*) FROM t WHERE doc = 'west'", "WITH x AS (SELECT * FROM t) SELECT count(*) FROM x WHERE x.doc = 'west'",
                'SELECT count(*) FROM t WHERE "Doc" = \'west\'', 'SELECT count(*) FROM t WHERE DOC = \'west\''):
        for dialect in ("postgres", "duckdb", "oracle", "sqlite"):
            verdict = check_values(sqlglot.parse_one(sql, read=dialect), visible, _Index(), dialect=dialect)
            assert isinstance(verdict, Refusal) and "t.Doc" in verdict.message, (dialect, sql, verdict)
    quoted = 'SELECT count(*) FROM t WHERE "DOC" = \'west\''
    assert check_values(sqlglot.parse_one(quoted, read="postgres"), visible, _Index(), dialect="postgres") is None
    verdict = check_values(sqlglot.parse_one(quoted, read="duckdb"), visible, _Index(), dialect="duckdb")
    assert isinstance(verdict, Refusal) and "t.Doc" in verdict.message, verdict


def test_a_using_key_is_the_joins_side():
    # Codex's second review of #178: the merged column of a USING join is the join's, and a
    # RIGHT join takes the right side's, `q.doc`, which is `t.region`; the walk discarded q
    # and took `t.doc` as the sole owner, refusing 'west' and passing 'a'. The ui lane
    # measured the sides on Postgres and DuckDB: inner and LEFT the left input's, RIGHT the
    # right's, FULL coalesced from both.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    join = "FROM t {side} JOIN t q(region, doc) USING (doc) WHERE doc = '{literal}'"
    assert con.execute("SELECT doc " + join.format(side="RIGHT", literal="west")).fetchall() == [("west",)]
    assert con.execute("SELECT doc " + join.format(side="LEFT", literal="west")).fetchall() == []
    assert con.execute("SELECT doc " + join.format(side="LEFT", literal="a")).fetchall() == [("a",)]
    assert con.execute("SELECT doc " + join.format(side="FULL", literal="west")).fetchall() == [("west",)]
    for literal in ("west", "a"):
        assert _check("SELECT count(*) " + join.format(side="RIGHT", literal=literal)) is None, literal   # q's, renamed by position
        assert _check("SELECT count(*) " + join.format(side="FULL", literal=literal)) is None, literal    # either side's
    _refused_on_doc("SELECT count(*) " + join.format(side="LEFT", literal="west"))                       # t's
    assert _check("SELECT count(*) " + join.format(side="LEFT", literal="a")) is None
    _refused_on_doc("SELECT count(*) FROM t JOIN t q(region, doc) USING (doc) WHERE doc = 'west'")       # inner: the left input's
    # a derived right side projecting the name from another column
    _refused_on_doc("SELECT count(*) FROM t LEFT JOIN (SELECT region AS doc FROM t) q USING (doc) WHERE doc = 'west'")
    assert _check("SELECT count(*) FROM t RIGHT JOIN (SELECT region AS doc FROM t) q USING (doc) WHERE doc = 'west'") is None
    # a chain: the later RIGHT join decides (the gate on the fourth round)
    chain = "FROM t JOIN t u(region, doc) USING (doc) RIGHT JOIN t v USING (doc) WHERE doc = 'west'"
    assert con.execute("SELECT doc " + chain).fetchall() == []   # v's doc, t.doc: no 'west'
    _refused_on_doc("SELECT count(*) " + chain)
    # a USING join inside a subquery is another scope's, and leaves the outer column's check on
    _refused_on_doc("SELECT count(*) FROM t WHERE doc = 'west' AND region IN (SELECT region FROM t a FULL JOIN t b USING (doc))")
    _refused_on_doc("SELECT count(*) FROM t WHERE doc = 'west' AND region IN (SELECT region FROM t a RIGHT JOIN t b(region, doc) USING (doc))")


def test_a_local_sources_name_the_walk_cannot_follow_stays_unresolved():
    # Codex's third review of #178: the inner `doc` is the USING output of a join inside the
    # derived table, which the walk cannot follow to one base column, and the outer unqualified
    # `doc` fell to the statement-wide reading, `t.doc`, where the engine compares `q.doc`,
    # `t.region`; qualified, `x.doc` was skipped, so the two spellings disagreed.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    sql = "SELECT doc FROM (SELECT doc FROM t RIGHT JOIN t q(region, doc) USING (doc)) x WHERE doc = 'west'"
    assert con.execute(sql).fetchall() == [("west",)]
    for spelling in ("doc", "x.doc"):
        for literal in ("west", "a"):
            assert _check(f"SELECT count(*) FROM (SELECT doc FROM t RIGHT JOIN t q(region, doc) USING (doc)) x WHERE {spelling} = '{literal}'") is None, (spelling, literal)
    # a local source projecting the name from one base column still answers, and a star or a
    # set operation leaves the statement-wide reading as before
    _refused_on_doc("SELECT count(*) FROM (SELECT doc FROM t) x WHERE doc = 'west'")
    _refused_on_doc("SELECT count(*) FROM (SELECT doc FROM t UNION ALL SELECT doc FROM t) x WHERE doc = 'west'")
    # a column list names the output by position, a function behind it (the gate on the fifth
    # round): the engine compares `upper(t.region)`, and the statement-wide reading said `t.doc`
    assert con.execute("WITH x(doc) AS (SELECT upper(region) FROM t) SELECT count(*) FROM x WHERE doc = 'WEST'").fetchall() == [(1,)]
    for literal in ("west", "WEST", "a"):
        assert _check(f"WITH x(doc) AS (SELECT upper(region) FROM t) SELECT count(*) FROM x WHERE doc = '{literal}'") is None, literal
        assert _check(f"SELECT count(*) FROM (SELECT upper(region) FROM t) AS x(doc) WHERE doc = '{literal}'") is None, literal
    # a list over a star leaves the positions unknown, and a listed name is the source's own
    # (Codex's fourth review of #178): `x(region, doc)` has `doc` for `t.region`, where the
    # statement-wide reading refused 'west' naming `t.doc`; `x(doc)` is `t.doc` by the position
    # alone, which the walk cannot know, so it skips too, as `x.doc` did (the gate's sixth
    # round had kept its refusal)
    assert con.execute("WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE doc = 'west'").fetchall() == [(1,)]
    assert con.execute("WITH x(doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE doc = 'west'").fetchall() == [(0,)]
    for shape in ("WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE {} = '{}'",
                  "SELECT count(*) FROM (SELECT * FROM t) x(region, doc) WHERE {} = '{}'",
                  "WITH x(region, doc) AS (SELECT *, 1 AS k FROM t) SELECT count(*) FROM x WHERE {} = '{}'",
                  "WITH x(doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE {} = '{}'"):
        for spelling in ("doc", "x.doc"):
            for literal in ("west", "a"):
                assert _check(shape.format(spelling, literal)) is None, (shape, spelling, literal)
    # the identity renaming skips too, its refusal, right by the position alone, given up on
    # purpose: the walk cannot tell it from `x(region, doc)` (the gate's seventh round)
    for literal in ("west", "a"):
        assert _check(f"SELECT count(*) FROM (SELECT * FROM t) d(doc, region) WHERE doc = '{literal}'") is None, literal
    # a projection the list renames away is no name of the output: `region AS doc` is `k`, and
    # the output's `doc` is `t.doc` through the star, refused as the statement-wide reading says
    assert con.execute("WITH x(k, m) AS (SELECT region AS doc, * FROM (SELECT region, doc FROM t) s) SELECT count(*) FROM x WHERE doc = 'west'").fetchall() == [(0,)]
    _refused_on_doc("WITH x(k, m) AS (SELECT region AS doc, * FROM (SELECT region, doc FROM t) s) SELECT count(*) FROM x WHERE doc = 'west'")
    # a projection past the star keeps its own name, the source's: `doc` is `upper(region)`
    assert con.execute("WITH x(k) AS (SELECT *, upper(region) AS doc FROM t) SELECT count(*) FROM x WHERE doc = 'WEST'").fetchall() == [(1,)]
    for literal in ("WEST", "a"):
        assert _check(f"WITH x(k) AS (SELECT *, upper(region) AS doc FROM t) SELECT count(*) FROM x WHERE doc = '{literal}'") is None, literal
    # a list renames the first projections away: x has no `doc`, and `doc` is t's
    assert con.execute("WITH x(k) AS (SELECT doc FROM t) SELECT count(*) FROM x, t WHERE doc = 'west'").fetchall() == [(0,)]
    _refused_on_doc("WITH x(k) AS (SELECT doc FROM t) SELECT count(*) FROM x, t WHERE doc = 'west'")


def test_a_natural_join_unsettles_only_the_name_both_sides_may_hold():
    # Codex's third review: a NATURAL join's keys are every name its sides share, and a RIGHT
    # one takes the right side's, `t NATURAL RIGHT JOIN (SELECT region AS doc FROM t) q`
    # compares `t.region` where the walk said `t.doc`. An inner or LEFT one takes the left
    # side's, `t.doc`, as a USING key does, and the absent value is refused (Codex's eighth
    # review: a skip on any NATURAL join let seven absent values through); a FULL one
    # coalesces both sides and answers nothing.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east')")
    assert con.execute("SELECT doc FROM t NATURAL RIGHT JOIN (SELECT region AS doc FROM t) q WHERE doc = 'west'").fetchall() == [("west",)]
    for side in ("", "LEFT "):
        assert con.execute(f"SELECT count(*) FROM t NATURAL {side}JOIN (SELECT region AS doc FROM t) q WHERE doc = 'west'").fetchall() == [(0,)]
        _refused_on_doc(f"SELECT count(*) FROM t NATURAL {side}JOIN (SELECT region AS doc FROM t) q WHERE doc = 'west'")
        assert _check(f"SELECT count(*) FROM t NATURAL {side}JOIN (SELECT region AS doc FROM t) q WHERE doc = 'a'") is None, side
    for side in ("RIGHT ", "FULL "):
        for literal in ("west", "a"):
            assert _check(f"SELECT count(*) FROM t NATURAL {side}JOIN (SELECT region AS doc FROM t) q WHERE doc = '{literal}'") is None, (side, literal)
    # a NATURAL join inside a subquery is another scope's, and leaves the outer column's check on
    _refused_on_doc("SELECT count(*) FROM t WHERE doc = 'west' AND region IN (SELECT region FROM t NATURAL JOIN t u)")


def test_a_natural_join_leaves_the_names_it_does_not_merge_alone():
    # Codex's eighth review of #178: a skip on any NATURAL join in the scope let seven absent
    # values through where main refused them; each compares `t.doc` on DuckDB and counts
    # nothing. A name the right side cannot hold the join merges nothing of, a correlated
    # scope included; one the left side cannot hold is the right side's; a FULL join's is
    # coalesced from both and answers nothing.
    visible = {**VISIBLE, "u": {"name"}, "v": {"doc", "k", "name"}, "w": {"k"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east'); "
                "CREATE TABLE u (name VARCHAR); INSERT INTO u VALUES ('n'); "
                "CREATE TABLE v (doc VARCHAR, k INTEGER, name VARCHAR); INSERT INTO v VALUES ('west', 1, 'n'); "
                "CREATE TABLE w (k VARCHAR); INSERT INTO w VALUES ('west')")
    for sql in ("SELECT count(*) FROM (SELECT * FROM (SELECT * FROM t JOIN u ON true) AS z) AS a NATURAL JOIN u AS b WHERE doc = 'west'",
                "SELECT count(*) FROM w NATURAL JOIN w AS b, t WHERE doc = 'west'",
                "SELECT count(*) FROM t JOIN t AS q USING (doc), w NATURAL JOIN w AS b WHERE doc = 'west'",
                "WITH x AS (SELECT doc FROM t NATURAL JOIN t AS b WHERE doc = 'west') SELECT count(*) FROM x",
                "SELECT count(*) FROM t NATURAL JOIN t AS b(region, doc) WHERE doc = 'west'",
                "SELECT count(*) FROM t NATURAL LEFT JOIN t AS b WHERE doc = 'west'",
                "SELECT count(*) FROM v WHERE EXISTS (SELECT 1 FROM t NATURAL RIGHT JOIN (SELECT region FROM t) AS b WHERE doc = 'west')"):
        assert con.execute(sql).fetchall() == [(0,)], sql
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)
    # A FULL join's merged column is coalesced from both sides: one answer where both resolve to
    # one base column (Codex's ninth review: `coalesce(t.doc, b.doc)` is `t.doc` twice), none
    # where they differ.
    for sql in ("SELECT count(*) FROM t NATURAL FULL JOIN t AS b WHERE doc = 'west'",
                "SELECT count(*) FROM t FULL JOIN t AS b USING (doc) WHERE doc = 'west'"):
        assert con.execute(sql).fetchall() == [(0,)], sql
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)
    sql = "SELECT count(*) FROM t NATURAL FULL JOIN (SELECT region AS doc FROM t) AS q WHERE doc = 'west'"
    assert con.execute(sql).fetchall() == [(1,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None
    # and none where a side cannot be named, or adds no owner under the grant (the review gate on
    # round 14): the coalesced column carries that side's values too, on either side
    for sql, granted in (("SELECT count(*) FROM t NATURAL FULL JOIN (SELECT * FROM (SELECT region AS doc FROM t) AS z CROSS JOIN u) AS q WHERE doc = 'west'", visible),
                         ("SELECT count(*) FROM t FULL JOIN v USING (doc) WHERE doc = 'west'", {**visible, "v": {"k", "name"}}),
                         ("SELECT count(*) FROM v FULL JOIN t AS b USING (doc) WHERE doc = 'west'", {**visible, "v": {"k", "name"}})):
        assert con.execute(sql).fetchall() == [(1,)], sql
        assert check_values(sqlglot.parse_one(sql, read="duckdb"), granted, _Index()) is None, sql
    # A source beside the FULL join that cannot hold the name is no side of it (the review gate on
    # round 15: counted as one, it blocked the answer and these skipped where main refused)
    for sql in ("SELECT count(*) FROM t FULL JOIN t AS b USING (doc), u WHERE doc = 'west'",
                "SELECT count(*) FROM t FULL JOIN t AS b USING (doc) CROSS JOIN w WHERE doc = 'west'",
                "SELECT count(*) FROM w, t FULL JOIN t AS b USING (doc) WHERE doc = 'west'",
                "SELECT count(*) FROM t NATURAL FULL JOIN t AS b, u WHERE doc = 'west'"):
        assert con.execute(sql).fetchall() == [(0,)], sql
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)
    # A comma binds more loosely than JOIN on DuckDB, so a source before it is no part of the FULL
    # join's left input even where it holds the name (the review gate on round 15: `v` among the
    # owners made them differ, and these skipped where main refused)
    for sql in ("SELECT count(*) FROM v, t FULL JOIN t AS b USING (doc) WHERE doc = 'west'",
                "SELECT count(*) FROM (SELECT region AS doc FROM t) AS q, t FULL JOIN t AS b USING (doc) WHERE doc = 'west'"):
        assert con.execute(sql).fetchall() == [(0,)], sql
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index(), dialect="duckdb")
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)
    # SQLite binds a comma like JOIN: the same statement is an ambiguous USING there, the parser
    # reads that comma as the CROSS JOIN it is, and the check keeps the sources before the join in
    # its left input, answering nothing
    lite = sqlite3.connect(":memory:")
    lite.executescript("CREATE TABLE t (doc TEXT, region TEXT); INSERT INTO t VALUES ('a', 'west'), ('b', 'east'); "
                       "CREATE TABLE v (doc TEXT, k INTEGER, name TEXT); INSERT INTO v VALUES ('west', 1, 'n')")
    sql = "SELECT count(*) FROM v, t FULL JOIN t AS b USING (doc) WHERE doc = 'west'"
    with pytest.raises(sqlite3.OperationalError, match="ambiguous"):
        lite.execute(sql)
    assert check_values(sqlglot.parse_one(sql, read="sqlite"), visible, _Index(), dialect="sqlite") is None
    # MySQL takes a bare JOIN, which binds like any JOIN and reads as the node of a comma, so there the
    # sources before it stay in the next join's left input: the merged `doc` is the left input's,
    # `t.doc` (the review gate on round 16; MySQL's grammar as documented, not measured here). Oracle's
    # CROSS APPLY reads as the node of a comma before a LATERAL, the same.
    for sql, engine in (("SELECT count(*) FROM t JOIN w NATURAL LEFT JOIN (SELECT region AS doc FROM t) AS q WHERE doc = 'west'", "mysql"),
                        ("SELECT count(*) FROM t CROSS APPLY (SELECT 1 AS k FROM dual) q FULL JOIN t b USING (doc) WHERE doc = 'west'", "oracle")):
        verdict = check_values(sqlglot.parse_one(sql, read=engine), visible, _Index(), dialect=engine)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)
    # A NATURAL join whose left input cannot hold the name under the grant makes the name its right
    # side's, and a FULL join before it is superseded (the review gate on round 15: the FULL join's
    # stale inputs answered nothing where the inner join's merged values are the right side's)
    hidden = {**visible, "v": {"k", "name"}}
    sql = "SELECT count(*) FROM v FULL JOIN v AS b USING (doc) NATURAL JOIN (SELECT doc FROM t) AS q WHERE doc = 'west'"
    assert con.execute(sql).fetchall() == [(0,)]
    verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), hidden, _Index(), dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, verdict
    sql = "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT name FROM u) AS a NATURAL JOIN v AS b WHERE doc = 'west')"
    assert con.execute(sql).fetchall() == [(2,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None
    # A join's left side is the sources joined before it: a later source holding the name does
    # not make an earlier NATURAL join a merge (the review gate on round 13: `doc` is `b.doc`,
    # `t.region`, and read with `c` on the left the first join dropped `b`, its only holder).
    sql = ("SELECT count(*) FROM w NATURAL JOIN (SELECT region AS doc, 'west' AS k FROM t) AS b "
           "NATURAL LEFT JOIN t AS c WHERE doc = 'west'")
    assert con.execute(sql).fetchall() == [(1,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None


def test_the_statement_wide_reading_returns_the_catalogs_spelling():
    # The gate on the fourth round: matched by fold, the fallback returned the name as typed,
    # which the index, keyed by the catalog's spelling, never held.
    class _Index:
        data = {("t", "Doc"): {"a", "b"}}

        def has(self, table, column):
            return (table, column) in self.data

        def contains(self, table, column, literal):
            return literal in self.data.get((table, column), set())

        def nearest(self, table, column, literal, k=8):
            return sorted(self.data.get((table, column), set()))[:k]

    visible = {"t": {"Doc", "region"}}
    sql = "SELECT count(*) FROM (SELECT * FROM t UNION ALL SELECT * FROM t) WHERE doc = 'west'"
    verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index(), dialect="duckdb")
    assert isinstance(verdict, Refusal) and "t.Doc" in verdict.message, (sql, verdict)
    # qualified, the derived source's own answer is nothing (a set operation), and stays so
    assert check_values(sqlglot.parse_one(sql.replace(") WHERE doc", ") u WHERE u.doc"), read="duckdb"), visible, _Index(), dialect="duckdb") is None


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
    # The gate's second round: a star over a join, a set operation and a recursive CTE resolve
    # to no owner in their scope alone, and the statement-wide reading, kept as the fallback,
    # refuses each as before.
    for sql in ("WITH x AS (SELECT * FROM t JOIN u ON t.region = u.name) SELECT count(*) FROM x WHERE doc = 'west'",
                "WITH x AS (SELECT t.* FROM t JOIN u ON t.region = u.name) SELECT count(*) FROM x WHERE doc = 'west'",
                "WITH x AS (SELECT doc FROM t UNION ALL SELECT doc FROM t) SELECT count(*) FROM x WHERE doc = 'west'",
                "WITH RECURSIVE r AS (SELECT doc FROM t UNION ALL SELECT doc FROM r WHERE false) SELECT count(*) FROM r WHERE doc = 'west'"):
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


def test_a_correlated_name_is_read_in_the_enclosing_scopes_before_the_statement_wide_reading():
    # Codex's sixth review of #178: `doc` inside the EXISTS names no column of `u`, so DuckDB
    # binds it to the enclosing scope's `x.doc`, `t.region` through the column list over a star,
    # whose base column the walk leaves unknown; read statement-wide it was `t.doc`, and the
    # present value 'west' was refused. The same through two levels of subquery, a LATERAL
    # derived table, a derived table DuckDB correlates without the keyword, and a scalar
    # subquery in the select list; the qualified `x.doc` skips the same way.
    visible = {**VISIBLE, "u": {"name"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east'); "
                "CREATE TABLE u (name VARCHAR); INSERT INTO u VALUES ('n')")
    for sql in ("WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE EXISTS (SELECT 1 FROM u WHERE doc = 'west')",
                "WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE EXISTS (SELECT 1 FROM u WHERE x.doc = 'west')",
                "WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x "
                "WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM u AS u2 WHERE doc = 'west'))",
                "WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x, LATERAL (SELECT 1 FROM u WHERE doc = 'west') AS d",
                "WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x, (SELECT 1 FROM u WHERE doc = 'west') AS d"):
        assert con.execute(sql).fetchall() == [(1,)], sql
        assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None, sql
    sql = "WITH x(region, doc) AS (SELECT * FROM t) SELECT (SELECT count(*) FROM u WHERE doc = 'west') FROM x"
    assert con.execute(sql).fetchall() == [(1,), (0,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None


def test_a_correlated_name_an_enclosing_base_table_holds_is_checked_against_it():
    # The controls: the enclosing scope's base table owns the name, under its alias too, and
    # the wrong literal counts nothing there, so the check refuses naming `t.doc`, by the walk
    # through the scopes and not by the statement-wide reading; and a name the subquery's own
    # source holds is that source's, `t.doc` under a correlated `u.name` (the gate's second
    # round). The inner scope's own source wins where both hold the name
    # (`test_a_subquerys_own_column_is_read_in_its_own_scope`).
    visible = {**VISIBLE, "u": {"name"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east'); "
                "CREATE TABLE u (name VARCHAR); INSERT INTO u VALUES ('n')")
    for sql in ("SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM u WHERE doc = 'west')",
                "SELECT count(*) FROM t AS q WHERE EXISTS (SELECT 1 FROM u WHERE q.doc = 'west')",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM u AS u2 WHERE doc = 'west'))",
                "SELECT count(*) FROM u WHERE EXISTS (SELECT 1 FROM t WHERE t.region = u.name AND doc = 'west')"):
        assert con.execute(sql).fetchall() == [(0,)], sql
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)


def test_a_source_whose_output_the_walk_cannot_name_keeps_the_statement_wide_reading():
    # The review gate on round 9: a derived table over a star over a join, or over a set
    # operation, may yield the name, and the engine binds there (`y.doc` is `v.doc`, unindexed),
    # so the scope is not passed for the enclosing one, where `t` would own the name and refuse
    # a present value; the statement-wide reading, kept, sees `t` and `v` and skips as before.
    # A source naming the column as an expression, through a star or not (`doc || '' AS doc`,
    # the gate's second round on this change), holds the name, and nothing is checked.
    visible = {**VISIBLE, "u": {"name"}, "v": {"doc", "k", "name"}, "w": {"k"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'); "
                "CREATE TABLE u (name VARCHAR); INSERT INTO u VALUES ('n'); "
                "CREATE TABLE v (doc VARCHAR, k INTEGER, name VARCHAR); INSERT INTO v VALUES ('west', 1, 'n'); "
                "CREATE TABLE w (k VARCHAR); INSERT INTO w VALUES ('west')")
    for sql in ("SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM u JOIN v ON true) AS y WHERE doc = 'west')",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT doc FROM v UNION ALL SELECT doc FROM v) AS y WHERE doc = 'west')",
                "WITH y AS (SELECT 1 AS k FROM (SELECT * FROM u JOIN v ON true) AS z WHERE doc = 'west') SELECT count(*) FROM t, y",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM (SELECT * FROM u JOIN v ON true) AS z) AS y WHERE doc = 'west')",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM (SELECT doc || '' AS doc FROM v) AS z) AS y WHERE doc = 'west')",
                "WITH y AS (SELECT 1 AS k FROM (SELECT * FROM (SELECT doc || '' AS doc FROM v) AS z) AS q WHERE doc = 'west') SELECT count(*) FROM t, y",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT k || '' AS doc FROM w) AS y WHERE doc = 'west')"):
        assert con.execute(sql).fetchall() == [(1,)], sql
        assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None, sql
    # Where no table of the statement but `t` holds the name, the statement-wide reading
    # refuses as before, the engine binding the name to `t` through the correlation: a star
    # over a join and a set operation, each over tables lacking it; and a plain star over a
    # source the walk names as lacking it, two levels down, is absent for sure.
    visible = {**VISIBLE, "u": {"name"}, "w": {"k"}}
    for sql in ("SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM u JOIN w ON true) AS y WHERE doc = 'west')",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT k FROM w UNION ALL SELECT k FROM w) AS y WHERE doc = 'west')",
                "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM (SELECT k FROM w) AS z) AS y WHERE doc = 'west')"):
        assert con.execute(sql).fetchall() == [(0,)], sql
        verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, (sql, verdict)
    # A plain star over one base table the walk names, so its absence is sure and the
    # enclosing scope's column list over a star skips (`y` has no `doc`; `x.doc` is `t.region`).
    visible = {**VISIBLE, "u": {"name"}}
    sql = "WITH x(region, doc) AS (SELECT * FROM t) SELECT count(*) FROM x WHERE EXISTS (SELECT 1 FROM (SELECT * FROM u) AS y WHERE doc = 'west')"
    assert con.execute(sql).fetchall() == [(1,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None


def test_a_natural_join_beside_sources_lacking_the_name_leaves_it_to_the_enclosing_scope():
    # Codex's seventh review of #178: `EXISTS (SELECT 1 FROM w NATURAL JOIN w AS b WHERE doc =
    # 'west')` compares the outer `t.doc`, neither `w` holding `doc`, and the NATURAL skip let an
    # absent value through where the statement-wide reading had refused it. Beside a source
    # holding the name the merged column is the join's, unsettled, and the skip stands.
    visible = {**VISIBLE, "v": {"doc", "k", "name"}, "w": {"k"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'); "
                "CREATE TABLE v (doc VARCHAR, k INTEGER, name VARCHAR); INSERT INTO v VALUES ('west', 1, 'n'); "
                "CREATE TABLE w (k VARCHAR); INSERT INTO w VALUES ('west')")
    sql = "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM w NATURAL JOIN w AS b WHERE doc = 'west')"
    assert con.execute(sql).fetchall() == [(0,)]
    verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, verdict
    sql = "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM v NATURAL JOIN v AS b WHERE doc = 'west')"
    assert con.execute(sql).fetchall() == [(1,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None
    # and a base table renamed by position may hold the name under any spelling (`z(doc)` is `w.k`)
    sql = "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM w AS z(doc) NATURAL JOIN w AS b(doc) WHERE doc = 'west')"
    assert con.execute(sql).fetchall() == [(1,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None
    # A local source beside the join is read through `_names` (the review gate on round 10): one
    # that cannot hold the name leaves it to the enclosing scope, one naming it as an expression
    # through a star keeps the skip.
    sql = "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT k FROM w) AS a NATURAL JOIN w AS b WHERE doc = 'west')"
    assert con.execute(sql).fetchall() == [(0,)]
    verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, verdict
    # the join's key is `k`, and only `a`, through its star, can hold `doc` (the gate on round 11)
    sql = ("SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM (SELECT doc || '' AS doc, k FROM v) AS z) AS a "
           "NATURAL JOIN (SELECT k FROM v) AS b WHERE doc = 'west')")
    assert con.execute(sql).fetchall() == [(1,)]
    assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None


def test_a_joins_side_whose_output_the_walk_cannot_name_is_not_read_statement_wide():
    # Codex's ninth review of #178: a RIGHT join made the name `q`'s, `q`'s output a star over a
    # join (or a union) the walk cannot name, and the statement-wide reading, kept for a scope
    # that cannot tell, resurrected the discarded left input and refused a present value against
    # `t.doc` where DuckDB compares `q.doc`, `t.region`. A merged column is one side's: where
    # that side cannot be named the scope answers nothing.
    visible = {**VISIBLE, "u": {"name"}}
    con = duckdb.connect()
    con.execute("CREATE TABLE t (doc VARCHAR, region VARCHAR); INSERT INTO t VALUES ('a', 'west'), ('b', 'east'); "
                "CREATE TABLE u (name VARCHAR); INSERT INTO u VALUES ('n')")
    for sql, rows in (("SELECT count(*) FROM t NATURAL RIGHT JOIN (SELECT * FROM (SELECT region AS doc FROM t) AS z CROSS JOIN u) AS q WHERE doc = 'west'", 1),
                      ("SELECT count(*) FROM t NATURAL RIGHT JOIN (SELECT * FROM (SELECT region AS doc FROM t UNION ALL SELECT region AS doc FROM t) AS z) AS q WHERE doc = 'west'", 2),
                      ("SELECT count(*) FROM t RIGHT JOIN (SELECT * FROM (SELECT region AS doc FROM t) AS z CROSS JOIN u) AS q USING (doc) WHERE doc = 'west'", 1),
                      # the left side cannot hold the name, the right side cannot be named: the name is the right's, and nothing is checked
                      ("SELECT count(*) FROM (SELECT name FROM u) AS a NATURAL JOIN (SELECT * FROM (SELECT region AS doc, 'n' AS name FROM t) AS z CROSS JOIN u) AS b WHERE doc = 'west'", 1)):
        assert con.execute(sql).fetchall() == [(rows,)], sql
        assert check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index()) is None, sql
    # and the left side's own holder still answers where the join leaves the name there
    sql = "SELECT count(*) FROM t NATURAL JOIN (SELECT * FROM (SELECT region AS doc FROM t) AS z CROSS JOIN u) AS q WHERE doc = 'west'"
    assert con.execute(sql).fetchall() == [(0,)]
    verdict = check_values(sqlglot.parse_one(sql, read="duckdb"), visible, _Index())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING and "t.doc" in verdict.message, verdict

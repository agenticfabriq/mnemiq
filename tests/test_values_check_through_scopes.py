"""M141: the value check reads a column through the CTE or derived table that renamed it.

`check_values` mapped qualifiers from base-table nodes only, so `x.doc` with `x` a CTE, or `s.d`
with `s` a derived table, resolved to nothing and was skipped. A skipped wrong literal executes;
`count(*)` then returns a single 0, which the sanity check accepts as an answer. A QUALIFIED column
is now followed through each scope's projection to the base column it reads, and skipped where a
projection is not a plain column (a function, a union) -- the check still never guesses. An
unqualified one is still resolved by name against the base tables, as before.
"""

from __future__ import annotations

import sqlglot

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
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


def _check(sql: str, row_filtered=None):
    return check_values(sqlglot.parse_one(sql, read="duckdb"), VISIBLE, _Index(), row_filtered)


def _refused(sql: str) -> Refusal:
    verdict = _check(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING, sql
    assert "t.doc" in verdict.message and "include: a, b" in verdict.message, verdict.message
    return verdict


def test_a_cte_column_is_checked_against_its_base_column():
    _refused("WITH x AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'nope'")


def test_a_derived_table_column_is_checked_through_its_rename():
    _refused("SELECT count(*) FROM (SELECT doc AS d FROM t) s WHERE s.d = 'nope'")


def test_a_rename_of_a_qualified_column_is_followed():
    _refused("WITH x AS (SELECT a.doc AS d FROM t a) SELECT 1 FROM x WHERE x.d = 'nope'")


def test_a_chain_of_ctes_and_an_alias_inside_one_are_followed():
    _refused("WITH x AS (SELECT a.doc FROM t a), y AS (SELECT doc FROM x) "
             "SELECT 1 FROM y WHERE y.doc IN ('a', 'nope')")


def test_a_present_literal_through_a_cte_passes():
    assert _check("WITH x AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'a'") is None


def test_a_rename_is_followed_to_the_column_it_reads_not_the_one_it_is_called():
    """`x.doc` is `t.region` here, which is not indexed; read by its name it would be `t.doc`,
    and 'west' would be refused against values of a different column."""
    sql = "WITH x AS (SELECT region AS doc FROM t) SELECT 1 FROM x WHERE x.doc = 'west'"
    assert _check(sql) is None


def test_a_projection_that_is_not_a_plain_column_is_not_guessed():
    for sql in ("WITH x AS (SELECT upper(doc) AS doc FROM t) SELECT 1 FROM x WHERE x.doc = 'Q'",
                "WITH x AS (SELECT doc FROM t UNION ALL SELECT 'z') "
                "SELECT 1 FROM x WHERE x.doc = 'z'"):
        assert _check(sql) is None, sql


def test_a_row_filtered_base_is_not_listed_through_a_cte_either():
    """M5 holds on the new path: the value index is unfiltered, so a row-filtered table's real
    values are withheld however the column is reached."""
    verdict = _check("WITH x AS (SELECT doc FROM t) SELECT 1 FROM x WHERE x.doc = 'nope'",
                     row_filtered={"t"})
    assert isinstance(verdict, Refusal) and "include" not in verdict.message
    assert "row-filtered" in verdict.message


def test_the_decider_refuses_the_count_that_would_have_answered_zero():
    """The join: `decide` hands the check its index, as the agent loop calls it."""
    verdict = decide("WITH x AS (SELECT doc FROM t) SELECT count(*) FROM x WHERE x.doc = 'nope'",
                     VISIBLE, dialect="duckdb", target="duckdb", values=_Index(),
                     policy=AccessPolicy(policy_schema=VISIBLE))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING, verdict


def test_a_masked_column_reached_through_a_cte_lists_nothing():
    """Review gate, the blocker on the first version: `check_cls` refuses a masked column in a
    predicate only where it is read directly, so following a CTE to it listed its raw values."""
    policy = AccessPolicy(masked={("t", "doc")}, policy_schema=VISIBLE)
    for sql in ("WITH x AS (SELECT doc FROM t) SELECT 1 FROM x WHERE x.doc = 'nope'",
                "SELECT s.d FROM (SELECT doc AS d FROM t) s WHERE s.d = 'nope'"):
        verdict = decide(sql, VISIBLE, dialect="duckdb", target="duckdb", values=_Index(),
                         policy=policy)
        assert not (isinstance(verdict, Refusal) and verdict.code is RefusalCode.VALUE_GROUNDING), (
            sql, verdict)

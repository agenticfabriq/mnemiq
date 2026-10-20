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

SCHEMA = {"claim": {"id", "secret"}, "pair": {"id", "secret"}, "one": {"secret"}, "sink": {"i", "s"}, "w": {"id"}}
HIDDEN = {"claim": {"id"}, "pair": {"id"}, "one": set(), "sink": {"i", "s"}, "w": {"id"}}
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
                    "CREATE TABLE sink (i INT, s VARCHAR); INSERT INTO sink VALUES (1, 'x'); "
                    "CREATE TABLE w (id INT); INSERT INTO w VALUES (1)")
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
    # A recursive CTE named after the base table reads the base table in its anchor (Codex's
    # fourth review): the whole body had been taken as the CTE and the statement as one over
    # CTEs alone, so the grant check returned before the join arm ran.
    "WITH RECURSIVE claim AS (SELECT id, secret FROM claim UNION ALL SELECT id, secret FROM claim WHERE false) SELECT count(*) AS n FROM claim JOIN (SELECT 'Z' AS secret) e USING (secret)",
    # A CTE named in another case than its reference, beside a granted base table of the
    # reference's spelling (Codex's third review): DuckDB folds, and reads the CTE.
    "WITH W AS (SELECT * FROM claim) SELECT count(*) AS n FROM w JOIN (SELECT 'Z' AS secret) e USING (secret)",
    "WITH W AS (SELECT * FROM claim) SELECT count(*) AS n FROM w NATURAL JOIN (SELECT 1 AS id, 'Z' AS secret) e",
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
    "WITH RECURSIVE claim AS (SELECT id, secret FROM claim UNION ALL SELECT id, secret FROM claim WHERE false) INSERT INTO sink (i) SELECT count(*) FROM claim JOIN (SELECT 'Z' AS secret) e USING (secret)",
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


def test_a_qualified_reference_is_the_base_table_whatever_a_cte_is_called():
    # The review gate on the CTE folding: `pg.claim` is the base table even beside a CTE named
    # `claim`, since a qualified name never names a CTE. A federated grant keyed `pg.claim`;
    # the decision alone, with no engine behind the catalog name.
    schema = {"pg.claim": {"id", "secret"}}
    hidden = {"pg.claim": {"id"}}
    for sql in ("WITH claim AS (SELECT 1 AS secret) SELECT count(*) AS n FROM pg.claim JOIN (SELECT 'Z' AS secret) e USING (secret)",
                "WITH claim AS (SELECT 1 AS id) SELECT count(*) AS n FROM pg.claim NATURAL JOIN (SELECT 1 AS id, 'Z' AS secret) e"):
        verdict = _decide(sql, hidden, policy=AccessPolicy(policy_schema=schema))
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    granted = _decide("SELECT count(*) AS n FROM pg.claim JOIN (SELECT 1 AS id) e USING (id)", schema, policy=AccessPolicy(policy_schema=schema))
    assert isinstance(granted, Approved), granted


def test_a_recursive_cte_named_after_the_base_table_expands_its_anchor_star_over_the_base_table(worlds):
    # M201 (the review gate on #170's fourth commit): the star resolver bound the recursive
    # name across the anchor too, so the star over the CTE was vouched for by nothing and the
    # hidden column came back.
    body = "WITH RECURSIVE claim AS (SELECT * FROM claim UNION ALL SELECT * FROM claim WHERE false) "
    star = body + "SELECT * FROM claim"
    assert [con.execute(star).fetchall() for con in worlds] == [[(1, "Z")], [(1, "A")]]
    verdict = _decide(star, HIDDEN)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, verdict
    assert isinstance(_decide(star, SCHEMA), Refusal)  # as `SELECT * FROM claim` is
    # A named read through the CTE is refused as well, by the positional rule (M183): the body
    # is a set operation of stars, which bind by position.
    for named in ("SELECT claim.id FROM claim", "SELECT claim.secret FROM claim"):
        assert isinstance(_decide(body + named, HIDDEN), Refusal), named
    counter = "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3) SELECT n FROM t"
    assert worlds[0].execute(counter).fetchall() == [(1,), (2,), (3,)]
    assert isinstance(_decide(counter, HIDDEN), Approved)


def test_on_oracle_a_quoted_cte_name_is_not_the_unquoted_reference():
    # Codex's fourth review: Oracle keeps `"claim"` and upper-folds `claim`, so the quoted CTE
    # does not stand in front of the base table the unquoted reference reads. Decision only.
    schema = {"claim": {"ID", "SECRET"}, "sink": {"I"}}
    hidden = {"claim": {"ID"}, "sink": {"I"}}
    for sql in ('WITH "claim" AS (SELECT I FROM sink) SELECT count(*) AS n FROM claim NATURAL JOIN (SELECT 1 AS ID, chr(90) AS SECRET FROM sink) e',
                'WITH "claim" AS (SELECT I FROM sink) SELECT count(*) AS n FROM claim JOIN (SELECT chr(90) AS SECRET FROM sink) e USING (SECRET)'):
        verdict = decide(sql, hidden, dialect="oracle", target="oracle", policy=AccessPolicy(policy_schema=schema))
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_whole_row_as_an_order_by_key_over_a_derived_star_stays_refused(worlds):
    # The ui lane's replay of #169: sqlglot calls `t` and `t.*` bare columns, DuckDB reads the
    # whole row, withheld column included; another check refuses both, pinned here.
    for sql in ("SELECT d.id FROM (SELECT * FROM pair t ORDER BY t LIMIT 1) d",
                "SELECT d.id FROM (SELECT * FROM pair t ORDER BY t.* LIMIT 1) d"):
        verdict = _decide(sql, HIDDEN)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


# The ui lane's replay of #170 at 3e98092: the recursive rule bound a CTE's own name on the right
# side of ANY set operation, and the name resolver bound it across the whole body as sqlglot does.
# To DuckDB only a UNION makes a CTE recursive, and only the LAST branch is the recursive part;
# everywhere else in the body the name is the base table (measured: each shape here is an oracle
# on the hidden column).
_OUTSIDE_THE_RECURSIVE_PART = [
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret INTERSECT SELECT id, secret FROM claim) SELECT id FROM claim", RefusalCode.UNKNOWN_COLUMN),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret INTERSECT ALL SELECT id, secret FROM claim) SELECT id FROM claim", RefusalCode.UNKNOWN_COLUMN),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret EXCEPT SELECT id, secret FROM claim) SELECT id FROM claim", RefusalCode.UNKNOWN_COLUMN),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret EXCEPT ALL SELECT id, secret FROM claim) SELECT id FROM claim", RefusalCode.UNKNOWN_COLUMN),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret INTERSECT SELECT * FROM claim) SELECT id FROM claim", RefusalCode.SELECT_STAR),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret EXCEPT SELECT * FROM claim) SELECT id FROM claim", RefusalCode.SELECT_STAR),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret UNION ALL SELECT id, secret FROM claim UNION ALL SELECT id, secret FROM claim WHERE false) SELECT secret FROM claim", RefusalCode.UNKNOWN_COLUMN),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id UNION ALL SELECT id FROM claim WHERE secret = 'Z' UNION ALL SELECT id FROM claim WHERE false) SELECT id FROM claim", RefusalCode.UNKNOWN_COLUMN),
    ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret UNION ALL SELECT * FROM claim UNION ALL SELECT * FROM claim WHERE false) SELECT secret FROM claim", RefusalCode.SELECT_STAR),
    ("WITH RECURSIVE claim AS (SELECT id, secret FROM claim UNION ALL SELECT id, secret FROM claim WHERE false) SELECT secret FROM claim", RefusalCode.UNKNOWN_COLUMN),
]


@pytest.mark.parametrize("sql, code", _OUTSIDE_THE_RECURSIVE_PART)
def test_a_recursive_ctes_own_name_is_the_base_table_outside_a_unions_last_branch(worlds, sql, code):
    first, second = (con.execute(sql).fetchall() for con in worlds)
    assert first != second, (sql, first, second)
    for policy in (COMPLETE, AccessPolicy()):
        verdict = _decide(sql, HIDDEN, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is code, (sql, verdict)
    denied = AccessPolicy(denied={("claim", "secret")}, policy_schema=SCHEMA)
    assert isinstance(_decide(sql, SCHEMA, policy=denied), Refusal), sql
    # Masked or row-filtered, the base read inside the body is rewritten at the source, or the
    # statement is refused: either way the two worlds answer alike.
    for policy in (AccessPolicy(masked={("claim", "secret")}, policy_schema=SCHEMA),
                   AccessPolicy(row_filters={"claim": "secret = 'Q'"}, policy_schema=SCHEMA)):
        verdict = _decide(sql, SCHEMA, policy=policy)
        if isinstance(verdict, Approved):
            answers = [con.execute(verdict.target_sql).fetchall() for con in worlds]
            assert answers[0] == answers[1], (sql, verdict.target_sql, answers)


def test_a_unions_last_branch_binds_the_recursive_cte_itself(worlds):
    # The control: in the recursive part the name is the CTE, whose `secret` is the anchor's
    # literal, so nothing hidden is read and the two worlds agree; a parenthesised right side
    # is the recursive part whole.
    for sql in ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret UNION ALL SELECT id, secret FROM claim WHERE secret = 'Q') SELECT secret FROM claim",
                "WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret UNION SELECT id, secret FROM claim WHERE secret = 'Q') SELECT secret FROM claim",
                "WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret UNION ALL (SELECT id, secret FROM claim WHERE secret = 'Q' UNION ALL SELECT id, secret FROM claim WHERE false)) SELECT secret FROM claim"):
        assert [con.execute(sql).fetchall() for con in worlds] == [[("Z",)], [("Z",)]], sql
        assert isinstance(_decide(sql, HIDDEN), Approved), sql


def test_a_recursive_body_outside_the_unions_last_branch_in_a_write_is_refused(worlds):
    for sql in ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret INTERSECT SELECT id, secret FROM claim) INSERT INTO sink (i) SELECT id FROM claim",
                "WITH RECURSIVE claim AS (SELECT 1 AS id UNION ALL SELECT id FROM claim WHERE secret = 'Z' UNION ALL SELECT id FROM claim WHERE false) INSERT INTO sink (i) SELECT id FROM claim"):
        for con in worlds:
            con.execute(sql)
        first, second = (con.execute("SELECT * FROM sink").fetchall() for con in worlds)
        assert first != second, (sql, first, second)
        verdict = _decide_write(sql, HIDDEN)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_mixed_set_operation_body_is_read_as_the_parser_groups_it(worlds):
    # The review gate on the sixth commit: sqlglot groups an unparenthesised chain left to
    # right, where DuckDB binds INTERSECT tighter than UNION and EXCEPT (measured). Where the
    # two agree the shape is an oracle and refused as a hidden read.
    for sql in ("WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret INTERSECT SELECT id, secret FROM claim UNION ALL SELECT id, secret FROM claim WHERE secret = 'Q') SELECT secret FROM claim",
                "WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret UNION ALL SELECT id, secret FROM claim WHERE secret = 'Q' EXCEPT SELECT id, secret FROM claim) SELECT secret FROM claim",
                "WITH RECURSIVE claim AS (SELECT 1 AS id, 'Z' AS secret EXCEPT SELECT id, secret FROM claim UNION ALL SELECT id, secret FROM claim WHERE secret = 'Q') SELECT secret FROM claim"):
        first, second = (con.execute(sql).fetchall() for con in worlds)
        assert first != second, (sql, first, second)
        verdict = _decide(sql, HIDDEN)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_chain_the_parser_groups_differently_from_the_engine_is_refused_before_any_rewrite(worlds):
    # Codex's sixth review of #170: `A UNION ALL B INTERSECT C` is `A UNION ALL (B INTERSECT C)`
    # to DuckDB, the CTE bound in both, and `(A UNION ALL B) INTERSECT C` to sqlglot, with no
    # recursive part; read as base-table references, B's and C's `claim` took the row filter
    # and the approved statement answered differently from the original. Refused as a shape,
    # as DuckDB, anywhere in the statement; parenthesised, grouped as written and read as before.
    recursive = "WITH RECURSIVE claim AS (SELECT 1 AS id UNION ALL SELECT id + 1 FROM claim WHERE id < 3 INTERSECT SELECT id + 1 FROM claim) SELECT id FROM claim"
    assert [con.execute(recursive).fetchall() for con in worlds] == [[(1,), (2,), (3,)]] * 2
    filtered = AccessPolicy(row_filters={"claim": "id = 99"}, policy_schema=SCHEMA)
    for policy in (COMPLETE, filtered, AccessPolicy()):
        verdict = _decide(recursive, SCHEMA, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.PARSE_ERROR, (policy, verdict)
    derived = "SELECT d.id FROM (SELECT 1 AS id UNION ALL SELECT 2 AS id INTERSECT SELECT 2 AS id) d"
    assert isinstance(_decide(derived, SCHEMA), Refusal)
    except_first = "SELECT d.id FROM (SELECT 1 AS id EXCEPT SELECT 2 AS id INTERSECT SELECT 2 AS id) d"
    assert isinstance(_decide(except_first, SCHEMA), Refusal)
    write = "WITH RECURSIVE claim AS (SELECT 1 AS id UNION ALL SELECT id + 1 FROM claim WHERE id < 3 INTERSECT SELECT id + 1 FROM claim) INSERT INTO sink (i) SELECT id FROM claim"
    assert isinstance(_decide_write(write, SCHEMA), Refusal)
    # Grouped alike by both, or parenthesised: not this refusal.
    for sql in ("SELECT d.id FROM (SELECT 1 AS id INTERSECT SELECT 1 AS id UNION ALL SELECT 2 AS id) d",
                "SELECT d.id FROM (SELECT 1 AS id UNION ALL (SELECT 2 AS id INTERSECT SELECT 2 AS id)) d",
                "SELECT d.id FROM ((SELECT 1 AS id UNION ALL SELECT 2 AS id) INTERSECT SELECT 2 AS id) d",
                "WITH RECURSIVE claim AS (SELECT 1 AS id UNION ALL (SELECT id + 1 FROM claim WHERE id < 3 INTERSECT SELECT id + 1 FROM claim)) SELECT id FROM claim"):
        worlds[0].execute(sql)
        assert isinstance(_decide(sql, SCHEMA), Approved), sql
    # Oracle groups left to right as the parser does: decision only, not refused as a shape.
    oracle = decide("SELECT d.id FROM (SELECT 1 AS id FROM sink UNION ALL SELECT 2 AS id FROM sink INTERSECT SELECT 2 AS id FROM sink) d",
                    {"sink": {"I"}}, dialect="oracle", target="oracle", policy=AccessPolicy(policy_schema={"sink": {"I"}}))
    assert not (isinstance(oracle, Refusal) and oracle.code is RefusalCode.PARSE_ERROR), oracle

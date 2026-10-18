"""M181 to M186: a column read through a derived table's star, or by position, reaches the base
table's withheld column.

`column_tables` settled a qualifier naming a derived table or CTE with no base table, so
`SELECT d.secret FROM (SELECT * FROM claim) d` read a hidden, denied or masked `claim.secret`
(M181): `check_access` skipped the column, `candidate_tables` answered no table, and the mask's
rewrite wrapped only a table whose masked column was NAMED. Where the derived table exposed
nothing under the name, the column was left unbound instead of binding outward as the engine
binds it (M182). The qualifier now resolves by EXPOSURE: in projection order, the first named
projection settles it in its own scope, and the first star reaching a base table that has the
column, by the policy's whole schema, binds it to that table; nothing exposed, the next scope out
answers. Four shapes bind by position rather than by name and are refused as unnamed: a star in a
branch of a set operation inside a derived table (M183), a column list over a star (M184),
DuckDB's `#n` (M185), and a PIVOT or UNPIVOT source, which crashed the resolver (M186). All
measured on main `dd781f3` and on #167 at `e19e850` through `decide` as duckdb; the ui lane's
replay of 11,337 recorded statements found no verdict change beyond the two expected.
"""
from __future__ import annotations

import duckdb
import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

FULL = {"claim": {"id", "secret"}, "other": {"id", "secret"}, "wide": {"id", "note"}}
ENGINES = ["duckdb", "oracle", "postgres"]


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INT, secret VARCHAR); INSERT INTO claim VALUES (1, 'CLAIM-SECRET')")
    con.execute("CREATE TABLE other (id INT, secret VARCHAR); INSERT INTO other VALUES (1, 'OTHER-SECRET')")
    con.execute("CREATE TABLE wide (id INT, note VARCHAR); INSERT INTO wide VALUES (1, 'WIDE-NOTE')")
    return con


def _hidden(table, column):
    visible = {t: set(c) for t, c in FULL.items()}
    visible[table].discard(column)
    return visible


def _policy(table, column, kind):
    return AccessPolicy(**{kind: {(table, column)}}, policy_schema=FULL)


def _decide(sql, visible, engine="duckdb", policy=None):
    return decide(sql, visible, dialect=engine, target=engine, policy=policy)


def _withheld_three_ways(con, sql, table, column, value):
    """Hidden: refused as unknown. Denied: refused. Masked: refused in a predicate, or approved
    and the value absent from what runs."""
    hidden = _decide(sql, _hidden(table, column))
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, (sql, hidden)
    denied = _decide(sql, FULL, policy=_policy(table, column, "denied"))
    assert isinstance(denied, Refusal) and denied.code is RefusalCode.UNAUTHORIZED_COLUMN, (sql, denied)
    masked = _decide(sql, FULL, policy=_policy(table, column, "masked"))
    if isinstance(masked, Approved):
        rows = con.execute(masked.target_sql).fetchall()
        assert value not in str(rows), (sql, masked.target_sql, rows)
    else:
        assert masked.code is RefusalCode.MASKED_COLUMN_IN_PREDICATE, (sql, masked)


# --- M181: a qualified column through a derived table's star reads the base table -------------

THROUGH_A_STAR = [
    "SELECT d.secret FROM (SELECT * FROM claim) d",
    "WITH d AS (SELECT * FROM claim) SELECT d.secret FROM d",
    "SELECT d.id FROM (SELECT * FROM claim) d WHERE d.secret = 'CLAIM-SECRET'",
    "SELECT d.secret FROM (SELECT c.* FROM claim c JOIN other o ON o.id = c.id) d",
    "SELECT d.secret FROM (SELECT * FROM (SELECT * FROM claim) e) d",
    "SELECT d.secret FROM (SELECT * FROM claim c JOIN wide w ON w.id = c.id) d",
    # The first source under the star lacks the name: DuckDB binds the one that has it.
    "SELECT d.secret FROM (SELECT * FROM wide w JOIN claim c ON c.id = w.id) d",
    "SELECT d.secret FROM (SELECT COLUMNS('.*') FROM claim) d",
    "SELECT d.secret FROM (SELECT *, 'NAMED' AS secret FROM claim) d",  # the star is first
    "SELECT d.SECRET FROM (SELECT * FROM claim) d",  # folded as the engine folds it
]


@pytest.mark.parametrize("sql", THROUGH_A_STAR)
def test_a_withheld_column_read_through_a_derived_star_is_not_read(con, sql):
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sql", THROUGH_A_STAR[:2])
def test_on_every_engine_a_hidden_column_through_a_derived_star_is_unknown(sql, engine):
    verdict = _decide(sql, _hidden("claim", "secret"), engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (engine, verdict)


@pytest.mark.parametrize("sql", THROUGH_A_STAR)
def test_a_granted_column_through_a_derived_star_is_approved_and_runs(con, sql):
    verdict = _decide(sql, FULL)
    assert isinstance(verdict, Approved), (sql, verdict)
    assert con.execute(verdict.target_sql).fetchall() == con.execute(sql).fetchall()


def test_the_policy_s_schema_binds_a_hidden_column_to_the_table_that_has_it():
    # The grant hides claim's secret, the policy's schema still lists it: through a star over
    # wide and claim the column binds to claim, and the refusal names claim; read through the
    # grant alone, no table lists the name and the first under the star is named instead.
    sql = "SELECT d.secret FROM (SELECT * FROM wide w JOIN claim c ON c.id = w.id) d"
    told = _decide(sql, _hidden("claim", "secret"), policy=AccessPolicy(policy_schema=FULL))
    assert isinstance(told, Refusal) and told.code is RefusalCode.UNKNOWN_COLUMN, told
    assert "'claim'" in told.message, told.message
    untold = _decide(sql, _hidden("claim", "secret"))
    assert isinstance(untold, Refusal) and untold.code is RefusalCode.UNKNOWN_COLUMN, untold
    assert "'wide'" in untold.message, untold.message


@pytest.mark.parametrize("schema", [None, {"claim": {"id", "secret"}}],
                         ids=["no policy schema", "claim alone"])
def test_a_policy_schema_that_omits_a_table_under_the_star_still_denies_and_masks(con, schema):
    # The review gate's case: the deny and mask passes read the grant's columns beside the
    # policy's, so `wide`, which the policy does not list, does not take the name first; read
    # through the policy's schema alone it did, and claim's mask was dropped as another table's.
    sql = "SELECT d.secret FROM (SELECT * FROM wide w JOIN claim c ON c.id = w.id) d"
    listed = {"policy_schema": schema} if schema else {}
    denied = _decide(sql, FULL, policy=AccessPolicy(denied={("claim", "secret")}, **listed))
    assert isinstance(denied, Refusal) and denied.code is RefusalCode.UNAUTHORIZED_COLUMN, denied
    masked = _decide(sql, FULL, policy=AccessPolicy(masked={("claim", "secret")}, **listed))
    assert isinstance(masked, Approved), masked
    assert con.execute(masked.target_sql).fetchall() == [(None,)], masked.target_sql


@pytest.mark.parametrize("schema", [None, FULL], ids=["plan as written", "visible map"])
def test_the_eval_criterion_attributes_a_masked_column_read_through_a_derived_star(schema):
    # The review gate's case: `touched` resolved with no schema, so the star over `wide JOIN
    # claim` bound the column to wide and the mask it measures went unreported, where a decider
    # that dropped claim's mask would then score as disclosing nothing against nothing.
    import sqlglot

    from mnemiq.eval.criterion import touched

    ast = sqlglot.parse_one("SELECT d.secret FROM (SELECT * FROM wide w JOIN claim c ON c.id = w.id) d",
                            read="duckdb")
    result = touched(ast, AccessPolicy(masked={("claim", "secret")}), schema, "duckdb")
    assert result.entries == {("mask", "claim")}, result
    assert result.unattributable == (), result


COMPLETE = AccessPolicy(policy_schema=FULL)  # production's: every column of the snapshot's


@pytest.mark.parametrize("sql", [
    "SELECT d.rn FROM (SELECT *, 1 AS rn FROM claim) d",
    "SELECT d.rn FROM (SELECT c.*, 1 AS rn FROM claim c) d",
    "WITH d AS (SELECT *, 1 AS rn FROM claim) SELECT d.rn FROM d",
    "SELECT rn FROM (SELECT *, 1 AS rn FROM claim) d",
])
def test_a_name_the_star_s_table_is_listed_in_full_without_is_the_later_projection_s(con, sql):
    # The ui lane's replay of #167: three recorded statements refused when a listed table
    # lacking the name fell closed. The policy schema lists claim in full, so claim has no `rn`
    # and the engine reads the literal; the qualified and the bare spelling agree.
    verdict = _decide(sql, FULL, policy=COMPLETE)
    assert isinstance(verdict, Approved), (sql, verdict)
    assert con.execute(verdict.target_sql).fetchall() == [(1,)]


@pytest.mark.parametrize("sql", [
    "SELECT secret FROM (SELECT *, 'NAMED' AS secret FROM claim) d",
    "SELECT secret FROM (SELECT c.*, 'NAMED' AS secret FROM claim c) d",
    "WITH w AS (SELECT *, 'NAMED' AS secret FROM claim) SELECT secret FROM w",
    "SELECT x.s FROM (SELECT secret AS s FROM (SELECT *, 'NAMED' AS secret FROM claim) d) x",
])
@pytest.mark.parametrize("policy", [None, COMPLETE], ids=["grant alone", "complete schema"])
def test_a_bare_column_named_after_a_star_reads_the_star_s_hidden_column_and_is_refused(sql, policy):
    # The unqualified twin of M181 (the ui lane's replay): the engine binds the star's column
    # first, so the later alias does not stand in for a hidden `secret`. Under the grant alone
    # the table may hide the name and the read falls closed; under the complete schema claim is
    # listed with it and the column binds there, unknown to this grant either way.
    verdict = _decide(sql, _hidden("claim", "secret"), policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_named_projection_before_the_star_answers_for_the_name(con):
    # DuckDB binds the first column of the name in projection order (measured): here the
    # literal, so claim's secret is not read and the statement is approved with it withheld.
    sql = "SELECT d.secret FROM (SELECT 'NAMED' AS secret, * FROM claim) d"
    for visible, policy in ((_hidden("claim", "secret"), None),
                            (FULL, _policy("claim", "secret", "denied"))):
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Approved), (verdict,)
        assert con.execute(verdict.target_sql).fetchall() == [("NAMED",)]


def test_a_replaced_star_column_holds_the_replacement_not_the_base_column(con):
    # `* REPLACE (id AS secret)`: the name `secret` carries `id`, a column node checked in its
    # scope, and claim's secret is not read; measured, the statement returns the id.
    sql = "SELECT d.secret FROM (SELECT * REPLACE (id AS secret) FROM claim) d"
    for visible, policy in ((_hidden("claim", "secret"), None),
                            (FULL, _policy("claim", "secret", "denied"))):
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Approved), (verdict,)
        assert con.execute(verdict.target_sql).fetchall() == [(1,)]


def test_a_renamed_star_column_is_a_column_node_and_checked_as_one(con):
    # `RENAME (secret AS id)` is `secret AS id` in the tree, so the inner `secret` is checked in
    # its own scope (measured safe before this change); `d.id` then reads it.
    sql = "SELECT d.id FROM (SELECT * RENAME (secret AS id) FROM claim) d"
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")


def test_a_bare_column_through_a_derived_star_is_still_refused():
    # M78's door: `secret` unqualified is every referenced table's.
    for visible, policy, code in ((_hidden("claim", "secret"), None, RefusalCode.UNKNOWN_COLUMN),
                                  (FULL, _policy("claim", "secret", "denied"),
                                   RefusalCode.UNAUTHORIZED_COLUMN)):
        verdict = _decide("SELECT secret FROM (SELECT * FROM claim) d", visible, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is code, verdict


# --- M182: a derived table exposing nothing under the name binds outward ----------------------

OUTWARD = [
    "SELECT (SELECT d.secret FROM (SELECT id FROM claim) d) FROM other d",
    "SELECT (SELECT D.secret FROM (SELECT id FROM claim) AS d) FROM other AS D",
    "SELECT (WITH d AS (SELECT id FROM claim) SELECT d.secret FROM d) FROM other d",
    "SELECT (SELECT d.secret FROM (SELECT * EXCLUDE (secret) FROM claim) d) FROM other d",
]


@pytest.mark.parametrize("sql", OUTWARD)
def test_a_qualifier_the_derived_table_does_not_expose_binds_to_the_outer_table(con, sql):
    _withheld_three_ways(con, sql, "other", "secret", "OTHER-SECRET")


@pytest.mark.parametrize("sql", OUTWARD)
def test_bound_outward_a_granted_column_is_approved_and_runs(con, sql):
    verdict = _decide(sql, FULL)
    assert isinstance(verdict, Approved), (sql, verdict)
    assert con.execute(verdict.target_sql).fetchall() == [("OTHER-SECRET",)]


def test_exposure_folds_the_column_s_case_as_the_engine_does():
    # The ui lane's edge: `d.SECRET` is `secret` on DuckDB, so the inner star over `other`
    # exposes it and the column is other's, denied; read as unexposed it would bind to claim.
    sql = "SELECT (SELECT d.SECRET FROM (SELECT * FROM other) d) FROM claim d"
    verdict = _decide(sql, FULL, policy=_policy("other", "secret", "denied"))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNAUTHORIZED_COLUMN, verdict


def test_a_table_the_policy_schema_lacks_exposes_any_name():
    # The ui lane's second edge: the policy lists claim alone, other's columns are known only
    # through `visible`, where secret is hidden; the star over `other` must still expose the
    # name, so the column binds to other and is unknown there, rather than to claim's grant.
    sql = "SELECT (SELECT d.secret FROM (SELECT * FROM other) d) FROM claim d"
    policy = AccessPolicy(policy_schema={"claim": {"id", "secret"}})
    verdict = _decide(sql, _hidden("other", "secret"), policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert isinstance(_decide(sql, FULL, policy=policy), Approved)


# --- M183, M184: shapes that bind by position are refused as unnamed --------------------------

BY_POSITION = [
    "SELECT d.secret FROM (SELECT * FROM claim UNION ALL SELECT * FROM wide) d",
    "SELECT secret FROM (SELECT * FROM claim UNION ALL SELECT * FROM wide) d",
    "SELECT d.secret FROM (SELECT id, secret FROM claim UNION ALL SELECT * FROM wide) d",
    "WITH d AS (SELECT * FROM claim UNION ALL SELECT * FROM wide) SELECT d.secret FROM d",
    "SELECT d.y FROM (SELECT * FROM claim) d(x, y)",
    "WITH d(x, y) AS (SELECT * FROM claim) SELECT d.y FROM d",
    "SELECT d.y FROM (SELECT * FROM (SELECT * FROM claim) e) d(x, y)",
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sql", BY_POSITION)
def test_a_star_that_binds_by_position_is_refused_before_any_grant_is_read(sql, engine):
    verdict = _decide(sql, FULL, engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (engine, verdict)
    assert "position" in verdict.message, verdict.message


def test_the_shapes_that_bind_by_position_read_a_withheld_column_on_duckdb(con):
    """The premise: these return wide's note or claim's secret under the name asked for."""
    rows = con.execute("SELECT d.secret FROM (SELECT * FROM claim UNION ALL SELECT * FROM wide) d").fetchall()
    assert ("WIDE-NOTE",) in rows
    assert con.execute("SELECT d.y FROM (SELECT * FROM claim) d(x, y)").fetchall() == [("CLAIM-SECRET",)]


@pytest.mark.parametrize("sql", [
    # Named branches and named projections bind by name: not these shapes.
    "SELECT d.secret FROM (SELECT id, secret FROM claim UNION ALL SELECT id, note FROM wide) d",
    "SELECT d.y FROM (SELECT id, secret FROM claim) d(x, y)",
    "WITH t(a, b) AS (SELECT * FROM (VALUES (1, 2))) SELECT t.b FROM t",  # the ui lane's 42
    "WITH t AS (SELECT 1 AS a UNION ALL SELECT 2) SELECT t.a FROM t",
])
def test_named_branches_and_a_list_over_named_projections_are_not_refused_as_positional(sql):
    verdict = _decide(sql, FULL)
    assert not (isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR), verdict


def test_a_withheld_column_in_a_named_union_branch_is_checked_in_its_scope(con):
    sql = "SELECT d.secret FROM (SELECT id, secret FROM claim UNION ALL SELECT id, note FROM wide) d"
    _withheld_three_ways(con, sql, "wide", "note", "WIDE-NOTE")


@pytest.mark.parametrize("sql", [
    # Codex's review of #167 at 246cd30: a column list on a table REFERENCE renames by position
    # too, over a base table and over a CTE whose body expands a star.
    "SELECT d.id FROM claim d(secret, id)",
    "WITH d AS (SELECT * FROM claim) SELECT e.id FROM d e(secret, id)",
    # And COLUMNS with an alias template renames every expanded column by a pattern.
    'SELECT d.x_secret FROM (SELECT COLUMNS(*) AS "x_\\0" FROM claim) d',
])
def test_a_column_list_on_a_table_reference_or_an_aliased_columns_expansion_is_refused(con, sql):
    rows = con.execute(sql).fetchall()
    assert rows == [("CLAIM-SECRET",)], rows  # the premise: the withheld column, under another name
    for visible, policy in ((_hidden("claim", "secret"), None),
                            (FULL, _policy("claim", "secret", "denied")),
                            (FULL, _policy("claim", "secret", "masked")),
                            (FULL, None)):
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


@pytest.mark.parametrize("sql", [
    # Over a CTE of named projections too, and over a table function: refused whatever stands
    # behind the name, since the CTE visible at the reference is not the root's (the review
    # gate: an inner WITH shadowing an outer name slipped a root-level lookup).
    "WITH d AS (SELECT id, secret FROM claim) SELECT e.x FROM d e(k, x)",
    "WITH d AS (SELECT id FROM other) SELECT id FROM "
    "(WITH d AS (SELECT * FROM claim) SELECT e.id FROM d e(secret, id)) z",
    "SELECT i FROM range(3) t(i)",
])
def test_a_column_list_on_any_table_reference_is_refused(sql):
    for visible, policy in ((_hidden("claim", "secret"), None), (FULL, COMPLETE),
                            (FULL, _policy("claim", "secret", "denied"))):
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (sql, verdict)


@pytest.mark.parametrize("sql", [
    # The review gate: an alias named after the hidden column must not switch the binding off
    # where the engine reads the source's column, in the projection or a WHERE.
    "SELECT secret AS secret FROM (SELECT * FROM claim) d",
    "SELECT upper(secret) AS secret FROM (SELECT * FROM claim) d",
    "SELECT id AS secret FROM (SELECT * FROM claim) d WHERE secret = 'CLAIM-SECRET'",
    "SELECT id AS k FROM (SELECT * FROM claim) d ORDER BY (SELECT secret) ",
])
def test_an_alias_named_after_a_withheld_column_does_not_stand_in_for_it(con, sql):
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")


def test_a_bare_name_that_is_the_query_s_own_alias_is_the_alias_where_the_engine_reads_it(con):
    # Measured on DuckDB: ORDER BY and HAVING read an output alias before a source column of the
    # name; GROUP BY reads the source column first. So `ORDER BY k` over `id AS k` is the alias
    # under either schema, while `GROUP BY k` binds through the star and is the alias only where
    # the complete schema says the source lacks `k`; under the grant alone it falls closed.
    ordered = "SELECT id AS k FROM (SELECT * FROM claim) d ORDER BY k"
    grouped = "SELECT id AS k, count(*) AS n FROM (SELECT * FROM claim) d GROUP BY k"
    for policy in (None, COMPLETE):
        verdict = _decide(ordered, {"claim": {"id"}}, policy=policy)
        assert isinstance(verdict, Approved), (policy, verdict)
        assert con.execute(verdict.target_sql).fetchall() == [(1,)]
    assert isinstance(_decide(grouped, {"claim": {"id"}}, policy=COMPLETE), Approved)
    closed = _decide(grouped, {"claim": {"id"}})
    assert isinstance(closed, Refusal) and closed.code is RefusalCode.UNKNOWN_COLUMN, closed
    # The alias wins in ORDER BY and HAVING even where the source has a column of the name
    # (measured: `-id AS secret ... ORDER BY secret` sorts by the alias, `HAVING secret > 1`
    # filters by it), so claim's hidden secret is not read. sqlglot drops the ORDER BY reference
    # from the scope's columns itself; HAVING is the clause the skip decides.
    for sql, rows in (("SELECT -id AS secret FROM (SELECT * FROM claim) d ORDER BY secret", [(-1,)]),
                      ("SELECT id AS k, count(*) AS n FROM (SELECT * FROM claim) d GROUP BY id "
                       "HAVING k > 0", [(1, 1)]),
                      ("SELECT id AS secret, count(*) AS n FROM (SELECT * FROM claim) d GROUP BY id "
                       "HAVING secret > 0", [(1, 1)])):
        aliased = _decide(sql, _hidden("claim", "secret"))
        assert isinstance(aliased, Approved), (sql, aliased)
        assert con.execute(aliased.target_sql).fetchall() == rows


# --- M185: DuckDB's positional reference ------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT #2 FROM claim",
    "SELECT id FROM claim WHERE #2 LIKE 'CLAIM%'",
    "SELECT #2 FROM (SELECT * FROM claim) d",
])
def test_a_column_by_number_is_refused(sql):
    verdict = _decide(sql, FULL)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, verdict
    assert "#" in verdict.message


# --- M186: a PIVOT or UNPIVOT source is unmodelled, not a crash -------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT value FROM (UNPIVOT claim ON COLUMNS(* EXCLUDE (id)) INTO NAME k VALUE value) u",
    "SELECT k FROM (UNPIVOT claim ON secret INTO NAME k VALUE v) u",
    "SELECT id FROM (PIVOT claim ON id USING first(secret)) p",
])
def test_a_pivot_source_is_refused_as_unmodelled(sql):
    verdict = _decide(sql, FULL)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_SOURCE, verdict

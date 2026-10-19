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

FULL = {"claim": {"id", "secret"}, "other": {"id", "secret"}, "wide": {"id", "note"},
        "twin": {"id", "secret_1"}}
ENGINES = ["duckdb", "oracle", "postgres"]


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INT, secret VARCHAR); INSERT INTO claim VALUES (1, 'CLAIM-SECRET')")
    con.execute("CREATE TABLE other (id INT, secret VARCHAR); INSERT INTO other VALUES (1, 'OTHER-SECRET')")
    con.execute("CREATE TABLE wide (id INT, note VARCHAR); INSERT INTO wide VALUES (1, 'WIDE-NOTE')")
    # `secret_1` on twin is a real column spelled like a name the engine generates (Codex's
    # fifth review of #167): a generated-name read must not bind to it.
    con.execute("CREATE TABLE twin (id INT, secret_1 VARCHAR); INSERT INTO twin VALUES (1, 'TWIN-S1')")
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


@pytest.mark.parametrize("sql", [
    # Codex's fourth review: DuckDB renames a later duplicate output name `secret_1`, so the
    # star's claim.secret behind a literal of the same name is read under that name; and over a
    # join, each later duplicate the same way.
    "SELECT d.secret_1 FROM (SELECT 'SAFE' AS secret, * FROM claim) d",
    "SELECT secret_1 FROM (SELECT 'SAFE' AS secret, * FROM claim) d",
    "SELECT d.secret_1 FROM (SELECT * FROM other o JOIN claim c ON c.id = o.id) d",
    # A second derived source beside the one with the star must not switch the bare binding off.
    "SELECT secret FROM (SELECT *, 'SAFE' AS secret FROM claim) d CROSS JOIN (SELECT 1 AS x) e",
    "SELECT secret FROM (SELECT 1 AS x) e CROSS JOIN (SELECT *, 'SAFE' AS secret FROM claim) d",
])
def test_a_renamed_duplicate_or_a_second_source_still_reaches_the_withheld_column(con, sql):
    rows = con.execute(sql).fetchall()
    assert rows == [("CLAIM-SECRET",)], rows  # the premise: DuckDB serves claim's secret
    if "secret_1" in sql:
        # A generated duplicate name binds to the table whose column it renames and is never a
        # listed column there, so every reading stops at the column check: fail-closed, with the
        # advice to name the columns distinctly.
        for visible, kind in ((_hidden("claim", "secret"), None), (FULL, "denied"), (FULL, "masked")):
            for schema in ({}, FULL):
                extra = AccessPolicy(**{kind: {("claim", "secret")}}, policy_schema=schema) if kind \
                    else AccessPolicy(policy_schema=schema)
                verdict = _decide(sql, visible, policy=extra)
                assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (
                    sql, kind, verdict)
    else:
        _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
        complete = _decide(sql, _hidden("claim", "secret"), policy=COMPLETE)
        assert isinstance(complete, Refusal) and complete.code is RefusalCode.UNKNOWN_COLUMN, complete
        granted = _decide(sql, FULL, policy=COMPLETE)  # the bare read of a granted column runs
        assert isinstance(granted, Approved), granted
        assert con.execute(granted.target_sql).fetchall() == [("CLAIM-SECRET",)]


@pytest.mark.parametrize("sql", [
    # The review gate: DuckDB's renaming of duplicates follows collision rules no count models
    # (an explicit `secret_1` after the star is itself renamed; a taken `secret_1` moves the
    # star's to `secret_2`; a duplicate one level down), so a derived output carrying a name
    # twice leaves the statement unresolved and every pass falls closed.
    "SELECT d.secret_1 FROM (SELECT 'SAFE' AS secret, *, 'X' AS secret_1 FROM claim) d",
    "SELECT d.secret_2 FROM (SELECT 'A' AS secret, 'B' AS secret_1, * FROM claim) d",
    "SELECT d.secret_1 FROM (SELECT * FROM (SELECT * FROM other o JOIN claim c ON c.id = o.id) x) d",
])
def test_a_derived_output_with_a_duplicate_name_falls_closed_under_the_complete_schema(con, sql):
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]  # the premise
    for visible, kind in ((_hidden("claim", "secret"), None), (FULL, "denied"), (FULL, "masked")):
        policy = AccessPolicy(**{kind: {("claim", "secret")}}, policy_schema=FULL) if kind else COMPLETE
        verdict = _decide(sql, visible, policy=policy)
        if isinstance(verdict, Approved):
            assert kind == "masked" and "CLAIM-SECRET" not in str(con.execute(verdict.target_sql).fetchall())
        else:
            assert verdict.code in (RefusalCode.UNKNOWN_COLUMN, RefusalCode.UNAUTHORIZED_COLUMN,
                                    RefusalCode.MASKED_COLUMN_IN_PREDICATE), (sql, kind, verdict)


def test_a_bare_name_no_source_of_its_scope_exposes_binds_outward(con):
    # The review gate: `secret` names nothing in `d`, so the engine reads the enclosing scope's
    # `other d`; a first repair had dropped this binding for bare names.
    sql = "SELECT (SELECT secret FROM (SELECT id FROM claim) d) FROM other d"
    _withheld_three_ways(con, sql, "other", "secret", "OTHER-SECRET")
    verdict = _decide(sql, FULL)
    assert isinstance(verdict, Approved), verdict
    assert con.execute(verdict.target_sql).fetchall() == [("OTHER-SECRET",)]


def test_a_generated_name_does_not_bind_to_a_table_that_happens_to_have_it(con):
    # Codex's fifth review: twin has a real `secret_1`, granted, and the engine's `secret_1`
    # over this derived table is claim's secret; bound to the first base table the read passed.
    sql = "SELECT d.secret_1 FROM (SELECT 'SAFE' AS secret, c.* FROM twin t CROSS JOIN claim c) d"
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    for visible, kind in ((_hidden("claim", "secret"), None), (FULL, "denied"), (FULL, "masked")):
        policy = AccessPolicy(**{kind: {("claim", "secret")}}, policy_schema=FULL) if kind else COMPLETE
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (kind, verdict)
        assert "makes up" in verdict.message
    honest = _decide("SELECT t.secret_1 FROM twin t", FULL, policy=COMPLETE)
    assert isinstance(honest, Approved), honest


def test_a_bare_name_beside_a_base_table_and_a_derived_source_resolves_in_from_order(con):
    # Codex's fifth review: a base table beside the derived source had switched the bare
    # binding off, and the inner alias then stood in for the hidden column.
    sql = "SELECT secret FROM wide w CROSS JOIN (SELECT *, 'SAFE' AS secret FROM claim) d"
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    for policy in (None, COMPLETE):
        hidden = _decide(sql, _hidden("claim", "secret"), policy=policy)
        assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    granted = _decide(sql, FULL, policy=COMPLETE)
    assert isinstance(granted, Approved), granted
    assert con.execute(granted.target_sql).fetchall() == [("CLAIM-SECRET",)]
    # A base table listed with the name binds it ahead of a derived source in FROM order.
    base_first = _decide("SELECT note FROM wide w CROSS JOIN (SELECT 'X' AS note FROM claim) d",
                         _hidden("wide", "note"), policy=COMPLETE)
    assert isinstance(base_first, Refusal) and base_first.code is RefusalCode.UNKNOWN_COLUMN, base_first
    # And settles it: a derived source the grant alone cannot show to lack the name does not
    # take it afterwards, since the engine refuses the name as ambiguous where both carry it
    # rather than reading either (premise below), so the listed base table is the only reading.
    with pytest.raises(duckdb.BinderException, match="Ambiguous"):
        con.execute("SELECT id FROM wide w CROSS JOIN (SELECT * FROM other) d")
    settled = _decide("SELECT note FROM wide w CROSS JOIN (SELECT * FROM other) d", FULL)
    assert isinstance(settled, Approved), settled


def test_a_grant_only_base_table_beside_a_derived_source_falls_closed_before_binding_outward(con):
    # The review gate on f83a78c: claim granted without secret, other with it, and the bare
    # `secret` of the inner scope, which the engine binds to claim, was bound outward to other.
    sql = ("SELECT (SELECT secret FROM claim c CROSS JOIN (SELECT id AS x FROM wide) d LIMIT 1) "
           "FROM other o")
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    hidden = _decide(sql, _hidden("claim", "secret"))
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    assert "'claim'" in hidden.message, hidden.message
    # A table listed in full and lacking the name does not have it, and the read goes outward,
    # where the engine binds it: claim has no note, so this reads wide's.
    outward = ("SELECT (SELECT note FROM claim c CROSS JOIN (SELECT id AS x FROM other) d LIMIT 1) "
               "FROM wide w")
    assert con.execute(outward).fetchall() == [("WIDE-NOTE",)]
    hidden_outward = _decide(outward, _hidden("wide", "note"), policy=COMPLETE)
    assert isinstance(hidden_outward, Refusal) and "'wide'" in hidden_outward.message, hidden_outward
    assert isinstance(_decide(outward, FULL, policy=COMPLETE), Approved)
    # From the grant alone, claim may hide a note, and the read falls closed on claim.
    grant_only = _decide(outward, FULL)
    assert isinstance(grant_only, Refusal) and "'claim'" in grant_only.message, grant_only


@pytest.mark.parametrize("sql", [
    "SELECT c.id AS k, k + 1 FROM claim c CROSS JOIN (SELECT id AS x FROM other) d",
    "SELECT c.id AS k FROM claim c CROSS JOIN (SELECT id AS x FROM other) d WHERE k > 0",
    "SELECT c.id AS k FROM claim c CROSS JOIN (SELECT id AS x FROM other) d GROUP BY k",
])
def test_the_query_s_own_alias_beside_a_grant_only_base_table_needs_the_complete_schema(con, sql):
    # The engine reads a projection alias laterally only once no source has the name, and from
    # the grant alone the resolver cannot tell whether claim has it (the review gate asked for
    # the alias reading here; Codex's sixth review read a hidden column through it, below). So
    # the complete schema, which the shipped `visible_schema` is, approves these, and the grant
    # alone refuses them on claim.
    assert con.execute(sql).fetchall()
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved), sql
    grant_only = _decide(sql, FULL)
    assert isinstance(grant_only, Refusal) and "'claim'" in grant_only.message, grant_only


@pytest.mark.parametrize("sql", [
    "SELECT id AS secret, secret FROM claim",
    "SELECT d.leak FROM (SELECT id AS secret, secret AS leak FROM claim) d",
    "SELECT id AS secret, (SELECT secret) AS leak FROM claim",
    "SELECT 'x' AS secret, count(*) FROM claim GROUP BY secret",
    "SELECT id AS secret FROM claim WHERE secret = 'CLAIM-SECRET'",
], ids=["projection", "through a derived alias", "correlated scalar subquery", "GROUP BY", "WHERE"])
def test_the_query_s_own_alias_does_not_read_a_hidden_source_column_over_base_tables_alone(con, sql):
    # Codex's seventh review of #167: the engine binds the source column before the alias in
    # the SELECT list, WHERE and GROUP BY, and `check_access`'s alias allowance passed the name
    # over base tables alone. Under the complete schema, which the shipped deciding paths pass,
    # the table listed with the name binds it; denied and masked hold through the policy's own
    # schema. From the grant alone the alias reading over base tables alone is `check_access`'s,
    # and the register's.
    assert con.execute(sql).fetchall()
    hidden = _decide(sql, _hidden("claim", "secret"), policy=COMPLETE)
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    assert "'claim'" in hidden.message, hidden.message
    denied = _decide(sql, FULL, policy=_policy("claim", "secret", "denied"))
    assert isinstance(denied, Refusal) and denied.code is RefusalCode.UNAUTHORIZED_COLUMN, denied
    masked = _decide(sql, FULL, policy=_policy("claim", "secret", "masked"))
    if isinstance(masked, Approved):
        assert "CLAIM-SECRET" not in str(con.execute(masked.target_sql).fetchall()), masked.target_sql
    else:
        assert masked.code is RefusalCode.MASKED_COLUMN_IN_PREDICATE, masked
    # A table listed in full and lacking the name does not have it, and the alias is read.
    lateral = _decide("SELECT id AS k, k + 1 FROM claim WHERE k > 0 GROUP BY k", FULL, policy=COMPLETE)
    assert isinstance(lateral, Approved), lateral
    assert isinstance(_decide("SELECT id AS k, k + 1 FROM claim", FULL), Approved)
    # And inside a subquery over base tables alone, from the grant alone (the review gate).
    inner = "SELECT (SELECT id AS k FROM wide GROUP BY k LIMIT 1) FROM claim"
    assert con.execute(inner).fetchall() == [(1,)]
    for policy in (None, COMPLETE):
        assert isinstance(_decide(inner, FULL, policy=policy), Approved), policy


def test_the_query_s_own_alias_does_not_read_a_hidden_source_column_beside_a_derived_source(con):
    # Codex's sixth review: the engine binds the source column before the alias.
    sql = "SELECT c.id AS secret, secret FROM claim c CROSS JOIN (SELECT id AS x FROM wide) d"
    assert con.execute(sql).fetchall() == [(1, "CLAIM-SECRET")]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    for policy in (None, COMPLETE):
        hidden = _decide(sql, _hidden("claim", "secret"), policy=policy)
        assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, (policy, hidden)


def test_a_suffixed_name_over_an_output_carrying_its_base_once_is_an_honest_column(con):
    # The ui lane's replay: `end_hour_24` beside one `end_hour` was refused as generated, and the
    # engine renames only a duplicate. Over one `note`, `note_1` is the explicit alias.
    sql = "SELECT d.note_1 FROM (SELECT note, id AS note_1 FROM wide) d"
    assert con.execute(sql).fetchall() == [(1,)]
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)
    assert isinstance(_decide(sql, FULL), Approved)
    # Over two `note`s the engine renames the second to `note_1`, so the read is refused.
    twice = "SELECT d.note_1 FROM (SELECT 'x' AS note, * FROM wide) d"
    assert con.execute(twice).fetchall() == [("WIDE-NOTE",)]
    verdict = _decide(twice, FULL, policy=COMPLETE)
    assert isinstance(verdict, Refusal) and "makes up" in verdict.message, verdict
    # A CTE's column list names its output, so the duplicate is counted through it too.
    listed = "WITH w(note) AS (SELECT id FROM claim) SELECT d.note_1 FROM (SELECT 'x' AS note, * FROM w) d"
    assert con.execute(listed).fetchall() == [(1,)]
    verdict = _decide(listed, FULL, policy=COMPLETE)
    assert isinstance(verdict, Refusal) and "makes up" in verdict.message, verdict


@pytest.mark.parametrize("sql", [
    "WITH w(n) AS (SELECT 1) SELECT (SELECT n FROM w) AS n FROM claim",
    "WITH RECURSIVE p(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM p WHERE n < 3) SELECT p.n FROM p CROSS JOIN claim",
    "WITH w(a, b) AS (SELECT 1, 2) SELECT b FROM w",
    "SELECT l.a FROM (SELECT 1 AS k) o, LATERAL (SELECT 1, 2) l(a, b)",
], ids=["scalar subquery over a listed CTE", "recursive CTE", "a list shorter than the output", "LATERAL list over values"])
def test_a_cte_or_lateral_column_list_names_its_output(con, sql):
    # The ui lane's replay: column-list names were known nowhere, so a bare read of one fell
    # closed outward (and the recursive form was refused on main). Each name stands for the
    # projection at its position, whose own nodes are checked in that scope.
    assert con.execute(sql).fetchall()
    for policy in (None, COMPLETE):
        assert isinstance(_decide(sql, FULL, policy=policy), Approved), (sql, policy)
    # A listed name over a hidden column is that column, checked where it is projected.
    listed = "WITH w(x) AS (SELECT secret FROM claim) SELECT x FROM w"
    _withheld_three_ways(con, listed, "claim", "secret", "CLAIM-SECRET")
    assert isinstance(_decide(listed, FULL, policy=COMPLETE), Approved)


@pytest.mark.parametrize("sql", [
    "WITH w(x) AS (SELECT 1 AS secret FROM wide) SELECT (SELECT w.secret FROM w) FROM claim w",
    "WITH w(x) AS (SELECT 1 AS secret FROM wide) SELECT (SELECT secret FROM w) FROM claim",
    "SELECT (SELECT l.secret FROM (SELECT 1 AS k) o, LATERAL (SELECT 1 AS secret FROM wide) l(x)) FROM claim l",
], ids=["qualified by the CTE's name", "bare", "qualified by a LATERAL's alias"])
def test_a_renamed_position_exposes_nothing_under_its_old_name(con, sql):
    # Codex's eighth review of #167: the list renames the projection, so the engine has no
    # `secret` in w and binds the reference outward, to the outer source of the same name; read
    # as the CTE's named projection, the reference passed and claim's secret came back.
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    hidden = _decide(sql, _hidden("claim", "secret"), policy=COMPLETE)
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)


@pytest.mark.parametrize("sql", [
    "SELECT secret FROM (SELECT *, 'SAFE' AS secret FROM claim)",
    "SELECT (SELECT max(secret) FROM (SELECT * FROM claim)) FROM other",
], ids=["bare over the star", "a scalar subquery over the star, from a table granting the name"])
def test_an_unaliased_derived_table_is_a_source(con, sql):
    # The ui lane's replay of #167: `FROM (SELECT ...)` with no alias, which DuckDB, SQLite and
    # Oracle accept, was read as no source at all, so its bare names bound outward or fell to
    # the flat rule, and claim's hidden secret came back where the aliased twin was refused.
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)
    # And a name it exposes by alias is its own: not refused on an outer table.
    inner = "SELECT (SELECT secret FROM (SELECT 'SAFE' AS secret)) FROM claim"
    assert con.execute(inner).fetchall() == [("SAFE",)]
    for policy in (None, COMPLETE):
        assert isinstance(_decide(inner, _hidden("claim", "secret"), policy=policy), Approved), policy
    cte = ("WITH f AS (SELECT id FROM (SELECT id, 1 AS rn FROM wide) WHERE rn = 1) "
           "SELECT f.id FROM f JOIN claim c ON c.id = f.id")
    assert con.execute(cte).fetchall() == [(1,)]
    for policy in (None, COMPLETE):
        assert isinstance(_decide(cte, FULL, policy=policy), Approved), policy


@pytest.mark.parametrize("sql", [
    "SELECT id AS secret, count(*) AS n FROM claim GROUP BY id HAVING max(secret) = 'CLAIM-SECRET'",
    "SELECT id AS secret, count(*) AS n FROM (SELECT * FROM claim) d GROUP BY id HAVING max(secret) = 'CLAIM-SECRET'",
    "WITH w AS (SELECT secret AS s FROM claim) SELECT count(*) AS n FROM w HAVING max(s) = 'CLAIM-SECRET'",
    "SELECT id AS secret FROM claim GROUP BY id ORDER BY max(secret) = 'CLAIM-SECRET'",
], ids=["over the table", "through a derived star", "through a CTE alias", "an ORDER BY aggregate"])
def test_a_source_column_inside_a_having_or_order_by_aggregate_is_read_in_its_scope(con, sql):
    # sqlglot lists no unqualified HAVING column in a scope, nor an ORDER BY one matching an
    # output name; DuckDB reads a source column inside an aggregate there (an output alias only
    # for a bare item), so the hidden column answered the predicate or the order unchecked
    # (TPC-DS q23 and q24 made the clause's reading necessary; the review gate, the ORDER BY).
    assert con.execute(sql).fetchall() in ([(1, 1)], [(1,)])
    hidden = _decide(sql, _hidden("claim", "secret"), policy=COMPLETE)
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden
    denied = _decide(sql, FULL, policy=_policy("claim", "secret", "denied"))
    assert isinstance(denied, Refusal), denied
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)


@pytest.mark.parametrize("sql", [
    "SELECT id AS x FROM claim UNION SELECT id FROM other ORDER BY x",
    "SELECT id AS x FROM claim UNION ALL SELECT id FROM other ORDER BY x DESC LIMIT 5",
    "(SELECT id AS x FROM claim) UNION (SELECT id FROM other) ORDER BY x",
    "(SELECT id AS x FROM claim UNION SELECT id FROM other) UNION ALL SELECT id FROM wide ORDER BY x",
    "SELECT id, sum(id) OVER (ORDER BY id) AS running FROM claim ORDER BY running",
])
def test_a_set_operation_s_or_a_window_s_order_by_reads_the_output_it_names(con, sql):
    # The review gate: a set operation's ORDER BY hangs off the union, so no SELECT was found
    # for it and ordinary sorted-UNION SQL was refused; its output is the first branch's.
    assert con.execute(sql).fetchall()
    for policy in (None, COMPLETE):
        assert isinstance(_decide(sql, FULL, policy=policy), Approved), (sql, policy)


def test_a_recursive_cte_s_list_names_its_anchor_even_where_nothing_reads_the_cte(con):
    # The ui lane's replay: sqlglot hands the recursive branch the anchor's scope, whose parent
    # is the set operation, not the CTE, so the list was not found and `a` was refused on main.
    sql = ("WITH RECURSIVE m(a) AS (SELECT 1 UNION ALL SELECT a + 1 FROM m WHERE a < 3), "
           "c AS (SELECT id FROM claim) SELECT count(*) AS n FROM c")
    assert con.execute(sql).fetchall() == [(1,)]
    for policy in (None, COMPLETE):
        assert isinstance(_decide(sql, FULL, policy=policy), Approved), policy


@pytest.mark.parametrize("sql", [
    "SELECT (SELECT max(secret) FROM claim) AS secret FROM wide",
    "SELECT (SELECT 1 AS secret), (SELECT max(secret) FROM claim) FROM wide",
], ids=["the outer alias", "a sibling subquery's alias"])
def test_an_alias_vouches_only_in_its_own_select(con, sql):
    # The ui lane's replay: the alias allowance was statement-wide, and no engine resolves a
    # name to another scope's alias, so the hidden `secret` passed on an alias elsewhere.
    assert con.execute(sql).fetchall()[0][-1] == "CLAIM-SECRET"
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)


@pytest.mark.parametrize("sql", [
    "WITH w(secret) AS (SELECT 1) SELECT secret FROM claim",
    "WITH w(secret) AS (SELECT 1) SELECT (SELECT secret FROM claim LIMIT 1) FROM w",
], ids=["beside the CTE", "a subquery over the base table, from the CTE"])
def test_a_cte_s_column_list_names_nothing_outside_the_cte(con, sql):
    # The review gate on #167: the list's names made known statement-wide let a bare `secret`
    # over claim pass as a known name. A list's name counts where a reference resolves to it.
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    hidden = _decide(sql, _hidden("claim", "secret"), policy=COMPLETE)
    assert isinstance(hidden, Refusal) and hidden.code is RefusalCode.UNKNOWN_COLUMN, hidden


@pytest.mark.parametrize("sql", [
    "SELECT l.y FROM (SELECT 1 AS k) o, LATERAL (SELECT * FROM claim) l(x, y)",
    "SELECT l.y FROM (SELECT 1 AS k) o CROSS JOIN LATERAL (SELECT * FROM claim) AS l(x, y)",
    "SELECT l.y FROM (SELECT 1 AS k) o JOIN LATERAL (SELECT * FROM claim) l(x, y) ON true",
], ids=["comma", "CROSS JOIN LATERAL", "JOIN LATERAL ON true"])
def test_a_column_list_on_a_lateral_over_a_star_is_refused_as_positional(con, sql):
    # The ui lane's replay of #167: approved and returning the withheld value under hidden,
    # denied and masked alike; the list sits on the LATERAL's alias, which the shape check read
    # on subqueries and CTEs alone.
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    for visible, policy in ((_hidden("claim", "secret"), None), (_hidden("claim", "secret"), COMPLETE),
                            (FULL, _policy("claim", "secret", "denied")), (FULL, _policy("claim", "secret", "masked")),
                            (FULL, COMPLETE)):
        verdict = _decide(sql, visible, policy=policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.SELECT_STAR, (policy, verdict)


@pytest.mark.parametrize("sql", [
    "SELECT l.secret FROM (SELECT 1 AS k) o, LATERAL (SELECT * FROM claim) l",
    "SELECT l.y FROM (SELECT 1 AS k) o, LATERAL (SELECT id, secret FROM claim) l(x, y)",
    "SELECT l.s FROM (SELECT 1 AS k) o, LATERAL (SELECT secret AS s FROM claim) l",
], ids=["through a LATERAL star", "a LATERAL list over named projections", "a LATERAL alias"])
def test_a_read_through_a_lateral_binds_through_the_derived_walk(con, sql):
    # The ui lane's replay: a qualified read through a LATERAL star was bound through no derived
    # walk and approved under hidden. A LATERAL source is its query, read as any derived source.
    assert con.execute(sql).fetchall() == [("CLAIM-SECRET",)]
    _withheld_three_ways(con, sql, "claim", "secret", "CLAIM-SECRET")
    assert isinstance(_decide(sql, FULL, policy=COMPLETE), Approved)


def test_a_generated_duplicate_name_is_refused_even_where_it_is_the_literal(con):
    # `(SELECT *, 'SAFE' AS secret FROM claim) d`: the star's secret is first, the literal is the
    # later duplicate, so DuckDB's `d.secret_1` is the literal. The engine's renaming is not
    # modelled (its collision rules defeated a count), so any `name_k` read over an output that
    # carries `name` falls closed, this one included; name the columns distinctly instead. An
    # honest column named with a suffix, over an output without the base, reads directly.
    sql = "SELECT d.secret_1 FROM (SELECT *, 'SAFE' AS secret FROM claim) d"
    assert con.execute(sql).fetchall() == [("SAFE",)]
    verdict = _decide(sql, FULL, policy=COMPLETE)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    honest = _decide("SELECT d.note_1 FROM (SELECT note AS note_1 FROM wide) d", FULL, policy=COMPLETE)
    assert isinstance(honest, Approved), honest


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

"""M30 + M7 — a write's READS are governed by the same RLS as a read's.

Two findings, one defect: `apply_row_and_mask` is the RLS implementation and the write decider
re-implemented it inline, so the two disagreed about *where* a filter binds (M30) and about
*whether a filter is valid at all* (M7).

The tests are written as PARITY tests on purpose. Asserting the write path's behaviour alone
would let the two implementations drift again in the other direction; asserting that both paths
answer the same question the same way is what makes the duplication impossible to reintroduce
without a red test.
"""

import duckdb
import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_VISIBLE = {"claim": {"id", "amount", "region"}, "scratch": {"id", "amount"}}
_SCHEMA = {k: set(v) for k, v in _VISIBLE.items()}
_FILTER = "region = 'west'"


def _policy(filt: str = _FILTER) -> AccessPolicy:
    return AccessPolicy(row_filters={"claim": filt}, policy_schema=_SCHEMA)


def _grants() -> GrantSet:
    return GrantSet(frozenset(_VISIBLE), writable=frozenset({"claim", "scratch"}))


class _OkAdapter:
    def execute(self, sql):
        return []


def _write(sql: str, policy: AccessPolicy | None = None):
    return decide_write(sql, _VISIBLE, _grants(), adapter=_OkAdapter(), dialect="duckdb",
                        writes_enabled=True, policy=policy if policy is not None else _policy())


# -- placement: every table a write READS is filtered, not just the one it writes -----------------

# The target of the mutation (A, B) already bound before this fix; C-G are every way a write
# reads, and none of them did. C is the one that matters: the governed rows land in an ungoverned
# table, where a later plain SELECT returns them forever.
_READS = [
    ("C_insert_select", "INSERT INTO scratch (id, amount) SELECT id, amount FROM claim"),
    ("D_update_subquery", "UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM claim)"),
    ("E_update_from", "UPDATE scratch SET amount = claim.amount FROM claim "
                      "WHERE scratch.id = claim.id"),
    ("F_delete_subquery", "DELETE FROM scratch WHERE id IN (SELECT id FROM claim)"),
    ("G_insert_cte", "INSERT INTO scratch (id, amount) "
                     "WITH c AS (SELECT id, amount FROM claim) SELECT id, amount FROM c"),
]
_TARGETS = [
    ("A_update_target", "UPDATE claim SET amount = 0 WHERE id = 1"),
    ("B_delete_target", "DELETE FROM claim WHERE id = 1"),
]


@pytest.mark.parametrize("label,sql", _READS + _TARGETS, ids=lambda v: v if isinstance(v, str) else "")
def test_every_governed_table_a_write_touches_is_filtered(label, sql):
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite), f"{label}: {verdict}"
    assert _FILTER.split("=")[0].strip() in verdict.plan_sql, (
        f"{label}: the write reads `claim` with no row filter -- {verdict.plan_sql}"
    )


def test_an_insert_target_is_not_filtered_because_an_insert_does_not_read_it():
    # The exemption is deliberate and must stay: filtering the target of an INSERT would refuse
    # rows on the way IN, which is not what a row filter means.
    verdict = _write("INSERT INTO claim (id, amount, region) VALUES (1, 2.0, 'east')")
    assert isinstance(verdict, ApprovedWrite)
    assert "region = 'west'" not in verdict.plan_sql


def test_the_target_is_excluded_by_identity_and_not_by_name():
    # Same table in two positions: the inner read must be wrapped, the outer target conjoined.
    # Excluding by NAME would skip both and reintroduce M31's flat-name-set bypass.
    verdict = _write("UPDATE claim SET amount = 0 "
                     "WHERE id IN (SELECT id FROM claim WHERE amount > 5)")
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.plan_sql.count("region = 'west'") == 2, verdict.plan_sql


# -- validation parity: both paths refuse the same filters for the same reason --------------------

# The read path proves a filter speaks only about what it is entitled to. The write path parsed it
# raw, so a policy typo crashed the decider and a filter naming an arbitrary table was spliced in
# unvalidated -- "the database as the control", which is the failure `decide_write`'s own docstring
# says M3 was.
_INVALID = [
    ("unparseable", "region = = 'west'"),
    ("not_a_predicate", "DELETE FROM claim"),
    ("column_not_on_table", "no_such_column = 1"),
    ("foreign_table", "region = (SELECT region FROM secret_table)"),
    ("the_stand_in", "__mnemiq_filtered__.region = 'west'"),
]


@pytest.mark.parametrize("label,filt", _INVALID, ids=[c[0] for c in _INVALID])
def test_an_invalid_row_filter_is_refused_identically_on_both_paths(label, filt):
    read = apply_row_and_mask(sqlglot.parse_one("SELECT id FROM claim", read="duckdb"),
                              _policy(filt), _VISIBLE, dialect="duckdb")
    assert isinstance(read, Refusal) and read.code == RefusalCode.INVALID_ROW_FILTER

    write = _write("UPDATE claim SET amount = 0 WHERE id = 1", _policy(filt))
    assert isinstance(write, Refusal), f"{label}: the write path accepted it -- {write}"
    assert write.code == read.code, f"{label}: {write.code} != {read.code}"


@pytest.mark.parametrize("label,filt", _INVALID, ids=[c[0] for c in _INVALID])
def test_no_exception_escapes_the_write_decider_for_an_invalid_filter(label, filt):
    # A policy typo must be a refusal, not a traceback in the caller. `sqlglot.parse_one` on a
    # malformed filter raised straight out of `decide_write`.
    for sql in ("UPDATE claim SET amount = 0 WHERE id = 1",
                "INSERT INTO scratch (id, amount) SELECT id, amount FROM claim"):
        verdict = _write(sql, _policy(filt))  # must not raise
        assert isinstance(verdict, (ApprovedWrite, Refusal))


# -- executed, not merely emitted -----------------------------------------------------------------


def _con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, amount DOUBLE, region VARCHAR)")
    con.execute("INSERT INTO claim VALUES (1, 10, 'west'), (2, 20, 'east'), (3, 30, 'west')")
    con.execute("CREATE TABLE scratch (id INTEGER, amount DOUBLE)")
    return con


@pytest.mark.parametrize("label,sql", _READS, ids=[c[0] for c in _READS])
def test_the_governed_write_moves_only_in_filter_rows(label, sql):
    """Run the approved SQL. A test that greps `plan_sql` passes on SQL that does not execute."""
    con = _con()
    con.execute("INSERT INTO scratch VALUES (1, 99), (2, 99), (3, 99)")
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite), verdict
    con.execute(verdict.target_sql)
    governed = con.execute("SELECT id, amount FROM scratch ORDER BY id, amount").fetchall()

    # The passing control: the same statement with no policy at all. If the two agree, the test
    # is measuring nothing -- which is how a guard that never fires reads as a guard that holds.
    ctl = _con()
    ctl.execute("INSERT INTO scratch VALUES (1, 99), (2, 99), (3, 99)")
    ungoverned = _write(sql, AccessPolicy())
    assert isinstance(ungoverned, ApprovedWrite)
    ctl.execute(ungoverned.target_sql)
    unfiltered = ctl.execute("SELECT id, amount FROM scratch ORDER BY id, amount").fetchall()

    assert governed != unfiltered, f"{label}: the filter changed nothing -- the control agrees"
    assert (2, 20.0) not in governed, f"{label}: the east row escaped into scratch -- {governed}"


# -- one implementation ---------------------------------------------------------------------------


_RENDERINGS = [
    ("filter", "SELECT id, amount FROM claim", _policy()),
    # A filter that reaches through another table (M28's shape) renders a derived table over a
    # subquery, and resolves that subquery against `policy_schema` rather than the caller's
    # scope -- a materially different rendering from the plain predicate above.
    #
    # There is deliberately no masked case: a write needs RAW access, so `decide_write` treats a
    # masked column as denied and refuses before any rewrite happens. The mask half of
    # `apply_row_and_mask` is unreachable from the write path by design, which is why the
    # equality here is over row filters only.
    ("subquery_filter", "SELECT id, amount FROM claim",
     AccessPolicy(row_filters={"claim": "id IN (SELECT id FROM entitlement)"},
                  policy_schema={**_SCHEMA, "entitlement": {"id"}})),
    ("two_references", "SELECT a.id, b.amount FROM claim AS a JOIN claim AS b ON a.id = b.id",
     _policy()),
    ("no_policy", "SELECT id, amount FROM claim", AccessPolicy()),
]


@pytest.mark.parametrize("label,select_sql,policy", _RENDERINGS, ids=[r[0] for r in _RENDERINGS])
def test_the_write_path_rewrites_a_read_exactly_as_the_read_path_does(label, select_sql, policy):
    """The security invariant behind M7, asserted as an equality between the two paths' OUTPUT.

    This is what the source-placement guard below was pretending to be. An INSERT's SELECT is a
    read, so the write path's rendering of it must equal `apply_row_and_mask`'s character for
    character -- across a plain filter, a filter that reaches through another table, two
    references to one filtered table, and no policy at all.

    Scope, stated because the previous version of this docstring overran it: four renderings are
    evidence about four renderings. A second implementation that agreed on all four and diverged
    on a fifth would pass here, and the 14 shapes below are what make that narrow. What this
    forecloses is the cheap way the duplicate came back the first time -- something that renders
    *nearly* the same.
    """
    read = sqlglot.parse_one(select_sql, read="duckdb")
    governed_read = apply_row_and_mask(read, policy, _VISIBLE, dialect="duckdb")
    assert not isinstance(governed_read, Refusal)

    verdict = _write(f"INSERT INTO scratch (id, amount) {select_sql}", policy)
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.plan_sql == (
        "INSERT INTO scratch (id, amount) " + governed_read.sql(dialect="duckdb")
    )


def test_the_write_decider_constructs_no_row_filter_of_its_own():
    """An ARCHITECTURAL guard on source placement -- not a security invariant, and it must not be
    read as one. The invariant is the equality above and the 14 shapes below.

    What it actually asserts: the three constructs that made up the deleted inline copy do not
    reappear in `decide_write.py`. It cannot see a second implementation moved to another module,
    spelled through an alias, or written with different AST calls -- and `rls.py` legitimately
    contains all three today, inside `apply_row_filters_to_write`. So the honest claim is that
    the duplicate has been RELOCATED to sit beside the original and share its validator, which is
    what stops them drifting; the claim is not that the strings are gone from the codebase.

    Kept anyway, because the specific regression it blocks is the specific one that happened:
    someone editing the write decider adds "just one" filter back where the old copy lived. It is
    the third version of this test. The first asserted a symbol name, which the better design
    does not satisfy; the second matched the prose of the comment explaining the fix, which is a
    test of the changelog.
    """
    import inspect

    from mnemiq.sql import decide_write as write_module
    from mnemiq.sql import rls as rls_module

    # Comments stripped: the assertion is about what the module DOES. The first version of this
    # test matched the prose of the comment explaining the fix, which is a test of the changelog.
    write_src = "\n".join(
        line for line in inspect.getsource(write_module).splitlines()
        if not line.strip().startswith("#")
    )
    assert "mnemiq.sql.rls" in write_src, "the write path must route through the RLS module"
    for built_here in ("exp.Where(this=", "_validate_filter", "row_filters.get"):
        assert built_here not in write_src, (
            f"the write decider builds its own filter ({built_here!r}) -- that is the second "
            "implementation M7 filed"
        )
    # And the wrap itself is constructed in exactly one place.
    assert inspect.getsource(rls_module).count("def _derived_table") == 1


# -- the class, not the instance ------------------------------------------------------------------

# M39's sharpest lesson is that fixing an instance is not fixing a class. The finding named two
# shapes (INSERT..SELECT, UPDATE..IN-subquery); the guard has to hold for every way a write can
# read. Each of these was run against the fix and each binds -- including the shapes that read a
# governed table from somewhere other than a FROM clause.
_EVERY_READ_SHAPE = [
    ("subquery_in_set", "UPDATE scratch SET amount = (SELECT max(amount) FROM claim) WHERE id = 1"),
    ("subquery_in_values",
     "INSERT INTO scratch (id, amount) VALUES (1, (SELECT max(amount) FROM claim))"),
    ("correlated_exists", "UPDATE scratch SET amount = 0 "
                          "WHERE EXISTS (SELECT 1 FROM claim WHERE claim.id = scratch.id)"),
    ("union_source", "INSERT INTO scratch (id, amount) "
                     "SELECT id, amount FROM claim UNION ALL SELECT id, amount FROM claim"),
    ("nested_cte", "INSERT INTO scratch (id, amount) WITH a AS (SELECT id, amount FROM claim), "
                   "b AS (SELECT id, amount FROM a) SELECT id, amount FROM b"),
    ("recursive_cte", "INSERT INTO scratch (id, amount) WITH RECURSIVE r AS "
                      "(SELECT id, amount FROM claim UNION ALL SELECT id, amount FROM r WHERE id > 0) "
                      "SELECT id, amount FROM r"),
    # M31's shape: a CTE named after the governed table. The reference inside the CTE's own body
    # reads the real table and must be wrapped; the outer reference is the CTE and must not be.
    ("cte_shadowing_the_name", "INSERT INTO scratch (id, amount) "
                               "WITH claim AS (SELECT id, amount FROM claim) SELECT id, amount FROM claim"),
    ("lateral_join", "INSERT INTO scratch (id, amount) SELECT s.id, c.amount FROM scratch s, "
                     "LATERAL (SELECT amount FROM claim WHERE claim.id = s.id) c"),
    ("scalar_in_where", "DELETE FROM scratch WHERE amount > (SELECT avg(amount) FROM claim)"),
    ("returning", "DELETE FROM scratch WHERE id = 1 RETURNING (SELECT max(amount) FROM claim)"),
    ("on_conflict_do_update", "INSERT INTO scratch (id, amount) VALUES (1, 2) "
                              "ON CONFLICT (id) DO UPDATE SET amount = (SELECT max(amount) FROM claim)"),
    ("window_function",
     "INSERT INTO scratch (id, amount) SELECT id, row_number() OVER (ORDER BY amount) FROM claim"),
    ("join", "INSERT INTO scratch (id, amount) SELECT c.id, c.amount FROM claim c "
             "JOIN scratch s ON s.id = c.id"),
    ("self_join_aliased", "INSERT INTO scratch (id, amount) SELECT a.id, b.amount FROM claim a "
                          "JOIN claim b ON a.id = b.id"),
]


@pytest.mark.parametrize("label,sql", _EVERY_READ_SHAPE, ids=[c[0] for c in _EVERY_READ_SHAPE])
def test_no_write_shape_reads_a_governed_table_unfiltered(label, sql):
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite), f"{label}: {verdict}"
    assert "region = 'west'" in verdict.plan_sql, (
        f"{label}: reads `claim` with no row filter -- {verdict.plan_sql}"
    )


@pytest.mark.parametrize("existing_where", ["id = 1 OR id = 2", "id = 1 OR 1 = 1"])
@pytest.mark.parametrize("filt", ["region = 'west'", "region = 'west' OR region = 'north'"])
def test_the_target_conjunction_binds_tighter_than_an_existing_or(existing_where, filt):
    """`WHERE a OR b AND filt` is not `WHERE (a OR b) AND filt`. The conjunction must parenthesise
    both sides or an OR in either half swallows the filter -- the oldest bypass in the genre."""
    verdict = _write(f"UPDATE claim SET amount = 0 WHERE {existing_where}", _policy(filt))
    assert isinstance(verdict, ApprovedWrite)

    con = _con()
    con.execute(verdict.target_sql)
    touched = {r[0] for r in con.execute("SELECT id FROM claim WHERE amount = 0").fetchall()}
    in_filter = {r[0] for r in con.execute(f"SELECT id FROM claim WHERE {filt}").fetchall()}
    assert touched <= in_filter, f"mutated rows outside the filter: {touched - in_filter}"


# -- the residual, recorded as a tripwire rather than as prose --------------------------------------


@pytest.mark.xfail(strict=True, reason="M30's residual: a statement-level WITH parks its CTEs in "
                                       "the write root's `with_` arg, outside where build_scope "
                                       "roots itself, so every guard sees an empty statement. This "
                                       "asserts the GOVERNED outcome, so it xpasses only when the "
                                       "shape is scope-resolved (M48's `_unscoped_ctes`) and "
                                       "nothing stands in its place. It does NOT distinguish "
                                       "'approved ungoverned' from 'refused' -- both leave it "
                                       "xfailing, which is why test_write_governance_floors.py "
                                       "asserts the disposition directly.")
@pytest.mark.parametrize("sql", [
    "WITH x AS (SELECT id, amount FROM claim) "
    "INSERT INTO scratch (id, amount) SELECT id, amount FROM x",
    "WITH x AS (SELECT id FROM claim) UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM x)",
    "WITH x AS (SELECT id FROM claim) DELETE FROM scratch WHERE id IN (SELECT id FROM x)",
])
def test_a_statement_level_with_is_still_invisible_to_the_rewrite(sql):
    """M30 is closed for every shape above and open for this one.

    The 14-shape sweep has a CTE case and still missed this: it spells the WITH *inside* the
    INSERT, which sqlglot parses into the projection where `build_scope` can see it. A leading
    WITH parses somewhere else entirely. Shapes generated from one spelling read as covering the
    class -- which is how a sweep can be exhaustive and blind at once.

    What this tripwire pinned when it was written was too narrow, and that is the more useful
    lesson: it asserts an RLS predicate on a table that IS granted, so it recorded a missing
    filter. Measured later, the same shape approved a read of a table with no grant at all and a
    reference to a column that does not exist -- the guards were not filtering less, they were
    not running. A tripwire built from the symptom you noticed pins the symptom you noticed.
    `test_write_governance_floors.py` carries the authorization dimensions.
    """
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite)
    assert "region = 'west'" in verdict.plan_sql

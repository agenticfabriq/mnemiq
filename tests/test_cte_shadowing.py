"""M31: a CTE named after a source object must not disarm the access controls.

`WITH t AS (SELECT ... FROM t) SELECT ... FROM t` is legal SQL in which the inner
reference reads the **base table** and the outer one reads the CTE. Three guards
used to skip both, because each asked only whether the name appeared in a flat set
of CTE names -- so table authorization, column deny/mask and row filters all fell
to the same blind spot at once.

Each test carries the equivalent un-shadowed query as a control: the point is not
that the shadowed form is refused, but that it is treated exactly like the query
it is a disguise for.
"""

from __future__ import annotations

import pytest
import sqlglot

from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Refusal
from mnemiq.sql.decide_write import decide_write
from mnemiq.authz.grants import GrantSet
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.scope import base_tables, column_tables
from mnemiq.sql.verdict import Approved, RefusalCode

DIALECT = "postgres"

VISIBLE = {
    "customer": {"customer_id", "store_id", "first_name", "email"},
    "rental": {"rental_id", "customer_id"},
}
FILTER = AccessPolicy(row_filters={"customer": "store_id = 1"})
MASK = AccessPolicy(masked={("customer", "email")})
DENY = AccessPolicy(denied={("customer", "email")})


def plan(sql: str, visible=VISIBLE, policy: AccessPolicy | None = None):
    return decide(
        sql, visible, dialect=DIALECT, target=DIALECT, policy=policy or AccessPolicy()
    )


# --- table authorization -------------------------------------------------------


def test_a_cte_cannot_launder_a_table_the_identity_has_no_grant_on():
    """The strongest form: `staff` is not in `visible` at all."""
    only_rental = {"rental": {"rental_id", "customer_id"}}

    control = plan("SELECT staff_id FROM staff", visible=only_rental)
    assert control.code is RefusalCode.UNAUTHORIZED_TABLE

    shadowed = plan(
        "WITH staff AS (SELECT staff_id FROM staff) SELECT staff_id FROM staff",
        visible=only_rental,
    )
    assert shadowed.code is RefusalCode.UNAUTHORIZED_TABLE, (
        "a CTE named after an ungranted table read it in full"
    )


# --- column-level policy -------------------------------------------------------


def test_a_cte_cannot_launder_a_denied_column():
    control = plan("SELECT email FROM customer", policy=DENY)
    assert control.code is RefusalCode.UNAUTHORIZED_COLUMN

    shadowed = plan(
        "WITH customer AS (SELECT customer_id, email FROM customer) SELECT email FROM customer",
        policy=DENY,
    )
    assert shadowed.code is RefusalCode.UNAUTHORIZED_COLUMN


def test_a_cte_cannot_launder_a_masked_column():
    control = plan("SELECT email FROM customer", policy=MASK)
    assert isinstance(control, Approved) and "NULL AS email" in control.plan_sql

    shadowed = plan(
        "WITH customer AS (SELECT customer_id, email FROM customer) SELECT email FROM customer",
        policy=MASK,
    )
    assert isinstance(shadowed, Approved)
    assert "NULL AS email" in shadowed.plan_sql, "the mask was dropped inside the CTE body"


# --- row filters ---------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # the plain shadow
        "WITH customer AS (SELECT customer_id, store_id FROM customer) "
        "SELECT customer_id FROM customer",
        # RECURSIVE: the parser puts the self-reference in the same place
        "WITH RECURSIVE customer AS (SELECT customer_id, store_id FROM customer) "
        "SELECT customer_id FROM customer",
        # the shadow is the SECOND cte, reading the real table through the first
        "WITH c1 AS (SELECT customer_id, store_id FROM customer), "
        "customer AS (SELECT customer_id, store_id FROM c1) SELECT customer_id FROM customer",
        # the outer query filters for a store the identity may not see
        "WITH customer AS (SELECT customer_id, store_id FROM customer) "
        "SELECT customer_id FROM customer WHERE store_id = 2",
    ],
    ids=["plain", "recursive", "nested", "reads-another-store"],
)
def test_a_cte_cannot_launder_a_row_filter(sql):
    control = plan("SELECT customer_id FROM customer", policy=FILTER)
    assert isinstance(control, Approved) and "store_id = 1" in control.plan_sql

    shadowed = plan(sql, policy=FILTER)
    assert isinstance(shadowed, Approved), f"unexpected refusal: {shadowed}"
    assert "store_id = 1" in shadowed.plan_sql, "the row filter was dropped inside the CTE body"


# --- what must keep working ----------------------------------------------------


def test_an_ordinary_cte_is_still_not_treated_as_a_source_object():
    """The reason the flat skip existed: a CTE name is not an object to authorize."""
    v = plan(
        "WITH recent AS (SELECT customer_id FROM rental) SELECT customer_id FROM recent",
        policy=FILTER,
    )
    assert isinstance(v, Approved), f"an ordinary CTE was refused: {v}"


def test_a_cte_selecting_from_a_filtered_table_is_filtered_at_the_base():
    v = plan(
        "WITH c AS (SELECT customer_id, store_id FROM customer) SELECT customer_id FROM c",
        policy=FILTER,
    )
    assert isinstance(v, Approved) and "store_id = 1" in v.plan_sql


def test_a_derived_table_sharing_the_name_was_already_correct():
    """Not a regression guard for the fix -- a guard that the fix did not break it."""
    v = plan(
        "SELECT customer_id FROM (SELECT customer_id, store_id FROM customer) AS customer",
        policy=FILTER,
    )
    assert isinstance(v, Approved) and "store_id = 1" in v.plan_sql


# -- M48: the same blind spot one layer down, in the resolver the three guards share -------------
#
# A write root parks its CTEs in its own WITH arg, outside the query `build_scope` roots itself
# at, so every guard saw an empty statement and approved it ungoverned. `check_write_shape`
# refuses this shape outright now (UNSCOPED_CTE), so these assert the FLOOR underneath that
# refusal: one guard's decision must not be the only thing standing between a write and an
# unresolved scope, because `base_tables` is the premise all three of them share.
#
# `build_scope` is `traverse_scope(...)[-1]`, and traverse_scope DOES visit the write root's
# CTEs -- taking only the last scope is what discards them. Measured, all three shapes below:
# traverse_scope recovers {claim, x} and the write target `scratch` appears in NO scope. For
# the INSERT that is correct, because an INSERT target is written and never read. For the
# UPDATE and DELETE it is not: those read the target to find their rows. So swapping the call
# site governs the reads and silently drops the target on two shapes of three -- the floor is
# not one call away.

_WRITE_WITH_LEADING_CTE = [
    "WITH x AS (SELECT id, amount FROM claim) "
    "INSERT INTO scratch (id, amount) SELECT id, amount FROM x",
    "WITH x AS (SELECT id FROM claim) UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM x)",
    "WITH x AS (SELECT id FROM claim) DELETE FROM scratch WHERE id IN (SELECT id FROM x)",
]


@pytest.mark.parametrize("sql", _WRITE_WITH_LEADING_CTE)
def test_the_resolver_over_reports_rather_than_answering_empty(sql):
    """`[]` would mean "this statement reads no real object", and `check_access` acts on that:
    its `if not alias_to_table: return None` early-out is justified by "their sources were
    checked above", and above, nothing was. Over-reporting refuses the statement on `x`, which
    is the direction to fail in -- so this floor DECLINES, it does not govern."""
    ast = sqlglot.parse_one(sql, read="duckdb")
    assert {t.name for t in base_tables(ast)} == {"claim", "scratch", "x"}
    assert column_tables(ast) is None


@pytest.mark.parametrize("sql", _WRITE_WITH_LEADING_CTE)
def test_the_resolver_floor_is_not_keyed_on_a_sqlglot_arg_name(sql):
    """`pyproject` declares `sqlglot>=25`. 25.x spells this arg `with` where 30.x spells it
    `with_`, so a name-keyed lookup returns None across much of the declared range and fails
    OPEN exactly where it must fail closed. Simulated by re-keying the node the parser produced:
    the floor is structural, so the spelling cannot reach it."""
    ast = sqlglot.parse_one(sql, read="duckdb")
    node = ast.args.pop("with_", None) or ast.args.pop("with", None)
    assert node is not None, "the parser produced neither spelling -- the fixture is inert"
    for spelling in ("with", "with_"):
        probe = ast.copy()
        probe.args[spelling] = node
        assert {t.name for t in base_tables(probe)} == {"claim", "scratch", "x"}, spelling


def test_the_same_cte_spelled_inside_the_insert_is_still_resolved():
    """The control that keeps this a floor and not a ban: this spelling parses into the
    projection, where `build_scope` sees it, and the resolver reads it exactly."""
    ast = sqlglot.parse_one(
        "INSERT INTO scratch (id, amount) "
        "WITH c AS (SELECT id, amount FROM claim) SELECT id, amount FROM c", read="duckdb")
    assert {t.name for t in base_tables(ast)} == {"claim"}
    assert column_tables(ast) is not None


@pytest.mark.parametrize("sql", [
    "WITH x AS (SELECT id FROM claim) SELECT id FROM x",
    "WITH claim AS (SELECT id FROM claim) SELECT id FROM claim",
    "SELECT * FROM (WITH y AS (SELECT id FROM claim) SELECT id FROM y) AS z",
])
def test_the_floor_never_fires_on_a_read(sql):
    """A read's WITH is always owned by the query `build_scope` roots at, so this is a
    write-root mechanism. Asserted because M49's ACL was claimed to depend on it and does not --
    the coupling was latent, and a test is what keeps that answer true."""
    ast = sqlglot.parse_one(sql, read="duckdb")
    assert {t.name for t in base_tables(ast)} == {"claim"}
    assert column_tables(ast) is not None


# -- the residual these floors do NOT cover, as tripwires rather than as prose ------------------
#
# A known residual is a strict xfail, not a sentence: prose cannot fail. Both were found by the
# review gate measuring claims I had written into the comments above and never run.
#
# Three reachable defects, kept apart because they fail through different guards.
#
# `strict=True` is what catches a decider that starts refusing everything: an XPASS becomes a
# failure. A control asserting a REFUSAL guards the opposite direction only when the thing it
# refuses is specific -- which is why the controls below pin a refusal CODE, not just a refusal.

_UPSERT_VISIBLE = {"claim": {"id", "amount"}, "scratch": {"id", "amount"}}
_UPSERT_GRANTS = GrantSet(objects=frozenset(_UPSERT_VISIBLE), writable=frozenset({"scratch"}))
_DENY_AMOUNT = AccessPolicy(denied={("scratch", "amount")})


def _upsert_verdict(sql, policy=_DENY_AMOUNT):
    return decide_write(sql, _UPSERT_VISIBLE, _UPSERT_GRANTS, adapter=None, dialect="duckdb",
                        policy=policy, views={}, writes_enabled=True)


def test_a_denied_column_on_the_target_is_refused_when_the_resolver_gives_up():
    """The control, and the reason the residual is invisible: `build_scope` returns None for
    this shape, so `base_tables` takes the `find_all` fallback, `scratch` lands in the read set
    and `check_cls` refuses. Every refusal below is this fallback, never a check on the target."""
    v = _upsert_verdict("UPDATE scratch SET amount = 0 WHERE id = 1")
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_COLUMN


@pytest.mark.xfail(strict=True, reason="An UPDATE/DELETE target is absent from `base_tables` "
                                       "whenever `build_scope` SUCCEEDS, because the target is "
                                       "not among `scope.sources`. `check_cls` builds its "
                                       "`referenced` set from `base_tables`, so a denied column "
                                       "ON THE TARGET can never match and the write is approved "
                                       "with it. Also disproves `_target_table`'s docstring, "
                                       "which M50's fix relies on.")
@pytest.mark.parametrize("sql", [
    "UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM claim)",
    "DELETE FROM scratch WHERE amount = 0 AND id IN (SELECT id FROM claim)",
])
def test_a_denied_column_on_the_target_is_refused_when_the_resolver_succeeds(sql):
    v = _upsert_verdict(sql)
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_COLUMN


@pytest.mark.xfail(strict=True, reason="`ON CONFLICT DO UPDATE` READS its target, so the premise "
                                       "that an INSERT target is written and never read -- which "
                                       "is why `base_tables` may omit it -- does not hold for an "
                                       "upsert. `column_tables` cannot resolve `scratch.amount` "
                                       "when `scratch` is not in `scope.sources`, so "
                                       "`_candidate_tables` falls through to `referenced` and "
                                       "evaluates `scratch.amount` against `claim` -- the wrong "
                                       "table, not a skipped column. Root cause is `base_tables` "
                                       "omitting the target, not a missing branch in check_cls.")
def test_an_upsert_may_not_read_a_denied_column_on_its_target():
    v = _upsert_verdict("INSERT INTO scratch (id, amount) SELECT id, amount FROM claim "
                        "ON CONFLICT (id) DO UPDATE SET amount = scratch.amount + 1")
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_COLUMN


def test_an_ordinary_governed_upsert_is_still_approved():
    """Guards the refuse-everything direction: nothing is denied, so this upsert must go
    through. `strict=True` on the xfails above does the same job from the other side."""
    v = _upsert_verdict("INSERT INTO scratch (id, amount) SELECT id, amount FROM claim "
                        "ON CONFLICT (id) DO UPDATE SET amount = scratch.amount + 1",
                        policy=AccessPolicy())
    assert not isinstance(v, Refusal)


# The third defect, and it is a statement SHAPE rather than a missing snapshot. An earlier draft
# blamed `Runtime.write` passing `{}`; measured, the approval does not depend on `visible` at all.
# `UPDATE scratch SET amount = 0 WHERE id IN (SELECT 1)` resolves to an EMPTY read set -- the
# subquery names no table and the target is in no `scope.sources` -- so `check_access` takes its
# `if not alias_to_table` early-out and no guard evaluates the target under any snapshot.

# Only snapshots that do NOT describe the target: refusing is the correct fixed verdict
# there. A snapshot that DOES describe it must stay approved, which is the control below.
_TARGET_UNDESCRIBED = [{}, {"claim": {"id"}}]
_SCRATCH_GRANTS = GrantSet(objects=frozenset({"scratch"}), writable=frozenset({"scratch"}))


def _shape_verdict(sql, visible):
    return decide_write(sql, visible, _SCRATCH_GRANTS, adapter=None, dialect="duckdb",
                        policy=AccessPolicy(), views={}, writes_enabled=True)


def test_check_access_does_run_for_this_statement_shape():
    """The discriminating control. Same shape, but the subquery names a table the caller cannot
    read, so `base_tables` is non-empty and the early-out is not taken. Without this, the xfail
    below cannot tell 'the guard ran and passed' from 'the guard never ran'."""
    v = _shape_verdict("UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM secret)",
                       {"scratch": {"id", "amount"}})
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_TABLE


@pytest.mark.xfail(strict=True, reason="A write whose read set resolves EMPTY skips table "
                                       "authorization entirely: `base_tables` is [] because the "
                                       "subquery names no table and an UPDATE target is in no "
                                       "`scope.sources`, so `check_access` early-outs on "
                                       "`if not alias_to_table`. Independent of the snapshot -- "
                                       "approved identically for an empty one, one omitting the "
                                       "target, and one describing it. `grants.allows_write` is "
                                       "the only guard the statement meets. Snapshot-independent: approved identically whether the snapshot omits the target or describes it -- the describing case is the control below, where approval is CORRECT and must survive the fix.")
@pytest.mark.parametrize("visible", _TARGET_UNDESCRIBED, ids=["no-snapshot", "omits-target"])
def test_a_write_with_an_empty_read_set_is_still_authorized_against_the_snapshot(visible):
    v = _shape_verdict("UPDATE scratch SET amount = 0 WHERE id IN (SELECT 1)", visible)
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_TABLE


def test_a_write_whose_target_the_snapshot_describes_stays_approved():
    """The positive control for the empty-read-set shape, and the reason the xfail above is not
    parametrized over this snapshot: here the caller is fully entitled -- target readable,
    writable, both columns real -- so approval is the CORRECT verdict and must survive any fix
    to the residual. Without this, closing the residual by refusing the shape outright would
    look like success."""
    v = _shape_verdict("UPDATE scratch SET amount = 0 WHERE id IN (SELECT 1)",
                       {"scratch": {"id", "amount"}})
    assert not isinstance(v, Refusal), v

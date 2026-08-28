"""The write path's governance is the READ path's governance, or it is not parity.

M30 + M7 unified the RLS *rewrite* and stopped there, and an adversarial review found the gap
that framing left: a read is governed by the rewrite AND by the floors around it, and the write
path had none of the floors. Three shapes walked straight through a write that a read refuses.

Each floor here is the same design as `check_views` (M27): refuse the shape that cannot be
governed, rather than model it and fail open on the shapes the model misses. Each is removable
the day the thing it stands in for is resolved, and each is paired with a control proving it
refuses the ungovernable shape rather than the whole feature.
"""

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract.semantic import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode


class _OkAdapter:
    def execute(self, sql):
        return []


# -- floor 1: a write reads through a governed view --------------------------------------------

# Measured before the fix, with the read path as the control:
#   read  via decide       REFUSED(UNGOVERNED_VIEW)
#   write via decide_write APPROVED  -> executed -> copied [(1,), (2,), (3,)], west-only is [(1,)]
# `decide_write` had no `views` parameter at all, so `check_views` could not run on a write even
# in principle. The RLS rewrite cannot see through a view, so a filter on the view's BASE table
# reached nothing and the rows landed, durably, in an ungoverned table.

_V_VISIBLE = {"claim_view": {"id", "amount", "region"}, "scratch": {"id", "amount"}}
_V_SCHEMA = {"claim": {"id", "amount", "region"}, **{k: set(v) for k, v in _V_VISIBLE.items()}}
_VIEWS = {"claim_view": ViewDefinition(object_id="claim_view",
                                       definition="SELECT * FROM claim", dialect="duckdb")}


def _v_policy(filtered: bool = True) -> AccessPolicy:
    return AccessPolicy(row_filters={"claim": "region = 'west'"} if filtered else {},
                        policy_schema=_V_SCHEMA)


def _v_grants() -> GrantSet:
    return GrantSet(frozenset(_V_VISIBLE), writable=frozenset({"scratch"}))


_VIEW_READERS = [
    ("insert", "INSERT INTO scratch (id, amount) SELECT id, amount FROM claim_view"),
    ("update", "UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM claim_view)"),
    ("delete", "DELETE FROM scratch WHERE id IN (SELECT id FROM claim_view)"),
]


@pytest.mark.parametrize("label,sql", _VIEW_READERS, ids=[c[0] for c in _VIEW_READERS])
def test_a_write_reading_a_governed_view_is_refused_as_a_read_is(label, sql):
    verdict = decide_write(sql, _V_VISIBLE, _v_grants(), adapter=_OkAdapter(),
                           policy=_v_policy(), views=_VIEWS, writes_enabled=True)
    assert isinstance(verdict, Refusal)
    assert verdict.code is RefusalCode.UNGOVERNED_VIEW


def test_the_read_path_refuses_the_same_view_the_write_path_now_does():
    """The control that makes the floor a PARITY claim rather than a write-path opinion."""
    read = decide("SELECT id, amount FROM claim_view", _V_VISIBLE, adapter=_OkAdapter(),
                  dialect="duckdb", target="duckdb", policy=_v_policy(), views=_VIEWS)
    assert isinstance(read, Refusal)
    assert read.code is RefusalCode.UNGOVERNED_VIEW


@pytest.mark.parametrize("label,sql", _VIEW_READERS, ids=[c[0] for c in _VIEW_READERS])
def test_an_unfiltered_view_is_still_writable(label, sql):
    """The floor refuses views over FILTERED tables, not views. Without this control the fix
    above is indistinguishable from banning views from the write path entirely."""
    verdict = decide_write(sql, _V_VISIBLE, _v_grants(), adapter=_OkAdapter(),
                           policy=_v_policy(filtered=False), views=_VIEWS, writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite)


def test_a_write_with_no_view_snapshot_is_unaffected():
    """Every existing caller passes no views; the floor must be inert for them."""
    verdict = decide_write("INSERT INTO scratch (id, amount) SELECT id, amount FROM scratch",
                           _V_VISIBLE, _v_grants(), adapter=_OkAdapter(),
                           policy=_v_policy(), writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite)


# -- floor 2: a statement-level WITH is invisible to every guard ---------------------------------

# sqlglot parks a LEADING `WITH` in the write root's `with_` arg, outside where `build_scope`
# roots itself -- so `base_tables` returns [] and each guard sees an empty statement. The M30
# xfails pinned only the RLS symptom, on a table that was granted. Measured, with the plain
# spellings as controls:
#   plain read of ungranted table  REFUSED(UNAUTHORIZED_TABLE)  |  leading WITH  APPROVED
#   plain unknown column           REFUSED(UNKNOWN_COLUMN)      |  leading WITH  APPROVED
#   -> executed, copying [(7, 700), (8, 800)] out of a table with no grant at all.
# So the residual was never only "the filter is missing"; it was "no guard runs". A floor is the
# honest response to that: refuse the spelling until its CTEs are inside the scope the guards
# walk. The equivalent INSERT-internal spelling stays approved and stays governed, which is what
# makes this a floor and not a ban on CTEs.

_W_VISIBLE = {"claim": {"id", "amount", "region"}, "scratch": {"id", "amount"}}
_W_SCHEMA = {k: set(v) for k, v in _W_VISIBLE.items()}


def _w_write(sql: str, visible=None, grants=None):
    visible = _W_VISIBLE if visible is None else visible
    grants = grants or GrantSet(frozenset(visible), writable=frozenset({"claim", "scratch"}))
    return decide_write(sql, visible, grants, adapter=_OkAdapter(), dialect="duckdb",
                        policy=AccessPolicy(row_filters={"claim": "region = 'west'"},
                                            policy_schema=_W_SCHEMA),
                        writes_enabled=True)


_LEADING_WITH = [
    "WITH x AS (SELECT id, amount FROM claim) "
    "INSERT INTO scratch (id, amount) SELECT id, amount FROM x",
    "WITH x AS (SELECT id FROM claim) UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM x)",
    "WITH x AS (SELECT id FROM claim) DELETE FROM scratch WHERE id IN (SELECT id FROM x)",
]


@pytest.mark.parametrize("sql", _LEADING_WITH)
def test_a_statement_level_with_is_refused_rather_than_approved_ungoverned(sql):
    verdict = _w_write(sql)
    assert isinstance(verdict, Refusal)
    assert verdict.code is RefusalCode.UNSCOPED_CTE


def test_the_floor_covers_the_authorization_dimension_not_only_the_filter():
    """The dimension the M30 xfails did not pin, and the one that made this critical: behind a
    leading WITH the identity read a table it holds no grant on whatsoever."""
    visible = {"scratch": {"id", "amount"}}  # `secret` is neither visible nor granted
    grants = GrantSet(frozenset(visible), writable=frozenset({"scratch"}))
    verdict = _w_write("WITH x AS (SELECT id, amount FROM secret) "
                       "INSERT INTO scratch (id, amount) SELECT id, amount FROM x",
                       visible=visible, grants=grants)
    assert isinstance(verdict, Refusal)


def test_the_same_cte_spelled_inside_the_insert_is_approved_and_governed():
    """The control that keeps this a floor rather than a ban: sqlglot parses this spelling into
    the projection, where `build_scope` sees it, so every guard runs and the filter binds."""
    verdict = _w_write("INSERT INTO scratch (id, amount) "
                       "WITH c AS (SELECT id, amount FROM claim) SELECT id, amount FROM c")
    assert isinstance(verdict, ApprovedWrite)
    assert "region = 'west'" in verdict.plan_sql


# -- floor 3: the target is resolved by exact spelling, or refused -------------------------------

# Measured: with `pg.claim` and `claim` both readable and only bare `claim` writable,
#   UPDATE claim    (control) APPROVED target='claim'
#   UPDATE pg.claim           APPROVED target='claim'   <- authorized against a DIFFERENT object
#   UPDATE other    (control) REFUSED(UNAUTHORIZED_WRITE)
# and executing it set pg.claim to 0 while bare claim kept 999. The grant check read `node.name`
# while the RLS half of the same change read `object_key` -- one change, two spellings of the
# same question. This is M50's root (no single object-id normalisation) in its worst direction:
# not a filter silently dropped, but a write authorized against an object it does not touch.

# `s` is visible on purpose. With it absent, `check_access` refuses UNAUTHORIZED_TABLE before the
# target resolver is ever reached, so the multi-target test below would be RED for a reason that
# has nothing to do with the floor -- passing its `isinstance(verdict, Refusal)` line on the
# wrong refusal and failing the next one. Not a false green; a test that cannot observe its
# subject in either direction, which is the same defect the runtime-write case had.
_Q_VISIBLE = {"claim": {"id", "amount"}, "pg.claim": {"id", "amount"}, "s": {"id"}}
_Q_POLICY = AccessPolicy(policy_schema={k: set(v) for k, v in _Q_VISIBLE.items()})


def _q_write(sql: str, writable: set[str]):
    return decide_write(sql, _Q_VISIBLE, GrantSet(frozenset(_Q_VISIBLE),
                                                  writable=frozenset(writable)),
                        adapter=_OkAdapter(), policy=_Q_POLICY, writes_enabled=True)


def test_a_qualified_target_is_not_authorized_by_a_bare_grant():
    verdict = _q_write("UPDATE pg.claim SET amount = 0 WHERE id = 1", {"claim"})
    assert isinstance(verdict, Refusal)
    assert verdict.code is RefusalCode.UNAUTHORIZED_WRITE
    assert verdict.subject == "pg.claim", "the refusal must name the object actually written"


def test_a_bare_target_is_still_authorized_by_a_bare_grant():
    verdict = _q_write("UPDATE claim SET amount = 0 WHERE id = 1", {"claim"})
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.target == "claim"


def test_a_qualified_grant_authorizes_the_qualified_target():
    """Exact spelling in BOTH directions -- otherwise the floor reads as 'qualifiers refuse'."""
    verdict = _q_write("UPDATE pg.claim SET amount = 0 WHERE id = 1", {"pg.claim"})
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.target == "pg.claim"


def test_an_insert_into_an_unknown_qualified_target_is_refused():
    """The shape with NO backstop, and the reason floor 3 cannot be pinned on UPDATE alone.

    An UPDATE/DELETE target is also a base table, so `check_access` refuses it when it is not
    visible. An INSERT target is not a base table and no guard before the grant check ever sees
    it: measured, `INSERT INTO pg.claim ... SELECT ... FROM claim` under `visible={'claim'}` and
    `writable={'claim'}` was APPROVED as `target='claim'` -- a write into an object the identity
    holds no grant on and that is not in the snapshot's visible set at all.
    """
    visible = {"claim": {"id", "amount"}}
    verdict = decide_write("INSERT INTO pg.claim (id, amount) SELECT id, amount FROM claim",
                           visible, GrantSet(frozenset(visible), writable=frozenset({"claim"})),
                           adapter=_OkAdapter(),
                           policy=AccessPolicy(policy_schema={"claim": {"id", "amount"}}),
                           writes_enabled=True)
    assert isinstance(verdict, Refusal)
    assert verdict.code is RefusalCode.UNAUTHORIZED_WRITE
    assert verdict.subject == "pg.claim"


def test_an_insert_into_a_bare_granted_target_still_works():
    visible = {"claim": {"id", "amount"}, "scratch": {"id", "amount"}}
    verdict = decide_write("INSERT INTO scratch (id, amount) SELECT id, amount FROM claim",
                           visible, GrantSet(frozenset(visible), writable=frozenset({"scratch"})),
                           adapter=_OkAdapter(), policy=AccessPolicy(), writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.target == "scratch"


def test_the_reported_tables_are_spelled_as_the_snapshot_spells_them():
    """`ApprovedWrite.tables` is built from `object_key`, like the read path's equivalent.

    Built from `.name` it reported this statement as having touched `claim` -- the same
    qualifier-dropping the target resolution above was fixed for, in the same function. Nothing
    in `src/` reads this field today, which is exactly why it needs a test: an unobserved field
    reverts silently, and the revert surfaces the first time write provenance is consumed.
    """
    visible = {"pg.claim": {"id", "amount"}, "scratch": {"id", "amount"}}
    verdict = decide_write("INSERT INTO scratch (id, amount) SELECT id, amount FROM pg.claim",
                           visible, GrantSet(frozenset(visible), writable=frozenset({"scratch"})),
                           adapter=_OkAdapter(), policy=AccessPolicy(), writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.tables == ["pg.claim", "scratch"]


def test_a_multi_target_delete_is_refused_rather_than_guessed_at():
    """`DELETE s FROM t JOIN s` puts the deleted table in `args['tables']` while `this` is `t`,
    so the resolver authorized `t` and the statement deletes from `s`. DuckDB happens to reject
    the syntax, so the EXPLAIN proof caught it -- a wrong authorization decision saved by a
    downstream parser error is not a control, and the next dialect need not oblige."""
    verdict = _q_write("DELETE s FROM claim JOIN s ON claim.id = s.id WHERE claim.id = 1",
                       {"claim", "s"})
    assert isinstance(verdict, Refusal)
    assert verdict.code is RefusalCode.AMBIGUOUS_WRITE_TARGET


# -- the floors hold against a real database ------------------------------------------------------


def test_no_floor_blocks_an_ordinary_governed_write_end_to_end():
    """One executed control for the whole file: the ordinary shape still runs, and still lands
    only filtered rows. Three refusals prove nothing if the feature stopped working."""
    verdict = _w_write("INSERT INTO scratch (id, amount) SELECT id, amount FROM claim")
    assert isinstance(verdict, ApprovedWrite)
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INT, amount INT, region TEXT)")
    con.execute("INSERT INTO claim VALUES (1,10,'west'),(2,20,'east'),(3,30,'east')")
    con.execute("CREATE TABLE scratch (id INT, amount INT)")
    con.execute(verdict.plan_sql)
    assert con.execute("SELECT id FROM scratch ORDER BY id").fetchall() == [(1,)]


# -- the floors reach production, not just the decider ---------------------------------------------


def test_runtime_write_supplies_the_view_snapshot_to_the_decider():
    """Floor 1 can be satisfied entirely while the gap it measured stays open in production.

    `Runtime.write` is the only production caller of `decide_write`, it holds `self.snapshot`,
    and it passed no views -- the read path builds the same map one line away in `plan_query`
    (`views = {v.object_id: v for v in snapshot.views}`). A parameter nobody passes is not a
    control, so this test drives the write through the Runtime rather than the decider.
    """
    from mnemiq.contract import Column, IdentityContext, Snapshot
    from mnemiq.runtime import Runtime

    snap = Snapshot(
        version="v1", source_id="acme", created_at="t",
        columns=[Column(id="claim_view.id", object_id="claim_view", name="id"),
                 # `claim` must be a known object or `check_views` refuses UNRESOLVABLE_VIEW
                 # -- snapshot incompleteness, not the row-filter floor this test is about.
                 Column(id="claim.id", object_id="claim", name="id"),
                 Column(id="claim.region", object_id="claim", name="region"),
                 Column(id="scratch.id", object_id="scratch", name="id")],
        views=[ViewDefinition(object_id="claim_view", definition="SELECT * FROM claim",
                              dialect="duckdb")],
    )

    class _Authz:
        def grants_for(self, _identity):
            return GrantSet(frozenset({"claim_view", "scratch"}),
                            writable=frozenset({"scratch"}),
                            row_filters={"claim": "region = 'west'"})

    class _Adapter:
        dialect = "duckdb"

        def execute(self, sql):
            return [] if sql.startswith("EXPLAIN") else [(1,)]

    class _WritesEnabled:
        write_enabled = True
        source_id = "acme"

    rt = Runtime(con=None, snapshot=snap, adapter=_Adapter(), agent=None, embedder=None,
                 authz=_Authz(), settings=_WritesEnabled())
    res = rt.write("INSERT INTO scratch (id) SELECT id FROM claim_view",
                   IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"]))
    assert res.approved is False
    # `WriteResult` carries no refusal code, so the assertion has to be on the message that
    # only UNGOVERNED_VIEW produces. `"view" in refusal` was not that: "claim_view" contains
    # "view", so it passed just as happily on UNRESOLVABLE_VIEW.
    assert "cannot apply that filter through a view" in (res.refusal or "")


def test_the_with_floor_is_not_keyed_on_a_sqlglot_arg_name():
    """The floor must survive the arg being spelled differently, because it is.

    `pyproject.toml` declares `sqlglot>=25`; the pinned 30.12.0 calls this arg `with_` and
    25.34.1 calls it `with`. A lookup by name returns None on the other spelling, the floor never
    fires, and the statement is approved with no guard having run -- failing OPEN, silently,
    on a dependency bump rather than on a code change.

    This pops whichever key this sqlglot uses and re-parks under the other one, so the detector
    is asked about a spelling it was not given on either version.
    `views.py` carries the same lesson from the same cause: a whitelist that read `args["from"]`
    where that sqlglot said `from_` enumerated no sources and approved everything.
    """
    import sqlglot

    from mnemiq.sql.decide_write import _has_unscoped_with

    ast = sqlglot.parse_one("WITH x AS (SELECT id FROM claim) "
                            "INSERT INTO scratch (id) SELECT id FROM x", read="duckdb")
    assert _has_unscoped_with(ast), "control: the current spelling must be detected"

    # Popped by whichever spelling this sqlglot uses, and re-parked under the OTHER one. Keyed
    # on a constant, this test raises KeyError on the version it defends against; re-parking
    # under a constant is worse -- on 25.x it would pop and restore the same key, re-running the
    # control above and passing a detector regressed to `args.get("with")`. The defect this test
    # exists to catch, twice, one level up in the test itself.
    popped = "with_" if "with_" in ast.args else "with"
    node = ast.args.pop(popped)
    assert node is not None
    ast.args["with" if popped == "with_" else "with_"] = node
    assert _has_unscoped_with(ast), "the floor must not depend on the arg's NAME"

    inside = sqlglot.parse_one("INSERT INTO scratch (id) WITH x AS (SELECT id FROM claim) "
                               "SELECT id FROM x", read="duckdb")
    assert not _has_unscoped_with(inside), "the governed spelling must not be swept up"


_ENTITLED = "id IN (SELECT id FROM entitlement WHERE who = 'u')"

# The last case is the only one that enters the conjoined-WHERE branch of
# `apply_row_filters_to_write`, and reaching it needs TWO things at once: an UPDATE/DELETE root
# AND a filter naming the write target. The first three filter `claim`, a read SOURCE, so they
# take the derived-table branch. Two earlier versions of this comment each got half of it: the
# first said UPDATE and DELETE exercise the conjoin by being UPDATE and DELETE (the shape alone
# is not enough), the second said the filter's table decides it (not enough either -- an INSERT
# whose TARGET is filtered still takes neither branch, because `apply_row_filters_to_write`
# returns on `not isinstance(ast, (exp.Update, exp.Delete))` before it reads the filter FOR THE
# TARGET, and that exemption is deliberate: an INSERT does not read its target. It has already
# read `row_filters` by then -- `apply_row_and_mask` runs first and consults them to wrap the
# READS, which is what filters `claim` in the insert case below. The early return skips the
# target's own predicate, not row filtering.)
#
# `rendering` and `absent` are asserted so the two branches are told apart in both directions.
# `"entitlement" in plan_sql` alone is satisfied by any construct naming the table, and even
# `") AS claim"` only proves a derived table named `claim` EXISTS, not that the predicate is
# inside it -- a rewrite that wrapped `claim` unfiltered and conjoined ITS predicate onto the
# target's WHERE would filter `scratch`'s rows by `claim`'s policy, the wrong table entirely, and
# still satisfy both. Pinning the predicate's position is what distinguishes them.
_PROVENANCE_SHAPES = [
    ("insert", "INSERT INTO scratch (id, amount) SELECT id, amount FROM claim",
     {"claim": _ENTITLED}, ["claim", "scratch"], f"WHERE {_ENTITLED}) AS claim",
     f"AND {_ENTITLED}"),
    ("update", "UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM claim)",
     {"claim": _ENTITLED}, ["claim", "scratch"], f"WHERE {_ENTITLED}) AS claim",
     f"AND {_ENTITLED}"),
    ("delete", "DELETE FROM scratch WHERE id IN (SELECT id FROM claim)",
     {"claim": _ENTITLED}, ["claim", "scratch"], f"WHERE {_ENTITLED}) AS claim",
     f"AND {_ENTITLED}"),
    ("update_target_filtered", "UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM claim)",
     {"scratch": _ENTITLED}, ["claim", "scratch"], f"AND {_ENTITLED}", ") AS claim"),
]


@pytest.mark.parametrize("label,sql,filters,expected,rendering,absent", _PROVENANCE_SHAPES,
                         ids=[p[0] for p in _PROVENANCE_SHAPES])
def test_write_provenance_does_not_disclose_the_caller_s_own_policy(label, sql, filters, expected,
                                                                    rendering, absent):
    """`tables` is read from the statement as ASKED, not from the rewritten tree.

    M28 on the write path. A row filter may reach through another table -- the only way to express
    child-table tenancy -- and the injected predicate names whatever the POLICY names. Reporting
    the rewrite therefore tells the caller which tables their policy consults, and an entitlements
    table is both the standard shape and one no caller is ever granted.

    Measured before the fix: this returned `['claim', 'entitlement', 'scratch']` to a caller who
    holds no grant on `entitlement` and cannot see it in their schema. The read path has computed
    this above its rewrite since M28; the write path computed it below.
    """
    visible = {"claim": {"id", "amount"}, "scratch": {"id", "amount"}}
    policy = AccessPolicy(
        row_filters=filters,
        policy_schema={**{k: set(v) for k, v in visible.items()},
                       "entitlement": {"id", "who"}},
    )
    verdict = decide_write(sql, visible,
                           GrantSet(frozenset(visible), writable=frozenset({"scratch"})),
                           adapter=_OkAdapter(), policy=policy, writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.tables == expected
    assert "entitlement" not in verdict.tables
    # The filter still binds -- provenance is narrowed, not the governance -- and it binds
    # through the branch this case exists to reach, not merely somewhere in the statement.
    assert "entitlement" in verdict.plan_sql
    assert rendering in verdict.plan_sql
    assert absent not in verdict.plan_sql

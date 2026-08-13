"""Six defects a second model found in one day's governance work, all reproduced first.

Worth stating why they existed together. M31 replaced a flat set of CTE names with a
scope-aware resolver, but only where it decided *which nodes are base tables* -- the
alias->table maps in the two consumers stayed global across scopes, and the leak moved there.
M27 then inlined views, which removes the very object a view-keyed policy is attached to. Each
fix was verified against the case that motivated it and not against the case next to it.
"""

from __future__ import annotations

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import ViewDefinition
from mnemiq.contract.semantic import Column, Snapshot
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy, build_access_policy
from mnemiq.sql.rls import _validate_filter
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Approved, Refusal

D = "postgres"


def plan(sql, visible, policy, views=None):
    return decide(sql, visible, dialect=D, target=D, policy=policy, views=views or {})


# --- F1: one alias map for every scope ------------------------------------------


TWO = {"inner_t": {"secret", "id"}, "outer_t": {"secret", "id"}}


def test_a_shadowed_alias_in_another_scope_cannot_reach_a_denied_column():
    """`q` means a different table in each leg. The alias map kept one entry, so the denied
    leg was resolved against the permitted table and returned."""
    v = plan("SELECT q.secret FROM inner_t q UNION ALL SELECT q.secret FROM outer_t q",
             TWO, AccessPolicy(denied={("inner_t", "secret")}))
    assert isinstance(v, Refusal), f"a denied column was returned: {getattr(v, 'plan_sql', v)}"


def test_the_same_shadowing_through_a_correlated_subquery():
    v = plan("SELECT o.id FROM outer_t o WHERE EXISTS "
             "(SELECT 1 FROM inner_t o WHERE o.secret = 'x')",
             TWO, AccessPolicy(denied={("inner_t", "secret")}))
    assert isinstance(v, Refusal), f"a denied column was reachable: {getattr(v, 'plan_sql', v)}"


def test_an_unshadowed_query_is_still_allowed():
    v = plan("SELECT o.id FROM outer_t o", TWO, AccessPolicy(denied={("inner_t", "secret")}))
    assert isinstance(v, Approved), v


def test_the_single_leg_control_still_refuses():
    v = plan("SELECT q.secret FROM inner_t q", TWO, AccessPolicy(denied={("inner_t", "secret")}))
    assert isinstance(v, Refusal)


# --- F2 / F3: inlining removes the object the policy is keyed to ------------------


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1", source_id="s", created_at="2026-01-01T00:00:00Z",
        columns=[
            Column(id="customer.store_id", object_id="customer", name="store_id"),
            Column(id="customer.id", object_id="customer", name="id"),
            Column(id="customer.email", object_id="customer", name="email", pii_level="pii"),
            Column(id="customer_list.id", object_id="customer_list", name="id"),
            Column(id="customer_list.sid", object_id="customer_list", name="sid"),
            Column(id="customer_list.email", object_id="customer_list", name="email",
                   pii_level="pii"),
        ],
        views=[ViewDefinition(
            object_id="customer_list", dialect=D,
            definition="SELECT id, store_id AS sid, email FROM customer")],
    )


def _through_the_real_builder(grants: GrantSet):
    """The production path. The M27 tests hand-built an AccessPolicy carrying a base-keyed
    filter for a caller with no base grant -- a policy `build_access_policy` cannot produce,
    so they validated a configuration that does not occur."""
    snap = _snapshot()
    return (visible_schema(snap, grants), build_access_policy(snap, grants),
            {v.object_id: v for v in snap.views})


def test_a_view_only_grant_is_filtered_when_the_filter_names_the_base():
    grants = GrantSet(frozenset({"customer_list"}), row_filters={"customer": "store_id = 1"})
    visible, policy, views = _through_the_real_builder(grants)
    v = plan("SELECT id FROM customer_list", visible, policy, views)
    assert isinstance(v, Approved), v
    assert "store_id = 1" in v.plan_sql, "the base filter never reached the inlined body"


def test_a_view_only_grant_is_filtered_when_the_filter_names_the_view():
    """The filter's column is in the view's OUTPUT, so it can be applied there -- but inlining
    removed the node it was keyed to before the rewrite ran."""
    grants = GrantSet(frozenset({"customer_list"}), row_filters={"customer_list": "sid = 1"})
    visible, policy, views = _through_the_real_builder(grants)
    v = plan("SELECT id FROM customer_list", visible, policy, views)
    assert isinstance(v, Approved), v
    assert "sid = 1" in v.plan_sql, "the view-keyed filter was silently unused"


def test_a_mask_keyed_to_a_view_column_survives_inlining():
    grants = GrantSet(frozenset({"customer_list"}), pii_mask=frozenset({"pii"}))
    visible, policy, views = _through_the_real_builder(grants)
    assert ("customer_list", "email") in policy.masked  # the real builder keys it to the view
    v = plan("SELECT email FROM customer_list", visible, policy, views)
    assert isinstance(v, Approved), v
    assert "NULL AS email" in v.plan_sql, "inlining dropped the mask"


def test_the_caller_still_only_learns_they_read_the_view():
    grants = GrantSet(frozenset({"customer_list"}), row_filters={"customer": "store_id = 1"})
    visible, policy, views = _through_the_real_builder(grants)
    v = plan("SELECT id FROM customer_list", visible, policy, views)
    assert v.tables == ["customer_list"]


# --- F4: qualification dropped when validating a subquery ------------------------


def test_a_differently_qualified_table_is_not_accepted_for_a_bare_key():
    got = _validate_filter(
        "customer_id IN (SELECT customer_id FROM private.entitlement WHERE principal = 'x')",
        {"customer_id"}, D, {"entitlement": {"customer_id", "principal"}})
    assert got is None, "private.entitlement was accepted against the key 'entitlement'"


def test_a_qualified_key_accepts_its_own_qualified_reference():
    """The other direction: federation keys objects `catalog.table`, and comparing that to a
    bare name rejected the only spelling that works."""
    got = _validate_filter(
        "customer_id IN (SELECT customer_id FROM pg.entitlement WHERE principal = 'x')",
        {"customer_id"}, D, {"pg.entitlement": {"customer_id", "principal"}})
    assert got is not None, "a federated policy could not name its own table"


# --- F5: a correlated reference is not an inner column ---------------------------


def test_a_correlated_reference_to_the_filtered_table_is_allowed():
    got = _validate_filter(
        "EXISTS (SELECT 1 FROM entitlement e WHERE e.customer_id = payer_id)",
        {"payer_id"}, D, {"entitlement": {"customer_id"}})
    assert got is not None, "the EXISTS spelling of a filter the IN spelling already allows"


def test_an_inner_column_belonging_to_nothing_is_still_refused():
    got = _validate_filter(
        "EXISTS (SELECT 1 FROM entitlement e WHERE e.nonexistent = payer_id)",
        {"payer_id"}, D, {"entitlement": {"customer_id"}})
    assert got is None


# --- F6: the trend has no ordering key -------------------------------------------


def test_a_recorded_run_is_stamped_so_the_latest_can_be_found(tmp_path):
    """`ORDER BY run_at DESC LIMIT 1` over rows that all carry "" returns an arbitrary one."""
    from mnemiq.eval.report import Report
    from mnemiq.eval.trend import last_run, record_run

    path = str(tmp_path / "trend.json")
    record_run(None, "acme", Report(total=10, correct=9, wrong=1), path=path)
    rec = last_run(None, "acme", path=path)
    assert rec.run_at, "a run with no timestamp cannot be ordered against another"


# --- second pass: what the fix itself broke or left half-done --------------------


def test_a_caller_no_policy_applies_to_is_not_put_through_the_rewrite():
    """Keeping dispositions for ungranted tables (F3's fix) made `policy.empty` false for
    everyone, so a caller with nothing enforceable started being inlined -- and an unparseable
    view then refused a query that used to work."""
    snap = Snapshot(
        version="v1", source_id="s", created_at="t",
        columns=[Column(id="v.id", object_id="v", name="id"),
                 Column(id="secret.ssn", object_id="secret", name="ssn", pii_level="pii")],
        views=[ViewDefinition(object_id="v", dialect=D, definition="SELECT FROM WHERE ((")],
    )
    g = GrantSet(frozenset({"v"}))          # `secret` is nothing to do with this caller
    pol, vis = build_access_policy(snap, g), visible_schema(snap, g)
    v = plan("SELECT id FROM v", vis, pol, {x.object_id: x for x in snap.views})
    assert isinstance(v, Approved), f"an unrelated PII column refused this caller: {v}"


def test_a_policys_own_subquery_is_not_inlined_and_not_filtered():
    """The two-visibility rule, broken by the fix that restored it elsewhere. A view named
    INSIDE a row filter belongs to the policy, so it reads with the policy author's reach --
    inlining it and then applying the caller's filter to its bases is the caller's reach."""
    snap = Snapshot(
        version="v1", source_id="s", created_at="t",
        columns=[Column(id="payment.payment_id", object_id="payment", name="payment_id"),
                 Column(id="payment.customer_id", object_id="payment", name="customer_id"),
                 Column(id="customer.customer_id", object_id="customer", name="customer_id"),
                 Column(id="customer.store_id", object_id="customer", name="store_id"),
                 Column(id="customer_view.customer_id", object_id="customer_view",
                        name="customer_id")],
        views=[ViewDefinition(object_id="customer_view", dialect=D,
                              definition="SELECT customer_id FROM customer")],
    )
    # The view is granted TOO, so `customer` is reachable and its filter survives the policy
    # builder. Without that this passes for the wrong reason -- the filter is dropped and there
    # is nothing left to leak, which is the unreachable-fixture trap that hid the original bug.
    g = GrantSet(frozenset({"payment", "customer_view"}), row_filters={
        "payment": "customer_id IN (SELECT customer_id FROM customer_view)",
        "customer": "store_id = 1"})
    pol, vis = build_access_policy(snap, g), visible_schema(snap, g)
    v = plan("SELECT payment_id FROM payment", vis, pol,
             {x.object_id: x for x in snap.views})
    assert isinstance(v, Approved), v
    assert "customer" in pol.row_filters, "the fixture no longer exercises the interaction"
    assert "store_id = 1" not in v.plan_sql, (
        "the caller's filter reached inside the policy's own subquery"
    )


def test_a_three_part_name_cannot_borrow_another_catalogs_key():
    """`object_key` keeps `db.table` and drops the catalog, so two catalogs collide."""
    got = _validate_filter(
        "customer_id IN (SELECT customer_id FROM other.public.entitlement)",
        {"customer_id"}, D, {"public.entitlement": {"customer_id"}})
    assert got is None, "a filter reached a table in a different catalog"


def test_a_qualified_inner_column_must_belong_to_the_table_it_names():
    """Correlation was allowed by unioning the outer table's column NAMES into the inner
    check, so an inner column qualified to a table that has no such column passed for
    sharing a name with one the outer table does have."""
    got = _validate_filter(
        "EXISTS (SELECT 1 FROM entitlement e WHERE e.payer_id = payer_id)",
        {"payer_id"}, D, {"entitlement": {"customer_id"}})
    assert got is None, "e.payer_id was accepted though entitlement has no payer_id"


def test_but_a_genuinely_correlated_unqualified_reference_still_works():
    got = _validate_filter(
        "EXISTS (SELECT 1 FROM entitlement e WHERE e.customer_id = payer_id)",
        {"payer_id"}, D, {"entitlement": {"customer_id"}})
    assert got is not None

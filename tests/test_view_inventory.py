"""M52: a view inventory that could not be READ is not one that is empty.

The producer has always known the difference -- `enrichment/pipeline.py` records `discover:views`
with `status='failed'` under a comment saying an empty list "must be treated as 'cannot reason
about', never as 'there are none'". Nothing read it, so a source that would not answer produced
the same `{}` as a source with no views, and the M27 view floor silently stopped firing.

Measured before this: with `row_filters={'claim'}` and `claim_v` defined over `claim`,
`SELECT id, amount FROM claim_v` was REFUSED `ungoverned_view` when discovery succeeded and
APPROVED when it failed -- from the identical call.

Tenth instance in this codebase of an absence and a failure sharing one value, after M2, M6, M34,
M40/M41, M45, `column_tables`, the provenance flag, M48 and M49.
"""
from __future__ import annotations

import sqlglot

from mnemiq.contract.semantic import Job, Snapshot, ViewDefinition
from mnemiq.sql.verdict import RefusalCode
from mnemiq.sql.views import VIEWS_UNAVAILABLE, ViewInventory, check_views, inventory_for

SQL = sqlglot.parse_one("SELECT id, amount FROM claim_v", read="duckdb")
FILTERED = {"claim"}
VIEW = ViewDefinition(object_id="claim_v", definition="SELECT id, region, amount FROM claim",
                      dialect="duckdb")


def _snapshot(views, jobs):
    return Snapshot(version="v", source_id="s", created_at="t", columns=[], views=views, jobs=jobs)


def _done(kind="done"):
    return [Job(id="discover:views", source_id="s", kind="discover", status=kind)]


# -- the collapse itself ------------------------------------------------------------------------

def test_an_unreadable_inventory_is_refused_rather_than_read_as_empty():
    r = check_views(SQL, VIEWS_UNAVAILABLE, FILTERED)
    assert r is not None and r.code is RefusalCode.VIEW_INVENTORY_UNAVAILABLE


def test_a_source_that_answered_and_has_no_views_is_the_control():
    """The value that must NOT be refused, and the reason this is a real distinction rather than
    a blanket refusal: an empty inventory from a source that ANSWERED means what it says."""
    assert check_views(SQL, ViewInventory({}), FILTERED) is None


def test_the_governed_view_is_still_refused_when_discovery_worked():
    """The other control: the floor this protects must still fire on its own case."""
    r = check_views(SQL, ViewInventory({"claim_v": VIEW}), FILTERED)
    assert r is not None and r.code is RefusalCode.UNGOVERNED_VIEW


def test_an_unreadable_inventory_costs_nothing_when_no_row_filters_exist():
    """The narrowing that keeps this from refusing every query on every source that hiccups:
    with no row filters there is nothing a view could carry a caller around."""
    assert check_views(SQL, VIEWS_UNAVAILABLE, set()) is None


# -- what a snapshot resolves to ----------------------------------------------------------------

def test_a_failed_discovery_job_makes_the_inventory_unavailable():
    inv = inventory_for(_snapshot([], _done("failed")))
    assert inv.available is False


def test_a_done_discovery_job_with_no_views_is_available():
    inv = inventory_for(_snapshot([], _done()))
    assert inv.available is True and dict(inv) == {}


def test_no_snapshot_at_all_is_unavailable_not_empty():
    """An engine holding no schema cannot assert a source has no views."""
    assert inventory_for(None).available is False


def test_a_snapshot_with_no_discovery_job_is_treated_as_available():
    """The documented residual, pinned so the choice is visible rather than implicit. The producer
    always emits the job, so its absence means a hand-built snapshot, not a failed read."""
    assert inventory_for(_snapshot([], [])).available is True


def test_the_inventory_survives_the_falsy_empty_dict_trap():
    """`views or {}` in either decider would drop the flag, because an unavailable inventory is an
    EMPTY dict and therefore falsy -- M45's exact shape, one field over. Pinned because the bug
    would be invisible: the refusal simply stops happening."""
    assert not VIEWS_UNAVAILABLE                      # falsy, which is the trap
    assert (VIEWS_UNAVAILABLE or {}).__class__ is dict   # what the old spelling yielded
    assert getattr({} if VIEWS_UNAVAILABLE is None else VIEWS_UNAVAILABLE, "available") is False


# -- through the deciders, which is where the falsy trap would actually bite --------------------

def test_the_read_decider_refuses_an_unreadable_inventory():
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import AccessPolicy
    v = decide("SELECT id, amount FROM claim_v", {"claim_v": {"id", "amount"}},
               policy=AccessPolicy(row_filters={"claim": "region = 'west'"}),
               views=VIEWS_UNAVAILABLE)
    assert getattr(v, "code", None) is RefusalCode.VIEW_INVENTORY_UNAVAILABLE, v


def test_the_write_decider_refuses_it_identically():
    """M46 is what happens when the two doors disagree about a view, so both are asserted."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write
    from mnemiq.sql.policy import AccessPolicy
    visible = {"claim_v": {"id", "amount"}, "scratch": {"id", "amount"}}
    v = decide_write("INSERT INTO scratch SELECT id, amount FROM claim_v", visible,
                     GrantSet(objects=frozenset(visible), writable=frozenset({"scratch"})),
                     adapter=None, dialect="duckdb",
                     policy=AccessPolicy(row_filters={"claim": "region = 'west'"}),
                     views=VIEWS_UNAVAILABLE, writes_enabled=True)
    assert getattr(v, "code", None) is RefusalCode.VIEW_INVENTORY_UNAVAILABLE, v


# -- the sibling instance upstream, in the policy builder ---------------------------------------
#
# The review gate BLOCKED the first version of this change over exactly this. `_reachable` built
# its view-body map straight from `snapshot.views`, so a failed discovery made it drop the base
# table's row filter out of `policy.row_filters` -- and `check_views` early-outs on
# `if not filtered` BEFORE it can refuse an unavailable inventory. The guard added here was
# bypassed by the very failure it exists for, one file upstream.

from mnemiq.authz.grants import GrantSet
from mnemiq.contract.semantic import Column
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import build_access_policy


def _col(obj, name):
    return Column(id=f"{obj}.{name}", object_id=obj, name=name, data_type="text")


def _view_snapshot(status):
    return Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[_col("claim", "id"), _col("claim", "region"), _col("claim_v", "id")],
        views=[] if status == "failed" else [VIEW_OVER_CLAIM],
        jobs=[Job(id="discover:views", source_id="s", kind="discover", status=status)],
    )


VIEW_OVER_CLAIM = ViewDefinition(object_id="claim_v", definition="SELECT id, region FROM claim",
                                 dialect="duckdb")
VIEW_ONLY_GRANT = GrantSet(objects=frozenset({"claim_v"}),
                           row_filters={"claim": "region = 'west'"})


def test_a_failed_discovery_does_not_narrow_the_row_filter_out_of_the_policy():
    """A view-only grant is what `_reachable`'s own docstring calls the standard shape. With the
    inventory unavailable the filter must SURVIVE, or the guard downstream never gets to run."""
    policy = build_access_policy(_view_snapshot("failed"), VIEW_ONLY_GRANT)
    assert dict(policy.row_filters) == {"claim": "region = 'west'"}


def test_the_view_only_read_is_refused_rather_than_returned_unfiltered():
    """End to end, which is what the BLOCKER measured: this returned Approved and every row of
    `claim`, unfiltered, through a view the caller was granted."""
    snapshot = _view_snapshot("failed")
    v = decide("SELECT id FROM claim_v", {"claim_v": {"id"}},
               policy=build_access_policy(snapshot, VIEW_ONLY_GRANT),
               views=inventory_for(snapshot))
    assert getattr(v, "code", None) is RefusalCode.VIEW_INVENTORY_UNAVAILABLE, v


def test_the_same_read_with_discovery_working_is_the_control():
    """Refused too, but for the reason the M27 floor exists -- so the test above is not passing
    on a decider that has simply started refusing everything."""
    snapshot = _view_snapshot("done")
    v = decide("SELECT id FROM claim_v", {"claim_v": {"id"}},
               policy=build_access_policy(snapshot, VIEW_ONLY_GRANT),
               views=inventory_for(snapshot))
    assert getattr(v, "code", None) is RefusalCode.UNGOVERNED_VIEW, v


def test_a_caller_with_no_row_filters_is_unaffected_by_an_unreadable_inventory():
    """The blast-radius control. Widening `reachable` must not make a policy-free caller
    suddenly refusable: with nothing filtered there is nothing to reach around."""
    snapshot = _view_snapshot("failed")
    grants = GrantSet(objects=frozenset({"claim_v"}))
    v = decide("SELECT id FROM claim_v", {"claim_v": {"id"}},
               policy=build_access_policy(snapshot, grants), views=inventory_for(snapshot))
    assert not hasattr(v, "code") or v.code is None, v


# -- federation, where the guard was inert entirely ---------------------------------------------

def test_a_federated_snapshot_carries_its_sources_views_and_jobs():
    """`merge_snapshots` passed neither `views=` nor `jobs=`, so every federated deployment had
    an empty view list -- the M27 floor could not fire there at all -- and `inventory_for` then
    read that emptiness as "no views" rather than "nobody asked"."""
    from mnemiq.config import SourceSpec
    from mnemiq.semantic.federation import merge_snapshots
    snap = Snapshot(version="v", source_id="s", created_at="t", columns=[],
                    views=[VIEW_OVER_CLAIM], jobs=_done())
    merged = merge_snapshots([(SourceSpec(id="s", kind="postgres", target="d", catalog="pg", schema="public"), snap)])
    assert [v.object_id for v in merged.views] == ["pg.claim_v"]
    assert [j.id for j in merged.jobs] == ["discover:views"]
    assert inventory_for(merged).available is True


def test_one_source_that_could_not_report_views_makes_the_union_unavailable():
    """The union is only as knowable as its least knowable member."""
    from mnemiq.config import SourceSpec
    from mnemiq.semantic.federation import merge_snapshots
    ok = Snapshot(version="v", source_id="a", created_at="t", columns=[],
                  views=[VIEW_OVER_CLAIM], jobs=_done())
    bad = Snapshot(version="v", source_id="b", created_at="t", columns=[], views=[],
                   jobs=_done("failed"))
    merged = merge_snapshots([
        (SourceSpec(id="a", kind="postgres", target="d", catalog="a", schema="public"), ok),
        (SourceSpec(id="b", kind="postgres", target="d", catalog="b", schema="public"), bad),
    ])
    assert inventory_for(merged).available is False


def test_an_unreadable_inventory_defers_before_the_first_model_call():
    """The refusal fires ahead of the AST walk, so every attempt would propose, be refused
    identically, and feed the same unfixable message back. Knowable before spending anything."""
    from mnemiq.contract import DeferralReason
    from mnemiq.generate.plan_query import plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    class ExplodingGenerator:
        def propose(self, packet, feedback):
            raise AssertionError("the model was asked despite a knowable outage")

    snapshot = _view_snapshot("failed")
    result = plan_query(
        packet=ContextPacket(question="how many claims?",
                             cards=[RetrievedCard(object_id="claim_v", card="claim_v", score=1.0)],
                             grant_fingerprint="f", enrichment_version=None),
        snapshot=snapshot, grants=VIEW_ONLY_GRANT, generator=ExplodingGenerator(),
    )
    assert result.code is DeferralReason.POLICY_UNAVAILABLE, result


# -- the federated spelling gap, found by Codex's stop-time review -------------------------------
#
# `merge_snapshots` qualifies a view's object_id (`claim_v` -> `pg.claim_v`) and leaves its BODY
# exactly as the source wrote it (`SELECT id, region FROM claim`). `_reachable` parses that body,
# reaches `claim`, never reaches `pg.claim`, and drops the federated filter from the policy as
# irrelevant -- so `check_views` sees an empty `filtered` and its early-out approves the read.
#
# Third time a narrowing in `_reachable` has silently disarmed the guard downstream of it, after
# the unparseable-body case it already handles and the unavailable-inventory case above.

def _federated():
    from mnemiq.config import SourceSpec
    from mnemiq.semantic.federation import merge_snapshots
    snap = Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[_col("claim", "id"), _col("claim", "region"), _col("claim_v", "id")],
        views=[VIEW_OVER_CLAIM], jobs=_done())
    spec = SourceSpec(id="s", kind="postgres", target="d", catalog="pg", schema="public")
    return merge_snapshots([(spec, snap)])


def test_a_federated_view_only_grant_does_not_bypass_the_row_filter():
    """Measured before the fix: row_filters={} and APPROVED -- every row of `claim` returned
    unfiltered through a granted view, while the identical single-source shapes refused."""
    fed = _federated()
    grants = GrantSet(objects=frozenset({"pg.claim_v"}),
                      row_filters={"pg.claim": "region = 'west'"})
    policy = build_access_policy(fed, grants)
    assert dict(policy.row_filters) == {"pg.claim": "region = 'west'"}
    v = decide("SELECT id FROM pg.claim_v", {"pg.claim_v": {"id"}},
               policy=policy, views=inventory_for(fed))
    assert getattr(v, "code", None) is RefusalCode.UNGOVERNED_VIEW, v


def test_a_federated_caller_with_no_row_filters_is_still_approved():
    """The blast-radius control: widening the spellings `_reachable` tries must not make a
    policy-free federated caller suddenly refusable."""
    fed = _federated()
    v = decide("SELECT id FROM pg.claim_v", {"pg.claim_v": {"id"}},
               policy=build_access_policy(fed, GrantSet(objects=frozenset({"pg.claim_v"}))),
               views=inventory_for(fed))
    assert not isinstance(v, type(None)) and getattr(v, "code", None) is None, v

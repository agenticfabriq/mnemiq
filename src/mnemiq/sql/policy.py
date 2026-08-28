from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot


@dataclass
class AccessPolicy:
    row_filters: dict[str, str] = field(default_factory=dict)
    denied: set[tuple[str, str]] = field(default_factory=set)
    masked: set[tuple[str, str]] = field(default_factory=set)
    # The POLICY AUTHOR's visibility: every table in the snapshot, not the caller's slice of
    # it. A row filter that reaches through another table -- the only way to express
    # child-table tenancy, since `payment` has no `store_id` -- is resolved against this and
    # never against the caller's scope. The policy is what *defines* the caller's boundary, so
    # resolving it through that boundary is circular; and the standard shape, an entitlements
    # table, is one no caller is ever granted (M28). Empty means no subquery filter can be
    # validated, and an unvalidatable filter is refused rather than guessed at.
    policy_schema: dict[str, set[str]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.row_filters and not self.denied and not self.masked


def _reachable(snapshot: Snapshot, grants: GrantSet) -> set[str]:
    """Objects this caller's query can end up reading: what they are granted, plus the bases
    behind any granted view, transitively.

    Not the same as "granted". A view-only grant carries no grant on the tables behind it, and
    after inlining those are what the query reads -- so scoping dispositions to grants alone
    deleted the policy exactly where it was needed. Scoping to *everything* instead was the
    over-correction: it made `policy.empty` false for every caller in a deployment that had any
    policy at all, so a caller with nothing enforceable was put through a rewrite that could
    refuse them over a view they had every right to read.
    """
    import sqlglot
    from sqlglot import exp

    from mnemiq.sql.qualify import object_key

    from mnemiq.sql.views import inventory_for

    inventory = inventory_for(snapshot)
    every = {c.object_id for c in snapshot.columns}
    reachable, frontier = set(grants.objects), list(grants.objects)
    if not inventory.available:
        # Same reason as the unparseable body below, and the same answer. We cannot see what any
        # view reads, so we cannot say the policy is irrelevant to this caller.
        #
        # This is load-bearing for M52 and was found by the review gate blocking that commit.
        # Narrowing here empties `row_filters`, and `check_views` early-outs on `if not filtered`
        # BEFORE it can refuse an unavailable inventory -- so the guard added for M52 was bypassed
        # by the very failure it exists for, one file upstream. Measured: a view-only grant plus a
        # failed `discover:views` job returned Approved on `SELECT ... FROM claim_v`, every row of
        # the filtered base table, unfiltered. The docstring above calls a view-only grant the
        # standard shape.
        return every | reachable
    bodies = dict(inventory)
    while frontier:
        view = bodies.get(frontier.pop())
        if view is None:
            continue
        try:
            parsed = sqlglot.parse_one(view.definition, read=view.dialect)
        except Exception:
            # We cannot see what this view reads, so we cannot say the policy is irrelevant.
            # Returning what we have left the policy EMPTY, which skipped the rewrite entirely
            # and approved the view unfiltered -- `check_views` never got the chance to refuse
            # it. Everything is reachable instead, so the policy stays live and the floor runs.
            return every | reachable
        for table in parsed.find_all(exp.Table):
            # `object_key`, the same spelling the inliner resolves with. A bare `.name` here
            # never reached `pg.base`, so its filter was dropped and the view read unfiltered.
            # Both spellings, for the same reason the floor matches both: a body may write
            # `public.base` where the snapshot's object-id is `base`, and reaching neither
            # dropped the filter before anything could apply it.
            for key in {object_key(table), table.name}:
                if key and key not in reachable:
                    reachable.add(key)
                    frontier.append(key)
    return reachable


def build_access_policy(snapshot: Snapshot, grants: GrantSet) -> AccessPolicy:
    """Resolve per-column dispositions from pii_level + clearance, and readable row filters."""
    reachable = _reachable(snapshot, grants)
    denied: set[tuple[str, str]] = set()
    masked: set[tuple[str, str]] = set()
    for c in snapshot.columns:
        if c.object_id not in reachable:
            continue  # not reachable even through a view -> nothing here can ever apply
        level = c.pii_level
        if not level or level == "none" or level in grants.pii_clearance:
            continue  # raw
        if level in grants.pii_mask:
            masked.add((c.object_id, c.name))
        else:
            denied.add((c.object_id, c.name))
    # Reachable, not granted -- same reasoning as the dispositions above.
    row_filters = {t: f for t, f in grants.row_filters.items() if t in reachable}
    policy_schema: dict[str, set[str]] = {}
    for c in snapshot.columns:
        policy_schema.setdefault(c.object_id, set()).add(c.name)
    return AccessPolicy(
        row_filters=row_filters, denied=denied, masked=masked, policy_schema=policy_schema
    )

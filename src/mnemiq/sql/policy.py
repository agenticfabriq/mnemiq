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

    bodies = {v.object_id: v for v in snapshot.views}
    reachable, frontier = set(grants.objects), list(grants.objects)
    while frontier:
        view = bodies.get(frontier.pop())
        if view is None:
            continue
        try:
            parsed = sqlglot.parse_one(view.definition, read=view.dialect)
        except Exception:
            continue  # unresolvable: `inline_views` refuses it, and refusing needs no policy
        for table in parsed.find_all(exp.Table):
            if table.name not in reachable:
                reachable.add(table.name)
                frontier.append(table.name)
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

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


def build_access_policy(snapshot: Snapshot, grants: GrantSet) -> AccessPolicy:
    """Resolve per-column dispositions from pii_level + clearance, and readable row filters."""
    denied: set[tuple[str, str]] = set()
    masked: set[tuple[str, str]] = set()
    for c in snapshot.columns:
        # Deliberately NOT skipped when the object is ungranted. A caller granted only a view
        # never has a grant on the tables behind it, and after inlining those tables are what
        # the query reads -- so dropping their dispositions here silently unmasked them. The
        # entries are inert for a table the query cannot reach: `check_access` refuses a direct
        # reference before CLS runs, and the rewrite only wraps nodes that are present.
        level = c.pii_level
        if not level or level == "none" or level in grants.pii_clearance:
            continue  # raw
        if level in grants.pii_mask:
            masked.add((c.object_id, c.name))
        else:
            denied.add((c.object_id, c.name))
    # Same reasoning as the dispositions above: a view-only grant carries no grant on the base
    # the filter names, and requiring one made the filter vanish exactly when it was needed.
    # A filter for a table the query never reaches is never applied.
    row_filters = dict(grants.row_filters)
    policy_schema: dict[str, set[str]] = {}
    for c in snapshot.columns:
        policy_schema.setdefault(c.object_id, set()).add(c.name)
    return AccessPolicy(
        row_filters=row_filters, denied=denied, masked=masked, policy_schema=policy_schema
    )

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
        if not grants.allows(c.object_id):
            continue  # not readable -> check_access handles it, not CLS
        level = c.pii_level
        if not level or level == "none" or level in grants.pii_clearance:
            continue  # raw
        if level in grants.pii_mask:
            masked.add((c.object_id, c.name))
        else:
            denied.add((c.object_id, c.name))
    row_filters = {t: f for t, f in grants.row_filters.items() if grants.allows(t)}
    policy_schema: dict[str, set[str]] = {}
    for c in snapshot.columns:
        policy_schema.setdefault(c.object_id, set()).add(c.name)
    return AccessPolicy(
        row_filters=row_filters, denied=denied, masked=masked, policy_schema=policy_schema
    )

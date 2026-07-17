from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot


@dataclass
class AccessPolicy:
    row_filters: dict[str, str] = field(default_factory=dict)
    denied: set[tuple[str, str]] = field(default_factory=set)
    masked: set[tuple[str, str]] = field(default_factory=set)

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
    return AccessPolicy(row_filters=row_filters, denied=denied, masked=masked)

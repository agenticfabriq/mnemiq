from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

from mnemiq.contract import IdentityContext


@dataclass(frozen=True)
class GrantSet:
    objects: frozenset[str]
    writable: frozenset[str] = frozenset()  # a writable object is always readable (in objects)
    # compare=False keeps GrantSet hashable despite the dict; equality is NOT the auth boundary
    # (the fingerprint is), so excluding the filters from __eq__/__hash__ is harmless.
    row_filters: dict[str, str] = field(default_factory=dict, compare=False)
    pii_clearance: frozenset[str] = frozenset()  # pii_levels seen RAW
    pii_mask: frozenset[str] = frozenset()  # pii_levels seen MASKED (else denied)

    def allows(self, object_id: str) -> bool:
        return object_id in self.objects

    def allows_write(self, object_id: str) -> bool:
        return object_id in self.writable

    @property
    def fingerprint(self) -> str:
        """Identifies the *access*, not the principal -- and it is the authorization boundary,
        so it hashes the FULL policy. Two identities with identical readable tables but different
        row filters / PII clearance / mask get different cache keys; a result computed under one
        policy can never be served under a narrower one (the Plan 07 cache key).
        """
        parts = [
            "R:" + "|".join(sorted(self.objects)),
            "F:" + "|".join(f"{k}={v}" for k, v in sorted(self.row_filters.items())),
            "C:" + "|".join(sorted(self.pii_clearance)),
            "M:" + "|".join(sorted(self.pii_mask)),
        ]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


EMPTY = GrantSet(frozenset())


class AuthzProvider(Protocol):
    def grants_for(self, identity: IdentityContext) -> GrantSet: ...


class DenyAll:
    """The default when nothing is configured. Absence of policy is not permission."""

    def grants_for(self, identity: IdentityContext) -> GrantSet:
        return EMPTY


class FileAuthzProvider:
    """Grants from a JSON policy: {"roles": {"analyst": ["claim", ...]}}.

    Stands in for the control plane, which will supply grants over its own interface. The
    engine never asks *how* the grants were decided -- only what they are. Any failure to
    read a policy denies everything: an authorization bug must fail loudly (no data) rather
    than silently (too much data).
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def grants_for(self, identity: IdentityContext) -> GrantSet:
        try:
            with open(self._path) as fh:
                policy = json.load(fh)
            roles = policy["roles"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return EMPTY

        objects: set[str] = set()
        writable: set[str] = set()
        row_filters: dict[str, str] = {}
        clearance: set[str] = set()
        mask: set[str] = set()
        for principal_role in [*identity.roles, *identity.groups]:
            granted = roles.get(principal_role)
            if isinstance(granted, list):  # list form: read-only (backward compatible)
                objects.update(str(o) for o in granted)
            elif isinstance(granted, dict):  # dict form: explicit read + write + RLS/CLS
                objects.update(str(o) for o in granted.get("read", []))
                writable.update(str(o) for o in granted.get("write", []))
                clearance.update(str(x) for x in granted.get("pii_clearance", []))
                mask.update(str(x) for x in granted.get("pii_mask", []))
                for table, filt in (granted.get("row_filters") or {}).items():
                    if table in row_filters:  # more roles = more visible rows: OR-combine
                        row_filters[table] = f"({row_filters[table]}) OR ({filt})"
                    else:
                        row_filters[table] = str(filt)
        objects |= writable  # a writable table is always readable (the decider requires it)
        return GrantSet(frozenset(objects), frozenset(writable), row_filters,
                        frozenset(clearance), frozenset(mask))

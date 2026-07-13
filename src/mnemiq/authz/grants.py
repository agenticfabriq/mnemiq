from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from mnemiq.contract import IdentityContext


@dataclass(frozen=True)
class GrantSet:
    objects: frozenset[str]

    def allows(self, object_id: str) -> bool:
        return object_id in self.objects

    @property
    def fingerprint(self) -> str:
        """Identifies the *access*, not the principal.

        Two identities with identical grants share cache entries; a result computed under
        broader access can never be served to narrower access (the Plan 07 cache key).
        """
        joined = "|".join(sorted(self.objects))
        return hashlib.sha256(joined.encode()).hexdigest()[:12]


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
        for principal_role in [*identity.roles, *identity.groups]:
            granted = roles.get(principal_role)
            if isinstance(granted, list):
                objects.update(str(o) for o in granted)
        return GrantSet(frozenset(objects))

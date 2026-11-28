from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from mnemiq.contract import IdentityContext
from mnemiq.sql.qualify import names_one_object

logger = logging.getLogger(__name__)


def classified_levels(snapshot) -> frozenset[str]:
    """Every level the snapshot classifies anything at, `none` aside: its columns' and, since
    M135, its certified dimensions'. What a grant meaning "everything cleared" has to clear --
    read from the columns alone, it left a dimension classified in Verity alone hidden from it."""
    return frozenset(
        level
        for level in (*(lvl for c in snapshot.columns for lvl in c.pii_levels()),
                      *(d.pii_level for d in snapshot.dimensions))
        if level and level != "none"
    )


@dataclass(frozen=True)
class GrantSet:
    objects: frozenset[str]
    writable: frozenset[str] = frozenset()  # a writable object is always readable (in objects)
    # compare=False keeps GrantSet hashable despite the dict; equality is NOT the auth boundary
    # (the fingerprint is), so excluding the filters from __eq__/__hash__ is harmless.
    row_filters: dict[str, str] = field(default_factory=dict, compare=False)
    pii_clearance: frozenset[str] = frozenset()  # pii_levels seen RAW
    pii_mask: frozenset[str] = frozenset()  # pii_levels seen MASKED (else denied)
    # False when the policy could not be READ, as opposed to a policy that legitimately grants
    # nothing. Both deny everything -- the difference is that one is an outage an operator must be
    # told about, and before this they were the same empty set (register M2).
    available: bool = True
    # The objects of which this request may not use what enrichment read from their rows (harvested values, and the
    # counts a request shows), nor, where no value was harvested, show the meaning the model wrote from those counts
    # (descriptions no certified record stated, code schemes, table facts, examples, codes not certified): in Superset
    # mode on a snapshot enriched directly from Oracle, every table Superset does not say, asked as the person, no row
    # rule applies to for them (`superset.grants.resolve_grants`). Empty everywhere else. Read through `values_shown`.
    values_withheld: frozenset[str] = frozenset()

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

        `available` is deliberately excluded: it describes whether we could READ the policy, not
        what the policy grants. Including it would repartition the cache during an outage, which
        is a second failure on top of the first.

        `values_withheld` is included where it holds anything, and only then, so every fingerprint
        taken where nothing is withheld is what it was. A turn answered while a table's values were
        shown may quote one, and a request withholding them must not be reminded of it
        (`scope_history` keeps only turns of the same fingerprint).
        """
        parts = [
            "R:" + "|".join(sorted(self.objects)),
            "F:" + "|".join(f"{k}={v}" for k, v in sorted(self.row_filters.items())),
            "C:" + "|".join(sorted(self.pii_clearance)),
            "M:" + "|".join(sorted(self.pii_mask)),
        ]
        if self.values_withheld:
            parts.append("W:" + "|".join(sorted(self.values_withheld)))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def values_shown(grants: GrantSet, object_id: str) -> bool:
    """Whether a request under `grants` may use the values enrichment harvested from the rows of `object_id`: only for
    an object the grants give the person (by its exact id, as they grant it), and not where `values_withheld` names it,
    nor where a row filter of the grants does, since the harvest read every row and the filter keeps some from this
    identity. Withheld and filtered names match ignoring case (`names_one_object`): matching more withholds more, never
    less.

    The one predicate the readers of harvested values decide by: in Superset mode, the snapshot a request reads and the
    value index it reads (`superset.harvested`). Requiring the grant makes what is shown an object the grants were
    decided for, never one they know nothing of."""
    return grants.allows(object_id) and not (names_one_object(object_id, grants.values_withheld)
                                             or names_one_object(object_id, grants.row_filters))


EMPTY = GrantSet(frozenset())
# Denies exactly as much as EMPTY. It exists so the engine can say *why* nothing is granted.
UNAVAILABLE = GrantSet(frozenset(), available=False)


def _role_row_filters(granted: object) -> dict:
    """One role's row filters, table to predicate, as the policy writes them; {} for a role that has none.

    The single reading of where a role's row filters live, for `grants_for`, which applies them, and
    `row_filtered_objects`, which lists the tables they cover so an advisory never quotes those tables' values. Two
    readings drift: a place one learns to read and the other does not is a filtered table whose values the advisory
    shows. A list-form grant reads without filters. A `row_filters` that is not a mapping raises TypeError: neither
    reader can say what it filters."""
    if not isinstance(granted, dict):
        return {}
    filters = granted.get("row_filters") or {}
    if not isinstance(filters, dict):
        raise TypeError(f"row_filters is a {type(filters).__name__}, not a mapping of table to predicate")
    return filters


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

    def policy_roles(self) -> list[str]:
        """The roles this policy declares, for advisory checks at boot.

        Optional on the provider protocol: a control-plane provider need not enumerate, and
        callers treat its absence as "cannot check" rather than "nothing to check".
        """
        try:
            with open(self._path) as fh:
                return sorted(json.load(fh)["roles"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return []

    def row_filtered_objects(self) -> frozenset[str] | None:
        """Every object some role of this policy reads through a row filter, from ONE read of the file.

        For advisories that must not show a filtered table's values. One read, so a policy edited while it is
        consulted cannot pair one version's roles with another's filters. None when the file cannot be read or does
        not have the shape `grants_for` reads, which is not a policy without filters.
        """
        try:
            with open(self._path) as fh:
                roles = json.load(fh)["roles"]
            return frozenset(str(table) for granted in roles.values() for table in _role_row_filters(granted))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return None

    def grants_for(self, identity: IdentityContext) -> GrantSet:
        try:
            with open(self._path) as fh:
                policy = json.load(fh)
            roles = policy["roles"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # Fail closed AND loudly, which is what the docstring above has always promised. It
            # used to return EMPTY silently, so a typo'd path denied everything and looked
            # identical to a policy that grants nothing -- and the deferral it produced was
            # counted as the engine abstaining correctly (register M2).
            logger.error("authorization policy %s could not be read (%s); denying everything",
                         self._path, exc)
            return UNAVAILABLE

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
                for table, filt in _role_row_filters(granted).items():
                    # Matched case-INSENSITIVELY. Unquoted identifiers are case-insensitive in
                    # all three engines, so two roles naming one table in different cases are two
                    # grants on the SAME table and must OR-combine like any other pair -- which is
                    # what the comment below has always said and the exact `in` never did.
                    # Measured before this: `claim` and `CLAIM` stayed separate keys and the
                    # QUERY's spelling then chose which role's predicate applied, `FROM claim`
                    # getting one and `FROM CLAIM` the other, from one policy and one identity.
                    existing = next((k for k in row_filters if k.lower() == table.lower()), None)
                    if existing is not None:  # more roles = more visible rows: OR-combine
                        row_filters[existing] = f"({row_filters[existing]}) OR ({filt})"
                    else:
                        row_filters[table] = str(filt)
        objects |= writable  # a writable table is always readable (the decider requires it)
        return GrantSet(frozenset(objects), frozenset(writable), row_filters,
                        frozenset(clearance), frozenset(mask))

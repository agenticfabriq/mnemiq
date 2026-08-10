"""Which granted tables a row filter fails to reach.

`row_filters` is `{table: predicate}` and each entry stands alone. Nothing in the policy
says that a payment belongs to a customer who belongs to a store, so filtering `customer`
leaves `payment` wide open -- and the engine is right to leave it open, because that is
what the policy said.

Securing a tenancy axis therefore means enumerating every table that hangs off it. Measured
against Pagila: a policy filtering customer/staff/inventory/store still let a store manager
read `SELECT SUM(amount) FROM payment` for the whole business -- 67,416.51 against their own
33,689.74, so the other store's revenue by subtraction, without naming it or touching a
filtered table. The policy was written by someone who had just argued for writing it
carefully, and who missed two tables out of twenty-two.

This does not change behaviour. It reads the foreign keys enrichment already collects and
says which tables the author probably meant to cover, at load, once.

Direction is the whole of the precision here. A `many_to_one` edge points from the CHILD to
its parent, so a payment row belongs to a customer. Warning on that is useful. Reachability
in the other direction is not: `customer.address_id -> address` does not make addresses the
property of a store, and filtering shared reference data would hide rows people should see.
So only child-to-parent edges are followed, and only towards a table that is actually
filtered.
"""

from __future__ import annotations

import logging

from mnemiq.authz.grants import GrantSet

logger = logging.getLogger(__name__)

_OWNING = {"many_to_one", "one_to_one"}


def unfiltered_dependents(grants: GrantSet, relationships) -> list[tuple[str, str]]:
    """Granted, unfiltered tables whose rows hang off a filtered one.

    Returns `(table, the filtered ancestor it reaches)`, sorted, so a caller can name both
    the hole and the filter it defeats.
    """
    filtered = set(grants.row_filters)
    if not filtered:
        return []  # no tenancy axis declared: nothing to be inconsistent with

    parents: dict[str, set[str]] = {}
    for rel in relationships:
        if getattr(rel, "cardinality", "") not in _OWNING:
            continue
        child, parent = getattr(rel, "from_", None), getattr(rel, "to", None)
        if child and parent:
            parents.setdefault(child, set()).add(parent)

    found: list[tuple[str, str]] = []
    for table in grants.objects:
        if table in filtered:
            continue
        # Walk child -> parent until a filtered ancestor turns up. Transitive on purpose: a
        # grandchild leaks exactly as well as a child.
        seen, stack = {table}, [table]
        while stack:
            for parent in parents.get(stack.pop(), ()):
                if parent in filtered:
                    found.append((table, parent))
                    stack = []
                    break
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
    return sorted(set(found))


def warn_unfiltered_dependents(grants: GrantSet, relationships, role: str = "") -> None:
    """Log the holes, fail-soft. A warning, never a refusal -- the policy is the operator's."""
    try:
        holes = unfiltered_dependents(grants, relationships)
    except Exception:  # noqa: BLE001 -- an advisory check must never stop a boot
        logger.debug("row-filter coverage check failed", exc_info=True)
        return
    if not holes:
        return
    who = f" for role '{role}'" if role else ""
    logger.warning(
        "row-filter coverage%s: %d granted table(s) hang off a filtered table but carry no "
        "filter, so their rows are readable across the boundary -- %s",
        who, len(holes),
        "; ".join(f"{child} (reaches filtered {parent})" for child, parent in holes),
    )

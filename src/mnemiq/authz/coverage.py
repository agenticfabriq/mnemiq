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
from mnemiq.contract import PII_LEVELS

logger = logging.getLogger(__name__)

_OWNING = {"many_to_one", "one_to_one"}


def unfiltered_dependents(grants: GrantSet, relationships,
                          views=None) -> list[tuple[str, str]]:
    """Granted, unfiltered tables -- and granted views -- whose rows hang off a filtered one.

    Returns `(table or view, the filtered ancestor it reaches)`, sorted, so a caller can name
    both the hole and the filter it defeats.

    `views` is `{object_id: ViewDefinition}`. A granted view has no foreign keys of its own, so
    without it a view over a dependent table was a door this never named (M29): measured, a view
    over `payment` under a filtered `customer` was approved and read every store's payments.
    Each view is followed to its base tables, through nested views, and the walk below runs from
    each base. A view reaching ANY filtered table is skipped: the deciders refuse every read of
    it (UNGOVERNED_VIEW), so it is not readable across the boundary.
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

    def ancestor(table: str) -> str | None:
        # Walk child -> parent until a filtered ancestor turns up. Transitive on purpose: a
        # grandchild leaks exactly as well as a child.
        seen, stack = {table}, [table]
        while stack:
            for parent in sorted(parents.get(stack.pop(), ())):
                if parent in filtered:
                    return parent
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return None

    views = views or {}
    tables = set(parents) | {p for ps in parents.values() for p in ps} | filtered
    found: list[tuple[str, str]] = []
    for name in grants.objects:
        if name in filtered:
            continue
        if name not in views:
            hit = ancestor(name)
            if hit is not None:
                found.append((name, hit))
            continue
        bases = _view_bases(name, views, tables)
        if bases & filtered:
            continue
        hits = sorted(h for h in (ancestor(b) for b in bases) if h is not None)
        if hits:
            found.append((name, hits[0]))
    return sorted(set(found))


def _view_bases(view: str, views, tables: set[str]) -> set[str]:
    """The tables a view reads, through nested views, as the relationship graph spells them.

    A body is written in its source's naming -- `payment` where a federated snapshot keys
    `pg.payment` -- so a mention is matched to a table by any spelling (`spellings`, the same
    fold `check_views` refuses by). Matching more only names more holes, which for a warning is
    the safe direction. A body that does not parse contributes nothing: the deciders refuse such
    a view (UNRESOLVABLE_VIEW).
    """
    from mnemiq.sql.views import _mentions, body_of, spellings

    by_spelling: dict[str, set[str]] = {}
    for table in tables:
        for spelled in spellings({table}):
            by_spelling.setdefault(spelled, set()).add(table)
    nested = {spelled: key for key in views for spelled in spellings({key})}
    out: set[str] = set()
    seen, stack = {view}, [view]
    while stack:
        body = body_of(views[stack.pop()])
        if body is None:
            continue
        for mention in _mentions(body):
            inner = nested.get(mention)
            if inner is not None and inner not in seen:
                seen.add(inner)
                stack.append(inner)
            out |= by_spelling.get(mention, set())
    return out


def unknown_pii_levels(grants: GrantSet) -> list[str]:
    """The `pii_clearance`/`pii_mask` values that are not PII levels, sorted.

    Clearance/mask are sets drawn from `PII_LEVELS`, and a column is read raw iff its level is in
    `pii_clearance`. The sensitive columns an operator means to reach are classified `pii`/`phi`,
    so a value outside the vocabulary -- a `low`/`medium`/`high` clearance scale, say -- can clear
    none of them: it denies rather than leaks, but silently (M40). (Nothing enforces `PII_LEVELS`
    on `Column.pii_level` itself; a producer that wrote an off-vocabulary level is a separate gap.)
    """
    return sorted((grants.pii_clearance | grants.pii_mask) - set(PII_LEVELS))


def warn_unknown_pii_levels(grants: GrantSet, role: str = "") -> None:
    """Log a clearance/mask value that names no PII level, fail-soft. A warning, never a
    refusal -- the policy is the operator's, but a grant that grants nothing must be visible."""
    try:
        unknown = unknown_pii_levels(grants)
    except Exception:  # noqa: BLE001 -- an advisory check must never stop a boot
        logger.debug("pii-level vocabulary check failed", exc_info=True)
        return
    if not unknown:
        return
    who = f" for role '{role}'" if role else ""
    logger.warning(
        "pii-level vocabulary%s: pii_clearance/pii_mask names %s, which is not a PII level "
        "(%s) -- the sensitive columns it was meant to reach are classified pii/phi, and a value "
        "outside that set clears none of them, so they stay denied",
        who, ", ".join(unknown), "/".join(PII_LEVELS),
    )


def warn_unknown_filter_targets(grants: GrantSet, snapshot, role: str = "") -> None:
    """Log, at boot and on a snapshot swap, a row-filter key that names no object (M50).

    Unlike the other advisories here this one describes a REFUSAL, not a hole: the deciders refuse
    every read for an identity holding such a key (`sql.rls.check_filter_targets`). It is logged
    here too because a refusal that only shows up when someone asks is found by a user, not by
    the operator who can fix it.
    """
    from mnemiq.sql.policy import unknown_filter_targets

    try:
        unknown = unknown_filter_targets(snapshot, grants.row_filters)
    except Exception:  # noqa: BLE001 -- an advisory check must never stop a boot
        logger.debug("row-filter target check failed", exc_info=True)
        return
    if not unknown:
        return
    who = f" for role '{role}'" if role else ""
    logger.warning(
        "row-filter targets%s: %s name no table or view in the snapshot, so every read for this "
        "role is refused until the policy names an object the snapshot holds",
        who, ", ".join(repr(k) for k in sorted(unknown)),
    )


def warn_unfiltered_dependents(grants: GrantSet, relationships, role: str = "",
                               views=None) -> None:
    """Log the holes, fail-soft. A warning, never a refusal -- the policy is the operator's."""
    try:
        holes = unfiltered_dependents(grants, relationships, views=views)
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

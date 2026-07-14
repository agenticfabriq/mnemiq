from __future__ import annotations

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot


def schema_map(snapshot: Snapshot) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for column in snapshot.columns:
        tables.setdefault(column.object_id, set()).add(column.name)
    return tables


def visible_schema(snapshot: Snapshot, grants: GrantSet) -> dict[str, set[str]]:
    """The schema as this identity is allowed to know it.

    The decider consults nothing else, so an ungranted table is not merely rejected -- inside
    the decider it does not exist.
    """
    return {t: c for t, c in schema_map(snapshot).items() if grants.allows(t)}

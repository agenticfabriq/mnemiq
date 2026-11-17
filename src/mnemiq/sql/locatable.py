"""The columns a repair hint may name, or a source probe may ask about, for one caller.

`repair_hints` tells a writer that a value missing from the column it filtered on "does appear in t.c". That sentence
says something about t.c's values, and finding it may mean asking the source. So a column is locatable only where the
caller could have read those values plainly itself:

- a table it is granted (`visible`, built from its grants), and a base table: never a view, which can read a table
  the caller sees only through a row filter (M27);
- no row filter on that table: the value index is built without one (M5), and a probe would run without one too;
- a column its policy neither denies nor masks: `visible` lists tables a caller may read, not which of their columns
  it may read raw, so a granted table's denied column is still in it (Codex's review);
- a string column, as the value index reads them, and not personal data by the index's own rules (a sensitive name,
  or a `pii` or `phi` level), cleared or not.

Computed columns are the decider's to drop (`decide`), since only the source can say which they are.
"""

from __future__ import annotations

from mnemiq.catalog import is_sensitive_name
from mnemiq.contract import Snapshot
from mnemiq.semantic.values import SENSITIVE_PII, _is_string_type
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.views import ViewInventory


def locatable_columns(snapshot: Snapshot, visible: dict[str, set[str]], policy: AccessPolicy,
                      views: ViewInventory | None) -> tuple[tuple[str, str], ...]:
    """Every `(table, column)` of `snapshot` this caller may be told holds a value, in snapshot order. The policy's
    names are folded once into sets, as `names_one_object` and `AccessPolicy.denies` fold them per call: per column,
    those scans made this quadratic, seconds on a wide schema (the final review)."""
    views_folded = {name.lower() for name in (views or ())}
    filtered = {name.lower() for name in policy.row_filters}
    withheld = {(t.lower(), c.lower()) for t, c in (*policy.denied, *policy.masked)}
    found: list[tuple[str, str]] = []
    for column in snapshot.columns:
        table, name = column.object_id, column.name
        if (table not in visible or name not in visible[table]
                or table.lower() in views_folded or table.lower() in filtered
                or (table.lower(), name.lower()) in withheld
                or not _is_string_type(column.data_type)
                or is_sensitive_name(name) or column.pii_levels() & SENSITIVE_PII):
            continue
        found.append((table, name))
    return tuple(dict.fromkeys(found))


def without_computed(pairs, adapter, snapshot: Snapshot) -> tuple[tuple[str, str], ...]:
    """`pairs` less every column the source reports computed on read, under the snapshot's table name or its source
    binding's (the catalogue names the source's, a binding can rename it: Codex's review), case folded, as Oracle
    reports them lower case; none at all when the source cannot say which those are. Every virtual column, not only
    the ones whose expression calls a function: stricter, and it needs no function inventory."""
    if adapter is None or not hasattr(adapter, "virtual_columns"):
        return tuple(pairs)
    try:
        computed = {(t.lower(), c.lower()) for t, c, _ in adapter.virtual_columns()}
    except Exception:  # noqa: BLE001 -- the catalogue could not say: nothing is located
        return ()
    physical = {b.object_id: b.source_object for b in snapshot.source_bindings}
    return tuple(pair for pair in pairs
                 if not any((name.lower(), pair[1].lower()) in computed for name in (pair[0], physical.get(pair[0], pair[0]))))

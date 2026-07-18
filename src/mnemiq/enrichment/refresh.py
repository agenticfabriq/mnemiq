from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.contract import Snapshot
from mnemiq.sql.schema import schema_map


@dataclass
class CatalogDiff:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.dropped)


def catalog_diff(adapter, snapshot: Snapshot) -> CatalogDiff:
    """Compare the live source catalog to the snapshot: added / changed (column set differs) /
    dropped / unchanged tables."""
    snap_cols = schema_map(snapshot)  # {table: {col, ...}}
    live_cols: dict[str, set[str]] = {}
    for table, col, _type in adapter.list_columns():
        live_cols.setdefault(table, set()).add(col)

    diff = CatalogDiff()
    for table in sorted(set(snap_cols) | set(live_cols)):
        if table not in snap_cols:
            diff.added.append(table)
        elif table not in live_cols:
            diff.dropped.append(table)
        elif snap_cols[table] != live_cols[table]:
            diff.changed.append(table)
        else:
            diff.unchanged.append(table)
    return diff

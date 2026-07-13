from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.adapters.base import SourceAdapter


@dataclass
class ColumnInfo:
    name: str
    data_type: str


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)


def introspect(adapter: SourceAdapter) -> list[TableInfo]:
    tables: dict[str, TableInfo] = {}
    for table, column, dtype in adapter.list_columns():
        tables.setdefault(table, TableInfo(table)).columns.append(ColumnInfo(column, dtype))
    return list(tables.values())

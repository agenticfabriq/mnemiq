from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.adapters.base import SourceAdapter

# Naming convention for identifier columns in the source catalog. Key columns are
# join candidates, never coded vocabularies -- "policy_identifier = 1" has no meaning.
KEY_SUFFIXES = ("_identifier", "_id")


def is_key_like(column: str) -> bool:
    return column.endswith(KEY_SUFFIXES)


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

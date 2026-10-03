"""A source seen through its manifest's `tables` list (M110).

Enrichment took every table the source listed -- for Oracle, every table the schema owner holds --
and nothing narrowed it. A design partner could not point it at the five tables their questions
use, so they copied those out of production into a local DuckDB file and evaluated that instead:
a different engine and dialect from the one the pilot existed to test.

A source in the manifest may now carry `"tables": [...]`, exact names or shell-style patterns
(`*`, `?`, `[...]`), matched without regard to case -- Oracle reports names upper-case, a manifest
is usually written lower. `adapter_for` wraps the adapter when the list is set, so every reader of
the catalog sees the same tables: introspection and so profiling, cards and the semantic layer;
declared foreign keys, kept only when both ends are in scope; view definitions; and `refresh`'s
catalog diff. Execution is untouched -- what the engine may query is decided by the snapshot and
the access policy, and a table outside the list is in neither.

A name or pattern that matches nothing raises `TablesNotFound` when the catalog is read: a typo
must not shrink the model silently.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any


class TablesNotFound(RuntimeError):
    """A manifest `tables` entry matched no table the source reports."""


class TableScopedAdapter:
    def __init__(self, inner: Any, patterns: tuple[str, ...], source_id: str) -> None:
        self._inner = inner
        self._patterns = tuple(p.casefold() for p in patterns)
        self._raw = tuple(patterns)
        self._source_id = source_id

    def __getattr__(self, name: str) -> Any:
        # Everything this view does not narrow -- execution, dialect, functions -- is the source's.
        return getattr(self._inner, name)

    def in_scope(self, table: str) -> bool:
        name = table.casefold()
        return any(fnmatchcase(name, p) for p in self._patterns)

    def _check(self, tables: set[str]) -> None:
        lowered = {t.casefold() for t in tables}
        missing = [raw for raw, p in zip(self._raw, self._patterns, strict=True)
                   if not any(fnmatchcase(t, p) for t in lowered)]
        if missing:
            raise TablesNotFound(
                f"source {self._source_id!r} lists table(s) it does not report: "
                f"{', '.join(missing)} -- check the names in the manifest's \"tables\" "
                "(matched without regard to case; * and ? are patterns)")

    def list_columns(self) -> list[tuple[str, str, str]]:
        rows = self._inner.list_columns()
        self._check({table for table, _col, _type in rows})
        return [row for row in rows if self.in_scope(row[0])]

    def introspect(self) -> list[str]:
        return [table for table in self._inner.introspect() if self.in_scope(table)]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        # Both ends: a key into a table the model never sees would join to nothing it can name.
        return [fk for fk in self._inner.foreign_keys()
                if self.in_scope(fk[0]) and self.in_scope(fk[2])]

    def view_definitions(self) -> list[tuple[str, str, str]]:
        return [view for view in self._inner.view_definitions() if self.in_scope(view[0])]

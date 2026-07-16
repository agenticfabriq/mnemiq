from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pyarrow as pa


class SourceAdapter(Protocol):
    # The SQL dialect this adapter actually executes. The engine transpiles its plan
    # (written in duckdb) to this before running -- so a SQLite source receives SQLite,
    # not un-transpiled DuckDB (which fails on YEAR/LISTAGG/etc.).
    dialect: str

    def introspect(self) -> list[str]: ...

    def list_columns(self) -> list[tuple[str, str, str]]: ...

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]: ...

    def execute(self, sql: str) -> list[tuple]: ...

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table: ...

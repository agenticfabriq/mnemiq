from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pyarrow as pa


class SourceAdapter(Protocol):
    def introspect(self) -> list[str]: ...

    def list_columns(self) -> list[tuple[str, str, str]]: ...

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]: ...

    def execute(self, sql: str) -> list[tuple]: ...

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table: ...

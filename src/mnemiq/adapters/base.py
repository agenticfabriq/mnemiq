from __future__ import annotations

from typing import Protocol


class SourceAdapter(Protocol):
    def introspect(self) -> list[str]: ...

    def execute(self, sql: str) -> list[tuple]: ...

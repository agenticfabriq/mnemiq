"""Where a request's statements run in Superset mode: through an adapter built for that request alone, carrying the
token the request arrived with.

`superset_executor` is the only place on the ask path that builds a Superset client or adapter
(`tests/test_superset_guards.py` holds every module to that). A request gets an adapter of its own, so the token one
person's request carries is never on another's, and what the adapter reads once and keeps, the catalogue and the
function inventory, is read again for the next request. The runtime itself holds `NoSourceConnection`, which runs
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import DelegatedToken, SupersetMcp

if TYPE_CHECKING:
    from mnemiq.config import Settings

_NO_SOURCE = "Superset mode opens no source connection of its own"


class SupersetCredentialMissing(Exception):
    """A request Superset mode cannot answer as anyone: it carries no delegated token, or it asks for a write, which
    Superset mode never makes."""


class NoSourceConnection:
    """The runtime's adapter in Superset mode, where it holds no connection to the source.

    It names the SQL dialect, which the runtime's boot reads, and runs nothing: every other `SourceAdapter` method
    raises. Each request's statements run through the adapter `superset_executor` builds for it instead."""

    def __init__(self, dialect: str) -> None:
        self.dialect = dialect

    def introspect(self) -> list[str]:
        raise RuntimeError(_NO_SOURCE)

    def list_columns(self) -> list[tuple[str, str, str]]:
        raise RuntimeError(_NO_SOURCE)

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        raise RuntimeError(_NO_SOURCE)

    def view_definitions(self) -> list[tuple[str, str, str]]:
        raise RuntimeError(_NO_SOURCE)

    def user_functions(self) -> list[str]:
        raise RuntimeError(_NO_SOURCE)

    def builtin_functions(self) -> list[str]:
        raise RuntimeError(_NO_SOURCE)

    def reachable_user_functions(self) -> list[str]:
        raise RuntimeError(_NO_SOURCE)

    def virtual_columns(self) -> list[tuple[str, str, str]]:
        raise RuntimeError(_NO_SOURCE)

    def execute(self, sql: str) -> list[tuple]:
        raise RuntimeError(_NO_SOURCE)

    def execute_arrow(self, sql: str, timeout_s: float | None = None):
        raise RuntimeError(_NO_SOURCE)


def superset_executor(settings: Settings, delegation: DelegatedToken) -> SupersetSqlAdapter:
    """The adapter one request's statements run through: Superset's MCP service at `settings.superset_mcp_url`, called
    with `delegation`, the request's own verified token, on the configured database and schema.

    The deadlines are the adapter's own: a statement's call waits for its time limit plus `TRANSPORT_MARGIN_S`, so a
    slow statement reads as Superset's answer that it ran out of time, never as Superset not answering. Refuses
    anything but a `DelegatedToken`: the ask path carries the asking person's token and nothing else."""
    if not isinstance(delegation, DelegatedToken):
        raise TypeError(f"Superset mode runs a request's statements with its delegated token, not a "
                        f"{type(delegation).__name__}")
    return SupersetSqlAdapter(SupersetMcp(settings.superset_mcp_url, delegation),
                              database_id=settings.superset_database_id, schema=settings.superset_schema)

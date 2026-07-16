"""MCP server: the read plane as scoped tools. stdio transport.

This is the AF seam: standalone uses the local policy; AF (v1) fronts these same tools and
injects a JWT-derived identity. The boundary is IdentityContext + AuthzProvider -- no AF here.
"""

from __future__ import annotations

import os

from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.runtime import Runtime, build_runtime


def _db_read(
    runtime: Runtime, identity: IdentityContext, question: str, mode: str | None = None
) -> dict:
    ans = runtime.ask(question, identity, mode=mode)
    trace = ans.trace
    return {
        "answer": ans.answer,
        "deferred": ans.deferred,
        "mode": ans.mode,
        "sql": trace.target_sql if trace else None,  # a deferral never carries a fabricated query
        "trace": (
            {
                "tables_used": trace.tables_used,
                "enrichment_version": trace.enrichment_version,
                "timing": trace.timing,
            }
            if trace
            else None
        ),
    }


def _get_schema(runtime: Runtime, identity: IdentityContext) -> dict:
    return {"tables": runtime.schema(identity)}


def _identity_from_env() -> IdentityContext:
    return IdentityContext(
        tenant_id=os.getenv("MNEMIQ_TENANT", "local"),
        principal_id=os.getenv("MNEMIQ_PRINCIPAL", "local"),
        roles=[r for r in os.getenv("MNEMIQ_ROLES", "").split(",") if r],
    )


def build_mcp(runtime: Runtime, identity: IdentityContext):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mnemiq")

    @mcp.tool()
    def db_read(question: str, mode: str | None = None) -> dict:
        """Answer a natural-language question over the database (read-only). Returns the
        answer, the SQL run, and an auditable trace; defers honestly when it cannot answer.
        mode: 'instant' (cheapest, no retries), 'thinking' (default, self-repairing), or
        'deep' (5 candidates + judge + agreement gate -- highest precision, ~6x cost)."""
        return _db_read(runtime, identity, question, mode=mode)

    @mcp.tool()
    def get_schema() -> dict:
        """List the tables you are allowed to query."""
        return _get_schema(runtime, identity)

    return mcp


def serve(settings: Settings) -> None:
    runtime = build_runtime(settings)
    identity = _identity_from_env()
    build_mcp(runtime, identity).run()  # stdio

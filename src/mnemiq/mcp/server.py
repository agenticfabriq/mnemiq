"""MCP server: the read plane as scoped tools. stdio transport.

This is the AF seam: standalone uses the local policy; AF (v1) fronts these same tools and
injects a JWT-derived identity. The boundary is IdentityContext + AuthzProvider -- no AF here.
"""

from __future__ import annotations


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
        # A governing agent has to choose between requesting a grant, rephrasing, and paging an
        # operator. Before this it got one boolean and a sentence for all three (register M6).
        "failed": ans.failed,
        "reason_code": str(ans.reason_code) if ans.reason_code else None,
        "mode": ans.mode,
        "sql": trace.target_sql if trace else None,  # a deferral never carries a fabricated query
        "preview": (
            {"columns": ans.preview.columns, "rows": ans.preview.rows,
             "row_count": ans.preview.row_count, "truncated": ans.preview.truncated}
            if ans.preview
            else None
        ),
        "trace": (
            {
                "tables_used": trace.tables_used,
                # M56: the marker travels with the list on every surface, not only the audit one.
                "lineage": {"tables": list(trace.tables_used),
                            "completeness": trace.lineage_completeness,
                            "unresolved": list(trace.lineage_unresolved),
                            "reasons": list(trace.lineage_reasons)},
                "enrichment_version": trace.enrichment_version,
                "timing": trace.timing,
            }
            if trace
            else None
        ),
    }


def _db_write(runtime: Runtime, identity: IdentityContext, sql: str) -> dict:
    res = runtime.write(sql, identity)
    return {
        "approved": res.approved,
        "target": res.target,
        "rows_affected": res.rows_affected,
        "refusal": res.refusal,
        "sql": res.target_sql,
    }


def _get_schema(runtime: Runtime, identity: IdentityContext) -> dict:
    return {"tables": runtime.schema(identity)}


def _identity_from_settings(settings: Settings | None) -> IdentityContext:
    from mnemiq.config import identity_from_settings

    return identity_from_settings(settings)


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
    def db_write(sql: str) -> dict:
        """Execute a single INSERT/UPDATE/DELETE (read/write governed separately from db_read).
        Refused by default: a write runs only when the identity has a write grant AND the
        deployment enabled writes. DDL and multi-statement input are never executed. Returns
        {approved, target, rows_affected, refusal, sql}; a governance plane records the result."""
        return _db_write(runtime, identity, sql)

    @mcp.tool()
    def get_schema() -> dict:
        """List the tables you are allowed to query."""
        return _get_schema(runtime, identity)

    return mcp


def serve(settings: Settings) -> None:
    runtime = build_runtime(settings)
    identity = _identity_from_settings(settings)
    build_mcp(runtime, identity).run()  # stdio

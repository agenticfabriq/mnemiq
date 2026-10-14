"""MCP server: the read plane as scoped tools. stdio transport.

This is the AF seam: standalone uses the local policy; AF (v1) fronts these same tools and
injects a JWT-derived identity. The boundary is IdentityContext + AuthzProvider -- no AF here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock

from mnemiq.adapters.federated import UnfederatableSource
from mnemiq.adapters.oracle import DriverMissing
from mnemiq.adapters.resolve import SourceUnconfigured, UnknownSourceKind
from mnemiq.adapters.scoped import TablesNotFound
from mnemiq.agent.modes import MODES
from mnemiq.agent.route import UnknownMode
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.duckdb_extensions import ExtensionNotInstalled, ExtensionsNotReady
from mnemiq.runtime import Runtime, SnapshotMissing, build_runtime
from mnemiq.server.serialize import verified_state

logger = logging.getLogger(__name__)


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
        # M89: the same derivation as the HTTP payload, imported rather than restated. A governing
        # agent deciding whether to act on an answer needs to know the verifier could not be
        # reached; hand-building a second dict here is how this surface would silently diverge,
        # which is the failure M56's marker comment warns about one field down.
        "verified": verified_state(ans.verify_layer),
        # A deferral never carries a fabricated query. The statement as asked, not as executed
        # (M51): the executed one carries the row policy's own predicate.
        "sql": trace.asked_sql if trace else None,
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
                  # Structured beside the sentence already in `answer`. An agent acting on a
                  # narrowed result needs the fact machine-readable; `null` means governance was
                  # not evaluated, `[]` that it narrowed nothing.
                  "narrowed": ([n.model_dump() for n in trace.narrowed]
                               if trace.narrowed is not None else None),
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
        "sql": res.asked_sql,  # M51: as asked; the executed write carries the row policy
        # Beside `rows_affected`, which is exactly where an agent needs it: 3 of 47 deleted reads
        # as success without this. `null` means the decision was not evaluated.
        "narrowed": ([{"object": n.object, "rows": n.rows, "columns": n.columns}
                      for n in res.narrowed] if res.narrowed is not None else None),
    }


def _get_schema(runtime: Runtime, identity: IdentityContext) -> dict:
    return {"tables": runtime.schema(identity)}


def _identity_from_settings(settings: Settings | None) -> IdentityContext:
    from mnemiq.config import identity_from_settings

    return identity_from_settings(settings)


def _mcp_classes():
    """The server class and the anticipated-failure exception of whichever `mcp` is installed.
    mcp 2.x renamed `FastMCP` to `MCPServer` and moved both (M121); the decorator and `run()` this
    module uses are the same in each, measured by the stdio startup test on mcp 1.28 and 2.3."""
    try:
        from mcp.server.mcpserver import MCPServer  # mcp 2.x
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
        from mcp.server.fastmcp.exceptions import ToolError
    return MCPServer, ToolError


# What a client is told for a setup failure is composed here, from what this layer knows -- never
# `str(exc)`, which carries the store path, the extension directory or a manifest's source ids
# (Codex's review of #114; SECURITY.md: configuration reaching a response). The full exception
# goes to the server log, which is where the engine puts what it withholds from the caller.
_INTERNAL = "mnemiq could not complete this call: an internal error, recorded in the server log."


class _DeploymentMode(UnknownMode):
    """An UnknownMode from the deployment's own MNEMIQ_MODE, raised building the runtime -- told
    apart from a mistyped per-call `mode`, which is the caller's to fix."""


def _client_message(exc: Exception) -> str:
    if isinstance(exc, SnapshotMissing) and isinstance(exc.__cause__, (SourceUnconfigured, UnknownSourceKind)):
        return ("This deployment's source is not configured: the operator's server log names what "
                "is missing.")
    if isinstance(exc, SnapshotMissing):
        return ("This deployment has no snapshot of its source yet: the operator runs "
                "`mnemiq enrich` then `mnemiq build`.")
    if isinstance(exc, SourceUnconfigured):
        return ("This deployment's source is not configured: the operator's server log names what "
                "is missing.")
    if isinstance(exc, ExtensionsNotReady):  # missing, or installed and refused at LOAD: fetch may not fix it
        return ("A DuckDB extension this deployment loads is missing or will not load, and "
                "MNEMIQ_LOCAL_ONLY forbids downloading one: the server log names which, and why.")
    if isinstance(exc, ExtensionNotInstalled):
        return ("A DuckDB extension this deployment loads is not installed, and MNEMIQ_LOCAL_ONLY "
                "forbids downloading it: the operator pre-seeds it with `mnemiq extensions fetch`; "
                "the server log names which.")
    if isinstance(exc, _DeploymentMode):
        return (f"This deployment's MNEMIQ_MODE names no mode: the operator sets it to one of "
                f"{sorted(MODES)}.")
    if isinstance(exc, UnknownMode):
        return f"Unknown mode; this deployment offers: {sorted(MODES)}."
    if isinstance(exc, TablesNotFound):
        return ("A table this deployment's source manifest lists does not exist in its source: the "
                "operator's server log names it.")
    if isinstance(exc, UnfederatableSource):
        return ("This deployment's source manifest federates a source DuckDB cannot attach: the "
                "operator's server log names it.")
    if isinstance(exc, DriverMissing):
        return "The Oracle driver is not installed: the operator installs `mnemiq[oracle]`."
    return _INTERNAL


_SETUP = (SnapshotMissing, SourceUnconfigured, ExtensionNotInstalled, ExtensionsNotReady,
          UnknownMode, UnfederatableSource, DriverMissing, TablesNotFound)


def build_mcp(runtime: Runtime | Callable[[], Runtime], identity: IdentityContext):
    """Register tools without opening the database when given a runtime factory.

    Initialize on the first tool call and reuse the runtime. Failed initialization
    remains a tool error and can be retried after the operator completes setup.
    """
    server_class, tool_error = _mcp_classes()
    mcp = server_class("mnemiq")
    runtime_lock = Lock()

    def get_runtime() -> Runtime:
        nonlocal runtime
        with runtime_lock:
            if callable(runtime):
                try:
                    runtime = runtime()
                except UnknownMode as exc:  # building the runtime: the deployment's MNEMIQ_MODE
                    raise _DeploymentMode(str(exc)) from exc
            return runtime

    def guarded(call: Callable[[], dict]) -> dict:
        """Every tool's body, so a failure reaches the client the same way on mcp 1.x and 2.x: a
        setup failure the operator can act on as its composed instruction, anything else as
        `_INTERNAL`. mcp 1.x put any exception's own words on the wire, and 2.1+ hides all but a
        ToolError's; both now get the message composed above, and the server log the rest."""
        try:
            return call()
        except tool_error:
            raise
        except _SETUP as exc:
            logger.warning("mnemiq MCP tool: setup failure: %s", exc)
            raise tool_error(_client_message(exc)) from exc
        except Exception as exc:
            logger.exception("mnemiq MCP tool failed")
            raise tool_error(_INTERNAL) from exc

    @mcp.tool()
    def db_read(question: str, mode: str | None = None) -> dict:
        """Answer a natural-language question over the database (read-only). Returns the
        answer, the SQL run, and an auditable trace; defers honestly when it cannot answer.
        mode: 'instant' (cheapest, no retries), 'thinking' (default, self-repairing), or
        'deep' (5 candidates + judge + agreement gate -- highest precision, ~6x cost)."""
        return guarded(lambda: _db_read(get_runtime(), identity, question, mode=mode))

    @mcp.tool()
    def db_write(sql: str) -> dict:
        """Execute a single INSERT/UPDATE/DELETE (read/write governed separately from db_read).
        Refused by default: a write runs only when the identity has a write grant AND the
        deployment enabled writes. DDL and multi-statement input are never executed. Returns
        {approved, target, rows_affected, refusal, sql}; a governance plane records the result."""
        return guarded(lambda: _db_write(get_runtime(), identity, sql))

    @mcp.tool()
    def get_schema() -> dict:
        """List the tables you are allowed to query."""
        return guarded(lambda: _get_schema(get_runtime(), identity))

    return mcp


def serve(settings: Settings) -> None:
    identity = _identity_from_settings(settings)
    build_mcp(lambda: build_runtime(settings), identity).run()  # stdio

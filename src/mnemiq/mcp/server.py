"""MCP server: the read plane as scoped tools, over stdio or streamable HTTP.

Over stdio the process is the caller, so one identity from settings serves every call. Over HTTP (served at /mcp by
`mnemiq serve --http`) each request carries its own bearer token, verified by `mnemiq.auth`, and each tool call is
answered as that token's identity, gated by its scopes. The boundary is IdentityContext + AuthzProvider.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from urllib.parse import urlsplit
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
from mnemiq.profiles import ProfileFileError, UnknownProfile
from mnemiq.runtime import Runtime, SnapshotMissing, build_runtime
from mnemiq.server.serialize import verified_state, write_payload

logger = logging.getLogger(__name__)


def _db_read(
    runtime: Runtime, identity: IdentityContext, question: str, mode: str | None = None,
    profile: str | None = None,
) -> dict:
    ans = runtime.ask(question, identity, mode=mode, profile=profile)
    trace = ans.trace
    return {
        "answer": ans.answer,
        "deferred": ans.deferred,
        # A governing agent has to choose between requesting a grant, rephrasing, and paging an
        # operator. Before this it got one boolean and a sentence for all three (register M6).
        "failed": ans.failed,
        # The rows stand and their sentence failed: the answer is the rows, and an agent can tell it from prose off.
        "prose_failed": ans.prose_failed,
        "reason_code": str(ans.reason_code) if ans.reason_code else None,
        "mode": ans.mode,
        "profile": ans.profile,
        "profile_version": ans.profile_version,
        "model": ans.model,
        "engine_commit": ans.engine_commit,
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
    # The one write serializer, shared with `mnemiq write --json`: the statement as asked, and
    # `narrowed` beside `rows_affected` (M51).
    return write_payload(runtime.write(sql, identity))


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
    if isinstance(exc, SnapshotMissing) and isinstance(exc.__cause__, ProfileFileError):
        return "This deployment's profile file cannot be run: the operator's server log says why."
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
    if isinstance(exc, UnknownProfile):  # unknown and forbidden read alike, and neither repeats the name
        return f"Unknown profile; you may pick: {exc.offered}."
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
          UnknownMode, UnknownProfile, UnfederatableSource, DriverMissing, TablesNotFound)


def build_mcp(runtime: Runtime | Callable[[], Runtime],
              identity: IdentityContext | Callable[[str], IdentityContext], *, server_kwargs: dict | None = None):
    """Register tools without opening the database when given a runtime factory.

    Initialize on the first tool call and reuse the runtime. Failed initialization
    remains a tool error and can be retried after the operator completes setup.

    `identity` is one IdentityContext for every call (stdio), or a resolver called on each tool call with the
    scope that tool needs, which answers that request's own identity or raises the tool error (HTTP).
    """
    server_class, tool_error = _mcp_classes()
    mcp = server_class("mnemiq", **(server_kwargs or {}))
    who = identity if callable(identity) else (lambda scope: identity)
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
    def db_read(question: str, mode: str | None = None, profile: str | None = None) -> dict:
        """Answer a natural-language question over the database (read-only). Returns the
        answer, the SQL run, and an auditable trace; defers honestly when it cannot answer.
        mode: 'instant' (cheapest, no retries), 'thinking' (default, self-repairing), or
        'deep' (5 candidates + judge + agreement gate -- highest precision, ~6x cost).
        profile: a model and its settings this deployment offers you; unset uses its default."""
        return guarded(lambda: _db_read(get_runtime(), who("db.read"), question, mode=mode, profile=profile))

    @mcp.tool()
    def db_write(sql: str) -> dict:
        """Execute a single INSERT/UPDATE/DELETE (read/write governed separately from db_read).
        Refused by default: a write runs only when the identity has a write grant AND the
        deployment enabled writes. DDL and multi-statement input are never executed. Returns
        {approved, target, rows_affected, refusal, sql}; a governance plane records the result."""
        return guarded(lambda: _db_write(get_runtime(), who("db.write"), sql))

    @mcp.tool()
    def get_schema() -> dict:
        """List the tables you are allowed to query."""
        return guarded(lambda: _get_schema(get_runtime(), who("db.read")))

    return mcp


def http_identity(cfg):
    """The per-call resolver for MCP over HTTP: the identity of the token this request carried, gated by the scope
    the tool needs. With no verified token in context it refuses; it never falls back to a settings identity."""
    from mcp.server.auth.middleware.auth_context import get_access_token

    from mnemiq.auth.resolve import identity_from_oidc
    from mnemiq.auth.token import TokenRejected

    _, tool_error = _mcp_classes()

    def who(scope: str) -> IdentityContext:
        token = get_access_token()
        if token is None or not token.claims:
            raise tool_error("no verified token")
        if scope not in token.scopes:
            raise tool_error(f"this token lacks the {scope} scope")
        try:
            return identity_from_oidc(token.claims, cfg)
        except TokenRejected as exc:  # the map went unreadable since the transport checked it, say
            raise tool_error(f"not permitted: {exc.reason}") from exc

    return who


def mcp_http_settings(cfg) -> dict:
    """What MCP over HTTP is served with, on either mcp: the external path /mcp, stateless (each request is judged
    by its own token, and no session outlives the token that opened it), and DNS-rebinding protection for the
    audience's own host only, which a proxy in front preserves."""
    from mcp.server.transport_security import TransportSecuritySettings

    parts = urlsplit(cfg.oidc.audience)
    origin = f"{parts.scheme}://{parts.netloc}"
    return {"streamable_http_path": "/mcp", "stateless_http": True,
            "transport_security": TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                                            allowed_hosts=[parts.netloc], allowed_origins=[origin])}


def _auth_settings(oidc):
    from mcp.server.auth.settings import AuthSettings

    kwargs = {"issuer_url": oidc.issuer, "resource_server_url": oidc.audience}
    # The verifier checks `aud` itself, exactly; the SDK's own resource check is looser and is left off where the
    # installed mcp has it.
    if "validate_token_resource" in AuthSettings.model_fields:
        kwargs["validate_token_resource"] = False
    return AuthSettings(**kwargs)


class _HeaderScreen:
    """ASGI middleware in front of the SDK's bearer check on /mcp, applying the header rules every door shares
    (`mnemiq.auth.doors.screen`): the SDK reads the first Authorization header and ignores everything else, so a
    valid token beside Open WebUI's headers, or beside a second Authorization, would otherwise be served."""

    def __init__(self, app, cfg, path: str = "/mcp") -> None:
        self.app, self.cfg, self.path = app, cfg, path

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == self.path:
            from mnemiq.auth.doors import DoorRefusal, screen

            headers: dict[str, list[str]] = {}
            for name, value in scope.get("headers", []):
                headers.setdefault(name.decode("latin-1").lower(), []).append(value.decode("latin-1"))
            try:
                screen(self.cfg, self.path, headers)
            except DoorRefusal as exc:
                logger.info("refused on %s: %s", self.path, exc.reason)
                from starlette.responses import JSONResponse

                response = JSONResponse({"error": "invalid_request", "error_description": exc.reason},
                                        status_code=exc.status,
                                        headers={"WWW-Authenticate": 'Bearer error="invalid_request"'})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def mcp_http_app(runtime, cfg, *, fetch: Callable[[], dict] | None = None):
    """The MCP door's Starlette app, which `server/app.py` mounts at exactly /mcp, with its protected-resource
    metadata. The resource URL is MNEMIQ_OIDC_AUDIENCE itself, so the two cannot disagree. mcp 2.x moved the path,
    statelessness and transport security from the server's constructor to `streamable_http_app()`; both are
    handled here, as `_mcp_classes` handles the class's rename."""
    from mnemiq.auth.resolve import identity_from_oidc
    from mnemiq.auth.token import JwtIdentityVerifier

    oidc = cfg.oidc
    verifier = JwtIdentityVerifier(oidc, audience=oidc.audience, refuse_audience=oidc.openwebui_audience,
                                   fetch=fetch, check=lambda claims: identity_from_oidc(claims, cfg))
    server_class, _ = _mcp_classes()
    http = mcp_http_settings(cfg)
    server_kwargs = {"token_verifier": verifier, "auth": _auth_settings(oidc)}
    if "streamable_http_path" in inspect.signature(server_class.streamable_http_app).parameters:  # mcp 2.x
        app = build_mcp(runtime, http_identity(cfg), server_kwargs=server_kwargs).streamable_http_app(**http)
    else:
        app = build_mcp(runtime, http_identity(cfg), server_kwargs={**server_kwargs, **http}).streamable_http_app()
    # Added last, so it is the outermost layer and runs before the SDK's authentication middleware.
    app.add_middleware(_HeaderScreen, cfg=cfg)
    return app


def serve(settings: Settings) -> None:
    if settings.mcp_transport == "http":
        raise SystemExit("MNEMIQ_MCP_TRANSPORT=http: MCP over HTTP is served by mnemiq serve --http, at /mcp")
    identity = _identity_from_settings(settings)
    build_mcp(lambda: build_runtime(settings), identity).run()  # stdio

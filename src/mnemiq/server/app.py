"""HTTP API over Runtime: the same seam mcp/server.py wraps, as JSON + SSE.

Deferrals are 200s with deferred=true -- refusal is an answer, not a transport error.
"""

from __future__ import annotations

import ipaddress
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mnemiq.agent.route import UnknownMode, offered_modes
from mnemiq.profiles import UnknownProfile
from mnemiq.contract import HistoryTurn, IdentityContext
from mnemiq.server.openai_compat import mount_openai
from mnemiq.server.serialize import answer_payload
from mnemiq.server.sse import chat_stream
from mnemiq.server.superset_refusals import superset_refusal

logger = logging.getLogger(__name__)

# The built workbench lands inside the package so a wheel picks it up with no manifest
# entry. It is gitignored: the repo carries the source, the build carries the bundle.
STATIC_DIR = Path(__file__).parent / "static"

_UNBUILT = """<!doctype html><meta charset="utf-8"><title>mnemiq</title>
<body style="font:14px ui-monospace,monospace;padding:3rem;max-width:34rem">
<h1 style="font-size:1rem">The workbench is not built yet.</h1>
<p>Build it, then reload this page:</p>
<pre>cd workbench &amp;&amp; pnpm install &amp;&amp; pnpm build</pre>
<p>The API is already serving: <code>/v1/ask</code>, <code>/v1/chat</code>,
<code>/v1/schema</code>.</p>
"""


class AskBody(BaseModel):
    question: str
    mode: str | None = None
    profile: str | None = None  # None: the deployment's default
    # Prior turns the caller wants this question read against. Untrusted: the engine
    # replays only those whose grant_fingerprint matches the CURRENT identity's
    # boundary, so echoing another identity's turn buys nothing (agent/history.py).
    history: list[HistoryTurn] | None = None


class _Workbench(StaticFiles):
    """Serves the built bundle with the two cache lifetimes it actually has.

    Vite content-hashes everything under `assets/`, so those URLs are immutable and
    may be kept forever. `index.html` is the one file whose name never changes, and
    it is what names the current hashes -- cached, it pins the browser to whatever
    bundle it was built against, and a rebuild silently changes nothing. It must be
    revalidated on every load; `no-cache` means "ask first", not "do not store", so
    an unchanged build still answers 304.
    """

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        immutable = "/assets/" in scope.get("path", "")
        response.headers["cache-control"] = (
            "public, max-age=31536000, immutable" if immutable else "no-cache"
        )
        return response


class _NotPermitted(HTTPException):
    """A 403 on /v1: a caller the door has verified but who is not permitted, by the door (another tenant, no
    `db.read`) or, in Superset mode, by Superset (`_refused`). Its body carries where Sign out ends at the identity
    provider, beside the unchanged `detail`, so the workbench can offer a Sign out that switches accounts. Not a leak:
    the caller is known, and the URL is the provider's public page. A 401 never carries it."""

    def __init__(self, detail: str, headers: dict | None, end_session_url: str | None) -> None:
        super().__init__(status_code=403, detail=detail, headers=headers)
        self.end_session_url = end_session_url


def caller_dependency(doors, door: str = "v1", *, delegated: bool = False):
    """The FastAPI dependency that turns a request into its verified caller at `door` ("v1" or "openai"), or into
    the door's refusal: 401 with `WWW-Authenticate: Bearer`, 403 for another tenant or a token without the
    `db.read` scope (with RFC 6750's insufficient-scope challenge), 503 for an unreadable map. On /v1 a 403's body
    also carries `end_session_url`, the doors' own, null when unset.

    `delegated` (on /v1, for Superset mode) gives the caller as a `Delegation`: verified the same way, with the Bearer
    that verification read, to be handed on to Superset unchanged.

    Every value of every header goes to the door, keyed by lowercase name, so a credential sent twice is seen as
    sent twice. A plain function, so FastAPI runs it in a worker thread: a key fetch may block."""
    from mnemiq.auth.doors import DoorRefusal

    judge = doors.v1_delegation if delegated and door == "v1" else getattr(doors, door)

    def caller(request: Request):
        headers: dict[str, list[str]] = {}
        for name, value in request.headers.items():
            headers.setdefault(name.lower(), []).append(value)
        try:
            return judge(headers)
        except DoorRefusal as exc:
            challenge = exc.challenge or ("Bearer" if exc.status == 401 else None)
            headers = {"WWW-Authenticate": challenge} if challenge else None
            if exc.status == 403 and door == "v1":
                raise _NotPermitted(exc.reason, headers, doors.end_session_url) from None
            raise HTTPException(status_code=exc.status, detail=exc.reason, headers=headers) from None

    return caller


@dataclass(frozen=True)
class _Asker:
    """Who a /v1 request asks as, and what the runtime is handed beside them: in Superset mode the token the door
    verified, as `delegation`, which every call the request makes to Superset carries; nothing in direct mode."""

    identity: IdentityContext
    handed: dict


def _refused(exc: Exception, doors) -> HTTPException | None:
    """The response a Superset refusal, or a failure Superset mode answers itself, becomes on /v1, or None for any
    other exception. A 401 asks for the Bearer again, as the door's own does, and a 403 carries where Sign out ends, as
    every /v1 403 does."""
    found = superset_refusal(exc)
    if found is None:
        return None
    log = logger.warning if found.status >= 500 else logger.info
    log("/v1 answered %d for %s: %s", found.status, type(exc).__name__, found.detail)
    if found.status == 401:
        return HTTPException(status_code=401, detail=found.detail, headers={"WWW-Authenticate": "Bearer"})
    if found.status == 403:
        return _NotPermitted(found.detail, None, doors.end_session_url if doors is not None else None)
    return HTTPException(status_code=found.status, detail=found.detail)


def build_app(
    runtime, identity, heartbeat_s: float = 15.0, static_dir: Path | None = None, *, doors=None, mcp_app=None
) -> FastAPI:
    """`identity` serves every caller when `doors` is None (no issuer configured). With doors, every /v1 route
    takes the caller its token names, needs the `db.read` scope as MCP's read tools do, and `identity` is never used.

    In Superset mode (`runtime.settings.superset_mode`) `/v1/ask`, `/v1/chat` and `/v1/schema` also hand the runtime
    the Bearer the door verified, as `delegation`, so Superset answers as the person it names, and each refusal from
    Superset reaches the person as `server/superset_refusals.py` says, never as a 500.

    No route here writes: `Runtime.write` is reached from MCP's `db_write` and the CLI alone. A write route added
    here would need the `db.write` scope too."""
    lifespan = None
    if mcp_app is not None:
        @asynccontextmanager
        async def lifespan(_app):
            # The MCP app's session manager runs for the life of the server, stateless or not.
            async with mcp_app.router.lifespan_context(mcp_app):
                yield

    app = FastAPI(title="mnemiq", version="v1", lifespan=lifespan)

    @app.exception_handler(_NotPermitted)
    async def not_permitted(request: Request, exc: _NotPermitted) -> JSONResponse:
        return JSONResponse({"detail": exc.detail, "end_session_url": exc.end_session_url},
                            status_code=exc.status_code, headers=exc.headers)

    if doors is None:
        def caller() -> IdentityContext:
            return identity
    else:
        caller = caller_dependency(doors, "v1")

    superset = getattr(getattr(runtime, "settings", None), "superset_mode", False) is True
    if not superset:
        def asker(who: IdentityContext = Depends(caller)) -> _Asker:
            return _Asker(who, {})
    elif doors is None:
        # No issuer, which startup refuses in Superset mode: no token to hand on, so the runtime refuses each request.
        def asker() -> _Asker:
            return _Asker(identity, {"delegation": None})
    else:
        verified = caller_dependency(doors, "v1", delegated=True)

        def asker(found=Depends(verified)) -> _Asker:
            return _Asker(found.identity, {"delegation": found.token})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/v1/whoami")
    def whoami(who: IdentityContext = Depends(caller)) -> dict:
        # What the workbench keys its stored threads by, so one browser never shows a user another's. `verified`
        # tells a token's identity from the one settings identity, whose threads the browser may keep.
        # `end_session_url` is where its Sign out ends, at the identity provider: operator configuration, never
        # anything the request carries, and told only to a caller the door has verified.
        return {"principal_id": who.principal_id, "tenant_id": who.tenant_id, "verified": doors is not None,
                "end_session_url": doors.end_session_url if doors is not None else None}

    @app.post("/v1/ask")
    def ask(body: AskBody, asking: _Asker = Depends(asker)) -> dict:
        try:
            return answer_payload(
                runtime.ask(body.question, asking.identity, mode=body.mode, history=body.history,
                            profile=body.profile, **asking.handed)
            )
        except UnknownProfile as exc:
            # Composed from what this caller may pick. Unknown and forbidden read alike, and neither repeats the name.
            raise HTTPException(status_code=422, detail=f"unknown profile; you may pick: {exc.offered}") from exc
        except UnknownMode as exc:
            # Built from the modes this runtime offers, not from `str(exc)`. The text this
            # particular exception carries happens to be safe -- the caller's own override and a
            # fixed list -- but "safe because this exception is safe" is the reasoning that puts an
            # error's words on the wire, and it stops holding the moment the message gains a field
            # or another exception reaches this handler. Same rule as the stream's error frame:
            # what the caller is told is composed here, from what this layer already knows.
            raise HTTPException(
                status_code=422,
                detail=f"unknown mode; this deployment offers: {offered_modes(getattr(runtime, 'router', None))}",
            ) from exc
        except Exception as exc:
            if (refused := _refused(exc, doors)) is None:
                raise
            raise refused from None

    @app.post("/v1/chat")
    def chat(body: AskBody, asking: _Asker = Depends(asker)) -> StreamingResponse:
        # The caller is resolved before the stream starts, so a refusal is a plain 401, never an error frame.
        return StreamingResponse(
            chat_stream(runtime, asking.identity, body.question, body.mode,
                        heartbeat_s=heartbeat_s, history=body.history, profile=body.profile, **asking.handed),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.get("/v1/schema")
    def schema(asking: _Asker = Depends(asker)) -> dict:
        # `scope` rather than `schema`: the opening screen's starters are access-scoped from the
        # same grant resolution as the cards, so the two cannot disagree (M32).
        try:
            return runtime.scope(asking.identity, **asking.handed)
        except Exception as exc:
            if (refused := _refused(exc, doors)) is None:
                raise
            raise refused from None

    if mcp_app is not None:
        from starlette.routing import Route

        # Each of the MCP app's own paths, exactly: /mcp and its protected-resource metadata. Routed whole to that
        # app, which sees the path unchanged, so its door is /mcp and never /mcp/mcp.
        for path in [getattr(route, "path", None) for route in mcp_app.routes]:
            if path:
                app.router.routes.append(Route(path, endpoint=mcp_app))

    mount_openai(app, runtime, doors, heartbeat_s=heartbeat_s)

    # Mounted last, so it can only claim paths no API route above already answered.
    built = static_dir if static_dir is not None else STATIC_DIR
    if (built / "index.html").is_file():
        app.mount("/", _Workbench(directory=built, html=True), name="workbench")
    else:

        @app.get("/", response_class=HTMLResponse)
        def unbuilt() -> str:
            return _UNBUILT

    return app


def _run_uvicorn(app, host: str, port: int) -> None:  # seam for tests
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def serve_http(settings, host: str, port: int) -> None:
    try:
        import uvicorn  # noqa: F401  -- fail before building a runtime
    except ImportError as exc:
        raise SystemExit(
            "the HTTP server needs the server extra: uv sync --extra server"
        ) from exc
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.config import identity_from_settings
    from mnemiq.runtime import build_runtime
    from mnemiq.superset.settings import check_superset_settings

    # Before the runtime: a configuration the doors cannot enforce stops here, having opened nothing.
    cfg = auth_config_from_settings(settings) if settings is not None else None
    if settings is not None:
        # In direct mode as well: a Superset database id set without the URL is refused there, rather than served
        # directly with mnemiq's own credential.
        check_superset_settings(settings, cfg)
    runtime = build_runtime(settings)
    doors = mcp_app = None
    if cfg is not None:
        from mnemiq.auth.doors import HttpDoors
        from mnemiq.auth.identity_map import IdentityMap, IdentityMapUnavailable

        doors = HttpDoors(cfg)
        if cfg.mcp_http:
            from mnemiq.mcp.server import mcp_http_app

            mcp_app = mcp_http_app(runtime, cfg)
        if cfg.map_path:
            try:
                IdentityMap.read(cfg.map_path, issuers=cfg.issuers,
                                 groups_passthrough=bool(cfg.oidc and cfg.oidc.groups_claim),
                                 oidc_issuer=cfg.oidc.issuer if cfg.oidc else None,
                                 ).warn_undeclared_roles(getattr(runtime, "authz", None))
            except IdentityMapUnavailable:
                pass  # logged where it was read; every request is refused until it is fixed
    elif not _loopback(host):
        logger.warning("serving every caller on %s as the one identity in MNEMIQ_PRINCIPAL/MNEMIQ_ROLES; set "
                       "MNEMIQ_OIDC_ISSUER to take each request's identity from its own token", host)
    _run_uvicorn(build_app(runtime, identity_from_settings(settings), doors=doors, mcp_app=mcp_app),
                 host=host, port=port)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

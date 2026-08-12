"""HTTP API over Runtime: the same seam mcp/server.py wraps, as JSON + SSE.

Deferrals are 200s with deferred=true -- refusal is an answer, not a transport error.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mnemiq.agent.route import UnknownMode
from mnemiq.contract import HistoryTurn
from mnemiq.server.serialize import answer_payload
from mnemiq.server.sse import chat_stream

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


def build_app(
    runtime, identity, heartbeat_s: float = 15.0, static_dir: Path | None = None
) -> FastAPI:
    app = FastAPI(title="mnemiq", version="v1")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/v1/ask")
    def ask(body: AskBody) -> dict:
        try:
            return answer_payload(
                runtime.ask(body.question, identity, mode=body.mode, history=body.history)
            )
        except UnknownMode as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/chat")
    def chat(body: AskBody) -> StreamingResponse:
        return StreamingResponse(
            chat_stream(runtime, identity, body.question, body.mode,
                        heartbeat_s=heartbeat_s, history=body.history),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.get("/v1/schema")
    def schema() -> dict:
        # `scope` rather than `schema`: the opening screen's starters are access-scoped from the
        # same grant resolution as the cards, so the two cannot disagree (M32).
        return runtime.scope(identity)

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
    from mnemiq.config import identity_from_settings
    from mnemiq.runtime import build_runtime

    runtime = build_runtime(settings)
    _run_uvicorn(build_app(runtime, identity_from_settings(settings)), host=host, port=port)

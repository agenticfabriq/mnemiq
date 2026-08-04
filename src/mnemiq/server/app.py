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
            return answer_payload(runtime.ask(body.question, identity, mode=body.mode))
        except UnknownMode as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/chat")
    def chat(body: AskBody) -> StreamingResponse:
        return StreamingResponse(
            chat_stream(runtime, identity, body.question, body.mode, heartbeat_s=heartbeat_s),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.get("/v1/schema")
    def schema() -> dict:
        return {"tables": runtime.schema(identity)}

    # Mounted last, so it can only claim paths no API route above already answered.
    built = static_dir if static_dir is not None else STATIC_DIR
    if (built / "index.html").is_file():
        app.mount("/", StaticFiles(directory=built, html=True), name="workbench")
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

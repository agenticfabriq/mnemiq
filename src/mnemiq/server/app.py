"""HTTP API over Runtime: the same seam mcp/server.py wraps, as JSON + SSE.

Deferrals are 200s with deferred=true -- refusal is an answer, not a transport error.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mnemiq.agent.route import UnknownMode
from mnemiq.server.serialize import answer_payload
from mnemiq.server.sse import chat_stream


class AskBody(BaseModel):
    question: str
    mode: str | None = None


def build_app(runtime, identity, heartbeat_s: float = 15.0) -> FastAPI:
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

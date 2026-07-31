"""HTTP API over Runtime: the same seam mcp/server.py wraps, as JSON + SSE.

Deferrals are 200s with deferred=true -- refusal is an answer, not a transport error.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mnemiq.agent.route import UnknownMode
from mnemiq.server.serialize import answer_payload


class AskBody(BaseModel):
    question: str
    mode: str | None = None


def build_app(runtime, identity) -> FastAPI:
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

    @app.get("/v1/schema")
    def schema() -> dict:
        return {"tables": runtime.schema(identity)}

    return app

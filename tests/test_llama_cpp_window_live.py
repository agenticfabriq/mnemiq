"""The prompt window on a real llama.cpp server: the count is what the server charges, the window is one request's.

Runs only with MNEMIQ_TEST_LLAMACPP_URL set to a running `llama-server` (any model; a router started with
`--models-dir` is asked for the first model it lists), e.g.

    llama-server -m <model.gguf> --port 18091 -c 8192 --parallel 2 --jinja
    MNEMIQ_TEST_LLAMACPP_URL=http://127.0.0.1:18091/v1 pytest tests/test_llama_cpp_window_live.py
"""

from __future__ import annotations

import os

import httpx
import pytest

from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot, SourceBinding
from mnemiq.llm.window import count_on_server, everything
from mnemiq.semantic.fit import fitter_for
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

URL = os.getenv("MNEMIQ_TEST_LLAMACPP_URL", "")
pytestmark = [pytest.mark.live_llm, pytest.mark.skipif(not URL, reason="no llama.cpp server (MNEMIQ_TEST_LLAMACPP_URL)")]
ROOT = URL.rstrip("/").removesuffix("/v1")


def _model() -> str:
    return httpx.get(f"{ROOT}/v1/models", timeout=30).json()["data"][0]["id"]


def _charged(system: str, user: str) -> int:
    """What the server's own chat endpoint charges for these messages."""
    reply = httpx.post(f"{ROOT}/v1/chat/completions", timeout=120, json={"model": _model(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 1, "temperature": 0})
    return reply.json()["usage"]["prompt_tokens"]


@pytest.mark.parametrize("user", [
    "QUESTION: how many lots?\n\ntable lots\ncolumns:\n- lot_id (VARCHAR, key) the lot.",
    "héllo wörld 12345 — 中文 " * 20,
])
def test_the_count_is_what_the_server_charges_and_the_window_is_the_models(user):
    got = count_on_server(URL, _model(), "You write SQL.", user)
    assert got is not None
    tokens, window = got
    assert tokens >= _charged("You write SQL.", user), "a count may err long, never short"
    assert tokens - _charged("You write SQL.", user) <= 1, "at most a leading BOS counted twice"
    # `/slots`, not `/props` (which the count reads): each slot's own context, the server's `-c` split across `--parallel`.
    slots = httpx.get(f"{ROOT}/slots", params={"model": _model()}, timeout=30).json()
    assert window == slots[0]["n_ctx"] > 0, "one request's window, not the server's whole -c"


def test_the_fit_counts_on_llama_cpp():
    snap = Snapshot(version="v", source_id="s", created_at="2026-11-25",
                    source_bindings=[SourceBinding(id="sb:lots", source_id="s", object_id="lots", source_object="lots",
                                                   binding_type="table")],
                    columns=[Column(id="lots.lot_id", object_id="lots", name="lot_id", data_type="text")])
    packet = ContextPacket(question="how many lots?", cards=[RetrievedCard(object_id="lots", card="table lots", score=1.0)],
                           grant_fingerprint="", enrichment_version=None)
    settings = Settings(llm_base_url=URL, llm_model=_model(), llm_api_key="local")
    fit = fitter_for(settings, "duckdb").measure(packet, snap, everything(snap))
    assert fit.counted_by == "server" and fit.budget is not None

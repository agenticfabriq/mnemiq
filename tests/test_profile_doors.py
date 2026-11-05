"""Each product door carries the profile, and refuses alike for unknown and forbidden.

A setting only `Runtime.ask` reads is a setting nobody can reach. The eval door refuses a profile file instead: it builds its
own agent and takes no mode, and `test_the_eval_door_refuses_settings_it_would_not_apply` holds that.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer
from mnemiq.cli import main
from mnemiq.contract import IdentityContext
from mnemiq.mcp.server import _client_message, _db_read
from mnemiq.profiles import UnknownProfile
from mnemiq.runtime import SnapshotMissing
from mnemiq.server.app import build_app


def _identity() -> IdentityContext:
    return IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


class _RT:
    def __init__(self, raises=None):
        self.raises = raises
        self.profile = "not asked"

    def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
        self.profile = profile
        if self.raises:
            raise self.raises
        return AgentAnswer(answer="A", mode="instant")


def _client(rt) -> TestClient:
    return TestClient(build_app(rt, _identity()))


def _events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_the_ask_door_passes_the_profile():
    rt = _RT()
    assert _client(rt).post("/v1/ask", json={"question": "q", "profile": "gpt"}).status_code == 200
    assert rt.profile == "gpt"


def test_the_ask_door_leaves_the_default_to_the_runtime():
    rt = _RT()
    _client(rt).post("/v1/ask", json={"question": "q"})
    assert rt.profile is None


def test_the_stream_door_passes_the_profile():
    rt = _RT()
    _client(rt).post("/v1/chat", json={"question": "q", "profile": "gpt"})
    assert rt.profile == "gpt"


def test_the_mcp_door_passes_the_profile():
    rt = _RT()
    _db_read(rt, _identity(), "q", profile="gpt")
    assert rt.profile == "gpt"


def test_the_cli_door_passes_the_profile(monkeypatch, capsys):
    rt = _RT()
    monkeypatch.setattr("mnemiq.cli.build_runtime", lambda settings: rt)
    assert main(["ask", "q", "--profile", "gpt"]) == 0
    assert rt.profile == "gpt"


def test_the_ask_door_refuses_with_only_what_this_caller_may_pick():
    response = _client(_RT(raises=UnknownProfile(["open", "qwen"]))).post(
        "/v1/ask", json={"question": "q", "profile": "secret-profile"})
    assert response.status_code == 422
    assert "open" in response.json()["detail"] and "qwen" in response.json()["detail"]
    assert "secret-profile" not in response.text


def test_the_stream_door_says_a_fixed_sentence():
    response = _client(_RT(raises=UnknownProfile(["open"]))).post(
        "/v1/chat", json={"question": "q", "profile": "secret-profile"})
    last = _events(response.text)[-1]
    assert last["type"] == "RUN_ERROR"
    assert last["message"] == "That profile is not one this deployment offers you."
    assert "secret-profile" not in response.text


def test_the_mcp_door_names_only_what_this_caller_may_pick():
    message = _client_message(UnknownProfile(["open", "qwen"]))
    assert "open" in message and "qwen" in message


def test_the_mcp_door_says_a_bad_file_is_the_operators():
    """A profile file that will not load is not a missing snapshot, and "run enrich" would send them the wrong way."""
    from mnemiq.profiles import ProfileFileError

    try:
        try:
            raise ProfileFileError("profile 'x' is wrong")
        except ProfileFileError as cause:
            raise SnapshotMissing(str(cause)) from cause
    except SnapshotMissing as exc:
        message = _client_message(exc)
    assert "enrich" not in message
    assert "profile" in message.lower()


def test_the_mcp_tool_turns_an_unknown_profile_into_its_instruction():
    pytest.importorskip("mcp")
    from mnemiq.mcp.server import _SETUP

    assert issubclass(UnknownProfile, _SETUP)


def test_the_cli_door_says_what_this_caller_may_pick(monkeypatch, capsys):
    monkeypatch.setattr("mnemiq.cli.build_runtime", lambda settings: _RT(raises=UnknownProfile(["open"])))
    assert main(["ask", "q", "--profile", "secret-profile"]) == 1
    err = capsys.readouterr().err
    assert "open" in err
    assert "secret-profile" not in err

"""MCP discovery must not require a database snapshot or provider credentials."""

import os
from pathlib import Path
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_unconfigured_stdio_server_lists_tools_and_survives_tool_errors(tmp_path):
    async def check():
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mnemiq.cli", "serve"],
            cwd=str(tmp_path),
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                "MNEMIQ_STORE_PATH": str(tmp_path / "empty.duckdb"),
            },
        )
        with anyio.fail_after(20):
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    assert {t.name for t in result.tools} == {
                        "db_read", "db_write", "get_schema"
                    }
                    assert not (tmp_path / "empty.duckdb").exists()
                    for name, arguments in [
                        ("get_schema", {}),
                        ("db_read", {"question": "How many rows?"}),
                        ("db_write", {"sql": "DELETE FROM orders"}),
                    ]:
                        result = await session.call_tool(name, arguments)
                        # `is_error` from mcp 2.x, `isError` before it (M121).
                        assert getattr(result, "is_error", getattr(result, "isError", None)) is True
                        assert "enrich" in result.content[0].text.lower()
                    await session.send_ping()

    anyio.run(check)


def test_a_mistyped_mode_reaches_the_client_as_its_instruction(tmp_path):
    """mcp 2.1 and later show a client only `Error executing tool <name>` for anything but ToolError
    (M121), so a setup failure the operator can fix is carried as one -- a mistyped MNEMIQ_MODE
    among them, raised building the runtime on the first call."""
    async def check():
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mnemiq.cli", "serve"],
            cwd=str(tmp_path),
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                "MNEMIQ_STORE_PATH": str(tmp_path / "empty.duckdb"),
                "MNEMIQ_MODE": "thinkin",
            },
        )
        with anyio.fail_after(20):
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("get_schema", {})
                    assert getattr(result, "is_error", getattr(result, "isError", None)) is True
                    assert "thinkin" in result.content[0].text, result.content[0].text

    anyio.run(check)


def test_a_mistyped_mode_on_one_call_reaches_the_client_as_its_instruction():
    """The per-call `mode` of `db_read`, routed after the runtime is built: an `UnknownMode` there
    lists the valid modes, and on mcp 2.1 and later it reached an agent as `Error executing tool db_read`
    unless carried as ToolError (review gate on M121)."""
    from mnemiq.agent.route import UnknownMode
    from mnemiq.contract import IdentityContext
    from mnemiq.mcp.server import _mcp_classes, build_mcp

    class _Runtime:
        def ask(self, question, identity, mode=None):
            raise UnknownMode(f"mode {mode!r} names no mode; valid modes: instant, thinking, deep")

    server = build_mcp(_Runtime(), IdentityContext(principal_id="p", tenant_id="t"))
    _, tool_error = _mcp_classes()

    async def call():
        await server.call_tool("db_read", {"question": "q", "mode": "thinkin"})

    try:
        anyio.run(call)
    except tool_error as exc:
        assert "valid modes" in str(exc), str(exc)
    else:
        raise AssertionError("a mistyped mode was answered")

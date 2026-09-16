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
                        assert result.isError
                        assert "enrich" in result.content[0].text.lower()
                    await session.send_ping()

    anyio.run(check)

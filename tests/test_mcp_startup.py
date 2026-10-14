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
                        # Composed in the MCP layer, never the exception's words: the store path
                        # it carries is configuration (Codex's review of #114).
                        assert str(tmp_path) not in result.content[0].text, result.content[0].text
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
                    text = result.content[0].text
                    # The offered modes, composed from the registry -- not the operator's value
                    # echoed back, which `thinkin` in `thinking` once passed for.
                    assert "Unknown mode" in text and "'deep', 'instant', 'thinking'" in text, text

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
        assert "this deployment offers: ['deep', 'instant', 'thinking']" in str(exc), str(exc)
    else:
        raise AssertionError("a mistyped mode was answered")



def _tool_error_from(raised: Exception) -> str:
    """The text a client gets when building the runtime raises `raised`, via the installed mcp."""
    from mnemiq.contract import IdentityContext
    from mnemiq.mcp.server import _mcp_classes, build_mcp

    def factory():
        raise raised

    server = build_mcp(factory, IdentityContext(principal_id="p", tenant_id="t"))
    _, tool_error = _mcp_classes()

    async def call():
        await server.call_tool("get_schema", {})

    try:
        anyio.run(call)
    except tool_error as exc:
        return str(exc)
    raise AssertionError("the call succeeded")


def test_each_setup_failure_reaches_the_client_as_its_instruction_without_its_details():
    """Codex's review of #114: the setup failures mcp 2.1+ would hide, and the details mcp 1.x
    showed. Each arrives as the instruction this layer composes; none carries the exception's own
    words -- here a distinctive path standing for the store path, manifest ids and the like."""
    from mnemiq.adapters.federated import UnfederatableSource
    from mnemiq.adapters.oracle import DriverMissing
    from mnemiq.adapters.resolve import SourceUnconfigured
    from mnemiq.duckdb_extensions import ExtensionNotInstalled
    from mnemiq.runtime import SnapshotMissing

    secret = "/srv/secret-layout/store.duckdb"
    unconfigured = SnapshotMissing(secret)
    unconfigured.__cause__ = SourceUnconfigured(secret)
    for raised, expected in [
        (SnapshotMissing(secret), "mnemiq enrich"),
        (unconfigured, "source is not configured"),
        (SourceUnconfigured(secret), "source is not configured"),
        (ExtensionNotInstalled(secret), "mnemiq extensions fetch"),
        (UnfederatableSource(secret), "federates a source DuckDB cannot attach"),
        (DriverMissing(secret), "mnemiq[oracle]"),
    ]:
        text = _tool_error_from(raised)
        assert expected in text, (type(raised).__name__, text)
        assert secret not in text, (type(raised).__name__, text)


def test_any_other_failure_reaches_the_client_as_an_internal_error_and_the_log_as_itself(caplog):
    """mcp 1.x put any exception's words on the wire; a DSN in one reached the client. Now the
    client is told only that an internal error was logged, and the log has the exception."""
    import logging

    dsn = "postgresql://svc:hunter2@db.internal/prod"
    with caplog.at_level(logging.WARNING, logger="mnemiq.mcp.server"):
        text = _tool_error_from(RuntimeError(f"could not connect to {dsn}"))
    assert "internal error, recorded in the server log" in text, text
    assert "hunter2" not in text and "db.internal" not in text, text
    assert dsn in caplog.text, "the operator still gets the exception, in the server log"


def test_a_missing_oracle_driver_is_its_own_error():
    """`DriverMissing`, not a bare RuntimeError, so a door can tell it from any other."""
    import sys

    from mnemiq.adapters.oracle import DriverMissing, OracleAdapter

    saved = sys.modules.get("oracledb")
    sys.modules["oracledb"] = None  # an import of it now raises ModuleNotFoundError
    try:
        try:
            OracleAdapter(dsn="db/x", user="u", password="p")
        except DriverMissing as exc:
            assert "mnemiq[oracle]" in str(exc)
        else:
            raise AssertionError("constructed without its driver")
    finally:
        if saved is None:
            sys.modules.pop("oracledb", None)
        else:
            sys.modules["oracledb"] = saved

"""What Superset imports when it loads the extension, once per process that builds the app: the web server, workers,
the CLI and the standalone MCP service.

The REST route is registered by importing `api`. The MCP tools register by importing `tools`, which needs the
`fastmcp` extra: in a process without it, `superset_core.mcp.decorators.tool` is a stub that raises
`NotImplementedError`, and an uncaught raise here would fail the whole extension, its REST route included.
"""

import logging

from agenticfabriq.mnemiq import api  # noqa: F401  (registers the token route and the forwarder)

logger = logging.getLogger(__name__)

try:
    from agenticfabriq.mnemiq import tools  # noqa: F401  (registers the MCP tools)
except NotImplementedError:
    logger.warning("mnemiq extension: MCP tools not registered (fastmcp is not installed in this process)")

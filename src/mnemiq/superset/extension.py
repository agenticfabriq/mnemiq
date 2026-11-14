"""The names of the extension's tools, and the one way to call them.

mnemiq's Superset extension registers its tools on Superset's MCP service under `TOOL_PREFIX`. A Superset that does not
have the extension installed does not say so: rc2's MCP service masks the not-found error, and the client receives only
`Internal error in <tool>: An unexpected error occurred. Error ID: <id>. ...`
(`GlobalErrorHandlerMiddleware._handle_error` in `superset/mcp_service/middleware.py`), the same text an unexpected
exception inside an installed tool produces. Superset's default tool search hides tools from `tools/list`, so the list
cannot settle it either. `call_tool` settles it by one probe: `dataset_metrics` with no datasets, which the installed
extension answers with an empty list and never fails on.
"""

from __future__ import annotations

from mnemiq.superset.client import SupersetToolError

__all__ = ["EXTENSION_TOOLS", "ExtensionMissing", "TOOL_PREFIX", "call_tool", "dataset_metrics"]

TOOL_PREFIX = "extensions.agenticfabriq.mnemiq."

# The base name of every tool the extension offers; the single source the client and the test fake build names from.
EXTENSION_TOOLS = ("create_dashboard", "claim_chart", "claim_dataset", "create_chart", "patch_chart",
                   "add_saved_metric", "delete_dataset", "set_layout", "dataset_metrics", "editors_of", "get_layout")

_PROBE = "dataset_metrics"
_MASKED = "An unexpected error occurred"


class ExtensionMissing(RuntimeError):
    """Superset does not offer the extension's tools. mnemiq's own sentence is the message; what Superset said is in
    `.detail` only."""

    def __init__(self, tool: str, *, detail: str = "") -> None:
        super().__init__(f"Superset does not offer mnemiq's extension tool {tool}: the extension is not installed there")
        self.detail = detail


def _is_not_offered(tool: str, failure: SupersetToolError) -> bool:
    """Whether `failure` reads as Superset not having `tool`: the masked internal error that names it, or an explicit
    unknown-tool error."""
    message = failure.detail.removeprefix("Error: ")
    return message.startswith(f"Internal error in {tool}: {_MASKED}") or "unknown tool" in message.lower()


def call_tool(client, base: str, request: dict) -> dict:
    """Call the extension's tool `base` as the client's caller. Raises `ExtensionMissing` when Superset does not offer
    it; any other failure is raised as the client raised it."""
    tool = TOOL_PREFIX + base
    try:
        return client.call(tool, request)
    except SupersetToolError as failure:
        if not _is_not_offered(tool, failure):
            raise
        original = failure
    probe = TOOL_PREFIX + _PROBE
    try:
        client.call(probe, {"dataset_ids": []})
    except SupersetToolError as failure:
        if _is_not_offered(probe, failure):
            raise ExtensionMissing(tool, detail=original.detail) from None
    raise original


def dataset_metrics(client, dataset_ids: tuple[int, ...]) -> list[dict]:
    """The extension's saved metrics of these datasets, each `{dataset_id, metrics: [...]}`; a dataset the caller cannot
    access is absent."""
    answer = call_tool(client, "dataset_metrics", {"dataset_ids": list(dataset_ids)})
    return list(answer.get("datasets") or [])

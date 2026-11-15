"""The names of the extension's tools, and the one way to call them.

mnemiq's Superset extension registers its tools on Superset's MCP service under `TOOL_PREFIX`. A Superset that does not
have the extension installed does not say so: rc2's MCP service masks the not-found error, and the client receives only
`Internal error in <tool>: An unexpected error occurred. Error ID: <id>. ...`
(`GlobalErrorHandlerMiddleware._handle_error` in `superset/mcp_service/middleware.py`), the same text an unexpected
exception inside an installed tool produces. Superset's default tool search hides tools from `tools/list`, so the list
cannot settle it either. `require_extension` settles it by one call of a probe: `dataset_metrics` with no datasets, which
the installed extension answers with an empty list and never fails on. `call_tool` asks it once a call fails that way.
"""

from __future__ import annotations

from mnemiq.superset.client import SupersetToolError

__all__ = ["DATASET_METRICS_BATCH", "EXTENSION_TOOLS", "ExtensionMissing", "TOOL_PREFIX", "call_tool",
           "dashboard_editorship", "dataset_metrics", "require_extension"]

TOOL_PREFIX = "extensions.agenticfabriq.mnemiq."

# The base name of every tool the extension offers; the single source the client and the test fake build names from.
EXTENSION_TOOLS = ("create_dashboard", "claim_chart", "claim_dataset", "create_chart", "patch_chart",
                   "add_saved_metric", "delete_dataset", "set_layout", "dataset_metrics", "editors_of", "get_layout")

# The most dataset ids one `dataset_metrics` call takes: the extension refuses a request naming more
# (`reads.MAX_DATASETS` in the extension).
DATASET_METRICS_BATCH = 20

# The probe, which the installed extension answers with an empty list and never fails on.
_PROBE = ("dataset_metrics", {"dataset_ids": []})
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


def require_extension(client) -> None:
    """Return when Superset offers the extension's tools, as the client's caller sees them; raise `ExtensionMissing`
    when it does not. One call, of the probe. Any other failure is raised as the client raised it."""
    probe, request = TOOL_PREFIX + _PROBE[0], _PROBE[1]
    try:
        client.call(probe, dict(request))
    except SupersetToolError as failure:
        if _is_not_offered(probe, failure):
            raise ExtensionMissing(probe, detail=failure.detail) from None
        raise


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
    try:
        require_extension(client)
    except ExtensionMissing:
        raise ExtensionMissing(tool, detail=original.detail) from None
    except SupersetToolError:
        pass  # the probe failed for its own reason: the tool's failure is what is raised
    raise original


def dataset_metrics(client, dataset_ids: tuple[int, ...]) -> list[dict]:
    """The extension's saved metrics of these datasets, each `{dataset_id, metrics: [...]}`, asked
    `DATASET_METRICS_BATCH` datasets a call, in order. A dataset the caller cannot access is absent, and so is one
    an answer leaves out for any other reason: a caller reads an absent dataset's metrics as unknown."""
    found: list[dict] = []
    ids = list(dataset_ids)
    for start in range(0, len(ids), DATASET_METRICS_BATCH):
        answer = call_tool(client, "dataset_metrics", {"dataset_ids": ids[start:start + DATASET_METRICS_BATCH]})
        found.extend(answer.get("datasets") or [])
    return found


def dashboard_editorship(client, dashboard_id: int) -> dict | None:
    """Whether the caller edits the dashboard, as the extension's `editors_of` answers it: `{dashboard_id, is_editor,
    editors}` (the editors as labels, for the person's panel only), `{dashboard_id, too_large}` for an answer the
    extension could not send whole, or None when the caller may not read the dashboard or it does not exist."""
    answer = call_tool(client, "editors_of", {"dashboard_ids": [dashboard_id]})
    entries = answer.get("dashboards") if isinstance(answer, dict) else None
    if not isinstance(entries, list):
        raise SupersetToolError("the extension's editors_of answered no list of dashboards", detail=str(answer)[:200])
    return next((entry for entry in entries if isinstance(entry, dict) and entry.get("dashboard_id") == dashboard_id),
                None)

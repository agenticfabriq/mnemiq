"""The extension's MCP tools: registration only. What each tool does is its function in `writes` (the eight that change
Superset) or `reads` (the three that only read).

Superset registers each under `extensions.agenticfabriq.mnemiq.<name>` (`superset/core/mcp/core_mcp_injection.py`,
`create_tool_decorator`) and wraps it in `mcp_auth_hook`, which runs it as the caller its token names. Each is gated by
a stock class permission with the method `write` (a read tool: `read`), so Superset's own RBAC (`can_write` or
`can_read` on Chart, Dataset or Dashboard) and the per-resource token scopes (`superset:chart:write` and the like)
govern it, and no role is granted anything
new. A tool takes one `request` object, as Superset's own tools do.

The names are mnemiq's `mnemiq.superset.extension.EXTENSION_TOOLS`; mnemiq calls them by those names.
"""

from __future__ import annotations

from typing import Any

from superset_core.mcp.decorators import tool

from agenticfabriq.mnemiq import reads, writes


@tool(name="create_dashboard", class_permission_name="Dashboard", method_permission_name="write",
      description="Create an unpublished, empty dashboard owned by the caller. request: {title, marker}.")
def create_dashboard(request: dict[str, Any]) -> dict[str, Any]:
    return writes.create_dashboard(request)


@tool(name="claim_chart", class_permission_name="Chart", method_permission_name="write",
      description="Make a chart a dashboard's: the dashboard's editors (with ENABLE_VIEWERS and "
                  "VIEWER_PROMISCUOUS_MODE on, the dataset rule instead), a build marker and its saved query. "
                  "request: {chart_id, dashboard_id, marker}.")
def claim_chart(request: dict[str, Any]) -> dict[str, Any]:
    return writes.claim_chart(request)


@tool(name="claim_dataset", class_permission_name="Dataset", method_permission_name="write",
      description="Make a dataset a dashboard's: a build marker, and its existing editors kept, with the caller and "
                  "only the dashboard's users who can already read it added. "
                  "request: {dataset_id, dashboard_id, marker}.")
def claim_dataset(request: dict[str, Any]) -> dict[str, Any]:
    return writes.claim_dataset(request)


@tool(name="create_chart", class_permission_name="Chart", method_permission_name="write",
      description="Create a funnel or sankey chart for a dashboard, after checking its query runs. "
                  "request: {dataset_id, viz_type, slice_name, params, dashboard_id, marker}.")
def create_chart(request: dict[str, Any]) -> dict[str, Any]:
    return writes.create_chart(request)


@tool(name="patch_chart", class_permission_name="Chart", method_permission_name="write",
      description="Set and unset named params keys of a chart, keeping every other key. "
                  "request: {chart_id, set, unset, expect_changed_on}.")
def patch_chart(request: dict[str, Any]) -> dict[str, Any]:
    return writes.patch_chart(request)


@tool(name="add_saved_metric", class_permission_name="Dataset", method_permission_name="write",
      description="Add one aggregating saved metric to a dataset, keeping its others. "
                  "request: {dataset_id, metric, expect_changed_on}.")
def add_saved_metric(request: dict[str, Any]) -> dict[str, Any]:
    return writes.add_saved_metric(request)


@tool(name="delete_dataset", class_permission_name="Dataset", method_permission_name="write",
      description="Move a dataset a build made for a dashboard to the trash once no chart uses it. "
                  "request: {dataset_id, dashboard_id}.")
def delete_dataset(request: dict[str, Any]) -> dict[str, Any]:
    return writes.delete_dataset(request)


@tool(name="set_layout", class_permission_name="Dashboard", method_permission_name="write",
      description="Write a dashboard's layout and with it its charts, keeping its filters and tabs. "
                  "request: {dashboard_id, positions, add_chart_ids, remove_chart_ids, expect_changed_on}.")
def set_layout(request: dict[str, Any]) -> dict[str, Any]:
    return writes.set_layout(request)


@tool(name="dataset_metrics", class_permission_name="Dataset", method_permission_name="read",
      description="Every saved metric of the datasets the caller may access, with its currency and certification. "
                  "request: {dataset_ids} (at most 20).")
def dataset_metrics(request: dict[str, Any]) -> dict[str, Any]:
    return reads.dataset_metrics(request)


@tool(name="editors_of", class_permission_name="Chart", method_permission_name="read",
      description="The editors' labels of the charts and datasets the caller may read. "
                  "request: {chart_ids, dataset_ids}.")
def editors_of(request: dict[str, Any]) -> dict[str, Any]:
    return reads.editors_of(request)


@tool(name="get_layout", class_permission_name="Dashboard", method_permission_name="read",
      description="A dashboard's stored layout as text, with its changed_on, when the caller may read it. "
                  "request: {dashboard_id}.")
def get_layout(request: dict[str, Any]) -> dict[str, Any]:
    return reads.get_layout(request)

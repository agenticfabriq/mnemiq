"""Where a build's charts go on its dashboard: the layout, in the shape Superset's dashboards keep (`position_json`,
version `v2`), which the extension's `set_layout` writes as the dashboard's `json_metadata.positions`.

Each chart is placed by its family. KPI tiles (a big number, a gauge) come first, four columns each, three to a row;
then trends, six columns each; then breakdowns, six columns each; then tables, the full twelve. Within a group the
charts keep the plan's order, and a row holds as many as fit the grid's twelve columns. Each row is a `ROW` under
`GRID_ID` holding its charts directly, as a dashboard built in Superset holds them, each chart a `CHART` node whose
`meta` names its chart id, its width in grid columns and its height in grid units.

Onto a dashboard that already has a layout, the build's rows come after what is there, which is kept as it is. Either
way every component's `parents`, the path from `ROOT_ID` that Superset's dashboard reads, is rebuilt from where the
component now sits.

Nothing here reads or sends anything.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Mapping, Sequence

from mnemiq.superset.charts import Family

__all__ = ["CHART_HEIGHT", "GRID_COLUMNS", "KPI_FAMILIES", "TABLE_FAMILIES", "TREND_FAMILIES", "layout_chart_ids",
           "layout_positions", "layout_without"]

# Superset's dashboard grid (`GRID_COLUMN_COUNT` in `superset-frontend/src/dashboard/util/constants.ts`), and the
# height rc2's MCP gives a chart it lays out (`generate_dashboard`, `superset/mcp_service/dashboard/tool/`).
GRID_COLUMNS = 12
CHART_HEIGHT = 50

KPI_FAMILIES = frozenset({Family.BIG_NUMBER, Family.GAUGE})
TREND_FAMILIES = frozenset({Family.LINE, Family.BAR, Family.AREA, Family.SCATTER, Family.MIXED, Family.WATERFALL})
TABLE_FAMILIES = frozenset({Family.TABLE, Family.PIVOT, Family.INTERACTIVE_PIVOT, Family.HANDLEBARS})
# Every other family is a breakdown: a pie, a treemap, a funnel, a sankey, a histogram, a box plot.

# Each group in the order the dashboard shows them, with the grid columns each of its charts takes.
_GROUPS = ((KPI_FAMILIES, 4), (TREND_FAMILIES, 6), (None, 6), (TABLE_FAMILIES, 12))

_ROOT, _GRID, _VERSION = "ROOT_ID", "GRID_ID", "DASHBOARD_VERSION_KEY"
_ROW_META = {"background": "BACKGROUND_TRANSPARENT"}


def _group(family: Family) -> int:
    for index, (families, _) in enumerate(_GROUPS):
        if families is not None and family in families:
            return index
    return next(index for index, (families, _) in enumerate(_GROUPS) if families is None)


def _empty() -> dict:
    """An empty dashboard's layout, as Superset's own (`getEmptyLayout`, `superset-frontend/src/dashboard/util/`)."""
    return {_VERSION: "v2", _ROOT: {"type": "ROOT", "id": _ROOT, "children": [_GRID]},
            _GRID: {"type": "GRID", "id": _GRID, "children": [], "parents": [_ROOT]}}


def layout_chart_ids(positions: Mapping) -> set[int]:
    """The chart ids a layout places: every `CHART` node's `meta.chartId`."""
    found: set[int] = set()
    for node in positions.values():
        if isinstance(node, Mapping) and node.get("type") == "CHART":
            chart_id = (node.get("meta") or {}).get("chartId")
            if type(chart_id) is int and chart_id > 0:
                found.add(chart_id)
    return found


def _base(onto: Mapping | None) -> dict:
    """A copy of the layout to add to, or an empty one; refused when it is not one whose grid rows can be added to."""
    if not onto:
        return _empty()
    if not all(isinstance(node, Mapping) for key, node in onto.items() if key != _VERSION):
        raise ValueError("the dashboard's layout holds a node that is not an object")
    root, grid = onto.get(_ROOT), onto.get(_GRID)
    if not isinstance(root, Mapping) or root.get("type") != "ROOT" or _GRID not in (root.get("children") or []):
        raise ValueError("the dashboard's layout has no root holding its grid")
    if not isinstance(grid, Mapping) or grid.get("type") != "GRID" or not isinstance(grid.get("children", []), list):
        raise ValueError("the dashboard's layout has no grid to add rows to")
    return copy.deepcopy(dict(onto))


def _free(positions: dict, wanted: str) -> str:
    """`wanted`, or the first of `wanted-2`, `wanted-3`, ... that no node of the layout is stored under."""
    key, n = wanted, 1
    while key in positions:
        n += 1
        key = f"{wanted}-{n}"
    return key


def _rebuild_parents(positions: dict) -> None:
    """Each node reached from the root, breadth first and once, gets its path from the root as its `parents`. A node
    no walk reaches (a header, a node left over) is left as it is."""
    visited = {_ROOT}
    queue: deque[tuple[str, list[str]]] = deque([(_ROOT, [])])
    while queue:
        key, path = queue.popleft()
        node = positions[key]
        if key != _ROOT:
            node["parents"] = path
        for child in node.get("children") or []:
            if isinstance(child, str) and child not in visited and isinstance(positions.get(child), dict):
                visited.add(child)
                queue.append((child, [*path, key]))


def layout_positions(charts: Sequence[tuple[int, Family]], *, onto: Mapping | None = None) -> dict:
    """The layout placing `charts`, each a chart id and its family, in their groups, onto the layout `onto` when the
    dashboard has one (its nodes kept, the build's rows after them). Refused: a chart id that is not a positive
    integer, one given twice, one the layout already places, and a layout without a root and a grid to add to."""
    ids = [chart_id for chart_id, _ in charts]
    if not all(type(chart_id) is int and chart_id > 0 for chart_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("a layout places distinct charts, each by its positive id")
    if not all(isinstance(family, Family) for _, family in charts):
        raise ValueError("a layout places each chart by its family")
    positions = _base(onto)
    if placed := set(ids) & layout_chart_ids(positions):
        raise ValueError(f"the dashboard's layout already places {sorted(placed)}")
    grid = positions[_GRID]
    grid["children"] = list(grid.get("children") or [])
    for index, (_, width) in enumerate(_GROUPS):
        group = [chart_id for chart_id, family in charts if _group(family) == index]
        per_row = GRID_COLUMNS // width
        for start in range(0, len(group), per_row):
            row_ids = group[start:start + per_row]
            row_key = _free(positions, f"ROW-mnemiq-{row_ids[0]}")
            chart_keys = []
            for chart_id in row_ids:
                chart_key = _free(positions, f"CHART-{chart_id}")
                positions[chart_key] = {"type": "CHART", "id": chart_key, "children": [],
                                        "meta": {"chartId": chart_id, "width": width, "height": CHART_HEIGHT}}
                chart_keys.append(chart_key)
            positions[row_key] = {"type": "ROW", "id": row_key, "children": chart_keys, "meta": dict(_ROW_META)}
            grid["children"].append(row_key)
    _rebuild_parents(positions)
    return positions


# The nodes a layout keeps when they are left empty: the root, the grid, and the tabs, which rc2 keeps under their ids
# (`process_tab_diff`, and the extension's `check_positions`, refuse a layout that drops one).
_KEPT_EMPTY = frozenset({"ROOT", "GRID", "TABS", "TAB"})


def layout_without(positions: Mapping, chart_ids) -> dict:
    """`positions` without the charts `chart_ids`: each one's `CHART` node gone from the layout and from its parent's
    children, then every row or column left with no child gone too, as Superset's editor drops an empty row. Every
    other node is kept as it is, and each node's `parents` is rebuilt from where it now sits."""
    removing = set(chart_ids)
    out = copy.deepcopy(dict(positions))
    gone = {key for key, node in out.items() if isinstance(node, Mapping) and node.get("type") == "CHART"
            and (node.get("meta") or {}).get("chartId") in removing}
    while gone:
        for key in gone:
            out.pop(key, None)
        for node in out.values():
            if isinstance(node, dict) and isinstance(node.get("children"), list):
                node["children"] = [child for child in node["children"] if child not in gone]
        gone = {key for key, node in out.items() if isinstance(node, dict) and node.get("type") not in _KEPT_EMPTY
                and node.get("type") in ("ROW", "COLUMN") and not node.get("children")}
    if _ROOT in out:
        _rebuild_parents(out)
    return out

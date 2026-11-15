"""A build's dashboard layout: where each chart goes, by its family, in the `position_json` shape Superset's dashboards
store, and how a build's charts are added to a layout a dashboard already has.

The layout is checked by the extension's own `check_positions`, the rule `set_layout` applies before it writes, so a
layout these tests pass is one the extension takes.
"""

from __future__ import annotations

import copy

import pytest

from mnemiq.superset.charts import Family
from mnemiq.superset.layout import (
    CHART_HEIGHT,
    GRID_COLUMNS,
    KPI_FAMILIES,
    TABLE_FAMILIES,
    TREND_FAMILIES,
    layout_chart_ids,
    layout_positions,
)
from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import params as extension  # noqa: E402

MIXED_ORDER = [(11, Family.TABLE), (12, Family.LINE), (13, Family.BIG_NUMBER), (14, Family.PIE), (15, Family.GAUGE),
               (16, Family.BAR), (17, Family.TREEMAP), (18, Family.PIVOT)]


def _rows(positions: dict) -> list[list[tuple[int, int]]]:
    """The grid's rows, top to bottom, each as its charts' (id, width), left to right."""
    rows = []
    for key in positions["GRID_ID"]["children"]:
        row = positions[key]
        assert row["type"] == "ROW", row
        rows.append([(positions[c]["meta"]["chartId"], positions[c]["meta"]["width"]) for c in row["children"]])
    return rows


def test_kpi_tiles_lead_then_trends_breakdowns_and_tables():
    """KPI tiles (big number, gauge) fill the first row at four columns each, then trends at six, breakdowns at six,
    and tables at twelve, each group in the plan's order."""
    rows = _rows(layout_positions(MIXED_ORDER))
    assert rows == [[(13, 4), (15, 4)], [(12, 6), (16, 6)], [(14, 6), (17, 6)], [(11, 12)], [(18, 12)]]


def test_a_row_never_holds_more_than_the_grid():
    """Four KPI tiles take a row of three and a row of one; three trends a row of two and one of one."""
    charts = [(i, Family.BIG_NUMBER) for i in (1, 2, 3, 4)] + [(i, Family.LINE) for i in (5, 6, 7)]
    rows = _rows(layout_positions(charts))
    assert rows == [[(1, 4), (2, 4), (3, 4)], [(4, 4)], [(5, 6), (6, 6)], [(7, 6)]]
    assert all(sum(width for _, width in row) <= GRID_COLUMNS for row in rows)


def test_every_family_has_one_place():
    places = [KPI_FAMILIES, TREND_FAMILIES, TABLE_FAMILIES]
    for family in Family:
        assert sum(family in place for place in places) <= 1, family
    assert KPI_FAMILIES == {Family.BIG_NUMBER, Family.GAUGE}
    assert {Family.LINE, Family.BAR, Family.AREA, Family.SCATTER, Family.MIXED} <= TREND_FAMILIES
    assert {Family.TABLE, Family.PIVOT} <= TABLE_FAMILIES


def _ancestry(positions: dict) -> dict[str, list[str]]:
    """Each node's path from the root, by walking the children from `ROOT_ID`."""
    found = {"ROOT_ID": []}
    queue = ["ROOT_ID"]
    while queue:
        key = queue.pop(0)
        for child in positions[key].get("children", []):
            found[child] = [*found[key], key]
            queue.append(child)
    return found


def test_every_component_has_its_parents():
    """Each component's `parents` is its path from the root, and every node is reached from the root."""
    positions = layout_positions(MIXED_ORDER)
    ancestry = _ancestry(positions)
    nodes = {key for key in positions if key != "DASHBOARD_VERSION_KEY"}
    assert set(ancestry) == nodes
    for key in nodes - {"ROOT_ID"}:
        assert positions[key]["parents"] == ancestry[key], key
    assert positions["DASHBOARD_VERSION_KEY"] == "v2"
    for key in nodes:
        assert positions[key]["id"] == key
    chart = positions["CHART-13"]
    assert chart["type"] == "CHART" and chart["children"] == []
    assert chart["meta"] == {"chartId": 13, "width": 4, "height": CHART_HEIGHT}


def test_a_new_layout_is_one_the_extension_takes():
    """The extension's own check of a layout (`check_positions`) passes it: it holds exactly the charts added."""
    positions = layout_positions(MIXED_ORDER)
    ids = [chart_id for chart_id, _ in MIXED_ORDER]
    assert layout_chart_ids(positions) == set(ids)
    assert extension.check_positions(positions, current_positions={}, current_chart_ids=[], add_chart_ids=ids) is None


def _existing() -> dict:
    """A layout made by hand: a header, a tab group with one tab holding a row with chart 7, and parents gone stale."""
    return {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
        "HEADER_ID": {"type": "HEADER", "id": "HEADER_ID", "meta": {"text": "Sales"}},
        "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": ["TABS-a"], "parents": ["ROOT_ID"]},
        "TABS-a": {"type": "TABS", "id": "TABS-a", "children": ["TAB-a"], "parents": ["ROOT_ID"], "meta": {}},
        "TAB-a": {"type": "TAB", "id": "TAB-a", "children": ["ROW-a"], "parents": [],
                  "meta": {"text": "Overview"}},
        "ROW-a": {"type": "ROW", "id": "ROW-a", "children": ["CHART-a"], "parents": ["nowhere"],
                  "meta": {"background": "BACKGROUND_TRANSPARENT"}},
        "CHART-a": {"type": "CHART", "id": "CHART-a", "children": [], "parents": [],
                    "meta": {"chartId": 7, "width": 5, "height": 40, "sliceName": "By hand"}},
    }


def test_charts_are_added_to_a_layout_the_dashboard_has():
    """Onto an existing layout, the build's rows come after what is there, which is kept as it was, every component's
    parents rebuilt; the extension's check passes it with the dashboard's chart and the added ones, no tab dropped."""
    existing = _existing()
    positions = layout_positions([(21, Family.BIG_NUMBER), (22, Family.LINE)], onto=existing)
    assert positions["GRID_ID"]["children"][0] == "TABS-a"
    assert _rows({**positions, "GRID_ID": {**positions["GRID_ID"],
                                           "children": positions["GRID_ID"]["children"][1:]}}) == [[(21, 4)], [(22, 6)]]
    for key, node in existing.items():
        if key in ("DASHBOARD_VERSION_KEY", "GRID_ID"):
            continue
        kept = {k: v for k, v in positions[key].items() if k != "parents"}
        assert kept == {k: v for k, v in node.items() if k != "parents"}, key
    ancestry = _ancestry(positions)
    for key in set(ancestry) - {"ROOT_ID"}:
        assert positions[key]["parents"] == ancestry[key], key
    assert positions["CHART-a"]["parents"] == ["ROOT_ID", "GRID_ID", "TABS-a", "TAB-a", "ROW-a"]
    assert layout_chart_ids(positions) == {7, 21, 22}
    assert extension.check_positions(positions, current_positions=existing, current_chart_ids=[7],
                                     add_chart_ids=[21, 22]) is None
    assert existing == _existing()  # the layout given is not changed


def test_an_empty_layout_is_a_new_one():
    """A dashboard shell's layout is empty: the build's layout is then a new one."""
    for empty in (None, {}):
        assert layout_positions([(1, Family.PIE)], onto=empty) == layout_positions([(1, Family.PIE)])


@pytest.mark.parametrize("onto", [
    {"ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": []}},
    {"GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": []}},
    {"ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
     "GRID_ID": {"type": "ROW", "id": "GRID_ID", "children": []}},
    {"ROOT_ID": "not a node"},
], ids=["no grid", "no root", "a grid that is not one", "a node that is not one"])
def test_a_layout_mnemiq_cannot_add_to_is_refused(onto):
    with pytest.raises(ValueError):
        layout_positions([(1, Family.PIE)], onto=onto)


@pytest.mark.parametrize("charts", [[(1, Family.PIE), (1, Family.LINE)], [(0, Family.PIE)], [(True, Family.PIE)]],
                         ids=["a chart twice", "an id that is no id", "a bool"])
def test_charts_are_distinct_ids(charts):
    with pytest.raises(ValueError):
        layout_positions(charts)


def test_a_chart_already_on_the_layout_is_not_placed_again():
    existing = _existing()
    with pytest.raises(ValueError):
        layout_positions([(7, Family.PIE)], onto=existing)


def test_row_ids_do_not_collide_with_the_layouts_own():
    """A row the build adds never takes the key of a node the layout already has."""
    existing = copy.deepcopy(_existing())
    first = layout_positions([(21, Family.PIE)])
    taken = next(key for key in first if key.startswith("ROW-"))
    existing[taken] = {"type": "ROW", "id": taken, "children": [], "parents": ["ROOT_ID", "GRID_ID"], "meta": {}}
    existing["GRID_ID"]["children"].append(taken)
    positions = layout_positions([(21, Family.PIE)], onto=existing)
    assert positions[taken] == {**existing[taken], "parents": ["ROOT_ID", "GRID_ID"]}
    added = [key for key in positions["GRID_ID"]["children"] if key not in existing["GRID_ID"]["children"]]
    assert len(added) == 1 and added[0] != taken and positions[added[0]]["children"] == ["CHART-21"]

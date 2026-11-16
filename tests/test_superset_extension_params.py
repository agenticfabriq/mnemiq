"""The pure half of the extension's write tools (`agenticfabriq.mnemiq.params`): the query context the extension saves
on every chart the builder makes, the patch merge, the layout check and the layout metadata, and the metric list a
saved-metric add sends. Nothing here imports Superset or Flask.

The expected query shapes are rc2's own, read from its frontend: `buildQueryObject` and `normalizeTimeColumn` in
`superset-frontend/packages/superset-ui-core/src/query/`, and each plugin's `buildQuery.ts`. The params are shaped as
rc2's MCP mappers write them (`superset/mcp_service/chart/chart_utils.py`, `map_*_config`) and as mnemiq's
`extension_chart_request` writes a funnel or a sankey.
"""

import copy
import json

import pytest

from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import params as ext  # noqa: E402

DATASET = 7
SALES = {"expressionType": "SQL", "sqlExpression": "SUM(amount)", "label": "sales"}  # an ad hoc SQL metric
ORDERS = "order_count"  # a saved metric, by name
WEST = {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "region", "operator": "==", "comparator": "West"}
WEST_Q = {"col": "region", "op": "==", "val": "West"}


def no_time_filter(column):
    """The neutral time filter rc2's MCP adds to a temporal x-axis (`_ensure_temporal_adhoc_filter`)."""
    return {"clause": "WHERE", "expressionType": "SIMPLE", "subject": column, "operator": "TEMPORAL_RANGE",
            "comparator": "No filter"}


def no_time_q(column):
    return {"col": column, "op": "TEMPORAL_RANGE", "val": "No filter"}


def axis(column, grain=None):
    """The column `normalizeTimeColumn` makes of a physical x-axis: BASE_AXIS, a column reference, the grain."""
    out = {"columnType": "BASE_AXIS", "sqlExpression": column, "label": column, "expressionType": "SQL",
           "isColumnReference": True}
    if grain:
        out["timeGrain"] = grain
    return out


def grained(column, grain):
    """The column the table and pivot plugins make of a temporal column: BASE_AXIS with the grain, no column
    reference flag (`plugin-chart-table/src/buildQuery.ts`, `plugin-chart-pivot-table/src/plugin/buildQuery.ts`)."""
    return {"timeGrain": grain, "columnType": "BASE_AXIS", "sqlExpression": column, "label": column,
            "expressionType": "SQL"}


# ---------------------------------------------------------------- one chart of each family, as it is saved

LINE = {"viz_type": "echarts_timeseries_line", "metrics": [SALES], "x_axis": "order_date",
        "granularity_sqla": "order_date", "time_grain_sqla": "P1M", "groupby": ["region"],
        "adhoc_filters": [WEST, no_time_filter("order_date")], "row_limit": 10000, "y_axis_format": ",.2f"}
BAR = {"viz_type": "echarts_timeseries_bar", "metrics": [SALES, ORDERS], "x_axis": "order_date",
       "granularity_sqla": "order_date", "time_grain_sqla": "P1D", "adhoc_filters": [no_time_filter("order_date")],
       "row_limit": 500}
AREA = {"viz_type": "echarts_area", "metrics": [SALES], "x_axis": "order_date", "granularity_sqla": "order_date",
        "time_grain_sqla": "P3M", "groupby": ["channel"], "adhoc_filters": [WEST, no_time_filter("order_date")],
        "row_limit": 10000, "stack": "Stack"}
# A non-temporal x-axis: MCP clears the grain and the time column (`configure_temporal_handling`).
SCATTER = {"viz_type": "echarts_timeseries_scatter", "metrics": [SALES], "x_axis": "units",
           "x_axis_sort_series_type": "name", "x_axis_sort_series_ascending": True, "time_grain_sqla": None,
           "granularity_sqla": None, "row_limit": 1000}
MIXED = {"viz_type": "mixed_timeseries", "x_axis": "order_date", "metrics": [SALES], "seriesType": "line",
         "metrics_b": [ORDERS], "seriesTypeB": "bar", "granularity_sqla": "order_date", "time_grain_sqla": "P1M",
         "groupby": ["region"], "groupby_b": ["channel"], "row_limit": 10000, "adhoc_filters": [WEST]}
TABLE = {"viz_type": "table", "all_columns": ["region"], "metrics": [SALES, ORDERS], "groupby": ["region"],
         "query_mode": "aggregate", "adhoc_filters": [WEST], "row_limit": 1000,
         "column_config": {"sales": {"d3NumberFormat": ",.2f"}}}
PIE = {"viz_type": "pie", "groupby": ["region"], "metric": SALES, "color_scheme": "supersetColors",
       "show_labels": True, "number_format": ",.2f", "sort_by_metric": True, "row_limit": 100, "donut": False,
       "adhoc_filters": [WEST]}
BIG_NUMBER = {"viz_type": "big_number_total", "metric": ORDERS, "y_axis_format": ",d",
              "adhoc_filters": [WEST, no_time_filter("order_date")]}
GAUGE = {"viz_type": "gauge_chart", "groupby": [], "metric": SALES, "sort_by_metric": True, "row_limit": 10,
         "min_val": None, "max_val": None, "number_format": ",.2f", "adhoc_filters": [WEST]}
PIVOT = {"viz_type": "pivot_table_v2", "groupbyRows": ["region"], "groupbyColumns": ["channel"], "metrics": [SALES],
         "aggregateFunction": "Sum", "rowTotals": True, "colTotals": True, "transposePivot": False,
         "combineMetric": False, "valueFormat": ",.2f", "metricsLayout": "COLUMNS", "rowOrder": "key_a_to_z",
         "colOrder": "key_a_to_z", "row_limit": 10000, "adhoc_filters": [WEST]}
TREEMAP = {"viz_type": "treemap_v2", "groupby": ["region", "channel"], "metric": SALES, "sort_by_metric": True,
           "row_limit": 10000, "color_scheme": "supersetColors", "adhoc_filters": [WEST]}
WATERFALL = {"viz_type": "waterfall", "x_axis": "order_date", "groupby": ["region"], "metric": SALES,
             "show_total": True, "row_limit": 10000, "time_grain_sqla": "P1M", "granularity_sqla": "order_date",
             "y_axis_format": ",.2f", "adhoc_filters": [WEST]}
FUNNEL = {"viz_type": "funnel", "datasource": f"{DATASET}__table", "metric": SALES, "groupby": ["stage"],
          "number_format": ",.2f", "currency_format": {"symbol": "USD", "symbolPosition": "prefix"},
          "adhoc_filters": [WEST], "row_limit": 10}
SANKEY = {"viz_type": "sankey_v2", "datasource": f"{DATASET}__table", "metric": ORDERS, "source": "from_stage",
          "target": "to_stage", "adhoc_filters": [WEST], "row_limit": 100}


def q(columns, metrics, filters, *, grain=None, row_limit=None, orderby=None, granularity=None):
    return {"columns": columns, "metrics": metrics, "filters": filters, "grain": grain, "row_limit": row_limit,
            "orderby": orderby, "granularity": granularity}


# Per family: the saved params, and per query what rc2's plugin would send, short of post-processing, totals and
# grouping sets.
CASES = {
    "echarts_timeseries_line": (LINE, [q([axis("order_date", "P1M"), "region"], [SALES],
                                         [WEST_Q, no_time_q("order_date")], grain="P1M", row_limit=10000,
                                         orderby=[[SALES, False]], granularity="order_date")]),
    "echarts_timeseries_bar": (BAR, [q([axis("order_date", "P1D")], [SALES, ORDERS], [no_time_q("order_date")],
                                       grain="P1D", row_limit=500, orderby=[[SALES, False]],
                                       granularity="order_date")]),
    "echarts_area": (AREA, [q([axis("order_date", "P3M"), "channel"], [SALES], [WEST_Q, no_time_q("order_date")],
                              grain="P3M", row_limit=10000, orderby=[[SALES, False]], granularity="order_date")]),
    "echarts_timeseries_scatter": (SCATTER, [q([axis("units")], [SALES], [], row_limit=1000,
                                               orderby=[[SALES, False]])]),
    "mixed_timeseries": (MIXED, [
        q([axis("order_date", "P1M"), "region"], [SALES], [WEST_Q], grain="P1M", row_limit=10000,
          orderby=[[SALES, False]], granularity="order_date"),
        q([axis("order_date", "P1M"), "channel"], [ORDERS], [WEST_Q], grain="P1M", row_limit=10000,
          orderby=[[ORDERS, False]], granularity="order_date")]),
    "table": (TABLE, [q(["region"], [SALES, ORDERS], [WEST_Q], row_limit=1000, orderby=[[SALES, False]])]),
    "pie": (PIE, [q(["region"], [SALES], [WEST_Q], row_limit=100, orderby=[[SALES, False]])]),
    "big_number_total": (BIG_NUMBER, [q([], [ORDERS], [WEST_Q, no_time_q("order_date")])]),
    "gauge_chart": (GAUGE, [q([], [SALES], [WEST_Q], row_limit=10, orderby=[[SALES, False]])]),
    "pivot_table_v2": (PIVOT, [q(["region", "channel"], [SALES], [WEST_Q], row_limit=10000,
                                 orderby=[[SALES, False]])]),
    "treemap_v2": (TREEMAP, [q(["region", "channel"], [SALES], [WEST_Q], row_limit=10000,
                               orderby=[[SALES, False], ["region", True], ["channel", True]])]),
    "waterfall": (WATERFALL, [q([axis("order_date", "P1M"), "region"], [SALES], [WEST_Q], grain="P1M",
                                row_limit=10000, orderby=[["order_date", True], ["region", True]],
                                granularity="order_date")]),
    "funnel": (FUNNEL, [q(["stage"], [SALES], [WEST_Q], row_limit=10)]),
    "sankey_v2": (SANKEY, [q(["from_stage", "to_stage"], [ORDERS], [WEST_Q], row_limit=100,
                             orderby=[["from_stage", True], ["to_stage", True]])]),
}

# The fourteen families the builder makes and checks, by name, so the module's set cannot lose one without this
# list changing too.
FOURTEEN = {"echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area", "echarts_timeseries_scatter",
               "mixed_timeseries", "table", "pie", "big_number_total", "gauge_chart", "pivot_table_v2", "treemap_v2",
               "waterfall", "funnel", "sankey_v2"}
# The types a person reaches by changing a line chart's type by hand, which rc2 builds with the very `buildQuery` the
# line chart uses (`plugin-chart-echarts/src/Timeseries/buildQuery.ts`, imported by `Timeseries/index.ts`,
# `Timeseries/Step/index.ts` and `Timeseries/Regular/SmoothLine/index.ts`; their keys are `VizType.Timeseries`,
# `VizType.Step` and `VizType.SmoothLine`).
LINE_LIKE = {"echarts_timeseries", "echarts_timeseries_step", "echarts_timeseries_smooth"}
# Each queries as the line chart does.
CASES.update({viz_type: ({**LINE, "viz_type": viz_type}, CASES["echarts_timeseries_line"][1]) for viz_type in LINE_LIKE})

BOX_PLOT = {"viz_type": "box_plot", "columns": ["region"], "groupby": [], "metrics": [SALES],
            "whiskerOptions": "Tukey", "row_limit": 10000}
NOT_BUILT = {
    "box_plot": BOX_PLOT,
    "histogram_v2": {"viz_type": "histogram_v2", "column": "amount", "groupby": [], "bins": 10, "row_limit": 10000},
    "handlebars": {"viz_type": "handlebars", "query_mode": "aggregate", "metrics": [SALES], "groupby": ["region"],
                   "handlebarsTemplate": "<p>{{data}}</p>"},
    "ag-grid-pivot-table": {"viz_type": "ag-grid-pivot-table", "groupby": ["region"], "metrics": [SALES]},
    "big_number": {"viz_type": "big_number", "metric": ORDERS, "granularity_sqla": "order_date",
                   "time_grain_sqla": "P1M", "show_trend_line": True},
    "word_cloud": {"viz_type": "word_cloud", "series": "region", "metric": SALES},
}


def test_built_families_are_the_fourteen_and_the_line_like_types():
    assert ext.BUILT_FAMILIES == frozenset(FOURTEEN | LINE_LIKE)
    assert set(CASES) == FOURTEEN | LINE_LIKE


def test_a_type_changed_by_hand_to_step_or_smooth_keeps_the_grain():
    """A line chart a person turns into a step, smooth or generic time-series chart keeps its query: the next patch
    regenerates the same query context, the grain on its x-axis included, rather than clearing it."""
    line = ext.query_context("echarts_timeseries_line", LINE, DATASET)
    for viz_type in sorted(LINE_LIKE):
        changed = ext.patch("echarts_timeseries_line", LINE, {"viz_type": viz_type}, [], DATASET)
        assert changed.viz_type == viz_type
        assert changed.query_context["queries"] == line["queries"]
        again = ext.patch(viz_type, changed.params, {"row_limit": 50}, [], DATASET)
        (axis_column, *_) = again.query_context["queries"][0]["columns"]
        assert axis_column == axis("order_date", "P1M") and again.query_context["queries"][0]["row_limit"] == 50


@pytest.mark.parametrize("viz_type", sorted(CASES))
def test_query_context_per_family(viz_type):
    params, expected = CASES[viz_type]
    found = ext.query_context(viz_type, params, DATASET)
    assert found["datasource"] == {"id": DATASET, "type": "table"}
    assert found["result_type"] == "full" and found["result_format"] == "json" and found["force"] is False
    assert len(found["queries"]) == len(expected)
    for query, want in zip(found["queries"], expected):
        assert "post_processing" not in query
        assert "grouping_sets" not in query and "time_offsets" not in query and "is_rowcount" not in query
        assert query["columns"] == want["columns"]
        assert query["metrics"] == want["metrics"]
        assert query["filters"] == want["filters"]
        assert query["extras"].get("time_grain_sqla") == want["grain"]
        assert query.get("row_limit") == want["row_limit"]
        assert query.get("orderby") == want["orderby"]
        assert query.get("granularity") == want["granularity"]
    # The JSON rc2 stores and later loads.
    assert json.loads(json.dumps(found)) == found


@pytest.mark.parametrize("viz_type", sorted(NOT_BUILT))
def test_query_context_is_none_outside_the_built_families(viz_type):
    assert ext.query_context(viz_type, NOT_BUILT[viz_type], DATASET) is None


def test_raw_mode_table_has_no_query_context():
    """A table gets a query context in aggregate mode only."""
    raw = {"viz_type": "table", "all_columns": ["region", "amount"], "columns": ["region", "amount"],
           "query_mode": "raw", "include_time": False, "order_desc": True, "row_limit": 1000}
    assert ext.query_context("table", raw, DATASET) is None
    # No `query_mode` and raw columns: rc2's `getQueryMode` reads that as raw too.
    assert ext.query_context("table", {"viz_type": "table", "all_columns": ["region"]}, DATASET) is None


def test_funnel_query_context():
    """One plain query of `columns` and `metrics`, the shape `buildQueryContext` makes; nothing else."""
    found = ext.query_context("funnel", FUNNEL, DATASET)
    assert found == {
        "datasource": {"id": DATASET, "type": "table"},
        "force": False,
        "queries": [{"columns": ["stage"], "metrics": [SALES], "filters": [WEST_Q],
                     "extras": {"having": "", "where": ""}, "row_limit": 10, "series_limit": 0,
                     "group_others_when_limit_reached": False, "order_desc": True}],
        "form_data": FUNNEL,
        "result_format": "json",
        "result_type": "full",
    }
    # Sorting by the metric orders by it, descending (`Funnel/buildQuery.ts`).
    sorted_funnel = ext.query_context("funnel", {**FUNNEL, "sort_by_metric": True}, DATASET)
    assert sorted_funnel["queries"][0]["orderby"] == [[SALES, False]]


def test_sankey_query_context():
    found = ext.query_context("sankey_v2", SANKEY, DATASET)
    assert found == {
        "datasource": {"id": DATASET, "type": "table"},
        "force": False,
        "queries": [{"columns": ["from_stage", "to_stage"], "metrics": [ORDERS], "filters": [WEST_Q],
                     "extras": {"having": "", "where": ""}, "row_limit": 100,
                     "orderby": [["from_stage", True], ["to_stage", True]], "series_limit": 0,
                     "group_others_when_limit_reached": False, "order_desc": True}],
        "form_data": SANKEY,
        "result_format": "json",
        "result_type": "full",
    }
    # `Sankey/buildQuery.ts` orders only under a row limit, and puts the metric first when sorting by it.
    unlimited = {k: v for k, v in SANKEY.items() if k != "row_limit"}
    assert "orderby" not in ext.query_context("sankey_v2", unlimited, DATASET)["queries"][0]
    by_metric = ext.query_context("sankey_v2", {**SANKEY, "sort_by_metric": True}, DATASET)
    assert by_metric["queries"][0]["orderby"] == [[ORDERS, False], ["from_stage", True], ["to_stage", True]]


def test_query_context_does_not_share_objects_with_params():
    before = copy.deepcopy(LINE)
    found = ext.query_context("echarts_timeseries_line", LINE, DATASET)
    found["queries"][0]["metrics"][0]["label"] = "changed"
    found["form_data"]["groupby"].append("changed")
    assert LINE == before


def test_free_form_and_having_filters_go_where_rc2_puts_them():
    """`processFilters`: a SIMPLE filter becomes `{col, op, val}` (no `val` for IS NULL); a SQL filter joins
    `extras.where` or `extras.having` in parentheses, a `--` comment closed by a newline; a SIMPLE HAVING filter is
    dropped, as rc2's frontend drops it."""
    params = {**PIE, "adhoc_filters": [
        WEST,
        {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "channel", "operator": "IS NULL"},
        {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "tier", "operator": "IN", "comparator": [1, 2]},
        {"clause": "WHERE", "expressionType": "SQL", "sqlExpression": "amount > 0 -- positive only"},
        {"clause": "WHERE", "expressionType": "SQL", "sqlExpression": "units < 9"},
        {"clause": "HAVING", "expressionType": "SQL", "sqlExpression": "SUM(amount) > 10"},
        {"clause": "HAVING", "expressionType": "SIMPLE", "subject": "sales", "operator": ">", "comparator": 5},
    ]}
    query = ext.query_context("pie", params, DATASET)["queries"][0]
    assert query["filters"] == [WEST_Q, {"col": "channel", "op": "IS NULL"},
                                {"col": "tier", "op": "IN", "val": [1, 2]}]
    assert query["extras"] == {"where": "(amount > 0 -- positive only\n) AND (units < 9)",
                               "having": "(SUM(amount) > 10)"}


def test_an_adhoc_x_axis_keeps_its_own_keys():
    custom = {"expressionType": "SQL", "label": "shipped", "sqlExpression": "order_date + interval '2 day'"}
    params = {**LINE, "x_axis": custom, "groupby": []}
    query = ext.query_context("echarts_timeseries_line", params, DATASET)["queries"][0]
    assert query["columns"] == [{"timeGrain": "P1M", "columnType": "BASE_AXIS", **custom}]
    assert "is_timeseries" not in query


def test_table_and_pivot_apply_the_grain_to_a_temporal_column():
    """A grain on a table or a pivot applies to a column Explore marked temporal: the first one in a table, moved to
    the front; every one in a pivot."""
    table = {**TABLE, "groupby": ["region", "order_date", "ship_date"], "time_grain_sqla": "P1W",
             "temporal_columns_lookup": {"order_date": True, "ship_date": True, "region": False}}
    query = ext.query_context("table", table, DATASET)["queries"][0]
    assert query["columns"] == [grained("order_date", "P1W"), "region", "ship_date"]
    pivot = {**PIVOT, "groupbyRows": ["order_date"], "groupbyColumns": ["channel", "ship_date"],
             "time_grain_sqla": "P1M", "temporal_columns_lookup": {"order_date": True, "ship_date": True}}
    query = ext.query_context("pivot_table_v2", pivot, DATASET)["queries"][0]
    assert query["columns"] == [grained("order_date", "P1M"), "channel", grained("ship_date", "P1M")]
    # Without a grain the columns stay plain.
    assert ext.query_context("table", {**table, "time_grain_sqla": None}, DATASET)["queries"][0]["columns"] == [
        "region", "order_date", "ship_date"]


def test_pivot_transposed_queries_columns_then_rows():
    query = ext.query_context("pivot_table_v2", {**PIVOT, "transposePivot": True}, DATASET)["queries"][0]
    assert query["columns"] == ["channel", "region"]


def test_table_sort_metric_and_percent_metrics():
    """`plugin-chart-table/src/buildQuery.ts`: a sort-by metric orders the table (ascending unless `order_desc`), and
    a percent metric is queried with the rest; only its share, a post-processing step, is left out."""
    params = {**TABLE, "timeseries_limit_metric": ORDERS, "order_desc": True, "percent_metrics": [SALES, "margin"]}
    query = ext.query_context("table", params, DATASET)["queries"][0]
    assert query["orderby"] == [[ORDERS, False]]
    assert query["metrics"] == [SALES, ORDERS, "margin"]


def test_mixed_secondary_query_reads_its_own_filters():
    params = {**MIXED, "adhoc_filters_b": [{**WEST, "comparator": "East"}]}
    first, second = ext.query_context("mixed_timeseries", params, DATASET)["queries"]
    assert first["filters"] == [WEST_Q]
    assert second["filters"] == [{**WEST_Q, "val": "East"}]
    # The query context's form data is the first query's, as rc2's mixed `buildQuery` returns it.
    found = ext.query_context("mixed_timeseries", params, DATASET)
    assert "metrics_b" not in found["form_data"] and found["form_data"]["metrics"] == [SALES]


def test_treemap_orders_only_under_a_row_limit():
    unlimited = {k: v for k, v in TREEMAP.items() if k != "row_limit"}
    assert "orderby" not in ext.query_context("treemap_v2", unlimited, DATASET)["queries"][0]
    assert "orderby" not in ext.query_context("treemap_v2", {**TREEMAP, "row_limit": 0}, DATASET)["queries"][0]


def test_query_context_refuses_a_bad_datasource_id_and_params():
    for bad in (0, -3, True, "7", 7.0, None):
        with pytest.raises(ValueError):
            ext.query_context("pie", PIE, bad)
    with pytest.raises(ValueError):
        ext.query_context("pie", ["not", "a", "mapping"], DATASET)
    # An order the frontend cannot parse fails its `buildQuery` (`extractQueryFields`), so it fails here too.
    with pytest.raises(ValueError):
        ext.query_context("pie", {**PIE, "order_by_cols": ["not json"]}, DATASET)


# ---------------------------------------------------------------- the patch merge

ANNOTATIONS = [{"name": "launch", "annotationType": "EVENT", "sourceType": "NATIVE", "value": 3, "show": True}]
CONDITIONAL = [{"colorScheme": "#ACE1C4", "column": "sales", "operator": ">", "targetValue": 100}]


def test_patch_sets_and_unsets_only_named_keys():
    params = {"viz_type": "pie", "metric": SALES, "groupby": ["region"], "row_limit": 100, "donut": False}
    before = copy.deepcopy(params)
    merged = ext.merge_params(params, {"row_limit": 25, "show_total": True}, ["donut"])
    assert merged == {"viz_type": "pie", "metric": SALES, "groupby": ["region"], "row_limit": 25,
                      "show_total": True}
    assert params == before
    # Unsetting a key that is not there is not an error, and changes nothing else.
    assert ext.merge_params(params, {}, ["absent"]) == params


def test_patch_keeps_unknown_keys():
    params = {**TABLE, "annotation_layers": ANNOTATIONS, "conditional_formatting": CONDITIONAL,
              "some_future_key": {"nested": [1, 2]}}
    merged = ext.merge_params(params, {"row_limit": 50}, [])
    for key in ("annotation_layers", "conditional_formatting", "some_future_key", "column_config", "adhoc_filters"):
        assert merged[key] == params[key]
    assert merged["row_limit"] == 50
    # The merged params do not share objects with the read ones.
    merged["annotation_layers"][0]["show"] = False
    assert params["annotation_layers"][0]["show"] is True


def test_patch_refuses_what_it_cannot_apply_cleanly():
    with pytest.raises(ext.PatchRefused) as both:
        ext.merge_params(PIE, {"row_limit": 5}, ["row_limit"])
    assert both.value.reason == "set_and_unset"
    # The chart's dataset is its row's binding, never a params key a patch moves.
    for to_set, to_unset, reason in (({"datasource": "8__table"}, [], "datasource"), ({}, ["datasource"], "datasource"),
                                     ({"datasource_id": 8}, [], "datasource"),
                                     ({"datasource_type": "query"}, [], "datasource"), ({}, ["viz_type"], "viz_type")):
        with pytest.raises(ext.PatchRefused) as found:
            ext.merge_params(FUNNEL, to_set, to_unset)
        assert found.value.reason == reason
    with pytest.raises(ext.PatchRefused):
        ext.merge_params(PIE, {}, "row_limit")  # a string is not a list of keys
    with pytest.raises(ext.PatchRefused):
        ext.merge_params(PIE, {1: "not a key"}, [])
    # A viz_type change to or from funnel or sankey is refused; another one is applied.
    for old, new in (("funnel", "pie"), ("pie", "funnel"), ("sankey_v2", "funnel"), ("table", "sankey_v2")):
        with pytest.raises(ext.PatchRefused) as found:
            ext.patch(old, {**PIE, "viz_type": old}, {"viz_type": new}, [], DATASET)
        assert found.value.reason == "viz_type"
    changed = ext.patch("echarts_timeseries_line", LINE, {"viz_type": "echarts_timeseries_bar"}, [], DATASET)
    assert changed.viz_type == "echarts_timeseries_bar" and changed.params["viz_type"] == "echarts_timeseries_bar"
    assert changed.query_context == ext.query_context("echarts_timeseries_bar", changed.params, DATASET)


def test_patch_regenerates_sankey_and_funnel_query_context():
    sankey = ext.patch("sankey_v2", SANKEY, {"row_limit": 5}, [], DATASET)
    query = sankey.query_context["queries"][0]
    assert query["row_limit"] == 5 and query["columns"] == ["from_stage", "to_stage"]
    assert query["orderby"] == [["from_stage", True], ["to_stage", True]]
    unlimited = ext.patch("sankey_v2", SANKEY, {}, ["row_limit"], DATASET)
    assert "row_limit" not in unlimited.query_context["queries"][0]
    assert "orderby" not in unlimited.query_context["queries"][0]

    funnel = ext.patch("funnel", FUNNEL, {"groupby": ["stage", "segment"], "sort_by_metric": True}, [], DATASET)
    query = funnel.query_context["queries"][0]
    assert query["columns"] == ["stage", "segment"] and query["orderby"] == [[SALES, False]]
    assert funnel.query_context["form_data"] == funnel.params
    assert funnel.viz_type == "funnel"


@pytest.mark.parametrize("viz_type", sorted(CASES))
def test_patch_regenerates_query_context_for_every_built_family(viz_type):
    params = {**CASES[viz_type][0], "annotation_layers": ANNOTATIONS}
    patched = ext.patch(viz_type, params, {"row_limit": 42}, [], DATASET)
    assert patched.params == {**params, "row_limit": 42}
    assert patched.query_context is not None
    assert patched.query_context == ext.query_context(viz_type, patched.params, DATASET)
    assert all(query["row_limit"] == 42 for query in patched.query_context["queries"])
    assert all("post_processing" not in query for query in patched.query_context["queries"])
    # The annotation layer stays in the params and stays out of the query: rc2 checks a guest's layers against the
    # chart's params (`security/manager.py`, `_annotation_layers_modified`), and the rows do not depend on it.
    assert all("annotation_layers" not in query for query in patched.query_context["queries"])


@pytest.mark.parametrize("viz_type", sorted(NOT_BUILT))
def test_patch_clears_query_context_outside_the_built_families(viz_type):
    patched = ext.patch(viz_type, NOT_BUILT[viz_type], {"row_limit": 42}, [], DATASET)
    assert patched.params["row_limit"] == 42
    assert patched.query_context is None


def test_patch_regenerates_the_grouping_it_changed():
    patched = ext.patch("echarts_timeseries_line", LINE, {"groupby": ["channel"], "time_grain_sqla": "P1W"}, [],
                        DATASET)
    assert patched.query_context["queries"][0]["columns"] == [axis("order_date", "P1W"), "channel"]
    pivot = ext.patch("pivot_table_v2", PIVOT, {"groupbyColumns": []}, [], DATASET)
    assert pivot.query_context["queries"][0]["columns"] == ["region"]


# ---------------------------------------------------------------- the layout check and metadata

def layout(chart_ids, tabs=()):
    """A dashboard layout (`position_json`) holding `chart_ids`, each in its own row; with `tabs`, the rows go in the
    first tab and every tab is kept."""
    nodes = {"DASHBOARD_VERSION_KEY": "v2",
             "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]}}
    holder = "GRID_ID"
    grid_children = []
    if tabs:
        nodes["TABS-1"] = {"type": "TABS", "id": "TABS-1", "children": list(tabs),
                           "parents": ["ROOT_ID", "GRID_ID"]}
        grid_children = ["TABS-1"]
        for tab in tabs:
            nodes[tab] = {"type": "TAB", "id": tab, "children": [], "meta": {"text": tab.lower()},
                          "parents": ["ROOT_ID", "GRID_ID", "TABS-1"]}
        holder = tabs[0]
    nodes["GRID_ID"] = {"type": "GRID", "id": "GRID_ID", "children": grid_children, "parents": ["ROOT_ID"]}
    for chart_id in chart_ids:
        row, chart = f"ROW-{chart_id}", f"CHART-{chart_id}"
        nodes[row] = {"type": "ROW", "id": row, "children": [chart], "meta": {"background": "BACKGROUND_TRANSPARENT"}}
        nodes[chart] = {"type": "CHART", "id": chart, "children": [],
                        "meta": {"chartId": chart_id, "width": 4, "height": 50}}
        nodes[holder]["children"].append(row)
    return nodes


def test_positions_chart_set_must_match():
    current = layout([1, 2])

    def check(positions, add=(), remove=()):
        return ext.check_positions(positions, current_positions=current, current_chart_ids=[1, 2],
                                   add_chart_ids=add, remove_chart_ids=remove)

    assert check(layout([1, 2])) is None
    assert check(layout([2, 1, 3]), add=[3]) is None
    assert check(layout([1, 3]), add=[3], remove=[2]) is None
    # A chart the call did not name is neither dropped nor added.
    assert check(layout([1])) == {"refused": "chart_set", "missing": [2], "unexpected": []}
    assert check(layout([1, 2, 3])) == {"refused": "chart_set", "missing": [], "unexpected": [3]}
    assert check(layout([1, 2]), add=[3]) == {"refused": "chart_set", "missing": [3], "unexpected": []}
    assert check(layout([1, 2]), remove=[2]) == {"refused": "chart_set", "missing": [], "unexpected": [2]}
    assert check(layout([1, 2]), add=[2], remove=[2]) == {"refused": "add_and_remove", "chart_ids": [2]}
    # The chart ids are read as rc2's `set_dash_metadata` reads them: any node's `meta.chartId`. A string or a
    # boolean id, a chart id on a node that is not a chart, or a node without a type is malformed.
    for bad in ({"chartId": "2", "width": 4, "height": 50}, {"chartId": True, "width": 4, "height": 50}):
        positions = layout([1, 2])
        positions["CHART-2"]["meta"] = bad
        assert check(positions) == {"refused": "malformed"}
    positions = layout([1, 2])
    positions["ROW-2"]["meta"]["chartId"] = 2
    assert check(positions) == {"refused": "malformed"}
    positions = layout([1, 2])
    del positions["ROW-1"]["type"]
    assert check(positions) == {"refused": "malformed"}
    positions = layout([1, 2])
    positions["CHART-1"]["meta"] = "not a mapping"
    assert check(positions) == {"refused": "malformed"}
    assert check(["not", "a", "layout"]) == {"refused": "malformed"}


def test_positions_may_not_drop_a_tab():
    current = layout([1, 2], tabs=("TAB-a", "TAB-b"))

    def check(positions):
        return ext.check_positions(positions, current_positions=current, current_chart_ids=[1, 2])

    assert check(layout([1, 2], tabs=("TAB-a", "TAB-b"))) is None
    assert check(layout([1, 2], tabs=("TAB-b", "TAB-a", "TAB-c"))) is None
    assert check(layout([1, 2], tabs=("TAB-a",))) == {"refused": "drops_tab", "tabs": ["TAB-b"]}
    assert check(layout([1, 2])) == {"refused": "drops_tab", "tabs": ["TAB-a", "TAB-b"]}
    # A node kept under the tab's id but no longer a tab drops it too.
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["TAB-b"]["type"] = "ROW"
    assert check(positions) == {"refused": "drops_tab", "tabs": ["TAB-b"]}
    # A dashboard with no layout yet has no tab to drop.
    assert ext.check_positions(layout([5]), current_positions=None, current_chart_ids=[], add_chart_ids=[5]) is None


def test_a_tab_is_kept_only_when_it_stays_reachable_under_its_id():
    """rc2 knows a dashboard's tabs as `Dashboard.tabs` (`superset/models/dashboard.py`) walks them: from `ROOT_ID`,
    through the children of the root, the grid, tab groups and tabs, each TAB node known by its own `id`. A save drops
    every node no walk from `ROOT_ID` reaches (`DashboardDAO.set_dash_metadata` runs `repair_position`,
    `superset/dashboards/layout.py`), and deactivates no report on it. So a tab still in the layout's nodes but no
    longer reached, or reached under another id, is a dropped tab; and a tab node stored under a key that is not its
    id is malformed."""
    current = layout([1, 2], tabs=("TAB-a", "TAB-b"))

    def check(positions):
        return ext.check_positions(positions, current_positions=current, current_chart_ids=[1, 2])

    # Detached: TAB-b's node is kept, and its tab group no longer lists it.
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["TABS-1"]["children"] = ["TAB-a"]
    assert check(positions) == {"refused": "drops_tab", "tabs": ["TAB-b"]}
    # The whole tab group detached from the grid.
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["GRID_ID"]["children"] = []
    assert check(positions) == {"refused": "drops_tab", "tabs": ["TAB-a", "TAB-b"]}
    # The tab group under a row, which the walk does not enter.
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["GRID_ID"]["children"] = ["ROW-x"]
    positions["ROW-x"] = {"type": "ROW", "id": "ROW-x", "children": ["TABS-1"], "meta": {}}
    assert check(positions) == {"refused": "drops_tab", "tabs": ["TAB-a", "TAB-b"]}
    # Re-keyed whole, its parent following: rc2 now knows it as TAB-c.
    assert check(layout([1, 2], tabs=("TAB-a", "TAB-c"))) == {"refused": "drops_tab", "tabs": ["TAB-b"]}
    # Its id changed under the same key, and its id kept under another key: neither is a layout rc2 reads one way.
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["TAB-b"]["id"] = "TAB-c"
    assert check(positions) == {"refused": "malformed"}
    positions = layout([1, 2], tabs=("TAB-a", "TAB-c"))
    positions["TAB-c"]["id"] = "TAB-b"
    assert check(positions) == {"refused": "malformed"}
    # The controls: the tabs reached by another path under the same ids (the tab group moved into a new outer tab),
    # and a node reached twice, are kept tabs.
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["GRID_ID"]["children"] = ["TABS-outer"]
    positions["TABS-outer"] = {"type": "TABS", "id": "TABS-outer", "children": ["TAB-outer"]}
    positions["TAB-outer"] = {"type": "TAB", "id": "TAB-outer", "children": ["TABS-1"], "meta": {"text": "outer"}}
    assert check(positions) is None
    positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    positions["TABS-1"]["children"] = ["TAB-a", "TAB-b", "TAB-a"]
    assert check(positions) is None


def test_an_unreachable_tab_of_the_current_layout_is_not_required():
    """A TAB node the current layout holds but does not reach is no tab rc2 knows (`Dashboard.tabs` does not list it,
    so no report can be on it) and the next save drops it anyway: a layout without it is not refused. The control: the
    same tab, reached, is required."""
    current = layout([1, 2], tabs=("TAB-a", "TAB-b"))
    current["TAB-z"] = {"type": "TAB", "id": "TAB-z", "children": [], "meta": {"text": "z"},
                        "parents": ["ROOT_ID", "GRID_ID", "TABS-1"]}

    def check(positions):
        return ext.check_positions(positions, current_positions=current, current_chart_ids=[1, 2])

    assert check(layout([1, 2], tabs=("TAB-a", "TAB-b"))) is None
    current["TABS-1"]["children"].append("TAB-z")
    assert check(layout([1, 2], tabs=("TAB-a", "TAB-b"))) == {"refused": "drops_tab", "tabs": ["TAB-z"]}


def test_a_node_whose_type_is_not_text_is_malformed():
    """Every layout rc2 writes types a node with text. A `type` that is a list or an object is no layout rc2 reads
    one way, so the new layout is refused as malformed, never failed on (a list or an object cannot be looked up in
    a set of types). A current layout holding such a node is read as `Dashboard.tabs` reads it, comparing the type
    with `==`: the node is passed over without being entered."""
    current = layout([1, 2], tabs=("TAB-a", "TAB-b"))

    def check(positions):
        return ext.check_positions(positions, current_positions=current, current_chart_ids=[1, 2])

    for key in ("ROOT_ID", "GRID_ID", "TABS-1", "TAB-a", "ROW-1", "CHART-1"):
        for bad in (["TABS"], {"type": "TAB"}):
            positions = layout([1, 2], tabs=("TAB-a", "TAB-b"))
            positions[key]["type"] = bad
            assert check(positions) == {"refused": "malformed"}, (key, bad)
    # The current tab group typed with a list: rc2 does not enter it, knows no tab under it, and none is required.
    current["TABS-1"]["type"] = ["TABS"]
    assert check(layout([1, 2])) is None


def _deleted_native_filter_ids(current_metadata, new_metadata):
    """What rc2's `UpdateDashboardCommand.process_native_filter_diff` (`superset/commands/dashboard/update.py`)
    counts as deleted, and so deactivates every report on: the filter ids the new metadata no longer lists."""
    current = {f["id"] for f in (current_metadata.get("native_filter_configuration") or []) if "id" in f}
    new = {f["id"] for f in (new_metadata.get("native_filter_configuration") or []) if "id" in f}
    return current - new


NATIVE_FILTERS = [{"id": "NATIVE_FILTER-region", "name": "Region", "filterType": "filter_select",
                   "targets": [{"datasetId": DATASET, "column": {"name": "region"}}]}]


def test_layout_metadata_keeps_native_filters():
    current = {"native_filter_configuration": NATIVE_FILTERS, "color_scheme": "supersetColors",
               "shared_label_colors": ["West"], "map_label_colors": {"West": "#1FA8C9"},
               "color_scheme_domain": ["#1FA8C9"], "color_namespace": "ns", "refresh_frequency": 0,
               "chart_configuration": {"1": {"id": 1}}, "positions": {"stale": True}}
    positions = layout([1, 2])
    metadata = ext.layout_metadata(positions, current)
    assert metadata["positions"] == positions
    assert metadata["native_filter_configuration"] == NATIVE_FILTERS
    assert _deleted_native_filter_ids(current, metadata) == set()
    # The control: positions alone would read to rc2 as every filter deleted.
    assert _deleted_native_filter_ids(current, {"positions": positions}) == {"NATIVE_FILTER-region"}
    # The keys rc2's `DashboardDAO.set_dash_metadata` resets or drops when they are absent go back as they are; the
    # keys it keeps from the stored metadata are not sent.
    assert metadata == {"positions": positions, "native_filter_configuration": NATIVE_FILTERS,
                        "shared_label_colors": ["West"], "map_label_colors": {"West": "#1FA8C9"},
                        "color_scheme_domain": ["#1FA8C9"], "color_namespace": "ns"}
    metadata["positions"]["CHART-1"]["meta"]["uuid"] = "x"
    assert "uuid" not in positions["CHART-1"]["meta"]
    # A dashboard without filters sends an empty list, which deletes none.
    assert ext.layout_metadata(positions, {}) == {"positions": positions, "native_filter_configuration": []}
    assert ext.layout_metadata(positions, None) == {"positions": positions, "native_filter_configuration": []}


def _legacy_filters_after_save(stored: dict, sent: dict) -> tuple:
    """What rc2's `DashboardDAO.set_dash_metadata` (`superset/daos/dashboard.py`) stores for the legacy
    `default_filters` and `filter_scopes` when a save sends `sent`, holding `positions`, over the stored metadata
    `stored`: it rewrites `default_filters` from the sent value alone (keeping the charts the layout holds), and
    keeps `filter_scopes` only when they are sent (`copy_filter_scopes`, the layout's charts again)."""
    md = copy.deepcopy(stored)
    chart_ids = [node.get("meta", {}).get("chartId") for node in sent["positions"].values() if isinstance(node, dict)]
    kept = {chart_id: chart_id for chart_id in chart_ids}
    scopes: dict = {}
    if "filter_scopes" in sent:
        sent_scopes = sent["filter_scopes"]
        sent_scopes = json.loads(sent_scopes or "{}") if isinstance(sent_scopes, str) else copy.deepcopy(sent_scopes)
        for filter_id, filter_scopes in sent_scopes.items():
            if kept.get(int(filter_id)):
                scopes[str(filter_id)] = filter_scopes
                for scope in filter_scopes.values():
                    scope["immune"] = [i for i in scope.get("immune", []) if int(i) in kept]
    try:
        defaults = json.loads(sent.get("default_filters", "{}"))
    except (json.JSONDecodeError, TypeError):
        defaults = {}
    md["default_filters"] = json.dumps({key: v for key, v in defaults.items() if int(key) in chart_ids})
    if scopes:
        md["filter_scopes"] = scopes
    else:
        md.pop("filter_scopes", None)
    return json.loads(md["default_filters"]), md.get("filter_scopes")


def test_layout_metadata_keeps_legacy_filters():
    """A dashboard made by hand may still carry the legacy filter settings. rc2 rebuilds both from what a save sends,
    so a layout save sends them back as they are stored, or it would undo them."""
    defaults = {"1": {"region": ["west"]}}
    scopes = {"1": {"region": {"scope": ["ROOT_ID"], "immune": [2]}}}
    stored = {"native_filter_configuration": [], "default_filters": json.dumps(defaults), "filter_scopes": scopes}
    positions = layout([1, 2])
    metadata = ext.layout_metadata(positions, stored)
    assert metadata["default_filters"] == stored["default_filters"]
    assert metadata["filter_scopes"] == scopes
    assert _legacy_filters_after_save(stored, metadata) == (defaults, scopes)
    # The control: a save without them stores no default filters and drops the scopes.
    assert _legacy_filters_after_save(stored, {"positions": positions, "native_filter_configuration": []}) == ({}, None)
    # Sent as a copy: what rc2 does to the sent scopes never reaches the stored ones.
    metadata["filter_scopes"]["1"]["region"]["immune"].append(9)
    assert scopes == {"1": {"region": {"scope": ["ROOT_ID"], "immune": [2]}}}
    # Stored as an object rather than JSON text, the default filters go back as the JSON text rc2 reads.
    as_object = ext.layout_metadata(positions, {"default_filters": defaults})
    assert json.loads(as_object["default_filters"]) == defaults
    assert _legacy_filters_after_save({"default_filters": defaults}, as_object) == (defaults, None)
    # Absent, neither is sent.
    assert not {"default_filters", "filter_scopes"} & set(ext.layout_metadata(positions, {}))


# ---------------------------------------------------------------- the saved metric add

def test_metric_stub_list_keeps_every_existing_id():
    existing = [
        {"id": 11, "metric_name": "count", "expression": "COUNT(*)", "d3format": ",d", "verbose_name": "Count"},
        {"id": 12, "metric_name": "sales", "expression": "SUM(amount)", "currency": {"symbol": "USD"}},
        {"id": 13, "metric_name": "margin", "expression": "SUM(amount - cost)", "extra": "{}"},
    ]
    new = {"metric_name": "avg_ticket", "expression": "AVG(amount)", "verbose_name": "Average ticket",
           "d3format": ",.2f", "currency": {"symbol": "USD", "symbolPosition": "prefix"}, "description": "per order"}
    stubs = ext.metric_stub_list(existing, new)
    # Every existing metric goes back as its id and name only, so `DatasetDAO.update_metrics` keeps it untouched;
    # the new one has no id, so it is created.
    assert stubs == [{"id": 11, "metric_name": "count"}, {"id": 12, "metric_name": "sales"},
                     {"id": 13, "metric_name": "margin"}, new]
    assert stubs[-1] is not new
    with pytest.raises(ext.MetricExists):
        ext.metric_stub_list(existing, {**new, "metric_name": "sales"})
    for bad in ({**new, "id": 12}, {"expression": "AVG(amount)"}, {"metric_name": "x"}):
        with pytest.raises(ValueError):
            ext.metric_stub_list(existing, bad)
    with pytest.raises(ValueError):
        ext.metric_stub_list([{"metric_name": "no id"}], new)
    assert ext.metric_stub_list([], new) == [new]


# ---------------------------------------------------------------- a histogram's own check query

HISTOGRAM = {"viz_type": "histogram_v2", "column": "amount", "groupby": ["region"], "bins": 10, "normalize": False,
             "cumulative": False, "row_limit": 10000, "adhoc_filters": [WEST]}


def test_histogram_check_context_selects_the_groups_then_the_column():
    """The observations and their grouping columns: the form data's `groupby`, then its `column`; no metrics and no
    post-processing; the form data's filters, read as the other families' builders read them."""
    context = ext.histogram_check_context(HISTOGRAM, DATASET)
    assert context["datasource"] == {"id": DATASET, "type": "table"}
    assert context["result_type"] == "full" and context["force"] is False
    (query,) = context["queries"]
    assert query["columns"] == ["region", "amount"]
    assert "metrics" not in query  # a plain row select, as the histogram's own `buildQuery` sends none
    assert query["filters"] == [WEST_Q, {"col": "region", "op": "IS NOT NULL"}, {"col": "amount", "op": "IS NOT NULL"}]
    assert query["post_processing"] == [{"operation": "histogram", "options": {
        "column": "amount", "groupby": ["region"], "bins": 10, "cumulative": False, "normalize": False}}]
    assert json.loads(json.dumps(context)) == context


def test_histogram_check_context_selects_the_column_alone_without_groups():
    (query,) = ext.histogram_check_context({"viz_type": "histogram_v2", "column": "amount"}, DATASET)["queries"]
    assert query["columns"] == ["amount"] and "metrics" not in query
    assert query["filters"] == [{"col": "amount", "op": "IS NOT NULL"}]


@pytest.mark.parametrize("bins", ["inf", "Infinity", "-inf", 1e999, float("inf")])
def test_histogram_check_context_refuses_bins_that_are_not_finite(bins):
    with pytest.raises(ValueError):
        ext.histogram_check_context({**HISTOGRAM, "bins": bins}, DATASET)


def test_histogram_check_context_keeps_an_adhoc_columns_own_object_in_its_null_filter():
    adhoc = {"label": "doubled", "sqlExpression": "amount * 2", "expressionType": "SQL"}
    (query,) = ext.histogram_check_context({"viz_type": "histogram_v2", "column": adhoc}, DATASET)["queries"]
    assert query["filters"] == [{"col": adhoc, "op": "IS NOT NULL"}]


@pytest.mark.parametrize("bins, parsed", [("absent", 5), ("12", 12), ("x", 5), (7.0, 7)])
def test_histogram_check_context_reads_bins_as_the_operator_does(bins, parsed):
    """`histogramOperator`: `Number(bins)`, or 5 where that is NaN; a missing `bins` is NaN."""
    form = {"viz_type": "histogram_v2", "column": "amount", "cumulative": True, "normalize": True, "bins": bins}
    if bins == "absent":
        form.pop("bins")
    (query,) = ext.histogram_check_context(form, DATASET)["queries"]
    assert query["post_processing"] == [{"operation": "histogram", "options": {
        "column": "amount", "groupby": [], "bins": parsed, "cumulative": True, "normalize": True}}]


_COUNT_ALL = [{"expressionType": "SQL", "sqlExpression": "COUNT(*)", "label": "COUNT(*)"}]


@pytest.mark.parametrize("having", [
    {"expressionType": "SIMPLE", "clause": "HAVING", "subject": "amount", "operator": ">", "comparator": 1},
    {"expressionType": "SQL", "clause": "HAVING", "sqlExpression": "SUM(amount) > 1"},
])
def test_histogram_check_context_aggregates_when_a_having_filter_is_present(having):
    """The histogram's `buildQuery` sends `COUNT(*)` as its metric only when an adhoc filter's clause is HAVING, so
    the query groups and a HAVING clause has an aggregate to hold; the groups then the column stay as they were."""
    form = {**HISTOGRAM, "adhoc_filters": [WEST, having]}
    (query,) = ext.histogram_check_context(form, DATASET)["queries"]
    assert query["metrics"] == _COUNT_ALL
    assert query["columns"] == ["region", "amount"]


def test_histogram_check_context_does_not_change_its_form_data():
    before = copy.deepcopy(HISTOGRAM)
    ext.histogram_check_context(HISTOGRAM, DATASET)
    assert HISTOGRAM == before


@pytest.mark.parametrize("form", [{"viz_type": "histogram_v2"}, {"column": None}, {"column": ""}, {"column": []}])
def test_histogram_check_context_needs_a_column(form):
    with pytest.raises(ValueError):
        ext.histogram_check_context({"groupby": ["region"], **form}, DATASET)


def test_histogram_check_context_needs_a_dataset():
    with pytest.raises(ValueError):
        ext.histogram_check_context(HISTOGRAM, 0)

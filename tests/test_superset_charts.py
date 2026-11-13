"""Chart specs for the fifteen v1 types: the request each family sends, and the plan checks that refuse a chart
Superset would show wrongly.

The request fields are those of rc2's `superset/mcp_service/chart/schemas.py` (`GenerateChartRequest` and the per-type
`*ChartConfig` models, which reject an unknown field), and are held here by value, since rc2 is not importable.
"""

from __future__ import annotations

import pytest

from mnemiq.superset.charts import (HANDLEBARS_TEMPLATE, MCP_FAMILIES, ChartSpec, Family, Filter, SavedRef, WrittenRef,
                                    extension_chart_request, generate_chart_request, plan_refusal)
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.saved_metrics import UNKNOWN, SavedMetric

USD = {"symbol": "USD", "symbolPosition": "prefix"}
K = MeasureKind


def _measure(name="revenue", kind=K.CURRENCY, currency="USD", expression="SUM(amount)") -> Measure:
    return Measure(name=name, label=name.title(), expression=expression, kind=kind, dataset_id=7, description="d",
                   asked_by="u", currency=currency if kind is K.CURRENCY else None)


def _spec(family: Family, measures=None, **more) -> ChartSpec:
    base = dict(key="c1", family=family, title="A chart", dataset_id=7,
                measures=tuple(measures if measures is not None else [WrittenRef(_measure())]),
                dimensions=("region", "channel"), time_column=None, time_grain=None, filters=(), row_limit=None,
                source=None, target=None)
    base.update(more)
    return ChartSpec(**base)


def _written(*measures: Measure) -> dict[str, Measure]:
    return {m.name: m for m in measures}


def _build(spec: ChartSpec, measures=None, saved=None) -> dict:
    return generate_chart_request(spec, measures or {}, saved or {})


WRITTEN_COLUMN = {"sql_expression": "SUM(amount)", "label": "Revenue"}
FORMAT = ",.2f"

# family -> (spec changes, the config as rc2's model for that chart_type wants it, with the written currency measure)
CASES = {
    Family.LINE: (dict(time_column="order_date", time_grain="P1M"),
                  {"chart_type": "xy", "kind": "line", "x": {"name": "order_date"}, "y": [WRITTEN_COLUMN],
                   "group_by": [{"name": "region"}, {"name": "channel"}], "time_grain": "P1M",
                   "temporal_column": "order_date", "y_axis": {"format": FORMAT}, "currency_format": USD}),
    Family.BAR: (dict(), {"chart_type": "xy", "kind": "bar", "x": {"name": "region"}, "y": [WRITTEN_COLUMN],
                          "group_by": [{"name": "channel"}], "y_axis": {"format": FORMAT}, "currency_format": USD}),
    Family.AREA: (dict(), {"chart_type": "xy", "kind": "area", "x": {"name": "region"}, "y": [WRITTEN_COLUMN],
                           "group_by": [{"name": "channel"}], "y_axis": {"format": FORMAT},
                           "currency_format": USD}),
    Family.SCATTER: (dict(), {"chart_type": "xy", "kind": "scatter", "x": {"name": "region"},
                              "y": [WRITTEN_COLUMN], "group_by": [{"name": "channel"}],
                              "y_axis": {"format": FORMAT}, "currency_format": USD}),
    Family.TABLE: (dict(), {"chart_type": "table", "query_mode": "aggregate",
                            "columns": [{"name": "region"}, {"name": "channel"}, WRITTEN_COLUMN],
                            "column_config": {"Revenue": {"d3NumberFormat": FORMAT}}}),
    Family.PIE: (dict(), {"chart_type": "pie", "dimension": {"name": "region"}, "metric": WRITTEN_COLUMN,
                          "number_format": FORMAT, "currency_format": USD}),
    Family.BIG_NUMBER: (dict(), {"chart_type": "big_number", "metric": WRITTEN_COLUMN, "show_trendline": False,
                                 "y_axis_format": FORMAT, "currency_format": USD}),
    Family.GAUGE: (dict(), {"chart_type": "gauge", "metric": WRITTEN_COLUMN,
                            "groupby": [{"name": "region"}, {"name": "channel"}],
                            "number_format": FORMAT, "currency_format": USD}),
    Family.HISTOGRAM: (dict(measures=[]), {"chart_type": "histogram", "column": {"name": "region"},
                                           "groupby": [{"name": "channel"}]}),
    Family.BOX_PLOT: (dict(measures=[WrittenRef(_measure(kind=K.NUMBER, name="score", expression="AVG(score)"))]),
                      {"chart_type": "box_plot",
                       "metrics": [{"sql_expression": "AVG(score)", "label": "Score"}],
                       "distribute_across": [{"name": "region"}], "dimensions": [{"name": "channel"}],
                       "number_format": FORMAT}),
    Family.PIVOT: (dict(), {"chart_type": "pivot_table", "rows": [{"name": "region"}],
                            "columns": [{"name": "channel"}], "metrics": [WRITTEN_COLUMN],
                            "value_format": FORMAT, "currency_format": USD}),
    Family.INTERACTIVE_PIVOT: (dict(), {"chart_type": "interactive_pivot", "rows": [{"name": "region"}],
                                        "columns": [{"name": "channel"}], "metrics": [WRITTEN_COLUMN],
                                        "value_format": FORMAT, "currency_format": USD}),
    Family.TREEMAP: (dict(), {"chart_type": "treemap_v2",
                              "groupby": [{"name": "region"}, {"name": "channel"}], "metric": WRITTEN_COLUMN}),
    Family.WATERFALL: (dict(), {"chart_type": "waterfall", "x_axis": {"name": "region"},
                                "breakdown": {"name": "channel"}, "metric": WRITTEN_COLUMN,
                                "y_axis_format": FORMAT, "currency_format": USD}),
    Family.MIXED: (dict(time_column="order_date", measures=[WrittenRef(_measure()),
                                                           WrittenRef(_measure("orders", K.COUNT,
                                                                               expression="COUNT(*)"))]),
                   {"chart_type": "mixed_timeseries", "x": {"name": "order_date"}, "y": [WRITTEN_COLUMN],
                    "y_secondary": [{"sql_expression": "COUNT(*)", "label": "Orders"}],
                    "group_by": [{"name": "region"}, {"name": "channel"}],
                    "group_by_secondary": [{"name": "region"}, {"name": "channel"}],
                    "primary_kind": "line", "secondary_kind": "bar", "temporal_column": "order_date",
                    "y_axis": {"format": FORMAT}, "currency_format": USD,
                    "y_axis_secondary": {"format": ",d"}}),
    Family.HANDLEBARS: (dict(measures=[WrittenRef(_measure("orders", K.COUNT, expression="COUNT(*)"))]),
                        {"chart_type": "handlebars", "handlebars_template": HANDLEBARS_TEMPLATE,
                         "query_mode": "aggregate",
                         "groupby": [{"name": "region"}, {"name": "channel"}],
                         "metrics": [{"sql_expression": "COUNT(*)", "label": "Orders"}]}),
}


def test_the_cases_cover_every_mcp_family():
    assert set(CASES) == set(MCP_FAMILIES)
    assert len(MCP_FAMILIES) == 16
    assert Family.FUNNEL not in MCP_FAMILIES and Family.SANKEY not in MCP_FAMILIES
    assert len(Family) == 18


@pytest.mark.parametrize("family", sorted(MCP_FAMILIES, key=str))
def test_family_request_matches_rc2_schema(family):
    """Each field is one of the `*ChartConfig` model's own (GenerateChartRequest.config, a union on `chart_type`)."""
    changes, config = CASES[family]
    spec = _spec(family, **changes)
    assert plan_refusal(spec, interactive_pivot_available=True) is None
    request = _build(spec)
    assert request["dataset_id"] == 7
    assert request["chart_name"] == "A chart"
    assert request["save_chart"] is True and request["generate_preview"] is False
    assert set(request) == {"dataset_id", "config", "chart_name", "save_chart", "generate_preview"}
    got = request["config"]
    for key, value in config.items():
        assert got[key] == value, key
    assert set(got) == set(config), set(got) ^ set(config)


def test_viz_types_are_the_ones_rc2_registers():
    assert {f: f.value for f in Family} == {
        Family.LINE: "echarts_timeseries_line", Family.BAR: "echarts_timeseries_bar", Family.AREA: "echarts_area",
        Family.SCATTER: "echarts_timeseries_scatter", Family.TABLE: "table", Family.PIE: "pie",
        Family.BIG_NUMBER: "big_number_total", Family.GAUGE: "gauge_chart", Family.HISTOGRAM: "histogram_v2",
        Family.BOX_PLOT: "box_plot", Family.PIVOT: "pivot_table_v2", Family.INTERACTIVE_PIVOT: "ag-grid-pivot-table",
        Family.TREEMAP: "treemap_v2", Family.WATERFALL: "waterfall", Family.MIXED: "mixed_timeseries",
        Family.HANDLEBARS: "handlebars", Family.FUNNEL: "funnel", Family.SANKEY: "sankey_v2"}


def test_a_saved_metric_is_named_not_written():
    saved = {"revenue_total": SavedMetric(7, "revenue_total", "SUM(amount)", None, None, USD, None, None)}
    spec = _spec(Family.PIE, measures=[SavedRef("revenue_total", K.CURRENCY)])
    config = _build(spec, saved=saved)["config"]
    assert config["metric"] == {"name": "revenue_total", "saved_metric": True}
    assert config["number_format"] == FORMAT and config["currency_format"] == USD


def test_a_saved_metric_of_an_unknown_currency_gets_the_kinds_format_without_one():
    saved = {"m": SavedMetric(7, "m", "SUM(amount)", None, None, UNKNOWN, UNKNOWN, None)}
    config = _build(_spec(Family.PIE, measures=[SavedRef("m", K.PERCENT)]), saved=saved)["config"]
    assert config["number_format"] == ",.1%" and "currency_format" not in config


def test_a_percent_measure_never_carries_a_currency():
    spec = _spec(Family.BIG_NUMBER, measures=[WrittenRef(_measure("rate", K.PERCENT, expression="AVG(r)"))])
    config = _build(spec)["config"]
    assert config["y_axis_format"] == ",.1%" and "currency_format" not in config


def test_currency_is_an_iso_code_in_currency_format():
    for family in (Family.PIE, Family.BIG_NUMBER, Family.GAUGE, Family.WATERFALL, Family.PIVOT):
        spec = _spec(family, measures=[WrittenRef(_measure(currency="EUR"))])
        symbol = _build(spec)["config"]["currency_format"]["symbol"]
        assert symbol == "EUR"
    assert _build(_spec(Family.PIE))["config"]["currency_format"] == USD


def test_filters_use_the_filter_config_fields():
    spec = _spec(Family.PIE, filters=(Filter("region", "IN", ["east", "west"]), Filter("amount", "IS NULL", None)),
                 row_limit=50)
    config = _build(spec)["config"]
    assert config["filters"] == [{"column": "region", "op": "IN", "value": ["east", "west"]},
                                 {"column": "amount", "op": "IS NULL"}]
    assert config["row_limit"] == 50


def test_funnel_and_sankey_go_to_the_extension():
    funnel = _spec(Family.FUNNEL, dimensions=("stage",))
    request = extension_chart_request(funnel, {}, {})
    assert request["viz_type"] == "funnel" and request["dataset_id"] == 7 and request["slice_name"] == "A chart"
    params = request["params"]
    assert params["viz_type"] == "funnel" and params["datasource"] == "7__table"
    assert params["groupby"] == ["stage"]
    assert params["metric"] == {"expressionType": "SQL", "sqlExpression": "SUM(amount)", "label": "Revenue"}
    assert params["number_format"] == FORMAT and params["currency_format"] == USD

    sankey = _spec(Family.SANKEY, measures=[WrittenRef(_measure("orders", K.COUNT, expression="COUNT(*)"))],
                   dimensions=(), source="origin", target="destination")
    params = extension_chart_request(sankey, {}, {})["params"]
    assert params["viz_type"] == "sankey_v2" and params["datasource"] == "7__table"
    assert (params["source"], params["target"]) == ("origin", "destination")
    assert params["metric"]["label"] == "Orders"
    assert "number_format" not in params and "currency_format" not in params


def test_a_saved_metric_on_funnel_is_its_name():
    spec = _spec(Family.FUNNEL, measures=[SavedRef("m", K.COUNT)], dimensions=("stage",))
    assert extension_chart_request(spec, {}, {})["params"]["metric"] == "m"


def test_the_two_requests_are_not_interchangeable():
    with pytest.raises(ValueError):
        _build(_spec(Family.FUNNEL))
    with pytest.raises(ValueError):
        extension_chart_request(_spec(Family.PIE), {}, {})


@pytest.mark.parametrize("family", [Family.SANKEY, Family.BOX_PLOT, Family.HISTOGRAM])
@pytest.mark.parametrize("kind", [K.CURRENCY, K.PERCENT])
def test_currency_on_sankey_box_plot_histogram_is_refused(family, kind):
    spec = _spec(family, measures=[WrittenRef(_measure("m", kind, expression="AVG(x)"))], source="a", target="b")
    reason = plan_refusal(spec, interactive_pivot_available=True)
    assert reason and kind.value in reason
    ok = _spec(family, measures=[WrittenRef(_measure("m", K.NUMBER, expression="AVG(x)"))], source="a", target="b")
    assert plan_refusal(ok, interactive_pivot_available=True) is None
    reused = _spec(family, measures=[SavedRef("m", kind)], source="a", target="b")
    assert plan_refusal(reused, interactive_pivot_available=True)


def test_two_kinds_on_one_axis_are_refused():
    two = [WrittenRef(_measure()), WrittenRef(_measure("orders", K.COUNT, expression="COUNT(*)"))]
    for family in (Family.LINE, Family.PIVOT, Family.INTERACTIVE_PIVOT, Family.BAR):
        reason = plan_refusal(_spec(family, measures=two), interactive_pivot_available=True)
        assert reason and "axis" in reason, family
    # a mixed chart's primary and secondary axes may differ, and a table has one format per column
    mixed = _spec(Family.MIXED, measures=two, time_column="d")
    assert plan_refusal(mixed, interactive_pivot_available=True) is None
    assert plan_refusal(_spec(Family.TABLE, measures=two), interactive_pivot_available=True) is None
    # but two kinds on the mixed chart's secondary axis are not
    three = [*two, WrittenRef(_measure("share", K.PERCENT, expression="AVG(s)"))]
    assert plan_refusal(_spec(Family.MIXED, measures=three, time_column="d"), interactive_pivot_available=True)


def test_two_currencies_on_one_axis_are_refused():
    two = [WrittenRef(_measure("a", currency="USD")), WrittenRef(_measure("b", currency="EUR"))]
    assert plan_refusal(_spec(Family.LINE, measures=two), interactive_pivot_available=True)
    same = [WrittenRef(_measure("a")), WrittenRef(_measure("b"))]
    assert plan_refusal(_spec(Family.LINE, measures=same), interactive_pivot_available=True) is None


def test_handlebars_takes_count_or_number_only():
    for kind, allowed in ((K.COUNT, True), (K.NUMBER, True), (K.CURRENCY, False), (K.PERCENT, False)):
        spec = _spec(Family.HANDLEBARS, measures=[WrittenRef(_measure("m", kind, expression="SUM(x)"))])
        assert (plan_refusal(spec, interactive_pivot_available=True) is None) is allowed, kind
    assert plan_refusal(_spec(Family.HANDLEBARS, measures=[SavedRef("m", K.PERCENT)]),
                        interactive_pivot_available=True)


def test_interactive_pivot_only_when_advertised():
    spec = _spec(Family.INTERACTIVE_PIVOT)
    assert plan_refusal(spec, interactive_pivot_available=False)
    assert plan_refusal(spec, interactive_pivot_available=True) is None
    assert plan_refusal(_spec(Family.PIVOT), interactive_pivot_available=False) is None


@pytest.mark.parametrize("family,changes", [
    (Family.PIE, dict(dimensions=())), (Family.PIE, dict(measures=[])),
    (Family.BIG_NUMBER, dict(measures=[WrittenRef(_measure()), WrittenRef(_measure("b"))])),
    (Family.MIXED, dict(time_column="d")), (Family.MIXED, dict(measures=[WrittenRef(_measure()),
                                                                          WrittenRef(_measure("b"))])),
    (Family.SANKEY, dict(source="a")), (Family.TREEMAP, dict(dimensions=())),
    (Family.WATERFALL, dict(dimensions=())), (Family.TABLE, dict(dimensions=(), measures=[])),
])
def test_a_spec_that_cannot_be_built_is_refused_not_a_crash(family, changes):
    assert plan_refusal(_spec(family, **changes), interactive_pivot_available=True)


def test_row_limit_is_held_to_what_the_chart_model_accepts():
    """`GaugeChartConfig.row_limit` is at most 10, and `BigNumberChartConfig` has no row_limit field."""
    assert plan_refusal(_spec(Family.GAUGE, row_limit=50), interactive_pivot_available=True)
    assert plan_refusal(_spec(Family.GAUGE, row_limit=10), interactive_pivot_available=True) is None
    assert plan_refusal(_spec(Family.PIE, row_limit=0), interactive_pivot_available=True)
    assert "row_limit" not in _build(_spec(Family.BIG_NUMBER, row_limit=50))["config"]

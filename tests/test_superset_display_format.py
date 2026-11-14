"""The number format and currency Superset's renderer shows for a chart's metric, decided from saved `form_data` and the
saved metric, and whether that display fits the measure's kind."""

from __future__ import annotations

import pytest

from mnemiq.superset.display_format import CANNOT_TELL, Displayed, displayed, fits
from mnemiq.superset.measures import MeasureKind
from mnemiq.superset.saved_metrics import UNKNOWN, SavedMetric

USD = {"symbol": "USD", "symbolPosition": "prefix"}
ADHOC = {"expressionType": "SQL", "sqlExpression": "SUM(amount)", "label": "revenue"}


def saved(d3format=None, currency=None):
    return SavedMetric(dataset_id=1, name="revenue", expression="SUM(amount)", verbose_name=None, d3format=d3format,
                       currency=currency, certified_by=None, warning_text=None)


def test_big_number_saved_d3format_wins():
    shown = displayed("big_number_total", {"y_axis_format": ",d"}, "revenue", saved(d3format=",.2f"))
    assert shown == Displayed(",.2f", None)
    # an ad hoc metric has no saved entry, so the chart's format shows
    assert displayed("big_number_total", {"y_axis_format": ",d"}, ADHOC, None) == Displayed(",d", None)
    # an empty saved format falls through (`||`)
    assert displayed("big_number", {"y_axis_format": ",d"}, "revenue", saved(d3format="")) == Displayed(",d", None)


def test_pie_ignores_saved_d3format():
    shown = displayed("pie", {"number_format": ",d"}, "revenue", saved(d3format=",.2f"))
    assert shown == Displayed(",d", None)
    # the default is merged when the key is absent, so the saved format still does not apply
    assert displayed("pie", {}, "revenue", saved(d3format=",.2f")) == Displayed("SMART_NUMBER", None)


def test_timeseries_saved_applies_only_when_key_absent():
    saved_metric = saved(d3format=",.2f")
    assert displayed("echarts_timeseries_bar", {"y_axis_format": ",d"}, "revenue", saved_metric).d3format == ",d"
    assert displayed("echarts_timeseries_bar", {}, "revenue", saved_metric).d3format == ",.2f"
    assert displayed("echarts_timeseries_bar", {}, ADHOC, None).d3format is None


def test_pivot_saved_currency_wins():
    form = {"valueFormat": ",d", "currency_format": {"symbol": "EUR", "symbolPosition": "prefix"}}
    shown = displayed("pivot_table_v2", form, "revenue", saved(d3format=",.2f", currency=USD))
    assert shown == Displayed(",.2f", USD)
    # no saved currency: the chart's applies
    assert displayed("pivot_table_v2", form, "revenue", saved()).currency["symbol"] == "EUR"


def test_waterfall_reads_chart_only():
    form = {"y_axis_format": ",d"}
    assert displayed("waterfall", form, "revenue", saved(d3format=",.2f", currency=USD)) == Displayed(",d", None)
    assert displayed("waterfall", {**form, "currency_format": USD}, ADHOC, None) == Displayed(",d", USD)


def test_sankey_fixed_two_decimals():
    assert displayed("sankey_v2", {"number_format": "$,d"}, "revenue", saved(currency=USD)) == Displayed(",.2f", None)


def test_table_column_config_wins_then_saved():
    form = {"column_config": {"revenue": {"d3NumberFormat": ",d", "currencyFormat": USD}}}
    assert displayed("table", form, "revenue", saved(d3format=",.2f")) == Displayed(",d", USD)
    # no per-column entry: the saved metric is the fallback
    assert displayed("table", {"column_config": {}}, "revenue", saved(d3format=",.2f", currency=USD)) == Displayed(",.2f", USD)
    # an ad hoc metric is keyed by its label and never matches a saved metric
    adhoc = {"column_config": {"revenue": {"d3NumberFormat": ",d"}}}
    assert displayed("table", adhoc, ADHOC, None) == Displayed(",d", None)
    assert displayed("table", {}, ADHOC, None) == Displayed(None, None)


def test_percent_with_currency_loses_its_sign():
    assert fits(MeasureKind.PERCENT, Displayed(",.1%", None))
    assert not fits(MeasureKind.PERCENT, Displayed(",.1%", USD))
    assert not fits(MeasureKind.PERCENT, Displayed("$,.1%", None))
    assert not fits(MeasureKind.PERCENT, Displayed(None, None))


def test_cleared_currency_on_a_written_measure_fails():
    form = {"number_format": None, "currency_format": None}
    shown = displayed("pie", form, ADHOC, None)
    assert shown == Displayed(None, None)
    assert not fits(MeasureKind.CURRENCY, shown)
    assert not fits(MeasureKind.CURRENCY, displayed("pie", {"number_format": ",.2f", "currency_format": {}}, ADHOC, None))


@pytest.mark.parametrize("viz_type", ["pie", "gauge_chart", "funnel", "treemap_v2"])
def test_saved_currency_falls_back_on_pie_gauge_funnel_treemap(viz_type):
    form = {"number_format": ",.2f", "currency_format": None}
    shown = displayed(viz_type, form, "revenue", saved(currency=USD))
    assert shown.currency == USD
    assert fits(MeasureKind.CURRENCY, shown)


def test_saved_currency_overrides_a_planned_percent():
    shown = displayed("pie", {"number_format": ",.1%"}, "revenue", saved(currency=USD))
    assert not fits(MeasureKind.PERCENT, shown)


@pytest.mark.parametrize("currency, expected", [
    (None, None), ({}, None), ({"symbol": ""}, None), ({"symbol": "USD"}, {"symbol": "USD"}),
    ({"symbol": "AUTO"}, CANNOT_TELL)])
def test_currency_states(currency, expected):
    form = {"number_format": ",.2f"}
    if currency is not None:
        form["currency_format"] = currency
    shown = displayed("pie", form, ADHOC, None)
    if expected is CANNOT_TELL:
        assert shown is CANNOT_TELL
    else:
        assert shown.currency == expected
    assert displayed("pie", {**form, "currency_format": None}, ADHOC, None).currency is None


def test_big_number_default_with_saved_currency_format_passes():
    assert fits(MeasureKind.CURRENCY, displayed("big_number_total", {}, "revenue", saved(d3format="$,.2f")))
    assert fits(MeasureKind.CURRENCY, displayed("big_number_total", {}, "revenue", saved(currency=USD)))
    assert not fits(MeasureKind.CURRENCY, displayed("big_number_total", {}, "revenue", saved()))


def test_a_symbol_that_is_not_an_iso_code_shows_no_currency():
    assert not fits(MeasureKind.CURRENCY, Displayed(",.2f", {"symbol": "$"}))


def test_handlebars_cannot_tell():
    assert displayed("handlebars", {}, "revenue", saved()) is CANNOT_TELL


def test_interactive_pivot_cannot_tell():
    assert displayed("ag-grid-pivot-table", {"valueFormat": ",d"}, "revenue", saved()) is CANNOT_TELL


def test_saved_currency_unknown_cannot_tell_where_it_decides():
    assert displayed("pivot_table_v2", {"valueFormat": ",d"}, "revenue", saved(currency=UNKNOWN)) is CANNOT_TELL
    # on a pie it decides only when the chart has no currency of its own
    assert displayed("pie", {"number_format": ",d"}, "revenue", saved(currency=UNKNOWN)) is CANNOT_TELL
    assert displayed("pie", {"number_format": ",d", "currency_format": USD}, "revenue",
                     saved(currency=UNKNOWN)) == Displayed(",d", USD)
    # a chart that never reads the saved currency is not affected
    assert displayed("waterfall", {}, "revenue", saved(currency=UNKNOWN)) == Displayed(None, None)


def test_forced_percent_modes_cannot_tell():
    assert displayed("echarts_timeseries_line", {"contributionMode": "row"}, ADHOC, None) is CANNOT_TELL
    assert displayed("pivot_table_v2", {"showValuesAs": "percent_row"}, ADHOC, None) is CANNOT_TELL


def test_mixed_secondary_axis_reads_its_own_keys():
    form = {"metrics": [ADHOC], "metrics_b": ["orders"], "y_axis_format": ",.2f", "y_axis_format_secondary": ",d",
            "currency_format": USD}
    assert displayed("mixed_timeseries", form, ADHOC, None) == Displayed(",.2f", USD)
    assert displayed("mixed_timeseries", form, "orders", None) == Displayed(",d", None)


def test_unknown_viz_type_cannot_tell():
    assert displayed("deck_scatter", {}, "revenue", None) is CANNOT_TELL


def test_a_dollar_format_counts_only_when_no_currency_symbol_is_set():
    assert not fits(MeasureKind.CURRENCY, Displayed("$,.2f", {"symbol": "$"}))
    assert fits(MeasureKind.CURRENCY, Displayed("$,.2f", None))
    assert fits(MeasureKind.CURRENCY, Displayed("$,.2f", {"symbol": ""}))
    assert fits(MeasureKind.CURRENCY, Displayed("$,.2f", {"symbol": "USD"}))


def test_a_saved_line_chart_reads_its_control_name_for_rebased_percent_change():
    saved_form = {"rebase_percent_change": True, "y_axis_format": ",.2f"}
    assert displayed("echarts_timeseries_line", saved_form, ADHOC, None) is CANNOT_TELL
    assert displayed("echarts_timeseries_line", {"rebase_percent_change": False}, ADHOC, None) is not CANNOT_TELL


def test_pivot_actual_values_is_the_default_display_not_a_percent():
    form = {"valueFormat": ",d", "showValuesAs": "actual"}
    assert displayed("pivot_table_v2", form, ADHOC, None) == Displayed(",d", None)
    assert displayed("pivot_table_v2", {**form, "showValuesAs": "percent_total"}, ADHOC, None) is CANNOT_TELL


@pytest.mark.parametrize("kind, shown, ok", [
    (MeasureKind.COUNT, Displayed(",.0%", None), False),
    (MeasureKind.COUNT, Displayed(",d", USD), False),
    (MeasureKind.NUMBER, Displayed("$,.2f", None), False),
    (MeasureKind.COUNT, Displayed(",d", None), True),
    (MeasureKind.NUMBER, Displayed("SMART_NUMBER", None), True),
], ids=["a count in percent", "a count in dollars", "a number in a $ format", "a count as ,d",
        "a number in the default format"])
def test_a_count_or_a_number_shows_neither_a_currency_nor_a_percent(kind, shown, ok):
    assert fits(kind, shown) is ok


def test_a_lower_case_code_shows_its_currency():
    """rc2's `getCurrencySymbol` (`currency-format/CurrencyFormatter.ts`) hands the symbol to `Intl.NumberFormat`, which
    takes a well-formed code in either case: `usd` shows `$`, and the percent beside it loses its `%`."""
    assert not fits(MeasureKind.PERCENT, Displayed(",.1%", {"symbol": "usd"}))
    assert fits(MeasureKind.CURRENCY, Displayed(",.2f", {"symbol": "usd"}))
    assert not fits(MeasureKind.NUMBER, Displayed(",.2f", {"symbol": "Eur"}))
    # what `Intl.NumberFormat` refuses shows no currency, and the percent keeps its `%`
    assert not fits(MeasureKind.CURRENCY, Displayed(",.2f", {"symbol": "us"}))
    assert fits(MeasureKind.PERCENT, Displayed(",.1%", {"symbol": "$"}))


TIMESERIES = ["echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area", "echarts_timeseries_scatter",
              "echarts_timeseries_smooth", "echarts_timeseries_step", "echarts_timeseries"]


@pytest.mark.parametrize("viz_type", TIMESERIES)
def test_series_stacked_to_100_percent_cannot_tell_on_every_timeseries_type(viz_type):
    """`Timeseries/transformProps.ts`, which every timeseries type renders through, forces a percent display for
    `isAreaExpand = stack === StackControlsValue.Expand`, whatever the type."""
    assert displayed(viz_type, {"stack": "Expand", "y_axis_format": ",.2f"}, ADHOC, None) is CANNOT_TELL
    assert displayed(viz_type, {"stack": "Stack", "y_axis_format": ",.2f"}, ADHOC, None) == Displayed(",.2f", None)


def test_an_unlabelled_metric_is_keyed_as_rc2_keys_it():
    """`getMetricLabel` (`superset-ui-core/src/query/getMetricLabel.ts`): an ad hoc metric with no label is keyed by its
    SQL, or a SIMPLE one by `AGGREGATE(column)`."""
    sql = {"expressionType": "SQL", "sqlExpression": "SUM(amount)"}
    simple = {"expressionType": "SIMPLE", "aggregate": "SUM", "column": {"column_name": "amount"}}
    form = {"column_config": {"SUM(amount)": {"d3NumberFormat": ",d"}}}
    assert displayed("table", form, sql, None) == Displayed(",d", None)
    assert displayed("table", form, simple, None) == Displayed(",d", None)
    # a mixed chart tells two unlabelled metrics apart by those keys too
    orders = {"expressionType": "SQL", "sqlExpression": "COUNT(*)"}
    mixed = {"metrics": [sql], "metrics_b": [orders], "y_axis_format": ",.2f", "y_axis_format_secondary": ",d"}
    assert displayed("mixed_timeseries", mixed, sql, None) == Displayed(",.2f", None)
    assert displayed("mixed_timeseries", mixed, orders, None) == Displayed(",d", None)

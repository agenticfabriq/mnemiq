"""The number format and currency Superset 7.0.0rc2's renderer shows for a chart's metric.

What shows is decided by the chart's saved `form_data` and, for a metric saved on the dataset, the saved metric's
`d3format` and `currency`, and which of the two wins differs by chart type. Each branch below names the renderer
file and symbol it mirrors (under `superset-frontend/plugins/` unless noted). The shared resolver is
`getValueFormatter` / `buildCustomFormatters` in `packages/superset-ui-core/src/currency-format/utils.ts`.

A saved metric applies only when the chart names it by its name (a string); an ad hoc metric (a dict) has no saved entry.
Anything the renderer decides from data mnemiq does not have (an AUTO currency, a forced percent display, a plugin
whose source is not in the tree) is `CANNOT_TELL`, never a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mnemiq.superset.measures import MeasureKind
from mnemiq.superset.saved_metrics import UNKNOWN, SavedMetric

__all__ = ["CANNOT_TELL", "Displayed", "displayed", "fits"]

AUTO = "AUTO"
_ISO = re.compile(r"^[A-Z]{3}$")


class _CannotTell:
    def __repr__(self) -> str:
        return "CANNOT_TELL"


CANNOT_TELL = _CannotTell()


@dataclass(frozen=True)
class Displayed:
    """The effective d3 format (None: the renderer's default, SMART_NUMBER) and currency (None: no currency)."""

    d3format: str | None
    currency: dict | None


_TIMESERIES = {"echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area", "echarts_timeseries_scatter"}
_BIG_NUMBER = {"big_number_total", "big_number"}
_CHART_WINS = {"pie", "gauge_chart", "funnel", "treemap_v2"}


def _saved_of(metric_key, saved: SavedMetric | None) -> SavedMetric | None:
    """`isSavedMetric`: only a metric the chart names by a string can match a saved metric."""
    return saved if isinstance(metric_key, str) else None


def _label(metric_key) -> str:
    return metric_key if isinstance(metric_key, str) else str(metric_key.get("label", ""))


def _named(metric, key) -> bool:
    return any(m == key or (isinstance(m, dict) and isinstance(key, dict) and m.get("label") == key.get("label"))
               for m in (metric or []))


def _symbol(currency) -> str:
    return currency.get("symbol") if isinstance(currency, dict) and currency.get("symbol") else ""


def _finish(d3format, currency):
    """`Displayed`, or `CANNOT_TELL` when the currency that decides is AUTO (it needs the data's detected currency)."""
    if _symbol(currency) == AUTO:
        return CANNOT_TELL
    return Displayed(d3format or None, currency if _symbol(currency) else None)


def _chart_then_saved(chart_currency, saved: SavedMetric | None):
    """`getEffectiveCurrencyFormat`: the chart's currency when it has a symbol, else the saved metric's. Returns the
    currency, or `CANNOT_TELL` when the saved currency would decide and could not be read."""
    if _symbol(chart_currency):
        return chart_currency
    if saved is None:
        return None
    return CANNOT_TELL if saved.currency is UNKNOWN else saved.currency


def _with_currency(d3format, chart_currency, saved: SavedMetric | None):
    currency = _chart_then_saved(chart_currency, saved)
    return CANNOT_TELL if currency is CANNOT_TELL else _finish(d3format, currency)


def displayed(viz_type: str, form_data: dict, metric_key, saved: SavedMetric | None):
    """What Superset shows for `metric_key` on a `viz_type` chart saved with `form_data`: a `Displayed`, or
    `CANNOT_TELL`. `saved` is the saved metric `metric_key` names, or None."""
    saved = _saved_of(metric_key, saved)
    if viz_type in _TIMESERIES:
        # Timeseries/transformProps.ts, `buildCustomFormatters(..., yAxisFormat, ...)` and `forcePercentFormatter`.
        if form_data.get("contributionMode") or form_data.get("rebasePercentChange") \
                or (viz_type == "echarts_area" and form_data.get("stack") == "Expand"):
            return CANNOT_TELL
        d3 = form_data.get("y_axis_format")
        if d3 is None and saved is not None:
            d3 = saved.d3format
        return _with_currency(d3, form_data.get("currency_format"), saved)
    if viz_type == "mixed_timeseries":
        # MixedTimeseries/transformProps.ts, `customFormatters` / `customFormattersSecondary` over `metrics` + `metricsB`.
        if form_data.get("contributionMode"):
            return CANNOT_TELL
        secondary = _named(form_data.get("metrics_b"), metric_key)
        suffix = "_secondary" if secondary else ""
        d3 = form_data.get(f"y_axis_format{suffix}")
        if d3 is None and saved is not None:
            d3 = saved.d3format
        return _with_currency(d3, form_data.get(f"currency_format{suffix}"), saved)
    if viz_type in _BIG_NUMBER:
        # BigNumber/BigNumberTotal/transformProps.ts and BigNumberWithTrendline/transformProps.ts:
        # `getValueFormatter(..., metricEntry?.d3format || yAxisFormat, currencyFormat, ...)`: the saved format wins.
        d3 = (saved.d3format if saved is not None else None) or form_data.get("y_axis_format")
        return _with_currency(d3, form_data.get("currency_format"), saved)
    if viz_type in _CHART_WINS:
        # Pie/transformProps.ts, Gauge/transformProps.ts, Funnel/transformProps.ts, Treemap/transformProps.ts:
        # `{...DEFAULT_<X>_FORM_DATA, ...formData}` makes `numberFormat` a string ('SMART_NUMBER'), so `d3Format ??
        # saved` takes the chart's. A key saved as null survives `convertKeysToCamelCase` and falls to the saved format.
        d3 = form_data["number_format"] if "number_format" in form_data else "SMART_NUMBER"
        if d3 is None and saved is not None:
            d3 = saved.d3format
        return _with_currency(d3, form_data.get("currency_format"), saved)
    if viz_type == "waterfall":
        # Waterfall/transformProps.ts, `defaultFormatter`: chart `currencyFormat?.symbol` only, no saved metric.
        return _finish(form_data.get("y_axis_format"), form_data.get("currency_format"))
    if viz_type == "sankey_v2":
        # Sankey/transformProps.ts: `getNumberFormatter(NumberFormats.FLOAT_2_POINT)`; no format controls.
        return Displayed(",.2f", None)
    if viz_type == "histogram_v2":
        # Histogram/transformProps.ts: `y_axis_format` for the counts, no currency.
        return Displayed(form_data.get("y_axis_format") or None, None)
    if viz_type == "box_plot":
        # BoxPlot/transformProps.ts: `getNumberFormatter(numberFormat)`; no currency, no saved metric.
        return Displayed(form_data.get("number_format") or None, None)
    if viz_type == "table":
        # plugin-chart-table/src/transformProps.ts, `processColumns`: `config.d3NumberFormat || savedFormat` and
        # `config.currencyFormat?.symbol ? config.currencyFormat : savedCurrency`, keyed by the result column name.
        config = (form_data.get("column_config") or {}).get(_label(metric_key)) or {}
        if config.get("d3SmallNumberFormat"):
            return CANNOT_TELL
        d3 = config.get("d3NumberFormat") or (saved.d3format if saved is not None else None)
        return _with_currency(d3, config.get("currencyFormat"), saved)
    if viz_type == "pivot_table_v2":
        # plugin-chart-pivot-table/src/PivotTableChart.tsx, `customFormatsArray`: `columnFormats[metric] || valueFormat`
        # and `currencyFormats[metric] || currencyFormat`: the saved metric wins both.
        if form_data.get("showValuesAs") or form_data.get("show_values_as"):
            return CANNOT_TELL
        d3 = (saved.d3format if saved is not None else None) or form_data.get("valueFormat")
        if saved is not None and saved.currency is UNKNOWN:
            return CANNOT_TELL
        currency = saved.currency if saved is not None and saved.currency is not None \
            else form_data.get("currency_format")
        return _finish(d3, currency)
    # handlebars (plugin-chart-handlebars: no number formatting by the plugin), the interactive pivot (its plugin is not
    # in the tree) and any other type.
    return CANNOT_TELL


def _currency_shown(shown: Displayed) -> bool:
    """A currency is on screen when it is an ISO code (`Intl.NumberFormat` throws on anything else and
    `CurrencyFormatter.format` then returns the bare number), or when the d3 format itself starts with `$`."""
    return bool(_ISO.match(_symbol(shown.currency))) or (shown.d3format or "").startswith("$")


def fits(kind: MeasureKind, shown: Displayed) -> bool:
    """Whether `shown` displays a `kind` measure as it should. A percent format beside any shown currency loses its
    `%` (`CurrencyFormatter.normalizeForCurrency`)."""
    currency = _currency_shown(shown)
    has_percent = "%" in (shown.d3format or "")
    if kind is MeasureKind.CURRENCY:
        return currency
    if kind is MeasureKind.PERCENT:
        return has_percent and not currency
    return not currency and not has_percent

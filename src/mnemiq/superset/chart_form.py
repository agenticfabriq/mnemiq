"""Reading a saved chart's `form_data` for the value check: what the chart's query computes, as an `Expectation` that
mnemiq's own SQL can compute too, or why mnemiq cannot compute the same (`NotCheckable`, with mnemiq's own reason).

Read as Superset 7.0.0rc2 reads it: the grouping columns in the order the chart's query selects them, the metrics its
renderer shows, the SIMPLE filters, and the time grain of an x-axis. A chart whose shown numbers are computed after its
query, or summed by its renderer from the rows of a measure that is not a sum or a count, or filtered in a way mnemiq
does not translate, is not checkable, and the reason says which. `chart_check.py` reads Superset, runs mnemiq's
statement and compares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.charts import Filter, not_null_condition
from mnemiq.superset.display_format import metric_label
from mnemiq.superset.settings import superset_dialect

__all__ = ["DatasetFacts", "Expectation", "NotCheckable", "expectation_from_form_data"]


@dataclass(frozen=True)
class DatasetFacts:
    """The chart's dataset, as `get_dataset_info` gives it to the builder. `database_id`, `schema` and
    `saved_metric_sql` (each saved metric's name and expression) are what mnemiq's own statement needs beyond whether
    the dataset can be checked: where the table is, and what a saved metric the chart names computes."""

    dataset_id: int
    is_virtual: bool
    table: str | None
    sql: str | None
    saved_metric_names: frozenset[str]
    database_id: int | None = None
    schema: str | None = None
    saved_metric_sql: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Expectation:
    """What the chart's query computes: each metric as its label and its SQL expression (a saved metric as its name
    twice), the grouping columns, the time grain applied to the first of them (the x-axis), and the SIMPLE filters in
    Superset's operator spelling."""

    metrics: tuple[tuple[str, str], ...]
    dimensions: tuple[str, ...]
    time_grain: str | None
    filters: tuple[Filter, ...]
    # The metrics, by index, whose groups with no value the chart's query leaves out: the builder's own SQL HAVING
    # `(<the metric's SQL>) IS NOT NULL` (`not_null_condition`), and no other.
    not_null: tuple[int, ...] = ()


@dataclass(frozen=True)
class NotCheckable:
    reason: str


# The types whose query `get_chart_data` builds with the x-axis first (`build_query_dicts_from_form_data`) or whose
# plugin's `buildQuery` does (waterfall), and which `time_grain_sqla` groups through that x-axis.
_X_AXIS = frozenset({"echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area", "echarts_timeseries_scatter",
                     "echarts_timeseries_smooth", "echarts_timeseries_step", "echarts_timeseries", "mixed_timeseries",
                     "waterfall"})
# The types `Timeseries/transformProps.ts` renders (`Timeseries/index.ts`, `Regular/{Line,Bar,Scatter,SmoothLine}`,
# `Area`, `Step`): not the waterfall, and not the mixed chart, which has a renderer of its own.
_TIMESERIES = _X_AXIS - {"waterfall", "mixed_timeseries"}
# One metric, grouped by `groupby`.
_GROUPED_ONE_METRIC = frozenset({"pie", "gauge_chart", "treemap_v2", "funnel"})
_ONE_METRIC = _GROUPED_ONE_METRIC | {"big_number_total", "waterfall", "sankey_v2", "big_number"}
_NOT_CHECKED_TYPES = {
    "big_number": "a big number with a trendline shows a series Superset resamples after its query",
    "box_plot": "a box plot shows quartiles Superset computes after its query",
    "histogram_v2": "a histogram shows bins Superset computes after its query",
    "handlebars": "a handlebars chart shows its rows through a template, which holds no number mnemiq can read",
    "ag-grid-pivot-table": "the interactive pivot table's plugin is not in Superset 7.0.0rc2, so what it shows is "
                           "unknown",
}
# The pivot shows each cell as its value only for these `showValuesAs` settings (`ShowValuesAsEnum.ACTUAL` in
# `plugin-chart-pivot-table/src/types.ts`, the control's default); the others show it as a share of a total.
_PIVOT_SHOWS_VALUE = frozenset({None, "", "actual"})
# The pivot's total and subtotal toggles. With a metric that is not additive, its `buildQuery` asks for every total
# shown in the same query as grouping sets, whose total rows carry null grouping values.
_PIVOT_TOTALS = ("rowTotals", "colTotals", "rowSubTotals", "colSubTotals")
# What `allMetricsAdditive` (`plugin-chart-pivot-table/src/plugin/utilities.ts`) counts as additive: a SIMPLE metric
# with one of these aggregates, whose totals the plugin reduces from the cells by `additiveReducerFor` (a sum, the
# minimum, the maximum). It counts a saved or SQL metric as never additive.
_ADDITIVE = frozenset({"SUM", "COUNT", "MIN", "MAX"})
# Of those, the aggregates whose sum over some groups is the same aggregate over them all: a sum of sums, a sum of
# counts. A sum of minimums is not a minimum. A saved metric never is; an SQL metric is when its whole expression is one
# of these (`_one_sum_or_count`).
_SUMMABLE = frozenset({"SUM", "COUNT"})

# Postgres's grain expressions (`_time_grain_expressions` in `superset/db_engine_specs/postgres.py`), the ones that are
# one `DATE_TRUNC`.
_GRAINS = {"PT1S": "second", "PT1M": "minute", "PT1H": "hour", "P1D": "day", "P1W": "week", "P1M": "month",
           "P3M": "quarter", "P1Y": "year"}
# A SIMPLE metric's aggregate (`sqla_aggregations` in `superset/connectors/sqla/models.py`).
_AGGREGATES = {"SUM": "SUM({})", "AVG": "AVG({})", "MIN": "MIN({})", "MAX": "MAX({})", "COUNT": "COUNT({})",
               "COUNT_DISTINCT": "COUNT(DISTINCT {})"}
# Superset's SIMPLE filter operators (`FilterOperator` in `superset/utils/core.py`) mnemiq writes the same way, each as
# `get_sqla_query` in `superset/models/helpers.py` writes it.
_COMPARISONS = {"==": "=", "!=": "<>", ">": ">", "<": "<", ">=": ">=", "<=": "<="}
_PATTERNS = {"LIKE": "LIKE", "NOT LIKE": "NOT LIKE", "ILIKE": "ILIKE", "NOT ILIKE": "NOT ILIKE"}
_NO_VALUE = {"IS NULL": "IS NULL", "IS NOT NULL": "IS NOT NULL", "IS TRUE": "IS TRUE", "IS FALSE": "IS FALSE"}
_LISTS = {"IN", "NOT IN"}
# The comparator of the neutral time binding MCP adds (`_ensure_temporal_adhoc_filter` in
# `superset/mcp_service/chart/chart_utils.py`; `NO_TIME_RANGE` in `superset/constants.py`).
_NO_TIME_RANGE = "No filter"
# Values Superset reads as a null and an empty string (`NULL_STRING`, `EMPTY_STRING` in `superset/constants.py`).
_SUPERSET_MARKERS = frozenset({"<NULL>", "<empty string>"})

_SQL_FILTER = "the chart has a SQL filter, which mnemiq does not translate into its own SQL"


# ---------------------------------------------------------------- reading form_data


def _metric_items(viz_type: str, form: dict) -> list:
    """The metrics the chart shows, as its renderer reads them from `form_data`: a one-metric type's `metric` alone
    (each plugin's `transformProps` shows `formData.metric`: `Pie/transformProps.ts`, `Gauge/`, `Treemap/`, `Funnel/`,
    `Waterfall/`, `Sankey/`, `BigNumber/BigNumberTotal/`), any other type's `metrics` (else its `metric`), and a mixed
    chart's `metrics_b` after them.

    This is not how rc2's data tool reads them: `resolve_metrics_and_groupby` in
    `superset/mcp_service/chart/chart_helpers.py` reads `metric` first (else the first of `metrics`) only for
    big_number, big_number_total and pop_kpi, and for every other type `metrics` first, else `metric`. Where its rows
    then lack the metric the chart shows, `_line_up` (`chart_check.py`) finds them not to carry the chart's columns,
    and nothing is checked."""
    if viz_type in _ONE_METRIC:
        items = [form["metric"]] if form.get("metric") else []
    else:
        items = list(form.get("metrics") or []) or ([form["metric"]] if form.get("metric") else [])
    if viz_type == "mixed_timeseries":
        items += list(form.get("metrics_b") or [])
    return items


def _quoted(name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=superset_dialect())


def _metric(item, dataset: DatasetFacts) -> tuple[str, str] | NotCheckable:
    if isinstance(item, str):
        if item not in dataset.saved_metric_names:
            return NotCheckable(f"the chart names the saved metric {item!r}, which its dataset does not have")
        return item, item
    label = metric_label(item)
    if not isinstance(item, dict) or not label:
        return NotCheckable("the chart holds a metric mnemiq cannot read")
    if item.get("expressionType") == "SQL":
        expression = item.get("sqlExpression")
        if not isinstance(expression, str) or not expression.strip():
            return NotCheckable(f"the metric {label!r} has no SQL")
    elif item.get("expressionType") == "SIMPLE":
        column = item.get("column") if isinstance(item.get("column"), dict) else {}
        name, aggregate = column.get("column_name"), item.get("aggregate")
        if aggregate not in _AGGREGATES or not isinstance(name, str) or not name:
            return NotCheckable(f"the metric {label!r} uses an aggregate mnemiq does not reproduce")
        if column.get("expression"):
            return NotCheckable(f"the metric {label!r} aggregates a custom SQL column, which mnemiq does not translate")
        expression = _AGGREGATES[aggregate].format(_quoted(name))
    else:
        return NotCheckable(f"the metric {label!r} is of a kind mnemiq cannot read")
    if label in dataset.saved_metric_names:
        # Superset names a column by label, and a saved metric by its name, so the two could not be told apart.
        return NotCheckable(f"an ad hoc metric is labelled {label!r}, the name of one of the dataset's saved metrics")
    return label, expression


def _column_list(value) -> list[str] | NotCheckable:
    items = [value] if isinstance(value, (str, dict)) else list(value or [])
    if any(not isinstance(item, str) or not item for item in items):
        return NotCheckable("the chart groups by a custom SQL column, which mnemiq does not translate")
    return items


def _table_of_rows(form: dict) -> bool:
    """Whether a table shows rows, not groups, as `getQueryMode` (`plugin-chart-table/src/buildQuery.ts`) reads it: the
    mode saved, else rows when it has columns of rows chosen."""
    mode = form.get("query_mode")
    if mode in ("aggregate", "raw"):
        return mode == "raw"
    return bool(form.get("all_columns"))


def _dimensions(viz_type: str, form: dict) -> list[str] | NotCheckable:
    """The chart's grouping columns, in the order its query selects them."""
    if viz_type in _X_AXIS:
        x_axis = form.get("x_axis")
        if not x_axis:
            return NotCheckable("the chart has no x-axis column")
        found = _column_list([x_axis, *_as_list(form.get("groupby"))])
    elif viz_type in _GROUPED_ONE_METRIC or viz_type == "table":
        if viz_type == "table" and _table_of_rows(form):
            return NotCheckable("a table of rows has no measure: its numbers are the rows themselves")
        found = _column_list(form.get("groupby"))
    elif viz_type == "pivot_table_v2":
        found = _column_list([*_as_list(form.get("groupbyRows")), *_as_list(form.get("groupbyColumns"))])
    elif viz_type == "sankey_v2":
        if not (form.get("source") and form.get("target")):
            return NotCheckable("the sankey chart has no source or no target")
        found = _column_list([form.get("source"), form.get("target")])
    elif viz_type == "big_number_total":
        found = []
    else:
        return NotCheckable(f"mnemiq does not check a {viz_type!r} chart")
    if isinstance(found, NotCheckable):
        return found
    return list(dict.fromkeys(found))


def _as_list(value) -> list:
    return [value] if isinstance(value, (str, dict)) else list(value or [])


def _after_the_query(viz_type: str, form: dict) -> str | None:
    """What the chart computes after its query, which `get_chart_data`'s rebuild never applies, or None."""
    suffixes = ("", "_b") if viz_type == "mixed_timeseries" else ("",)
    for suffix in suffixes:
        if str(form.get(f"rolling_type{suffix}") or "None") != "None":
            return "a rolling window"
        if form.get(f"time_compare{suffix}"):
            return "a time comparison"
        if form.get(f"resample_rule{suffix}"):
            return "resampling"
    if form.get("contributionMode"):
        return "a contribution, each value as its share of a total"
    if form.get("percent_metrics"):
        return "percentage metrics"
    if form.get("stack") == "Expand":
        return "series stacked to 100%"
    if form.get("forecastEnabled"):
        return "a forecast"
    # The line chart's control (`Timeseries/Regular/Line/controlPanel.tsx`), which `Timeseries/transformProps.ts` reads
    # camel-cased and applies to every point.
    if form.get("rebase_percent_change") or form.get("rebasePercentChange"):
        return "each series as its percent change from its first point"
    return None


def _set(form: dict, *keys: str) -> bool:
    """Whether a renderer reads any of `keys` as set, as JavaScript reads a value as true: not absent, null, false, 0 or
    the empty string. A saved chart holds the control's own name; the renderer reads it camel-cased."""
    return any(form.get(key) not in (None, False, 0, "") for key in keys)


def _one_sum_or_count(expression) -> bool:
    """Whether an SQL metric's whole expression, parsed as Postgres, is one SUM, or one COUNT of a column or of every
    row, and nothing more: no DISTINCT, no FILTER or OVER, nothing around the aggregate, no aggregate, window or query
    within the sum. mnemiq writes every measure as an SQL metric, so this is how a measure's totals are known to be the
    sums of its groups. SQL that does not parse, or is not one expression, is not, and nor is SQL holding a template
    marker: Superset renders it before running it, and sqlglot reads `{{ col }}` as something else (a struct)."""
    if not isinstance(expression, str) or not expression.strip():
        return False
    if any(marker in expression for marker in JINJA_DELIMITERS):
        return False
    try:
        node = sqlglot.parse_one(expression, read=superset_dialect())
    except (SqlglotError, ValueError):
        return False
    if isinstance(node, exp.Sum):
        inner = node.this
        return (isinstance(inner, exp.Expression) and not isinstance(inner, exp.Distinct)
                and inner.find(exp.AggFunc, exp.Window, exp.Subquery, exp.Select) is None)
    if isinstance(node, exp.Count):
        return isinstance(node.this, (exp.Star, exp.Column)) and not node.expressions
    return False


def _summable(item) -> bool:
    """Whether a sum of the metric's values over some groups is the metric over them all: a SIMPLE SUM or COUNT, or an
    SQL metric that is one SUM or one COUNT whole (`_one_sum_or_count`). A saved metric is not read here, so it never
    is."""
    if not isinstance(item, dict):
        return False
    if item.get("expressionType") == "SIMPLE":
        return item.get("aggregate") in _SUMMABLE
    return item.get("expressionType") == "SQL" and _one_sum_or_count(item.get("sqlExpression"))


def _several(metrics: list, groupby) -> bool:
    """Whether a chart's metrics and grouping make several series, as `isMultiSeries = groupBy.length ||
    metrics?.length > 1` reads them (`Timeseries/transformProps.ts`)."""
    return bool(_as_list(groupby)) or len(metrics) > 1


def _series_total(viz_type: str, form: dict) -> str | None:
    """A total across the chart's series that its renderer sums and shows, which no query returns, or None. It is a
    measure's own number only when the series are the groups of one SUM or COUNT; two measures added together never
    are.

    - The time series' rich tooltip (`Timeseries/transformProps.ts`) adds a "Total" row when `allowTotal &&
      showTooltipTotal`, and shows each series as a share of it when `showTooltipPercentage`, where `allowTotal =
      Boolean(isMultiSeries) && richTooltip && !isForecast`. The renderer reads the form over `DEFAULT_FORM_DATA`
      (`Timeseries/constants.ts`), where `richTooltip` is true and there is no `showTooltipTotal`, while the Explore
      control defaults it to true. A saved key is read in either spelling, since the renderer reads them camel-cased.
      The mixed chart's tooltip shows no total.
    - A stack labelled with its total (`stack`, `only_total` and `show_value`): the top series' label shows
      `totalStackedValues` (`transformSeries` in `Timeseries/transformers.ts`). Each side of a mixed chart does so with
      its own keys (`stackB`, `only_totalB`, `show_valueB`, read as `stackB`, `onlyTotalB`, `showValueB` in
      `MixedTimeseries/transformProps.ts`)."""
    if viz_type in _TIMESERIES:
        rich_keys = [key for key in ("rich_tooltip", "richTooltip") if key in form]
        rich = not rich_keys or _set(form, *rich_keys)
        tooltip = rich and _set(form, "showTooltipTotal", "show_tooltip_total", "showTooltipPercentage",
                                "show_tooltip_percentage")
        sides = [("", _metric_items(viz_type, form), form.get("groupby"))]
    elif viz_type == "mixed_timeseries":
        tooltip = False
        sides = [("", list(form.get("metrics") or []), form.get("groupby")),
                 ("B", list(form.get("metrics_b") or []), form.get("groupby_b"))]
    else:
        return None
    for suffix, metrics, groupby in sides:
        if not _several(metrics, groupby) or (len(metrics) == 1 and _summable(metrics[0])):
            continue
        if tooltip:
            return ("the chart's tooltip shows a total of its series, a sum its renderer computes across them, which "
                    "is a measure's own number only for one measure that is a SUM or a COUNT")
        if form.get(f"stack{suffix}") and _set(form, f"only_total{suffix}", f"onlyTotal{suffix}") \
                and _set(form, f"show_value{suffix}", f"showValue{suffix}"):
            return ("the chart labels each stack with the total of its series, a sum its renderer computes across "
                    "them, which is a measure's own number only for one measure that is a SUM or a COUNT")
    return None


def _renderer_sum(viz_type: str, form: dict, dimensions: list[str]) -> str | None:
    """A number the chart shows that its renderer sums from the query's rows, which neither side's rows hold, or None.
    Each is a sum of numbers the check does compare, so it is the measure's own only for a SUM or a COUNT; a pie's
    Other slice also hides the slices it folds in."""
    if (total := _series_total(viz_type, form)) is not None:
        return total
    if viz_type == "pie" and _set(form, "threshold_for_other", "thresholdForOther"):
        # `Pie/transformProps.ts`: `if (thresholdForOther)` folds the slices under it into one, `otherSum`.
        return ("the pie folds its smallest slices into an Other slice its renderer sums, which mnemiq does not "
                "reproduce")
    if viz_type == "pie" and _set(form, "show_total", "showTotal"):
        # `Pie/transformProps.ts`: `totalValue`, shown as `t('Total: %s', ...)` when `showTotal`.
        shown = "the pie's total"
    elif viz_type == "treemap_v2" and len(dimensions) > 1:
        # `treeBuilder` (`plugin-chart-echarts/src/utils/treeBuilder.ts`) sets each parent's value to the sum of its
        # children's, which `Treemap/transformProps.ts` labels and shows in `formatTooltip`.
        shown = "each of the treemap's parents"
    elif viz_type == "waterfall":
        # `Waterfall/transformProps.ts`: `formatTooltip` shows every bar's `totalSum`, the running sum of the steps so
        # far, whatever `showTotal` says; with `showTotal` (the `show_total` control, which MCP's
        # `map_waterfall_config` writes, True by default) `transformer` also adds total bars summed from the steps.
        shown = "the waterfall's running totals"
    elif viz_type == "sankey_v2":
        # `Sankey/transformProps.ts`: its `seriesData` gives a node no value of its own, so each node shows the sum of
        # its links, and `nodeValues` (the larger of a node's `incomingFlows` and `outgoingFlows` sums) is what
        # `tooltipFormatter` shows each link as a share of.
        shown = "each of the sankey's nodes"
    else:
        return None
    if all(_summable(item) for item in _metric_items(viz_type, form)):
        return None
    return (f"the chart shows {shown} as a sum its renderer computes from the groups, which is that measure's own "
            "number only for a SUM or a COUNT")


def _neutral_range(value) -> str | None:
    return None if value in (None, "", _NO_TIME_RANGE) else value


def _row_limit(form: dict) -> int | None:
    try:
        limit = int(form.get("row_limit"))
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


@dataclass(frozen=True)
class Ranking:
    """How the chart's query ranks its rows under its row limit: `ORDER BY <metric> DESC|ASC LIMIT <limit>`, the
    metric being the expectation's metric at `metric_index`. rc2 adds no NULLS LAST, so Postgres puts NULL first under
    DESC, and no tiebreak."""

    metric_index: int
    descending: bool
    limit: int


def _saved_flag(value, *, absent: bool) -> bool | None:
    """A saved boolean the query reads (`order_desc`, `sort_by_metric`): absent is `absent`; anything but a boolean is
    not read, since the frontend and the backend may read it apart."""
    if value is _ABSENT:
        return absent
    return value if isinstance(value, bool) else None


_ABSENT = object()


def _is_form_metric(item) -> bool:
    """`isQueryFormMetric` (`superset-ui-core`): a saved metric's name, any string, or a SIMPLE or SQL ad hoc one."""
    return isinstance(item, str) or (isinstance(item, dict) and item.get("expressionType") in ("SIMPLE", "SQL"))


def _ranked_metric(item, expectation: Expectation, dataset: DatasetFacts) -> int | None:
    found = _metric(item, dataset)
    if isinstance(found, NotCheckable) or found not in expectation.metrics:
        return None
    return expectation.metrics.index(found)


# The form keys `extractQueryFields` (`superset-ui-core`) gathers into a query's metrics besides `metrics`, in the
# form's key order: one holding a value can come first in the query's metrics, and the query then ranks by it.
_OTHER_METRIC_ALIASES = ("metric", "metric_2", "secondary_metric", "left_metric", "right_metric", "x", "y", "size")


def ranking_of(form: dict, expectation: Expectation, *, dataset: DatasetFacts) -> Ranking | None:
    """The ranking the chart's query cuts its rows by, read from its saved form exactly as rc2 orders it, or None where
    mnemiq does not read one: a type whose query does not rank by its measure (mixed, pivot, funnel, sankey,
    waterfall), a sort mnemiq does not read, or a ranking metric the chart does not show.

    - The time-series types: `normalizeOrderBy` (`superset-ui-core`), by the series limit metric `buildQueryObject`
      picks (`series_limit_metric` when it is a metric, a string included, else `timeseries_limit_metric`), else, when
      that is empty, `metrics[0]`; descending unless `order_desc` is false (`buildQueryObject` reads it absent as
      true). A valid `order_by_cols` or `orderby` (both reach the query through `extractQueryFields`) would take
      precedence, so any is not read.
    - A table (`plugin-chart-table` `buildQuery`): `timeseries_limit_metric`, ascending unless `order_desc` is true,
      else `metrics[0]` descending whatever `order_desc` says.
    - A pie, gauge or treemap: its `metric`, descending, only when `sort_by_metric` is on: each plugin's `buildQuery`
      orders by the metric only then, and the treemap breaks ties by its groups, which mnemiq accepts either way.

    A query that skips rows (`row_offset`, which `buildQueryObject` passes on) keeps rows that are not its first N,
    and a time-series chart or table with another metric alias holding a value may rank by that alias: neither is
    read."""
    limit = _row_limit(form)
    viz_type = str(form.get("viz_type") or "")
    if limit is None or form.get("row_offset") not in (None, 0):
        return None
    if (viz_type in _TIMESERIES or viz_type == "table") and any(form.get(alias) for alias in _OTHER_METRIC_ALIASES):
        return None
    if viz_type in _TIMESERIES:
        if form.get("order_by_cols") or form.get("orderby"):
            return None
        item = form.get("series_limit_metric")
        if not _is_form_metric(item):
            item = form.get("timeseries_limit_metric")
        if item is None or item in ("", {}, []):
            item = (_as_list(form.get("metrics")) or [None])[0]
        descending = _saved_flag(form.get("order_desc", _ABSENT), absent=True)
    elif viz_type == "table":
        item = form.get("timeseries_limit_metric")
        if item:
            descending = _saved_flag(form.get("order_desc", _ABSENT), absent=False)
        else:
            item, descending = (_as_list(form.get("metrics")) or [None])[0], True
    elif viz_type in ("pie", "gauge_chart", "treemap_v2"):
        if _saved_flag(form.get("sort_by_metric", _ABSENT), absent=False) is not True:
            return None
        item, descending = form.get("metric"), True
    else:
        return None
    if item is None or descending is None:
        return None
    index = _ranked_metric(item, expectation, dataset)
    return None if index is None else Ranking(index, descending, limit)


def _second_series(form: dict, dimensions: list[str]) -> NotCheckable | None:
    """Why a mixed chart's second series is shown otherwise than Superset's rows hold it, or None. The renderer takes
    each `_b` key the form holds in place of the primary's, an empty `adhoc_filters_b` included
    (`retainFormDataSuffix` in `plugin-chart-echarts/src/utils/formDataSuffix.ts`); the data tool's rebuild
    (`_build_mixed_timeseries_secondary` in `superset/mcp_service/chart/chart_helpers.py`) reads an empty one as
    absent and filters the series with the primary's filters. It reads `row_limit_b` only when its caller passes no
    row limit, and `get_chart_data` always passes one (`_coerce_row_limit` of the request's limit or the form's
    `row_limit`), so the second series comes back under the primary's limit. So each `_b` key present must say what
    the primary's says."""
    if _as_list(form.get("groupby_b")) != _as_list(form.get("groupby")):
        return NotCheckable("the mixed chart groups its second series differently from its first")
    if "adhoc_filters_b" in form:
        mine, theirs = _filters({"adhoc_filters": form["adhoc_filters_b"]}), _filters(form)
        if isinstance(mine, NotCheckable) or (not isinstance(theirs, NotCheckable) and mine != theirs):
            return NotCheckable("the mixed chart filters its second series differently from its first")
    if "time_range_b" in form and _neutral_range(form["time_range_b"]) != _neutral_range(form.get("time_range")):
        return NotCheckable("the mixed chart keeps its second series within another time range")
    if "row_limit_b" in form and _row_limit({"row_limit": form["row_limit_b"]}) != _row_limit(form):
        return NotCheckable("the mixed chart limits the rows of its second series differently from its first")
    if len(dimensions) > 1 and (form.get("series_limit_b") or form.get("limit_b")):
        return NotCheckable("the mixed chart keeps only the top groups of its second series (a series limit)")
    return None


def _pivot(form: dict) -> NotCheckable | None:
    """Why a pivot table shows its cells otherwise than Superset's rows hold them, or None."""
    if form.get("showValuesAs") not in _PIVOT_SHOWS_VALUE:
        return NotCheckable("the pivot table shows each cell as its share of a total")
    # The totals shown are the cells' own only for a metric whose total the cells give: one the plugin reduces from them
    # itself, or an SQL SUM or COUNT, whose totals from grouping sets are the sums of its cells.
    additive = [(isinstance(m, dict) and m.get("expressionType") == "SIMPLE" and m.get("aggregate") in _ADDITIVE)
                or _summable(m) for m in form.get("metrics") or []]
    if not (additive and all(additive)) and any(form.get(key) for key in _PIVOT_TOTALS):
        return NotCheckable("the pivot table shows totals of a metric that is not a sum or a count, which Superset "
                            "computes in the same query as its cells (grouping sets) and mnemiq does not reproduce")
    if (form.get("currency_format") or {}).get("symbol") == "AUTO":
        return NotCheckable("the pivot table's AUTO currency adds the dataset's currency column to its grouping")
    return None


def _scalar(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (str, int)) and value not in _SUPERSET_MARKERS


def _filters(form: dict) -> tuple[Filter, ...] | NotCheckable:
    """The chart's filters, as `prepare_form_data_for_query` and `split_adhoc_filters_into_base_filters`
    (`superset/utils/core.py`) gather them: SIMPLE WHERE filters only, the neutral time binding dropped."""
    if any(isinstance(form.get(key), str) and form[key].strip() for key in ("where", "having")):
        return NotCheckable(_SQL_FILTER)
    if form.get("extra_filters") or form.get("extra_form_data") or form.get("filters"):
        return NotCheckable("the chart carries filters from a dashboard or an older Superset form, which mnemiq does "
                            "not translate")
    if form.get("time_range") not in (None, "", _NO_TIME_RANGE):
        return NotCheckable("the chart keeps rows within a time range, whose bounds Superset reads from the time now")
    found = []
    for item in form.get("adhoc_filters") or []:
        if not isinstance(item, dict):
            return NotCheckable("the chart holds a filter mnemiq cannot read")
        if item.get("expressionType") == "SQL":
            return NotCheckable(_SQL_FILTER)
        if item.get("expressionType") != "SIMPLE":
            return NotCheckable("the chart holds a filter mnemiq cannot read")
        if item.get("clause", "WHERE") != "WHERE":
            return NotCheckable("the chart filters its aggregated values (HAVING), which mnemiq does not translate")
        column, op, value = item.get("subject"), item.get("operator"), item.get("comparator")
        if not isinstance(column, str) or not column:
            return NotCheckable("the chart filters on a custom SQL column, which mnemiq does not translate")
        if op == "TEMPORAL_RANGE":
            if value in (None, _NO_TIME_RANGE):
                continue
            return NotCheckable("the chart keeps rows within a time range, whose bounds Superset reads from the time "
                                "now")
        if op in _NO_VALUE:
            found.append(Filter(column, op, None))
        elif op in _LISTS:
            values = _as_list(value)
            if not values or not all(_scalar(v) for v in values):
                return NotCheckable(f"the chart's filter on {column!r} holds a value mnemiq does not translate")
            found.append(Filter(column, op, values))
        elif op in _COMPARISONS or op in _PATTERNS:
            if not _scalar(value):
                return NotCheckable(f"the chart's filter on {column!r} holds a value mnemiq does not translate")
            found.append(Filter(column, op, value))
        else:
            return NotCheckable(f"the chart filters with {op!r}, an operator mnemiq does not translate")
    return tuple(found)


def expectation_from_form_data(form_data: dict, *, dataset: DatasetFacts) -> Expectation | NotCheckable:
    """What the chart's query computes, read from its `form_data` (whose `viz_type` names its type), or why mnemiq
    cannot compute the same."""
    form = form_data
    viz_type = str(form.get("viz_type") or "")
    if viz_type in _NOT_CHECKED_TYPES:
        return NotCheckable(_NOT_CHECKED_TYPES[viz_type])
    dimensions = _dimensions(viz_type, form)
    if isinstance(dimensions, NotCheckable):
        return dimensions
    if (after := _after_the_query(viz_type, form)) is not None:
        return NotCheckable(f"the chart shows numbers computed after its query ({after}), which mnemiq does not "
                            "reproduce")
    if (summed := _renderer_sum(viz_type, form, dimensions)) is not None:
        return NotCheckable(summed)
    if viz_type == "table" and _set(form, "show_totals"):
        # `plugin-chart-table/src/buildQuery.ts` adds, for `formData.show_totals` in aggregate mode, a query of its own
        # over no grouping (`getTotalsMetrics`, `toTotalsAggregate`), which the data tool's rebuild leaves out and
        # whose one row `_line_up` (`chart_check.py`) passes over in a saved query context's answer.
        return NotCheckable("the table shows a totals row Superset computes in a query of its own, which mnemiq does "
                            "not reproduce")
    if viz_type in _X_AXIS and len(dimensions) > 1 and (form.get("series_limit") or form.get("limit")):
        return NotCheckable("the chart keeps only its top series (a series limit), which Superset's data tool does not "
                            "apply")
    if viz_type == "mixed_timeseries" and (second := _second_series(form, dimensions)) is not None:
        return second
    if viz_type == "pivot_table_v2" and (pivot := _pivot(form)) is not None:
        return pivot
    grain = form.get("time_grain_sqla") or None
    if grain is not None and viz_type in ("table", "pivot_table_v2"):
        return NotCheckable("the chart applies a time grain to whichever of its grouping columns the dataset marks "
                            "temporal, which mnemiq does not read")
    if grain is not None and viz_type not in _X_AXIS and form.get("x_axis") in dimensions:
        # `buildQueryContext` runs `normalizeTimeColumn` (`superset-ui-core/src/query/normalizeTimeColumn.ts`) for
        # every chart, so a grouping column that is also the form's `x_axis` is shown by the grain.
        return NotCheckable("the chart groups its x-axis column by a time grain, which Superset's data tool does not "
                            "apply")
    if viz_type not in _X_AXIS:
        grain = None
    if grain is not None and grain not in _GRAINS:
        return NotCheckable(f"the chart's time grain {grain!r} is one mnemiq does not reproduce")
    metrics = []
    for item in _metric_items(viz_type, form):
        metric = _metric(item, dataset)
        if isinstance(metric, NotCheckable):
            return metric
        metrics.append(metric)
    if not metrics:
        return NotCheckable("the chart has no metric to compare")
    labels = [label for label, _ in metrics]
    if len(set(labels)) != len(labels):
        return NotCheckable("two of the chart's metrics share a label, so their values cannot be told apart")
    for label in labels:
        if label.casefold() in {name.casefold() for name in dimensions}:
            return NotCheckable(f"the metric label {label!r} is also the name of a grouping column, so their values "
                                "cannot be told apart")
    # Only on the types the builder gives the HAVING to, those it ranks; on any other it is a SQL filter as before.
    not_null, others = _not_null(form, metrics, dataset) if viz_type in _LEFT_OUT_TYPES else ((), [])
    filters = _filters({**form, "adhoc_filters": others} if not_null else form)
    if isinstance(filters, NotCheckable):
        return filters
    return Expectation(metrics=tuple(metrics), dimensions=tuple(dimensions), time_grain=grain, filters=filters,
                       not_null=not_null)


# The chart types the builder makes with a HAVING that leaves out groups with no value (`charts.leaves_out_no_value`).
_LEFT_OUT_TYPES = frozenset({"echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area",
                             "echarts_timeseries_scatter", "table", "pie", "gauge_chart", "treemap_v2"})


def _not_null(form: dict, metrics: list[tuple[str, str]], dataset: DatasetFacts) -> tuple[tuple[int, ...], list]:
    """The metrics whose groups with no value the chart leaves out by the builder's own SQL HAVING, exactly
    `not_null_condition(<the metric's SQL>)` (a saved metric's SQL as the dataset holds it), at most one per metric;
    and every other filter, which `_filters` reads as before."""
    saved = dict(dataset.saved_metric_sql)
    conditions = {not_null_condition(saved.get(label, sql) if label in dataset.saved_metric_names else sql): index
                  for index, (label, sql) in enumerate(metrics)}
    found: list[int] = []
    others = []
    for item in form.get("adhoc_filters") or []:
        text = item.get("sqlExpression") if isinstance(item, dict) else None
        index = conditions.get(text.strip()) if isinstance(text, str) else None
        if isinstance(item, dict) and item.get("clause") == "HAVING" and item.get("expressionType") == "SQL" \
                and index is not None and index not in found:
            found.append(index)
        else:
            others.append(item)
    return tuple(sorted(found)), others

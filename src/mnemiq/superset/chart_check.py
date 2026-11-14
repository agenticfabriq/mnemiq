"""The value check: whether a saved chart shows the numbers mnemiq computes for it, in a format that fits each measure.

What to compare is read from the chart's current `form_data`, never from mnemiq's plan, so a chart changed by hand is
checked as it now is. Both sides are read as the builder: the chart's rows through MCP's `get_chart_data`, and mnemiq's
own through the request's adapter, its statement decided as every ask-path statement is. The values ride back in the
`ChartCheck` for the response and are kept nowhere, logged nowhere.

Whatever mnemiq cannot reproduce faithfully is `NOT_CHECKED`, with mnemiq's reason, and never `CHECKED`. Three facts of
Superset 7.0.0rc2 decide most of it:

- `get_chart_data` (`superset/mcp_service/chart/tool/get_chart_data.py`) runs a chart's saved query context when it has
  one, and otherwise rebuilds one from `form_data` (`build_query_dicts_from_form_data` in
  `superset/mcp_service/chart/chart_helpers.py`) with no post-processing. Every chart MCP's `generate_chart` saves has
  none. So a chart whose shown numbers come after the query (a rolling window, a time comparison, a contribution, a
  forecast, quartiles, bins) is not checked.
- That rebuild puts a string x-axis among the query's columns as it is: `time_grain_sqla` reaches a query only through a
  `BASE_AXIS` column (`apply_time_grain_to_base_axis` in `superset/common/utils/time_grain_utils.py`;
  `adhoc_column_to_sqla` in `superset/connectors/sqla/models.py`). A chart with a time grain is compared grouped by the
  grain first, as the chart shows it; when Superset's rows instead match mnemiq's without the grain, the numbers the
  chart shows were never in Superset's answer, and the chart is not checked, saying so.
- Superset's text never enters a reason: a refusal can name a table or a user the caller may not see. It goes to the
  log, as the client logs it.

A virtual dataset is never checked with mnemiq's own SQL: a row rule attached to the virtual dataset alone applies to
queries of that dataset, not to SQL over its tables. A saved metric on one is checked through `query_dataset`
(`superset/mcp_service/dataset/tool/query_dataset.py`), which takes saved metric names only, and a measure written on
the chart is not checked.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import StrEnum

from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.execute.runner import DEFAULT_TIMEOUT_S, ExecutionError
from mnemiq.sql.guard import MAX_ROWS
from mnemiq.sql.verdict import Refusal
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.charts import Filter
from mnemiq.superset.client import SupersetToolError, SupersetUnavailable
from mnemiq.superset.display_format import CANNOT_TELL, Displayed, displayed, fits
from mnemiq.superset.measures import Measure, MeasureKind, measure_statement, run_decided
from mnemiq.superset.saved_metrics import SavedMetric

__all__ = ["ChartCheck", "ChartStatus", "DatasetFacts", "Expectation", "NotCheckable", "check_chart",
           "expectation_from_form_data", "read_dataset_facts"]

logger = logging.getLogger(__name__)


class ChartStatus(StrEnum):
    CHECKED = "checked"
    MISMATCH = "mismatch"
    EMPTY = "empty"
    CUT = "cut"
    FORMAT_WRONG = "format_wrong"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class ChartCheck:
    """One chart's check. `chart_values` and `mnemiq_values` are the rows each side gave, each a tuple of the grouping
    values then the metric values, for the response only."""

    chart_id: int
    status: ChartStatus
    reason: str
    chart_values: tuple | None
    mnemiq_values: tuple | None


@dataclass(frozen=True)
class DatasetFacts:
    """The chart's dataset, as `get_dataset_info` gives it to the builder. `database_id`, `schema` and
    `saved_metric_sql` (each saved metric's name and expression) are what mnemiq's own statement needs beyond the
    plan's five fields: where the table is, and what a saved metric the chart names computes."""

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


@dataclass(frozen=True)
class NotCheckable:
    reason: str


# `get_chart_info` returns `form_data` only when asked for it (`DEFAULT_GET_CHART_INFO_COLUMNS` in
# `superset/mcp_service/chart/schemas.py` leaves it out).
_CHART_FIELDS = ["id", "slice_name", "viz_type", "form_data", "changed_on"]
_DATASET_FIELDS = ["id", "table_name", "schema", "database_id", "is_virtual", "sql", "metrics"]

# The types whose query `get_chart_data` builds with the x-axis first (`build_query_dicts_from_form_data`) or whose
# plugin's `buildQuery` does (waterfall), and which `time_grain_sqla` groups through that x-axis.
_X_AXIS = frozenset({"echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area", "echarts_timeseries_scatter",
                     "echarts_timeseries_smooth", "echarts_timeseries_step", "echarts_timeseries", "mixed_timeseries",
                     "waterfall"})
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
# with one of these aggregates. A saved or SQL metric never is.
_ADDITIVE = frozenset({"SUM", "COUNT", "MIN", "MAX"})

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

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}(:?\d{2})?)?)?")

_SQL_FILTER = "the chart has a SQL filter, which mnemiq does not translate into its own SQL"
_UNREADABLE = "mnemiq could not read the chart's settings: "


# ---------------------------------------------------------------- reading form_data


def _label(item) -> str | None:
    """The column Superset names a metric's values by (`get_metric_name` in `superset/utils/core.py`)."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("label"), str) and item["label"]:
        return item["label"]
    if item.get("expressionType") == "SQL" and isinstance(item.get("sqlExpression"), str):
        return item["sqlExpression"]
    column = (item.get("column") or {}).get("column_name") if isinstance(item.get("column"), dict) else None
    if item.get("expressionType") == "SIMPLE" and column and item.get("aggregate"):
        return f"{item['aggregate']}({column})"
    return column


def _metric_items(viz_type: str, form: dict) -> list:
    """The chart's metrics as `form_data` holds them, read as `resolve_metrics_and_groupby` does: a one-metric type's
    `metric` (else its first of `metrics`), any other type's `metrics` (else its `metric`), and a mixed chart's
    `metrics_b` after them."""
    if viz_type in _ONE_METRIC:
        one = form.get("metric") or next(iter(form.get("metrics") or []), None)
        items = [one] if one else []
    else:
        items = list(form.get("metrics") or []) or ([form["metric"]] if form.get("metric") else [])
    if viz_type == "mixed_timeseries":
        items += list(form.get("metrics_b") or [])
    return items


def _quoted(name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect="postgres")


def _metric(item, dataset: DatasetFacts) -> tuple[str, str] | NotCheckable:
    if isinstance(item, str):
        if item not in dataset.saved_metric_names:
            return NotCheckable(f"the chart names the saved metric {item!r}, which its dataset does not have")
        return item, item
    label = _label(item)
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


def _dimensions(viz_type: str, form: dict) -> list[str] | NotCheckable:
    """The chart's grouping columns, in the order its query selects them."""
    if viz_type in _X_AXIS:
        x_axis = form.get("x_axis")
        if not x_axis:
            return NotCheckable("the chart has no x-axis column")
        found = _column_list([x_axis, *_as_list(form.get("groupby"))])
    elif viz_type in _GROUPED_ONE_METRIC or viz_type == "table":
        if viz_type == "table" and form.get("query_mode") == "raw":
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


def _neutral_range(value) -> str | None:
    return None if value in (None, "", _NO_TIME_RANGE) else value


def _second_series(form: dict, dimensions: list[str]) -> NotCheckable | None:
    """Why a mixed chart's second series is shown otherwise than Superset's rows hold it, or None. The renderer takes
    each `_b` key the form holds in place of the primary's, an empty `adhoc_filters_b` included
    (`retainFormDataSuffix` in `plugin-chart-echarts/src/utils/formDataSuffix.ts`); the data tool's rebuild
    (`_build_mixed_timeseries_secondary` in `superset/mcp_service/chart/chart_helpers.py`) reads an empty one as
    absent and filters the series with the primary's filters, and never reads `row_limit_b`. So each `_b` key present
    must say what the primary's says."""
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
    additive = [isinstance(m, dict) and m.get("expressionType") == "SIMPLE" and m.get("aggregate") in _ADDITIVE
                for m in form.get("metrics") or []]
    if not (additive and all(additive)) and any(form.get(key) for key in _PIVOT_TOTALS):
        return NotCheckable("the pivot table asks for its totals in the same query as its cells (grouping sets), whose "
                            "total rows mnemiq does not reproduce")
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
    filters = _filters(form)
    if isinstance(filters, NotCheckable):
        return filters
    return Expectation(metrics=tuple(metrics), dimensions=tuple(dimensions), time_grain=grain, filters=filters)


# ---------------------------------------------------------------- reading Superset


def _failed(answer: dict) -> bool:
    """An MCP error model answered as the tool's result (`MCPBaseError`, `DatasetError`): it carries `error_type`."""
    return bool(answer.get("error_type") or answer.get("error"))


def _log_error(what: str, answer: dict) -> None:
    logger.warning("the value check: Superset answered %s with an error (%s): %s", what, answer.get("error_type"),
                   answer.get("error") or answer.get("message"))


def _read_chart(client, chart_id: int) -> tuple[str, dict] | NotCheckable:
    """The chart's type and `form_data`, the type as the chart row holds it (`get_chart_data` reads it the same way)."""
    try:
        answer = client.call("get_chart_info", {"identifier": chart_id, "select_columns": _CHART_FIELDS})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"{_UNREADABLE}{failure}")
    if _failed(answer):
        _log_error("get_chart_info", answer)
        return NotCheckable(f"{_UNREADABLE}Superset answered with an error, which is in the log")
    form = answer.get("form_data")
    if answer.get("_response_truncated") or (isinstance(form, dict) and form.get("_truncated")):
        # The size guard (`truncate_oversized_response` in `superset/mcp_service/utils/response_size_utils.py`) clips
        # strings and summarizes dicts anywhere in the answer, `form_data` among them.
        return NotCheckable(f"{_UNREADABLE}Superset cut them to fit its response size")
    if not isinstance(form, dict):
        # `redact_chart_data_model_fields` (`superset/mcp_service/privacy.py`) withholds them from a caller without
        # the data-model privilege.
        return NotCheckable(f"{_UNREADABLE}Superset does not show them to this builder")
    viz_type = answer.get("viz_type") or form.get("viz_type") or ""
    return viz_type, {**form, "viz_type": viz_type}


def _dataset_of(form: dict, dataset_id: int) -> int | NotCheckable:
    """The dataset the chart reads now: the one its `form_data` names once saved in Explore, as
    `resolve_form_data_datasource` reads it, else the one mnemiq built it on (MCP's `generate_chart` saves no
    `datasource` in the chart's params, and `get_chart_info` returns no dataset id)."""
    combined = form.get("datasource")
    if isinstance(combined, str) and "__" in combined:
        head, _, kind = combined.partition("__")
        if kind != "table" or not head.isdecimal():
            return NotCheckable("the chart reads a datasource that is not a dataset mnemiq checks")
        return int(head)
    return dataset_id


def read_dataset_facts(client, dataset_id: int) -> DatasetFacts | NotCheckable:
    try:
        answer = client.call("get_dataset_info", {"identifier": dataset_id, "select_columns": _DATASET_FIELDS})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"mnemiq could not read the chart's dataset: {failure}")
    if _failed(answer):
        _log_error("get_dataset_info", answer)
        return NotCheckable("mnemiq could not read the chart's dataset: Superset answered with an error, which is in "
                            "the log")
    if answer.get("_response_truncated"):
        return NotCheckable("mnemiq could not read the chart's dataset whole: Superset cut it to fit its response size")
    metrics = [m for m in answer.get("metrics") or [] if isinstance(m, dict) and isinstance(m.get("metric_name"), str)]
    sql = answer.get("sql")
    return DatasetFacts(
        dataset_id=dataset_id,
        # Fails closed: a dataset that does not say it is physical, or that carries SQL, is read as virtual.
        is_virtual=answer.get("is_virtual") is not False or bool(sql),
        table=answer.get("table_name"), sql=sql,
        saved_metric_names=frozenset(m["metric_name"] for m in metrics),
        database_id=answer.get("database_id"), schema=answer.get("schema"),
        saved_metric_sql=tuple((m["metric_name"], m.get("expression") or "") for m in metrics))


@dataclass(frozen=True)
class _Rows:
    """One side's rows: each group's metric values by its grouping values, in the order the side gave them."""

    values: dict[tuple, list]
    largest: int = 0  # the most rows any one of the side's results held
    cap: int | None = None  # the row cap put on the side's one read, when it reached it: it may hold only some groups


def _canonical(value):
    """A value as both sides can be compared on: a timestamp as a naive UTC `datetime` whichever way it was spelled (an
    ISO string from `get_chart_data`, a `datetime` from mnemiq's own result), a decimal as a float."""
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return tuple(_canonical(v) for v in value)
    return value


def _line_up(sets: list[tuple[list, list]], labels: list[str], dimensions: tuple[str, ...]) -> _Rows | NotCheckable:
    """Superset's rows by group, each result that holds every grouping column lending the metrics it holds: a mixed
    chart answers one result per axis. Every metric must be held by one."""
    values: dict[tuple, list] = {}
    held: set[int] = set()
    largest = 0
    for columns, rows in sets:
        present = [i for i, label in enumerate(labels) if label in columns]
        if not present or not all(name in columns for name in dimensions):
            continue
        held.update(present)
        largest = max(largest, len(rows))
        seen: set[tuple] = set()
        for row in rows:
            key = tuple(_canonical(row.get(name)) for name in dimensions)
            if key in seen:
                return NotCheckable("Superset's rows for this chart repeat a group, so they cannot be lined up with "
                                    "mnemiq's")
            seen.add(key)
            slot = values.setdefault(key, [None] * len(labels))
            for i in present:
                slot[i] = _canonical(row.get(labels[i]))
    if held != set(range(len(labels))):
        return NotCheckable("Superset's rows for this chart do not carry the columns its settings name, so they cannot "
                            "be lined up with mnemiq's")
    return _Rows(values, largest)


def _chart_rows(client, chart_id: int, expectation: Expectation) -> _Rows | ChartCheck:
    try:
        answer = client.call("get_chart_data", {"identifier": chart_id})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, f"mnemiq could not read the chart's data: {failure}")
    if answer.get("_response_truncated"):
        # `truncate_query_result` (`superset/mcp_service/utils/response_size_utils.py`) keeps the rows that fit.
        return _outcome(chart_id, ChartStatus.CUT, "Superset cut the chart's rows to fit its response size, so they "
                                                   "are not all there to compare")
    if _failed(answer):
        if answer.get("error_type") == "NoData":  # rc2 answers a query with no rows this way
            return _Rows({})
        _log_error("get_chart_data", answer)
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, "mnemiq could not read the chart's data: Superset answered "
                                                           "with an error, which is in the log")
    results = answer.get("query_results")
    if isinstance(results, list) and results:
        sets = [(list(r.get("columns") or []), list(r.get("data") or [])) for r in results if isinstance(r, dict)]
    else:
        sets = [([c.get("name") for c in answer.get("columns") or [] if isinstance(c, dict)],
                 list(answer.get("data") or []))]
    lined = _line_up(sets, [label for label, _ in expectation.metrics], expectation.dimensions)
    if isinstance(lined, NotCheckable):
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, lined.reason)
    return lined


# ---------------------------------------------------------------- mnemiq's own numbers


def _literal(value) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return exp.Literal.number(value).sql(dialect="postgres")
    return exp.Literal.string(str(value)).sql(dialect="postgres")


def _filter_sql(item: Filter) -> str:
    column = _quoted(item.column)
    if item.op in _NO_VALUE:
        return f"{column} {_NO_VALUE[item.op]}"
    if item.op in _LISTS:
        listed = f"{column} IN ({', '.join(_literal(v) for v in item.value)})"
        return f"NOT {listed}" if item.op == "NOT IN" else listed
    operator = _COMPARISONS.get(item.op) or _PATTERNS[item.op]
    return f"{column} {operator} {_literal(item.value)}"


def _statement(expectation: Expectation, dataset: DatasetFacts, *, grained: bool) -> str:
    """mnemiq's statement for the chart's query, through `measure_statement`: each metric aliased by position, the
    x-axis grouped by the grain when `grained`, as Postgres's grain expression writes it."""
    saved = dict(dataset.saved_metric_sql)
    measures = tuple(Measure(name=f"m{i}", label=label,
                             expression=saved[label] if label in dataset.saved_metric_names else expression,
                             kind=MeasureKind.NUMBER, dataset_id=dataset.dataset_id, description="", asked_by="")
                     for i, (label, expression) in enumerate(expectation.metrics))
    columns = [_quoted(name) for name in expectation.dimensions]
    if grained and expectation.time_grain:
        columns[0] = f"DATE_TRUNC('{_GRAINS[expectation.time_grain]}', {columns[0]}) AS {columns[0]}"
    return measure_statement(measures, _quoted(dataset.table), tuple(columns),
                             tuple(_filter_sql(item) for item in expectation.filters))


def _raise_terminal(adapter) -> None:
    """A refusal of the caller met while the statement was decided or run stays the same for every later call, so it
    ends the check, as it ends a question (`SupersetSqlAdapter.terminal_refusal`)."""
    if (refused := getattr(adapter, "terminal_refusal", None)) is not None:
        raise refused


def _own_rows(sql: str, expectation: Expectation, *, chart_id: int, adapter, snapshot: Snapshot, grants: GrantSet,
              timeout_s: float, guard_fanout: bool | None) -> _Rows | NotCheckable:
    try:
        decided = run_decided(sql, adapter=adapter, snapshot=snapshot, grants=grants, timeout_s=timeout_s,
                              guard_fanout=guard_fanout)
    except ExecutionError as failure:
        _raise_terminal(adapter)
        logger.warning("the value check: mnemiq's statement for chart %s did not run: %s %s", chart_id,
                       failure.message, failure.source_detail or "")
        return NotCheckable("mnemiq's own statement for this chart did not run; why is in the log")
    _raise_terminal(adapter)
    if isinstance(decided, Refusal):
        logger.warning("the value check: mnemiq refused its statement for chart %s (%s, %r): %s", chart_id,
                       decided.code.value, decided.subject, decided.source_detail or "")
        return NotCheckable(f"mnemiq refused its own statement for this chart ({decided.code.value}): "
                            f"{decided.message}")
    table, width = decided.table, len(expectation.dimensions)
    if table.num_rows and table.num_columns != width + len(expectation.metrics):
        return NotCheckable("mnemiq's own statement for this chart returned other columns than it asked for")
    values: dict[tuple, list] = {}
    for row in zip(*(table.column(i).to_pylist() for i in range(table.num_columns))) if table.num_rows else ():
        key = tuple(_canonical(v) for v in row[:width])
        if key in values:
            # Two groups the database told apart read as one once made comparable (`2026-01-01` and
            # `2026-01-01 00:00:00` in a text column): which one the chart's row is cannot be told.
            return NotCheckable("mnemiq's own rows for this chart repeat a group, so they cannot be lined up with the "
                                "chart's")
        values[key] = [_canonical(v) for v in row[width:]]
    cap = decided.approved.row_cap
    return _Rows(values, table.num_rows, cap=cap if cap is not None and table.num_rows >= cap else None)


def _dataset_rows(client, expectation: Expectation, dataset: DatasetFacts) -> _Rows | NotCheckable:
    """A virtual dataset's saved metrics through `query_dataset`, which applies the dataset's own row rules."""
    request = {"dataset_id": dataset.dataset_id, "metrics": [label for label, _ in expectation.metrics],
               "columns": list(expectation.dimensions),
               "filters": [{"col": f.column, "op": f.op, "val": f.value} for f in expectation.filters],
               "row_limit": MAX_ROWS}
    try:
        answer = client.call("query_dataset", request)
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"mnemiq could not read the dataset's numbers: {failure}")
    if _failed(answer):
        _log_error("query_dataset", answer)
        return NotCheckable("mnemiq could not read the dataset's numbers: Superset answered with an error, which is "
                            "in the log")
    if answer.get("_response_truncated"):
        return NotCheckable("Superset cut the dataset's numbers to fit its response size, so they are not all there "
                            "to compare")
    rows = list(answer.get("data") or [])
    values = {tuple(_canonical(row.get(name)) for name in expectation.dimensions):
              [_canonical(row.get(label)) for label, _ in expectation.metrics] for row in rows}
    if len(values) != len(rows):
        return NotCheckable("the dataset's rows repeat a group, so they cannot be lined up with the chart's")
    return _Rows(values, len(rows), cap=MAX_ROWS if len(rows) >= MAX_ROWS else None)


# ---------------------------------------------------------------- comparing


def _same(a, b) -> bool:
    """Two values agree: integers exactly, other numbers within `rel_tol=1e-9`, anything else by equality."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9)
    return a == b


def _shown(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _values(rows: dict[tuple, list], order: list[tuple]) -> tuple:
    return tuple((*(_shown(v) for v in key), *(_shown(v) for v in rows[key])) for key in order)


def _outcome(chart_id: int, status: ChartStatus, reason: str, chart: tuple | None = None,
             mine: tuple | None = None) -> ChartCheck:
    return ChartCheck(chart_id, status, reason, chart, mine)


# What mnemiq's side of the comparison is, for the reasons: its own decided SQL over the physical table, or Superset's
# own query of a virtual dataset's saved metrics.
_OWN_SQL = "mnemiq's own SQL"
_QUERY_DATASET = "Superset's own query of the dataset's saved metrics (query_dataset)"


def _compare(chart_id: int, chart: _Rows, mine: _Rows, row_limit: int | None, *, source: str) -> ChartCheck:
    shown = list(chart.values)
    order = [key for key in shown if key in mine.values] + [key for key in mine.values if key not in chart.values]
    both = (_values(chart.values, shown), _values(mine.values, order))

    def outcome(status: ChartStatus, reason: str) -> ChartCheck:
        values = both if status in (ChartStatus.CHECKED, ChartStatus.MISMATCH, ChartStatus.EMPTY) else (None, None)
        return _outcome(chart_id, status, reason, *values)

    if any(key in mine.values and not all(map(_same, chart.values[key], mine.values[key])) for key in shown):
        return outcome(ChartStatus.MISMATCH, f"the chart shows a value {source} does not compute")
    if mine.cap is not None:
        return outcome(ChartStatus.NOT_CHECKED, f"the chart has more groups than {source} returns in one read "
                                                f"({mine.cap:,}), so not all of them were compared")
    if any(key not in mine.values for key in shown):
        return outcome(ChartStatus.MISMATCH, f"the chart shows a group {source} does not find")
    if len(mine.values) > len(shown):
        if row_limit is not None and chart.largest >= row_limit:
            return outcome(ChartStatus.NOT_CHECKED, f"the chart's row limit ({row_limit:,}) keeps only some of its "
                                                    "groups, so not all of them were compared")
        return outcome(ChartStatus.MISMATCH, f"{source} finds a group the chart does not show")
    if not shown:
        return outcome(ChartStatus.EMPTY, f"the chart shows no rows, and {source} finds none")
    return outcome(ChartStatus.CHECKED, f"the chart's numbers match {source}")


def _all_found(chart: _Rows, mine: _Rows) -> bool:
    """Positive evidence that the chart's rows are `mine`: there are some, and each is there with equal values."""
    return bool(chart.values) and all(key in mine.values and all(map(_same, values, mine.values[key]))
                                      for key, values in chart.values.items())


def _row_limit(form: dict) -> int | None:
    try:
        limit = int(form.get("row_limit"))
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


# ---------------------------------------------------------------- the format


def _describe(shown: Displayed) -> str:
    fmt = f"in the format {shown.d3format!r}" if shown.d3format else "in Superset's default number format"
    symbol = (shown.currency or {}).get("symbol")
    if not symbol:
        return f"{fmt}, with no currency"
    if not re.fullmatch(r"[A-Z]{3}", str(symbol)):
        return f"{fmt}, with the currency symbol {symbol!r}, which Superset's renderer does not show"
    return f"{fmt}, with the currency {symbol}"


def _format_problem(viz_type: str, form: dict, kinds: dict[str, MeasureKind], saved: dict) -> str | None:
    """Why a metric mnemiq knows the kind of is shown in a format that does not fit it, or None. A format the renderer
    decides from what mnemiq cannot see (`CANNOT_TELL`), and a saved metric whose own record the caller did not pass,
    are not judged."""
    for item in _metric_items(viz_type, form):
        label = _label(item)
        kind = kinds.get(label) if label else None
        if kind is None:
            continue
        record: SavedMetric | None = None
        if isinstance(item, str):
            record = saved.get(item)
            if record is None:
                continue
        shown = displayed(viz_type, form, item, record)
        if shown is CANNOT_TELL or fits(kind, shown):
            continue
        return f"the chart shows {label!r} {_describe(shown)}, which does not fit a {kind.value} measure"
    return None


# ---------------------------------------------------------------- the check


def _numbers(client, adapter, *, chart_id: int, form: dict, dataset_id: int, snapshot: Snapshot, grants: GrantSet,
             guard_fanout: bool | None, timeout_s: float) -> ChartCheck:
    """The chart's numbers against mnemiq's own: any status but `FORMAT_WRONG`."""
    def not_checked(reason: str) -> ChartCheck:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, reason)

    dataset_id = _dataset_of(form, dataset_id)
    if isinstance(dataset_id, NotCheckable):
        return not_checked(dataset_id.reason)
    dataset = read_dataset_facts(client, dataset_id)
    if isinstance(dataset, NotCheckable):
        return not_checked(dataset.reason)
    expectation = expectation_from_form_data(form, dataset=dataset)
    if isinstance(expectation, NotCheckable):
        return not_checked(expectation.reason)
    sql = ""
    if dataset.is_virtual:
        written = [label for label, _ in expectation.metrics if label not in dataset.saved_metric_names]
        if written:
            return not_checked(f"the measure {written[0]!r} is written on the chart over a virtual dataset: mnemiq "
                               "never checks a virtual dataset with its own SQL, which a row rule attached to that "
                               "dataset alone would not narrow, and Superset's query_dataset takes saved metrics only")
        if expectation.time_grain:
            return not_checked("the chart groups a virtual dataset by a time grain, which Superset's query_dataset "
                               "cannot ask for")
    else:
        if getattr(adapter, "dialect", None) != "postgres":
            return not_checked("mnemiq's own statement for a chart is written for Postgres, Superset mode's one "
                               "dialect, and this adapter speaks another")
        if dataset.database_id != getattr(adapter, "database_id", dataset.database_id) \
                or (dataset.schema or None) != getattr(adapter, "schema", dataset.schema):
            return not_checked("the chart's dataset is on another database or schema than the one mnemiq reads")
        if not dataset.table:
            return not_checked("the chart's dataset names no table")
        saved = dict(dataset.saved_metric_sql)
        if empty := [label for label, _ in expectation.metrics if label in saved and not saved[label].strip()]:
            return not_checked(f"Superset gave no expression for the saved metric {empty[0]!r}")
        sql = _statement(expectation, dataset, grained=True)
        # Checked on the statement itself, so a saved metric's own expression is covered too: `execute_sql` renders a
        # statement as a template, and the adapter would refuse it later in words about the statement, not the chart.
        if (marker := next((m for m in JINJA_DELIMITERS if m in sql), None)) is not None:
            return not_checked(f"the chart's settings hold {marker}, which Superset reads as a template marker, so "
                               "mnemiq cannot compute the same and sends nothing")
    chart = _chart_rows(client, chart_id, expectation)
    if isinstance(chart, ChartCheck):
        return chart
    if dataset.is_virtual:
        mine, source = _dataset_rows(client, expectation, dataset), _QUERY_DATASET
    else:
        mine, source = _own_rows(sql, expectation, chart_id=chart_id, adapter=adapter, snapshot=snapshot, grants=grants,
                                 timeout_s=timeout_s, guard_fanout=guard_fanout), _OWN_SQL
    if isinstance(mine, NotCheckable):
        return not_checked(mine.reason)
    found = _compare(chart_id, chart, mine, _row_limit(form), source=source)
    if found.status is ChartStatus.MISMATCH and expectation.time_grain and source is _OWN_SQL:
        plain = _own_rows(_statement(expectation, dataset, grained=False), expectation, chart_id=chart_id,
                          adapter=adapter, snapshot=snapshot, grants=grants, timeout_s=timeout_s,
                          guard_fanout=guard_fanout)
        # Only on positive evidence: every one of the chart's rows found among mnemiq's ungrouped ones, with equal
        # values. An ungrouped read at its row cap, or an empty chart, proves nothing, and the grouped finding stands.
        if not isinstance(plain, NotCheckable) and _all_found(chart, plain):
            return not_checked("Superset's data tool returned this chart's rows without its time grain, as it does for "
                               "a chart with no saved query context, so the numbers the chart shows by its time grain "
                               "were not compared")
    return found


def check_chart(client, adapter, *, chart_id: int, dataset_id: int, kinds: dict[str, MeasureKind],
                saved: dict[str, SavedMetric], snapshot: Snapshot, grants: GrantSet, guard_fanout: bool | None,
                timeout_s: float = DEFAULT_TIMEOUT_S) -> ChartCheck:
    """Check chart `chart_id` as `client`'s caller, the builder: `client` is `adapter.client`, and `adapter` runs
    mnemiq's own statement under `grants` and `snapshot`. `dataset_id` is the dataset mnemiq built the chart on, which
    a chart saved since in Explore overrides with its own. `kinds` gives the kind of each metric mnemiq knows, by the
    label Superset names its values by; `saved` the dataset's saved metrics mnemiq read, by name.

    A wrong format is reported whatever the numbers, unless they are a mismatch or the chart is empty. A refusal of the
    caller (`SupersetRefusal`) is raised: it stays the same for every later call the request makes."""
    read = _read_chart(client, chart_id)
    if isinstance(read, NotCheckable):
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, read.reason)
    viz_type, form = read
    problem = _format_problem(viz_type, form, kinds, saved)
    found = _numbers(client, adapter, chart_id=chart_id, form=form, dataset_id=dataset_id, snapshot=snapshot,
                     grants=grants, guard_fanout=guard_fanout, timeout_s=timeout_s)
    if problem is None or found.status in (ChartStatus.MISMATCH, ChartStatus.EMPTY):
        return found
    reason = problem if found.status is ChartStatus.CHECKED else f"{problem}. Its numbers were not compared: " \
                                                                  f"{found.reason}"
    return _outcome(chart_id, ChartStatus.FORMAT_WRONG, reason, found.chart_values, found.mnemiq_values)

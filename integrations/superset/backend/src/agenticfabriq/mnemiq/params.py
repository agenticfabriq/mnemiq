"""The pure parts of the extension's write tools: what a write computes, with nothing written.

- `query_context`: the query context the extension saves on a chart the builder makes. rc2's chart data paths run a
  chart's saved query context when it has one; without one, MCP's `get_chart_data` rebuilds the query from the
  chart's params and drops the time grain on the way (`superset/mcp_service/chart/chart_helpers.py`,
  `build_query_dicts_from_form_data`).
- `merge_params` and `patch`: an edit that sets and unsets named params keys and keeps every other one.
- `check_positions` and `layout_metadata`: a layout save that changes membership only as asked, keeps every tab rc2
  knows under its id, and keeps the native filters.
- `metric_stub_list`: the metric list that adds one saved metric and keeps every other one.
- `parse_marker` and `format_marker`: the build marker. mnemiq carries the same two (`mnemiq/superset/marker.py`),
  and one table of cases runs through both.

Standard library only, and no Superset or Flask import, so it runs in Superset's process (Python 3.11) and in
mnemiq's unit tests alike.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["BUILT_FAMILIES", "MARKER_PATTERN", "Marker", "MetricExists", "PatchRefused", "Patched", "check_positions",
           "format_marker", "layout_metadata", "merge_params", "metric_stub_list", "parse_marker", "patch",
           "query_context"]


# ---------------------------------------------------------------- the query context
#
# Each family's query is built as its rc2 frontend plugin's `buildQuery` builds it, then `buildQueryContext`
# (`superset-frontend/packages/superset-ui-core/src/query/buildQueryContext.ts`) wraps it, and nothing more:
# - no `post_processing`, no totals query, no `grouping_sets`, no `time_offsets` (time comparison). What comes back
#   is the rows the chart's query selects, before the frontend reshapes them;
# - no `annotation_layers`: they add annotation data, not rows, and rc2 checks a guest's layers against the chart's
#   params, not its query context (`superset/security/manager.py`, `_annotation_layers_modified`);
# - none of what a dashboard or a browser adds at run time: `extra_form_data`, `extra_filters`, `url_params`, a
#   table's server pagination.
# The shared core below mirrors `buildQueryObject` (`.../src/query/buildQueryObject.ts`) and the helpers it calls.

_BASE_AXIS = "BASE_AXIS"
_UNARY_OPERATORS = frozenset({"IS NOT NULL", "IS NULL"})  # `.../src/query/types/Operator.ts`, `UNARY_OPERATORS`

# `extractQueryFields` (`.../src/query/extractQueryFields.ts`): form-data keys read as a query's metrics, columns,
# group-by or order.
_QUERY_FIELD_ALIASES = {"metric": "metrics", "metric_2": "metrics", "secondary_metric": "metrics",
                        "left_metric": "metrics", "right_metric": "metrics", "x": "metrics", "y": "metrics",
                        "size": "metrics", "all_columns": "columns", "series": "groupby", "order_by_cols": "orderby"}
# The form-data keys `buildQueryObject` takes for itself before the rest reach `extractQueryFields`.
_OWN_KEYS = frozenset({"annotation_layers", "extra_form_data", "time_range", "since", "until", "row_limit",
                       "row_offset", "order_desc", "limit", "timeseries_limit_metric", "granularity", "url_params",
                       "custom_params", "series_columns", "series_limit", "series_limit_metric",
                       "group_others_when_limit_reached", "query_mode"})


def _ensure_list(value: Any) -> list:
    """`ensureIsArray` (`.../src/utils/ensureIsArray.ts`)."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _js_number(value: Any) -> int | float | None:
    """JavaScript's `Number(value)` for the values a form carries, None where it gives NaN or an infinity (which
    `JSON.stringify` sends as null)."""
    if value is None:
        return None
    if isinstance(value, bool):
        number: float = int(value)
    elif _is_number(value):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return int(number) if float(number).is_integer() else number


def _is_empty(value: Any) -> bool:
    """lodash's `isEmpty`: true for null, an empty string, list or object, and any number or boolean."""
    if isinstance(value, (str, list, dict)):
        return len(value) == 0
    return True


def _is_adhoc_column(column: Any) -> bool:
    """`isAdhocColumn` (`.../src/query/types/Column.ts`)."""
    return (isinstance(column, Mapping) and "sqlExpression" in column and "label" in column
            and ("expressionType" not in column or column["expressionType"] == "SQL"))


def _is_query_form_column(column: Any) -> bool:
    """`isQueryFormColumn`: a physical column (any string) or an adhoc one."""
    return isinstance(column, str) or _is_adhoc_column(column)


def _is_query_form_metric(metric: Any) -> bool:
    """`isQueryFormMetric` (`.../src/query/types/Metric.ts`): a saved metric's name or a SIMPLE or SQL adhoc one."""
    return isinstance(metric, str) or (isinstance(metric, Mapping)
                                       and metric.get("expressionType") in ("SIMPLE", "SQL"))


def _column_label(column: Any) -> Any:
    """`getColumnLabel` (`.../src/query/getColumnLabel.ts`)."""
    if isinstance(column, str):
        return column
    if isinstance(column, Mapping):
        return column.get("label") or column.get("sqlExpression")
    return None


def _metric_label(metric: Any) -> Any:
    """`getMetricLabel` (`.../src/query/getMetricLabel.ts`)."""
    if isinstance(metric, str):
        return metric
    if not isinstance(metric, Mapping):
        return None
    if metric.get("label"):
        return metric["label"]
    if metric.get("expressionType") == "SIMPLE":
        column = metric.get("column") if isinstance(metric.get("column"), Mapping) else {}
        return f"{metric.get('aggregate')}({column.get('columnName') or column.get('column_name')})"
    return metric.get("sqlExpression")


def _dedupe(items: list, label) -> list:
    """`removeDuplicates` (`.../src/utils/removeDuplicates.ts`) by a label: the first of each label stays. JavaScript's
    `Set` compares a string, number or boolean by value and anything else by identity, so an unusual label never
    matches another."""
    seen: set = set()
    out = []
    for item in items:
        found = label(item)
        key = (type(found).__name__, found) if found is None or isinstance(found, (str, int, float)) else id(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _parse_order(item: Any) -> Any:
    if not isinstance(item, str):
        return item
    try:
        return json.loads(item)
    except ValueError:
        # `extractQueryFields` throws "Found invalid orderby options", so the chart has no query to save.
        raise ValueError("an order in the chart's params is not one the chart can read") from None


def _query_fields(fd: Mapping) -> dict:
    """`extractQueryFields`: the metrics, columns and order the form data names."""
    mode = fd.get("query_mode")
    columns: list = []
    metrics: list = []
    orderby: list = []
    for key, value in fd.items():
        if key in _OWN_KEYS or value is None:
            continue
        field = _QUERY_FIELD_ALIASES.get(key, key)
        if mode == "aggregate" and field == "columns":
            continue
        if mode == "raw" and field in ("groupby", "metrics"):
            continue
        if field == "groupby":
            field = "columns"
        if field == "metrics":
            metrics.extend(_ensure_list(value))
        elif field == "columns":
            columns.extend(_ensure_list(value))
        elif field == "orderby":
            orderby.extend(_ensure_list(value))
    out: dict = {"columns": _dedupe([c for c in columns if c != ""], _column_label),
                 "metrics": None if mode == "raw" else _dedupe(metrics, _metric_label)}
    if orderby:
        out["orderby"] = [_parse_order(item) for item in orderby]
    return out


def _sanitize_clause(clause: str) -> str:
    """`sanitizeClause` (`.../src/query/processFilters.ts`): parenthesised, a trailing `--` comment closed."""
    return f"({clause}\n)" if "--" in clause else f"({clause})"


def _simple_filter(item: Mapping) -> dict:
    """`convertFilter` (`.../src/query/convertFilter.ts`): `{col, op, val}`, with no `val` for IS NULL or IS NOT
    NULL. A key the filter lacks is left out, as `JSON.stringify` leaves out an undefined one."""
    out: dict = {}
    if "subject" in item:
        out["col"] = item["subject"]
    if "operator" in item:
        out["op"] = item["operator"]
    if item.get("operator") not in _UNARY_OPERATORS and "comparator" in item:
        out["val"] = item["comparator"]
    return out


def _filters_and_extras(fd: Mapping) -> tuple[list, dict]:
    """`processFilters` (`.../src/query/processFilters.ts`) with `extractExtras` (`.../src/query/extractExtras.ts`):
    a SIMPLE WHERE filter becomes a query filter and a SIMPLE HAVING one is dropped, as rc2's frontend drops it; a
    SQL filter joins `extras.where` or `extras.having`; the grain is `extras.time_grain_sqla`."""
    filters: list = []
    where: list[str] = []
    having: list[str] = []
    legacy_where = fd.get("where")
    if legacy_where:
        if not isinstance(legacy_where, str):
            raise ValueError("the chart's `where` is not SQL text")
        where.append(legacy_where)
    for item in _ensure_list(fd.get("adhoc_filters")):
        if not isinstance(item, Mapping):
            raise ValueError("an adhoc filter in the chart's params is not an object")
        if item.get("expressionType") == "SIMPLE":
            if item.get("clause") == "WHERE":
                filters.append(_simple_filter(item))
            continue
        sql = item.get("sqlExpression")
        if not isinstance(sql, str):
            raise ValueError("a SQL adhoc filter in the chart's params has no SQL")
        (where if item.get("clause") == "WHERE" else having).append(sql)
    extras: dict = {"having": " AND ".join(_sanitize_clause(c) for c in having),
                    "where": " AND ".join(_sanitize_clause(c) for c in where)}
    if fd.get("time_grain_sqla"):
        extras["time_grain_sqla"] = fd["time_grain_sqla"]
    return filters, extras


def _base_query(fd: Mapping) -> dict:
    """`buildQueryObject`: the query every plugin starts from, with the keys it sends as undefined left out."""
    fields = _query_fields(fd)
    filters, extras = _filters_and_extras(fd)
    query: dict = {"filters": filters, "extras": extras, "columns": fields["columns"]}
    if fields["metrics"] is not None:
        query["metrics"] = fields["metrics"]
    if "orderby" in fields:
        query["orderby"] = fields["orderby"]
    if fd.get("time_range"):
        query["time_range"] = fd["time_range"]
    granularity = fd.get("granularity") or fd.get("granularity_sqla")
    if granularity:
        query["granularity"] = granularity
    for key in ("row_limit", "row_offset"):
        number = _js_number(fd.get(key))
        if number is not None:
            query[key] = number
    if fd.get("series_columns") is not None:
        query["series_columns"] = fd["series_columns"]
    series_limit = fd.get("series_limit")
    if series_limit is None:
        series_limit = _js_number(fd.get("limit")) if fd.get("limit") is not None else 0
    query["series_limit"] = 0 if series_limit is None else series_limit
    limit_metric = fd.get("series_limit_metric")
    if not _is_query_form_metric(limit_metric):
        limit_metric = fd.get("timeseries_limit_metric")
    if limit_metric is not None:
        query["series_limit_metric"] = limit_metric
    others = fd.get("group_others_when_limit_reached")
    query["group_others_when_limit_reached"] = False if others is None else others
    query["order_desc"] = fd["order_desc"] if "order_desc" in fd else True
    return query


def _set_or_drop(query: dict, key: str, value: Any) -> None:
    if value is None:
        query.pop(key, None)
    else:
        query[key] = value


def _x_axis_column(fd: Mapping) -> Any:
    """`getXAxisColumn` (`.../src/query/getXAxis.ts`)."""
    if not (fd.get("granularity_sqla") or fd.get("x_axis")):
        return None
    return fd["x_axis"] if _is_query_form_column(fd.get("x_axis")) else "__timestamp"


def _normalize_time_column(fd: Mapping, query: dict) -> dict:
    """`normalizeTimeColumn` (`.../src/query/normalizeTimeColumn.ts`), which `buildQueryContext` applies to every
    query when the form has an x-axis: the x-axis column becomes a BASE_AXIS column carrying the grain, so rc2
    truncates it to that grain (`superset/connectors/sqla/models.py`, `SqlaTable.adhoc_column_to_sqla`)."""
    x_axis = fd.get("x_axis")
    columns = query.get("columns")
    if not _is_query_form_column(x_axis) or not x_axis or not isinstance(columns, list):
        return query
    for index, column in enumerate(columns):
        if (isinstance(column, str) and isinstance(x_axis, str) and column == x_axis) or (
                _is_adhoc_column(column) and _is_adhoc_column(x_axis)
                and column["sqlExpression"] == x_axis["sqlExpression"]):
            break
    else:
        return query
    grain = query.get("extras", {}).get("time_grain_sqla")
    if _is_adhoc_column(columns[index]):
        axis = {"timeGrain": grain, "columnType": _BASE_AXIS, **columns[index]}
    else:
        axis = {"timeGrain": grain, "columnType": _BASE_AXIS, "sqlExpression": x_axis, "label": x_axis,
                "expressionType": "SQL", "isColumnReference": True}
    if axis.get("timeGrain") is None:
        del axis["timeGrain"]
    out = {key: value for key, value in query.items() if key != "is_timeseries"}
    out["columns"] = [*columns[:index], axis, *columns[index + 1:]]
    return out


def _normalize_order_by(query: Mapping) -> dict:
    """`normalizeOrderBy` (`.../src/query/normalizeOrderBy.ts`): a valid order is kept; otherwise the series limit
    metric, else the first metric, in the direction `order_desc` gives."""
    orderby = query.get("orderby")
    if isinstance(orderby, list) and orderby:
        first = orderby[0]
        if isinstance(first, list) and len(first) == 2 and not _is_empty(first[0]) and isinstance(first[1], bool):
            return dict(query)
    out = {k: v for k, v in query.items() if k not in ("series_limit_metric", "legacy_order_by", "order_desc",
                                                        "orderby")}
    ascending = not query.get("order_desc")
    limit_metric = query.get("series_limit_metric")
    metrics = query.get("metrics")
    if limit_metric is not None and not _is_empty(limit_metric):
        out["orderby"] = [[limit_metric, ascending]]
    elif isinstance(metrics, list) and metrics:
        out["orderby"] = [[metrics[0], ascending]]
    return out


def _extra_metrics(fd: Mapping) -> list:
    """`extractExtraMetrics` (`superset-frontend/packages/superset-ui-chart-controls/src/operators/utils/
    extractExtraMetrics.ts`): the sort metric of an x-axis sorted by a metric the chart does not plot."""
    groupby = fd.get("groupby") or []
    limit_metric = (_ensure_list(fd.get("timeseries_limit_metric")) or [None])[0]
    sort = fd.get("x_axis_sort")
    if (isinstance(groupby, (list, str)) and len(groupby)) or not limit_metric:
        return []
    if _metric_label(limit_metric) != sort:
        return []
    if any(_metric_label(metric) == sort for metric in _ensure_list(fd.get("metrics"))):
        return []
    return [limit_metric]


def _timeseries(fd: dict) -> list[tuple[dict, dict]]:
    """Line, bar, area and scatter, and the step, smooth and generic time-series charts a line chart becomes by hand:
    `plugin-chart-echarts/src/Timeseries/buildQuery.ts`, `buildQuery` (each kind registers it: `Timeseries/Area/
    index.ts`, `Timeseries/Regular/{Line,Bar,Scatter,SmoothLine}/index.ts`, `Timeseries/Step/index.ts`,
    `Timeseries/index.ts`). The x-axis, then the
    group-by; left out are the pivot, resample, rolling, time-compare, contribution, rename, sort, flatten and prophet
    operators, and `time_offsets`."""
    base = _base_query(fd)
    groupby = fd.get("groupby")
    x_set = _is_query_form_column(fd.get("x_axis"))
    query = dict(base)
    query["metrics"] = [*(base.get("metrics") or []), *_extra_metrics(fd)]
    query["columns"] = [*(_ensure_list(_x_axis_column(fd)) if x_set else []), *_ensure_list(groupby)]
    _set_or_drop(query, "series_columns", groupby)
    if not x_set:
        query["is_timeseries"] = True
    _set_or_drop(query, "orderby", _normalize_order_by(base).get("orderby"))
    return [(fd, query)]


def _remove_suffix(fd: Mapping, suffix: str) -> dict:
    """`removeFormDataSuffix` (`plugin-chart-echarts/src/utils/formDataSuffix.ts`)."""
    return {key: value for key, value in fd.items() if not key.endswith(suffix)}


def _retain_suffix(fd: Mapping, suffix: str) -> dict:
    """`retainFormDataSuffix` (the same file): the suffixed keys, renamed, win over their unsuffixed twins."""
    out: dict = {}
    for key, value in sorted(fd.items(), key=lambda item: 0 if item[0].endswith(suffix) else 1):
        if key.endswith(suffix):
            out[key[:-len(suffix)]] = value
        elif key not in out:
            out[key] = value
    return out


def _mixed(fd: dict) -> list[tuple[dict, dict]]:
    """Mixed time series: `plugin-chart-echarts/src/MixedTimeseries/buildQuery.ts`, `buildQuery`. Two queries, one
    per side (the `_b` keys are the second's), each the x-axis then its own group-by. Left out: each side's pivot,
    resample, rolling, time-compare, rename and flatten operators, and `time_offsets`."""
    x_set = _is_query_form_column(fd.get("x_axis"))
    out = []
    for side in (_remove_suffix(fd, "_b"), _retain_suffix(fd, "_b")):
        query = _base_query(side)
        query["columns"] = [*(_ensure_list(_x_axis_column(fd)) if x_set else []), *_ensure_list(side.get("groupby"))]
        _set_or_drop(query, "series_columns", side.get("groupby"))
        if not x_set:
            query["is_timeseries"] = True
        _set_or_drop(query, "orderby", _normalize_order_by(query).get("orderby"))
        out.append((side, query))
    return out


def _table_query_mode(fd: Mapping) -> str:
    """`getQueryMode` (`plugin-chart-table/src/buildQuery.ts`)."""
    mode = fd.get("query_mode")
    if mode in ("aggregate", "raw"):
        return mode
    raw = fd.get("all_columns")
    return "raw" if isinstance(raw, (list, str)) and len(raw) > 0 else "aggregate"


def _table(fd: dict) -> list[tuple[dict, dict]] | None:
    """Table, aggregate mode only: `plugin-chart-table/src/buildQuery.ts`, `buildQuery`. The group-by
    with the first column Explore marked temporal made a BASE_AXIS column with the grain and moved to the front, the
    metrics and percent metrics, ordered by the sort-by metric or else the first metric descending. Left out: the
    percent-metric `contribution` and time-compare operators, `time_offsets`, the totals and all-records percent
    queries, and server pagination."""
    if _table_query_mode(fd) != "aggregate":
        return None
    base = _base_query(fd)
    metrics = list(base.get("metrics") or [])
    orderby = base.get("orderby") or []
    sort_metric = (_ensure_list(fd.get("timeseries_limit_metric")) or [None])[0]
    order_desc = fd["order_desc"] if "order_desc" in fd else False
    if sort_metric:
        orderby = [[sort_metric, not order_desc]]
    elif metrics:
        orderby = [[metrics[0], False]]
    if fd.get("percent_metrics"):
        metrics = _dedupe([*metrics, *_ensure_list(fd["percent_metrics"])], _metric_label)
    grain = fd.get("time_grain_sqla")
    lookup = fd.get("temporal_columns_lookup")
    lookup = lookup if isinstance(lookup, Mapping) else {}
    temporal = None
    kept = []
    for column in base["columns"]:
        if temporal is None and isinstance(column, str) and grain and lookup.get(column):
            temporal = {"timeGrain": grain, "columnType": _BASE_AXIS, "sqlExpression": column, "label": column,
                        "expressionType": "SQL"}
            continue
        kept.append(column)
    query = dict(base)
    query["columns"] = [temporal, *kept] if temporal else kept
    query["orderby"] = orderby
    query["metrics"] = metrics
    return [(fd, query)]


def _sorted_by_metric(fd: dict) -> list[tuple[dict, dict]]:
    """Pie (`plugin-chart-echarts/src/Pie/buildQuery.ts`), gauge (`Gauge/buildQuery.ts`) and funnel
    (`Funnel/buildQuery.ts`), each `buildQuery`: the group-by and the metric, ordered by the metric descending when
    `sort_by_metric` is on. Left out: the pie's `contribution` operator, which adds a share column; the values are
    the query's."""
    query = _base_query(fd)
    if fd.get("sort_by_metric"):
        query["orderby"] = [[fd.get("metric"), False]]
    return [(fd, query)]


def _big_number_total(fd: dict) -> list[tuple[dict, dict]]:
    """Big number without a trendline: `plugin-chart-echarts/src/BigNumber/BigNumberTotal/buildQuery.ts`,
    `buildQuery`, which sends the base query as it is."""
    return [(fd, _base_query(fd))]


def _pivot(fd: dict) -> list[tuple[dict, dict]]:
    """Pivot table: `plugin-chart-pivot-table/src/plugin/buildQuery.ts`, `buildQuery`. The rows then the columns
    (the columns first when transposed), one of each, a column Explore marked temporal (or the time column) made a
    BASE_AXIS column with the grain; ordered by the series limit metric or else the first metric. Left out: the
    `grouping_sets` a non-additive metric's totals would add. The leaf rows are the query's; totals and subtotals are
    the frontend's."""
    grain = fd.get("time_grain_sqla")
    first, second = ("groupbyColumns", "groupbyRows") if fd.get("transposePivot") else ("groupbyRows",
                                                                                         "groupbyColumns")
    lookup = fd.get("temporal_columns_lookup")
    lookup = lookup if isinstance(lookup, Mapping) else {}
    seen: set = set()
    columns = []
    for column in [*_ensure_list(fd.get(first)), *_ensure_list(fd.get(second))]:
        if isinstance(column, str):  # a JavaScript `Set` keeps one of each string, and objects by identity
            if column in seen:
                continue
            seen.add(column)
        if isinstance(column, str) and grain and (lookup.get(column) or fd.get("granularity_sqla") == column):
            column = {"timeGrain": grain, "columnType": _BASE_AXIS, "sqlExpression": column, "label": column,
                      "expressionType": "SQL"}
        columns.append(column)
    query = _base_query(fd)
    limit_metric, metrics = query.get("series_limit_metric"), query.get("metrics")
    if limit_metric:
        orderby = [[limit_metric, not query.get("order_desc")]]
    elif isinstance(metrics, list) and metrics and metrics[0]:
        orderby = [[metrics[0], not query.get("order_desc")]]
    else:
        orderby = None
    _set_or_drop(query, "orderby", orderby)
    query["columns"] = columns
    return [(fd, query)]


def _orders_under_row_limit(row_limit: Any) -> bool:
    """`applyOrderBy`'s condition (`plugin-chart-echarts/src/utils/orderby.ts`): a row limit is set and is not 0."""
    if row_limit is None:
        return False
    if isinstance(row_limit, str):
        number = _js_number(row_limit)
        return number is None or number != 0
    if _is_number(row_limit):
        return math.isnan(row_limit) or row_limit != 0
    return True


def _treemap(fd: dict) -> list[tuple[dict, dict]]:
    """Treemap: `plugin-chart-echarts/src/Treemap/buildQuery.ts`, `buildQuery`. The group-by and the metric, under a
    row limit ordered by the metric (when sorting by it) then each group-by column ascending."""
    query = _base_query(fd)
    metric = fd.get("metric")
    orderby: list = [[metric, False]] if fd.get("sort_by_metric") and metric else []
    orderby += [[column, True] for column in _ensure_list(fd.get("groupby")) if isinstance(column, str) and column]
    if _orders_under_row_limit(fd.get("row_limit")) and orderby:
        query["orderby"] = orderby
    return [(fd, query)]


def _waterfall(fd: dict) -> list[tuple[dict, dict]]:
    """Waterfall: `plugin-chart-echarts/src/Waterfall/buildQuery.ts`, `buildQuery`. The x-axis (or the time column)
    then the breakdown, each ordered ascending; `buildQueryContext` then gives the x-axis its grain. The running
    totals are the frontend's."""
    query = _base_query(fd)
    columns = [*_ensure_list(fd.get("x_axis") or fd.get("granularity_sqla")), *_ensure_list(fd.get("groupby"))]
    query["columns"] = columns
    query["orderby"] = [[column, True] for column in columns]
    return [(fd, query)]


def _sankey(fd: dict) -> list[tuple[dict, dict]]:
    """Sankey: `plugin-chart-echarts/src/Sankey/buildQuery.ts`, `buildQuery`. The source and the target and the
    metric, under a row limit ordered by the metric (when sorting by it) then source and target ascending. The plugin
    sends them as `groupby`, which rc2 loads as `columns` (`superset/charts/schemas.py`, `ChartDataQueryObjectSchema`,
    `rename_deprecated_fields`); they are sent here as `columns` directly."""
    query = _base_query(fd)
    source, target, metric = fd.get("source"), fd.get("target"), fd.get("metric")
    orderby: list = [[metric, False]] if fd.get("sort_by_metric") and metric else []
    orderby += [[column, True] for column in (source, target) if column]
    row_limit = fd.get("row_limit")
    query["columns"] = [source, target]
    if row_limit is not None and not (_is_number(row_limit) and row_limit == 0) and orderby:
        query["orderby"] = orderby
    return [(fd, query)]


_BUILDERS = {
    "echarts_timeseries_line": _timeseries,
    "echarts_timeseries_bar": _timeseries,
    "echarts_area": _timeseries,
    "echarts_timeseries_scatter": _timeseries,
    # Not made by the builder, and the types a person reaches by changing a line chart's type in Explore; built with
    # the same `buildQuery` (`VizType.Step`, `VizType.SmoothLine`, `VizType.Timeseries`), so a later patch keeps the
    # chart's grain.
    "echarts_timeseries_step": _timeseries,
    "echarts_timeseries_smooth": _timeseries,
    "echarts_timeseries": _timeseries,
    "mixed_timeseries": _mixed,
    "table": _table,
    "pie": _sorted_by_metric,
    "big_number_total": _big_number_total,
    "gauge_chart": _sorted_by_metric,
    "pivot_table_v2": _pivot,
    "treemap_v2": _treemap,
    "waterfall": _waterfall,
    "funnel": _sorted_by_metric,
    "sankey_v2": _sankey,
}

# The types that get a query context: the families the builder makes, and the line-like types above. A box plot, a
# histogram, handlebars and the interactive pivot get none, nor does anything else, including a big number with a
# trendline.
BUILT_FAMILIES: frozenset[str] = frozenset(_BUILDERS)


def query_context(viz_type: str, params: Mapping, datasource_id: int) -> dict | None:
    """The query context to save on a chart of `viz_type` with these params (its form data) on dataset
    `datasource_id`, in the shape rc2's `ChartDataQueryContextSchema` loads (`superset/charts/schemas.py`); None for a
    family outside `BUILT_FAMILIES` and for a table in raw mode. Raises ValueError for a bad `datasource_id` or for
    params rc2's frontend could not build a query from either."""
    if isinstance(datasource_id, bool) or not isinstance(datasource_id, int) or datasource_id < 1:
        raise ValueError("datasource_id must be a positive integer")
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    build = _BUILDERS.get(viz_type)
    if build is None:
        return None
    built = build(copy.deepcopy(dict(params)))
    if built is None:
        return None
    context = {
        "datasource": {"id": datasource_id, "type": "table"},
        "force": False,
        "queries": [_normalize_time_column(form, query) for form, query in built],
        # `buildQueryContext` sends its form data; the mixed chart's is the first side's, as its `buildQuery` returns.
        "form_data": built[0][0],
        "result_format": "json",
        "result_type": "full",
    }
    # Exactly what is stored: JSON, sharing nothing with `params`.
    return json.loads(json.dumps(context))


def histogram_check_context(form_data: Mapping, datasource_id: int) -> dict:
    """The query context the extension runs to check a histogram (`histogram_v2`) before it is saved: the observations
    and their grouping columns, the form data's `groupby` then its `column`, no metrics and no post-processing, under
    the form data's filters, read as the other families' queries read them. rc2's own compile check builds a histogram's
    query from metrics and groupby and so finds nothing to select, because the histogram names its column under
    `column` alone. Not a chart's saved query context. Raises ValueError for a bad `datasource_id`, for form data
    that is not a mapping, and for form data without a `column`."""
    if isinstance(datasource_id, bool) or not isinstance(datasource_id, int) or datasource_id < 1:
        raise ValueError("datasource_id must be a positive integer")
    if not isinstance(form_data, Mapping):
        raise ValueError("params must be a mapping")
    column = form_data.get("column")
    if not (isinstance(column, str) and column) and not _is_adhoc_column(column):
        raise ValueError("a histogram's params name no column")
    filters, extras = _filters_and_extras(form_data)
    groups = [c for c in _ensure_list(form_data.get("groupby")) if c != ""]
    context = {
        "datasource": {"id": datasource_id, "type": "table"},
        "force": False,
        "queries": [{"columns": _dedupe([*groups, column], _column_label), "metrics": [], "filters": filters,
                     "extras": extras}],
        "result_format": "json",
        "result_type": "full",
    }
    return json.loads(json.dumps(context))


# ---------------------------------------------------------------- the patch

class PatchRefused(ValueError):
    """A patch the tool does not apply. `reason` is one of: `not_a_mapping`, `key` (a key that is not a string),
    `set_and_unset` (one key in both), `datasource` (the chart's dataset binding), `viz_type` (unsetting it, or a
    change to or from funnel or sankey, which the extension alone makes)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# A chart's dataset is its row's binding; a params patch never moves it.
_BINDING_KEYS = frozenset({"datasource", "datasource_id", "datasource_type"})
_EXTENSION_MADE = frozenset({"funnel", "sankey_v2"})


def merge_params(params: Mapping, to_set: Mapping, to_unset: Iterable[str]) -> dict:
    """`params` with the keys of `to_set` set to their values and the keys in `to_unset` removed, every other key
    (annotation layers, conditional formatting, anything mnemiq does not know) kept as it was. Shares nothing with its
    inputs."""
    if not isinstance(params, Mapping) or not isinstance(to_set, Mapping):
        raise PatchRefused("not_a_mapping")
    if isinstance(to_unset, (str, bytes)):
        raise PatchRefused("key")
    unset = list(to_unset)
    keys = [*to_set, *unset]
    if any(not isinstance(key, str) for key in keys):
        raise PatchRefused("key")
    if set(to_set) & set(unset):
        raise PatchRefused("set_and_unset")
    if _BINDING_KEYS & set(keys):
        raise PatchRefused("datasource")
    if "viz_type" in unset:
        raise PatchRefused("viz_type")
    merged = copy.deepcopy(dict(params))
    for key in unset:
        merged.pop(key, None)
    merged.update(copy.deepcopy(dict(to_set)))
    return merged


@dataclass(frozen=True)
class Patched:
    """A patched chart: its params, its viz type, and the query context to save (None: save none)."""

    params: dict
    viz_type: str
    query_context: dict | None


def patch(viz_type: str, params: Mapping, to_set: Mapping, to_unset: Iterable[str], datasource_id: int) -> Patched:
    """The chart `patch_chart` writes: `viz_type` and `params` are the chart's now, `datasource_id` its dataset. The
    query context is regenerated from the merged params for a family in `BUILT_FAMILIES` and is None for the rest.
    A `viz_type` in `to_set` changes the chart's type, except to or from funnel or sankey."""
    merged = merge_params(params, to_set, to_unset)
    new_type = to_set.get("viz_type", viz_type)
    if not isinstance(new_type, str) or not new_type:
        raise PatchRefused("viz_type")
    if new_type != viz_type and (viz_type in _EXTENSION_MADE or new_type in _EXTENSION_MADE):
        raise PatchRefused("viz_type")
    return Patched(merged, new_type, query_context(new_type, merged, datasource_id))


# ---------------------------------------------------------------- the layout

def _is_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _layout_chart_ids(positions: Mapping) -> set[int] | None:
    """The charts a layout makes members, read as rc2's `DashboardDAO.set_dash_metadata` (`superset/daos/
    dashboard.py`) reads them: every node's `meta.chartId`. None for a layout it would fail on or misread: a node
    without a type or whose type is not text, a `meta` that is not an object, a chart node without a positive integer
    id, or a chart id on a node that is not a chart."""
    found: set[int] = set()
    for node in positions.values():
        if not isinstance(node, Mapping):
            continue
        if not isinstance(node.get("type"), str):
            return None
        meta = node.get("meta", {})
        if not isinstance(meta, Mapping):
            return None
        chart_id = meta.get("chartId")
        if node["type"] == "CHART":
            if not _is_id(chart_id):
                return None
            found.add(chart_id)
        elif chart_id is not None:
            return None
    return found


# The node types whose children `Dashboard.tabs` walks into; it reads past every other node without entering it.
_WALKED = frozenset({"ROOT", "GRID", "TABS", "TAB"})


def _reachable_tabs(positions: Any) -> list[tuple[str, Mapping]]:
    """The TAB nodes of a layout that rc2 knows as its tabs, each with the key it is stored under, read as
    `Dashboard.tabs` (`superset/models/dashboard.py`) reads them: breadth first from `ROOT_ID`, entering the children
    of the root, the grid, tab groups and tabs only, each node reached once (a node a second parent lists is passed
    over), a node without a type entered no further. A report on a tab names it by its `id`. A node no walk from
    `ROOT_ID` reaches is dropped by the next save (`DashboardDAO.set_dash_metadata` runs `repair_position`,
    `superset/dashboards/layout.py`), so it is no tab at all."""
    if not isinstance(positions, Mapping) or not isinstance(positions.get("ROOT_ID"), Mapping):
        return []
    found: list[tuple[str, Mapping]] = []
    visited = {"ROOT_ID"}
    queue: deque[tuple[str, Mapping]] = deque([("ROOT_ID", positions["ROOT_ID"])])
    while queue:
        key, node = queue.popleft()
        if "type" not in node:
            continue
        children = node.get("children", [])
        for child_key in children if isinstance(children, list) else []:
            child = positions.get(child_key) if isinstance(child_key, str) else None
            if not isinstance(child, Mapping) or child_key in visited:
                continue
            visited.add(child_key)
            # Compared as rc2 compares it, with `==`: a type that is not text is no walked type, and never a lookup
            # that fails on a list or an object.
            if isinstance(node["type"], str) and node["type"] in _WALKED:
                queue.append((child_key, child))
        if node["type"] == "TAB":
            found.append((key, node))
    return found


def check_positions(positions: Any, *, current_positions: Any, current_chart_ids: Iterable[int],
                    add_chart_ids: Iterable[int] = (), remove_chart_ids: Iterable[int] = ()) -> dict | None:
    """Why `set_layout` refuses `positions`, or None. `current_positions` is the dashboard's layout now and
    `current_chart_ids` its charts. Refused:
    - `malformed`: not a layout `set_dash_metadata` reads cleanly, a chart id that is not a positive integer, or a
      tab stored under a key that is not its id (rc2 walks a layout by its keys and knows a tab by its id);
    - `add_and_remove`: a chart both added and removed;
    - `chart_set`: the layout's charts are not exactly the current ones plus the added minus the removed. Writing
      `positions` replaces the dashboard's membership wholesale, so a chart left out would be dropped from it;
    - `drops_tab`: a tab of the current layout, as rc2 knows its tabs (`_reachable_tabs`), is not one of the new
      layout's under the same id: gone, no longer a tab, detached from the walk from `ROOT_ID`, or under another id.
      rc2 deactivates the reports on a deleted tab only when `position_json` itself is sent
      (`UpdateDashboardCommand.process_tab_diff`), which a `positions` save is not. A tab the current layout holds
      but does not reach is not required: rc2 does not know it, and the save drops it anyway."""
    add, remove = list(add_chart_ids), list(remove_chart_ids)
    current = list(current_chart_ids)
    if not isinstance(positions, Mapping) or not all(_is_id(i) for i in [*add, *remove, *current]):
        return {"refused": "malformed"}
    both = set(add) & set(remove)
    if both:
        return {"refused": "add_and_remove", "chart_ids": sorted(both)}
    found = _layout_chart_ids(positions)
    tabs = _reachable_tabs(positions)
    if found is None or any(node.get("id") != key for key, node in tabs):
        return {"refused": "malformed"}
    expected = (set(current) | set(add)) - set(remove)
    if found != expected:
        return {"refused": "chart_set", "missing": sorted(expected - found), "unexpected": sorted(found - expected)}
    known = {node["id"] for _, node in _reachable_tabs(current_positions) if isinstance(node.get("id"), str)}
    dropped = sorted(known - {key for key, _ in tabs})
    if dropped:
        return {"refused": "drops_tab", "tabs": dropped}
    return None


# Keys `set_dash_metadata` resets to a default or drops when the metadata sent lacks them; sent back as they are
# stored. Every other stored key it keeps by itself.
_RESET_WHEN_ABSENT = ("color_namespace", "shared_label_colors", "map_label_colors", "color_scheme_domain")


def layout_metadata(positions: Mapping, current_json_metadata: Mapping | None) -> dict:
    """The `json_metadata` a layout save sends through `UpdateDashboardCommand`: the `positions` (which set the
    layout and the membership) and the dashboard's current `native_filter_configuration`. Without the filters,
    `UpdateDashboardCommand.process_native_filter_diff` reads every filter as deleted and deactivates each report on
    one. The keys `set_dash_metadata` would otherwise reset go back as stored.

    So do the legacy `default_filters` and `filter_scopes` when the dashboard has them: `set_dash_metadata` rewrites
    `default_filters` from the value sent alone and drops `filter_scopes` when none are sent, which would undo them on
    a dashboard made by hand. It reads `default_filters` as JSON text, so a value stored as an object goes back as
    its JSON."""
    current = current_json_metadata if isinstance(current_json_metadata, Mapping) else {}
    metadata = {"positions": copy.deepcopy(dict(positions)),
                "native_filter_configuration": copy.deepcopy(current.get("native_filter_configuration", []))}
    for key in _RESET_WHEN_ABSENT:
        if key in current:
            metadata[key] = copy.deepcopy(current[key])
    if "default_filters" in current:
        stored = current["default_filters"]
        metadata["default_filters"] = json.dumps(stored) if isinstance(stored, Mapping) else stored
    if "filter_scopes" in current:
        metadata["filter_scopes"] = copy.deepcopy(current["filter_scopes"])
    return metadata


# ---------------------------------------------------------------- the saved metric add

class MetricExists(ValueError):
    """The dataset already has a metric of that name."""


def metric_stub_list(existing: Iterable[Mapping], new: Mapping) -> list[dict]:
    """The `metrics` for `UpdateDatasetCommand` that add `new` and keep every metric the dataset has. rc2 replaces the
    whole list (`superset/daos/dataset.py`, `DatasetDAO.update_metrics`): a metric with an id is updated with only the
    keys sent, one without is created, and one not sent is deleted. So each existing metric goes back as its id and
    name alone, unchanged, and `new` has no id."""
    stubs = []
    for metric in existing:
        if not isinstance(metric, Mapping) or not _is_id(metric.get("id")) or not isinstance(
                metric.get("metric_name"), str):
            raise ValueError("an existing metric has no id or no name")
        stubs.append({"id": metric["id"], "metric_name": metric["metric_name"]})
    if not isinstance(new, Mapping) or "id" in new:
        raise ValueError("the new metric must be an object without an id")
    name, expression = new.get("metric_name"), new.get("expression")
    if not (isinstance(name, str) and name.strip() and isinstance(expression, str) and expression.strip()):
        raise ValueError("the new metric needs a name and an expression")
    if any(stub["metric_name"] == name for stub in stubs):
        raise MetricExists(name)
    return [*stubs, copy.deepcopy(dict(new))]


# ---------------------------------------------------------------- the build marker
#
# Kept identical to mnemiq's `mnemiq/superset/marker.py`; `tests/test_superset_marker_grammar.py` runs one table of
# cases through both.

MARKER_PATTERN = re.compile(
    r"mnemiq:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}):([a-z0-9_]{1,40}):([1-9][0-9]*|new)")


@dataclass(frozen=True)
class Marker:
    """A build's marker: the build, the step, and the dashboard (None: `new`, the dashboard shell's own)."""

    build_id: str
    step_key: str
    dashboard_id: int | None


def parse_marker(description: str | None) -> Marker | None:
    """The marker on the last line of `description`, or None: an empty or missing description, a last line that is
    not exactly a marker, and a marker on any other line are all no marker."""
    lines = (description or "").rstrip().splitlines()
    if not lines:
        return None
    found = MARKER_PATTERN.fullmatch(lines[-1])
    if found is None:
        return None
    build_id, step_key, dashboard = found.groups()
    if dashboard == "new":
        return Marker(build_id, step_key, None)
    try:
        return Marker(build_id, step_key, int(dashboard))
    except ValueError:  # more digits than Python converts to an integer: no dashboard has that id
        return None


def format_marker(build_id: str, step_key: str, dashboard_id: int | None) -> str:
    """The marker line for these fields (`dashboard_id` None writes `new`). Refuses fields that would not parse back
    to themselves."""
    if dashboard_id is None:
        tail = "new"
    elif _is_id(dashboard_id):
        tail = str(dashboard_id)
    else:
        raise ValueError("a marker's dashboard id is a positive integer, or None for new")
    if not isinstance(build_id, str) or not isinstance(step_key, str):
        raise ValueError("a marker's build id and step key are strings")
    line = f"mnemiq:{build_id}:{step_key}:{tail}"
    if parse_marker(line) != Marker(build_id, step_key, dashboard_id):
        raise ValueError("these fields do not make a marker")
    return line

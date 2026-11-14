"""Chart specs for the fifteen v1 chart types, and the requests that build them.

A `ChartSpec` says what a chart shows, in mnemiq's words. `generate_chart_request` turns it into the request rc2's MCP
`generate_chart` takes (`GenerateChartRequest` in `superset/mcp_service/chart/schemas.py`, whose per-type `*ChartConfig`
models reject a field they do not know, so no request here carries one they lack). Funnel and sankey are not among MCP's
chart types; `extension_chart_request` builds the request for the extension's `create_chart`, with the `form_data`
Superset's own funnel and sankey plugins read.

Nothing here sends anything. Every number a chart shows is formatted from the kind of the measure it plots
(`display_format`), and a chart that would show a kind wrongly is refused by `plan_refusal` before it is built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from mnemiq.superset.measures import Measure, MeasureKind, display_format
from mnemiq.superset.saved_metrics import SavedMetric

__all__ = ["ChartSpec", "Family", "Filter", "MCP_FAMILIES", "MeasureRef", "SavedRef", "WrittenRef",
           "extension_chart_request", "generate_chart_request", "owed_format_params", "plan_refusal"]


class Family(StrEnum):
    """A chart family and its Superset `viz_type`. MCP's one `xy` chart type is split into its four kinds."""

    LINE = "echarts_timeseries_line"
    BAR = "echarts_timeseries_bar"
    AREA = "echarts_area"
    SCATTER = "echarts_timeseries_scatter"
    TABLE = "table"
    PIE = "pie"
    BIG_NUMBER = "big_number_total"
    GAUGE = "gauge_chart"
    HISTOGRAM = "histogram_v2"
    BOX_PLOT = "box_plot"
    PIVOT = "pivot_table_v2"
    INTERACTIVE_PIVOT = "ag-grid-pivot-table"
    TREEMAP = "treemap_v2"
    WATERFALL = "waterfall"
    MIXED = "mixed_timeseries"
    HANDLEBARS = "handlebars"
    FUNNEL = "funnel"
    SANKEY = "sankey_v2"


EXTENSION_FAMILIES = frozenset({Family.FUNNEL, Family.SANKEY})
MCP_FAMILIES = frozenset(set(Family) - EXTENSION_FAMILIES)

# The one template a handlebars chart is built with: each row as its columns' names and values, which the chart's
# author then restyles in Superset. rc2's `HandlebarsChartConfig` requires a template.
HANDLEBARS_TEMPLATE = "<table>{{#each data}}<tr>{{#each this}}<td>{{@key}}</td><td>{{this}}</td>{{/each}}</tr>{{/each}}</table>"

# The most rows rc2's `*ChartConfig.row_limit` accepts; a chart without a `row_limit` field (the big number) has none.
_ROW_LIMIT_MAX = {Family.GAUGE: 10, Family.PIE: 10000, Family.TREEMAP: 10000, Family.HISTOGRAM: 100000}
_NO_ROW_LIMIT = {Family.BIG_NUMBER}

_XY_KINDS = {Family.LINE: "line", Family.BAR: "bar", Family.AREA: "area", Family.SCATTER: "scatter"}


@dataclass(frozen=True)
class SavedRef:
    """A saved metric the chart reuses, with the kind the planner gave it."""

    name: str
    kind: MeasureKind


@dataclass(frozen=True)
class WrittenRef:
    """A measure the chart writes into its own request."""

    measure: Measure


MeasureRef = SavedRef | WrittenRef


@dataclass(frozen=True)
class Filter:
    column: str
    op: str
    value: str | float | list | None = None


@dataclass(frozen=True)
class ChartSpec:
    key: str
    family: Family
    title: str
    dataset_id: int
    measures: tuple[MeasureRef, ...]
    dimensions: tuple[str, ...]
    time_column: str | None
    time_grain: str | None
    filters: tuple[Filter, ...]
    row_limit: int | None
    source: str | None
    target: str | None


# ---------------------------------------------------------------- the plan check

_ONE_MEASURE = {Family.PIE, Family.BIG_NUMBER, Family.GAUGE, Family.TREEMAP, Family.WATERFALL, Family.FUNNEL,
                Family.SANKEY}
_NO_MEASURE_NEEDED = {Family.HISTOGRAM}
_NEEDS_DIMENSION = {Family.PIE, Family.TREEMAP, Family.FUNNEL, Family.HISTOGRAM, Family.PIVOT,
                    Family.INTERACTIVE_PIVOT, Family.TABLE}
_NO_CURRENCY_OR_PERCENT = {Family.SANKEY, Family.BOX_PLOT, Family.HISTOGRAM}
_ONE_FORMAT_PER_AXIS_EXEMPT = {Family.TABLE, Family.HANDLEBARS}


def _kind(ref: MeasureRef) -> MeasureKind:
    return ref.kind if isinstance(ref, SavedRef) else ref.measure.kind


_ISO = re.compile(r"^[A-Z]{3}$")


def _currency(ref: MeasureRef, saved: dict[str, SavedMetric], deployment: str) -> str | None:
    """The ISO code a currency measure shows its currency in: its own, or a reused saved metric's own when that is an
    ISO code, else the deployment currency (`superset_currency`, `MNEMIQ_SUPERSET_CURRENCY`): a saved metric whose
    currency is none, `UNKNOWN` or a symbol is shown in the deployment's. A measure of another kind has none."""
    if _kind(ref) is not MeasureKind.CURRENCY:
        return None
    if isinstance(ref, WrittenRef):
        return ref.measure.currency or deployment
    held = saved.get(ref.name)
    own = held.currency.get("symbol") if held is not None and isinstance(held.currency, dict) else None
    return own if isinstance(own, str) and _ISO.match(own) else deployment


def _axis_refusal(refs: tuple[MeasureRef, ...], axis: str, saved: dict[str, SavedMetric],
                  deployment: str) -> str | None:
    kinds = {_kind(ref) for ref in refs}
    if len(kinds) > 1:
        return (f"the {axis}axis would show {' and '.join(sorted(k.value for k in kinds))} measures in one number "
                "format; plot one kind on an axis")
    # Each as the request shows it: `_format` gives the whole axis the first measure's currency.
    currencies = {_currency(ref, saved, deployment) for ref in refs if _kind(ref) is MeasureKind.CURRENCY}
    if len(currencies) > 1:
        return f"the {axis}axis would show more than one currency; plot one currency on an axis"
    return None


def plan_refusal(spec: ChartSpec, *, interactive_pivot_available: bool, saved: dict[str, SavedMetric],
                 currency: str) -> str | None:
    """Why `spec` cannot be built as asked, or None. The checks are on what the chart would show wrongly or cannot be
    built from, never on what Superset would accept. `saved` holds the reused saved metrics by name and `currency` is
    the deployment's ISO code, as the request builders take them: two currency measures on one axis are refused when
    the request would show them in different currencies, each resolved as `generate_chart_request` resolves it (a saved
    metric's own ISO code, a written measure's own currency, else the deployment's; a saved metric whose currency is
    `UNKNOWN` is the deployment's)."""
    family, refs = spec.family, spec.measures
    if family is Family.INTERACTIVE_PIVOT and not interactive_pivot_available:
        return "this Superset does not offer the interactive pivot table; use the pivot table"
    if not refs and family not in _NO_MEASURE_NEEDED:
        return f"a {family.name.lower()} chart needs a measure"
    if spec.row_limit is not None and family not in _NO_ROW_LIMIT:
        if not 1 <= spec.row_limit <= _ROW_LIMIT_MAX.get(family, 50000):
            return (f"a {family.name.lower()} chart takes 1 to {_ROW_LIMIT_MAX.get(family, 50000)} rows, not "
                    f"{spec.row_limit}")
    if family in _ONE_MEASURE and len(refs) != 1:
        return f"a {family.name.lower()} chart plots exactly one measure, not {len(refs)}"
    if family in _NO_CURRENCY_OR_PERCENT:
        for ref in refs:
            if _kind(ref) in (MeasureKind.CURRENCY, MeasureKind.PERCENT):
                return (f"a {family.name.lower()} chart cannot show a {_kind(ref).value} measure: Superset has no "
                        "format for it there")
    if family is Family.HANDLEBARS:
        for ref in refs:
            if _kind(ref) not in (MeasureKind.COUNT, MeasureKind.NUMBER):
                return f"a handlebars chart takes count or number measures only, not {_kind(ref).value}"
    if family in _NEEDS_DIMENSION and not spec.dimensions:
        return f"a {family.name.lower()} chart needs a dimension"
    if family is Family.SANKEY and not (spec.source and spec.target):
        return "a sankey chart needs a source and a target"
    if family is Family.WATERFALL and not (spec.time_column or spec.dimensions):
        return "a waterfall chart needs a column for its steps"
    if family is Family.BOX_PLOT and not (spec.time_column or spec.dimensions):
        return "a box plot needs a column to distribute across"
    if family is Family.MIXED:
        if not spec.time_column:
            return "a mixed chart needs a time column"
        if len(refs) < 2:
            return "a mixed chart needs a primary and a secondary measure"
        return (_axis_refusal(refs[:1], "primary ", saved, currency)
                or _axis_refusal(refs[1:], "secondary ", saved, currency))
    if family not in _ONE_FORMAT_PER_AXIS_EXEMPT:
        return _axis_refusal(refs, "", saved, currency)
    return None


# ---------------------------------------------------------------- the requests

def _resolve(ref: MeasureRef, measures: dict[str, Measure]) -> Measure | SavedRef:
    if isinstance(ref, WrittenRef):
        return measures.get(ref.measure.name, ref.measure)
    return ref


def _column(ref: MeasureRef, measures: dict[str, Measure]) -> dict:
    """The `ColumnRef` of a measure: a saved metric by name, a written one as its expression and label."""
    held = _resolve(ref, measures)
    if isinstance(held, SavedRef):
        return {"name": held.name, "saved_metric": True}
    return {"sql_expression": held.expression, "label": held.label}


def _label(ref: MeasureRef, measures: dict[str, Measure]) -> str:
    held = _resolve(ref, measures)
    return held.name if isinstance(held, SavedRef) else held.label


def _format(refs: tuple[MeasureRef, ...], saved: dict[str, SavedMetric], deployment: str) -> tuple[str, dict | None]:
    """The d3 format and currency of the first measure in `refs`; `plan_refusal` has made the rest agree."""
    return display_format(_kind(refs[0]), _currency(refs[0], saved, deployment))


def _filter(item: Filter) -> dict:
    out: dict = {"column": item.column, "op": item.op}
    if item.value is not None:
        out["value"] = item.value
    return out


def _cols(names) -> list[dict]:
    return [{"name": name} for name in names]


def _shared(spec: ChartSpec, config: dict) -> dict:
    if spec.filters:
        config["filters"] = [_filter(f) for f in spec.filters]
    if spec.row_limit is not None and spec.family not in _NO_ROW_LIMIT:
        config["row_limit"] = spec.row_limit
    return config


def _with_currency(config: dict, key: str, currency: dict | None) -> None:
    if currency is not None:
        config[key] = currency


def _config(spec: ChartSpec, measures: dict[str, Measure], saved: dict[str, SavedMetric], deployment: str) -> dict:
    family, refs, dims = spec.family, spec.measures, list(spec.dimensions)
    columns = [_column(ref, measures) for ref in refs]
    config: dict = {}
    if family in _XY_KINDS:
        x = spec.time_column or (dims[0] if dims else None)
        rest = dims if spec.time_column else dims[1:]
        config = {"chart_type": "xy", "kind": _XY_KINDS[family], "y": columns}
        if x is not None:
            config["x"] = {"name": x}
        if rest:
            config["group_by"] = _cols(rest)
        if spec.time_grain:
            config["time_grain"] = spec.time_grain
        fmt, currency = _format(refs, saved, deployment)
        config["y_axis"] = {"format": fmt}
        _with_currency(config, "currency_format", currency)
    elif family is Family.TABLE:
        config = {"chart_type": "table", "query_mode": "aggregate" if refs else "raw",
                  "columns": [*_cols(dims), *columns]}
        formats = {_label(ref, measures): {"d3NumberFormat": display_format(_kind(ref), None)[0]} for ref in refs}
        if formats:
            config["column_config"] = formats
    elif family is Family.PIE:
        config = {"chart_type": "pie", "dimension": {"name": dims[0]}, "metric": columns[0]}
        config["number_format"], currency = _format(refs, saved, deployment)
        _with_currency(config, "currency_format", currency)
    elif family is Family.BIG_NUMBER:
        config = {"chart_type": "big_number", "metric": columns[0], "show_trendline": False}
        config["y_axis_format"], currency = _format(refs, saved, deployment)
        _with_currency(config, "currency_format", currency)
        if spec.time_grain:
            config["time_grain"] = spec.time_grain
    elif family is Family.GAUGE:
        config = {"chart_type": "gauge", "metric": columns[0]}
        if dims:
            config["groupby"] = _cols(dims)
        config["number_format"], currency = _format(refs, saved, deployment)
        _with_currency(config, "currency_format", currency)
    elif family is Family.HISTOGRAM:
        config = {"chart_type": "histogram", "column": {"name": dims[0]}}
        if dims[1:]:
            config["groupby"] = _cols(dims[1:])
    elif family is Family.BOX_PLOT:
        across = [spec.time_column] if spec.time_column else dims[:1]
        split = dims if spec.time_column else dims[1:]
        config = {"chart_type": "box_plot", "metrics": columns, "distribute_across": _cols(across)}
        if split:
            config["dimensions"] = _cols(split)
        config["number_format"] = _format(refs, saved, deployment)[0]
    elif family in (Family.PIVOT, Family.INTERACTIVE_PIVOT):
        config = {"chart_type": "pivot_table" if family is Family.PIVOT else "interactive_pivot",
                  "rows": _cols(dims[:1]), "columns": _cols(dims[1:]), "metrics": columns}
        if family is Family.PIVOT and not dims[1:]:
            del config["columns"]
        if family is Family.INTERACTIVE_PIVOT and spec.time_grain:
            config["time_grain"] = spec.time_grain
        config["value_format"], currency = _format(refs, saved, deployment)
        _with_currency(config, "currency_format", currency)
    elif family is Family.TREEMAP:
        config = {"chart_type": "treemap_v2", "groupby": _cols(dims), "metric": columns[0]}
    elif family is Family.WATERFALL:
        x = spec.time_column or dims[0]
        breakdown = dims[1:] if not spec.time_column else dims
        config = {"chart_type": "waterfall", "x_axis": {"name": x}, "metric": columns[0]}
        if breakdown:
            config["breakdown"] = {"name": breakdown[0]}
        if spec.time_grain:
            config["time_grain"] = spec.time_grain
        config["y_axis_format"], currency = _format(refs, saved, deployment)
        _with_currency(config, "currency_format", currency)
    elif family is Family.MIXED:
        config = {"chart_type": "mixed_timeseries", "x": {"name": spec.time_column}, "y": columns[:1],
                  "y_secondary": columns[1:], "primary_kind": "line", "secondary_kind": "bar"}
        if dims:
            config["group_by"] = _cols(dims)
            config["group_by_secondary"] = _cols(dims)
        if spec.time_grain:
            config["time_grain"] = spec.time_grain
        fmt, currency = _format(refs[:1], saved, deployment)
        config["y_axis"] = {"format": fmt}
        _with_currency(config, "currency_format", currency)
        fmt, currency = _format(refs[1:], saved, deployment)
        config["y_axis_secondary"] = {"format": fmt}
        _with_currency(config, "currency_format_secondary", currency)
    elif family is Family.HANDLEBARS:
        config = {"chart_type": "handlebars", "handlebars_template": HANDLEBARS_TEMPLATE,
                  "query_mode": "aggregate", "metrics": columns}
        if dims:
            config["groupby"] = _cols(dims)
    if spec.time_column:
        config["temporal_column"] = spec.time_column
    return _shared(spec, config)


def generate_chart_request(spec: ChartSpec, measures: dict[str, Measure], saved: dict[str, SavedMetric], *,
                           currency: str) -> dict:
    """The request for MCP's `generate_chart` that saves `spec` as a chart: `save_chart` on and no preview, since the
    chart is read where it is saved. `measures` holds the written measures by name, and wins over the one a
    `WrittenRef` carries; `saved` holds the reused saved metrics by name. The caller has passed `plan_refusal`."""
    if spec.family in EXTENSION_FAMILIES:
        raise ValueError(f"{spec.family.name.lower()} is not an MCP chart type; build it with extension_chart_request")
    return {"dataset_id": spec.dataset_id, "config": _config(spec, measures, saved, currency), "chart_name": spec.title,
            "save_chart": True, "generate_preview": False}


def owed_format_params(spec: ChartSpec, measures: dict[str, Measure], saved: dict[str, SavedMetric], *,
                       currency: str) -> dict:
    """The top-level `params` keys to set after the chart is created, for a format rc2's renderers read and MCP's
    models cannot send: a table's per-column `currencyFormat` (`TableColumnConfig` has only `d3NumberFormat`) and a
    treemap's `number_format` and `currency_format` (`TreemapChartConfig` has no format field). Empty when the request
    already carries the measure's whole format. The table's value is the whole `column_config`, every column the request
    set included, since a patch replaces a top-level key. Histogram and handlebars have no metric format key at all, so
    they owe nothing."""
    refs, deployment = spec.measures, currency
    if spec.family is Family.TABLE:
        if not any(_kind(ref) is MeasureKind.CURRENCY and _currency(ref, saved, deployment) for ref in refs):
            return {}
        config = {}
        for ref in refs:
            fmt, currency = display_format(_kind(ref), _currency(ref, saved, deployment))
            config[_label(ref, measures)] = {"d3NumberFormat": fmt, **({"currencyFormat": currency} if currency else {})}
        return {"column_config": config}
    if spec.family is Family.TREEMAP:
        fmt, shown = _format(refs, saved, deployment)
        return {"number_format": fmt, **({"currency_format": shown} if shown else {})}
    return {}


# Superset's simple-filter operators, as the explore filter control writes them, for the operators FilterConfig allows.
_SIMPLE_OPERATOR = {"=": "==", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "LIKE": "LIKE",
                    "ILIKE": "ILIKE", "NOT LIKE": "NOT LIKE", "IN": "IN", "NOT IN": "NOT IN",
                    "IS NULL": "IS NULL", "IS NOT NULL": "IS NOT NULL"}


def _adhoc_filter(item: Filter) -> dict:
    out = {"expressionType": "SIMPLE", "clause": "WHERE", "subject": item.column,
           "operator": _SIMPLE_OPERATOR[item.op]}
    if item.value is not None:
        out["comparator"] = item.value
    return out


def _form_metric(ref: MeasureRef, measures: dict[str, Measure]):
    """A form_data metric: a saved metric's name, or an adhoc SQL metric."""
    held = _resolve(ref, measures)
    if isinstance(held, SavedRef):
        return held.name
    return {"expressionType": "SQL", "sqlExpression": held.expression, "label": held.label}


def extension_chart_request(spec: ChartSpec, measures: dict[str, Measure], saved: dict[str, SavedMetric], *,
                            currency: str) -> dict:
    """The request for the extension's `create_chart` for a funnel or a sankey: the minimal `form_data` their plugins
    read. A sankey has no number format control, so it carries none."""
    if spec.family not in EXTENSION_FAMILIES:
        raise ValueError(f"{spec.family.name.lower()} is an MCP chart type; build it with generate_chart_request")
    params: dict = {"viz_type": spec.family.value, "datasource": f"{spec.dataset_id}__table",
                    "metric": _form_metric(spec.measures[0], measures)}
    if spec.family is Family.FUNNEL:
        params["groupby"] = list(spec.dimensions)
        params["number_format"], shown = _format(spec.measures, saved, currency)
        _with_currency(params, "currency_format", shown)
    else:
        params["source"], params["target"] = spec.source, spec.target
    if spec.filters:
        params["adhoc_filters"] = [_adhoc_filter(f) for f in spec.filters]
    if spec.row_limit is not None:
        params["row_limit"] = spec.row_limit
    return {"dataset_id": spec.dataset_id, "slice_name": spec.title, "viz_type": spec.family.value,
            "params": params}

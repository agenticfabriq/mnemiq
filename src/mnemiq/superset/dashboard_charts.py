"""A dashboard's charts, read once as the builder, and which planned charts the dashboard already shows.

A build asked for onto a dashboard plans only what the dashboard does not already show. Each planned chart is compared
with each chart on the dashboard as Superset now holds it (`read_dashboard_charts`: the layout the extension's
`get_layout` answers, then each chart's `get_chart_info`), and a planned chart is left out only when one of them shows
exactly the same thing: the same type and dataset, the same measures by definition (normalized SQL, a saved metric by
its expression on the dataset), the same grouping and x-axis columns, the same time grain, the same filters and the
same row limit. Anything mnemiq cannot read the same way on both sides (a chart it cannot read whole, a filter in SQL,
a dataset the build is still to make) is never a match, so a doubt builds the chart rather than leaving it out.

A chart's view is read with the value check's own reader (`chart_form.expectation_from_form_data`), so "shows the
same" means "computes the same" as the check reads it; a planned chart's view is what that reader finds on the chart
the builder makes from it (`charts.config_chart_request`, mapped as rc2's MCP maps it).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from sqlglot.errors import SqlglotError

from mnemiq.superset.chart_check import (
    ChartRead,
    DatasetFacts,
    NotCheckable,
    dataset_of,
    expectation_from_form_data,
    read_chart,
    read_dataset_facts,
)
from mnemiq.superset.chart_form import _row_limit
from mnemiq.superset.charts import ChartSpec, Family, SavedRef, WrittenRef, leaves_out_no_value
from mnemiq.superset.extension import call_tool
from mnemiq.superset.layout import layout_chart_ids
from mnemiq.superset.saved_metrics import normalized

__all__ = ["DashboardChart", "LeftOut", "datasets_built_on", "leave_out_shown", "read_dashboard_charts"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardChart:
    """A chart the dashboard's layout places, as the builder reads it, with the dataset it reads now (None when that
    cannot be told)."""

    read: ChartRead
    dataset_id: int | None


@dataclass(frozen=True)
class LeftOut:
    """A part of the goal left out of the plan: the planned chart's title, and the dashboard's chart that already shows
    it, by id and title."""

    part: str
    chart_id: int
    title: str | None


def datasets_built_on(measures: Iterable) -> dict[int, int]:
    """Each chart mnemiq made on the dashboard, by id, with the dataset it made it on: what the dashboard's kept
    measures (`BuildStore.measures_for`) record, since a chart rc2's MCP maps saves no dataset in its params."""
    return {chart_id: row.measure.dataset_id for row in measures for chart_id in row.chart_ids
            if type(row.measure.dataset_id) is int and row.measure.dataset_id > 0}


def read_dashboard_charts(client, dashboard_id: int, *, built_on: Mapping[int, int] | None = None,
                          before_call: Callable[[], None] | None = None) -> tuple[DashboardChart, ...]:
    """The charts dashboard `dashboard_id`'s layout places, in id order, each read once as the caller. A layout that
    cannot be read gives none: every planned chart is then built, which is what a build onto it did before.

    A chart's dataset is the one its `form_data` names (a chart saved in Explore names it), else the one `built_on`
    gives for a chart mnemiq made (the dashboard's measures record it), else unknown."""
    answer = call_tool(client, "get_layout", {"dashboard_id": dashboard_id}, before_call=before_call)
    try:
        layout = json.loads(answer["position_json"]) if answer.get("position_json") else {}
    except (KeyError, TypeError, ValueError):
        layout = None
    if answer.get("dashboard_id") != dashboard_id or not isinstance(layout, dict):
        logger.info("the layout of dashboard %s could not be read, so no chart of it is compared", dashboard_id)
        return ()
    charts = []
    for chart_id in sorted(layout_chart_ids(layout)):
        if before_call is not None:
            before_call()
        read = read_chart(client, chart_id)
        made_on = (built_on or {}).get(chart_id, 0)
        dataset = dataset_of(read.form_data, made_on) if read.form_data is not None else None
        charts.append(DashboardChart(read, dataset if type(dataset) is int and dataset > 0 else None))
    return tuple(charts)


# ---------------------------------------------------------------- what a chart shows

@dataclass(frozen=True)
class _View:
    """What a chart shows, in terms both sides can be read in."""

    viz_type: str
    dataset_id: int
    measures: tuple[str, ...]
    dimensions: tuple[str, ...]
    time_grain: str | None
    filters: frozenset
    row_limit: int | None
    not_null: tuple[int, ...]


_XY = {Family.LINE, Family.BAR, Family.AREA, Family.SCATTER}
# What rc2's MCP saves as a chart's row limit when its config sets none (each `*ChartConfig.row_limit` default in
# `superset/mcp_service/chart/schemas.py`). The big number has no row limit, and a funnel or a sankey the extension
# makes keeps none unless the plan sets one.
_MCP_ROW_LIMIT = {Family.LINE: 10000, Family.BAR: 10000, Family.AREA: 10000, Family.SCATTER: 10000,
                  Family.TABLE: 1000, Family.PIE: 100, Family.GAUGE: 10, Family.TREEMAP: 100, Family.PIVOT: 10000}
# rc2's spelling of each filter operator mnemiq plans (`charts._SIMPLE_OPERATOR`, MCP's `map_filter_operator`).
_OPERATOR = {"=": "==", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "LIKE": "LIKE", "ILIKE": "ILIKE",
             "NOT LIKE": "NOT LIKE", "NOT ILIKE": "NOT ILIKE", "IN": "IN", "NOT IN": "NOT IN", "IS NULL": "IS NULL",
             "IS NOT NULL": "IS NOT NULL"}


def _definition(expression: str) -> str:
    """A measure's SQL as `builder._same_definition` compares it: normalized, else as written."""
    try:
        return normalized(expression)
    except SqlglotError:
        return expression.strip()


def _hashable(value):
    return tuple(_hashable(item) for item in value) if isinstance(value, list | tuple) else value


def _filters(items) -> frozenset:
    return frozenset((item.column, item.op, _hashable(item.value)) for item in items)


def _shown_view(chart: DashboardChart, facts: DatasetFacts) -> _View | None:
    """What the dashboard's chart shows, as the value check reads it; None when that cannot be read."""
    form = chart.read.form_data
    if form is None or chart.dataset_id != facts.dataset_id:
        return None
    expectation = expectation_from_form_data(form, dataset=facts)
    if isinstance(expectation, NotCheckable):
        return None
    saved = dict(facts.saved_metric_sql)
    measures = tuple(_definition(saved[label] if label in facts.saved_metric_names and label == sql else sql)
                     for label, sql in expectation.metrics)
    return _View(viz_type=chart.read.viz_type, dataset_id=facts.dataset_id, measures=measures,
                 dimensions=expectation.dimensions, time_grain=expectation.time_grain,
                 filters=_filters(expectation.filters), row_limit=_row_limit(form), not_null=expectation.not_null)


def _planned_dimensions(spec: ChartSpec) -> tuple[str, ...] | None:
    """The grouping columns the chart made from `spec` selects, in order, as the value check reads them."""
    dims = list(spec.dimensions)
    if spec.family in _XY:
        x = spec.time_column or (dims[0] if dims else None)
        if x is None:
            return None
        found = [x, *(dims if spec.time_column else dims[1:])]
    elif spec.family in (Family.TABLE, Family.PIE, Family.GAUGE, Family.TREEMAP, Family.FUNNEL, Family.PIVOT):
        found = dims
    elif spec.family is Family.SANKEY:
        found = [spec.source, spec.target]
    elif spec.family is Family.BIG_NUMBER:
        found = []
    else:  # a type the value check never reads the same way (waterfall, mixed, box plot, histogram, handlebars)
        return None
    return tuple(dict.fromkeys(found))


def _planned_view(spec: ChartSpec, measures: Mapping[str, object], facts: DatasetFacts) -> _View | None:
    """What the chart the builder makes from `spec` shows; None when it could not equal a chart's read view."""
    if spec.dataset_id != facts.dataset_id:
        return None
    dimensions = _planned_dimensions(spec)
    if dimensions is None:
        return None
    saved = dict(facts.saved_metric_sql)
    shown = []
    for ref in spec.measures:
        if isinstance(ref, SavedRef):
            if ref.name not in saved:
                return None
            shown.append(_definition(saved[ref.name]))
        elif isinstance(ref, WrittenRef):
            shown.append(_definition(getattr(measures.get(ref.measure.name), "expression", ref.measure.expression)))
        else:
            return None
    if spec.filters and any(item.op not in _OPERATOR for item in spec.filters):
        return None
    filters = _filters(replace(item, op=_OPERATOR[item.op]) for item in spec.filters)
    limit = spec.row_limit if spec.row_limit is not None else _MCP_ROW_LIMIT.get(spec.family)
    return _View(viz_type=spec.family.value, dataset_id=spec.dataset_id, measures=tuple(shown),
                 dimensions=dimensions, time_grain=spec.time_grain if spec.family in _XY else None,
                 filters=filters, row_limit=limit, not_null=(0,) if leaves_out_no_value(spec) else ())


# ---------------------------------------------------------------- the plan, less what is shown

def leave_out_shown(plan, charts: Sequence[DashboardChart], *, client,
                    before_call: Callable[[], None] | None = None):
    """`plan` without each chart the dashboard's `charts` already show, and those left out (`plan.left_out`), each
    with the chart that shows it. A measure or a dataset no chart that is kept uses is left out with them. The
    datasets the planned charts share with the dashboard's are read once each, as the caller."""
    on = {chart.dataset_id for chart in charts if chart.dataset_id is not None}
    facts: dict[int, DatasetFacts] = {}
    for dataset_id in sorted({spec.dataset_id for spec in plan.charts} & on):
        if before_call is not None:
            before_call()
        found = read_dataset_facts(client, dataset_id)
        if isinstance(found, DatasetFacts):
            facts[dataset_id] = found
    measures = {measure.name: measure for measure in plan.measures}
    left: list[LeftOut] = []
    kept: list[ChartSpec] = []
    for spec in plan.charts:
        dataset = facts.get(spec.dataset_id)
        view = _planned_view(spec, measures, dataset) if dataset is not None else None
        match = next((chart for chart in charts if view is not None and dataset is not None
                      and _shown_view(chart, dataset) == view), None) if view is not None else None
        if match is None:
            kept.append(spec)
        else:
            left.append(LeftOut(part=spec.title, chart_id=match.read.chart_id, title=match.read.title))
    if not left:
        return plan
    written = {ref.measure.name for spec in kept for ref in spec.measures if isinstance(ref, WrittenRef)}
    still = tuple(measure for measure in plan.measures if measure.name in written)
    used = {spec.dataset_id for spec in kept} | {measure.dataset_id for measure in still}
    return replace(plan, charts=tuple(kept), measures=still,
                   datasets=tuple(dataset for dataset in plan.datasets if dataset.placeholder in used),
                   left_out=tuple(left))

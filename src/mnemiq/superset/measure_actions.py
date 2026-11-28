"""A dashboard's measures, as an editor of the dashboard reviews and changes them: listed with each one's value
computed now, as the person asking, and edited, removed or saved to its dataset, each change made through Superset as
that person.

**Listed.** The measures the dashboard keeps (`BuildStore.measures_for`), each with the charts the dashboard still
places that show it, by id and title, and each chart's last check. A chart someone took off the dashboard drops out of
the list. A measure's value is computed each time it is listed, as the person asking, and kept nowhere:
- on a physical dataset, mnemiq's own statement over the dataset's table, decided under the person's grants and run
  through Superset as them, so Superset's row rules apply to it as to any statement they run;
- on a virtual dataset, only through Superset: a saved metric by Superset's own query of the dataset
  (`query_dataset`), a written measure by a chart of it that shows it as one number (`get_chart_data`). Never
  mnemiq's own SQL, since a row rule attached to a virtual dataset applies only through it. Otherwise there is no
  value, and `value_note` says why.
A value says what it covers (`ValueBasis`): mnemiq's statement and Superset's query of the dataset cover every row
the person can read, under no chart's filters, while a chart's one number is under that chart's own filters. A
measure whose charts all carry its condition as a filter shows them right while its value here covers every row.

**A copy** of a measure on a chart is a metric the chart holds that is still the measure as the dashboard keeps it: an
SQL metric with the same label and the same SQL once normalized, or a saved metric's name. A chart the measure is kept
on that holds no copy is one someone changed by hand, and an action leaves it as it is and names it.

**Every action asks first, then changes.** Whether the person edits each chart (and, to save, the dataset) is read
before anything is written (the extension's `editors_of`). Edit and Save change nothing when the person does not
edit every chart holding a copy, and name that chart's editors, so the copies never split; Remove takes such a chart
off the dashboard instead and names it with its dataset. A chart whose settings cannot be read whole is named and left
as it is; a chart Superset no longer holds is gone. No action runs while a build is building on the dashboard (the
route refuses it), since a build keeps the dashboard's measures as it last writes them.
- **Edit** writes the new definition into every copy (`patch_chart`, only the keys it touches), and checks each chart
  it changed again.
- **Remove** deletes the charts that show the measure (MCP's `delete_chart`, which moves them to Superset's trash),
  after the person saw which: the ids it is sent must be the measure's charts now. A chart the person does not edit is
  taken off the dashboard instead and named; one a report holds is named and left. A dataset mnemiq made for the
  measure is deleted once no chart uses it (`delete_dataset`).
- **Save** adds the measure to its dataset as a saved metric (`add_saved_metric`), then switches every copy to it.

Editors' labels go only to the answer, for the panel: never to a model.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

from mnemiq.execute.runner import DEFAULT_TIMEOUT_S, ExecutionError
from mnemiq.superset.chart_check import (
    ChartRead,
    ChartStatus,
    DatasetFacts,
    NotCheckable,
    check_chart,
    expectation_from_form_data,
    read_chart,
    read_dataset_facts,
)
from mnemiq.superset.chart_form import _quoted
from mnemiq.superset.charts import not_null_condition
from mnemiq.superset.client import SupersetToolError, SupersetUnavailable
from mnemiq.superset.dashboard_charts import _definition
from mnemiq.superset.display_format import metric_label
from mnemiq.superset.extension import call_tool
from mnemiq.superset.history import run_decided
from mnemiq.superset.layout import layout_chart_ids, layout_without
from mnemiq.superset.measures import Measure, MeasureKind, MeasureRefused, check_measure, display_format, \
    measure_statement, scales_by_100
from mnemiq.superset.names import UnreadableStatement, from_superset
from mnemiq.superset.saved_metrics import read_saved_metrics
from mnemiq.sql.verdict import Refusal

__all__ = ["ActionRefused", "ActionResult", "Asking", "ChartOf", "ChartsChanged", "DashboardUnreadable",
           "EditorsOf", "MeasureNotFound", "MeasureView", "edit_measure", "list_measures", "remove_measure",
           "save_measure"]

logger = logging.getLogger(__name__)

NOT_KEPT = "This dashboard keeps no measure of that name."
NOT_READABLE = "This person can read no dashboard of that id."
OUTDATED = ("Superset's mnemiq extension is older than this mnemiq and does not say whether this person edits each "
            "chart; update the extension installed there.")
VIRTUAL_NO_VALUE = ("its dataset is virtual, and no chart of it shows it as one number, so mnemiq computes it only "
                    "through Superset's charts")


class MeasureNotFound(LookupError):
    def __init__(self) -> None:
        super().__init__(NOT_KEPT)


class DashboardUnreadable(LookupError):
    def __init__(self) -> None:
        super().__init__(NOT_READABLE)


class ActionRefused(Exception):
    """The action changes nothing, for the reason given in mnemiq's words."""


class ChartsChanged(Exception):
    """Remove was sent other charts than the measure has now: `charts` are the ones it has, for the panel to show."""

    def __init__(self, charts: tuple[ChartOf, ...]) -> None:
        super().__init__("The measure's charts changed since they were shown, so nothing was removed; here are the "
                         "charts it has now.")
        self.charts = charts


@dataclass(frozen=True)
class ChartOf:
    chart_id: int
    title: str | None
    check: ChartStatus | None


class ValueBasis(StrEnum):
    """What a listed value covers. `READABLE_ROWS`: every row of the measure's dataset the person can read, under no
    chart's filters (mnemiq's own statement, or Superset's query of the dataset). `CHART`: the one number a chart shows,
    under that chart's own filters."""

    READABLE_ROWS = "readable_rows"
    CHART = "chart"


class Valued(NamedTuple):
    """A measure's value as listed: the value, or None and `note` saying why; what it covers, and the chart it was read
    from when that is a chart. No value says nothing of what it would cover."""

    value: float | None
    note: str | None
    basis: ValueBasis | None
    chart_id: int | None


@dataclass(frozen=True)
class MeasureView:
    """A measure as it is listed: its definition, the saved metric it is (None: one written on its charts), its charts,
    and its value computed now as the person asking (None, with `value_note` saying why, when there is none)."""

    measure: Measure
    saved_metric: str | None
    charts: tuple[ChartOf, ...]
    value: float | None
    value_note: str | None
    value_basis: ValueBasis | None
    value_chart_id: int | None


@dataclass(frozen=True)
class EditorsOf:
    """An object the person does not edit, and its editors' labels."""

    editors: tuple[str, ...]
    chart_id: int | None = None
    dataset_id: int | None = None


@dataclass(frozen=True)
class ActionResult:
    changed: tuple[int, ...] = ()
    left_by_hand: tuple[int, ...] = ()
    not_editor: tuple[EditorsOf, ...] = ()
    blocked_by_report: tuple[int, ...] = ()
    could_not_read: tuple[int, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class Asking:
    """Who asks, about which dashboard: the request's adapter (every Superset call goes through its client, as the
    person), the build store, and what the person's statements are decided under."""

    adapter: object
    store: object
    tenant: str
    principal: str
    dashboard_id: int
    snapshot: object
    grants: object
    settings: object
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def client(self):
        return self.adapter.client


# ---------------------------------------------------------------- reading


def _placed(asking: Asking) -> set[int]:
    """The charts the dashboard's layout places, as the person reads it; `DashboardUnreadable` when they cannot."""
    answer = call_tool(asking.client, "get_layout", {"dashboard_id": asking.dashboard_id})
    if answer.get("dashboard_id") != asking.dashboard_id:
        raise DashboardUnreadable()
    try:
        layout = json.loads(answer["position_json"]) if answer.get("position_json") else {}
    except (TypeError, ValueError):
        layout = {}
    return layout_chart_ids(layout) if isinstance(layout, dict) else set()


def _kept(asking: Asking) -> dict:
    return {row.measure.name: row for row in asking.store.measures_for(asking.tenant, asking.dashboard_id)}


def _row(asking: Asking, name: str):
    row = _kept(asking).get(name)
    if row is None:
        raise MeasureNotFound()
    return row


def _charts_of(row, placed: set[int], reads: Mapping[int, ChartRead]) -> tuple[ChartOf, ...]:
    """The measure's charts the dashboard places now, a chart Superset no longer holds (in its trash) left out."""
    return tuple(ChartOf(chart_id, reads[chart_id].title if chart_id in reads else None,
                         row.check_status.get(chart_id)) for chart_id in row.chart_ids
                 if chart_id in placed and not (chart_id in reads and reads[chart_id].missing))


def _is_copy(item, row) -> bool:
    """Whether `item`, a metric a chart holds, is still the measure: its SQL with the same label and the same
    definition, or, once it is saved on its dataset, the saved metric's name."""
    if row.saved_metric is not None and item == row.saved_metric:
        return True
    if not isinstance(item, dict) or item.get("expressionType") != "SQL":
        return False
    text = item.get("sqlExpression")
    return (isinstance(text, str) and metric_label(item) == row.measure.label
            and _definition(text) == _definition(row.measure.expression))


def _copies(form: dict, row) -> list[tuple[str, int | None]]:
    """Where `form` holds a copy of the measure: `("metrics", i)`, `("metrics_b", i)` or `("metric", None)`."""
    found: list[tuple[str, int | None]] = []
    for key in ("metrics", "metrics_b"):
        items = form.get(key)
        if isinstance(items, list):
            found += [(key, index) for index, item in enumerate(items) if _is_copy(item, row)]
    if _is_copy(form.get("metric"), row):
        found.append(("metric", None))
    return found


def _editorship(asking: Asking, *, chart_ids: Iterable[int] = (), dataset_ids: Iterable[int] = ()) -> dict:
    """Whether the person edits each chart and dataset, with its editors' labels: `{("chart", id): (bool, labels)}`.
    An object they cannot read is absent; an extension that does not say whether refuses the action."""
    charts, datasets = sorted(set(chart_ids)), sorted(set(dataset_ids))
    found: dict = {}
    if not charts and not datasets:
        return found
    answer = call_tool(asking.client, "editors_of", {"chart_ids": charts, "dataset_ids": datasets})
    for kind, key, plural in (("chart", "chart_id", "charts"), ("dataset", "dataset_id", "datasets")):
        for entry in answer.get(plural) or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("is_editor"), bool):
                raise ActionRefused(OUTDATED)
            found[(kind, entry.get(key))] = (entry["is_editor"], tuple(str(e) for e in entry.get("editors") or []))
    return found


def _not_editors(editorship: dict, kind: str, ids: Iterable[int],
                 dataset_id: int | None = None) -> tuple[EditorsOf, ...]:
    """Each object of `ids` the person does not edit, with its editors' labels; a chart with the dataset it is on."""
    out = []
    for object_id in ids:
        is_editor, editors = editorship.get((kind, object_id), (False, ()))
        if not is_editor:
            out.append(EditorsOf(editors=editors, chart_id=object_id, dataset_id=dataset_id) if kind == "chart"
                       else EditorsOf(editors=editors, dataset_id=object_id))
    return tuple(out)


# ---------------------------------------------------------------- the list


def _value(asking: Asking, row, facts: DatasetFacts | NotCheckable,
           reads: Mapping[int, ChartRead]) -> Valued:
    if isinstance(facts, NotCheckable):
        return _no_value("mnemiq could not read the measure's dataset")
    if not facts.is_virtual:
        if not facts.table:
            return _no_value("mnemiq could not read the measure's table")
        if (facts.database_id, facts.schema) != (getattr(asking.adapter, "database_id", facts.database_id),
                                                 getattr(asking.adapter, "schema", facts.schema)):
            return _no_value("the measure's dataset reads another database or schema than the one mnemiq answers from")
        sql = measure_statement(dataclasses.replace(row.measure, name="value"), _quoted(facts.table))
        try:
            decided = run_decided(sql, adapter=asking.adapter, snapshot=asking.snapshot, grants=asking.grants,
                                  timeout_s=asking.timeout_s, guard_fanout=asking.settings.guard_fanout)
        except ExecutionError as failure:
            logger.warning("a measure's value did not run: %s %s", failure.message, failure.source_detail or "")
            return _no_value("mnemiq's statement for the value did not run; why is in the log")
        if isinstance(decided, Refusal):
            return _no_value(f"mnemiq refused the statement for its value ({decided.code.value}): {decided.message}")
        table = decided.table
        if table.num_rows == 0 or table.num_columns == 0 or table.column(0)[0].as_py() is None:
            return _no_value("no row this person can read has a value for it")
        return _covering(_finite(table.column(0)[0].as_py()), ValueBasis.READABLE_ROWS)
    if row.saved_metric is not None:
        return _covering(_one_number(asking.client, "query_dataset", {"dataset_id": facts.dataset_id,
                                                            "metrics": [row.saved_metric], "columns": [],
                                                            "row_limit": 1}, row.saved_metric),
                         ValueBasis.READABLE_ROWS)
    for chart_id in row.chart_ids:
        read = reads.get(chart_id)
        if read is not None and read.form_data is not None and read.viz_type == "big_number_total" \
                and _copies(read.form_data, row) == [("metric", None)]:
            return _covering(_one_number(asking.client, "get_chart_data", {"identifier": chart_id}, row.measure.label),
                             ValueBasis.CHART, chart_id)
    return _no_value(VIRTUAL_NO_VALUE)


def _no_value(note: str) -> Valued:
    return Valued(None, note, None, None)


def _covering(found: tuple[float | None, str | None], basis: ValueBasis, chart_id: int | None = None) -> Valued:
    """`found` with what it covers, or, when it is no value, its note alone."""
    value, note = found
    return Valued(value, note, basis, chart_id) if value is not None else _no_value(note)


def _one_number(client, tool: str, request: dict, label: str) -> tuple[float | None, str | None]:
    """The one value Superset's `tool` answers under `label`, as the person."""
    try:
        answer = client.call(tool, request)
    except (SupersetUnavailable, SupersetToolError) as failure:
        logger.warning("Superset did not answer a measure's value (%s)", type(failure).__name__)
        return None, "Superset did not answer for its value"
    rows = answer.get("data") if isinstance(answer.get("data"), list) else answer.get("rows")
    if answer.get("error") or answer.get("error_type") or not isinstance(rows, list) or len(rows) != 1 \
            or not isinstance(rows[0], dict) or label not in rows[0]:
        return None, "Superset's answer for its value could not be read as one number"
    value = rows[0][label]
    if value is None:
        return None, "no row this person can read has a value for it"
    return _finite(value)


def _finite(value) -> tuple[float | None, str | None]:
    """A value as a finite number, or no value and why: an infinity or a NaN is not a number to show."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "Superset's answer for its value is not a number"
    if not math.isfinite(number):
        return None, "its value is not a finite number"
    return number, None


def list_measures(asking: Asking) -> list[MeasureView]:
    """The dashboard's measures, each value computed now as the person asking."""
    placed = _placed(asking)
    rows = list(_kept(asking).values())
    reads = {chart_id: read_chart(asking.client, chart_id)
             for chart_id in sorted({c for row in rows for c in row.chart_ids} & placed)}
    facts: dict[int, DatasetFacts | NotCheckable] = {}
    views = []
    for row in rows:
        dataset_id = row.measure.dataset_id
        if dataset_id not in facts:
            facts[dataset_id] = read_dataset_facts(asking.client, dataset_id)
        valued = _value(asking, row, facts[dataset_id], reads)
        views.append(MeasureView(measure=row.measure, saved_metric=row.saved_metric,
                                 charts=_charts_of(row, placed, reads), value=valued.value, value_note=valued.note,
                                 value_basis=valued.basis, value_chart_id=valued.chart_id))
    return views


# ---------------------------------------------------------------- the copies on each chart


@dataclass(frozen=True)
class _Copies:
    """The measure's charts now: each read, those holding a copy, those changed by hand, those unreadable."""

    reads: dict[int, ChartRead]
    holding: dict[int, list[tuple[str, int | None]]]
    by_hand: tuple[int, ...]
    unreadable: tuple[int, ...]
    # Placed, but Superset no longer holds them: deleted, or in its trash.
    gone: tuple[int, ...] = ()


def _the_copies(asking: Asking, row, placed: set[int]) -> _Copies:
    reads, holding, by_hand, unreadable, gone = {}, {}, [], [], []
    for chart_id in row.chart_ids:
        if chart_id not in placed:
            continue
        read = read_chart(asking.client, chart_id)
        reads[chart_id] = read
        if read.missing:
            gone.append(chart_id)
        elif read.form_data is None:
            unreadable.append(chart_id)
        elif places := _copies(read.form_data, row):
            holding[chart_id] = places
        else:
            by_hand.append(chart_id)
    return _Copies(reads, holding, tuple(by_hand), tuple(unreadable), tuple(gone))


def _patch(asking: Asking, chart_id: int, read: ChartRead, to_set: dict, *, again) -> str:
    """`patch_chart` with `to_set` (a key set to None is unset), read again and recomputed once
    (`again(read) -> to_set | None`) when the chart changed meanwhile: "ok", "not_editor", or "changing" (twice
    stale, or no copy left on the second read)."""
    for attempt in range(2):
        answer = call_tool(asking.client, "patch_chart", {
            "chart_id": chart_id, "set": {key: value for key, value in to_set.items() if value is not None},
            "unset": sorted(key for key, value in to_set.items() if value is None),
            "expect_changed_on": read.changed_on})
        if answer.get("ok") is True:
            return "ok"
        if answer.get("not_editor"):
            return "not_editor"
        if not answer.get("stale") or attempt:
            logger.warning("patch_chart answered a measure action with %s", sorted(answer))
            return "changing"
        read = read_chart(asking.client, chart_id)
        recomputed = again(read) if read.form_data is not None else None
        if recomputed is None:
            return "changing"
        to_set = recomputed
    return "changing"


def _recheck(asking: Asking, chart_ids: Iterable[int], dataset_id: int) -> dict[int, ChartStatus]:
    """Each changed chart checked again, with the kind of every measure the dashboard keeps on it."""
    kept = list(_kept(asking).values())
    saved = {metric.name: metric for metric in read_saved_metrics(asking.client, dataset_ids=(dataset_id,))}
    statuses = {}
    for chart_id in chart_ids:
        kinds: dict[str, MeasureKind] = {}
        for row in kept:
            if chart_id in row.chart_ids:
                kinds[row.saved_metric or row.measure.label] = row.measure.kind
        found = check_chart(asking.adapter, chart_id=chart_id, dataset_id=dataset_id, kinds=kinds, saved=saved,
                            snapshot=asking.snapshot, grants=asking.grants,
                            guard_fanout=asking.settings.guard_fanout)
        statuses[chart_id] = found.status
    return statuses


def _put(asking: Asking, row, *, measure: Measure | None = None, chart_ids=None, check_status=None,
         saved_metric: str | None | object = ...) -> None:
    asking.store.put_measure(asking.tenant, asking.dashboard_id, measure or row.measure,
                             chart_ids=row.chart_ids if chart_ids is None else chart_ids,
                             made_dataset_ids=row.made_dataset_ids,
                             check_status=dict(row.check_status) if check_status is None else check_status,
                             saved_metric=row.saved_metric if saved_metric is ... else saved_metric)


def _sentence(done: str, result: ActionResult) -> str:
    parts = [done]
    if result.left_by_hand:
        parts.append("left as changed by hand: " + ", ".join(f"chart {c}" for c in result.left_by_hand))
    if result.could_not_read:
        parts.append("not read whole, so left as they are: " + ", ".join(f"chart {c}" for c in result.could_not_read))
    if result.blocked_by_report:
        parts.append("held by a report, so left on the dashboard: "
                     + ", ".join(f"chart {c}" for c in result.blocked_by_report))
    if result.not_editor:
        parts.append("not edited by this person: " + "; ".join(
            (f"chart {e.chart_id}" + (f" on dataset {e.dataset_id}" if e.dataset_id is not None else "")
             if e.chart_id is not None else f"dataset {e.dataset_id}")
            + (f" (its editors are {', '.join(e.editors)})" if e.editors else "") for e in result.not_editor))
    return "; ".join(parts) + "."


# ---------------------------------------------------------------- edit


# The params key each chart type shows its one measure's format under (as `charts._config` writes them, through MCP's
# mappers), and its currency under `currency_format`. A table keeps a format per column (`column_config`).
_FORMAT_KEY = {"echarts_timeseries_line": "y_axis_format", "echarts_timeseries_bar": "y_axis_format",
               "echarts_area": "y_axis_format", "echarts_timeseries_scatter": "y_axis_format",
               "big_number_total": "y_axis_format", "waterfall": "y_axis_format", "pie": "number_format",
               "gauge_chart": "number_format", "treemap_v2": "number_format", "funnel": "number_format",
               "pivot_table_v2": "value_format"}


def _replaced(item: dict, new: Measure) -> dict:
    return {**item, "sqlExpression": new.expression, "label": new.label, "hasCustomLabel": True}


def _format_set(viz_type: str, form: dict, old: Measure, new: Measure, currency: str | None) -> dict | str:
    """The format keys the chart needs once the measure is `new`, or why the chart cannot take it."""
    if (old.kind, old.currency, old.label) == (new.kind, new.currency, new.label):
        return {}
    fmt, shown = display_format(new.kind, currency if new.kind is MeasureKind.CURRENCY else None)
    if viz_type == "table":
        config = dict(form.get("column_config") or {})
        before = config.pop(old.label, {})
        config[new.label] = {**(before if isinstance(before, dict) else {}), "d3NumberFormat": fmt}
        config[new.label].pop("currencyFormat", None)
        if shown:
            config[new.label]["currencyFormat"] = shown
        return {"column_config": config}
    if (old.kind, old.currency) == (new.kind, new.currency):
        return {}
    key = _FORMAT_KEY.get(viz_type)
    metrics = (form.get("metrics") if isinstance(form.get("metrics"), list) else []) + \
              (form.get("metrics_b") if isinstance(form.get("metrics_b"), list) else [])
    if key is None or len(metrics) > 1:
        return "it shows other measures on the same axis, or keeps no format mnemiq writes"
    return {key: fmt, "currency_format": shown}


def _edit_set(viz_type: str, form: dict, row, new: Measure, currency: str | None) -> dict | str | None:
    """The params keys that make every copy on the chart `new`; None when it holds no copy now."""
    places = _copies(form, row)
    if not places:
        return None
    to_set: dict = {}
    for key, index in places:
        if index is None:
            to_set[key] = _replaced(form[key], new)
        else:
            items = list(to_set.get(key) or form[key])
            items[index] = _replaced(items[index], new)
            to_set[key] = items
    for key in ("timeseries_limit_metric", "series_limit_metric"):
        if _is_copy(form.get(key), row):
            to_set[key] = _replaced(form[key], new)
    old_condition = not_null_condition(row.measure.expression.strip())
    filters = form.get("adhoc_filters") if isinstance(form.get("adhoc_filters"), list) else []
    if any(isinstance(f, dict) and f.get("clause") == "HAVING" and str(f.get("sqlExpression", "")).strip()
           == old_condition for f in filters):
        to_set["adhoc_filters"] = [
            {**f, "sqlExpression": not_null_condition(new.expression)}
            if isinstance(f, dict) and f.get("clause") == "HAVING"
            and str(f.get("sqlExpression", "")).strip() == old_condition else f for f in filters]
    formats = _format_set(viz_type, form, row.measure, new, currency)
    if isinstance(formats, str):
        return formats
    to_set.update(formats)
    return to_set


def _source(asking: Asking, row, facts) -> str:
    if isinstance(facts, NotCheckable):
        raise ActionRefused("mnemiq could not read the measure's dataset, so nothing was changed.")
    if facts.is_virtual:
        if row.measure.dataset_id not in row.made_dataset_ids or not facts.sql:
            raise ActionRefused("The measure's dataset is virtual and not one mnemiq made for this dashboard, so "
                                "mnemiq cannot check a new definition on it; nothing was changed.")
        # Superset holds the SQL in its own spelling, as the build sent it; the deciders read each table as stored.
        try:
            return f'({from_superset(facts.sql)}) AS "virtual_table"'
        except UnreadableStatement:
            raise ActionRefused("mnemiq could not read the SQL of the measure's dataset, so it cannot check a new "
                                "definition on it; nothing was changed.") from None
    return _quoted(facts.table)


def _label_taken(asking: Asking, row, label: str, facts, copies: _Copies) -> str | None:
    """Why `label` cannot be the measure's: Superset keys a chart's values by label, so it may be no other measure's
    the dashboard keeps, no saved metric's name on its dataset, and no other metric's on a chart that shows it."""
    folded = label.casefold()
    if any(other.measure.name != row.measure.name and other.measure.label.casefold() == folded
           for other in _kept(asking).values()):
        return "another measure of this dashboard has that label"
    if isinstance(facts, DatasetFacts) and folded in {name.casefold() for name in facts.saved_metric_names}:
        return "a saved metric of its dataset has that name"
    for read in copies.reads.values():
        form = read.form_data or {}
        items = [*(form.get("metrics") if isinstance(form.get("metrics"), list) else []),
                 *(form.get("metrics_b") if isinstance(form.get("metrics_b"), list) else []), form.get("metric")]
        if any(item is not None and not _is_copy(item, row) and (metric_label(item) or "").casefold() == folded
               for item in items):
            return f"chart {read.chart_id} shows another metric under that label"
    return None


def edit_measure(asking: Asking, name: str, *, label: str, expression: str, kind: MeasureKind,
                 currency: str | None, description: str) -> ActionResult:
    """The measure `name` made `label`, `expression`, `kind`, `currency` and `description` on every copy, once the
    new definition passes mnemiq's checks and the person edits every chart holding a copy. The copies never split:
    when a chart keeps changing under the edit, the charts already changed are put back, and the dashboard keeps
    the old definition."""
    row = _row(asking, name)
    if row.saved_metric is not None:
        raise ActionRefused("This measure is a saved metric of its dataset: edit it in Superset's dataset editor, "
                            "where every chart that uses it follows.")
    new = dataclasses.replace(row.measure, label=label, expression=expression, kind=MeasureKind(kind),
                              currency=currency if MeasureKind(kind) is MeasureKind.CURRENCY else None,
                              description=description)
    if new.kind is MeasureKind.PERCENT and scales_by_100(new.expression):
        raise ActionRefused("A percent measure is a fraction, which Superset shows times 100: write it without the "
                            "multiplication by 100.")
    placed = _placed(asking)
    copies = _the_copies(asking, row, placed)
    facts = read_dataset_facts(asking.client, row.measure.dataset_id)
    if new.label.casefold() != row.measure.label.casefold() and \
            (taken := _label_taken(asking, row, new.label, facts, copies)) is not None:
        raise ActionRefused(f"The label {new.label!r} cannot be used: {taken}; nothing was changed.")
    dimensions: set[str] = set()
    for read in copies.reads.values():
        if read.form_data is not None and isinstance(facts, DatasetFacts) and \
                not isinstance(found := expectation_from_form_data(read.form_data, dataset=facts), NotCheckable):
            dimensions.update(found.dimensions)
    checked = check_measure(new, source=_source(asking, row, facts), dimension_names=tuple(sorted(dimensions)),
                            adapter=asking.adapter, snapshot=asking.snapshot, grants=asking.grants,
                            guard_fanout=asking.settings.guard_fanout)
    if isinstance(checked, MeasureRefused):
        raise ActionRefused(f"The new definition is refused, so nothing was changed: {checked.reason}.")
    deployment = asking.settings.superset_currency
    plans: dict[int, dict] = {}
    for chart_id in copies.holding:
        read = copies.reads[chart_id]
        found = _edit_set(read.viz_type, read.form_data, row, new, new.currency or deployment)
        if isinstance(found, str):
            raise ActionRefused(f"Chart {chart_id} cannot take the new kind or currency ({found}), so nothing was "
                                "changed; change its kind there by hand, or remove the measure and plan it again.")
        plans[chart_id] = found
    editorship = _editorship(asking, chart_ids=plans)
    blocked = _not_editors(editorship, "chart", plans, row.measure.dataset_id)
    if blocked:
        result = ActionResult(left_by_hand=copies.by_hand, not_editor=blocked, could_not_read=copies.unreadable)
        return dataclasses.replace(result, message=_sentence("Nothing was changed, since this person does not edit "
                                                             "every chart that shows the measure", result))

    def towards(target_row, to: Measure):
        return lambda read: (found if isinstance(found := _edit_set(read.viz_type, read.form_data, target_row, to,
                                                                    to.currency or deployment), dict) else None)

    changed, failed = [], []
    for chart_id, to_set in plans.items():
        outcome = _patch(asking, chart_id, copies.reads[chart_id], to_set, again=towards(row, new))
        (changed if outcome == "ok" else failed).append(chart_id)
    if failed:
        # Put back what was changed, so no chart holds a definition the dashboard does not keep.
        edited_row = dataclasses.replace(row, measure=new)
        split = []
        for chart_id in changed:
            read = read_chart(asking.client, chart_id)
            back = towards(edited_row, row.measure)(read) if read.form_data is not None else None
            if back is None or _patch(asking, chart_id, read, back, again=towards(edited_row, row.measure)) != "ok":
                split.append(chart_id)
        result = ActionResult(left_by_hand=copies.by_hand, could_not_read=(*copies.unreadable, *failed))
        said = "Nothing was changed: " + ", ".join(f"chart {c}" for c in failed) + \
            " kept changing while it was edited, so the charts already changed were put back"
        if split:
            said += "; " + ", ".join(f"chart {c}" for c in split) + " could not be put back and shows the new " \
                    "definition, so edit it by hand or edit the measure again"
        return dataclasses.replace(result, message=_sentence(said, result))
    status = dict(row.check_status)
    status.update(_recheck_with(asking, changed, row.measure.dataset_id, {row.measure.name: new}))
    _put(asking, row, measure=new, check_status=status)
    result = ActionResult(changed=tuple(changed), left_by_hand=copies.by_hand, could_not_read=copies.unreadable)
    return dataclasses.replace(result, message=_sentence(
        f"The measure {name!r} now reads as its new definition on {len(changed)} chart"
        f"{'' if len(changed) == 1 else 's'}, each checked again", result))


def _recheck_with(asking: Asking, chart_ids: Iterable[int], dataset_id: int, replacing: Mapping[str, Measure]):
    """`_recheck`, each measure of `replacing` read as its new definition (not yet kept)."""
    if not chart_ids:
        return {}
    kept = list(_kept(asking).values())
    saved = {metric.name: metric for metric in read_saved_metrics(asking.client, dataset_ids=(dataset_id,))}
    statuses = {}
    for chart_id in chart_ids:
        kinds: dict[str, MeasureKind] = {}
        for row in kept:
            if chart_id in row.chart_ids:
                measure = replacing.get(row.measure.name, row.measure)
                kinds[row.saved_metric or measure.label] = measure.kind
        statuses[chart_id] = check_chart(asking.adapter, chart_id=chart_id, dataset_id=dataset_id, kinds=kinds,
                                         saved=saved, snapshot=asking.snapshot, grants=asking.grants,
                                         guard_fanout=asking.settings.guard_fanout).status
    return statuses


# ---------------------------------------------------------------- remove


def remove_measure(asking: Asking, name: str, *, chart_ids: Iterable[int]) -> ActionResult:
    """Delete the charts that show measure `name`, `chart_ids` being the ones the person was shown (the list's): a
    chart Superset no longer holds is not among them, and is taken off the layout as gone. Only a chart that still
    holds a copy is deleted: one changed by hand is left and named, as is one that cannot be read whole. A chart the
    person does not edit is taken off the dashboard instead, and named with its dataset and its editors; one a
    report holds is named and left. A dataset mnemiq made for the measure is deleted once no chart uses it."""
    row = _row(asking, name)
    placed = _placed(asking)
    copies = _the_copies(asking, row, placed)
    now = tuple(chart_id for chart_id in row.chart_ids if chart_id in placed and chart_id not in copies.gone)
    if sorted(set(chart_ids)) != sorted(now):
        raise ChartsChanged(_charts_of(row, placed, copies.reads))
    editorship = _editorship(asking, chart_ids=copies.holding)
    deleted, reported, off, failed, gone, permanent = [], [], [], [], list(copies.gone), False
    for chart_id in copies.holding:
        is_editor, _ = editorship.get(("chart", chart_id), (False, ()))
        if not is_editor:
            off.append(chart_id)
            continue
        answer = asking.client.call("delete_chart", {"identifier": chart_id})
        if answer.get("success") is True:
            deleted.append(chart_id)
            permanent = permanent or answer.get("soft_deleted") is False
        elif answer.get("error_type") == "ReportsExist":
            reported.append(chart_id)
        elif answer.get("error_type") == "NotFound":
            gone.append(chart_id)
        elif answer.get("permission_denied"):
            off.append(chart_id)
        else:
            logger.warning("delete_chart answered a remove with %s", answer.get("error_type"))
            failed.append(chart_id)
    taking = set(deleted) | set(off) | set(gone)
    layout_note = None
    if taking:
        try:
            _take_off(asking, taking)
        except ActionRefused as refused:
            layout_note = str(refused)
    for dataset_id in row.made_dataset_ids:
        answer = call_tool(asking.client, "delete_dataset", {"dataset_id": dataset_id,
                                                             "dashboard_id": asking.dashboard_id})
        if answer.get("ok") is not True:
            logger.info("a dataset mnemiq made for a removed measure was kept (%s)", sorted(answer))
    left_here = set(reported) | set(failed) | set(copies.unreadable)
    for other in _kept(asking).values():
        if other.measure.name == name:
            kept = tuple(chart_id for chart_id in other.chart_ids if chart_id in left_here)
            if kept:
                _put(asking, other, chart_ids=kept,
                     check_status={c: s for c, s in other.check_status.items() if c in kept})
            else:
                asking.store.delete_measure(asking.tenant, asking.dashboard_id, name)
            continue
        left = tuple(chart_id for chart_id in other.chart_ids if chart_id not in taking)
        if left != tuple(other.chart_ids):
            _put(asking, other, chart_ids=left,
                 check_status={c: s for c, s in other.check_status.items() if c in left})
    result = ActionResult(changed=tuple(deleted), left_by_hand=copies.by_hand,
                          not_editor=_not_editors(editorship, "chart", off, row.measure.dataset_id),
                          blocked_by_report=tuple(reported), could_not_read=(*copies.unreadable, *failed))
    where = "deleted for good (this Superset keeps no trash)" if permanent else "moved to Superset's trash"
    said = (f"The measure {name!r} was removed: {len(deleted)} chart{'' if len(deleted) == 1 else 's'} {where}"
            + (", the rest taken off the dashboard" if off else ""))
    if layout_note:
        said += f"; {layout_note}"
    return dataclasses.replace(result, message=_sentence(said, result))


def _take_off(asking: Asking, chart_ids: set[int]) -> None:
    """The charts taken off the dashboard's layout, which sets its charts; read and computed again once when the
    dashboard changed meanwhile. `ActionRefused` when it cannot be done."""
    for _ in range(2):
        answer = call_tool(asking.client, "get_layout", {"dashboard_id": asking.dashboard_id})
        try:
            current = json.loads(answer["position_json"]) if answer.get("position_json") else {}
        except (TypeError, ValueError):
            current = None
        if not isinstance(current, dict) or not isinstance(answer.get("changed_on"), str):
            raise ActionRefused(f"The layout of dashboard {asking.dashboard_id} could not be read, so the charts "
                                "were not taken off it")
        removing = sorted(chart_ids & layout_chart_ids(current))
        if not removing:
            return
        answer = call_tool(asking.client, "set_layout", {
            "dashboard_id": asking.dashboard_id, "positions": layout_without(current, removing), "add_chart_ids": [],
            "remove_chart_ids": removing, "expect_changed_on": answer["changed_on"]})
        if answer.get("ok") is True:
            return
        if not answer.get("stale"):
            raise ActionRefused(f"The charts could not be taken off dashboard {asking.dashboard_id} "
                                f"({sorted(answer)})")
    raise ActionRefused(f"Dashboard {asking.dashboard_id} kept changing while its charts were taken off it")


# ---------------------------------------------------------------- save


def _switch(read: ChartRead, row, name: str) -> dict | None:
    """The params keys that make each written copy on the chart the saved metric `name`; None when it holds none."""
    form = read.form_data or {}
    places = [(key, index) for key, index in _copies(form, row)
              if (form[key] if index is None else form[key][index]) != name]
    if not places:
        return None
    to_set: dict = {}
    for key, index in places:
        if index is None:
            to_set[key] = name
        else:
            items = list(to_set.get(key) or form[key])
            items[index] = name
            to_set[key] = items
    if read.viz_type == "table":
        config = dict(form.get("column_config") or {})
        if row.measure.label in config:
            config[name] = config.pop(row.measure.label)
            to_set["column_config"] = config
    return to_set


def save_measure(asking: Asking, name: str) -> ActionResult:
    """Save measure `name` on its dataset as a saved metric, then switch every copy to it. Like Edit, it changes
    nothing unless the person edits the dataset and every chart holding a copy, and names whom to ask otherwise, so
    the copies never split. Asked again (an earlier save whose answer was lost, or copies that could not be switched),
    it switches what is left, onto a saved metric of the measure's own definition."""
    row = _row(asking, name)
    measure = row.measure
    dataset_editors = _editorship(asking, dataset_ids=(measure.dataset_id,))
    blocked = _not_editors(dataset_editors, "dataset", (measure.dataset_id,))
    if blocked:
        result = ActionResult(not_editor=blocked)
        return dataclasses.replace(result, message=_sentence("Nothing was saved, since this person does not edit the "
                                                             "measure's dataset", result))
    placed = _placed(asking)
    copies = _the_copies(asking, row, placed)
    switching = {chart_id: switch for chart_id in copies.holding
                 if (switch := _switch(copies.reads[chart_id], row, measure.name)) is not None}
    editorship = _editorship(asking, chart_ids=switching)
    not_editor = _not_editors(editorship, "chart", switching, measure.dataset_id)
    if not_editor:
        result = ActionResult(left_by_hand=copies.by_hand, not_editor=not_editor, could_not_read=copies.unreadable)
        return dataclasses.replace(result, message=_sentence("Nothing was saved, since this person does not edit "
                                                             "every chart that shows the measure", result))
    if row.saved_metric is None:
        fmt, shown = display_format(measure.kind, measure.currency or (
            asking.settings.superset_currency if measure.kind is MeasureKind.CURRENCY else None))
        metric = {"metric_name": measure.name, "expression": measure.expression, "verbose_name": measure.label,
                  "d3format": fmt, "description": measure.description or None, "currency": shown}
        _add_metric(asking, measure.dataset_id, metric)
    saved_row = dataclasses.replace(row, saved_metric=measure.name)
    _put(asking, row, saved_metric=measure.name)
    switched, failed = [], []
    for chart_id, to_set in switching.items():
        outcome = _patch(asking, chart_id, copies.reads[chart_id], to_set,
                         again=lambda read: _switch(read, row, measure.name))
        (switched if outcome == "ok" else failed).append(chart_id)
    if switched:
        status = {**row.check_status, **_recheck(asking, switched, measure.dataset_id)}
        _put(asking, saved_row, check_status=status, saved_metric=measure.name)
    result = ActionResult(changed=tuple(switched), left_by_hand=copies.by_hand,
                          could_not_read=(*copies.unreadable, *failed))
    said = (f"The measure {name!r} is saved on dataset {measure.dataset_id}, and {len(switched)} chart"
            f"{'' if len(switched) == 1 else 's'} now use{'s' if len(switched) == 1 else ''} it")
    if failed:
        said += "; save again to switch the charts that kept changing"
    return dataclasses.replace(result, message=_sentence(said, result))


def _add_metric(asking: Asking, dataset_id: int, metric: dict) -> None:
    """The saved metric added to the dataset, read and sent again once when the dataset changed meanwhile. One the
    dataset already holds under the name with the same definition (an earlier save whose answer was lost) is taken
    as saved; one with another definition refuses the save."""
    for _ in range(2):
        info = asking.client.call("get_dataset_info", {"identifier": dataset_id,
                                                       "select_columns": ["id", "changed_on", "metrics"]})
        changed_on = info.get("changed_on") if isinstance(info, dict) else None
        if not isinstance(changed_on, str):
            raise ActionRefused("mnemiq could not read the measure's dataset, so nothing was saved.")
        held = {m.get("metric_name"): m.get("expression") for m in info.get("metrics") or [] if isinstance(m, dict)}
        if metric["metric_name"] in held:
            if _definition(str(held[metric["metric_name"]] or "")) == _definition(metric["expression"]):
                return
            raise ActionRefused(f"Dataset {dataset_id} already has a saved metric named {metric['metric_name']!r} "
                                "with another definition, so nothing was saved.")
        answer = call_tool(asking.client, "add_saved_metric", {"dataset_id": dataset_id, "metric": metric,
                                                               "expect_changed_on": changed_on})
        if answer.get("ok") is True:
            return
        if answer.get("refused") == "row_level":
            raise ActionRefused("Superset reads the measure as giving one value per row, so it was not saved.")
        if not answer.get("stale") and not answer.get("exists"):
            raise ActionRefused(f"Superset did not save the metric ({sorted(answer)}), so nothing was changed.")
    raise ActionRefused(f"Dataset {dataset_id} kept changing while the metric was saved; nothing was changed.")

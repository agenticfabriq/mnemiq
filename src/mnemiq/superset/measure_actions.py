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

**A copy** of a measure on a chart is a metric the chart holds that is still the measure as the dashboard keeps it: an
SQL metric with the same label and the same SQL once normalized, or a saved metric's name. A chart the measure is kept
on that holds no copy is one someone changed by hand, and an action leaves it as it is and names it.

**Every action asks first, then changes.** Whether the person edits each chart (and, to save, the dataset) is read
before anything is written (the extension's `editors_of`), so an action that one chart would refuse changes nothing
and names that chart's editors. A chart whose settings cannot be read whole is named and left as it is.
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True)
class MeasureView:
    """A measure as it is listed: its definition, the saved metric it is (None: one written on its charts), its charts,
    and its value computed now as the person asking (None, with `value_note` saying why, when there is none)."""

    measure: Measure
    saved_metric: str | None
    charts: tuple[ChartOf, ...]
    value: float | None
    value_note: str | None


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
    return tuple(ChartOf(chart_id, reads[chart_id].title if chart_id in reads else None,
                         row.check_status.get(chart_id)) for chart_id in row.chart_ids if chart_id in placed)


def _is_copy(item, row) -> bool:
    if row.saved_metric is not None:
        return item == row.saved_metric
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


def _not_editors(editorship: dict, kind: str, ids: Iterable[int]) -> tuple[EditorsOf, ...]:
    out = []
    for object_id in ids:
        is_editor, editors = editorship.get((kind, object_id), (False, ()))
        if not is_editor:
            out.append(EditorsOf(editors=editors, **({"chart_id": object_id} if kind == "chart"
                                                     else {"dataset_id": object_id})))
    return tuple(out)


# ---------------------------------------------------------------- the list


def _value(asking: Asking, row, facts: DatasetFacts | NotCheckable,
           reads: Mapping[int, ChartRead]) -> tuple[float | None, str | None]:
    if isinstance(facts, NotCheckable):
        return None, "mnemiq could not read the measure's dataset"
    if not facts.is_virtual:
        if not facts.table:
            return None, "mnemiq could not read the measure's table"
        sql = measure_statement(dataclasses.replace(row.measure, name="value"), _quoted(facts.table))
        try:
            decided = run_decided(sql, adapter=asking.adapter, snapshot=asking.snapshot, grants=asking.grants,
                                  timeout_s=asking.timeout_s, guard_fanout=asking.settings.guard_fanout)
        except ExecutionError as failure:
            logger.warning("a measure's value did not run: %s %s", failure.message, failure.source_detail or "")
            return None, "mnemiq's statement for the value did not run; why is in the log"
        if isinstance(decided, Refusal):
            return None, f"mnemiq refused the statement for its value ({decided.code.value}): {decided.message}"
        table = decided.table
        if table.num_rows == 0 or table.num_columns == 0 or table.column(0)[0].as_py() is None:
            return None, "no row this person can read has a value for it"
        return float(table.column(0)[0].as_py()), None
    if row.saved_metric is not None:
        return _one_number(asking.client, "query_dataset", {"dataset_id": facts.dataset_id,
                                                            "metrics": [row.saved_metric], "columns": [],
                                                            "row_limit": 1}, row.saved_metric)
    for chart_id in row.chart_ids:
        read = reads.get(chart_id)
        if read is not None and read.form_data is not None and read.viz_type == "big_number_total" \
                and _copies(read.form_data, row) == [("metric", None)]:
            return _one_number(asking.client, "get_chart_data", {"identifier": chart_id}, row.measure.label)
    return None, VIRTUAL_NO_VALUE


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
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, "Superset's answer for its value is not a number"


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
        value, note = _value(asking, row, facts[dataset_id], reads)
        views.append(MeasureView(measure=row.measure, saved_metric=row.saved_metric,
                                 charts=_charts_of(row, placed, reads), value=value, value_note=note))
    return views


# ---------------------------------------------------------------- the copies on each chart


@dataclass(frozen=True)
class _Copies:
    """The measure's charts now: each read, those holding a copy, those changed by hand, those unreadable."""

    reads: dict[int, ChartRead]
    holding: dict[int, list[tuple[str, int | None]]]
    by_hand: tuple[int, ...]
    unreadable: tuple[int, ...]


def _the_copies(asking: Asking, row, placed: set[int]) -> _Copies:
    reads, holding, by_hand, unreadable = {}, {}, [], []
    for chart_id in row.chart_ids:
        if chart_id not in placed:
            continue
        read = read_chart(asking.client, chart_id)
        reads[chart_id] = read
        if read.form_data is None:
            unreadable.append(chart_id)
        elif places := _copies(read.form_data, row):
            holding[chart_id] = places
        else:
            by_hand.append(chart_id)
    return _Copies(reads, holding, tuple(by_hand), tuple(unreadable))


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
            (f"chart {e.chart_id}" if e.chart_id is not None else f"dataset {e.dataset_id}")
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
        return f'({facts.sql}) AS "virtual_table"'
    return _quoted(facts.table)


def edit_measure(asking: Asking, name: str, *, label: str, expression: str, kind: MeasureKind,
                 currency: str | None, description: str) -> ActionResult:
    """The measure `name` made `label`, `expression`, `kind`, `currency` and `description` on every copy, once the
    new definition passes mnemiq's checks and the person edits every chart holding a copy."""
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
    blocked = _not_editors(editorship, "chart", plans)
    if blocked:
        result = ActionResult(left_by_hand=copies.by_hand, not_editor=blocked, could_not_read=copies.unreadable)
        return dataclasses.replace(result, message=_sentence("Nothing was changed, since this person does not edit "
                                                             "every chart that shows the measure", result))
    changed, unreadable, refused = [], list(copies.unreadable), []
    for chart_id, to_set in plans.items():
        outcome = _patch(asking, chart_id, copies.reads[chart_id], to_set,
                         again=lambda read: (found if isinstance(
                             found := _edit_set(read.viz_type, read.form_data, row, new, new.currency or deployment),
                             dict) else None))
        if outcome == "ok":
            changed.append(chart_id)
        elif outcome == "not_editor":
            refused.append(chart_id)
        else:
            unreadable.append(chart_id)
    status = dict(row.check_status)
    _put(asking, row, measure=new, check_status=status)
    if changed:
        status.update(_recheck(asking, changed, row.measure.dataset_id))
        _put(asking, row, measure=new, check_status=status)
    result = ActionResult(changed=tuple(changed), left_by_hand=copies.by_hand,
                          not_editor=_not_editors({}, "chart", refused), could_not_read=tuple(unreadable))
    return dataclasses.replace(result, message=_sentence(
        f"The measure {name!r} now reads as its new definition on {len(changed)} chart"
        f"{'' if len(changed) == 1 else 's'}, each checked again", result))


# ---------------------------------------------------------------- remove


def remove_measure(asking: Asking, name: str, *, chart_ids: Iterable[int]) -> ActionResult:
    """Delete the charts that show measure `name`, `chart_ids` being the ones the person was shown. A chart they do
    not edit is taken off the dashboard and named; one a report holds is named and left."""
    row = _row(asking, name)
    placed = _placed(asking)
    now = tuple(chart_id for chart_id in row.chart_ids if chart_id in placed)
    if sorted(set(chart_ids)) != sorted(now):
        reads = {chart_id: read_chart(asking.client, chart_id) for chart_id in now}
        raise ChartsChanged(_charts_of(row, placed, reads))
    editorship = _editorship(asking, chart_ids=now)
    deleted, reported, off, failed = [], [], [], []
    for chart_id in now:
        is_editor, _ = editorship.get(("chart", chart_id), (False, ()))
        if not is_editor:
            off.append(chart_id)
            continue
        answer = asking.client.call("delete_chart", {"identifier": chart_id})
        if answer.get("success") is True:
            deleted.append(chart_id)
        elif answer.get("error_type") == "ReportsExist":
            reported.append(chart_id)
        elif answer.get("permission_denied"):
            off.append(chart_id)
        else:
            logger.warning("delete_chart answered a remove with %s", answer.get("error_type"))
            failed.append(chart_id)
    gone = set(deleted) | set(off)
    if gone:
        _take_off(asking, gone)
    for dataset_id in row.made_dataset_ids:
        answer = call_tool(asking.client, "delete_dataset", {"dataset_id": dataset_id,
                                                             "dashboard_id": asking.dashboard_id})
        if answer.get("ok") is not True:
            logger.info("a dataset mnemiq made for a removed measure was kept (%s)", sorted(answer))
    for other in _kept(asking).values():
        left = tuple(chart_id for chart_id in other.chart_ids if chart_id not in gone)
        if other.measure.name == name:
            if reported or failed:
                kept = tuple(chart_id for chart_id in other.chart_ids if chart_id in set(reported) | set(failed))
                _put(asking, other, chart_ids=kept,
                     check_status={c: s for c, s in other.check_status.items() if c in kept})
            else:
                asking.store.delete_measure(asking.tenant, asking.dashboard_id, name)
        elif left != tuple(other.chart_ids):
            _put(asking, other, chart_ids=left,
                 check_status={c: s for c, s in other.check_status.items() if c in left})
    result = ActionResult(changed=tuple(deleted), not_editor=_not_editors(editorship, "chart", off),
                          blocked_by_report=tuple(reported), could_not_read=tuple(failed))
    return dataclasses.replace(result, message=_sentence(
        f"The measure {name!r} was removed: {len(deleted)} chart{'' if len(deleted) == 1 else 's'} moved to "
        "Superset's trash" + (", the rest taken off the dashboard" if off else ""), result))


def _take_off(asking: Asking, chart_ids: set[int]) -> None:
    """The charts taken off the dashboard's layout, which sets its charts; read and computed again once when the
    dashboard changed meanwhile."""
    for _ in range(2):
        answer = call_tool(asking.client, "get_layout", {"dashboard_id": asking.dashboard_id})
        current = json.loads(answer["position_json"]) if answer.get("position_json") else {}
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
                                f"({sorted(answer)}); the measure's charts are as the answer says.")
    raise ActionRefused(f"Dashboard {asking.dashboard_id} kept changing while its charts were taken off it.")


# ---------------------------------------------------------------- save


def save_measure(asking: Asking, name: str) -> ActionResult:
    """Save measure `name` on its dataset as a saved metric, then switch every copy to it. A person who does not edit
    the dataset is told its editors, and nothing changes."""
    row = _row(asking, name)
    if row.saved_metric is not None:
        raise ActionRefused("This measure is already a saved metric of its dataset.")
    measure = row.measure
    dataset_editors = _editorship(asking, dataset_ids=(measure.dataset_id,))
    blocked = _not_editors(dataset_editors, "dataset", (measure.dataset_id,))
    if blocked:
        result = ActionResult(not_editor=blocked)
        return dataclasses.replace(result, message=_sentence("Nothing was saved, since this person does not edit the "
                                                             "measure's dataset", result))
    placed = _placed(asking)
    copies = _the_copies(asking, row, placed)
    editorship = _editorship(asking, chart_ids=copies.holding)
    not_editor = _not_editors(editorship, "chart", copies.holding)
    fmt, shown = display_format(measure.kind, measure.currency or (
        asking.settings.superset_currency if measure.kind is MeasureKind.CURRENCY else None))
    metric = {"metric_name": measure.name, "expression": measure.expression, "verbose_name": measure.label,
              "d3format": fmt, "description": measure.description or None, "currency": shown}
    _add_metric(asking, measure.dataset_id, metric)
    switched, unreadable = [], list(copies.unreadable)
    saved_row = dataclasses.replace(row, saved_metric=measure.name)
    others = {entry.chart_id for entry in not_editor}
    for chart_id in copies.holding:
        if chart_id in others:
            continue
        read = copies.reads[chart_id]

        def switch(read: ChartRead) -> dict | None:
            places = _copies(read.form_data, row)
            if not places:
                return None
            to_set: dict = {}
            for key, index in places:
                if index is None:
                    to_set[key] = measure.name
                else:
                    items = list(to_set.get(key) or read.form_data[key])
                    items[index] = measure.name
                    to_set[key] = items
            if read.viz_type == "table":
                config = dict(read.form_data.get("column_config") or {})
                if measure.label in config:
                    config[measure.name] = config.pop(measure.label)
                    to_set["column_config"] = config
            return to_set

        outcome = _patch(asking, chart_id, read, switch(read) or {}, again=switch)
        (switched if outcome == "ok" else unreadable).append(chart_id)
    _put(asking, saved_row, saved_metric=measure.name)
    status = dict(row.check_status)
    if switched:
        status.update(_recheck(asking, switched, measure.dataset_id))
        _put(asking, saved_row, check_status=status, saved_metric=measure.name)
    result = ActionResult(changed=tuple(switched), left_by_hand=copies.by_hand, not_editor=not_editor,
                          could_not_read=tuple(unreadable))
    return dataclasses.replace(result, message=_sentence(
        f"The measure {name!r} is saved on dataset {measure.dataset_id}, and {len(switched)} chart"
        f"{'' if len(switched) == 1 else 's'} now use{'s' if len(switched) == 1 else ''} it", result))


def _add_metric(asking: Asking, dataset_id: int, metric: dict) -> None:
    """The saved metric added to the dataset, read and sent again once when the dataset changed meanwhile."""
    for _ in range(2):
        info = asking.client.call("get_dataset_info", {"identifier": dataset_id, "select_columns": ["id", "changed_on"]})
        changed_on = info.get("changed_on") if isinstance(info, dict) else None
        if not isinstance(changed_on, str):
            raise ActionRefused("mnemiq could not read the measure's dataset, so nothing was saved.")
        answer = call_tool(asking.client, "add_saved_metric", {"dataset_id": dataset_id, "metric": metric,
                                                               "expect_changed_on": changed_on})
        if answer.get("ok") is True:
            return
        if answer.get("exists"):
            raise ActionRefused(f"Dataset {dataset_id} already has a saved metric named {metric['metric_name']!r}, "
                                "so nothing was saved.")
        if answer.get("refused") == "row_level":
            raise ActionRefused("Superset reads the measure as giving one value per row, so it was not saved.")
        if not answer.get("stale"):
            raise ActionRefused(f"Superset did not save the metric ({sorted(answer)}), so nothing was changed.")
    raise ActionRefused(f"Dataset {dataset_id} kept changing while the metric was saved; nothing was changed.")

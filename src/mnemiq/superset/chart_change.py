"""Changing a dashboard by chat: a person's request, read by the profile's model against what the dashboard's charts
show now, made into changes of the charts' own settings and written through the extension's `patch_chart`, which sets
and unsets only the keys named. Whatever a chart's settings hold that the request does not name, a hand edit among
them, stays as it is: mnemiq never renders a chart again from its plan, and never sends MCP's `update_chart`, which
replaces a chart's settings wholesale.

**What the model sees** (`dashboard_summary`): each chart the dashboard places, by id, title and type, its dataset's
table, its measures by label, its grouping and x-axis columns, its time grain, row limit and filters, and the columns
of its table the person may read. Never an editor, a value read from the data, or Superset's words.

**What the model may answer** (`Interpretation`): changes of charts (keys to set or unset, filters to add, the
columns whose filters to drop), charts to remove, or that the request adds a chart, which the route builds onto the
dashboard as a build (planned as every build is, so only what the dashboard lacks is made).

**What is written.** Only keys a chart's type may have set by chat (`SETTABLE`), with values mnemiq reads as valid
for that key; any other is dropped and named. A chart's type changes only when the change names it, and only among the
x-y types. A change is sent with the `changed_on` mnemiq read: a chart changed meanwhile answers stale, and is read
again and changed once more; changed again, it is named and left. A chart the person does not edit is named; a chart
the dashboard does not place is never read or written.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from mnemiq.sql.schema import visible_schema
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.chart_check import (
    ChartRead,
    DatasetFacts,
    NotCheckable,
    dataset_of,
    expectation_from_form_data,
    read_chart,
    read_dataset_facts,
)
from mnemiq.superset.chart_form import _GRAINS, _row_limit
from mnemiq.superset.charts import Family, Filter, _adhoc_filter
from mnemiq.superset.dashboard_charts import _OPERATOR, datasets_built_on
from mnemiq.superset.display_format import metric_label
from mnemiq.superset.extension import call_tool
from mnemiq.superset.measure_actions import (
    Asking,
    EditorsOf,
    _editorship,
    _kept,
    _not_editors,
    _placed,
    _put,
    _recheck,
    _take_off,
)

__all__ = ["SETTABLE", "ChangeAnswer", "ChartChange", "FakeInterpreter", "Interpretation", "LLMChangeInterpreter",
           "change_dashboard", "change_prompt", "dashboard_summary"]

logger = logging.getLogger(__name__)

_XY = (Family.LINE, Family.BAR, Family.AREA, Family.SCATTER)
_XY_TYPES = frozenset(family.value for family in _XY)
_FORMATS = frozenset({"y_axis_format", "number_format", "value_format", "currency_format"})

# The `params` keys a chat change may set or unset, by the chart's type. Filters are changed through `add_filters`
# and `remove_filters`, never by sending a chart's whole filter list, so its other filters and the builder's
# leave-out condition are kept.
SETTABLE: dict[Family, frozenset[str]] = {
    **{family: frozenset({"time_grain_sqla", "row_limit", "groupby", "x_axis", "orientation", "viz_type",
                          "y_axis_format", "currency_format", "order_desc"}) for family in _XY},
    Family.TABLE: frozenset({"row_limit", "groupby", "order_desc"}),
    Family.PIE: frozenset({"row_limit", "groupby", "number_format", "currency_format"}),
    Family.BIG_NUMBER: frozenset({"y_axis_format", "currency_format"}),
    Family.GAUGE: frozenset({"row_limit", "groupby", "number_format", "currency_format"}),
    Family.TREEMAP: frozenset({"row_limit", "groupby", "number_format", "currency_format"}),
    Family.PIVOT: frozenset({"value_format", "currency_format", "groupbyRows", "groupbyColumns"}),
    Family.FUNNEL: frozenset({"row_limit", "groupby", "number_format", "currency_format"}),
    Family.SANKEY: frozenset({"row_limit", "source", "target"}),
    Family.WATERFALL: frozenset({"time_grain_sqla", "x_axis", "y_axis_format", "currency_format"}),
}
# The types a chart's filters may be changed on by chat: those whose filters are SIMPLE WHERE filters in `adhoc_filters`.
_FILTERABLE = frozenset(SETTABLE)
_MAX_ROW_LIMIT = 50000
_FORMAT_TEXT = re.compile(r"^[^{}%]{0,40}%?$|^SMART_NUMBER$")
_ISO = re.compile(r"^[A-Z]{3}$")
_MAX_REQUEST = 2000
CHANGE_MAX_TOKENS = 2000


@dataclass(frozen=True)
class ChartChange:
    chart_id: int
    set: Mapping = field(default_factory=dict)
    unset: tuple[str, ...] = ()
    add_filters: tuple[Mapping, ...] = ()
    remove_filters: tuple[str, ...] = ()


@dataclass(frozen=True)
class Interpretation:
    """What the model read the request as. `reason` says why nothing could be done, when nothing is."""

    changes: tuple[ChartChange, ...] = ()
    add_chart: bool = False
    remove_chart_ids: tuple[int, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ChangeAnswer:
    """What a request changed: charts changed, charts left (each with mnemiq's reason), keys dropped (each chart and
    key), charts the person does not edit (with their editors), charts removed, whether a chart is to be added (the
    route starts the build), and mnemiq's words."""

    changed: tuple[int, ...] = ()
    left: tuple[tuple[int, str], ...] = ()
    dropped: tuple[tuple[int, str], ...] = ()
    not_editor: tuple[EditorsOf, ...] = ()
    removed: tuple[int, ...] = ()
    add_chart: bool = False
    message: str = ""


# ---------------------------------------------------------------- what the model sees


def _family(viz_type: str) -> Family | None:
    try:
        return Family(viz_type)
    except ValueError:
        return None


def _table_columns(asking: Asking, facts) -> list[str]:
    if not isinstance(facts, DatasetFacts) or facts.is_virtual or not facts.table:
        return []
    return sorted(visible_schema(asking.snapshot, asking.grants).get(facts.table, set()))


def _chart_summary(read: ChartRead, facts) -> dict:
    form = read.form_data or {}
    summary = {"id": read.chart_id, "title": read.title, "type": (_family(read.viz_type).name.lower()
                                                                 if _family(read.viz_type) else read.viz_type)}
    if isinstance(facts, DatasetFacts):
        summary["table"] = facts.table if not facts.is_virtual else None
        expectation = expectation_from_form_data(form, dataset=facts) if form else None
        if expectation is not None and not isinstance(expectation, NotCheckable):
            summary["measures"] = [label for label, _ in expectation.metrics]
            summary["grouping"] = list(expectation.dimensions)
            summary["filters"] = [{"column": f.column, "op": f.op, "value": f.value} for f in expectation.filters]
        else:
            items = form.get("metrics") if isinstance(form.get("metrics"), list) else [form.get("metric")]
            summary["measures"] = [label for item in items if (label := metric_label(item))]
    summary["time_grain"] = form.get("time_grain_sqla") or None
    summary["row_limit"] = _row_limit(form)
    if not form:
        summary["unreadable"] = True
    return summary


def dashboard_summary(asking: Asking) -> dict:
    """What the model is shown of the dashboard: each chart it places, read now as the person (`_chart_summary`),
    and the columns of each chart's table they may read."""
    placed = _placed(asking)
    built_on = datasets_built_on(asking.store.measures_for(asking.tenant, asking.dashboard_id))
    charts, facts = [], {}
    for chart_id in sorted(placed):
        read = read_chart(asking.client, chart_id)
        dataset = dataset_of(read.form_data, built_on.get(chart_id, 0)) if read.form_data else built_on.get(chart_id)
        if type(dataset) is int and dataset > 0 and dataset not in facts:
            facts[dataset] = read_dataset_facts(asking.client, dataset)
        found = facts.get(dataset) if type(dataset) is int else None
        summary = _chart_summary(read, found)
        summary["columns"] = _table_columns(asking, found)
        charts.append(summary)
    return {"charts": charts}


_SYSTEM = """You change a Superset dashboard as a person asks, by naming the settings of its charts to change. \
Answer with one JSON object:
{"changes": [...], "add_chart": false, "remove_chart_ids": [], "reason": null}

changes: one item per chart to change, {"chart_id": its id, "set": {setting: value}, "unset": [setting], \
"add_filters": [{"column": a column of the chart's table, "op": one of = != > < >= <= LIKE ILIKE IN "NOT IN" \
"IS NULL" "IS NOT NULL", "value": the value, or a list for IN}], "remove_filters": [a column whose filters to drop]}. \
Name only what the request asks to change; never a setting it does not mention.
The settings: time_grain_sqla (one of GRAINS), row_limit (a whole number), groupby (a list of columns), x_axis \
(a column), orientation (vertical or horizontal), viz_type (echarts_timeseries_line, echarts_timeseries_bar, \
echarts_area or echarts_timeseries_scatter, only for a chart of one of those types and only when the request asks \
for another type), y_axis_format, number_format, value_format, order_desc (true or false).
add_chart: true when the request asks for a chart the dashboard does not have; it is then planned and built.
remove_chart_ids: the charts the request asks to remove.
reason: when the request cannot be done with these, why, in one sentence; otherwise null."""

_RESPONSE_FORMAT = {"response_format": {"type": "json_schema", "json_schema": {"name": "dashboard_change", "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["changes", "add_chart", "remove_chart_ids", "reason"],
    "properties": {
        "changes": {"type": "array", "items": {"type": "object", "required": ["chart_id"], "properties": {
            "chart_id": {"type": "integer"}, "set": {"type": "object"}, "unset": {"type": "array",
                                                                                "items": {"type": "string"}},
            "add_filters": {"type": "array", "items": {"type": "object"}},
            "remove_filters": {"type": "array", "items": {"type": "string"}}}}},
        "add_chart": {"type": "boolean"}, "remove_chart_ids": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": ["string", "null"]}}}}}}


def change_prompt(request: str, summary: Mapping) -> tuple[str, str]:
    """The system and user prompts the model is sent for `request`: everything it sees."""
    system = _SYSTEM.replace("GRAINS", ", ".join(_GRAINS))
    return system, f"Request: {request}\n\nThe dashboard's charts:{json.dumps(summary, ensure_ascii=False)}"


def _int(value) -> int | None:
    return value if type(value) is int and value > 0 else None


def interpretation_from(doc: object) -> Interpretation:
    """The model's reply read strictly: anything not of the shape asked for is left out."""
    if not isinstance(doc, Mapping):
        return Interpretation(reason="the model did not answer with a change")
    changes = []
    for item in doc.get("changes") or []:
        if not isinstance(item, Mapping) or _int(item.get("chart_id")) is None:
            continue
        changes.append(ChartChange(
            chart_id=item["chart_id"], set=dict(item["set"]) if isinstance(item.get("set"), Mapping) else {},
            unset=tuple(k for k in item.get("unset") or [] if isinstance(k, str)),
            add_filters=tuple(f for f in item.get("add_filters") or [] if isinstance(f, Mapping)),
            remove_filters=tuple(c for c in item.get("remove_filters") or [] if isinstance(c, str))))
    removed = tuple(c for c in doc.get("remove_chart_ids") or [] if _int(c) is not None)
    reason = doc.get("reason") if isinstance(doc.get("reason"), str) else None
    return Interpretation(changes=tuple(changes), add_chart=doc.get("add_chart") is True, remove_chart_ids=removed,
                          reason=reason[:500] if reason else None)


class LLMChangeInterpreter:
    """Reads a request with the profile's model, its reply held to `_RESPONSE_FORMAT`."""

    def __init__(self, client, max_tokens: int = CHANGE_MAX_TOKENS) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def interpret(self, request: str, summary: Mapping) -> Interpretation:
        system, user = change_prompt(request, summary)
        raw = self._client.complete(system, user, max_tokens=self._max_tokens, extra_body=_RESPONSE_FORMAT)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        try:
            doc = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            doc = None
        return interpretation_from(doc)


class FakeInterpreter:
    """Answers the interpretations it was given, in order, the last again for any later call, recording each request
    and summary."""

    def __init__(self, answer: Interpretation | Sequence[Interpretation]) -> None:
        self._answers = list(answer) if isinstance(answer, (list, tuple)) else [answer]
        self.requests: list[str] = []
        self.summaries: list[Mapping] = []

    def interpret(self, request: str, summary: Mapping) -> Interpretation:
        self.requests.append(request)
        self.summaries.append(summary)
        return self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]


# ---------------------------------------------------------------- what is written


def _template(value) -> bool:
    return any(marker in json.dumps(value) for marker in JINJA_DELIMITERS)


def _valid(key: str, value, *, family: Family, columns: set[str]) -> bool:
    """Whether `value` is one mnemiq writes for `key` on a chart of `family`."""
    if _template(value):
        return False
    if key == "time_grain_sqla":
        return value in _GRAINS
    if key == "row_limit":
        return type(value) is int and 0 < value <= _MAX_ROW_LIMIT
    if key in ("groupby", "groupbyRows", "groupbyColumns"):
        return isinstance(value, list) and all(isinstance(c, str) and c in columns for c in value)
    if key in ("x_axis", "source", "target"):
        return isinstance(value, str) and value in columns
    if key == "orientation":
        return value in ("vertical", "horizontal")
    if key == "viz_type":
        return family in _XY and value in _XY_TYPES
    if key == "order_desc":
        return isinstance(value, bool)
    if key == "currency_format":
        return isinstance(value, Mapping) and set(value) <= {"symbol", "symbolPosition"} \
            and isinstance(value.get("symbol"), str) and _ISO.match(value["symbol"]) is not None \
            and value.get("symbolPosition", "prefix") in ("prefix", "suffix")
    if key in _FORMATS:
        return isinstance(value, str) and _FORMAT_TEXT.match(value) is not None
    return False


def _filter_of(item: Mapping, columns: set[str]) -> Filter | None:
    column, op, value = item.get("column"), item.get("op"), item.get("value")
    if not isinstance(column, str) or column not in columns or op not in _OPERATOR or _template(value):
        return None
    if op in ("IN", "NOT IN"):
        if not isinstance(value, list) or not value or not all(isinstance(v, str | int | float) for v in value):
            return None
    elif op in ("IS NULL", "IS NOT NULL"):
        value = None
    elif not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    return Filter(column, op, value)


@dataclass(frozen=True)
class _Planned:
    """A chart's change as it is sent: the keys to set and unset, and what was dropped."""

    to_set: dict
    to_unset: list
    dropped: list


def _planned(change: ChartChange, read: ChartRead, columns: set[str]) -> _Planned:
    family = _family(read.viz_type)
    settable = SETTABLE.get(family, frozenset()) if family else frozenset()
    to_set, to_unset, dropped = {}, [], []
    for key, value in change.set.items():
        if key in settable and _valid(key, value, family=family, columns=columns):
            to_set[key] = value
        else:
            dropped.append(key)
    for key in change.unset:
        if key in settable and key not in ("viz_type", "x_axis"):
            to_unset.append(key)
        else:
            dropped.append(key)
    if change.add_filters or change.remove_filters:
        if family not in _FILTERABLE:
            dropped.append("adhoc_filters")
        else:
            form = read.form_data or {}
            current = form.get("adhoc_filters") if isinstance(form.get("adhoc_filters"), list) else []
            removing = set(change.remove_filters)
            kept = [f for f in current if not (isinstance(f, Mapping) and f.get("expressionType") == "SIMPLE"
                                                and f.get("clause", "WHERE") == "WHERE"
                                                and f.get("subject") in removing)]
            added = [_filter_of(item, columns) for item in change.add_filters]
            if any(item is None for item in added):
                dropped.append("adhoc_filters")
            else:
                to_set["adhoc_filters"] = kept + [_adhoc_filter(item) for item in added]
    return _Planned(to_set, to_unset, sorted(set(dropped)))


def _send(asking: Asking, chart_id: int, read: ChartRead, planned: _Planned) -> dict:
    return call_tool(asking.client, "patch_chart", {"chart_id": chart_id, "set": planned.to_set,
                                                     "unset": planned.to_unset,
                                                     "expect_changed_on": read.changed_on})


def change_dashboard(asking: Asking, request: str, *, interpreter) -> ChangeAnswer:
    """Make `request` on the dashboard, as the person asking: read the dashboard's charts, have `interpreter` read the
    request against them, and write only what it names, as `SETTABLE` and `_valid` allow."""
    request = " ".join((request or "").split())[:_MAX_REQUEST]
    summary = dashboard_summary(asking)
    placed = {chart["id"] for chart in summary["charts"]}
    columns_of = {chart["id"]: set(chart.get("columns") or []) for chart in summary["charts"]}
    read = interpreter.interpret(request, summary)
    left: list[tuple[int, str]] = []
    dropped: list[tuple[int, str]] = []
    wanted: dict[int, tuple[ChartChange, ChartRead, _Planned]] = {}
    for change in read.changes:
        if change.chart_id not in placed:
            left.append((change.chart_id, "it is not a chart of this dashboard"))
            continue
        current = read_chart(asking.client, change.chart_id)
        if current.form_data is None:
            left.append((change.chart_id, current.unreadable or "its settings could not be read"))
            continue
        planned = _planned(change, current, columns_of.get(change.chart_id, set()))
        dropped += [(change.chart_id, key) for key in planned.dropped]
        if planned.to_set or planned.to_unset:
            wanted[change.chart_id] = (change, current, planned)
    removing = tuple(chart_id for chart_id in read.remove_chart_ids if chart_id in placed)
    left += [(chart_id, "it is not a chart of this dashboard") for chart_id in read.remove_chart_ids
             if chart_id not in placed]
    editorship = _editorship(asking, chart_ids=[*wanted, *removing])
    not_editor = list(_not_editors(editorship, "chart", wanted))
    for entry in not_editor:
        wanted.pop(entry.chart_id, None)
    changed = []
    for chart_id, (change, current, planned) in wanted.items():
        for attempt in range(2):
            answer = _send(asking, chart_id, current, planned)
            if answer.get("ok") is True:
                changed.append(chart_id)
                break
            if answer.get("not_editor"):
                not_editor.append(EditorsOf(editors=tuple(str(e) for e in answer.get("editors") or []),
                                            chart_id=chart_id))
                break
            if not answer.get("stale") or attempt:
                left.append((chart_id, "it kept changing while it was changed" if answer.get("stale")
                             else "Superset did not take the change"))
                break
            current = read_chart(asking.client, chart_id)
            if current.form_data is None:
                left.append((chart_id, current.unreadable or "its settings could not be read"))
                break
            planned = _planned(change, current, columns_of.get(chart_id, set()))
    removed = _removed(asking, removing, editorship, not_editor)
    if changed:
        _recheck_kept(asking, changed)
    answer = ChangeAnswer(changed=tuple(changed), left=tuple(left), dropped=tuple(sorted(dropped)),
                          not_editor=tuple(not_editor), removed=removed, add_chart=read.add_chart)
    return dataclasses.replace(answer, message=_said(answer, read.reason))


def _removed(asking: Asking, removing: tuple[int, ...], editorship: dict, not_editor: list) -> tuple[int, ...]:
    """Each chart to remove deleted (moved to Superset's trash) when the person edits it, else named; both off the
    layout, and out of the dashboard's kept measures."""
    if not removing:
        return ()
    deleted, off = [], []
    for chart_id in removing:
        is_editor, editors = editorship.get(("chart", chart_id), (False, ()))
        if not is_editor:
            not_editor.append(EditorsOf(editors=editors, chart_id=chart_id))
            continue
        answer = asking.client.call("delete_chart", {"identifier": chart_id})
        (deleted if answer.get("success") is True else off).append(chart_id)
    gone = set(deleted)
    if gone:
        _take_off(asking, gone)
        for row in _kept(asking).values():
            if gone & set(row.chart_ids):
                left = tuple(c for c in row.chart_ids if c not in gone)
                _put(asking, row, chart_ids=left, check_status={c: s for c, s in row.check_status.items()
                                                                if c in left})
    return tuple(deleted)


def _recheck_kept(asking: Asking, changed: list[int]) -> None:
    """Each changed chart that shows a measure the dashboard keeps is checked again, and its status kept."""
    for row in list(_kept(asking).values()):
        mine = [chart_id for chart_id in changed if chart_id in row.chart_ids]
        if mine:
            statuses = _recheck(asking, mine, row.measure.dataset_id)
            _put(asking, row, check_status={**row.check_status, **statuses})


def _said(answer: ChangeAnswer, reason: str | None) -> str:
    parts = []
    if answer.changed:
        parts.append("changed: " + ", ".join(f"chart {c}" for c in answer.changed))
    if answer.removed:
        parts.append("moved to Superset's trash: " + ", ".join(f"chart {c}" for c in answer.removed))
    if answer.add_chart:
        parts.append("the chart asked for is planned and built onto the dashboard")
    if answer.dropped:
        parts.append("not changed by chat: " + ", ".join(f"{key} on chart {c}" for c, key in answer.dropped))
    if answer.left:
        parts.append("left as they are: " + "; ".join(f"chart {c} ({why})" for c, why in answer.left))
    if answer.not_editor:
        parts.append("not edited by this person: " + "; ".join(
            f"chart {e.chart_id}" + (f" (its editors are {', '.join(e.editors)})" if e.editors else "")
            for e in answer.not_editor))
    if not parts:
        return reason or "Nothing on the dashboard was changed."
    return "; ".join(parts).capitalize() + "."

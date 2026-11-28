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
dashboard as a build (planned as every build is, so only what the dashboard lacks is made). A chart to remove is only
proposed (`would_remove`): nothing is deleted on the model's reading, only on the person's confirmation.

**What is written.** Only keys a chart's type may have set by chat (`SETTABLE`), with values mnemiq reads as valid
for that key; any other is dropped and named. A chart's type changes only when the change names it, and only among the
x-y types. A change is sent with the `changed_on` mnemiq read: a chart changed meanwhile answers stale, and is read
again and changed once more; changed again, it is named and left. A chart the person does not edit is named; a chart
the dashboard does not place is never read or written.
"""

from __future__ import annotations

import dataclasses
import hashlib
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
from mnemiq.superset.names import stored_name
from mnemiq.superset.planner import SCALAR
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

__all__ = ["NOT_ANSWERED", "PREVIEW_JSON_SHAPE", "SETTABLE", "STALE_SINCE_PREVIEW", "ChangeAnswer", "ChangeInterrupted",
           "ChangePreview", "ChartChange",
           "FakeInterpreter", "Interpretation", "LLMChangeInterpreter", "PlannedEdit", "apply_change",
           "change_dashboard", "change_prompt", "dashboard_summary", "preview_change"]

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
    # The charts the request reads as removing, by id and title: a proposal the person confirms, never acted on.
    would_remove: tuple[tuple[int, str | None], ...] = ()
    # A confirmed removal deleted at least one chart for good (a Superset keeping no trash).
    permanent: bool = False


# ---------------------------------------------------------------- what the model sees


def _family(viz_type: str) -> Family | None:
    try:
        return Family(viz_type)
    except ValueError:
        return None


def _table_columns(asking: Asking, facts) -> list[str]:
    """The columns of a physical dataset's table the person may read, as the snapshot names them: the table Superset
    reads for the dataset's name (`stored_name`)."""
    if not isinstance(facts, DatasetFacts) or facts.is_virtual or not facts.table:
        return []
    return sorted(visible_schema(asking.snapshot, asking.grants).get(stored_name(facts.table), set()))


def _chart_summary(read: ChartRead, facts) -> dict:
    form = read.form_data or {}
    summary = {"id": read.chart_id, "title": read.title, "type": (_family(read.viz_type).name.lower()
                                                                 if _family(read.viz_type) else read.viz_type)}
    if isinstance(facts, DatasetFacts):
        # In the snapshot's spelling, as the columns shown beside it are (`_table_columns`).
        summary["table"] = stored_name(facts.table) if not facts.is_virtual and facts.table is not None else None
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
        found = _facts_of(asking, read, built_on, facts)
        summary = _chart_summary(read, found)
        summary["columns"] = _table_columns(asking, found)
        charts.append(summary)
    return {"charts": charts}


def _facts_of(asking: Asking, read: ChartRead, built_on: Mapping[int, int], facts: dict):
    """The facts of the dataset `read` shows, read once per dataset into `facts`; None when it has none mnemiq reads."""
    chart_id = read.chart_id
    dataset = dataset_of(read.form_data, built_on.get(chart_id, 0)) if read.form_data else built_on.get(chart_id)
    if type(dataset) is int and dataset > 0 and dataset not in facts:
        facts[dataset] = read_dataset_facts(asking.client, dataset)
    return facts.get(dataset) if type(dataset) is int else None


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
                def same(a: Mapping, b: Mapping) -> bool:
                    return all(a.get(k) == b.get(k) for k in ("expressionType", "clause", "subject", "operator",
                                                              "comparator"))

                # A change sent again (its answer lost) adds no second copy of a filter the chart already holds.
                new = [f for f in (_adhoc_filter(item) for item in added)
                       if not any(isinstance(old, Mapping) and same(old, f) for old in kept)]
                if new or removing:
                    to_set["adhoc_filters"] = kept + new
    return _Planned(to_set, to_unset, sorted(set(dropped)))


def _send(asking: Asking, chart_id: int, changed_on: str | None, planned: _Planned) -> dict:
    return call_tool(asking.client, "patch_chart", {"chart_id": chart_id, "set": planned.to_set,
                                                     "unset": planned.to_unset,
                                                     "expect_changed_on": changed_on})


# A previewed change no longer matches the dashboard or a chart it names: what a confirm answers for each edit it did
# not write.
STALE_SINCE_PREVIEW = "it changed since the preview; ask again"
# A chart a strict apply was writing, or about to remove, when Superset stopped answering: it may or may not be done.
NOT_ANSWERED = "Superset stopped answering before this was done; check the dashboard"


class ChangeInterrupted(Exception):
    """A strict apply that failed after its first write was sent: `answer` names what was written and what may or may
    not have been. The failure itself is the cause."""

    def __init__(self, answer: ChangeAnswer) -> None:
        super().__init__("a change stopped partway")
        self.answer = answer


@dataclass(frozen=True)
class PlannedEdit:
    """One chart's change as a preview names it: the keys it will set and unset, against the chart's `changed_on` as
    the preview read it, and mnemiq's words for it (`said`). `recorded` is what the model read, held to the keys and
    filters that made the plan, and `digest` names the plan exactly: a confirm plans the chart again from `recorded`,
    as it reads now, and writes only when that plan has this digest. `change` (the model's reading as it came) and
    `columns` (the person's columns of its table) are held in memory only, for today's retry of a stale write."""

    chart_id: int
    title: str | None
    changed_on: str | None
    to_set: Mapping
    to_unset: tuple[str, ...]
    said: str
    digest: str
    recorded: ChartChange
    change: ChartChange | None = None
    columns: frozenset[str] | None = None


@dataclass(frozen=True)
class ChangePreview:
    """What a request will do to a dashboard, read and planned with nothing written: its edits, the charts it reads
    as removing (`would_remove`), whether it asks for a chart the dashboard lacks, and what it leaves, drops, or may not
    change, with mnemiq's words. `dashboard_changed_on` is the dashboard's, read before anything else."""

    dashboard_id: int
    dashboard_changed_on: str | None
    edits: tuple[PlannedEdit, ...] = ()
    would_remove: tuple[tuple[int, str | None], ...] = ()
    add_chart: bool = False
    # Each chart it would remove, with the `changed_on` the preview read it at: a confirmed removal is held to it.
    would_remove_changed_on: tuple[tuple[int, str], ...] = ()
    left: tuple[tuple[int, str], ...] = ()
    dropped: tuple[tuple[int, str], ...] = ()
    not_editor: tuple[EditorsOf, ...] = ()
    message: str = ""
    reason: str | None = None

    def to_json(self) -> dict:
        """The preview as a conversation keeps it (`PREVIEW_JSON_SHAPE`): each edit's `recorded` reading, a filter's
        literal under `operand`, never the model's own `value` key; no edit's keys to send, which its digest names."""
        return {"dashboard_id": self.dashboard_id, "dashboard_changed_on": self.dashboard_changed_on,
                "edits": [{"chart_id": edit.chart_id, "title": edit.title, "changed_on": edit.changed_on,
                           "said": edit.said, "digest": edit.digest, "change": _change_json(edit.recorded)}
                          for edit in self.edits],
                "would_remove": [{"chart_id": chart_id, "title": title,
                                  "changed_on": dict(self.would_remove_changed_on).get(chart_id)}
                                 for chart_id, title in self.would_remove],
                "add_chart": self.add_chart,
                "left": [{"chart_id": chart_id, "reason": reason} for chart_id, reason in self.left],
                "dropped": [{"chart_id": chart_id, "key": key} for chart_id, key in self.dropped],
                "not_editor": [{"chart_id": e.chart_id, "editors": list(e.editors)} for e in self.not_editor],
                "message": self.message, "reason": self.reason}

    @classmethod
    def from_json(cls, doc: Mapping) -> ChangePreview:
        """`to_json`'s record read back: each edit with its `recorded` reading and digest, nothing to send until a
        strict apply plans it again."""
        return cls(
            dashboard_id=doc["dashboard_id"], dashboard_changed_on=doc.get("dashboard_changed_on"),
            edits=tuple(PlannedEdit(chart_id=edit["chart_id"], title=edit.get("title"),
                                    changed_on=edit.get("changed_on"), to_set={}, to_unset=(), said=edit["said"],
                                    digest=edit["digest"], recorded=_change_of(edit["chart_id"], edit["change"]))
                        for edit in doc.get("edits") or []),
            would_remove=tuple((item["chart_id"], item.get("title")) for item in doc.get("would_remove") or []),
            would_remove_changed_on=tuple((item["chart_id"], item["changed_on"]) for item in doc.get("would_remove") or []
                                          if isinstance(item.get("changed_on"), str)),
            add_chart=doc.get("add_chart") is True,
            left=tuple((item["chart_id"], item["reason"]) for item in doc.get("left") or []),
            dropped=tuple((item["chart_id"], item["key"]) for item in doc.get("dropped") or []),
            not_editor=tuple(EditorsOf(editors=tuple(item.get("editors") or ()), chart_id=item["chart_id"])
                             for item in doc.get("not_editor") or []),
            message=doc.get("message") or "", reason=doc.get("reason"))


_SETTING_JSON = (SCALAR, [SCALAR], {"symbol": SCALAR, "symbolPosition": SCALAR})
_CHANGE_JSON = {"set": {key: _SETTING_JSON for key in sorted(set().union(*SETTABLE.values()))},
                "unset": [SCALAR],
                "add_filters": [{"column": SCALAR, "op": SCALAR, "operand": (SCALAR, [SCALAR])}],
                "remove_filters": [SCALAR]}
# What `ChangePreview.to_json` writes, in `planner.PLAN_JSON_SHAPE`'s form: the conversation store's allow-list of a
# kept preview. This module's tests hold the writer to it.
PREVIEW_JSON_SHAPE: dict = {
    "dashboard_id": SCALAR, "dashboard_changed_on": SCALAR,
    "edits": [{"chart_id": SCALAR, "title": SCALAR, "changed_on": SCALAR, "said": SCALAR, "digest": SCALAR,
               "change": _CHANGE_JSON}],
    "would_remove": [{"chart_id": SCALAR, "title": SCALAR, "changed_on": SCALAR}], "add_chart": SCALAR,
    "left": [{"chart_id": SCALAR, "reason": SCALAR}], "dropped": [{"chart_id": SCALAR, "key": SCALAR}],
    "not_editor": [{"chart_id": SCALAR, "editors": [SCALAR]}], "message": SCALAR, "reason": SCALAR,
}


def _change_json(change: ChartChange) -> dict:
    return {"set": dict(change.set), "unset": list(change.unset),
            "add_filters": [{"column": item.get("column"), "op": item.get("op"), "operand": item.get("value")}
                            for item in change.add_filters],
            "remove_filters": list(change.remove_filters)}


def _change_of(chart_id: int, doc: Mapping) -> ChartChange:
    return ChartChange(chart_id=chart_id, set=dict(doc.get("set") or {}), unset=tuple(doc.get("unset") or ()),
                       add_filters=tuple({"column": item.get("column"), "op": item.get("op"),
                                          "value": item.get("operand")} for item in doc.get("add_filters") or []),
                       remove_filters=tuple(doc.get("remove_filters") or ()))


def _digest(planned: _Planned) -> str:
    """What a chart's write sends, named: equal for two plans exactly when they set and unset the same."""
    return hashlib.sha256(json.dumps({"set": planned.to_set, "unset": list(planned.to_unset)}, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def _recorded(change: ChartChange, planned: _Planned) -> ChartChange:
    """`change` held to what made `planned`: the keys it sets and unsets, and its filters only when the plan writes
    them. Planned again against the same chart, it plans the same."""
    filters = "adhoc_filters" in planned.to_set
    # A chart's filters are changed through `add_filters` and `remove_filters` only: an `adhoc_filters` the model put
    # in `set` is never settable, and its plan's `adhoc_filters` comes from the filters it adds and drops.
    return ChartChange(chart_id=change.chart_id,
                       set={key: value for key, value in change.set.items()
                            if key in planned.to_set and key != "adhoc_filters"},
                       unset=tuple(key for key in change.unset if key in planned.to_unset),
                       add_filters=tuple({"column": item.get("column"), "op": item.get("op"),
                                          "value": item.get("value")} for item in change.add_filters)
                       if filters else (),
                       remove_filters=tuple(change.remove_filters) if filters else ())


def _family_word(viz_type) -> str:
    family = _family(viz_type) if isinstance(viz_type, str) else None
    return family.name.lower().replace("_", " ") if family else str(viz_type)


def _edit_said(read: ChartRead, change: ChartChange, planned: _Planned) -> str:
    """mnemiq's words for one chart's planned change: its title, then each thing that changes."""
    parts = []
    for key, value in planned.to_set.items():
        if key == "viz_type":
            parts.append(f"{_family_word(read.viz_type)} → {_family_word(value)}")
        elif key == "time_grain_sqla":
            parts.append(f"time grain {value}")
        elif key == "row_limit":
            parts.append(f"at most {value} rows")
        elif key in ("groupby", "groupbyRows", "groupbyColumns"):
            parts.append(f"{'grouped' if key == 'groupby' else key} by {', '.join(map(str, value)) or 'nothing'}")
        elif key == "order_desc":
            parts.append("largest first" if value else "smallest first")
        elif key == "adhoc_filters":
            for item in change.add_filters:
                operand = "" if item.get("value") is None else f" {item.get('value')}"
                parts.append(f"filter {item.get('column')} {item.get('op')}{operand}")
            for column in change.remove_filters:
                parts.append(f"no filter on {column}")
        else:
            parts.append(f"{key} {value}")
    parts += [f"{key} cleared" for key in planned.to_unset]
    return f"{read.title or f'chart {read.chart_id}'}: {'; '.join(parts)}"


def _dashboard_changed_on(asking: Asking) -> str | None:
    answer = call_tool(asking.client, "get_layout", {"dashboard_id": asking.dashboard_id})
    if answer.get("dashboard_id") != asking.dashboard_id or not isinstance(answer.get("changed_on"), str):
        return None
    return answer["changed_on"]


def preview_change(asking: Asking, request: str, *, interpreter) -> ChangePreview:
    """What `request` would do on the dashboard, as the person asking, with nothing written: the dashboard's
    `changed_on` read first, then its charts, `interpreter`'s reading of the request against them, and each chart's
    change planned as `SETTABLE` and `_valid` allow. A chart the request reads as removing is only proposed."""
    request = " ".join((request or "").split())[:_MAX_REQUEST]
    stamped = _dashboard_changed_on(asking)
    summary = dashboard_summary(asking)
    placed = {chart["id"] for chart in summary["charts"]}
    columns_of = {chart["id"]: set(chart.get("columns") or []) for chart in summary["charts"]}
    read = interpreter.interpret(request, summary)
    left: list[tuple[int, str]] = []
    dropped: list[tuple[int, str]] = []
    wanted: dict[int, PlannedEdit] = {}
    for change in read.changes:
        if change.chart_id not in placed:
            left.append((change.chart_id, "it is not a chart of this dashboard"))
            continue
        current = read_chart(asking.client, change.chart_id)
        if current.form_data is None:
            left.append((change.chart_id, current.unreadable or "its settings could not be read"))
            continue
        columns = columns_of.get(change.chart_id, set())
        planned = _planned(change, current, columns)
        dropped += [(change.chart_id, key) for key in planned.dropped]
        if planned.to_set or planned.to_unset:
            wanted[change.chart_id] = PlannedEdit(
                chart_id=change.chart_id, title=current.title, changed_on=current.changed_on,
                to_set=planned.to_set, to_unset=tuple(planned.to_unset), said=_edit_said(current, change, planned),
                digest=_digest(planned), recorded=_recorded(change, planned), change=change,
                columns=frozenset(columns))
    titles = {chart["id"]: chart.get("title") for chart in summary["charts"]}
    would_remove = tuple((chart_id, titles.get(chart_id)) for chart_id in dict.fromkeys(read.remove_chart_ids)
                         if chart_id in placed)
    left += [(chart_id, "it is not a chart of this dashboard") for chart_id in read.remove_chart_ids
             if chart_id not in placed]
    editorship = _editorship(asking, chart_ids=wanted)
    not_editor = tuple(_not_editors(editorship, "chart", wanted))
    for entry in not_editor:
        wanted.pop(entry.chart_id, None)
    removing_read = tuple((chart_id, read_chart(asking.client, chart_id).changed_on) for chart_id, _ in would_remove)
    preview = ChangePreview(dashboard_id=asking.dashboard_id, dashboard_changed_on=stamped,
                            edits=tuple(wanted.values()), would_remove=would_remove, add_chart=read.add_chart,
                            would_remove_changed_on=tuple((c, at) for c, at in removing_read if isinstance(at, str)),
                            left=tuple(left), dropped=tuple(sorted(dropped)), not_editor=not_editor,
                            reason=read.reason)
    return dataclasses.replace(preview, message=_previewed(preview))


def apply_change(asking: Asking, preview: ChangePreview, *, strict: bool, remove: Sequence[int] = ()) -> ChangeAnswer:
    """Write `preview`'s edits, as the person asking, with no model call, then check again each changed chart that
    shows a measure the dashboard keeps. `strict` is a confirm's: the person approved the preview whole, so nothing is
    written unless the dashboard, every edited chart and every chart in `remove` are as the preview read them and each
    edit plans again to its digest; a write Superset still answers stale stops the apply there, and `remove`, charts
    the preview would remove that the person confirmed, are removed only once every edit is written. A strict apply
    that fails after its first write raises `ChangeInterrupted`, naming what was written. Not strict is the change
    asked by the changes route, its preview made just before: a stale write is read again, planned again and retried
    once, and it removes nothing."""
    if preview.dashboard_id != asking.dashboard_id:
        raise ValueError("a preview is applied to the dashboard it was made on")
    remove = tuple(dict.fromkeys(remove))
    if remove and not strict:
        raise ValueError("a removal is confirmed by a strict apply of the preview that proposed it")
    if not set(remove) <= {chart_id for chart_id, _ in preview.would_remove}:
        raise ValueError("only a chart the preview would remove can be confirmed for removal")
    if not strict:
        changed, left, not_editor = _writes(asking, preview)
        removed = ChangeAnswer()
    else:
        changed, left, not_editor, removed = _strict_writes(asking, preview, remove)
    if changed:
        _recheck_kept(asking, changed)
    answer = ChangeAnswer(changed=tuple(changed), left=preview.left + tuple(left) + removed.left,
                          dropped=preview.dropped, not_editor=preview.not_editor + tuple(not_editor) + removed.not_editor,
                          removed=removed.removed, permanent=removed.permanent, add_chart=preview.add_chart,
                          # A confirm settles what the preview proposed: nothing is left to confirm.
                          would_remove=() if strict else preview.would_remove)
    return dataclasses.replace(answer, message=_said(answer, preview.reason))


def _writes(asking: Asking, preview: ChangePreview) -> tuple[list, list, list]:
    changed, left, not_editor = [], [], []
    for edit in preview.edits:
        if edit.change is None or edit.columns is None:
            raise ValueError("a change applied on its own preview's reading needs the preview that read it")
        changed_on, planned = edit.changed_on, _Planned(dict(edit.to_set), list(edit.to_unset), [])
        for attempt in range(2):
            answer = _send(asking, edit.chart_id, changed_on, planned)
            if answer.get("ok") is True:
                changed.append(edit.chart_id)
                break
            if answer.get("not_editor"):
                not_editor.append(EditorsOf(editors=tuple(str(e) for e in answer.get("editors") or []),
                                            chart_id=edit.chart_id))
                break
            if not answer.get("stale") or attempt:
                left.append((edit.chart_id, "it kept changing while it was changed" if answer.get("stale")
                             else "Superset did not take the change"))
                break
            current = read_chart(asking.client, edit.chart_id)
            if current.form_data is None:
                left.append((edit.chart_id, current.unreadable or "its settings could not be read"))
                break
            changed_on, planned = current.changed_on, _planned(edit.change, current, set(edit.columns))
    return changed, left, not_editor


def _strict_writes(asking: Asking, preview: ChangePreview,
                   remove: tuple[int, ...]) -> tuple[list, list, list, ChangeAnswer]:
    stale = [(edit.chart_id, STALE_SINCE_PREVIEW) for edit in preview.edits]
    stale += [(chart_id, STALE_SINCE_PREVIEW) for chart_id in remove]
    nothing = ChangeAnswer()
    if not preview.edits and not remove:
        return [], [], [], nothing
    if preview.dashboard_changed_on is None or _dashboard_changed_on(asking) != preview.dashboard_changed_on:
        return [], stale, [], nothing
    built_on = datasets_built_on(asking.store.measures_for(asking.tenant, asking.dashboard_id))
    facts: dict = {}
    sends = []
    for edit in preview.edits:
        current = read_chart(asking.client, edit.chart_id)
        if current.form_data is None or current.changed_on is None or current.changed_on != edit.changed_on:
            return [], stale, [], nothing
        columns = set(_table_columns(asking, _facts_of(asking, current, built_on, facts)))
        planned = _planned(edit.recorded, current, columns)
        if _digest(planned) != edit.digest:
            return [], stale, [], nothing
        sends.append((edit, planned))
    read_at = dict(preview.would_remove_changed_on)
    for chart_id in remove:
        current = read_chart(asking.client, chart_id)
        if read_at.get(chart_id) is None or current.changed_on != read_at[chart_id]:
            return [], stale, [], nothing
    # Read again just before the first write: a layout changed while the charts were read stops the apply too. What
    # is left open is the writes themselves, each of which `patch_chart` holds to its own chart's `changed_on`.
    if _dashboard_changed_on(asking) != preview.dashboard_changed_on:
        return [], stale, [], nothing
    changed, left, not_editor = [], [], []
    for index, (edit, planned) in enumerate(sends):
        try:
            answer = _send(asking, edit.chart_id, edit.changed_on, planned)
        except Exception as exc:
            unknown = [(later.chart_id, NOT_ANSWERED) for later, _ in sends[index:]]
            raise ChangeInterrupted(ChangeAnswer(changed=tuple(changed), left=tuple(left + unknown), not_editor=tuple(
                not_editor))) from exc
        if answer.get("ok") is True:
            changed.append(edit.chart_id)
        elif answer.get("not_editor"):
            not_editor.append(EditorsOf(editors=tuple(str(e) for e in answer.get("editors") or []),
                                        chart_id=edit.chart_id))
        elif answer.get("stale"):
            # Superset writes one chart at a time: what was written stays, and nothing more is.
            left += [(later.chart_id, STALE_SINCE_PREVIEW) for later, _ in sends[index:]]
            left += [(chart_id, STALE_SINCE_PREVIEW) for chart_id in remove]
            return changed, left, not_editor, nothing
        else:
            left.append((edit.chart_id, "Superset did not take the change"))
    if not remove:
        return changed, left, not_editor, nothing
    try:
        removed = _confirmed(asking, remove)
    except Exception as exc:
        raise ChangeInterrupted(ChangeAnswer(changed=tuple(changed), left=tuple(
            left + [(chart_id, NOT_ANSWERED) for chart_id in remove]), not_editor=tuple(not_editor))) from exc
    return changed, left, not_editor, removed


def change_dashboard(asking: Asking, request: str, *, interpreter,
                     confirmed_removal: Sequence[int] = ()) -> ChangeAnswer:
    """Make `request` on the dashboard, as the person asking: read the dashboard's charts, have `interpreter` read the
    request against them, and write only what it names, as `SETTABLE` and `_valid` allow (`preview_change`, then
    `apply_change` not strict). A chart the request reads as removing is only proposed (`would_remove`);
    `confirmed_removal` names the charts the person confirmed, which are removed and nothing else is done (the model
    is not asked)."""
    if confirmed_removal:
        return _confirmed(asking, tuple(dict.fromkeys(confirmed_removal)))
    return apply_change(asking, preview_change(asking, request, interpreter=interpreter), strict=False)


def _confirmed(asking: Asking, removing: tuple[int, ...]) -> ChangeAnswer:
    """The charts the person confirmed removing: each one the dashboard places and they edit deleted (moved to
    Superset's trash, or deleted for good where Superset keeps none), then taken off the layout and out of the
    dashboard's kept measures. One they do not edit is named with its editors; one Superset will not delete (a
    report holds it) is named with the reason and left; one Superset no longer holds is taken off as gone."""
    placed = _placed(asking)
    left = [(chart_id, "it is not a chart of this dashboard") for chart_id in removing if chart_id not in placed]
    removing = tuple(chart_id for chart_id in removing if chart_id in placed)
    editorship = _editorship(asking, chart_ids=removing)
    not_editor, deleted, gone, permanent = [], [], [], False
    for chart_id in removing:
        is_editor, editors = editorship.get(("chart", chart_id), (None, ()))
        if is_editor is False:
            not_editor.append(EditorsOf(editors=editors, chart_id=chart_id))
            continue
        answer = asking.client.call("delete_chart", {"identifier": chart_id})
        if answer.get("success") is True:
            deleted.append(chart_id)
            permanent = permanent or answer.get("soft_deleted") is False
        elif answer.get("error_type") == "NotFound":
            gone.append(chart_id)
        elif answer.get("error_type") == "ReportsExist":
            left.append((chart_id, "a report holds it"))
        elif answer.get("permission_denied"):
            not_editor.append(EditorsOf(editors=editors, chart_id=chart_id))
        else:
            logger.warning("delete_chart answered a confirmed removal with %s", answer.get("error_type"))
            left.append((chart_id, "Superset did not delete it"))
    taken = set(deleted) | set(gone)
    if taken:
        try:
            _take_off(asking, taken)
        except Exception as exc:  # what was deleted is answered whatever the layout's write did
            left.append((min(taken), f"it was deleted, but the layout was not written ({exc})"))
        for row in _kept(asking).values():
            if taken & set(row.chart_ids):
                kept = tuple(c for c in row.chart_ids if c not in taken)
                _put(asking, row, chart_ids=kept, check_status={c: s for c, s in row.check_status.items()
                                                                if c in kept})
    answer = ChangeAnswer(removed=tuple(deleted + gone), left=tuple(left), not_editor=tuple(not_editor),
                          permanent=permanent)
    return dataclasses.replace(answer, message=_said(answer, None))


def _recheck_kept(asking: Asking, changed: list[int]) -> None:
    """Each changed chart that shows a measure the dashboard keeps is checked again, and its status kept."""
    for row in list(_kept(asking).values()):
        mine = [chart_id for chart_id in changed if chart_id in row.chart_ids]
        if mine:
            statuses = _recheck(asking, mine, row.measure.dataset_id)
            _put(asking, row, check_status={**row.check_status, **statuses})


def _previewed(preview: ChangePreview) -> str:
    """mnemiq's words for a preview: each edit it would make, then what else the request reads as, leaves or drops."""
    nothing = "Nothing on the dashboard was changed."
    rest = _said(ChangeAnswer(left=preview.left, dropped=preview.dropped, not_editor=preview.not_editor,
                              add_chart=preview.add_chart, would_remove=preview.would_remove), None)
    if preview.edits:
        head = "To change: " + "; ".join(edit.said for edit in preview.edits) + "."
        return head if rest == nothing else f"{head} {rest}"
    return preview.reason or "Nothing on the dashboard would change." if rest == nothing else rest


def _said(answer: ChangeAnswer, reason: str | None) -> str:
    parts = []
    if answer.changed:
        parts.append("changed: " + ", ".join(f"chart {c}" for c in answer.changed))
    if answer.removed:
        where = "deleted for good (this Superset keeps no trash)" if answer.permanent else "moved to Superset's trash"
        parts.append(f"{where}: " + ", ".join(f"chart {c}" for c in answer.removed))
    if answer.would_remove:
        parts.append("to remove, once confirmed: " + ", ".join(
            f"chart {c}" + (f" ({title!r})" if title else "") for c, title in answer.would_remove))
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

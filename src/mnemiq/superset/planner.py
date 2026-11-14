"""The plan step of a dashboard build: from the builder's goal to a plan whose every part mnemiq has checked.

A plan is the measures a dashboard shows, the charts that show them, the datasets mnemiq makes for them, and, where a
dashboard the builder can see already covers the goal, the offer to reuse it. Nothing is built here: the plan is handed
to the build, which runs it one step per request.

**What the model sees** (`PlanContext`) is what the builder may: the tables and columns the grants leave visible
(`visible_schema`), the saved metrics of those tables' datasets by name and expression, and the dashboards whose titles
share a word with the goal, by id and title. Never an editor's name (a saved metric's certifier, a dashboard's owners),
never Superset's error text, never a value: nothing a plan is made from is a number computed from the data. A part
refused goes back to the model with mnemiq's own reason, which names what the model wrote and nothing Superset said.

**What is checked.** Each written measure meets `check_measure`, its statement decided under the builder's grants as
every ask-path statement is, and then `reusable`: a saved metric on its dataset that says the same thing replaces it.
Each chart meets `plan_refusal` and the plan's own rules here (its key, its columns, its filters, the saved metrics it
names). A refused part goes back to the model once; refused again, it refuses the plan, and the reason names it.

**Datasets.** A chart on a table with no dataset gets a planned `create_dataset`; a measure that needs a join gets a
planned `create_virtual_dataset`, the join decided with the measure by `check_measure`. A planned dataset has no id
until it is made, so the plan's charts and measures carry its negative `placeholder` until `with_made_datasets` puts
the made id in its place. Every dataset Superset makes carries a saved metric `count` (`COUNT_METRIC`), which is
Superset's, never one of mnemiq's measures.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.sql.identifiers import resolve_identifier
from mnemiq.sql.schema import column_types, visible_schema
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.charts import ChartSpec, Family, Filter, SavedRef, WrittenRef, plan_refusal
from mnemiq.superset.grants import _candidate_names, _read_datasets, _tables, _Unreadable
from mnemiq.superset.marker import format_marker
from mnemiq.superset.measures import Measure, MeasureKind, MeasureRefused, check_measure
from mnemiq.superset.saved_metrics import SavedMetric, read_saved_metrics, reusable
from mnemiq.superset.settings import SUPERSET_DIALECT

__all__ = ["COUNT_METRIC", "FILTER_OPERATORS", "GRAINS", "PLANNER_MAX_TOKENS", "BuildPlan", "FakePlanner",
           "LLMPlanner", "PlanColumn", "PlanContext", "PlanDashboard", "PlanMetric", "PlanRefused", "PlanTable",
           "PlannedDataset", "Planner", "RefusedPart", "Repair", "choose_dashboard", "draft_of", "plan_build",
           "plan_from_draft", "plan_from_json", "plan_prompt", "plan_to_json", "with_made_datasets"]

logger = logging.getLogger(__name__)

# The saved metric every dataset Superset makes carries: `CreateDatasetCommand.run` calls `fetch_metadata`, which adds
# the engine spec's metrics (`BaseEngineSpec.get_metrics` in `superset/db_engine_specs/base.py`), for a physical and a
# virtual dataset alike.
COUNT_METRIC = ("count", "COUNT(*)")

# rc2's `FilterConfig.op` (`superset/mcp_service/chart/schemas.py`): the operators a chart's filter may use.
FILTER_OPERATORS = ("=", "!=", ">", "<", ">=", "<=", "LIKE", "ILIKE", "NOT LIKE", "IN", "NOT IN", "IS NULL",
                    "IS NOT NULL")
_NO_OPERAND = frozenset({"IS NULL", "IS NOT NULL"})
_LIST_OPERAND = frozenset({"IN", "NOT IN"})
# `FilterConfig.sanitize_value`'s `max_length`.
_MAX_OPERAND = 1000

# The time grains a chart may group by, by the word the model writes: the ones the value check reproduces as one
# Postgres `DATE_TRUNC` (`_GRAINS` in `mnemiq.superset.chart_form`), so a chart on any of them can be checked.
GRAINS = {"second": "PT1S", "minute": "PT1M", "hour": "PT1H", "day": "P1D", "week": "P1W", "month": "P1M",
          "quarter": "P3M", "year": "P1Y"}
_GRAIN_WORD = {code: word for word, code in GRAINS.items()}

# The keys a build's own steps take besides its datasets' and charts' (`STEP_KINDS` in `mnemiq.superset.build_store`),
# which no chart's key may be.
_RESERVED_KEYS = frozenset({"dashboard", "layout", "check"})
# A build id the marker grammar reads, to hold a chart's key to the marker's step key (`format_marker`).
_ANY_BUILD = "00000000-0000-0000-0000-000000000000"

# Bounds on what one plan sends Superset: each written measure is one decided statement, and a dashboard of more charts
# than this is not one a goal is read into.
MAX_CHARTS = 16
MAX_MEASURES = 24
MAX_GOAL = 2000
# `slice_name` and a dataset's name are at most 250 characters in rc2's models.
MAX_TITLE = 250
MAX_REASON = 500

# Which dashboards are offered: at most this many of the goal's words are searched, and this many dashboards shown.
MAX_SEARCH_WORDS = 6
DASHBOARD_PAGE = 25
MAX_DASHBOARDS = 20
_WORD = re.compile(r"[^\W_]+")
_STOP_WORDS = frozenset({"and", "the", "for", "with", "from", "into", "per", "our", "all", "any", "this", "that",
                         "show", "build", "make", "create", "dashboard", "dashboards", "chart", "charts"})

# The room the model is given to write a plan in, as the generator is given room to write SQL.
PLANNER_MAX_TOKENS = 4000

DATASETS_UNREADABLE = ("Superset's list of this builder's datasets could not be read, so no plan was made; mnemiq's "
                       "log records why.")
NOTHING_VISIBLE = "This builder can read no table here, so there is nothing to plan a dashboard on."


# ---------------------------------------------------------------- the plan

@dataclass(frozen=True)
class PlannedDataset:
    """A dataset the build makes. Physical, on `table` (the snapshot's table, as the model and the deciders name it), or
    virtual, on `join` (an unparenthesized join, as `check_measure` takes one). The plan check fills in what the build
    sends: a physical dataset's `table_name` (as Superset stores the table), a virtual one's `sql` and `name`."""

    key: str
    placeholder: int
    table: str | None = None
    join: str | None = None
    table_name: str | None = None
    sql: str | None = None
    name: str | None = None

    @property
    def virtual(self) -> bool:
        return self.join is not None


@dataclass(frozen=True)
class BuildPlan:
    """A plan: its charts, the measures mnemiq writes for them (never a saved metric reused, never Superset's `count`),
    the dashboards offered for reuse, and the datasets the build makes. While `reuse_candidates` holds a dashboard, the
    build waits for the builder's answer (`choose_dashboard`)."""

    charts: tuple[ChartSpec, ...]
    measures: tuple[Measure, ...]
    reuse_candidates: tuple[int, ...] = ()
    reuse_reason: str | None = None
    datasets: tuple[PlannedDataset, ...] = ()

    @property
    def awaiting_choice(self) -> bool:
        """Whether nothing may be built until the builder says to reuse the dashboard offered, or to start fresh."""
        return bool(self.reuse_candidates)


@dataclass(frozen=True)
class RefusedPart:
    """A part of a plan mnemiq refused, as the model named it (`measure 'x'`, `chart 'y'`, `the join '...'`), with
    mnemiq's own reason."""

    part: str
    reason: str


@dataclass(frozen=True)
class PlanRefused:
    """No plan. `parts` names each refused part; `draft` is the plan as the model wrote it, when one could be read, for
    its one repair."""

    reason: str
    parts: tuple[RefusedPart, ...] = ()
    draft: str | None = None


# ---------------------------------------------------------------- what the model sees

@dataclass(frozen=True)
class PlanColumn:
    name: str
    type: str | None


@dataclass(frozen=True)
class PlanMetric:
    """A saved metric, by its name and expression only."""

    name: str
    expression: str


@dataclass(frozen=True)
class PlanTable:
    """A table the builder can read: its columns, the id of its dataset (None: the build makes one), and that dataset's
    saved metrics, or the `count` Superset gives the dataset the build makes."""

    name: str
    columns: tuple[PlanColumn, ...]
    dataset_id: int | None
    saved_metrics: tuple[PlanMetric, ...]


@dataclass(frozen=True)
class PlanDashboard:
    id: int
    title: str


@dataclass(frozen=True)
class Repair:
    """The one repair: the plan as the model wrote it, and the parts mnemiq refused, each with its reason."""

    previous: str | None
    refused: tuple[RefusedPart, ...]


@dataclass(frozen=True)
class PlanContext:
    tables: tuple[PlanTable, ...]
    dashboards: tuple[PlanDashboard, ...]
    families: tuple[Family, ...]
    repair: Repair | None = None


class Planner(Protocol):
    def plan(self, goal: str, context: PlanContext) -> BuildPlan | PlanRefused: ...


# ---------------------------------------------------------------- the model's channel

_KINDS = [kind.value for kind in MeasureKind]


def _family_name(family: Family) -> str:
    return family.name.lower()


_FAMILY_BY_NAME = {_family_name(family): family for family in Family}

_FILTER_SCHEMA = {"type": "object", "properties": {"column": {"type": "string"},
                                                    "op": {"type": "string", "enum": list(FILTER_OPERATORS)},
                                                    "operand": {}},
                  "required": ["column", "op"]}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "measures": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "label": {"type": "string"}, "expression": {"type": "string"},
            "kind": {"type": "string", "enum": _KINDS}, "table": {"type": "string"}, "join": {"type": "string"},
            "saved": {"type": "boolean"}, "description": {"type": "string"}, "currency": {"type": "string"}},
            "required": ["name", "kind"]}},
        "charts": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "family": {"type": "string", "enum": sorted(_FAMILY_BY_NAME)},
            "title": {"type": "string"}, "measures": {"type": "array", "items": {"type": "string"}},
            "dimensions": {"type": "array", "items": {"type": "string"}}, "time_column": {"type": "string"},
            "grain": {"type": "string", "enum": list(GRAINS)}, "filters": {"type": "array", "items": _FILTER_SCHEMA},
            "row_limit": {"type": "integer"}, "source": {"type": "string"}, "target": {"type": "string"},
            "table": {"type": "string"}, "join": {"type": "string"}},
            "required": ["key", "family", "title", "measures", "dimensions"]}},
        "reuse_dashboard": {"type": ["object", "null"], "properties": {"id": {"type": "integer"},
                                                                       "reason": {"type": "string"}}},
    },
    "required": ["measures", "charts"],
}
# Without `strict`, as the generator sends its own schema: strict also demands every property in `required`.
_RESPONSE_FORMAT = {"response_format": {"type": "json_schema",
                                        "json_schema": {"name": "dashboard_plan", "schema": PLAN_SCHEMA}}}

_SYSTEM = """You plan a Superset dashboard for a builder's goal, from the tables, saved metrics and dashboards you are \
shown. Answer with one JSON object and nothing else:
{"measures": [...], "charts": [...], "reuse_dashboard": null}

Measures are the numbers the charts show. Each is either
- written: {"name": lower case letters, digits and _, "label": the name shown on the chart, "expression": one SQL \
aggregate expression in PostgreSQL over the measure's rows, "kind": one of currency, percent, count, number, "table": \
a table's name, or "join": "a JOIN b ON ...", "description": optional, "currency": an ISO 4217 code, optional}; or
- a saved metric of a table's dataset: {"name": the saved metric's name, "saved": true, "kind": ..., "table": ...}.
Use a saved metric where one says what a measure needs. Write a join only where a measure needs columns of more than \
one table. On a join, name each column alone (amount, not claim.amount): a joined dataset names its columns without \
their tables, and a column two joined tables share is not in it. A written measure's name and label are never a saved \
metric's name on its table, and no two measures share a label. A table with no dataset gets one, with Superset's saved \
metric count (COUNT(*)).

Charts: {"key": lower case letters, digits and _, at most 40, one per chart, "family": one of FAMILIES, "title", \
"measures": [measure names], "dimensions": [columns], "time_column": a date column or omitted, "grain": one of GRAINS \
or omitted, "filters": [{"column", "op": one of =, !=, >, <, >=, <=, LIKE, ILIKE, NOT LIKE, IN, NOT IN, IS NULL, IS \
NOT NULL, "operand": one value, a list for IN and NOT IN, omitted for IS NULL and IS NOT NULL}], "row_limit": \
optional, "source" and "target": a sankey's two columns, "table" or "join": the chart's rows, needed only when it has \
no measures}. A chart's measures, dimensions and filters are all on one table or one join.
- pie, big_number, gauge, treemap, waterfall, funnel and sankey plot exactly one measure; mixed plots two or more on a \
time column, the first on its primary axis;
- pie, treemap, funnel, histogram, pivot and table need a dimension; waterfall and box_plot need a dimension or a time \
column;
- a histogram takes no measure: its first dimension is the column whose spread it shows;
- sankey, box_plot and histogram cannot show currency or percent; handlebars shows count or number only;
- one axis shows one kind of measure.

reuse_dashboard: {"id": a dashboard's id, "reason": why it already covers the goal}, only for one of the dashboards \
listed and only when it covers the goal; the builder is then asked whether to reuse it. Otherwise null."""


def plan_prompt(goal: str, context: PlanContext) -> tuple[str, str]:
    """The system and user prompts the model is sent for `goal`: everything it sees."""
    families = ", ".join(_family_name(family) for family in context.families)
    system = _SYSTEM.replace("FAMILIES", families).replace("GRAINS", ", ".join(GRAINS))
    document = {
        "tables": [{"name": table.name,
                    "columns": [{"name": c.name, **({"type": c.type} if c.type else {})} for c in table.columns],
                    "has_dataset": table.dataset_id is not None,
                    "saved_metrics": [{"name": m.name, "expression": m.expression} for m in table.saved_metrics]}
                   for table in context.tables],
        "dashboards": [{"id": d.id, "title": d.title} for d in context.dashboards],
    }
    user = f"Goal: {goal}\n\nWhat this builder can see:\n{json.dumps(document, ensure_ascii=False)}"
    if context.repair is not None:
        lines = "\n".join(f"- {part.part}: {part.reason}" for part in context.repair.refused)
        previous = f"Your plan was:\n{context.repair.previous}\n\n" if context.repair.previous else ""
        user += (f"\n\n{previous}mnemiq refused these parts of it:\n{lines}\n\nPlan again. Keep every part that was "
                 "not refused as it was, and change the refused ones so that their reasons no longer apply.")
    return system, user


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _reply_object(raw: str) -> dict | PlanRefused:
    """The JSON object a reply holds, read as the generator's `_parse` reads one: first brace to last. A reply that
    holds none is a refusal, never an exception."""
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return PlanRefused("the model did not answer with a plan on the JSON channel")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return PlanRefused("the model's plan was malformed JSON")
    if not isinstance(payload, dict):
        return PlanRefused("the model's plan was not a JSON object")
    return payload


class LLMPlanner:
    """Plans with the profile's model, its reply held to `PLAN_SCHEMA` with a `json_schema` `response_format`."""

    def __init__(self, client, max_tokens: int = PLANNER_MAX_TOKENS) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def plan(self, goal: str, context: PlanContext) -> BuildPlan | PlanRefused:
        system, user = plan_prompt(goal, context)
        raw = self._client.complete(system, user, max_tokens=self._max_tokens, extra_body=_RESPONSE_FORMAT)
        draft = _reply_object(raw)
        return draft if isinstance(draft, PlanRefused) else plan_from_draft(draft, context)


class FakePlanner:
    """Returns the plans it was given, in order, and the last again for any later call, recording each context."""

    def __init__(self, plan: BuildPlan | PlanRefused | Sequence[BuildPlan | PlanRefused]) -> None:
        self._plans = list(plan) if isinstance(plan, (list, tuple)) else [plan]
        if not self._plans:
            raise ValueError("a fake planner needs a plan")
        self.goals: list[str] = []
        self.contexts: list[PlanContext] = []

    def plan(self, goal: str, context: PlanContext) -> BuildPlan | PlanRefused:
        self.goals.append(goal)
        self.contexts.append(context)
        return self._plans.pop(0) if len(self._plans) > 1 else self._plans[0]


# ---------------------------------------------------------------- the model's plan, read

def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


def _join_key(join: str) -> str | None:
    """The join as sqlglot writes it back, so two spellings of one join plan one dataset; None when it does not
    parse."""
    try:
        tree = sqlglot.parse_one(f"SELECT 1 FROM {join}", read=SUPERSET_DIALECT)
    except SqlglotError:
        return None
    if not isinstance(tree, exp.Select) or tree.args.get("from_") is None:
        return None
    return tree.sql(dialect=SUPERSET_DIALECT, normalize=True)


def plan_from_draft(draft: Mapping[str, Any], context: PlanContext) -> BuildPlan | PlanRefused:
    """The plan a model's JSON object describes, its tables and joins read against `context`: a table with a dataset is
    that dataset, and a table without one, or a join, a dataset the build makes. A part that cannot be read is refused,
    with the draft kept for the repair."""
    refused: list[RefusedPart] = []
    tables = {table.name: table for table in context.tables}
    planned: dict[tuple[str, str], PlannedDataset] = {}

    def refuse(part: str, reason: str) -> None:
        refused.append(RefusedPart(part, reason))

    def dataset_of(item: Mapping, part: str) -> int | None:
        table, join = item.get("table"), item.get("join")
        if (table is None) == (join is None):
            refuse(part, "name its rows with a table or a join, one of the two")
            return None
        if table is not None:
            if not isinstance(table, str) or table not in tables:
                refuse(part, f"{table!r} is not a table this builder can read")
                return None
            if tables[table].dataset_id is not None:
                return tables[table].dataset_id
            key = ("table", table)
        else:
            normal = _join_key(join) if isinstance(join, str) and join.strip() else None
            if normal is None:
                refuse(part, f"the join {join!r} is not one SQL join")
                return None
            key = ("join", normal)
        if key not in planned:
            n = len(planned) + 1
            planned[key] = PlannedDataset(key=f"dataset_{n}", placeholder=-n, table=table,
                                          join=join.strip() if isinstance(join, str) else None)
        return planned[key].placeholder

    written: dict[str, Measure] = {}
    saved: dict[str, tuple[SavedRef, int]] = {}
    measures = draft.get("measures")
    if not isinstance(measures, list):
        refuse("the plan", "its measures are a list")
        measures = []
    for index, item in enumerate(measures, 1):
        name = _text(item.get("name")) if isinstance(item, dict) else None
        part = f"measure {name!r}" if name else f"measure {index}"
        if not isinstance(item, dict) or name is None:
            refuse(part, "a measure is an object with a name")
            continue
        if name in written or name in saved:
            refuse(part, "two measures have this name")
            continue
        try:
            kind = MeasureKind(item.get("kind"))
        except ValueError:
            refuse(part, f"its kind {item.get('kind')!r} is not one of {', '.join(_KINDS)}")
            kind = None
        dataset = dataset_of(item, part)
        if item.get("saved") is True:
            if kind is not None and dataset is not None:
                saved[name] = (SavedRef(name, kind), dataset)
            continue
        label, expression = item.get("label"), item.get("expression")
        description, currency = item.get("description", ""), item.get("currency")
        if not isinstance(label, str) or not isinstance(expression, str) or not expression.strip():
            refuse(part, "a written measure has a label and an expression")
            continue
        if not isinstance(description, str) or not (currency is None or isinstance(currency, str)):
            refuse(part, "its description and currency are text")
            continue
        if kind is not None and dataset is not None:
            written[name] = Measure(name=name, label=label.strip(), expression=expression.strip(), kind=kind,
                                    dataset_id=dataset, description=description.strip(), asked_by="",
                                    currency=currency.strip() if currency else None)

    charts: list[ChartSpec] = []
    items = draft.get("charts")
    if not isinstance(items, list):
        refuse("the plan", "its charts are a list")
        items = []
    for index, item in enumerate(items, 1):
        key = item.get("key") if isinstance(item, dict) else None
        part = f"chart {key!r}" if isinstance(key, str) else f"chart {index}"
        if not isinstance(item, dict) or not isinstance(key, str):
            refuse(part, "a chart is an object with a key")
            continue
        before = len(refused)
        family = _FAMILY_BY_NAME.get(item.get("family")) if isinstance(item.get("family"), str) else None
        if family is None:
            refuse(part, f"its family {item.get('family')!r} is not one of "
                         f"{', '.join(_family_name(f) for f in context.families)}")
        title = item.get("title")
        if not isinstance(title, str):
            refuse(part, "a chart has a title")
        refs, of_refs = [], set()
        names = item.get("measures")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            refuse(part, "its measures are a list of measure names")
            names = []
        for name in names:
            if name in written:
                refs.append(WrittenRef(written[name]))
                of_refs.add(written[name].dataset_id)
            elif name in saved:
                refs.append(saved[name][0])
                of_refs.add(saved[name][1])
            else:
                refuse(part, f"it plots {name!r}, which is no measure of the plan")
        own = dataset_of(item, part) if item.get("table") is not None or item.get("join") is not None else None
        if len(of_refs) > 1:
            refuse(part, "its measures are on more than one table or join; a chart's measures are on one")
        elif own is not None and of_refs and of_refs != {own}:
            refuse(part, "its table or join is not its measures'")
        dataset = own if own is not None else next(iter(of_refs), None)
        dimensions = item.get("dimensions", [])
        if not isinstance(dimensions, list) or not all(isinstance(d, str) for d in dimensions):
            refuse(part, "its dimensions are a list of column names")
            dimensions = []
        grain = item.get("grain")
        if grain is not None and grain not in GRAINS and grain not in _GRAIN_WORD:
            refuse(part, f"its grain {grain!r} is not one of {', '.join(GRAINS)}")
        filters = []
        for each in item.get("filters") or []:
            if not isinstance(each, dict) or not isinstance(each.get("column"), str) \
                    or not isinstance(each.get("op"), str):
                refuse(part, "a filter is an object with a column and an op")
                continue
            filters.append(Filter(each["column"], each["op"], each.get("operand", each.get("value"))))
        row_limit = item.get("row_limit")
        if row_limit is not None and (type(row_limit) is not int):
            refuse(part, "its row_limit is a whole number")
        texts = {name: item.get(name) for name in ("time_column", "source", "target")}
        if any(value is not None and not isinstance(value, str) for value in texts.values()):
            refuse(part, "its time_column, source and target are column names")
        if dataset is None and len(refused) == before:
            refuse(part, "it names no table or join for its rows")
        if len(refused) > before:
            continue
        charts.append(ChartSpec(key=key, family=family, title=title.strip(), dataset_id=dataset,
                                measures=tuple(refs), dimensions=tuple(dimensions),
                                time_column=texts["time_column"], time_grain=GRAINS.get(grain, grain),
                                filters=tuple(filters), row_limit=row_limit, source=texts["source"],
                                target=texts["target"]))

    reuse, candidates, reason = draft.get("reuse_dashboard"), (), None
    if isinstance(reuse, dict) and type(reuse.get("id")) is int:
        candidates = (reuse["id"],)
        reason = reuse.get("reason").strip() if isinstance(reuse.get("reason"), str) else None
    elif type(reuse) is int:
        candidates = (reuse,)
    elif reuse is not None:
        refuse("the dashboard to reuse", 'it is {"id": a dashboard\'s id, "reason": "..."}, or null')

    draft_text = json.dumps(draft, ensure_ascii=False)
    if refused:
        return PlanRefused(_naming(refused), tuple(refused), draft=draft_text)
    return BuildPlan(charts=tuple(charts), measures=tuple(written.values()), reuse_candidates=candidates,
                     reuse_reason=reason, datasets=tuple(planned.values()))


def draft_of(plan: BuildPlan, context: PlanContext) -> dict:
    """`plan` as the model writes one (`plan_from_draft` reads it back), its datasets named by table or join: what the
    model is shown of its own plan when a part goes back for repair."""
    by_id = {table.dataset_id: table.name for table in context.tables if table.dataset_id is not None}
    planned = {dataset.placeholder: dataset for dataset in plan.datasets}

    def rows(dataset_id: int) -> dict:
        if dataset_id in planned:
            made = planned[dataset_id]
            return {"join": made.join} if made.virtual else {"table": made.table}
        return {"table": by_id.get(dataset_id, f"dataset {dataset_id}")}

    measures: list[dict] = []
    for measure in plan.measures:
        doc = {"name": measure.name, "label": measure.label, "expression": measure.expression,
               "kind": measure.kind.value, **rows(measure.dataset_id)}
        if measure.description:
            doc["description"] = measure.description
        if measure.currency:
            doc["currency"] = measure.currency
        measures.append(doc)
    seen = {measure.name for measure in plan.measures}
    charts: list[dict] = []
    for chart in plan.charts:
        for ref in chart.measures:
            if isinstance(ref, SavedRef) and ref.name not in seen:
                seen.add(ref.name)
                measures.append({"name": ref.name, "saved": True, "kind": ref.kind.value, **rows(chart.dataset_id)})
        doc = {"key": chart.key, "family": _family_name(chart.family), "title": chart.title,
               "measures": [ref.name if isinstance(ref, SavedRef) else ref.measure.name for ref in chart.measures],
               "dimensions": list(chart.dimensions)}
        if not chart.measures:
            doc.update(rows(chart.dataset_id))
        for name, value in (("time_column", chart.time_column), ("grain", _GRAIN_WORD.get(chart.time_grain or "",
                                                                                            chart.time_grain)),
                            ("row_limit", chart.row_limit), ("source", chart.source), ("target", chart.target)):
            if value is not None:
                doc[name] = value
        if chart.filters:
            doc["filters"] = [{"column": f.column, "op": f.op, **({} if f.value is None else {"operand": f.value})}
                              for f in chart.filters]
        charts.append(doc)
    reuse = None
    if plan.reuse_candidates:
        reuse = {"id": plan.reuse_candidates[0], **({"reason": plan.reuse_reason} if plan.reuse_reason else {})}
    return {"measures": measures, "charts": charts, "reuse_dashboard": reuse}


def _naming(parts: Sequence[RefusedPart]) -> str:
    return "mnemiq could not plan this dashboard: " + "; ".join(f"{p.part}: {p.reason}" for p in parts)


# ---------------------------------------------------------------- what the builder can see

@dataclass(frozen=True)
class _Catalogue:
    """What a plan is checked against: the visible tables' columns, each table's name as stored, its existing
    dataset, and that dataset's saved metrics as the builder reads them."""

    columns: dict[str, frozenset[str]]
    stored: dict[str, str]
    datasets: dict[str, int]
    saved: dict[int, tuple[SavedMetric, ...]]
    types: dict[str, dict[str, str]] = field(default_factory=dict)


def _datasets_by_table(adapter, stored: dict[str, str]) -> dict[str, int]:
    """Each visible table's physical dataset on the configured database and schema, as `resolve_grants` reads them
    (`_read_datasets`, `_candidate_names`): the one named exactly as the table is stored, else the first by id."""
    physical = [d for d in _read_datasets(adapter)
                if d.database_id == adapter.database_id and d.schema == adapter.schema and not d.sql]
    found: dict[str, int] = {}
    for table, name in stored.items():
        named = [d for d in physical if name in _candidate_names(d.table_name)]
        if named:
            found[table] = min(named, key=lambda d: (d.table_name != name, d.id)).id
    return found


def _catalogue(client, adapter, snapshot: Snapshot, grants: GrantSet) -> _Catalogue | PlanRefused:
    visible = visible_schema(snapshot, grants)
    stored = {table: table for table in visible}
    for name, tables in _tables(snapshot).items():
        for table in tables:
            if table in visible:
                stored[table] = name
    try:
        datasets = _datasets_by_table(adapter, stored)
    except _Unreadable:  # logged where it was found
        return PlanRefused(DATASETS_UNREADABLE)
    saved: dict[int, list[SavedMetric]] = {dataset: [] for dataset in datasets.values()}
    for metric in read_saved_metrics(client, dataset_ids=tuple(sorted(set(datasets.values())))):
        saved.setdefault(metric.dataset_id, []).append(metric)
    types = column_types(snapshot)
    return _Catalogue(columns={table: frozenset(columns) for table, columns in visible.items()}, stored=stored,
                      datasets=datasets, saved={d: tuple(sorted(ms, key=lambda m: m.name)) for d, ms in saved.items()},
                      types={table: types.get(table, {}) for table in visible})


def _words(text: str) -> list[str]:
    """The words of `text` a dashboard's title is matched on, in order, once each: folded, of three letters or more,
    and not one of the few that every goal and title has."""
    words: list[str] = []
    for word in _WORD.findall(text.casefold()):
        if len(word) >= 3 and word not in _STOP_WORDS and word not in words:
            words.append(word)
    return words


def _dashboards(client, goal: str) -> tuple[PlanDashboard, ...]:
    """The dashboards the builder can see whose titles share a word with the goal, by id and title only: Superset's
    `list_dashboards` searched for each of the goal's words, asked for those two columns alone."""
    words = _words(goal)
    found: dict[int, str] = {}
    for word in words[:MAX_SEARCH_WORDS]:
        answer = client.call("list_dashboards", {"search": word, "select_columns": ["id", "dashboard_title"],
                                                 "page": 1, "page_size": DASHBOARD_PAGE})
        listed = answer.get("dashboards") if isinstance(answer, dict) else None
        if not isinstance(listed, list):
            logger.info("list_dashboards answered with no list of dashboards; none is offered for %r", word)
            continue
        for item in listed:
            if not isinstance(item, dict):
                continue
            dashboard, title = item.get("id"), item.get("dashboard_title")
            if type(dashboard) is int and dashboard > 0 and isinstance(title, str) and set(_words(title)) & set(words):
                found.setdefault(dashboard, title.strip()[:MAX_TITLE])
    return tuple(PlanDashboard(d, found[d]) for d in sorted(found)[:MAX_DASHBOARDS])


def _context(catalogue: _Catalogue, dashboards, families) -> PlanContext:
    tables = []
    for table in sorted(catalogue.columns):
        dataset = catalogue.datasets.get(table)
        metrics = (tuple(PlanMetric(m.name, m.expression) for m in catalogue.saved.get(dataset, ()))
                   if dataset is not None else (PlanMetric(*COUNT_METRIC),))
        columns = tuple(PlanColumn(c, catalogue.types[table].get(c)) for c in sorted(catalogue.columns[table]))
        tables.append(PlanTable(table, columns, dataset, metrics))
    return PlanContext(tables=tuple(tables), dashboards=dashboards, families=families)


# ---------------------------------------------------------------- the plan check

@dataclass(frozen=True)
class _Rows:
    """A dataset a part of the plan is on: its columns, the source its measures are decided over, its saved metrics,
    how the model named it, and the planned dataset when the build makes it."""

    columns: frozenset[str]
    source: str
    saved: tuple[SavedMetric, ...]
    part: str
    planned: PlannedDataset | None = None


def _quoted(name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=SUPERSET_DIALECT)


def _made_count(dataset_id: int) -> SavedMetric:
    name, expression = COUNT_METRIC
    return SavedMetric(dataset_id=dataset_id, name=name, expression=expression, verbose_name="COUNT(*)",
                       d3format=None, currency=None, certified_by=None, warning_text=None)


def _joined(planned: PlannedDataset, catalogue: _Catalogue) -> _Rows | str:
    """The rows of a planned join, with the SELECT its virtual dataset is made from, or why the join is refused. The
    dataset names each column one joined table alone has, without its table, so a measure decided over the join names
    the same column on the dataset."""
    # The join is the virtual dataset's SQL, which Superset renders as a template.
    if (marker := next((d for d in JINJA_DELIMITERS if d in planned.join), None)) is not None:
        return f"it holds {marker}, which Superset reads as a template marker; write it without {marker}"
    try:
        tree = sqlglot.parse_one(f"SELECT 1 FROM {planned.join}", read=SUPERSET_DIALECT)
    except SqlglotError:
        return "it is not one SQL join"
    if not isinstance(tree, exp.Select) or {key for key, value in tree.args.items() if value} - {"expressions",
                                                                                                 "from_", "joins"}:
        return "it is a join of tables and nothing else: no WHERE, GROUP BY or LIMIT"
    if any(tree.find_all(exp.Subquery, exp.Values, exp.Lateral, exp.Unnest)):
        return "it joins tables only, never a subquery"
    joins = tree.args.get("joins") or []
    if not joins or any(join.args.get("on") is None or join.args.get("using") for join in joins):
        return "it is a JOIN of two or more tables, each joined ON a condition"
    nodes = list(tree.find_all(exp.Table))
    names = []
    for node in nodes:
        name = resolve_identifier(node.this, SUPERSET_DIALECT)
        if node.args.get("db") or node.args.get("catalog"):
            return "it names its tables without a schema"
        if name not in catalogue.columns:
            return f"{node.name!r} is not a table this builder can read"
        if name in names:
            return f"it joins {node.name!r} to itself"
        names.append(name)
    counted: dict[str, int] = {}
    for name in names:
        for column in catalogue.columns[name]:
            counted[column] = counted.get(column, 0) + 1
    picked = []
    for node, name in zip(nodes, names):
        qualifier = (node.args["alias"].this if node.args.get("alias") else node.this).sql(dialect=SUPERSET_DIALECT)
        picked.extend(f"{qualifier}.{_quoted(c)} AS {_quoted(c)}" for c in sorted(catalogue.columns[name])
                      if counted[c] == 1)
    if not picked:
        return "its tables have no column of their own to show"
    exposed = frozenset(c for c, n in counted.items() if n == 1)
    made = replace(planned, sql=f"SELECT {', '.join(picked)} FROM {planned.join}",
                   name="_".join(["mnemiq", *names])[:MAX_TITLE])
    return _Rows(exposed, planned.join, (_made_count(planned.placeholder),), f"the join {planned.join!r}", made)


def _column_problem(expression: str, columns: frozenset[str]) -> str | None:
    """Why a measure's expression cannot run on a joined dataset as it was decided over the join: a column named with
    its table, or one the dataset does not have."""
    try:
        tree = sqlglot.parse_one(f"SELECT {expression}", read=SUPERSET_DIALECT)
    except SqlglotError:
        return None  # `check_measure` says why
    for column in tree.find_all(exp.Column):
        if column.table:
            return (f"name {column.sql(dialect=SUPERSET_DIALECT)} alone, as {column.name}: the joined dataset names "
                    "its columns without their tables")
        name = resolve_identifier(column.this, SUPERSET_DIALECT)
        if name not in columns:
            return f"{column.name!r} is not a column of the joined dataset: it is in both tables, or in neither"
    return None


def _filter_problem(item: Filter, columns: frozenset[str]) -> str | None:
    if item.column not in columns:
        return f"the filter's column {item.column!r} is not a column of its rows"
    if item.op not in FILTER_OPERATORS:
        return f"the filter operator {item.op!r} is not one of {', '.join(FILTER_OPERATORS)}"
    operand = item.value
    if item.op in _NO_OPERAND:
        return None if operand is None else f"{item.op} takes no operand"
    if item.op in _LIST_OPERAND:
        if not isinstance(operand, list) or not operand or not all(_scalar(x) for x in operand):
            return f"{item.op} takes a list of one or more values"
        items = operand
    elif operand is None:
        return f"{item.op} takes an operand"
    elif not _scalar(operand):
        return f"{item.op} takes one operand, not a list"
    else:
        items = [operand]
    if any(isinstance(x, str) and len(x) > _MAX_OPERAND for x in items):
        return f"an operand is at most {_MAX_OPERAND} characters"
    return None


def _chart_columns(chart: ChartSpec) -> list[str]:
    return [*chart.dimensions, *(c for c in (chart.time_column, chart.source, chart.target) if c)]


def _check(plan: BuildPlan, catalogue: _Catalogue, context: PlanContext, *, adapter, snapshot: Snapshot,
           grants: GrantSet, settings, interactive_pivot_available: bool) -> BuildPlan | tuple[RefusedPart, ...]:
    """The plan with its reusable measures replaced by the saved metrics that say the same, or every part refused, each
    with mnemiq's own reason."""
    refused: list[RefusedPart] = []

    def refuse(part: str, reason: str) -> None:
        refused.append(RefusedPart(part, reason))

    if not plan.charts:
        refuse("the plan", "it has no chart")
    if len(plan.charts) > MAX_CHARTS:
        refuse("the plan", f"it has {len(plan.charts)} charts; a plan has at most {MAX_CHARTS}")

    # Only what a chart plots is built: a measure or a dataset no chart uses is left out.
    plotted = {ref.measure.name for chart in plan.charts for ref in chart.measures if isinstance(ref, WrittenRef)}
    measures = [m for m in plan.measures if m.name in plotted]
    if len(measures) > MAX_MEASURES:
        refuse("the plan", f"it writes {len(measures)} measures; a plan writes at most {MAX_MEASURES}")
    used = {m.dataset_id for m in measures} | {chart.dataset_id for chart in plan.charts}

    # The datasets parts can be on: each visible table's existing one, and each planned one the plan uses.
    by_dataset = {dataset: table for table, dataset in catalogue.datasets.items()}
    rows: dict[int, _Rows] = {
        dataset: _Rows(catalogue.columns[table], _quoted(catalogue.stored[table]), catalogue.saved.get(dataset, ()),
                       f"the table {table!r}")
        for dataset, table in by_dataset.items()}
    keys: set[str] = set()
    for planned in plan.datasets:
        if planned.placeholder not in used:
            continue
        part = f"the join {planned.join!r}" if planned.virtual else f"the table {planned.table!r}"
        try:
            format_marker(_ANY_BUILD, planned.key, None)
            good_key = planned.key not in keys and planned.key not in _RESERVED_KEYS
        except ValueError:
            good_key = False
        if not good_key or planned.placeholder >= 0 or planned.placeholder in rows \
                or (planned.table is None) == (planned.join is None):
            refuse(part, "a planned dataset has a key of its own, a negative placeholder, and a table or a join")
            continue
        keys.add(planned.key)
        if planned.virtual:
            joined = _joined(planned, catalogue)
            if isinstance(joined, str):
                refuse(part, joined)
                continue
            if not any(m.dataset_id == planned.placeholder for m in measures):
                refuse(part, "a joined dataset is made only for a written measure that needs the join")
                continue
            rows[planned.placeholder] = joined
        elif planned.table not in catalogue.columns:
            refuse(part, "it is not a table this builder can read")
        elif planned.table in catalogue.datasets:
            refuse(part, "the table already has a dataset; plan on it")
        else:
            rows[planned.placeholder] = _Rows(catalogue.columns[planned.table],
                                              _quoted(catalogue.stored[planned.table]),
                                              (_made_count(planned.placeholder),), part,
                                              replace(planned, table_name=catalogue.stored[planned.table]))

    # Each written measure: decided, then reused where a saved metric says the same, else held apart from every name
    # its chart would show beside it.
    names: dict[str, str] = {}
    labels: dict[str, str] = {}
    reused: dict[str, SavedRef] = {}
    kept: list[Measure] = []
    columns_of: dict[str, set[str]] = {}
    for chart in plan.charts:
        for ref in chart.measures:
            if isinstance(ref, WrittenRef):
                columns_of.setdefault(ref.measure.name, set()).update(_chart_columns(chart))
    for measure in measures:
        part = f"measure {measure.name!r}"
        if measure.name.casefold() in names:
            refuse(part, "two measures have this name")
            continue
        names[measure.name.casefold()] = measure.name
        if measure.label.strip().casefold() in labels:
            refuse(part, f"its label {measure.label!r} is also measure {labels[measure.label.strip().casefold()]!r}'s;"
                         " give each measure a label of its own")
            continue
        labels[measure.label.strip().casefold()] = measure.name
        on = rows.get(measure.dataset_id)
        if on is None:
            refuse(part, "its rows are not a table this builder can read, or a join the plan makes")
            continue
        if on.planned is not None and on.planned.virtual \
                and (problem := _column_problem(measure.expression, on.columns)) is not None:
            refuse(part, problem)
            continue
        verdict = check_measure(measure, source=on.source, dimension_names=tuple(sorted(columns_of[measure.name])),
                                adapter=adapter, snapshot=snapshot, grants=grants, guard_fanout=settings.guard_fanout)
        if isinstance(verdict, MeasureRefused):
            refuse(part, verdict.reason)
            continue
        hit = reusable(measure, list(on.saved))
        if hit is not None:
            reused[measure.name] = SavedRef(hit.name, measure.kind)
            continue
        taken = {metric.name.casefold() for metric in on.saved}
        for what, text in (("name", measure.name), ("label", measure.label.strip())):
            if text.casefold() in taken:
                made = " (Superset gives every dataset it makes one named count)" if on.planned is not None else ""
                refuse(part, f"its {what} {text!r} is also a saved metric's on its dataset{made}; Superset would show "
                             "two metrics of one name, so name it differently")
                break
        else:
            kept.append(measure)
    by_name = {measure.name: measure for measure in measures}

    # Each chart, with the measures it plots as the build will send them.
    charts: list[ChartSpec] = []
    chart_keys: set[str] = set()
    for chart in plan.charts:
        part = f"chart {chart.key!r}"
        before = len(refused)
        try:
            format_marker(_ANY_BUILD, chart.key, None)
            if chart.key in _RESERVED_KEYS or chart.key in keys:
                raise ValueError
        except ValueError:
            refuse(part, "a chart's key is 1 to 40 of a-z, 0-9 and _, and not dashboard, layout, check or a dataset's")
        if chart.key in chart_keys:
            refuse(part, "two charts have this key")
        chart_keys.add(chart.key)
        if not chart.title.strip() or len(chart.title) > MAX_TITLE:
            refuse(part, f"a chart's title is 1 to {MAX_TITLE} characters")
        on = rows.get(chart.dataset_id)
        if on is None:
            refuse(part, "its rows are not a table this builder can read, or a join the plan makes")
            charts.append(chart)
            continue
        refs = []
        for ref in chart.measures:
            if isinstance(ref, WrittenRef):
                measure = by_name.get(ref.measure.name)
                if measure is None:
                    refuse(part, f"it plots {ref.measure.name!r}, which is no measure of the plan")
                elif measure.dataset_id != chart.dataset_id:
                    refuse(part, f"it plots {measure.name!r}, whose rows are not the chart's")
                else:
                    refs.append(reused.get(measure.name, WrittenRef(measure)))
            else:
                refs.append(ref)
        spec = replace(chart, measures=tuple(refs))
        columns = _chart_columns(spec)
        folded = {c.casefold() for c in columns}
        saved = {metric.name: metric for metric in on.saved}
        for ref in refs:
            if not isinstance(ref, SavedRef):
                continue
            if ref.name not in saved:
                refuse(part, f"{on.part} has no saved metric named {ref.name!r} this builder can reach")
            elif ref.name.casefold() in folded:
                refuse(part, f"the saved metric {ref.name!r} is also the name of a dimension of the chart")
        shown = [(ref.name if isinstance(ref, SavedRef) else ref.measure.label.strip()).casefold() for ref in refs]
        if len(set(shown)) < len(shown):
            refuse(part, "it plots one metric twice")
        if spec.family is Family.HISTOGRAM and refs:
            refuse(part, "a histogram shows the spread of its first dimension and takes no measure")
        for column in columns:
            if column not in on.columns:
                refuse(part, f"{column!r} is not a column of {on.part}")
        if spec.time_grain is not None:
            if spec.time_grain not in _GRAIN_WORD:
                refuse(part, f"its grain is one of {', '.join(GRAINS)}")
            elif not spec.time_column:
                refuse(part, "a grain groups a time column, and the chart has none")
        for item in spec.filters:
            if (problem := _filter_problem(item, on.columns)) is not None:
                refuse(part, problem)
        if len(refused) == before:
            why = plan_refusal(spec, interactive_pivot_available=interactive_pivot_available,
                               saved={ref.name: saved[ref.name] for ref in refs if isinstance(ref, SavedRef)},
                               currency=settings.superset_currency)
            if why is not None:
                refuse(part, why)
        charts.append(spec)

    offered = {dashboard.id for dashboard in context.dashboards}
    if len(plan.reuse_candidates) > 1 or not set(plan.reuse_candidates) <= offered:
        refuse("the dashboard to reuse", "reuse only one of the dashboards listed, by its id, or none")
    if plan.reuse_reason is not None and len(plan.reuse_reason) > MAX_REASON:
        refuse("the dashboard to reuse", f"its reason is at most {MAX_REASON} characters")

    if refused:
        return tuple(refused)
    still = {m.dataset_id for m in kept} | {chart.dataset_id for chart in charts}
    return BuildPlan(charts=tuple(charts), measures=tuple(kept), reuse_candidates=plan.reuse_candidates,
                     reuse_reason=plan.reuse_reason,
                     datasets=tuple(rows[d.placeholder].planned for d in plan.datasets if d.placeholder in still))


# ---------------------------------------------------------------- the step

def plan_build(goal: str, *, planner: Planner, client, adapter, snapshot: Snapshot, grants: GrantSet, settings,
               interactive_pivot_available: bool = False) -> BuildPlan | PlanRefused:
    """The checked plan for `goal`, or why there is none. `client` is the request's Superset client, the one its
    adapter (`superset_executor(settings, delegation)`) carries, and every call goes through it as the builder; nothing
    is written. A part refused goes back to `planner` once, with mnemiq's reason; refused again, the plan is refused,
    its reason naming each part."""
    if client is not getattr(adapter, "client", None):
        raise ValueError("the planner calls Superset only through the client of the request's adapter")
    goal = (goal or "").strip()
    if not goal or len(goal) > MAX_GOAL:
        return PlanRefused(f"a goal is 1 to {MAX_GOAL} characters")
    catalogue = _catalogue(client, adapter, snapshot, grants)
    if isinstance(catalogue, PlanRefused):
        return catalogue
    if not catalogue.columns:
        return PlanRefused(NOTHING_VISIBLE)
    families = tuple(f for f in Family if f is not Family.INTERACTIVE_PIVOT or interactive_pivot_available)
    context = _context(catalogue, _dashboards(client, goal), families)
    parts: tuple[RefusedPart, ...] = ()
    for attempt in range(2):
        answer = planner.plan(goal, context)
        if isinstance(answer, PlanRefused):
            if not answer.parts and attempt == 0:
                return answer
            if not answer.parts:  # the repair could not be read: the parts it was to repair stay refused
                return PlanRefused(f"{_naming(parts)}; the repaired plan could not be read: {answer.reason}", parts)
            parts, previous = answer.parts, answer.draft
        else:
            checked = _check(answer, catalogue, context, adapter=adapter, snapshot=snapshot, grants=grants,
                             settings=settings, interactive_pivot_available=interactive_pivot_available)
            if (terminal := getattr(adapter, "terminal_refusal", None)) is not None:
                # Superset refused the caller, or the caller's roles, while a measure was decided: every later call
                # meets the same refusal, so the request ends here, as `superset_history` ends it.
                logger.warning("Superset refused this request (%s) while its plan was checked; the request ends here",
                               type(terminal).__name__)
                raise terminal
            if isinstance(checked, BuildPlan):
                return checked
            parts, previous = checked, json.dumps(draft_of(answer, context), ensure_ascii=False)
        if attempt == 0:
            context = replace(context, repair=Repair(previous=previous, refused=parts))
    return PlanRefused(_naming(parts), parts)


# ---------------------------------------------------------------- after the plan

def choose_dashboard(plan: BuildPlan, answer: str) -> tuple[BuildPlan, int | None]:
    """The builder's answer to the dashboard offered: `reuse` builds on it, `fresh` builds a new one. The plan comes
    back with no offer open, beside the dashboard to build on (None: a new one)."""
    if not plan.reuse_candidates:
        raise ValueError("this plan offers no dashboard to reuse")
    settled = replace(plan, reuse_candidates=(), reuse_reason=None)
    if answer == "reuse":
        return settled, plan.reuse_candidates[0]
    if answer == "fresh":
        return settled, None
    raise ValueError("the answer to a dashboard offered is reuse or fresh")


def with_made_datasets(plan: BuildPlan, made: Mapping[str, int]) -> BuildPlan:
    """`plan` with each made dataset's id, by its key, in place of its placeholder in the measures and charts."""
    placeholders = {dataset.key: dataset.placeholder for dataset in plan.datasets}
    ids: dict[int, int] = {}
    for key, dataset_id in made.items():
        if key not in placeholders:
            raise ValueError(f"the plan makes no dataset {key!r}")
        if type(dataset_id) is not int or dataset_id <= 0:
            raise ValueError("a made dataset's id is a positive integer")
        ids[placeholders[key]] = dataset_id

    def measure(m: Measure) -> Measure:
        return replace(m, dataset_id=ids.get(m.dataset_id, m.dataset_id))

    charts = tuple(replace(chart, dataset_id=ids.get(chart.dataset_id, chart.dataset_id),
                           measures=tuple(WrittenRef(measure(ref.measure)) if isinstance(ref, WrittenRef) else ref
                                          for ref in chart.measures))
                   for chart in plan.charts)
    return replace(plan, charts=charts, measures=tuple(measure(m) for m in plan.measures))


# A stored plan's own version, read back by `plan_from_json`.
_FORMAT = 1


def plan_to_json(plan: BuildPlan) -> dict:
    """The plan as the build store keeps it: plain JSON, and no key a value is kept under (`VALUE_SHAPED_KEYS`), so a
    filter's operand is kept as `operand`. `plan_from_json` reads it back."""
    def filter_doc(item: Filter) -> dict:
        return {"column": item.column, "op": item.op, "operand": item.value}

    def ref_doc(ref) -> dict:
        return {"saved": ref.name, "kind": ref.kind.value} if isinstance(ref, SavedRef) else {"written":
                                                                                              ref.measure.name}

    return {
        "format": _FORMAT,
        "measures": [{"name": m.name, "label": m.label, "expression": m.expression, "kind": m.kind.value,
                      "dataset_id": m.dataset_id, "description": m.description, "asked_by": m.asked_by,
                      "currency": m.currency} for m in plan.measures],
        "charts": [{"key": c.key, "family": c.family.value, "title": c.title, "dataset_id": c.dataset_id,
                    "measures": [ref_doc(ref) for ref in c.measures], "dimensions": list(c.dimensions),
                    "time_column": c.time_column, "time_grain": c.time_grain,
                    "filters": [filter_doc(f) for f in c.filters], "row_limit": c.row_limit, "source": c.source,
                    "target": c.target} for c in plan.charts],
        "datasets": [{"key": d.key, "placeholder": d.placeholder, "table": d.table, "join": d.join,
                      "table_name": d.table_name, "sql": d.sql, "name": d.name} for d in plan.datasets],
        "reuse": {"candidates": list(plan.reuse_candidates), "reason": plan.reuse_reason},
    }


def plan_from_json(doc: Mapping[str, Any]) -> BuildPlan:
    """The plan `plan_to_json` kept. Raises ValueError for a document it did not write."""
    if not isinstance(doc, Mapping) or doc.get("format") != _FORMAT:
        raise ValueError("not a plan this version of mnemiq keeps")
    try:
        measures = tuple(Measure(name=m["name"], label=m["label"], expression=m["expression"],
                                 kind=MeasureKind(m["kind"]), dataset_id=m["dataset_id"],
                                 description=m["description"], asked_by=m["asked_by"], currency=m["currency"])
                         for m in doc["measures"])
        by_name = {m.name: m for m in measures}

        def ref(item: Mapping):
            if "saved" in item:
                return SavedRef(item["saved"], MeasureKind(item["kind"]))
            return WrittenRef(by_name[item["written"]])

        charts = tuple(ChartSpec(key=c["key"], family=Family(c["family"]), title=c["title"],
                                 dataset_id=c["dataset_id"], measures=tuple(ref(r) for r in c["measures"]),
                                 dimensions=tuple(c["dimensions"]), time_column=c["time_column"],
                                 time_grain=c["time_grain"],
                                 filters=tuple(Filter(f["column"], f["op"], f["operand"]) for f in c["filters"]),
                                 row_limit=c["row_limit"], source=c["source"], target=c["target"])
                       for c in doc["charts"])
        datasets = tuple(PlannedDataset(**d) for d in doc["datasets"])
        reuse = doc["reuse"]
        return BuildPlan(charts=charts, measures=measures, reuse_candidates=tuple(reuse["candidates"]),
                         reuse_reason=reuse["reason"], datasets=datasets)
    except (KeyError, TypeError) as exc:
        raise ValueError("a kept plan is missing a part") from exc

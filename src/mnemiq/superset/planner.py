"""The plan step of a dashboard build: from the builder's goal to a plan whose every part mnemiq has checked.

A plan is the measures a dashboard shows, the charts that show them, the datasets mnemiq makes for them, and, where a
dashboard the builder can see already covers the goal, the offer to reuse it. Nothing is built here: the plan is handed
to the build, which runs it one step per request.

**What the model sees** (`PlanContext`) is what the builder may: the tables and columns the grants leave visible
(`visible_schema`), the saved metrics of those tables' datasets by name and expression, and the dashboards whose titles
share a word with the goal, by id and title. Never an editor's name (a saved metric's certifier, a dashboard's owners),
never Superset's error text, and no value in the plan's first prompt: nothing a plan is made from is a number computed
from the data. A filter's literal that no row the builder can read holds is refused, and only then does the model see
values: those of the column its own plan filtered on, read as the builder. A part refused goes back to the model with
mnemiq's own reason, which names what the model wrote and nothing Superset said, those values aside.

**What is checked.** Each written measure meets `check_measure`, its statement decided under the builder's grants as
every ask-path statement is, then the values it filters on (`absent_literals`), and then `reusable`: a saved metric on
its dataset that says the same thing replaces it. Each chart meets `plan_refusal` and the plan's own rules here (its
key, its columns, its filters, the saved metrics it names, no negated filter beside `IS NULL` on one column), then the
values its filters compare a column with, and then whether a column it filters with `!=`, `NOT IN` or `NOT LIKE` holds
NULL, which such a filter leaves out. A refused part goes back to the model once; refused again, it refuses the plan,
and the reason names it.

**A joined table's saved metric on a join** is carried onto it as its table defines it (`carry`), once for each
metric, kind, table and join. It is refused where its table is on a side an outer join may fill with NULL, where the
fan-out guard is on and the join does not leave its table's rows at their own grain (`tables_off_their_grain`),
whatever it aggregates, and where its expression reads anything but its own table's columns. Otherwise its expression
is printed twice from one parse: each column qualified as the join names its table, decided by `check_measure` over
the join, and each column named as the joined dataset names it, which the chart sends. It is then a written measure
of the plan, under the plan's name for it and its metric's verbose name, but for its values, which are its table's
and never probed.

**Datasets.** A chart on a table with no dataset gets a planned dataset over the table; a measure that needs a join,
written or carried, gets a planned virtual dataset over the join's SQL, the join decided with the measure by
`check_measure`. The builder
makes each with the extension's `create_dataset`, a virtual one with its `sql`. A planned dataset has no id
until it is made, so the plan's charts and measures carry its negative `placeholder` until `with_made_datasets` puts
the made id in its place. Every dataset Superset makes carries a saved metric `count` (`COUNT_METRIC`), which is
Superset's, never one of mnemiq's measures.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.execute.runner import DEFAULT_TIMEOUT_S
from mnemiq.llm.client import ContextTooLong, PromptCut
from mnemiq.sql.fanout_check import OffGrain, guard_on, key_facts, tables_off_their_grain
from mnemiq.sql.identifiers import resolve_identifier
from mnemiq.sql.schema import column_types, visible_schema
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.chart_check import read_dataset_facts
from mnemiq.superset.chart_form import DatasetFacts, NotCheckable
from mnemiq.superset.charts import ChartSpec, Family, Filter, SavedRef, WrittenRef, plan_refusal
from mnemiq.superset.dashboard_charts import LeftOut, leave_out_shown, read_dashboard_charts
from mnemiq.superset.grants import UnreadableAnswer, read_datasets, table_bindings
from mnemiq.superset.literals import NEGATED_OPERATORS, Absent, Probes, absent_literals, filter_statement, holds_null
from mnemiq.superset.marker import format_marker
from mnemiq.superset.measures import MAX_LABEL, Measure, MeasureKind, MeasureRefused, check_measure, scales_by_100, \
    superset_text_problem
from mnemiq.superset.saved_metrics import Carried, SavedMetric, carry, read_saved_metrics, reusable, same_definition
from mnemiq.superset.settings import SUPERSET_DIALECT

__all__ = ["COUNT_METRIC", "FILTER_OPERATORS", "GRAINS", "PLANNER_MAX_TOKENS", "PLAN_JSON_SHAPE", "SCALAR",
           "BuildPlan", "FakePlanner",
           "LLMPlanner", "PlanColumn", "PlanContext", "PlanDashboard", "PlanMetric", "PlanRefused", "PlanTable",
           "PlannedDataset", "Planner", "RefusedPart", "Repair", "draft_of", "plan_build",
           "plan_from_draft", "plan_from_json", "plan_prompt", "plan_to_json", "with_made_datasets"]

logger = logging.getLogger(__name__)

# The saved metric every dataset Superset makes carries: `CreateDatasetCommand.run` calls `fetch_metadata`, which adds
# the engine spec's metrics (`BaseEngineSpec.get_metrics` in `superset/db_engine_specs/base.py`), for a physical and a
# virtual dataset alike.
COUNT_METRIC = ("count", "COUNT(*)")

# The families whose x-axis is the chart's time column when it has one. rc2's time-series query orders by the first
# measure (`normalizeOrderBy`) before its LIMIT and the axis sorts by time afterwards, so a row limit keeps the largest
# points, not the latest: such a chart takes no row limit.
_TIME_AXIS_FAMILIES = frozenset({Family.LINE, Family.BAR, Family.AREA, Family.SCATTER, Family.MIXED})
_TIME_AXIS_ROW_LIMIT = ("a chart on a time axis takes no row_limit: a row limit keeps the points its query ranks "
                        "first, the largest by its first measure, not the latest; to show a recent span, filter its "
                        "time column with >= a start date instead")

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
MAX_DESCRIPTION = 500
# Postgres keeps the first 63 bytes of a name (`NAMEDATALEN` - 1): a longer alias would be cut, and two could meet.
MAX_IDENTIFIER = 63

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
# A planning prompt the model's server refuses, or cuts, as longer than its window: the goal and the schema it is
# planned on are too large for one prompt, so the plan is refused, in mnemiq's words. The provider's own words go to
# the log only. A model that does not answer at all is an outage, which the plan step leaves to its caller.
PLAN_TOO_LONG = ("No plan was made: the goal and what this builder can see (the tables, their columns and saved "
                 "metrics) are too large to plan in one prompt, and the model server refused the planning prompt as "
                 "longer than its context window. Raising the server's window (vLLM: --max-model-len) up to the "
                 "model's own limit lets it plan on a schema this large.")
PLAN_CUT = ("No plan was made: the goal and what this builder can see (the tables, their columns and saved metrics) "
            "are too large to plan in one prompt, and the model server cut the planning prompt to fit its context "
            "window, so the model never saw all of it. Raising the server's window (vLLM: --max-model-len) up to the "
            "model's own limit lets it plan on a schema this large.")


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
    the dashboard offered for reuse, and the datasets the build makes. A plan that offers a dashboard is never built:
    the offer says the dashboard already covers the goal, so reusing it is opening it. Asked again with that
    dashboard's id, the build adds this goal's planned charts to it, and asked for a fresh one, it makes a new
    dashboard; either way the plan is made again, with no dashboard offered."""

    charts: tuple[ChartSpec, ...]
    measures: tuple[Measure, ...]
    reuse_candidates: tuple[int, ...] = ()
    reuse_reason: str | None = None
    datasets: tuple[PlannedDataset, ...] = ()
    # Built onto a dashboard: each planned chart the dashboard already shows, left out of `charts`, with the chart that
    # shows it (`dashboard_charts.leave_out_shown`). Answered with the plan, never kept with the build.
    left_out: tuple[LeftOut, ...] = ()

    @property
    def awaiting_choice(self) -> bool:
        """Whether the plan offers a dashboard that already covers the goal, so nothing is built from it: the builder
        opens that dashboard, or asks again (on it, or for a fresh one)."""
        return bool(self.reuse_candidates)


@dataclass(frozen=True)
class RefusedPart:
    """A part of a plan mnemiq refused, as the model named it (`measure 'x'`, `chart 'y'`, `the join '...'`), with
    mnemiq's own reason."""

    part: str
    reason: str
    # False where no rewrite of the part would pass (`MeasureRefused.repairable`): the plan stops there.
    repairable: bool = True


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
    # Built onto a dashboard: the measures it keeps that this builder may be told of (`_readable_kept`), whose names
    # and labels a written measure leaves to them.
    kept: tuple[Measure, ...] = ()


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
            "saved_metric": {"type": "string"}, "description": {"type": "string"},
            "currency": {"type": "string"}},
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
aggregate expression in PostgreSQL over the measure's rows, "kind": one of currency, percent, count, number \
(a percent is a fraction from 0 to 1, which Superset shows as a percent: 0.25 shows as 25%; never multiply it by \
100), "table": a table's name, or "join": "a JOIN b ON ...", "description": optional, "currency": an ISO 4217 \
code, optional}; or
- a saved metric of a table's dataset: {"name": the plan's own name for it, "saved_metric": the saved metric's name, \
"kind": ..., "table": ...}, or, on a join, {"name", "saved_metric", "kind", "table": the joined table whose saved \
metric it is, "join": "a JOIN b ON ..."}. The same saved metric on two tables, or on two joins, is two measures, each \
with a name of its own.
Use a saved metric where one says what a measure needs, on a join too: never write your own measure for what a joined \
table's saved metric says. Write a join only where the charts need columns of more than one table. A join is tables \
joined with JOIN, LEFT JOIN, RIGHT JOIN or FULL JOIN, each ON a condition on their columns, each table named alone or \
with an alias. On a join, a written measure names each column alone (amount, not claim.amount), and only a column one \
joined table alone has. A column two joined tables share is in the joined dataset as <table>_<column> \
(claim_created_at), for dimensions, filters, time columns, source and target only. A saved metric goes onto a join as \
its table defines it, mnemiq naming its columns as the join names them, and only where the join repeats none of its \
table's rows: its table joined to tables with at most one row for each of its rows, with JOIN (which leaves out its \
rows no other table matches) or on the side a LEFT or RIGHT JOIN keeps every row of (the left of a LEFT JOIN, the \
right of a RIGHT JOIN), never in a FULL JOIN; and only a saved metric that names a column of its own table, so never \
count (COUNT(*)). A written measure's name and label are never a saved metric's name on its table, and no two \
measures share a label. A table with no dataset gets one, with Superset's saved metric count (COUNT(*)).

Charts: {"key": lower case letters, digits and _, at most 40, one per chart, "family": one of FAMILIES, "title", \
"measures": [measure names], "dimensions": [columns], "time_column": a date or timestamp column, or omitted, \
"grain": one of GRAINS or omitted, "filters": [{"column", "op": one of =, !=, >, <, >=, <=, LIKE, ILIKE, NOT LIKE, IN, \
NOT IN, IS NULL, IS NOT NULL, "operand": one value, a list for IN and NOT IN, omitted for IS NULL and IS NOT NULL}], \
"row_limit": optional, never on a chart with a time_column or a date or timestamp first dimension, "source" and \
"target": a sankey's two columns, "table" or "join": the chart's rows, needed only when it has no measures}. \
A chart's measures, dimensions and filters are all on one table or one join.
- pie, big_number, gauge, treemap, waterfall, funnel and sankey plot exactly one measure; mixed plots two or more on a \
time column, the first on its primary axis;
- pie, treemap, funnel, histogram, pivot and table need a dimension; waterfall and box_plot need a dimension or a time \
column;
- a histogram takes no measure: its first dimension is the column whose spread it shows;
- sankey, box_plot and histogram cannot show currency or percent; handlebars shows count or number only;
- one axis shows one kind of measure.

A condition that says which rows a written measure covers (open, won, overdue, active) goes inside its expression, \
as CASE WHEN ... THEN ... END with no ELSE on that condition, so a row it leaves out adds nothing: an ELSE 0 there is \
counted by COUNT and averaged in by AVG. A share keeps its own 0 for the rows it covers: a win rate among closed deals \
is AVG(CASE WHEN closed THEN CASE WHEN won THEN 1.0 ELSE 0 END END), closed and won standing for their conditions. The \
measure then means its name on every chart and wherever it is saved; a chart's filters narrow only what that chart \
shows. A negated condition (!=, NOT IN, NOT LIKE) also leaves out the rows \
where its column is NULL: where those rows belong to the measure, keep them with COALESCE or IS NULL OR. Where a chart \
should not show the groups a measure's condition leaves out (a closed stage, shown empty), filter them out on that \
chart as well.

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
    if context.kept:
        # Only measures on datasets this builder reads whole (`_readable_kept`); a table's own dataset is named.
        tables = {table.dataset_id: table.name for table in context.tables if table.dataset_id is not None}
        kept = [{"name": m.name, "label": m.label, "expression": m.expression, "kind": m.kind.value,
                 **({"table": tables[m.dataset_id]} if m.dataset_id in tables else {})} for m in context.kept]
        user += (f"\n\nThe dashboard this builds onto: this dashboard already keeps these measures, each on its own "
                 f"dataset:\n{json.dumps(kept, ensure_ascii=False)}\nA written measure's name and label are never one "
                 "of theirs, unless it is that measure: the same expression and kind, on its table.")
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


def _join_tables(text: str) -> set[str] | None:
    """The tables a FROM clause joins, each name as the dialect resolves it, when `text` is one SQL join (a FROM with at
    least one JOIN); None when it is not (a bare table name among them)."""
    try:
        tree = sqlglot.parse_one(f"SELECT 1 FROM {text}", read=SUPERSET_DIALECT)
    except SqlglotError:
        return None
    if not isinstance(tree, exp.Select) or tree.args.get("from_") is None or not tree.args.get("joins"):
        return None
    sources = [tree.args["from_"].this, *(join.this for join in tree.args["joins"])]
    if not all(isinstance(source, exp.Table) for source in sources):
        return None
    return {resolve_identifier(source.this, SUPERSET_DIALECT) for source in sources}


def _on_the_join(chart: Mapping[str, Any], table: str, joined: set[str],
                 tables: Mapping[str, "PlanTable"]) -> dict | None:
    """`chart`, which names `table`, moved onto a join of `joined` tables: its columns renamed as the join's dataset
    names `table`'s columns (`_joined`: a column one joined table alone has keeps its name, one two share is
    `<table>_<column>`), so each still reads the column of `table` the chart named. None, and the chart stays where it
    was, when `table` is not one of the join's, a joined table is not one the builder can read, or a column the chart
    names is not `table`'s."""
    if table not in joined or not joined <= tables.keys():
        return None
    having: dict[str, int] = {}
    for name in joined:
        for column in tables[name].columns:
            having[column.name] = having.get(column.name, 0) + 1
    own = {column.name for column in tables[table].columns}

    def shown(column: object) -> str | None:
        if not isinstance(column, str) or column not in own:
            return None
        return column if having[column] == 1 else f"{table}_{column}"

    moved = {key: value for key, value in chart.items() if key != "table"}
    dimensions = chart.get("dimensions", [])
    if isinstance(dimensions, list):
        moved["dimensions"] = [shown(d) for d in dimensions]
        if None in moved["dimensions"]:
            return None
    for key in ("time_column", "source", "target"):
        if chart.get(key) is not None:
            moved[key] = shown(chart[key])
            if moved[key] is None:
                return None
    if isinstance(chart.get("filters"), list):
        moved["filters"] = []
        for each in chart["filters"]:
            if not isinstance(each, dict) or shown(each.get("column")) is None:
                return None
            moved["filters"].append({**each, "column": shown(each["column"])})
    return moved


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
        # A join written where a table goes (Qwen3.8-27B wrote one): text that names no table and is one SQL join is
        # read as the join it is.
        if join is None and isinstance(table, str) and table not in tables and _join_tables(table) is not None:
            table, join = None, table
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
        # A saved metric named with both its table and a join is that table's, carried onto the join: its rows are the
        # join's, and its table one the builder can read. Which of the join's tables it is, the plan check reads.
        carried = "saved_metric" in item and item.get("table") is not None and item.get("join") is not None
        table = item.get("table") if carried else None
        dataset = dataset_of({"join": item.get("join")} if carried else item, part)
        if carried and (not isinstance(table, str) or table not in tables):
            refuse(part, f"{table!r} is not a table this builder can read")
            table = None
        if "saved_metric" in item:
            metric = _text(item.get("saved_metric"))
            if metric is None:
                refuse(part, "its saved_metric is the name of a saved metric on its table")
            elif kind is not None and dataset is not None and (table is not None or not carried):
                saved[name] = (SavedRef(metric, kind, table=table, handle=name) if carried else SavedRef(metric, kind),
                               dataset)
            continue
        label, expression = item.get("label"), item.get("expression")
        description, currency = item.get("description", ""), item.get("currency")
        if not isinstance(label, str) or not isinstance(expression, str) or not expression.strip():
            refuse(part, "a written measure has a label and an expression; a saved one names its saved_metric")
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
        # A chart's rows are its measures'. One that names a table its measures' join joins (Qwen3.8-27B named
        # `orders` for a measure over `order_items JOIN orders`) is planned on that join, as one that names none is;
        # the check reads its columns there.
        theirs = next(iter(of_refs)) if len(of_refs) == 1 else None
        joined = next((d for d in planned.values() if d.placeholder == theirs and d.join), None)
        named = item.get("table")
        moved = None
        if item.get("join") is None and isinstance(named, str) and named in tables and joined is not None:
            moved = _on_the_join(item, named, _join_tables(joined.join) or set(), tables)
        if moved is not None:
            own, item = theirs, moved
        else:
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
    # A saved metric is a measure of the plan once for each table and kind it is plotted with, under a handle of its
    # own: the metric's name where that is free, else the metric's name and its table's. A joined table's metric,
    # carried onto a join, is named with its table and the join, under the plan's own name for it.
    taken = {measure.name for measure in plan.measures}
    handles: dict[tuple[str, MeasureKind, int, str | None, str | None], str] = {}

    def slug(text: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", text.casefold()).strip("_")

    def handle(ref: SavedRef, dataset_id: int) -> str:
        key = (ref.name, ref.kind, dataset_id, ref.table, ref.handle)
        if key not in handles:
            where = rows(dataset_id)
            if ref.table is not None:
                base = [ref.handle] if ref.handle else [ref.name, f"{ref.name}_{slug(ref.table)}"]
                where = {"table": ref.table, **({"join": where["join"]} if "join" in where else {})}
            else:
                base = [ref.name, f"{ref.name}_{slug(next(iter(where.values())) or '')}"]
            name, n = next((b for b in base if b not in taken), None), 2
            while name is None:
                name = None if f"{base[-1]}_{n}" in taken else f"{base[-1]}_{n}"
                n += 1
            taken.add(name)
            handles[key] = name
            measures.append({"name": name, "saved_metric": ref.name, "kind": ref.kind.value, **where})
        return handles[key]

    charts: list[dict] = []
    for chart in plan.charts:
        names = [handle(ref, chart.dataset_id) if isinstance(ref, SavedRef) else ref.measure.name
                 for ref in chart.measures]
        doc = {"key": chart.key, "family": _family_name(chart.family), "title": chart.title, "measures": names,
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
    """Each visible table's physical dataset on the configured database and schema (`read_datasets`): the first by id
    of those named exactly as the table is stored. rc2 reads a physical dataset FROM `table(self.table_name,
    schema=self.schema)` (`SqlaTable.get_sqla_table` in `superset/connectors/sqla/models.py`), which SQLAlchemy quotes
    when the name is not all lower case, so a dataset `Party` reads `"Party"`, never a stored `party`."""
    physical = [d for d in read_datasets(adapter)
                if d.database_id == adapter.database_id and d.schema == adapter.schema and not d.sql]
    found: dict[str, int] = {}
    for table, name in stored.items():
        named = [d.id for d in physical if d.table_name == name]
        if named:
            found[table] = min(named)
    return found


def _catalogue(client, adapter, snapshot: Snapshot, grants: GrantSet) -> _Catalogue | PlanRefused:
    visible = visible_schema(snapshot, grants)
    stored = {table: table for table in visible}
    for name, tables in table_bindings(snapshot).items():
        for table in tables:
            if table in visible:
                stored[table] = name
    try:
        datasets = _datasets_by_table(adapter, stored)
    except UnreadableAnswer:  # logged where it was found
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
    """A dataset a part of the plan is on: the columns a chart may name and their types, the source its measures are
    decided over, its saved metrics, how the model named it, and the planned dataset when the build makes it. On a
    join, `measured` holds the columns a measure may name, `owners` each column's table, and `shared` the names the
    dataset gives each column two joined tables share. `origins` gives each column a chart may name the snapshot's table
    and column it reads, as a chart filter's value is asked of the source (`absent_literals`), and `shown` the other way:
    each joined table's column, by table and column, to the name the dataset gives it. `qualifiers` gives each joined
    table its qualifier in the join (its alias, else its name, as written there, quoting and all), and `nullable` the
    joined tables an outer join may fill with NULL: in join order from the FROM's table, a LEFT JOIN's table, every
    table before a RIGHT JOIN, and every table up to a FULL JOIN's."""

    columns: frozenset[str]
    source: str
    saved: tuple[SavedMetric, ...]
    part: str
    types: Mapping[str, str | None]
    planned: PlannedDataset | None = None
    measured: frozenset[str] | None = None
    owners: Mapping[str, str] = field(default_factory=dict)
    shared: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    origins: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    shown: Mapping[tuple[str, str], str] = field(default_factory=dict)
    qualifiers: Mapping[str, exp.Identifier] = field(default_factory=dict)
    nullable: frozenset[str] = frozenset()


def _own_columns(table: str, columns: frozenset[str]) -> dict[str, tuple[str, str]]:
    """A table's columns, each read from itself: a physical dataset's `_Rows.origins`."""
    return {column: (table, column) for column in columns}


def _quoted(name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=SUPERSET_DIALECT)


def _made_count(dataset_id: int) -> SavedMetric:
    name, expression = COUNT_METRIC
    return SavedMetric(dataset_id=dataset_id, name=name, expression=expression, verbose_name="COUNT(*)",
                       d3format=None, currency=None, certified_by=None, warning_text=None)


def _virtual_name(key: str, tables: Sequence[str]) -> str:
    """A virtual dataset's name: its key, unique within the build, then its tables, cut to the 250 characters rc2's
    `table_name` holds. A `:` ends the key, which no key holds, so two keys never give one name, whatever the tables
    are called. The build adds its own id to make it unique across builds."""
    return f"mnemiq_{key}:{'_'.join(tables)}"[:MAX_TITLE]


# A join's own arguments mnemiq reads: its table, its ON condition, and LEFT, RIGHT or FULL (`side`) with INNER or
# OUTER (`kind`). NATURAL (`method`), USING and every other clause are refused.
_JOIN_ARGS = frozenset({"this", "on", "side", "kind"})
# How a join is parsed: as the FROM of a SELECT.
_JOIN_PREFIX = "SELECT 1 FROM "

# Postgres's time types a time column groups by: a date or a timestamp, never a time of day.
_TIME_OF_DAY = (exp.DataType.Type.TIME, exp.DataType.Type.TIMETZ)


def _temporal(data_type: str | None) -> bool:
    """Whether a column of `data_type`, as the snapshot records it, is a date or a timestamp. An unknown type is not."""
    if not data_type:
        return False
    try:
        built = exp.DataType.build(data_type, dialect=SUPERSET_DIALECT, udt=True)
    except (SqlglotError, ValueError):
        return False
    return built.is_type(*exp.DataType.TEMPORAL_TYPES) and not built.is_type(*_TIME_OF_DAY)


def joined_names(tables: Sequence[tuple[str, Collection[str]]]) -> dict[tuple[str, str], str]:
    """The name a joined dataset gives each (table, column), `tables` being each joined table and its columns in join
    order: a column only one table has keeps its name, and a column two or more tables share is `<table>_<column>`."""
    having: dict[str, int] = {}
    for _, columns in tables:
        for column in columns:
            having[column] = having.get(column, 0) + 1
    return {(table, column): column if having[column] == 1 else f"{table}_{column}"
            for table, columns in tables for column in columns}


def _joined(planned: PlannedDataset, catalogue: _Catalogue) -> _Rows | str:
    """The rows of a planned join, with the SELECT its virtual dataset is made from, or why the join is refused.

    The join is held to an allow-list: its SELECT is the only one in it, and its only tables are the FROM's and each
    JOIN's, so nothing in an ON condition (an EXISTS, an IN or a scalar subquery, a CTE) reads another table. Each table
    is named alone or with an alias (`_table_problem`), each join is a plain, LEFT, RIGHT or FULL JOIN ON a condition,
    and the conditions name only the joined tables' columns (`_condition_problem`). The dataset runs the join as sqlglot
    prints it, the text each measure over it was decided on.

    The dataset names a column one joined table alone has without its table, so a measure decided over the join names
    the same column on the dataset; a measure names only those. A column two joined tables share is in the dataset once
    for each, as `<table>_<column>`, for a chart's dimensions, filters, time column, source and target."""
    # The join is the virtual dataset's SQL, which Superset renders as a template.
    if (marker := next((d for d in JINJA_DELIMITERS if d in planned.join), None)) is not None:
        return f"it holds {marker}, which Superset reads as a template marker; write it without {marker}"
    try:
        tree = sqlglot.parse_one(f"{_JOIN_PREFIX}{planned.join}", read=SUPERSET_DIALECT)
    except SqlglotError:
        return "it is not one SQL join"
    if not isinstance(tree, exp.Select) or {key for key, value in tree.args.items() if value} - {"expressions",
                                                                                                 "from_", "joins"}:
        return "it is a join of tables and nothing else: no WHERE, GROUP BY or LIMIT"
    joins = tree.args.get("joins") or []
    for join in joins:
        if join.args.get("method"):
            return f"a {join.args['method']} JOIN is not one mnemiq reads: join each table ON a condition"
        if {key for key, value in join.args.items() if value} - _JOIN_ARGS \
                or (join.args.get("kind") or "INNER").upper() not in ("INNER", "OUTER"):
            return "it joins with JOIN, LEFT JOIN, RIGHT JOIN or FULL JOIN, each ON a condition"
    if not joins or any(join.args.get("on") is None for join in joins):
        return "it is a JOIN of two or more tables, each joined ON a condition"
    nodes = [tree.args["from_"].this, *(join.this for join in joins)]
    if len(list(tree.find_all(exp.Select))) != 1 or not all(isinstance(node, exp.Table) for node in nodes) \
            or {id(table) for table in tree.find_all(exp.Table)} != {id(node) for node in nodes}:
        return ("it joins tables, each ON a condition on their columns: no subquery, EXISTS, CTE or table function "
                "anywhere in it")
    names: list[str] = []
    qualifiers: dict[str | None, str] = {}
    qualifier_of: dict[str, exp.Identifier] = {}
    for node in nodes:
        if (problem := _table_problem(node)) is not None:
            return problem
        name = resolve_identifier(node.this, SUPERSET_DIALECT)
        if name not in catalogue.columns:
            return f"{node.name!r} is not a table this builder can read"
        if name in names:
            return f"it joins {node.name!r} to itself"
        names.append(name)
        alias = node.args.get("alias")
        qualifiers[resolve_identifier(alias.this if alias else node.this, SUPERSET_DIALECT)] = name
        qualifier_of[name] = (alias.this if alias else node.this).copy()
    if (problem := _condition_problem(tree, qualifiers, catalogue)) is not None:
        return problem
    # `names[index]` is the table the join at `index` (from 1) brings in, and `names[:index]` every table before it.
    nullable: set[str] = set()
    for index, join in enumerate(joins, 1):
        side = (join.args.get("side") or "").upper()
        if side == "LEFT":
            nullable.add(names[index])
        elif side == "RIGHT":
            nullable.update(names[:index])
        elif side == "FULL":
            nullable.update(names[:index + 1])
    having: dict[str, list[str]] = {}
    for name in names:
        for column in catalogue.columns[name]:
            having.setdefault(column, []).append(name)
    picked: list[exp.Expression] = []
    owners: dict[str, str] = {}
    origins: dict[str, tuple[str, str]] = {}
    shown_as: dict[tuple[str, str], str] = {}
    types: dict[str, str | None] = {}
    named = joined_names([(name, catalogue.columns[name]) for name in names])
    for node, name in zip(nodes, names):
        qualifier = node.args["alias"].this if node.args.get("alias") else node.this
        for column in sorted(catalogue.columns[name]):
            shown = named[(name, column)]
            if shown in owners:
                return f"the joined dataset would have two columns named {shown!r}"
            if len(shown.encode()) > MAX_IDENTIFIER:
                return f"the joined dataset's column {shown!r} is longer than the {MAX_IDENTIFIER} bytes of a name"
            owners[shown], types[shown] = name, catalogue.types[name].get(column)
            origins[shown] = (name, column)
            shown_as[(name, column)] = shown
            picked.append(exp.alias_(exp.Column(this=exp.to_identifier(column, quoted=True), table=qualifier.copy()),
                                     exp.to_identifier(shown, quoted=True)))
    shared = {column: tuple(f"{table}_{column}" for table in tables) for column, tables in having.items()
              if len(tables) > 1}
    # The dataset's SELECT is the parsed join's own tree with these columns, printed as the deciders print each
    # measure's statement over the same join text: the dataset runs the FROM that was decided, never the raw text.
    select = tree.copy()
    select.set("expressions", picked)
    made = PlannedDataset(key=planned.key, placeholder=planned.placeholder, join=planned.join,
                          sql=select.sql(dialect=SUPERSET_DIALECT), name=_virtual_name(planned.key, names))
    return _Rows(frozenset(owners), planned.join, (_made_count(planned.placeholder),), f"the join {planned.join!r}",
                 types, made, measured=frozenset(c for c, tables in having.items() if len(tables) == 1),
                 owners=owners, shared=shared, origins=origins, shown=shown_as, qualifiers=qualifier_of,
                 nullable=frozenset(nullable))


def _table_problem(node: exp.Table) -> str | None:
    """Why a joined table is not named alone, or with an alias alone: a function in its place, a schema, ONLY, a
    TABLESAMPLE (which would make the dataset a sample of the table), or an alias's column list."""
    if not isinstance(node.this, exp.Identifier):
        return f"it joins tables only, never a function such as {node.this.sql(dialect=SUPERSET_DIALECT)}"
    clauses = {key for key, value in node.args.items() if value} - {"this", "alias"}
    if clauses & {"db", "catalog"}:
        return "it names its tables without a schema"
    if "only" in clauses:
        return f"it names {node.name!r} without ONLY"
    if "sample" in clauses:
        return f"it reads {node.name!r} whole: a TABLESAMPLE would make the dataset a sample of it"
    if clauses:
        return f"it names {node.name!r} alone, with an alias at most"
    alias = node.args.get("alias")
    if alias is not None and {key for key, value in alias.args.items() if value} - {"this"}:
        return f"the alias of {node.name!r} takes no column list: the dataset names the table's own columns"
    return None


def _condition_problem(tree: exp.Select, qualifiers: Mapping[str | None, str], catalogue: _Catalogue) -> str | None:
    """Why a join's ON conditions name something other than a joined table's column: a column of a table not in the
    join, one no joined table has, or a `TABLE` read as a column (`= ANY (TABLE t)`, which sqlglot reads that way)."""
    for column in tree.find_all(exp.Column):
        written = column.sql(dialect=SUPERSET_DIALECT)
        name = resolve_identifier(column.this, SUPERSET_DIALECT)
        if column.name.casefold() == "table":
            return f"its conditions name {written}: a table is joined, never read in a condition"
        if column.args.get("db") or column.args.get("catalog"):
            return f"its conditions name {written} with a schema; name a joined table's column"
        if column.table:
            owner = qualifiers.get(resolve_identifier(column.args.get("table"), SUPERSET_DIALECT))
            if owner is None:
                return f"its conditions name {written}, and {column.table!r} is no table of the join"
            if name not in catalogue.columns[owner]:
                return f"its conditions name {written}, which is no column of {owner!r}"
        elif not any(name in catalogue.columns[owner] for owner in qualifiers.values()):
            return f"its conditions name {column.name!r}, which is no column of a joined table"
    return None


def _expression_columns(expression: str) -> list[exp.Column] | None:
    """The columns `expression` names, or None when it does not parse (`check_measure` says why)."""
    try:
        return list(sqlglot.parse_one(f"SELECT {expression}", read=SUPERSET_DIALECT).find_all(exp.Column))
    except SqlglotError:
        return None


def _column_problem(columns: list[exp.Column], rows: _Rows) -> str | None:
    """Why a measure's expression cannot run on a joined dataset as it was decided over the join: a column named with
    its table, or one the dataset does not name as the join does."""
    for column in columns:
        if column.table:
            return (f"name {column.sql(dialect=SUPERSET_DIALECT)} alone, as {column.name}: the joined dataset names "
                    "its columns without their tables")
        name = resolve_identifier(column.this, SUPERSET_DIALECT)
        if name in rows.shared:
            return (f"{column.name!r} is in more than one of the joined tables, so a measure cannot name it; a measure "
                    "names only a column one joined table alone has")
        if name not in rows.measured:
            return (f"{column.name!r} is not a column a measure on the join may name: only a column one joined table "
                    "alone has, named alone")
    return None


def _scaled_saved(name: str) -> str:
    return (f"the saved metric {name!r} is on a 0 to 100 scale, so it cannot be shown as a percent: plan it with kind "
            "number, or write the fraction")


# Where a join's tables are named by aliases that differ only in letter case, which the own-grain reading cannot tell
# apart (`tables_off_their_grain` reads no such join), the refusal says so.
_FOLDED_ALIASES = "; give the joined tables aliases that differ by more than letter case"
_ISO_CODE = re.compile(r"[A-Z]{3}")


def _carried(ref: SavedRef, on: _Rows, catalogue: _Catalogue,
             off_grain: Mapping[str, OffGrain] | None) -> tuple[SavedMetric, Carried] | str:
    """`ref`, a saved metric of `ref.table`, one of the join `on`'s tables, carried onto the join as its table defines
    it (`carry`), or why it cannot be: the table has no such metric the builder can reach; it is on a side an outer
    join may fill with NULL, where the metric is not defined; the join does not leave its rows at their own grain
    (`off_grain`, the tables `tables_off_their_grain` names, given only when the fan-out guard is on); the metric is
    on a 0 to 100 scale and planned as a percent; or `carry` refuses its expression."""
    table = ref.table
    metric = next((m for m in catalogue.saved.get(catalogue.datasets.get(table), ()) if m.name == ref.name), None)
    if metric is None:
        return f"{table} has no saved metric named {ref.name!r} this builder can reach"
    if table in on.nullable:
        return (f"{table} is on a side the join may fill with NULL, where its saved metric {ref.name!r} is not "
                f"defined: put {table} on the side the join keeps every row of, or join with JOIN")
    if off_grain is not None and (off := off_grain.get(table)) is not None:
        # Folded as the own-grain reading folds an alias, to lower case.
        folded = len({qualifier.name.lower() for qualifier in on.qualifiers.values()}) < len(on.qualifiers)
        # "may repeat" where mnemiq cannot tell (`OffGrain.repeats`): it never says the join does what it cannot read.
        repeats = "repeats" if off.repeats else "may repeat"
        return (f"the saved metric {ref.name!r} is defined on {table}'s own rows, and this join {repeats} them: "
                f"{off.reason}; join {table} only to tables it meets at most once per row, or plan the chart on "
                f"{table} alone{_FOLDED_ALIASES if folded else ''}")
    if ref.kind is MeasureKind.PERCENT and scales_by_100(metric.expression):
        return _scaled_saved(ref.name)
    carried = carry(metric, table=table, stored=catalogue.stored[table], columns=catalogue.columns[table],
                    qualifier=on.qualifiers[table],
                    shown={column: on.shown[(table, column)] for column in catalogue.columns[table]})
    return carried if isinstance(carried, str) else (metric, carried)


def _carried_label(metric: SavedMetric) -> str:
    """A carried metric's label: its verbose name where that is one Superset keeps as written, else its name."""
    text = metric.verbose_name
    if isinstance(text, str) and text.strip() and len(text) <= MAX_LABEL and not any(
            marker in text for marker in JINJA_DELIMITERS) and superset_text_problem(text, "a label") is None:
        return text
    return metric.name


def _carried_currency(metric: SavedMetric) -> str | None:
    """A carried metric's currency: its own symbol where that is an ISO 4217 code, else none."""
    symbol = metric.currency.get("symbol") if isinstance(metric.currency, dict) else None
    return symbol if isinstance(symbol, str) and _ISO_CODE.fullmatch(symbol) else None


def _column_refusal(column: str, rows: _Rows) -> str:
    if column in rows.shared:
        return (f"{column!r} is in more than one of the joined tables; the joined dataset names it "
                f"{' or '.join(rows.shared[column])}")
    return f"{column!r} is not a column of {rows.part}"


def _filter_problem(item: Filter) -> str | None:
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
    if any(isinstance(x, float) and not math.isfinite(x) for x in items):
        return "an operand is a finite number: JSON carries no NaN or Infinity"
    if any(isinstance(x, str) and len(x) > _MAX_OPERAND for x in items):
        return f"an operand is at most {_MAX_OPERAND} characters"
    return None


def _chart_columns(chart: ChartSpec) -> list[str]:
    return [*chart.dimensions, *(c for c in (chart.time_column, chart.source, chart.target) if c)]


def _absent_reason(absent: Absent) -> str:
    """Why a part that filters on a value no row the builder can read holds is refused, naming the values its column
    holds where the source said. Each value is quoted (`repr`), so one holding ", " never reads as two, and one holding
    a line break never starts a line of the repair's prompt that reads as another part's refusal. A named value that
    differs from the plan's only in letter case (`casefold`, so 'STRASSE' matches 'straße') is called out, one clause
    each in the order named, among the named values only: no further read is made. The reason ends with what to do: a
    value the column holds replaces the plan's only where it is the value the goal means, and the plan's own value is
    kept only where the goal itself names it. A value swapped in, or the filter left out, would build a chart of a
    different question that still checks; kept, a goal value the builder cannot read is refused again, and the plan is
    refused naming what the builder can see."""
    literal = absent.literal
    keep = f"keep {literal!r} only if the goal itself names it"
    if absent.held is None:
        held = f"the source did not say which values it holds; {keep}"
    elif not absent.held:
        held = f"no row the builder can read holds any value of {absent.column}; {keep}"
    else:
        folded = literal.casefold()
        case = "".join(f"; {value!r} differs from {literal!r} only in letter case"
                       for value in absent.held if value != literal and value.casefold() == folded)
        held = (f"{absent.column} holds: {', '.join(repr(value) for value in absent.held)}"
                f"{' and more' if absent.more else ''}{case}; filter on one of these if it is the value the goal "
                f"means, and {keep}")
    return f"it filters {absent.column} on {literal!r}, which no row the builder can read holds; {held}"


def _filter_refusals(chart: ChartSpec, on: _Rows, *, adapter, snapshot: Snapshot, grants: GrantSet,
                     probes: Probes) -> list[str]:
    """The reasons `chart` is refused for a filter on a value no row the builder can read holds: each filter that
    compares a column with strings by `=`, `!=`, `IN` or `NOT IN`, asked of the table and column the chart's rows read
    it from (`_Rows.origins`), as the builder. A negated filter on such a value excludes nothing, so it is refused as
    the others are."""
    reasons: dict[str, None] = {}
    for item in chart.filters:
        origin = on.origins.get(item.column)
        sql = filter_statement(*origin, item.op, item.value, adapter.dialect) if origin is not None else None
        if sql is None:
            continue
        for absent in absent_literals(sql, adapter=adapter, snapshot=snapshot, grants=grants,
                                      timeout_s=DEFAULT_TIMEOUT_S, probes=probes):
            reasons.setdefault(_absent_reason(absent))
    return list(reasons)


def _null_contradictions(chart: ChartSpec) -> list[str]:
    """The reasons `chart` is refused for a negated filter (`!=`, `NOT IN`, `NOT LIKE`) beside `IS NULL` on the same
    column: a chart's filters all apply, and a NULL never meets a negation, so no row meets both. Decided from the plan
    alone, among the chart's own checks, so nothing is asked of the source for it."""
    nulls = {item.column for item in chart.filters if item.op == "IS NULL"}
    reasons: dict[str, None] = {}
    for item in chart.filters:
        if item.op in NEGATED_OPERATORS and item.column in nulls:
            reasons.setdefault(f"it filters {item.column} with both {item.op} and IS NULL, and no row can meet both; "
                               f"keep IS NULL alone if the rows where {item.column} is NULL are meant, otherwise drop "
                               "it")
    return list(reasons)


def _null_refusals(chart: ChartSpec, on: _Rows, *, adapter, snapshot: Snapshot, grants: GrantSet,
                   probes: Probes) -> list[str]:
    """The reasons `chart` is refused for a negated filter (`!=`, `NOT IN`, `NOT LIKE`) on a column a row the builder
    can read holds NULL in: SQL reads NULL compared with anything as unknown, so the filter leaves those rows out
    beside the ones it names, which the goal may not mean. A chart that also filters the column `IS NOT NULL` has said
    those rows are left out, and is not asked; one that filters it `IS NULL` beside the negation never reaches here
    (`_null_contradictions`). Asked of the table and column the chart's rows read it from (`_Rows.origins`), as the
    builder (`holds_null`); a column the source says nothing of refuses nothing."""
    stated = {item.column for item in chart.filters if item.op == "IS NOT NULL"}
    reasons: dict[str, None] = {}
    for item in chart.filters:
        if item.op not in NEGATED_OPERATORS or item.column in stated:
            continue
        origin = on.origins.get(item.column)
        if origin is not None and holds_null(*origin, adapter=adapter, snapshot=snapshot, grants=grants,
                                             timeout_s=DEFAULT_TIMEOUT_S, probes=probes):
            reasons.setdefault(f"it filters {item.column} with {item.op}, which leaves out every row where "
                               f"{item.column} is NULL, and rows the builder can read hold NULL there; replace the "
                               f"filter with IS NULL if only those rows are meant, add {item.column} IS NOT NULL if "
                               "leaving them out is meant, or filter on another column")
    return list(reasons)


def _check(plan: BuildPlan, catalogue: _Catalogue, context: PlanContext, *, adapter, snapshot: Snapshot,
           grants: GrantSet, settings, interactive_pivot_available: bool,
           on_dashboard: Sequence[Measure] = ()) -> BuildPlan | tuple[RefusedPart, ...]:
    """The plan with its reusable measures replaced by the saved metrics that say the same, and each joined table's
    saved metric carried onto its join as a measure written from its expression, or every part refused, each with
    mnemiq's own reason."""
    refused: list[RefusedPart] = []
    # Every probe of a filter's value this check makes, the measures' and the charts', shares one budget and one memo.
    probes = Probes()

    def refuse(part: str, reason: str, *, repairable: bool = True) -> None:
        refused.append(RefusedPart(part, reason, repairable))

    if not plan.charts:
        refuse("the plan", "it has no chart")
    if len(plan.charts) > MAX_CHARTS:
        refuse("the plan", f"it has {len(plan.charts)} charts; a plan has at most {MAX_CHARTS}")

    # Only what a chart plots is built: a measure or a dataset no chart uses is left out.
    plotted = {ref.measure.name for chart in plan.charts for ref in chart.measures if isinstance(ref, WrittenRef)}
    measures = [m for m in plan.measures if m.name in plotted]
    # Each joined table's saved metric a chart carries onto its join: one measure for each metric, kind, table and
    # join, decided against the columns of every chart that plots it.
    to_carry: dict[tuple[str, MeasureKind, str, int], SavedRef] = {}
    carried_columns: dict[tuple[str, MeasureKind, str, int], set[str]] = {}
    for chart in plan.charts:
        for ref in chart.measures:
            if isinstance(ref, SavedRef) and ref.table is not None:
                key = (ref.name, ref.kind, ref.table, chart.dataset_id)
                to_carry.setdefault(key, ref)
                carried_columns.setdefault(key, set()).update(_chart_columns(chart))
    if len(measures) + len(to_carry) > MAX_MEASURES:
        refuse("the plan", f"it writes {len(measures) + len(to_carry)} measures; a plan writes at most "
                           f"{MAX_MEASURES}")
    used = {m.dataset_id for m in measures} | {chart.dataset_id for chart in plan.charts}

    # The datasets parts can be on: each visible table's existing one, and each planned one the plan uses.
    by_dataset = {dataset: table for table, dataset in catalogue.datasets.items()}
    rows: dict[int, _Rows] = {
        dataset: _Rows(catalogue.columns[table], _quoted(catalogue.stored[table]), catalogue.saved.get(dataset, ()),
                       f"the table {table!r}", catalogue.types[table],
                       origins=_own_columns(table, catalogue.columns[table]))
        for dataset, table in by_dataset.items()}
    keys: set[str] = set()
    planned_tables: set[str] = set()
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
            if not any(m.dataset_id == planned.placeholder for m in measures) \
                    and not any(key[3] == planned.placeholder for key in to_carry):
                refuse(part, "a joined dataset is made only for a written measure, or a joined table's saved metric, "
                             "that needs the join")
                continue
            rows[planned.placeholder] = joined
        elif planned.table not in catalogue.columns:
            refuse(part, "it is not a table this builder can read")
        elif planned.table in catalogue.datasets:
            refuse(part, "the table already has a dataset; plan on it")
        elif planned.table in planned_tables:
            refuse(part, "the plan already makes a dataset on this table; plan both on that one")
        else:
            planned_tables.add(planned.table)
            # Built from what the check read, never from what the planner wrote beside it.
            made = PlannedDataset(key=planned.key, placeholder=planned.placeholder, table=planned.table,
                                  table_name=catalogue.stored[planned.table])
            rows[planned.placeholder] = _Rows(catalogue.columns[planned.table],
                                              _quoted(catalogue.stored[planned.table]),
                                              (_made_count(planned.placeholder),), part,
                                              catalogue.types[planned.table], made,
                                              origins=_own_columns(planned.table, catalogue.columns[planned.table]))

    # Each written measure: decided, then reused where a saved metric says the same, else held apart from every name
    # its chart would show beside it.
    names: dict[str, str] = {}
    labels: dict[str, str] = {}
    # The dashboard's own measures, when the plan builds onto one, every one of them, those the builder may not be
    # told of among them: one definition for each name, one measure for each label (`same_definition`: a dataset is
    # part of a measure). A refusal they cause says nothing of them.
    kept_names = {m.name.casefold(): m for m in on_dashboard}
    kept_labels = {m.label.strip().casefold(): m for m in on_dashboard}
    reused: dict[str, SavedRef] = {}
    kept: list[Measure] = []
    columns_of: dict[str, set[str]] = {}
    # The joined tables each planned join's measures and charts read, so a join only one of them is read from is
    # refused: a virtual dataset is made only where a join is needed.
    reads: dict[int, set[str]] = {}
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
        held = kept_names.get(measure.name.casefold())
        if held is not None and not same_definition(held, measure):
            # Its definition is not said: its dataset may be no table this builder can read.
            refuse(part, "this dashboard keeps a measure of this name with another definition, and keeps one "
                         "definition for each name: give this measure a name and a label this dashboard does not keep")
            continue
        other = kept_labels.get(measure.label.strip().casefold())
        if other is not None and other.name.casefold() != measure.name.casefold():
            refuse(part, f"its label {measure.label!r} is one this dashboard keeps on another measure: give this "
                         "measure a label this dashboard does not keep")
            continue
        if len(measure.description) > MAX_DESCRIPTION:
            refuse(part, f"its description is at most {MAX_DESCRIPTION} characters")
            continue
        on = rows.get(measure.dataset_id)
        if on is None:
            refuse(part, "its rows are not a table this builder can read, or a join the plan makes")
            continue
        if on.measured is not None:
            named = _expression_columns(measure.expression) or []
            if (problem := _column_problem(named, on)) is not None:
                refuse(part, problem)
                continue
            reads.setdefault(measure.dataset_id, set()).update(
                on.owners[n] for c in named if (n := resolve_identifier(c.this, SUPERSET_DIALECT)) in on.owners)
        verdict = check_measure(measure, source=on.source, dimension_names=tuple(sorted(columns_of[measure.name])),
                                adapter=adapter, snapshot=snapshot, grants=grants, guard_fanout=settings.guard_fanout)
        if isinstance(verdict, MeasureRefused):
            refuse(part, verdict.reason, repairable=verdict.repairable)
            continue
        # Both checks, so a measure refused for both is told both in its one repair.
        scaled = measure.kind is MeasureKind.PERCENT and scales_by_100(measure.expression)
        if scaled:
            refuse(part, "a percent measure is a fraction from 0 to 1, which Superset shows as a percent: write it "
                         "without multiplying by 100")
        absent = absent_literals(verdict.asked_sql, adapter=adapter, snapshot=snapshot, grants=grants,
                                 timeout_s=DEFAULT_TIMEOUT_S, probes=probes) if verdict.asked_sql else []
        for reason in dict.fromkeys(_absent_reason(item) for item in absent):
            refuse(part, reason)
        if scaled or absent:
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

    # Each joined table's saved metric on a join: carried as its table defines it (`_carried`), held apart from every
    # name and label of the plan, decided over the join as a written measure is (its values are its table's, so none is
    # probed), and kept as the measure its charts write, its expression naming the joined dataset's columns.
    carried: dict[tuple[str, MeasureKind, str, int], WrittenRef] = {}
    guarded = bool(to_carry) and guard_on(settings.guard_fanout, snapshot)
    grains: dict[int, Mapping[str, str]] = {}
    for key, ref in to_carry.items():
        # A reference with no handle, which only a planner other than the model's channel makes, goes by its metric's
        # name.
        name, placeholder = ref.handle or ref.name, key[3]
        part = f"measure {name!r}"
        on = rows.get(placeholder)
        if on is None:  # the chart's own check refuses its rows
            continue
        if on.measured is None:
            refuse(part, f"a joined table's saved metric goes onto a join only, and its chart's rows are {on.part}")
            continue
        if ref.table not in on.qualifiers:
            refuse(part, f"{ref.table} is not one of the tables of {on.part}")
            continue
        reads.setdefault(placeholder, set()).add(ref.table)
        if guarded and placeholder not in grains:
            grains[placeholder] = tables_off_their_grain(
                sqlglot.parse_one(f"{_JOIN_PREFIX}{on.source}", read=SUPERSET_DIALECT),
                visible_schema(snapshot, grants), key_facts(snapshot), SUPERSET_DIALECT)
        found = _carried(ref, on, catalogue, grains.get(placeholder))
        if isinstance(found, str):
            refuse(part, found)
            continue
        metric, rendered = found
        # Its label is its metric's own, so where that is another measure's label (the plan's, or one the dashboard
        # keeps), a saved metric's name on the join, or a dimension of a chart that plots it, its table is appended;
        # where that meets one too, it is refused.
        base = _carried_label(metric)
        saved_names = {m.name.casefold() for m in on.saved}
        taken = set(labels) | saved_names | set(kept_labels)
        dimensions = {column.casefold() for column in carried_columns[key]}
        label = base
        if base.casefold() in taken or base.casefold() in dimensions:
            label = f"{base} ({ref.table})"
            if label.casefold() in dimensions or (base.casefold() in dimensions and label.casefold() in taken):
                clashing = base if base.casefold() in dimensions else label
                refuse(part, f"the saved metric {metric.name!r}'s label {clashing!r} is also a dimension of a chart "
                             f"that plots it; drop that dimension or plan the chart on {ref.table} alone")
                continue
            if label.casefold() in taken:
                refuse(part, f"its label {base!r} is another measure's, or a saved metric's on the join, and so is "
                             f"{label!r}; give the other measure a label of its own")
                continue
        if name.casefold() in names:
            refuse(part, "two measures have this name")
            continue
        if name.casefold() in saved_names:
            refuse(part, f"its name {name!r} is also a saved metric's on its dataset (Superset gives every dataset it "
                         "makes one named count); Superset would show two metrics of one name, so name it differently")
            continue
        if name.casefold() in kept_names:  # on a dataset the build makes: never the kept measure
            refuse(part, "this dashboard keeps a measure of this name, and keeps one definition for each name: give "
                         "this measure a name this dashboard does not keep")
            continue
        names[name.casefold()], labels[label.casefold()] = name, name
        decided = Measure(name=name, label=label, expression=rendered.decided, kind=ref.kind, dataset_id=placeholder,
                          description=f"The saved metric {metric.name!r} of {ref.table!r}, carried onto the join.",
                          asked_by="", currency=_carried_currency(metric))
        verdict = check_measure(decided, source=on.source, dimension_names=tuple(sorted(carried_columns[key])),
                                adapter=adapter, snapshot=snapshot, grants=grants, guard_fanout=settings.guard_fanout)
        if isinstance(verdict, MeasureRefused):
            refuse(part, verdict.reason, repairable=verdict.repairable)
            continue
        measure = replace(decided, expression=rendered.superset)
        kept.append(measure)
        carried[key] = WrittenRef(measure)

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
        elif (problem := superset_text_problem(chart.title, "a chart's title")) is not None:
            refuse(part, problem)
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
            elif ref.table is not None:
                # The measure the carried pass wrote for it; refused there, the reference as planned, with the reason
                # the measure was given, so the chart's own checks still run.
                refs.append(carried.get((ref.name, ref.kind, ref.table, chart.dataset_id), ref))
            else:
                refs.append(ref)
        spec = replace(chart, measures=tuple(refs))
        columns = _chart_columns(spec)
        folded = {c.casefold() for c in columns}
        saved = {metric.name: metric for metric in on.saved}
        for ref in refs:
            if not isinstance(ref, SavedRef) or ref.table is not None:
                continue
            if ref.name not in saved:
                refuse(part, f"{on.part} has no saved metric named {ref.name!r} this builder can reach")
            elif ref.name.casefold() in folded:
                refuse(part, f"the saved metric {ref.name!r} is also the name of a dimension of the chart")
            elif ref.kind is MeasureKind.PERCENT and scales_by_100(saved[ref.name].expression):
                refuse(part, _scaled_saved(ref.name))
        shown = [(ref.name if isinstance(ref, SavedRef) else ref.measure.label.strip()).casefold() for ref in refs]
        if len(set(shown)) < len(shown):
            refuse(part, "it plots one metric twice")
        if spec.family is Family.HISTOGRAM and refs:
            refuse(part, "a histogram shows the spread of its first dimension and takes no measure")
        for column in [*columns, *(item.column for item in spec.filters)]:
            if column not in on.columns:
                refuse(part, _column_refusal(column, on))
            elif on.owners:
                reads.setdefault(spec.dataset_id, set()).add(on.owners[column])
        if spec.time_column and spec.time_column in on.columns and not _temporal(on.types.get(spec.time_column)):
            refuse(part, f"its time column {spec.time_column!r} is not a date or a timestamp")
        if spec.time_grain is not None:
            if spec.time_grain not in _GRAIN_WORD:
                refuse(part, f"its grain is one of {', '.join(GRAINS)}")
            elif not spec.time_column:
                refuse(part, "a grain groups a time column, and the chart has none")
        if spec.row_limit is not None and spec.family in _TIME_AXIS_FAMILIES:
            # The x-axis is the time column, or, for an x-y chart without one, its first dimension: a date there is
            # a time axis all the same.
            axis = spec.time_column or (spec.dimensions[0] if spec.dimensions and spec.family is not Family.MIXED
                                        else None)
            if axis is not None and (axis == spec.time_column or _temporal(on.types.get(axis))):
                refuse(part, _TIME_AXIS_ROW_LIMIT)
        for item in spec.filters:
            if (problem := _filter_problem(item)) is not None:
                refuse(part, problem)
        for reason in _null_contradictions(spec):
            refuse(part, reason)
        if len(refused) == before:
            why = plan_refusal(spec, interactive_pivot_available=interactive_pivot_available,
                               saved={ref.name: saved[ref.name] for ref in refs
                                      if isinstance(ref, SavedRef) and ref.table is None},
                               currency=settings.superset_currency)
            if why is not None:
                refuse(part, why)
        if len(refused) == before:
            # A chart's filter reasons go back together: a value no row holds, and the NULL rows a negation leaves out.
            for reason in [*_filter_refusals(spec, on, adapter=adapter, snapshot=snapshot, grants=grants,
                                             probes=probes),
                           *_null_refusals(spec, on, adapter=adapter, snapshot=snapshot, grants=grants,
                                           probes=probes)]:
                refuse(part, reason)
        charts.append(spec)

    for placeholder, on in rows.items():
        if on.measured is not None and len(read := reads.get(placeholder, set())) < 2:
            what = f"only {', '.join(sorted(read))}" if read else "none of the joined tables' columns"
            refuse(on.part, f"its measures and charts read {what}: a joined dataset is made only where they read more "
                            "than one table, so plan them on that table")

    offered = {dashboard.id for dashboard in context.dashboards}
    if len(plan.reuse_candidates) > 1 or not set(plan.reuse_candidates) <= offered:
        refuse("the dashboard to reuse", "reuse only one of the dashboards listed, by its id, or none")
    if plan.reuse_reason is not None and len(plan.reuse_reason) > MAX_REASON:
        refuse("the dashboard to reuse", f"its reason is at most {MAX_REASON} characters")

    if refused:
        return tuple(refused)
    if any(isinstance(ref, SavedRef) and ref.table is not None for chart in charts for ref in chart.measures):
        raise AssertionError("a joined table's saved metric reached a checked plan without being carried")
    still = {m.dataset_id for m in kept} | {chart.dataset_id for chart in charts}
    return BuildPlan(charts=tuple(charts), measures=tuple(kept), reuse_candidates=plan.reuse_candidates,
                     reuse_reason=plan.reuse_reason,
                     datasets=tuple(rows[d.placeholder].planned for d in plan.datasets if d.placeholder in still))


# ---------------------------------------------------------------- the step

def plan_build(goal: str, *, planner: Planner, client, adapter, snapshot: Snapshot, grants: GrantSet, settings,
               interactive_pivot_available: bool = False, offer_reuse: bool = True, onto: int | None = None,
               built_on: Mapping[int, int] | None = None, kept: Sequence[Measure] = ()) -> BuildPlan | PlanRefused:
    """The checked plan for `goal`, or why there is none. `client` is the request's Superset client, the one its
    adapter (`superset_executor(settings, delegation)`) carries, and every call goes through it as the builder; nothing
    is written. A part refused goes back to `planner` once, with mnemiq's reason; refused again, the plan is refused,
    its reason naming each part. A part no rewrite could pass (`RefusedPart.repairable`) refuses the plan at once.

    With `offer_reuse` False (a build asked for on a dashboard of its own), no dashboard is looked for, and a plan that
    names one to reuse is refused.

    `onto` names the dashboard the build adds to: once the plan is checked, its charts are read, once each, and each
    planned chart one of them already shows is left out (`BuildPlan.left_out`), with the measures and datasets only it
    used. `built_on` gives the dataset of each chart mnemiq made there (`dashboard_charts.datasets_built_on`), and
    `kept` the measures the dashboard keeps: the model is shown those on datasets the builder reads whole, and a
    written measure that takes any kept name with another definition, or a kept label under another name, is refused
    for repair, so the build does not stop on it (`measure_conflict`) once confirmed.

    A planning prompt the model's server refuses or cuts as longer than its window (`ContextTooLong`, `PromptCut`)
    refuses the plan in mnemiq's words, on the first plan or its repair: the goal and what the builder can see are too
    large for one prompt, which is not fitted to the window. A model that does not answer (`ModelUnavailable`) is an
    outage, not a plan refused: it is raised as it is, for the caller to answer."""
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
    kept = tuple(kept)
    context = replace(_context(catalogue, _dashboards(client, goal) if offer_reuse else (), families),
                      kept=_readable_kept(kept, catalogue, adapter, client, snapshot=snapshot, grants=grants,
                                          settings=settings))
    parts: tuple[RefusedPart, ...] = ()
    for attempt in range(2):
        try:
            answer = planner.plan(goal, context)
        except (ContextTooLong, PromptCut) as exc:
            return _too_large(exc)
        if isinstance(answer, PlanRefused):
            if not answer.parts and attempt == 0:
                return answer
            if not answer.parts:  # the repair could not be read: the parts it was to repair stay refused
                return PlanRefused(f"{_naming(parts)}; the repaired plan could not be read: {answer.reason}", parts)
            parts, previous = answer.parts, answer.draft
        else:
            checked = _check(answer, catalogue, context, adapter=adapter, snapshot=snapshot, grants=grants,
                             settings=settings, interactive_pivot_available=interactive_pivot_available,
                             on_dashboard=kept)
            if (terminal := getattr(adapter, "terminal_refusal", None)) is not None:
                # Superset refused the caller, or the caller's roles, while a measure was decided: every later call
                # meets the same refusal, so the request ends here, as `superset_history` ends it.
                logger.warning("Superset refused this request (%s) while its plan was checked; the request ends here",
                               type(terminal).__name__)
                raise terminal
            if isinstance(checked, BuildPlan):
                if onto is None:
                    return checked
                shown = read_dashboard_charts(client, onto, built_on=built_on)
                return leave_out_shown(checked, shown, client=client)
            if not all(part.repairable for part in checked):
                return PlanRefused(_naming(checked), checked)
            parts, previous = checked, json.dumps(draft_of(answer, context), ensure_ascii=False)
        if attempt == 0:
            context = replace(context, repair=Repair(previous=previous, refused=parts))
    return PlanRefused(_naming(parts), parts)


def _readable_kept(kept: Sequence[Measure], catalogue: _Catalogue, adapter, client, *, snapshot: Snapshot,
                   grants: GrantSet, settings) -> tuple[Measure, ...]:
    """The kept measures this builder may be told of: each one mnemiq would run for them. Each measure's own statement
    over its dataset is decided as any statement this builder asks is (`check_measure`), whose deciders read the
    dataset's SQL and the measure's expression as the engine does, their tables, scopes, subqueries and functions:
    a visible table's own dataset is its table; any other is the dataset as Superset answers it to the builder. A
    readable dataset does not make every measure on it readable, since a measure written by a builder with wider grants
    can read another table itself. The rest stay from the model; the plan check still holds their names and labels
    apart. Fails closed: a dataset Superset does not answer, one on another database or schema, or a statement the
    deciders refuse keeps its measure out."""
    own = {dataset: _quoted(catalogue.stored[table]) for table, dataset in catalogue.datasets.items()}
    stored = set(catalogue.stored.values())
    facts: dict[int, DatasetFacts | NotCheckable] = {}
    told = []
    for measure in kept:
        source = own.get(measure.dataset_id)
        if source is None:
            if measure.dataset_id not in facts:
                facts[measure.dataset_id] = read_dataset_facts(client, measure.dataset_id)
            source = _source_for_builder(facts[measure.dataset_id], adapter, stored)
        if source is None:
            continue
        verdict = check_measure(measure, source=source, adapter=adapter, snapshot=snapshot, grants=grants,
                                guard_fanout=settings.guard_fanout)
        if not isinstance(verdict, MeasureRefused):
            told.append(measure)
    return tuple(told)


def _source_for_builder(facts: DatasetFacts | NotCheckable, adapter, stored: set[str]) -> str | None:
    """A dataset, as Superset answered it to the builder, as the FROM of a statement over it (as Superset reads it:
    a table, or `(<its SQL>) AS "virtual_table"`), or None where it is not on the deployment's database and schema, or
    is a physical one over a table the builder's catalogue does not hold."""
    if not isinstance(facts, DatasetFacts) or facts.database_id != adapter.database_id \
            or facts.schema != adapter.schema:
        return None
    if not facts.is_virtual:
        return _quoted(facts.table) if facts.table in stored else None
    return f'({facts.sql}) AS "virtual_table"' if facts.sql else None


def _too_large(exc: ContextTooLong | PromptCut) -> PlanRefused:
    logger.warning("the model's server took a build's planning prompt as too long (%s): %s", exc.kind, exc)
    return PlanRefused(PLAN_CUT if isinstance(exc, PromptCut) else PLAN_TOO_LONG)


# ---------------------------------------------------------------- after the plan

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

# One JSON value in `PLAN_JSON_SHAPE`: a string, a number, a bool or null.
SCALAR = "scalar"
_FILTER_JSON = {"column": SCALAR, "op": SCALAR, "operand": (SCALAR, [SCALAR])}
# What `plan_to_json` writes, declared beside it, the build store's allow-list of a kept plan (`build_store._plan`):
# an object is a dict of the keys it may hold, each with its value's shape; a list is a one-item list of its items'
# shape; a tuple is any one of its shapes. The planner's tests hold the writer to it, both ways.
PLAN_JSON_SHAPE: dict = {
    "format": SCALAR,
    "measures": [{"name": SCALAR, "label": SCALAR, "expression": SCALAR, "kind": SCALAR, "dataset_id": SCALAR,
                  "description": SCALAR, "asked_by": SCALAR, "currency": SCALAR}],
    "charts": [{"key": SCALAR, "family": SCALAR, "title": SCALAR, "dataset_id": SCALAR,
                "measures": [({"saved": SCALAR, "kind": SCALAR}, {"written": SCALAR})], "dimensions": [SCALAR],
                "time_column": SCALAR, "time_grain": SCALAR, "filters": [_FILTER_JSON], "row_limit": SCALAR,
                "source": SCALAR, "target": SCALAR}],
    "datasets": [{"key": SCALAR, "placeholder": SCALAR, "table": SCALAR, "join": SCALAR, "table_name": SCALAR,
                  "sql": SCALAR, "name": SCALAR}],
    "reuse": {"candidates": [SCALAR], "reason": SCALAR},
}


def plan_to_json(plan: BuildPlan) -> dict:
    """The plan as the build store keeps it: plain JSON in `PLAN_JSON_SHAPE`, and no key a value is kept under
    (`VALUE_SHAPED_KEYS`), so a filter's operand is kept as `operand`. `plan_from_json` reads it back."""
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

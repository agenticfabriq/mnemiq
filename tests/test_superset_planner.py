"""The plan step: from a builder's goal to a plan whose every part mnemiq has checked.

Superset is a client in this process that answers each tool from a handler and records every call (no wire, so no
transport is exercised here: `read_saved_metrics` and the adapter's statements are the code under test's own callers).
The model is `FakePlanner` or `LLMPlanner` over a scripted model client. Nothing here calls a real model.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from types import SimpleNamespace

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot
from mnemiq.contract.semantic import Job
from mnemiq.sql.fanout_check import PARTITIONS_JOB
from mnemiq.superset import build_store
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.chart_form import _GRAINS
from mnemiq.superset.client import SupersetToolError
from mnemiq.superset.charts import ChartSpec, Family, Filter, SavedRef, WrittenRef
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.measures import Measure, MeasureKind, scales_by_100
from mnemiq.superset.planner import (
    COUNT_METRIC,
    FILTER_OPERATORS,
    GRAINS,
    BuildPlan,
    FakePlanner,
    LLMPlanner,
    PlannedDataset,
    PlanRefused,
    draft_of,
    plan_build,
    plan_from_draft,
    plan_from_json,
    plan_prompt,
    plan_to_json,
    with_made_datasets,
)
from tests.superset_shared import CHANGED_OR_REFUSED_BY_RC2, KEPT_BY_RC2

K = MeasureKind
EMPTY = {"success": True, "columns": [], "rows": []}
SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="policy.policy_id", object_id="policy", name="policy_id", data_type="INTEGER"),
    Column(id="policy.region", object_id="policy", name="region", data_type="VARCHAR"),
    Column(id="policy.premium", object_id="policy", name="premium", data_type="NUMERIC"),
    Column(id="policy.start_date", object_id="policy", name="start_date", data_type="DATE"),
    Column(id="policy.created_at", object_id="policy", name="created_at", data_type="timestamp without time zone"),
    Column(id="claim.claim_id", object_id="claim", name="claim_id", data_type="INTEGER"),
    Column(id="claim.policy_id", object_id="claim", name="policy_id", data_type="INTEGER"),
    Column(id="claim.amount", object_id="claim", name="amount", data_type="NUMERIC"),
    Column(id="claim.status", object_id="claim", name="status", data_type="VARCHAR"),
    Column(id="claim.created_at", object_id="claim", name="created_at", data_type="timestamp with time zone"),
    Column(id="agent.agent_id", object_id="agent", name="agent_id", data_type="INTEGER"),
    Column(id="agent.office", object_id="agent", name="office", data_type="VARCHAR"),
    Column(id="payroll.salary", object_id="payroll", name="salary", data_type="NUMERIC"),
])
GRANTS = GrantSet(frozenset({"policy", "claim", "agent"}))
GOAL = "Premium by region and month"
WRITES = ("generate_chart", "create_dataset", "create_virtual_dataset", "delete_chart", "remove_chart_from_dashboard",
          *(TOOL_PREFIX + name for name in ("create_dashboard", "create_dataset", "create_chart", "patch_chart",
                                            "add_saved_metric", "delete_dataset", "set_layout")))


class Tools:
    """Stands in for the request's Superset client: each tool answered by a handler of the call's request, and every
    call recorded. A tool with no handler is a call the test did not expect."""

    def __init__(self, handlers: dict) -> None:
        self.handlers = dict(handlers)
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool: str, request: dict, *, timeout_s: float | None = None) -> dict:
        self.calls.append((tool, copy.deepcopy(request)))
        if tool not in self.handlers:
            raise AssertionError(f"unexpected call of {tool}")
        answer = self.handlers[tool](request)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def tools(self) -> list[str]:
        return [tool for tool, _ in self.calls]

    def statements(self) -> list[str]:
        return [request["sql"] for tool, request in self.calls if tool == "execute_sql"]


def _dataset(dataset_id: int, table: str, sql: str | None = None, **more) -> dict:
    return {"id": dataset_id, "table_name": table, "schema": "public", "database_id": 7, "sql": sql, **more}


def _metric(dataset_id: int, name: str, expression: str, **more) -> dict:
    return {"name": name, "verbose_name": more.get("verbose_name"), "description": more.get("description"),
            "expression": expression, "d3format": None, "warning_text": more.get("warning_text"), "source": "builtin",
            "dataset_id": dataset_id, "dataset_name": f"t{dataset_id}", "view_id": None, "view_name": None,
            "compatible_dimensions": []}


# The plan check's two statements about a filter's value (`mnemiq.superset.literals`): whether a column holds it
# (`probe_sql`), and the values the column holds (`held_values_sql`).
_PROBE = re.compile(r'SELECT 1 FROM "(?P<table>[^"]+)" WHERE "(?P<column>[^"]+)" = \'(?P<value>(?:[^\']|\'\')*)\' '
                    r'LIMIT 1')
_HELD_READ = re.compile(r'SELECT DISTINCT "(?P<column>[^"]+)" FROM "(?P<table>[^"]+)" WHERE "[^"]+" IS NOT NULL '
                        r'ORDER BY 1 LIMIT (?P<limit>\d+)')
# The plan check's probe of whether a column holds NULL (`null_statement`).
_NULL_PROBE = re.compile(r'SELECT 1 FROM "(?P<table>[^"]+)" WHERE "(?P<column>[^"]+)" IS NULL LIMIT 1')


def _null_answer(sql: str, nulls) -> dict | None:
    """Superset's answer to a probe of whether a column holds NULL, a row where `(table, column)` is in `nulls`, or
    None for any other statement."""
    if (probe := _NULL_PROBE.fullmatch(sql)) is None:
        return None
    found = (probe["table"], probe["column"]) in nulls
    return {"success": True, "columns": [{"name": "?column?", "type": "int64"}],
            "rows": [{"?column?": 1}] if found else []}


def _value_answer(sql: str, held: dict | None) -> dict | None:
    """Superset's answer to a probe of a filter's value or a read of a column's values, from `held` (the values each
    `(table, column)` holds; None: every probe finds its value), or None for any other statement."""
    one = [{"name": "?column?", "type": "int64"}]
    if (probe := _PROBE.fullmatch(sql)) is not None:
        found = held is None or probe["value"].replace("''", "'") in held.get((probe["table"], probe["column"]), [])
        return {"success": True, "columns": one, "rows": [{"?column?": 1}] if found else []}
    if (read := _HELD_READ.fullmatch(sql)) is not None:
        values = sorted((held or {}).get((read["table"], read["column"]), []))[:int(read["limit"])]
        return {"success": True, "columns": [{"name": read["column"], "type": "object"}],
                "rows": [{read["column"]: value} for value in values]}
    return None


def _tools(*, metrics=None, extension=None, dashboards=(), execute=None, datasets=None, held=None,
           nulls=frozenset(), infos=None) -> Tools:
    """policy is dataset 1 and claim dataset 2; agent has none, and a virtual dataset over policy is never a table's.
    `held` gives the values each `(table, column)` holds, for the plan check's probes of a filter's values; without
    it, every value probed is found. `nulls` holds each `(table, column)` that holds NULL; no other does."""
    metrics = metrics or {}
    extension = extension or {}
    datasets = datasets if datasets is not None else [_dataset(1, "policy"), _dataset(2, "claim"),
                                                      _dataset(9, "policy_view", "SELECT region FROM policy")]

    def list_metrics(request):
        rows = metrics.get(request["dataset_id"], [])
        return {"metrics": rows, "total_count": len(rows), "page": 1, "page_size": request["page_size"],
                "total_pages": 1, "success": True}

    def dataset_metrics(request):
        return {"datasets": [{"dataset_id": i, "metrics": extension.get(i, [])} for i in request["dataset_ids"]]}

    def dataset_info(request):
        # As Superset answers the builder: a dataset it reads, else an error.
        found = (infos or {}).get(request["identifier"])
        return dict(found) if found is not None else SupersetToolError("no dataset of that id for this user")

    def list_dashboards(request):
        word = request["search"].casefold()
        return {"dashboards": [dict(item) for item in dashboards if word in item["dashboard_title"].casefold()],
                "count": 0, "total_pages": 1}

    return Tools({
        "list_datasets": lambda request: {"datasets": datasets, "count": len(datasets)},
        "list_metrics": list_metrics,
        TOOL_PREFIX + "dataset_metrics": dataset_metrics,
        "list_dashboards": list_dashboards,
        "get_dataset_info": dataset_info,
        "execute_sql": lambda request: (_null_answer(request["sql"], nulls) or _value_answer(request["sql"], held)
                                        or (execute or (lambda _: EMPTY))(request)),
    })


def _adapter(tools: Tools) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(tools, database_id=7, schema="public")


def _settings() -> Settings:
    return Settings(guard_fanout=False, superset_currency="USD")


def _plan(goal: str, planner, tools: Tools, *, grants=GRANTS, snapshot=SNAPSHOT, settings: Settings | None = None,
          **more):
    adapter = _adapter(tools)
    return plan_build(goal, planner=planner, client=adapter.client, adapter=adapter, snapshot=snapshot, grants=grants,
                      settings=settings or _settings(), **more)


def _measure(name="total_premium", label="Total premium", expression="SUM(premium)", kind=K.CURRENCY, dataset_id=1,
             currency=None) -> Measure:
    return Measure(name=name, label=label, expression=expression, kind=kind, dataset_id=dataset_id, description="",
                   asked_by="", currency=currency)


def _chart(key, family, refs, *, dataset_id=1, dims=(), time_column=None, grain=None, filters=(), row_limit=None,
           source=None, target=None) -> ChartSpec:
    return ChartSpec(key=key, family=family, title=key.replace("_", " ").capitalize(), dataset_id=dataset_id,
                     measures=tuple(refs), dimensions=tuple(dims), time_column=time_column, time_grain=grain,
                     filters=tuple(filters), row_limit=row_limit, source=source, target=target)


PREMIUM = _measure()
CLAIMS = _measure("claim_count", "Claims", "COUNT(claim_id)", K.COUNT, dataset_id=2)


def _valid_plan(**changes) -> BuildPlan:
    fields = dict(
        charts=(
            _chart("premium_by_month", Family.LINE, [WrittenRef(PREMIUM)], time_column="start_date", grain="P1M"),
            _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                   filters=[Filter("region", "IN", ["north", "south"])]),
            _chart("claims", Family.BIG_NUMBER, [WrittenRef(CLAIMS)], dataset_id=2),
        ),
        measures=(PREMIUM, CLAIMS),
        reuse_candidates=(),
    )
    fields.update(changes)
    return BuildPlan(**fields)


def _parts(refused: PlanRefused) -> dict[str, str]:
    return {part.part: part.reason for part in refused.parts}


# ---------------------------------------------------------------- a plan, checked, repaired, and offering a dashboard

def test_valid_plan_passes():
    tools = _tools()
    planner = FakePlanner(_valid_plan())
    plan = _plan(GOAL, planner, tools)
    assert plan == _valid_plan()
    assert len(planner.contexts) == 1 and planner.contexts[0].repair is None
    assert not plan.awaiting_choice
    # Each written measure's statement was decided, its source the table's quoted name.
    statements = "\n".join(tools.statements())
    assert 'SUM(premium) AS total_premium FROM "policy"' in statements
    assert 'COUNT(claim_id) AS claim_count FROM "claim"' in statements
    assert not set(tools.tools()) & set(WRITES)


def test_refused_measure_is_repaired_once():
    bad = _measure(expression="SUM(premium) OVER ()")
    first = _valid_plan(measures=(bad, CLAIMS), charts=(
        _chart("premium_by_region", Family.BAR, [WrittenRef(bad)], dims=["region"]),
        _chart("claims", Family.BIG_NUMBER, [WrittenRef(CLAIMS)], dataset_id=2)))
    second = _valid_plan()
    planner = FakePlanner([first, second])
    plan = _plan(GOAL, planner, _tools())
    assert plan == second
    assert len(planner.contexts) == 2
    repair = planner.contexts[1].repair
    assert [part.part for part in repair.refused] == ["measure 'total_premium'"]
    assert "does not aggregate" in repair.refused[0].reason
    assert "SUM(premium) OVER ()" in repair.previous
    # The repair goes back with the same view of the data: only the repair is new.
    assert dataclasses.replace(planner.contexts[1], repair=None) == planner.contexts[0]


def test_second_refusal_refuses_the_plan():
    bad = _measure(expression="SUM(premium) OVER ()")
    plan = _valid_plan(measures=(bad,), charts=(_chart("premium_by_region", Family.BAR, [WrittenRef(bad)],
                                                       dims=["region"]),))
    planner = FakePlanner(plan)
    refused = _plan(GOAL, planner, _tools())
    assert isinstance(refused, PlanRefused)
    assert len(planner.contexts) == 2, "one repair, never a third round"
    assert "measure 'total_premium'" in refused.reason and "does not aggregate" in refused.reason
    assert list(_parts(refused)) == ["measure 'total_premium'"]


def test_planted_saved_metric_replaces_an_equivalent_written_one():
    metrics = {1: [_metric(1, "premium_total", "SUM(premium)"), _metric(1, "sum_premium", "sum( premium )")]}
    extension = {1: [{"metric_name": "premium_total", "currency": None, "certified_by": None},
                     {"metric_name": "sum_premium", "currency": None, "certified_by": "finance"}]}
    plan = _plan(GOAL, FakePlanner(_valid_plan()), _tools(metrics=metrics, extension=extension))
    assert isinstance(plan, BuildPlan)
    assert [m.name for m in plan.measures] == ["claim_count"], "the written premium measure is not kept"
    # The certified one of the two equivalent saved metrics, with the kind the planner gave the measure.
    for chart in plan.charts[:2]:
        assert chart.measures == (SavedRef("sum_premium", K.CURRENCY),)
    assert plan.charts[2].measures == (WrittenRef(CLAIMS),)


def test_existing_dashboard_is_offered_not_taken():
    dashboards = [{"id": 12, "dashboard_title": "Premium by region", "owners": [{"first_name": "Avery"}]},
                  {"id": 30, "dashboard_title": "Regional premiums overview"},
                  {"id": 31, "dashboard_title": "Monthly ops"}]
    tools = _tools(dashboards=dashboards)
    planner = FakePlanner(_valid_plan(reuse_candidates=(12,), reuse_reason="it already shows premium by region"))
    plan = _plan(GOAL, planner, tools)
    # Only titles sharing a word with the goal: "Regional premiums" shares none, "Monthly" is not "month".
    assert [(d.id, d.title) for d in planner.contexts[0].dashboards] == [(12, "Premium by region")]
    searched = [request for tool, request in tools.calls if tool == "list_dashboards"]
    assert {request["search"] for request in searched} == {"premium", "region", "month"}
    assert all(request["select_columns"] == ["id", "dashboard_title"] for request in searched)
    assert plan.reuse_candidates == (12,) and plan.awaiting_choice
    assert not set(tools.tools()) & set(WRITES), "nothing is built from a plan that offers a dashboard"
    assert plan_from_json(json.loads(json.dumps(plan_to_json(plan)))) == plan


class ScriptedModel:
    """Stands in for `LLMClient`: answers each call with the next scripted reply and records what it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.sent: list[dict] = []

    def complete(self, system: str, user: str, max_tokens: int = 512, extra_body: dict | None = None,
                 temperature: float | None = None) -> str:
        self.sent.append({"system": system, "user": user, "max_tokens": max_tokens, "extra_body": extra_body})
        return self.replies.pop(0)


def _draft(expression: str) -> str:
    return json.dumps({
        "measures": [{"name": "total_premium", "label": "Total premium", "expression": expression,
                      "kind": "currency", "table": "policy"}],
        "charts": [{"key": "premium_by_month", "family": "line", "title": "Premium by month",
                    "measures": ["total_premium"], "dimensions": [], "time_column": "start_date", "grain": "month"},
                   {"key": "premium_by_region", "family": "bar", "title": "Premium by region",
                    "measures": ["total_premium"], "dimensions": ["region"]}],
        "reuse_dashboard": None})


NAMES = ("Avery", "Quillfeather", "Morwenna", "Valemont", "Rowan", "Pikeworth")
SENTINEL = "987654"
SUPERSET_TEXT = "SUPERSET-ERROR-TEXT"


def test_prompt_holds_no_editor_names_or_values():
    def execute(request):
        if "premium * 2" in request["sql"]:
            return {"success": False, "error": f"{SUPERSET_TEXT}: column premium is ambiguous for Avery Quillfeather",
                    "error_type": "GENERIC_DB_ENGINE_ERROR"}
        if "total_premium" not in request["sql"]:
            return EMPTY  # the catalogue's own reads
        return {"success": True, "columns": [{"name": "total_premium", "type": "float64"}],
                "rows": [{"total_premium": 987654.321}]}

    tools = _tools(
        datasets=[_dataset(1, "policy", owners=[{"first_name": "Morwenna", "last_name": "Valemont"}]),
                  _dataset(2, "claim")],
        metrics={1: [_metric(1, "sum_claims_paid", "SUM(premium) - 1", description="Owned by Avery Quillfeather",
                             warning_text="Ask Rowan Pikeworth", verbose_name="Rowan's metric")]},
        extension={1: [{"metric_name": "sum_claims_paid", "currency": None, "certified_by": "Morwenna Valemont",
                        "certification_details": "Signed by Morwenna Valemont",
                        "warning_text": "Ask Rowan Pikeworth"}]},
        dashboards=[{"id": 12, "dashboard_title": "Premium by region", "certified_by": "Morwenna Valemont",
                     "owners": [{"first_name": "Avery", "last_name": "Quillfeather"}],
                     "changed_by_name": "Rowan Pikeworth"}],
        execute=execute)
    model = ScriptedModel([_draft("SUM(premium * 2)"), _draft("SUM(premium)")])
    planner = LLMPlanner(model)
    plan = _plan(GOAL, planner, tools)
    assert isinstance(plan, BuildPlan), plan
    assert len(model.sent) == 2, "the refused measure went back once"
    answered = [execute(request) for tool, request in tools.calls if tool == "execute_sql"]
    assert any(SENTINEL in json.dumps(answer) for answer in answered), "the value was planted where it is read"
    assert any(SUPERSET_TEXT in json.dumps(answer) for answer in answered), "Superset's text was planted too"
    for sent in model.sent:
        whole = sent["system"] + "\n" + sent["user"]
        for name in NAMES:
            assert name not in whole, f"an editor's name reached the prompt: {name}"
        assert SENTINEL not in whole and SUPERSET_TEXT not in whole and "ambiguous" not in whole
    # The search reads the prompt that matters: the schema, the saved metric and the dashboard are in it.
    first = model.sent[0]["user"]
    for shown in ("policy", "start_date", "sum_claims_paid", "SUM(premium) - 1", "Premium by region", GOAL):
        assert shown in first
    assert "payroll" not in first and "salary" not in first, "an ungranted table is not shown"
    # The repair carries mnemiq's reason and the model's own earlier plan, nothing of Superset's.
    repair = model.sent[1]["user"]
    assert "The source rejected this query" in repair and "premium * 2" in repair
    # The context object itself, whole, holds none of them either.
    context = planner_contexts(tools)
    for name in (*NAMES, SENTINEL, SUPERSET_TEXT):
        assert name not in context


def planner_contexts(tools: Tools) -> str:
    """Every context a plan made on `tools` hands its planner, serialized whole."""
    recorder = FakePlanner(_valid_plan())
    _plan(GOAL, recorder, tools)
    return json.dumps([dataclasses.asdict(context) for context in recorder.contexts], default=str)


# ---------------------------------------------------------------- the checks a plan's parts meet

def test_written_label_equal_to_a_saved_metric_name_is_refused():
    metrics = {1: [_metric(1, "premium_total", "AVG(premium)")]}
    clash = _measure(label="premium_total")
    plan = _valid_plan(measures=(clash,), charts=(_chart("by_region", Family.BAR, [WrittenRef(clash)],
                                                         dims=["region"]),))
    refused = _plan(GOAL, FakePlanner(plan), _tools(metrics=metrics))
    assert isinstance(refused, PlanRefused)
    assert "saved metric" in _parts(refused)["measure 'total_premium'"]


def test_two_written_measures_with_one_label_are_refused():
    other = _measure("premium_avg", expression="AVG(premium)", label="TOTAL PREMIUM")
    plan = _valid_plan(measures=(PREMIUM, other), charts=(
        _chart("by_region", Family.BAR, [WrittenRef(PREMIUM), WrittenRef(other)], dims=["region"]),))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "measure 'premium_avg'" in _parts(refused)
    assert "label" in _parts(refused)["measure 'premium_avg'"]
    assert "measure 'total_premium'" not in _parts(refused)


# A dashboard keeps each measure on one dataset: a build onto it that makes a dataset of its own makes the same formula
# a measure of its own, which needs a name and a label the dashboard does not keep.
ELSEWHERE = 53  # a dataset mnemiq made for the dashboard, which is no table's of the catalogue


# Superset's answer for dataset ELSEWHERE as a dataset mnemiq made over a join of two tables the builder reads.
READABLE_JOIN = {"id": ELSEWHERE, "table_name": "dataset_1 b1", "schema": "public", "database_id": 7, "is_virtual": True,
                 "sql": "SELECT p.premium, p.region FROM policy AS p JOIN claim AS c ON c.policy_id = p.policy_id",
                 "metrics": []}


def _one_bar(measure: Measure) -> BuildPlan:
    return _valid_plan(measures=(measure,), charts=(_chart("by_region", Family.BAR, [WrittenRef(measure)],
                                                           dims=["region"]),))


def test_a_measure_named_as_one_the_dashboard_keeps_elsewhere_goes_back_for_a_name_of_its_own():
    """The dashboard keeps `total_premium` (SUM(premium)) on a dataset of its own; the plan writes the same formula on
    the policy table. The plan's measure is refused, repairable, in words that say the dashboard keeps that name; the
    renamed plan passes. Before, the plan passed and its build stopped at its first step on `measure_conflict`."""
    kept = dataclasses.replace(PREMIUM, dataset_id=ELSEWHERE)
    renamed = _measure("premium_on_policy", "Premium")
    planner = FakePlanner([_one_bar(PREMIUM), _one_bar(renamed)])
    assert _plan(GOAL, planner, _tools(), kept=(kept,)) == _one_bar(renamed)
    [part] = planner.contexts[1].repair.refused
    assert part.part == "measure 'total_premium'" and part.repairable, part
    assert "this dashboard keeps" in part.reason and "name" in part.reason, part
    # The kept measure's dataset is no table this builder can see: its definition is not said.
    assert "SUM(premium)" not in part.reason and "premium" not in part.reason.replace("total_premium", ""), part
    refused = _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(kept,))
    assert isinstance(refused, PlanRefused) and "this dashboard keeps" in _parts(refused)["measure 'total_premium'"]


def test_a_measure_labelled_as_one_the_dashboard_keeps_is_refused():
    """A label is a dashboard's too: Superset keys a chart's values by label, and the panel's Edit holds every label
    of a dashboard's measures apart, so a new name with a kept label is refused as well, whatever its dataset."""
    kept = dataclasses.replace(PREMIUM, name="premium_kept", dataset_id=ELSEWHERE)
    refused = _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(kept,))
    assert isinstance(refused, PlanRefused), refused
    reason = _parts(refused)["measure 'total_premium'"]
    assert "this dashboard keeps" in reason and "label" in reason, reason
    on_policy = dataclasses.replace(kept, dataset_id=1)
    refused = _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(on_policy,))
    assert isinstance(refused, PlanRefused) and "label" in _parts(refused)["measure 'total_premium'"]


def test_a_kept_measure_on_a_dataset_the_builder_cannot_read_reaches_the_model_in_no_part():
    """The measures a dashboard keeps are the store's, read for the dashboard, not for the builder: one on a dataset
    Superset does not answer the builder for, or over a table the builder's grants leave out, or on another database,
    reaches the model in no part (name, label, kind or definition), and a planned measure that meets its name or label
    is refused in words that repeat neither."""
    secret = _measure("secret_margin", "Secret Margin", "SUM(salary)", dataset_id=ELSEWHERE)
    # As Superset answers each to the builder: payroll is no table the builder's grants hold, read plainly, through a
    # CTE of its own name, through an upper-case CTE that the unquoted name passes by, and inside a string a function
    # runs as SQL; and a dataset on another database.
    payroll = {54: "SELECT salary FROM payroll",
               56: "WITH payroll AS (SELECT * FROM payroll) SELECT salary FROM payroll",
               57: 'WITH "PAYROLL" AS (SELECT premium AS salary FROM policy) SELECT salary FROM PAYROLL',
               58: "SELECT query_to_xml('SELECT * FROM payroll', true, false, '') AS salary FROM policy"}
    infos = {i: dict(READABLE_JOIN, id=i, sql=sql) for i, sql in payroll.items()}
    infos[55] = dict(READABLE_JOIN, id=55, database_id=8)  # read whole, were it on this database
    hidden = (secret, *(dataclasses.replace(secret, dataset_id=i, name=f"secret_{i}", label=f"Secret {i}")
                        for i in payroll),
              dataclasses.replace(secret, dataset_id=55, name="secret_55", label="Secret 55", expression="SUM(premium)"),
              # On policy's own dataset, which the builder reads, but reading payroll itself: made by a builder with
              # wider grants.
              dataclasses.replace(secret, dataset_id=1, name="secret_sub", label="Secret Sub",
                                  expression="SUM((SELECT MAX(salary) FROM payroll))"))
    tools = _tools(infos=infos)
    told = FakePlanner(_one_bar(PREMIUM))
    assert _plan(GOAL, told, tools, kept=hidden) == _one_bar(PREMIUM)
    user = plan_prompt(GOAL, told.contexts[0])[1]
    assert "keeps" not in user and all(word not in user for word in ("secret", "Secret", "salary")), user

    by_name = dataclasses.replace(secret, name="total_premium")
    refused = _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(by_name,))
    reason = _parts(refused)["measure 'total_premium'"]
    assert "this dashboard keeps" in reason and "Secret" not in reason and "salary" not in reason, reason
    by_label = dataclasses.replace(secret, label="Total premium")
    refused = _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(by_label,))
    reason = _parts(refused)["measure 'total_premium'"]
    assert "this dashboard keeps" in reason and "label" in reason and "secret" not in reason, reason


def test_the_measure_the_dashboard_keeps_is_planned_again_as_it_is():
    """The same measure (its name, its kind, its dataset, its expression once normalized) is the one the dashboard
    keeps, which the build adds the new charts to: it passes, as its build merges it."""
    kept = dataclasses.replace(PREMIUM, expression="sum( premium )")
    assert _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(kept,)) == _one_bar(PREMIUM)


def test_a_kept_name_on_the_same_dataset_with_another_expression_is_refused():
    kept = dataclasses.replace(PREMIUM, expression="AVG(premium)")
    refused = _plan(GOAL, FakePlanner(_one_bar(PREMIUM)), _tools(), kept=(kept,))
    assert isinstance(refused, PlanRefused) and "this dashboard keeps" in _parts(refused)["measure 'total_premium'"]


def test_the_planner_is_told_the_measures_the_dashboard_keeps_and_a_fresh_build_hears_nothing_of_them():
    """The measures a dashboard keeps reach the model in the user prompt, each with its name, label and kind; its
    expression and table only where its dataset is a table this builder can see, since a dashboard's editor may read
    fewer tables than its measures are on. A build with none kept is sent the prompt it was sent before."""
    # A second physical dataset over policy, which is no table's own dataset of the catalogue.
    second = {"id": 60, "table_name": "policy", "schema": "public", "database_id": 7, "is_virtual": False, "sql": None,
              "metrics": []}
    kept = (dataclasses.replace(PREMIUM, dataset_id=ELSEWHERE), dataclasses.replace(CLAIMS, name="claims_kept"),
            _measure("premium_avg", "Average premium", "AVG(premium)", dataset_id=60))
    told = FakePlanner(_one_bar(_measure("premium_on_policy", "Premium")))
    _plan(GOAL, told, _tools(infos={ELSEWHERE: READABLE_JOIN, 60: second}), kept=kept)
    system, user = plan_prompt(GOAL, told.contexts[0])
    assert "this dashboard already keeps" in user and "never one of theirs" in user, user
    assert '"name": "total_premium"' in user and '"label": "Total premium"' in user, user
    assert '"expression": "SUM(premium)"' in user and '"expression": "AVG(premium)"' in user, user
    assert '"expression": "COUNT(claim_id)"' in user and '"table": "claim"' in user, user
    fresh = FakePlanner(_one_bar(PREMIUM))
    _plan(GOAL, fresh, _tools())
    plain_system, plain_user = plan_prompt(GOAL, fresh.contexts[0])
    assert plain_system == system and "keeps" not in plain_user
    assert plain_user == plan_prompt(GOAL, dataclasses.replace(told.contexts[0], kept=()))[1]


def test_saved_ref_named_like_a_dimension_is_refused():
    metrics = {1: [_metric(1, "region", "COUNT(DISTINCT region)")]}
    plan = _valid_plan(measures=(), charts=(_chart("by_region", Family.BAR, [SavedRef("region", K.COUNT)],
                                                   dims=["region"]),))
    refused = _plan(GOAL, FakePlanner(plan), _tools(metrics=metrics))
    assert isinstance(refused, PlanRefused)
    assert "dimension" in _parts(refused)["chart 'by_region'"]


def test_saved_ref_must_exist_on_the_charts_dataset():
    metrics = {2: [_metric(2, "paid", "SUM(amount)")]}
    plan = _valid_plan(measures=(), charts=(_chart("by_region", Family.BAR, [SavedRef("paid", K.CURRENCY)],
                                                   dims=["region"]),))
    refused = _plan(GOAL, FakePlanner(plan), _tools(metrics=metrics))
    assert "no saved metric" in _parts(refused)["chart 'by_region'"]
    ok = _valid_plan(measures=(), charts=(_chart("by_status", Family.BAR, [SavedRef("paid", K.CURRENCY)],
                                                 dataset_id=2, dims=["status"]),))
    assert _plan(GOAL, FakePlanner(ok), _tools(metrics=metrics)) == ok


@pytest.mark.parametrize("item, why", [
    (Filter("region", "BETWEEN", ["a", "b"]), "operator"),
    (Filter("region", "IN", "north"), "list"),
    (Filter("region", "IN", []), "list"),
    (Filter("region", "IS NULL", "x"), "no operand"),
    (Filter("region", "=", None), "operand"),
    (Filter("region", "=", ["a"]), "one operand"),
    (Filter("salary", "=", "x"), "not a column"),
])
def test_a_filter_outside_filterconfig_is_refused(item, why):
    plan = _valid_plan(charts=(_chart("by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                                      filters=[item]),), measures=(PREMIUM,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused)
    assert why in _parts(refused)["chart 'by_region'"]


def test_every_filterconfig_operator_is_accepted():
    """The operators are rc2's `FilterConfig.op` (`superset/mcp_service/chart/schemas.py`), the keys of the charts
    module's adhoc-filter map."""
    assert set(FILTER_OPERATORS) == {"=", ">", "<", ">=", "<=", "!=", "LIKE", "ILIKE", "NOT LIKE", "IN", "NOT IN",
                                     "IS NULL", "IS NOT NULL"}
    operand = {"IN": ["a"], "NOT IN": ["a"], "IS NULL": None, "IS NOT NULL": None}
    # IS NULL filters another column: beside a negated filter on one column it selects no row, and is refused.
    filters = [Filter("premium" if op == "IS NULL" else "region", op, operand.get(op, "a")) for op in FILTER_OPERATORS]
    plan = _valid_plan(charts=(_chart("by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                                      filters=filters),), measures=(PREMIUM,))
    assert _plan(GOAL, FakePlanner(plan), _tools()) == plan


def test_a_histogram_takes_no_measure():
    """A count, which `plan_refusal` lets a histogram show, so only the plan's own rule refuses it."""
    policies = _measure("policies", "Policies", "COUNT(policy_id)", K.COUNT)
    with_one = _valid_plan(measures=(policies,), charts=(_chart("spread", Family.HISTOGRAM, [WrittenRef(policies)],
                                                                dims=["premium"]),))
    refused = _plan(GOAL, FakePlanner(with_one), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "takes no measure" in _parts(refused)["chart 'spread'"]
    without = _valid_plan(measures=(), charts=(_chart("spread", Family.HISTOGRAM, [], dims=["premium"]),))
    assert _plan(GOAL, FakePlanner(without), _tools()) == without


def test_chart_columns_are_its_datasets():
    plan = _valid_plan(charts=(_chart("by_status", Family.BAR, [WrittenRef(PREMIUM)], dims=["status"]),),
                       measures=(PREMIUM,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "status" in _parts(refused)["chart 'by_status'"]


@pytest.mark.parametrize("key", ["Premium", "premium-by-month", "a" * 41, "", "dashboard", "layout", "check"])
def test_chart_keys_are_held_to_the_marker_grammar(key):
    plan = _valid_plan(charts=(_chart(key, Family.BAR, [WrittenRef(PREMIUM)], dims=["region"]),), measures=(PREMIUM,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused) and f"chart {key!r}" in _parts(refused)


def test_two_charts_with_one_key_are_refused():
    chart = _chart("premium", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"])
    refused = _plan(GOAL, FakePlanner(_valid_plan(charts=(chart, chart), measures=(PREMIUM,))), _tools())
    assert "key" in _parts(refused)["chart 'premium'"]


def test_plan_refusal_reaches_the_plan():
    """A chart family's own rule (`plan_refusal`): a pie plots exactly one measure."""
    average = _measure("premium_avg", "Average premium", "AVG(premium)")
    plan = _valid_plan(measures=(PREMIUM, average), charts=(
        _chart("share", Family.PIE, [WrittenRef(PREMIUM), WrittenRef(average)], dims=["region"]),))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert "exactly one measure" in _parts(refused)["chart 'share'"]


def test_a_dashboard_not_offered_cannot_be_reused():
    refused = _plan(GOAL, FakePlanner(_valid_plan(reuse_candidates=(99,))), _tools())
    assert isinstance(refused, PlanRefused) and "the dashboard to reuse" in _parts(refused)


# ---------------------------------------------------------------- datasets

def test_a_chart_on_a_table_with_no_dataset_plans_create_dataset():
    rows = _measure("agents", "Agents", "COUNT(DISTINCT agent_id)", K.COUNT, dataset_id=-1)
    planned = PlannedDataset(key="dataset_1", placeholder=-1, table="agent")
    plan = BuildPlan(charts=(_chart("agents_by_office", Family.BAR, [WrittenRef(rows)], dataset_id=-1,
                                    dims=["office"]),), measures=(rows,), reuse_candidates=(), datasets=(planned,))
    checked = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(checked, BuildPlan), checked
    assert checked.datasets == (dataclasses.replace(planned, table_name="agent"),)
    made = with_made_datasets(checked, {"dataset_1": 41})
    assert made.measures[0].dataset_id == 41 and made.charts[0].dataset_id == 41
    assert made.charts[0].measures[0].measure.dataset_id == 41
    with pytest.raises(ValueError):
        with_made_datasets(checked, {"dataset_9": 41})


def test_a_dataset_superset_makes_carries_count_so_no_measure_takes_its_name():
    """`CreateDatasetCommand.run` calls `fetch_metadata`, which adds the engine spec's metrics
    (`BaseEngineSpec.get_metrics` in `superset/db_engine_specs/base.py`: `count`, `COUNT(*)`), so every dataset a build
    makes, physical or virtual, already has a saved metric named `count`."""
    assert COUNT_METRIC == ("count", "COUNT(*)")
    planned = PlannedDataset(key="dataset_1", placeholder=-1, table="agent")
    named = _measure("count", "Agents", "COUNT(DISTINCT agent_id)", K.COUNT, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("agents", Family.BAR, [WrittenRef(named)], dataset_id=-1, dims=["office"]),),
                     measures=(named,), reuse_candidates=(), datasets=(planned,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "count" in _parts(refused)["measure 'count'"]
    # A written COUNT(*) is that metric already: it is reused, and is no measure of mnemiq's.
    rows = _measure("rows", "Rows", "COUNT(*)", K.COUNT, dataset_id=-1)
    plan = dataclasses.replace(plan, measures=(rows,), charts=(
        _chart("agents", Family.BAR, [WrittenRef(rows)], dataset_id=-1, dims=["office"]),))
    checked = _plan(GOAL, FakePlanner(plan), _tools())
    assert checked.measures == () and checked.charts[0].measures == (SavedRef("count", K.COUNT),)


JOIN = "policy JOIN claim ON claim.policy_id = policy.policy_id"


def test_a_measure_that_needs_a_join_plans_a_virtual_dataset():
    paid = _measure("paid", "Paid", "SUM(amount)", K.CURRENCY, dataset_id=-1)
    planned = PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN)
    plan = BuildPlan(charts=(_chart("paid_by_region", Family.BAR, [WrittenRef(paid)], dataset_id=-1,
                                    dims=["region"]),), measures=(paid,), reuse_candidates=(), datasets=(planned,))
    tools = _tools()
    checked = _plan(GOAL, FakePlanner(plan), tools)
    assert isinstance(checked, BuildPlan), checked
    # The join was decided with the measure (`check_measure`, its source the join itself).
    assert any(f"SUM(amount) AS paid FROM {JOIN}" in sql for sql in tools.statements())
    (made,) = checked.datasets
    # A column one joined table alone has is named alone; one both have is named after its table as well.
    assert made.sql == ('SELECT policy."created_at" AS "policy_created_at", policy."policy_id" AS "policy_policy_id", '
                        'policy."premium" AS "premium", policy."region" AS "region", '
                        'policy."start_date" AS "start_date", claim."amount" AS "amount", '
                        'claim."claim_id" AS "claim_id", claim."created_at" AS "claim_created_at", '
                        'claim."policy_id" AS "claim_policy_id", claim."status" AS "status" FROM ' + JOIN)
    assert made.name == "mnemiq_dataset_1:policy_claim"
    assert (made.key, made.placeholder, made.join, made.table, made.table_name) == ("dataset_1", -1, JOIN, None, None)


@pytest.mark.parametrize("expression, dims, why", [
    ("SUM(claim.amount)", ["region"], "alone"),
    ("COUNT(policy_id)", ["region"], "policy_id"),
    ("SUM(amount)", ["policy_id"], "policy_id"),
    ("MAX(claim_created_at)", ["region"], "may name"),
])
def test_a_joined_dataset_names_its_columns_alone(expression, dims, why):
    paid = _measure("paid", "Paid", expression, K.NUMBER, dataset_id=-1)
    planned = PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN)
    plan = BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=dims),),
                     measures=(paid,), reuse_candidates=(), datasets=(planned,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused)
    assert why in " ".join(_parts(refused).values())


def test_a_join_with_no_written_measure_is_refused():
    planned = PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN)
    plan = BuildPlan(charts=(_chart("spread", Family.HISTOGRAM, [], dataset_id=-1, dims=["amount"]),),
                     measures=(), reuse_candidates=(), datasets=(planned,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert f"the join {JOIN!r}" in _parts(refused)


def test_a_join_on_an_ungranted_table_is_refused():
    join = "policy JOIN payroll ON payroll.salary = policy.premium"
    paid = _measure("paid", "Paid", "SUM(premium)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"]),),
                     measures=(paid,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused) and f"the join {join!r}" in _parts(refused)


# ---------------------------------------------------------------- the model's channel

def test_llm_planner_asks_for_a_json_schema_reply():
    model = ScriptedModel([_draft("SUM(premium)")])
    plan = _plan(GOAL, LLMPlanner(model), _tools())
    assert isinstance(plan, BuildPlan)
    body = model.sent[0]["extra_body"]
    assert body is not None and body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == {"measures", "charts"}
    assert "strict" not in body["response_format"]["json_schema"]
    assert [chart.time_grain for chart in plan.charts] == ["P1M", None]
    assert plan.measures == (PREMIUM,)


@pytest.mark.parametrize("reply", ["no json here", "{not json", "[1, 2]", ""])
def test_an_unreadable_reply_is_a_refusal_never_an_exception(reply):
    model = ScriptedModel([reply])
    refused = _plan(GOAL, LLMPlanner(model), _tools())
    assert isinstance(refused, PlanRefused)
    assert len(model.sent) == 1, "a reply with no plan in it has no part to repair"


def test_a_fenced_reply_is_read():
    model = ScriptedModel(["Here is the plan:\n```json\n" + _draft("SUM(premium)") + "\n```"])
    assert isinstance(_plan(GOAL, LLMPlanner(model), _tools()), BuildPlan)


def test_a_malformed_part_goes_back_once():
    broken = json.loads(_draft("SUM(premium)"))
    broken["measures"][0]["kind"] = "money"
    model = ScriptedModel([json.dumps(broken), _draft("SUM(premium)")])
    plan = _plan(GOAL, LLMPlanner(model), _tools())
    assert isinstance(plan, BuildPlan)
    assert "measure 'total_premium'" in model.sent[1]["user"] and '"money"' in model.sent[1]["user"]


def test_a_draft_table_with_no_dataset_is_planned():
    draft = {"measures": [{"name": "agents", "label": "Agents", "expression": "COUNT(DISTINCT agent_id)",
                           "kind": "count", "table": "agent"},
                          {"name": "paid", "label": "Paid", "expression": "SUM(amount)", "kind": "currency",
                           "join": JOIN}],
             "charts": [{"key": "agents", "family": "bar", "title": "Agents", "measures": ["agents"],
                         "dimensions": ["office"]},
                        {"key": "paid", "family": "bar", "title": "Paid", "measures": ["paid"],
                         "dimensions": ["region"],
                         "filters": [{"column": "status", "op": "IN", "operand": ["open"]}]}]}
    plan = _plan(GOAL, LLMPlanner(ScriptedModel([json.dumps(draft)])), _tools())
    assert isinstance(plan, BuildPlan), plan
    assert [(d.key, d.placeholder, d.table, d.join) for d in plan.datasets] == [
        ("dataset_1", -1, "agent", None), ("dataset_2", -2, None, JOIN)]
    assert [m.dataset_id for m in plan.measures] == [-1, -2]
    assert plan.charts[1].filters == (Filter("status", "IN", ["open"]),)


def test_an_unreadable_repair_still_names_the_refused_part():
    model = ScriptedModel([_draft("SUM(premium) OVER ()"), "no plan this time"])
    refused = _plan(GOAL, LLMPlanner(model), _tools())
    assert isinstance(refused, PlanRefused) and len(model.sent) == 2
    assert "measure 'total_premium'" in refused.reason and "could not be read" in refused.reason


def test_a_join_holding_a_template_marker_is_refused():
    join = "policy JOIN claim ON claim.policy_id = policy.policy_id AND '{{' = '{{'"
    paid = _measure("paid", "Paid", "SUM(amount)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"]),),
                     measures=(paid,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))
    tools = _tools()
    refused = _plan(GOAL, FakePlanner(plan), tools)
    assert f"the join {join!r}" in _parts(refused)
    assert "template marker" in _parts(refused)[f"the join {join!r}"]
    assert not any("{{" in sql for sql in tools.statements()), "nothing holding the marker was sent"


def test_the_model_reads_its_own_plan_back_as_it_wrote_it():
    """`draft_of` is what a repair shows the model of its plan, so `plan_from_draft` reads it back to the same plan."""
    from mnemiq.superset.planner import draft_of, plan_from_draft

    draft = json.loads(_draft("SUM(premium)"))
    draft["measures"].append({"name": "paid", "label": "Paid", "expression": "SUM(amount)", "kind": "currency",
                              "join": JOIN, "currency": "EUR", "description": "claims paid"})
    draft["measures"].append({"name": "agents_n", "saved_metric": "count", "kind": "count", "table": "agent"})
    draft["charts"] += [{"key": "paid", "family": "pie", "title": "Paid", "measures": ["paid"],
                         "dimensions": ["status"], "filters": [{"column": "status", "op": "IS NOT NULL"}]},
                        {"key": "agents", "family": "big_number", "title": "Agents", "measures": ["agents_n"],
                         "dimensions": []},
                        {"key": "spread", "family": "histogram", "title": "Spread", "measures": [],
                         "dimensions": ["premium"], "table": "policy", "row_limit": 500}]
    draft["reuse_dashboard"] = {"id": 12, "reason": "it covers it"}
    recorder = FakePlanner(_valid_plan())
    _plan(GOAL, recorder, _tools())
    context = recorder.contexts[0]
    plan = plan_from_draft(draft, context)
    assert isinstance(plan, BuildPlan), plan
    assert plan_from_draft(draft_of(plan, context), context) == plan


def test_a_draft_cannot_name_an_ungranted_table():
    draft = json.loads(_draft("SUM(salary)"))
    draft["measures"][0]["table"] = "payroll"
    model = ScriptedModel([json.dumps(draft), json.dumps(draft)])
    refused = _plan(GOAL, LLMPlanner(model), _tools())
    assert isinstance(refused, PlanRefused) and "measure 'total_premium'" in _parts(refused)


# ---------------------------------------------------------------- what is kept, and its edges

def test_plan_json_round_trips_and_keeps_no_value_shaped_key():
    planned = PlannedDataset(key="dataset_1", placeholder=-1, table="agent", table_name="agent")
    rows = _measure("agents", "Agents", "COUNT(DISTINCT agent_id)", K.COUNT, dataset_id=-1)
    plan = _valid_plan(reuse_candidates=(12,), reuse_reason="covers it", datasets=(planned,),
                       measures=(PREMIUM, CLAIMS, rows),
                       charts=(*_valid_plan().charts,
                               _chart("agents", Family.BAR, [WrittenRef(rows), SavedRef("count", K.COUNT)],
                                      dataset_id=-1, dims=["office"],
                                      filters=[Filter("office", "=", "north"), Filter("office", "IS NOT NULL")])))
    doc = plan_to_json(plan)

    def keys(node, path=""):
        if isinstance(node, dict):
            for key, item in node.items():
                yield f"{path}.{key}", key
                yield from keys(item, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                yield from keys(item, f"{path}[{index}]")

    found = list(keys(doc))
    assert not [path for path, key in found if key.casefold() in build_store.VALUE_SHAPED_KEYS]
    assert ".charts[1].filters[0].operand" in {path for path, _ in found}, "the planted twin's key is read"
    assert build_store._plan(doc) == json.loads(json.dumps(doc)), "the store takes it as it is"
    assert plan_from_json(json.loads(json.dumps(doc))) == plan


def test_the_store_guard_catches_a_filter_kept_under_value():
    """The planted twin: a plan that kept a filter's operand under `value` is one the store refuses."""
    doc = plan_to_json(_valid_plan())
    doc["charts"][1]["filters"][0]["value"] = doc["charts"][1]["filters"][0].pop("operand")
    with pytest.raises(ValueError):
        build_store._plan(doc)


def test_grains_are_those_the_value_check_reproduces():
    assert set(GRAINS.values()) == set(_GRAINS)


def test_the_client_is_the_adapters():
    tools = _tools()
    with pytest.raises(ValueError):
        plan_build(GOAL, planner=FakePlanner(_valid_plan()), client=_tools(), adapter=_adapter(tools),
                   snapshot=SNAPSHOT, grants=GRANTS, settings=_settings())


def test_a_refusal_of_the_caller_ends_the_request_not_a_repair():
    """A token Superset stopped accepting while a measure was decided stays refused for every later call, so the
    request ends there, as `superset_history` ends it: nothing is sent back to the model to repair."""
    from mnemiq.superset.client import SupersetRefusedCaller

    def execute(request):
        if "total_premium" in request["sql"]:
            return SupersetRefusedCaller("Superset did not accept this sign-in", detail="401")
        return EMPTY

    planner = FakePlanner(_valid_plan())
    with pytest.raises(SupersetRefusedCaller):
        _plan(GOAL, planner, _tools(execute=execute))
    assert len(planner.contexts) == 1


def test_no_visible_table_plans_nothing():
    planner = FakePlanner(_valid_plan())
    refused = _plan(GOAL, planner, _tools(), grants=GrantSet(frozenset()))
    assert isinstance(refused, PlanRefused) and planner.contexts == []


def test_the_planner_is_built_on_the_profiles_client():
    from mnemiq.assembly import build_components

    import duckdb

    settings = Settings(llm_base_url="http://x", llm_api_key="k", llm_model="profile-model")
    kit = build_components(settings, SimpleNamespace(dialect="postgres"), duckdb.connect(":memory:"))
    assert isinstance(kit.planner, LLMPlanner)
    assert kit.planner._client is kit.client and kit.client._model == "profile-model"


# ---------------------------------------------------------------- what each part of a plan is held to

COUNTS = {1: [_metric(1, "count", "COUNT(*)")], 2: [_metric(2, "count", "COUNT(*)")]}


def test_a_saved_metric_has_a_handle_of_its_own_in_the_plan():
    """`count` on two tables. A saved measure's plan-local name is its handle; the metric's own name beside it."""
    from mnemiq.superset.planner import draft_of, plan_from_draft

    draft = {"measures": [{"name": "policies", "saved_metric": "count", "kind": "count", "table": "policy"},
                          {"name": "claims", "saved_metric": "count", "kind": "count", "table": "claim"}],
             "charts": [{"key": "policies", "family": "big_number", "title": "Policies", "measures": ["policies"],
                         "dimensions": []},
                        {"key": "claims", "family": "big_number", "title": "Claims", "measures": ["claims"],
                         "dimensions": []}]}
    recorder = FakePlanner(_valid_plan())
    _plan(GOAL, recorder, _tools(metrics=COUNTS))
    model = ScriptedModel([json.dumps(draft), json.dumps(draft)])
    plan = _plan(GOAL, LLMPlanner(model), _tools(metrics=COUNTS))
    assert isinstance(plan, BuildPlan), plan
    assert len(model.sent) == 1, "nothing to repair"
    assert [(c.dataset_id, c.measures) for c in plan.charts] == [(1, (SavedRef("count", K.COUNT),)),
                                                                 (2, (SavedRef("count", K.COUNT),))]
    shown = draft_of(plan, recorder.contexts[0])
    assert len({m["name"] for m in shown["measures"]}) == 2, "each count keeps a handle of its own"
    assert plan_from_draft(shown, recorder.contexts[0]) == plan


@pytest.mark.parametrize("join", [
    JOIN + " AND EXISTS (SELECT 1 FROM agent)",
    JOIN + " AND claim.status IN (SELECT office FROM agent)",
    JOIN + " AND claim.amount > (SELECT MAX(premium) FROM policy)",
    JOIN + " AND 1 = (SELECT 1)",
    "policy JOIN (WITH c AS (SELECT * FROM claim) SELECT * FROM c) AS claim ON claim.policy_id = policy.policy_id",
])
def test_a_join_holds_tables_and_conditions_only(join):
    """The only tables are the FROM's and each JOIN's, and the only SELECT is the join's own."""
    paid = _measure("paid", "Paid", "SUM(amount)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"]),),
                     measures=(paid,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert f"the join {join!r}" in _parts(refused)


@pytest.mark.parametrize("join, dims", [
    (JOIN, ["region"]),
    (JOIN + " JOIN agent ON agent.agent_id = claim.claim_id", ["office"]),
    ("policy INNER JOIN claim ON claim.policy_id = policy.policy_id", ["region"]),
    ("policy LEFT OUTER JOIN claim ON claim.policy_id = policy.policy_id", ["region"]),
    ("policy FULL JOIN claim ON claim.policy_id = policy.policy_id", ["region"]),
])
def test_a_join_of_two_or_three_tables_is_planned(join, dims):
    paid = _measure("paid", "Paid", "SUM(amount)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=dims),),
                     measures=(paid,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))
    checked = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(checked, BuildPlan), checked
    assert checked.datasets[0].sql.endswith(" FROM " + join)


def test_a_dataset_is_a_tables_only_when_named_exactly_as_stored():
    """rc2 reads a physical dataset FROM `table(self.table_name, schema=self.schema)`
    (`SqlaTable.get_sqla_table` in `superset/connectors/sqla/models.py`), which SQLAlchemy quotes for a mixed-case
    name: a dataset named `Agent` reads `"Agent"`, never the stored `agent`."""
    recorder = FakePlanner(_valid_plan())
    _plan(GOAL, recorder, _tools(datasets=[_dataset(1, "policy"), _dataset(2, "claim"), _dataset(5, "Agent")]))
    tables = {table.name: table.dataset_id for table in recorder.contexts[0].tables}
    assert tables == {"agent": None, "claim": 2, "policy": 1}


def test_a_shared_column_is_a_charts_under_its_tables_name():
    """Both tables carry created_at; the chart groups by claim's, as the joined dataset names it."""
    paid = _measure("paid", "Paid", "SUM(amount)", K.CURRENCY, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("paid_by_month", Family.LINE, [WrittenRef(paid)], dataset_id=-1,
                                    dims=["region"], time_column="claim_created_at", grain="P1M",
                                    filters=[Filter("policy_policy_id", "IS NOT NULL")]),),
                     measures=(paid,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))
    checked = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(checked, BuildPlan), checked
    assert 'claim."created_at" AS "claim_created_at"' in checked.datasets[0].sql
    # Named bare, the shared column is refused, and the reason names the two it can be.
    bare = dataclasses.replace(plan, charts=(dataclasses.replace(plan.charts[0], time_column="created_at"),))
    refused = _plan(GOAL, FakePlanner(bare), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "policy_created_at" in _parts(refused)["chart 'paid_by_month'"]


def test_reuse_is_not_offered_when_the_build_asks_for_none():
    """With `offer_reuse=False` no dashboard is listed, and a candidate the model names is refused."""
    tools = _tools(dashboards=[{"id": 12, "dashboard_title": "Premium by region"}])
    recorder = FakePlanner(_valid_plan())
    assert _plan(GOAL, recorder, tools, offer_reuse=False) == _valid_plan()
    assert "list_dashboards" not in tools.tools() and recorder.contexts[0].dashboards == ()
    refused = _plan(GOAL, FakePlanner(_valid_plan(reuse_candidates=(12,))), tools, offer_reuse=False)
    assert isinstance(refused, PlanRefused), refused
    assert "the dashboard to reuse" in _parts(refused)
    assert "list_dashboards" not in tools.tools()


class _FailsOnTheRepair(FakePlanner):
    """Answers with its plan first, then fails as the model's client fails: the outage comes on the repair."""

    def __init__(self, plan: BuildPlan, failure: Exception) -> None:
        super().__init__(plan)
        self.failure = failure

    def plan(self, goal, context):
        if self.contexts:
            self.contexts.append(context)
            raise self.failure
        return super().plan(goal, context)


def _repaired_plan_meets(failure: Exception):
    bad = _valid_plan(measures=(_measure(expression="SUM(premium) OVER ()"),), charts=(
        _chart("premium_by_region", Family.BAR, [WrittenRef(_measure(expression="SUM(premium) OVER ()"))],
               dims=["region"]),))
    planner = _FailsOnTheRepair(bad, failure)
    return planner, (lambda: _plan(GOAL, planner, _tools()))


def test_a_prompt_too_large_on_the_repair_ends_the_plan_in_mnemiqs_words():
    """The model's server refuses the repair's prompt as longer than its window (the repair carries the plan back, so
    it is the longer prompt): the plan is refused there, saying the goal and what the builder can see are too large
    for one prompt, and the provider's words are not in it."""
    from mnemiq.llm.client import ContextTooLong

    planner, plan = _repaired_plan_meets(ContextTooLong("model-host.internal: maximum context length is 32768"))
    refused = plan()
    assert isinstance(refused, PlanRefused), refused
    assert len(planner.contexts) == 2 and "the goal and what this builder can see" in refused.reason
    assert "model-host" not in refused.reason


def test_a_model_that_does_not_answer_the_repair_is_raised_for_the_route_to_answer():
    """An outage is not a plan refused: the model's `ModelUnavailable` leaves the plan step as it is, on the repair
    as on the first plan, for the route to answer as an outage."""
    from mnemiq.llm.client import ModelUnavailable

    planner, plan = _repaired_plan_meets(ModelUnavailable("model-host.internal timed out", kind="connect"))
    with pytest.raises(ModelUnavailable):
        plan()
    assert len(planner.contexts) == 2


def test_a_refusal_the_model_cannot_repair_stops_the_plan():
    """Superset not answering a measure's proof (`source_unavailable`) is nobody's statement to repair."""
    from mnemiq.superset.client import SupersetUnavailable
    from mnemiq.superset.measures import MeasureRefused, check_measure

    def execute(request):
        if "total_premium" in request["sql"]:
            return SupersetUnavailable("Superset did not answer", detail="timed out")
        return EMPTY

    tools = _tools(execute=execute)
    refused = check_measure(PREMIUM, source='"policy"', adapter=_adapter(tools), snapshot=SNAPSHOT, grants=GRANTS,
                            guard_fanout=False)
    assert isinstance(refused, MeasureRefused) and refused.repairable is False
    planner = FakePlanner(_valid_plan())
    refused = _plan(GOAL, planner, _tools(execute=execute))
    assert isinstance(refused, PlanRefused), refused
    assert len(planner.contexts) == 1, "not sent back to the model"
    assert _parts(refused)["measure 'total_premium'"].startswith("source_unavailable")
    # A refusal the model can repair still goes back.
    assert MeasureRefused("x").repairable is True


def test_a_time_column_is_a_date_or_a_timestamp():
    """A time column is a date or a timestamp by the type the snapshot records; a time of day or an unknown type
    is not one."""
    from mnemiq.superset.planner import _temporal

    assert [_temporal(t) for t in ("DATE", "timestamp without time zone", "timestamp with time zone", "TIMESTAMPTZ")] \
        == [True, True, True, True]
    assert [_temporal(t) for t in ("time without time zone", "VARCHAR", "INTEGER", None, "")] == [False] * 5
    plan = _valid_plan(charts=(_chart("by_region", Family.LINE, [WrittenRef(PREMIUM)], time_column="region",
                                      grain="P1M"),), measures=(PREMIUM,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "date" in _parts(refused)["chart 'by_region'"]


def test_a_virtual_datasets_name_carries_its_key():
    """Unique within a build by its key, and never longer than rc2's 250."""
    from mnemiq.superset.planner import _virtual_name

    assert _virtual_name("dataset_1", ["policy", "claim"]) == "mnemiq_dataset_1:policy_claim"
    long = [f"t{i}_" + "x" * 60 for i in range(6)]
    one, two = _virtual_name("dataset_1", long), _virtual_name("dataset_2", long)
    assert len(one) == 250 and len(two) == 250 and one != two and one.startswith("mnemiq_dataset_1:")


def test_two_joins_get_two_names():
    paid = _measure("paid", "Paid", "SUM(amount)", K.NUMBER, dataset_id=-1)
    offices = _measure("offices", "Offices", "COUNT(DISTINCT office)", K.COUNT, dataset_id=-2)
    other = "claim JOIN agent ON agent.agent_id = claim.claim_id"
    plan = BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"]),
                             _chart("offices", Family.BAR, [WrittenRef(offices)], dataset_id=-2, dims=["status"])),
                     measures=(paid, offices), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),
                               PlannedDataset(key="dataset_2", placeholder=-2, join=other)))
    checked = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(checked, BuildPlan), checked
    assert [d.name for d in checked.datasets] == ["mnemiq_dataset_1:policy_claim", "mnemiq_dataset_2:claim_agent"]


def test_a_checked_dataset_holds_only_what_the_check_wrote():
    """A planner's own sql, name or table_name never reaches the build."""
    rows = _measure("agents", "Agents", "COUNT(DISTINCT agent_id)", K.COUNT, dataset_id=-1)
    paid = _measure("paid", "Paid", "SUM(amount)", K.NUMBER, dataset_id=-2)
    plan = BuildPlan(charts=(_chart("agents", Family.BAR, [WrittenRef(rows)], dataset_id=-1, dims=["office"]),
                             _chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-2, dims=["region"])),
                     measures=(rows, paid), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, table="agent", table_name="payroll",
                                              sql="SELECT salary FROM payroll", name="planted"),
                               PlannedDataset(key="dataset_2", placeholder=-2, join=JOIN, table_name="payroll",
                                              sql="SELECT salary FROM payroll", name="planted")))
    checked = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(checked, BuildPlan), checked
    physical, virtual = checked.datasets
    assert physical == PlannedDataset(key="dataset_1", placeholder=-1, table="agent", table_name="agent")
    assert virtual.table_name is None and "payroll" not in virtual.sql and virtual.name != "planted"


def test_a_join_read_from_one_table_is_refused():
    """A virtual dataset is made only where a join is needed: a join whose measures and charts read one of its
    tables alone is refused."""
    total = _measure("total", "Total", "SUM(premium)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("total", Family.BAR, [WrittenRef(total)], dataset_id=-1, dims=["region"]),),
                     measures=(total,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "only" in _parts(refused)[f"the join {JOIN!r}"]


def test_the_planner_imports_no_private_name():
    """What the planner reads of another module is that module's public interface. Parsed with `ast`."""
    import ast
    import inspect

    from mnemiq.superset import planner

    def private(source: str) -> list[str]:
        return [f"{node.module}.{alias.name}" for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ImportFrom) for alias in node.names if alias.name.startswith("_")]

    assert private("from mnemiq.superset.grants import _read_datasets, resolve_grants") == [
        "mnemiq.superset.grants._read_datasets"]
    assert private("from mnemiq.superset.grants import read_datasets") == []
    assert private(inspect.getsource(planner)) == []


def test_two_planned_datasets_on_one_table_are_refused():
    """Two planned datasets on one table are refused: a table gets one dataset, which both charts plan on."""
    one = _measure("agents", "Agents", "COUNT(DISTINCT agent_id)", K.COUNT, dataset_id=-1)
    two = _measure("offices", "Offices", "COUNT(DISTINCT office)", K.COUNT, dataset_id=-2)
    plan = BuildPlan(charts=(_chart("agents", Family.BAR, [WrittenRef(one)], dataset_id=-1, dims=["office"]),
                             _chart("offices", Family.BAR, [WrittenRef(two)], dataset_id=-2, dims=["agent_id"])),
                     measures=(one, two), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, table="agent"),
                               PlannedDataset(key="dataset_2", placeholder=-2, table="agent")))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "the table 'agent'" in _parts(refused)


@pytest.mark.parametrize("item", [Filter("premium", ">", float("nan")), Filter("premium", "<", float("inf")),
                                  Filter("premium", "IN", [1.0, float("-inf")])])
def test_an_operand_json_cannot_carry_is_refused(item):
    """An operand is a finite number: JSON carries no NaN or Infinity."""
    plan = _valid_plan(charts=(_chart("by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                                      filters=[item]),), measures=(PREMIUM,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "finite" in _parts(refused)["chart 'by_region'"]


def test_a_measures_description_is_bounded():
    """A measure's description is at most 500 characters."""
    long = dataclasses.replace(PREMIUM, description="d" * 501)
    plan = _valid_plan(charts=(_chart("by_region", Family.BAR, [WrittenRef(long)], dims=["region"]),),
                       measures=(long,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "description" in _parts(refused)["measure 'total_premium'"]
    fits = dataclasses.replace(PREMIUM, description="d" * 500)
    plan = _valid_plan(charts=(_chart("by_region", Family.BAR, [WrittenRef(fits)], dims=["region"]),),
                       measures=(fits,))
    assert _plan(GOAL, FakePlanner(plan), _tools()) == plan


# ---------------------------------------------------------------- what a join is held to

def _joined_plan(join: str) -> BuildPlan:
    paid = _measure("paid", "Paid", "SUM(amount)", K.NUMBER, dataset_id=-1)
    return BuildPlan(charts=(_chart("paid", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"]),),
                     measures=(paid,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))


@pytest.mark.parametrize("join, why", [
    ("policy JOIN claim AS c(policy_id, claim_id) ON c.policy_id = policy.policy_id", "column list"),
    ("policy JOIN generate_series(1, 3) g ON true", "function"),
    ("policy JOIN agent() ON true", "function"),
    ("policy NATURAL JOIN claim", "NATURAL"),
    ("policy NATURAL JOIN claim ON claim.policy_id = policy.policy_id", "NATURAL"),
    ("policy JOIN ONLY claim ON claim.policy_id = policy.policy_id", "ONLY"),
    ("ONLY policy JOIN claim ON claim.policy_id = policy.policy_id", "ONLY"),
    ("policy JOIN claim TABLESAMPLE SYSTEM (10) ON claim.policy_id = policy.policy_id", "TABLESAMPLE"),
    (JOIN + " AND claim.policy_id = ANY (TABLE payroll)", "TABLE"),
    (JOIN + " AND claim.policy_id = agent_id", "agent_id"),
    (JOIN + " AND agent.agent_id = 1", "no table of the join"),
    (JOIN + " AND claim.office = 'x'", "office"),
    ("policy SEMI JOIN claim ON claim.policy_id = policy.policy_id", "LEFT JOIN"),
    ("policy ANTI JOIN claim ON claim.policy_id = policy.policy_id", "LEFT JOIN"),
    ("policy CROSS JOIN claim ON claim.policy_id = policy.policy_id", "LEFT JOIN"),
])
def test_a_joined_table_is_named_alone_and_joined_on_its_columns(join, why):
    """Each table is named alone, with an alias at most, and joined ON conditions on the joined tables' columns:
    a column list, a function, NATURAL, ONLY, TABLESAMPLE (which would make the dataset a sample), TABLE, a column
    of no joined table, and SEMI, ANTI and CROSS joins are refused."""
    refused = _plan(GOAL, FakePlanner(_joined_plan(join)), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert f"the join {join!r}" in _parts(refused)
    assert why in _parts(refused)[f"the join {join!r}"]


@pytest.mark.parametrize("written, rendered", [
    ("policy  join claim on claim.policy_id=policy.policy_id", JOIN),
    ("policy p LEFT JOIN claim c ON c.policy_id = p.policy_id",
     "policy AS p LEFT JOIN claim AS c ON c.policy_id = p.policy_id"),
])
def test_a_virtual_dataset_runs_the_join_as_it_was_decided(written, rendered):
    """The dataset's SQL is the parsed join's rendering, the text the dry run decided, not the raw
    text."""
    tools = _tools()
    checked = _plan(GOAL, FakePlanner(_joined_plan(written)), tools)
    assert isinstance(checked, BuildPlan), checked
    (made,) = checked.datasets
    assert made.join == written, "the plan keeps the join as the model wrote it"
    assert made.sql.endswith(" FROM " + rendered)
    assert any(f"AS paid FROM {rendered}" in sql for sql in tools.statements())


def test_virtual_dataset_names_cannot_meet():
    """A key holds no `:`, so the name's key part is the key and nothing else."""
    from mnemiq.superset.planner import _virtual_name

    assert _virtual_name("a", ["b", "c_d"]) != _virtual_name("a_b", ["c", "d"])


def test_table_in_a_condition_is_refused_even_where_a_column_is_named_table():
    """`= ANY (TABLE payroll)` reads as a column `TABLE`; where claim has a column of that name, only the rule on the
    name itself refuses it, since every joined table's column would otherwise match."""
    with_table = SNAPSHOT.model_copy(update={"columns": [
        *SNAPSHOT.columns, Column(id="claim.table", object_id="claim", name="table", data_type="VARCHAR")]})
    join = JOIN + " AND claim.policy_id = ANY (TABLE payroll)"
    refused = _plan(GOAL, FakePlanner(_joined_plan(join)), _tools(), snapshot=with_table)
    assert isinstance(refused, PlanRefused), refused
    assert f"the join {join!r}" in _parts(refused)
    assert "never read in a condition" in _parts(refused)[f"the join {join!r}"]


# ---------------------------------------------------------------- what rc2 does to a chart's name and a measure's label

# `CHANGED_OR_REFUSED_BY_RC2` and `KEPT_BY_RC2` (`tests/superset_shared.py`): the texts rc2's MCP stores as other
# text or refuses, and the ones it keeps as written.


@pytest.mark.parametrize("case", list(CHANGED_OR_REFUSED_BY_RC2))
def test_a_chart_title_rc2_would_change_or_refuse_is_refused(case):
    title = CHANGED_OR_REFUSED_BY_RC2[case]
    chart = dataclasses.replace(_chart("premium", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"]), title=title)
    refused = _plan(GOAL, FakePlanner(_valid_plan(charts=(chart,), measures=(PREMIUM,))), _tools())
    assert isinstance(refused, PlanRefused), case
    assert "title" in _parts(refused)["chart 'premium'"] and title not in refused.reason


@pytest.mark.parametrize("title", KEPT_BY_RC2)
def test_a_chart_title_rc2_keeps_is_accepted(title):
    chart = dataclasses.replace(_chart("premium", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"]), title=title)
    planned = _plan(GOAL, FakePlanner(_valid_plan(charts=(chart,), measures=(PREMIUM,))), _tools())
    assert isinstance(planned, BuildPlan), planned
    assert planned.charts[0].title == title


def test_the_text_rule_is_rc2s_own_patterns():
    """The patterns are rc2's, read from its source: `_DANGEROUS_URL_SCHEME_RE` and the handler check in
    `_check_dangerous_patterns`. One rule holds a chart's title and a measure's label."""
    from mnemiq.superset import measures

    assert measures._RC2_REFUSED[0].pattern == r"\b(javascript|vbscript|data):"
    assert measures._RC2_REFUSED[1].pattern == r"on\w+\s*="
    assert all(p.flags & re.IGNORECASE for p in measures._RC2_REFUSED)


def _labelled(label: str, *, family: Family = Family.BAR, currency: str | None = None) -> BuildPlan:
    measure = dataclasses.replace(PREMIUM, label=label, currency=currency)
    return _valid_plan(charts=(_chart("premium_by_region", family, [WrittenRef(measure)], dims=["region"]),),
                       measures=(measure,))


@pytest.mark.parametrize("case", list(CHANGED_OR_REFUSED_BY_RC2))
def test_a_measure_label_rc2_would_change_or_refuse_goes_back_for_repair(case):
    """A written measure's label meets the rule a chart's title does: rc2 saves an SQL metric's label as other text, or
    refuses it (`ColumnRef.sanitize_label`), and a table's per-column format and the value check find the measure by
    the label mnemiq sent. Refused in mnemiq's words, which do not repeat the label, it goes back to the model once,
    and the repaired plan passes."""
    label = CHANGED_OR_REFUSED_BY_RC2[case]
    planner = FakePlanner([_labelled(label), _labelled("Total premium")])
    assert _plan(GOAL, planner, _tools()) == _labelled("Total premium")
    [part] = planner.contexts[1].repair.refused
    assert part.part == "measure 'total_premium'" and part.repairable, part
    assert "label" in part.reason and label not in part.reason, part
    refused = _plan(GOAL, FakePlanner(_labelled(label)), _tools())
    assert isinstance(refused, PlanRefused) and "label" in _parts(refused)["measure 'total_premium'"]


@pytest.mark.parametrize("label", KEPT_BY_RC2)
def test_a_measure_label_rc2_keeps_is_accepted(label):
    assert _plan(GOAL, FakePlanner(_labelled(label)), _tools()) == _labelled(label)


def test_a_currency_tables_label_rc2_would_escape_is_refused_at_plan_time():
    """`Premium < 1y` on a currency table: rc2 would save the label as `Premium &lt; 1y`, the table's per-column
    currency would be set under `Premium < 1y`, and no format would be judged. The plan check refuses it."""
    refused = _plan(GOAL, FakePlanner(_labelled("Premium < 1y", family=Family.TABLE, currency="USD")), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "label" in _parts(refused)["measure 'total_premium'"]


# ---------------------------------------------------------------- the plan's JSON, declared beside its writer


def _every_part_plan() -> BuildPlan:
    planned = PlannedDataset(key="dataset_1", placeholder=-1, table="agent", table_name="agent")
    joined = PlannedDataset(key="dataset_2", placeholder=-2, join=JOIN, sql="SELECT 1", name="mnemiq_dataset_2:x")
    rows = _measure("agents", "Agents", "COUNT(DISTINCT agent_id)", K.COUNT, dataset_id=-1, currency="USD")
    return _valid_plan(reuse_candidates=(12,), reuse_reason="covers it", datasets=(planned, joined),
                       measures=(PREMIUM, CLAIMS, rows),
                       charts=(*_valid_plan().charts,
                               _chart("agents", Family.SANKEY, [WrittenRef(rows), SavedRef("count", K.COUNT)],
                                      dataset_id=-1, dims=["office"], time_column="start_date", grain="P1M",
                                      row_limit=10, source="office", target="office",
                                      filters=[Filter("office", "=", "north"), Filter("office", "IS NOT NULL"),
                                               Filter("office", "IN", ["a", "b"])])))


def _declared_paths(shape, path="") -> set[str]:
    from mnemiq.superset.planner import SCALAR

    if isinstance(shape, dict):
        found = set()
        for key, item in shape.items():
            found |= {f"{path}.{key}"} | _declared_paths(item, f"{path}.{key}")
        return found
    if isinstance(shape, list):
        return _declared_paths(shape[0], f"{path}[]")
    if isinstance(shape, tuple):
        return set().union(*(_declared_paths(item, path) for item in shape))
    assert shape is SCALAR
    return set()


def _written_paths(doc, path="") -> set[str]:
    if isinstance(doc, dict):
        found = set()
        for key, item in doc.items():
            found |= {f"{path}.{key}"} | _written_paths(item, f"{path}.{key}")
        return found
    if isinstance(doc, list):
        return set().union(set(), *(_written_paths(item, f"{path}[]") for item in doc))
    return set()


@pytest.mark.parametrize("plan", [_valid_plan(), _every_part_plan(), _valid_plan(charts=(), measures=())],
                         ids=["a plan", "every part", "nothing"])
def test_what_plan_to_json_writes_stays_within_its_declared_shape(plan):
    """The store keeps a plan only within `PLAN_JSON_SHAPE`, declared beside `plan_to_json`: whatever the writer
    writes passes."""
    doc = plan_to_json(plan)
    assert build_store._plan(doc) == json.loads(json.dumps(doc))


def test_every_key_the_shape_declares_is_one_the_writer_writes():
    """Nothing declared goes unwritten: the shape is the writer's keys, no wider."""
    from mnemiq.superset.planner import PLAN_JSON_SHAPE

    assert _declared_paths(PLAN_JSON_SHAPE) == _written_paths(plan_to_json(_every_part_plan()))


# ---------------------------------------------------------------- a percent is a fraction


def test_a_percent_scaled_by_100_is_found():
    for text in ("100.0 * COUNT(*) FILTER (WHERE x) / COUNT(*)", "COUNT(*) * 100 / NULLIF(COUNT(*), 0)",
                 "SUM(a) / SUM(b) / 0.01", "SUM(a)::numeric * 100::numeric / COUNT(*)",
                 "ROUND(100.0 * AVG(a), 1)", "AVG(x) * '100'::numeric", "AVG(x) * '100'", "AVG(x) / '0.01'::numeric",
                 "AVG(x) * -100", "AVG(x) * (10 * 10)", "AVG(x) / (1.0 / 100)", "SUM(a) / (SUM(b) / 100.0)",
                 "SUM(a) / SUM(b) / (1/100.0)", "SUM(a) / NULLIF(SUM(b) / 100.0, 0)",
                 "AVG(CASE WHEN x THEN 100 ELSE 0 END)", "SUM(CASE WHEN x THEN 100.0 ELSE 0 END) / COUNT(*)",
                 "SUM(a) / (SUM(b) * 0.01)", "AVG(CASE WHEN x THEN 0 ELSE 100 END)"):
        assert scales_by_100(text), text
    for text in ("COUNT(*) FILTER (WHERE x)::float / NULLIF(COUNT(*), 0)", "SUM(a) / 1000.0", "SUM(a) * 1000",
                 "SUM(a) * 'x100'", "not sql ((", "SUM(a) / 100", "ROUND(SUM(a), 100)",
                 "COUNT(*) FILTER (WHERE n > 100) / COUNT(*)::float", "AVG(CASE WHEN n > 100 THEN 1 END)",
                 "SUM(dollars) / SUM(cents / 100.0)", "SUM(a) * 'sNaN'", "SUM(a) / (1 / 0)",
                 "AVG(CASE WHEN x THEN 1.0 ELSE 0 END)"):
        assert not scales_by_100(text), text


def _percent_plan(expression: str, kind) -> BuildPlan:
    rate = _measure("win_rate", "Win rate", expression, kind=kind)
    return _valid_plan(measures=(rate,), charts=(_chart("rate_by_region", Family.BAR, [WrittenRef(rate)],
                                                        dims=["region"]),))


def test_a_percent_measure_multiplied_by_100_is_refused_at_plan_time():
    planner = FakePlanner(_percent_plan("100.0 * AVG(premium)", K.PERCENT))
    refused = _plan(GOAL, planner, _tools())
    assert isinstance(refused, PlanRefused)
    assert len(planner.contexts) == 2, "repairable: it went back once"
    assert _parts(refused)["measure 'win_rate'"] == (
        "a percent measure is a fraction from 0 to 1, which Superset shows as a percent: write it without "
        "multiplying by 100")
    assert planner.contexts[1].repair.refused[0].repairable
    # The same expression as a plain number is not refused for it.
    assert isinstance(_plan(GOAL, FakePlanner(_percent_plan("100.0 * AVG(premium)", K.NUMBER)), _tools()), BuildPlan)
    # A fraction as a percent passes.
    assert isinstance(_plan(GOAL, FakePlanner(_percent_plan("AVG(premium) / MAX(premium)", K.PERCENT)), _tools()),
                      BuildPlan)


TIME_AXIS_ROW_LIMIT = ("a chart on a time axis takes no row_limit: a row limit keeps the points its query ranks first, "
                       "the largest by its first measure, not the latest; to show a recent span, filter its time column "
                       "with >= a start date instead")
AVERAGE_PREMIUM = _measure("average_premium", "Average premium", "AVG(premium)")


@pytest.mark.parametrize("family", [Family.LINE, Family.BAR, Family.AREA, Family.SCATTER, Family.MIXED])
def test_a_row_limit_on_a_time_axis_chart_is_refused_with_a_time_range_repair(family):
    """A time axis's query orders by its first measure, then the axis sorts by time: under a row limit it keeps the
    largest points, not the latest. The plan goes back once with the reason, which points to a time filter instead."""
    refs = [WrittenRef(PREMIUM), WrittenRef(AVERAGE_PREMIUM)] if family is Family.MIXED else [WrittenRef(PREMIUM)]
    chart = _chart("premium_by_month", family, refs, time_column="start_date", grain="P1M", row_limit=12)
    planner = FakePlanner(_valid_plan(charts=(chart,), measures=(PREMIUM, AVERAGE_PREMIUM)))
    refused = _plan(GOAL, planner, _tools())
    assert isinstance(refused, PlanRefused)
    assert len(planner.contexts) == 2, "repairable: it went back once"
    assert _parts(refused)["chart 'premium_by_month'"] == TIME_AXIS_ROW_LIMIT
    assert planner.contexts[1].repair.refused[0].repairable


@pytest.mark.parametrize("family", [Family.LINE, Family.BAR, Family.AREA, Family.SCATTER])
def test_a_row_limit_on_a_date_first_dimension_is_refused_too(family):
    """With no time column, an x-y chart is drawn on its first dimension: a date there is a time axis all the same, so
    moving the date out of `time_column` does not get a row limit past the check."""
    chart = _chart("premium_by_day", family, [WrittenRef(PREMIUM)], dims=["start_date"], row_limit=12)
    planner = FakePlanner(_valid_plan(charts=(chart,), measures=(PREMIUM,)))
    refused = _plan(GOAL, planner, _tools())
    assert isinstance(refused, PlanRefused)
    assert _parts(refused)["chart 'premium_by_day'"] == TIME_AXIS_ROW_LIMIT


def test_a_row_limit_on_a_categorical_chart_still_plans():
    """A bar by region keeps its top groups under a row limit: a ranking, which the chart check reads as one."""
    chart = _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"], row_limit=20)
    assert isinstance(_plan(GOAL, FakePlanner(_valid_plan(charts=(chart,), measures=(PREMIUM,))), _tools()), BuildPlan)


def test_the_prompt_says_a_time_axis_takes_no_row_limit():
    model = ScriptedModel([_draft("SUM(premium)")])
    _plan(GOAL, LLMPlanner(model), _tools())
    assert '"row_limit": optional, never on a chart with a time_column or a date or timestamp first dimension,' \
        in model.sent[0]["system"]


def test_the_prompt_says_a_percent_is_a_fraction():
    model = ScriptedModel([_draft("SUM(premium)")])
    _plan(GOAL, LLMPlanner(model), _tools())
    system = model.sent[0]["system"]
    assert "a percent is a fraction from 0 to 1" in system and "never multiply it by 100" in system


def test_the_prompt_keeps_a_measures_condition_in_its_expression():
    """A condition that defines a measure (open, won) belongs in its expression, so the measure means its name on every
    chart, in the measures section and in a saved metric; a chart's filters only narrow that chart."""
    model = ScriptedModel([_draft("SUM(premium)")])
    _plan(GOAL, LLMPlanner(model), _tools())
    system = model.sent[0]["system"]
    assert "goes inside its expression" in system
    assert "a chart's filters narrow only what that chart shows" in system
    assert "filter them out on that chart as well" in system
    # No ELSE on the condition: an ELSE 0 is a row COUNT counts and AVG averages in, the same wrong meaning again.
    assert "with no ELSE on that condition" in system
    assert "an ELSE 0 there is counted by COUNT and averaged in by AVG" in system
    assert "at 0)" not in system
    # A negated condition also drops the NULL rows; the model is asked to keep them where they belong.
    assert "keep them with COALESCE or IS NULL OR" in system
    # The "no ELSE" is the condition that picks the measure's rows; a share keeps its own 0 for the rows it covers, or
    # a win rate among closed deals, its inner ELSE 0 dropped, reads 100%.
    assert "A condition that says which rows a written measure covers" in system
    assert "A share keeps its own 0 for the rows it covers" in system
    assert "AVG(CASE WHEN closed THEN CASE WHEN won THEN 1.0 ELSE 0 END END)" in system


def test_the_prompts_share_form_passes_the_plan_check_as_a_percent():
    """The share the prompt shows (its own 0 inside the condition that picks its rows) is one mnemiq's checks accept
    as a percent, so the prompt never asks for what the plan check refuses."""
    share = "AVG(CASE WHEN region = 'west' THEN CASE WHEN premium > 100 THEN 1.0 ELSE 0 END END)"
    assert isinstance(_plan(GOAL, FakePlanner(_percent_plan(share, K.PERCENT)), _tools()), BuildPlan)


def test_a_saved_metric_on_a_0_to_100_scale_cannot_be_planned_as_a_percent():
    metrics = {1: [_metric(1, "win_pct", "100.0 * AVG(premium)")]}
    chart = _chart("rate", Family.BIG_NUMBER, [SavedRef("win_pct", K.PERCENT)])
    refused = _plan(GOAL, FakePlanner(_valid_plan(measures=(), charts=(chart,))), _tools(metrics=metrics))
    assert isinstance(refused, PlanRefused)
    assert _parts(refused)["chart 'rate'"] == (
        "the saved metric 'win_pct' is on a 0 to 100 scale, so it cannot be shown as a percent: plan it with kind "
        "number, or write the fraction")
    number = _chart("rate", Family.BIG_NUMBER, [SavedRef("win_pct", K.NUMBER)])
    assert isinstance(_plan(GOAL, FakePlanner(_valid_plan(measures=(), charts=(number,))),
                            _tools(metrics=metrics)), BuildPlan)


# ---------------------------------------------------------------- a filter's value no row holds

# What a refusal for a value no row holds advises after the values it names, and after saying it names none.
def keep_the_goals(literal: str) -> str:
    return (f"; filter on one of these if it is the value the goal means, and keep {literal!r} only if the goal "
            "itself names it")


def keep_it(literal: str) -> str:
    return f"; keep {literal!r} only if the goal itself names it"


def _reason(column: str, literal: str, held: str, case: tuple[str, ...] = ()) -> str:
    """The refusal of a part that filters `column` on `literal`, no row the builder can read holding it; `held` is the
    values the column holds as the reason quotes them, `case` the named values that differ from `literal` only in
    letter case, in the order named."""
    hints = "".join(f"; {value!r} differs from {literal!r} only in letter case" for value in case)
    return (f"it filters {column} on {literal!r}, which no row the builder can read holds; {column} holds: {held}"
            f"{hints}{keep_the_goals(literal)}")


def _won_plan(literal: str) -> BuildPlan:
    won = _measure("won_amount", "Won amount", f"SUM(CASE WHEN status = '{literal}' THEN amount END)", K.CURRENCY,
                   dataset_id=2)
    return _valid_plan(measures=(won,), charts=(_chart("won", Family.BIG_NUMBER, [WrittenRef(won)], dataset_id=2),))


def test_a_measure_filtering_a_value_no_row_holds_is_refused_with_the_held_values():
    """The measure is decided, and then each value it filters on is asked of the source as the builder: 'Won' is held
    by no row, so the measure is refused, repairably, and the repair is told the values `claim.status` holds. Only
    then does the model see a value: the first prompt names none."""
    from mnemiq.superset.planner import RefusedPart, plan_prompt

    held = {("claim", "status"): ["WON", "LOST"]}
    planner = FakePlanner(_won_plan("Won"))
    refused = _plan(GOAL, planner, _tools(held=held))
    assert isinstance(refused, PlanRefused), refused
    reason = _reason("claim.status", "Won", "'LOST', 'WON'", case=("WON",))
    assert _parts(refused) == {"measure 'won_amount'": reason}
    assert len(planner.contexts) == 2, "repairable: it went back once"
    assert planner.contexts[1].repair.refused == (RefusedPart("measure 'won_amount'", reason, True),)
    assert reason in plan_prompt(GOAL, planner.contexts[1])[1]
    first = "\n".join(plan_prompt(GOAL, planner.contexts[0]))
    assert "WON" not in first and "LOST" not in first
    # The repair that filters a value the column holds passes, and so does the first plan where 'Won' is held.
    fixed = FakePlanner([_won_plan("Won"), _won_plan("WON")])
    assert isinstance(_plan(GOAL, fixed, _tools(held=held)), BuildPlan)
    assert isinstance(_plan(GOAL, FakePlanner(_won_plan("Won")), _tools(held={("claim", "status"): ["Won"]})),
                      BuildPlan)


def test_a_held_value_differing_only_in_case_is_named():
    refused = _plan(GOAL, FakePlanner(_won_plan("Won")), _tools(held={("claim", "status"): ["LOST", "WON"]}))
    assert _parts(refused) == {"measure 'won_amount'": (
        "it filters claim.status on 'Won', which no row the builder can read holds; claim.status holds: 'LOST', 'WON'; "
        "'WON' differs from 'Won' only in letter case; filter on one of these if it is the value the goal means, and "
        "keep 'Won' only if the goal itself names it")}


def test_two_case_only_variants_are_both_named():
    held = {("claim", "status"): ["WON", "won", "LOST"]}
    refused = _plan(GOAL, FakePlanner(_won_plan("Won")), _tools(held=held))
    assert _parts(refused) == {"measure 'won_amount'": _reason(
        "claim.status", "Won", "'LOST', 'WON', 'won'", case=("WON", "won"))}
    [reason] = _parts(refused).values()
    assert reason.index("'WON' differs") < reason.index("'won' differs")


def test_no_case_hint_without_a_case_only_match():
    refused = _plan(GOAL, FakePlanner(_won_plan("Nope")), _tools(held={("claim", "status"): ["LOST", "WON"]}))
    [reason] = _parts(refused).values()
    assert "letter case" not in reason
    assert reason == _reason("claim.status", "Nope", "'LOST', 'WON'")


def test_casefold_matches_a_non_ascii_case_variant():
    refused = _plan(GOAL, FakePlanner(_won_plan("straße")), _tools(held={("claim", "status"): ["STRASSE"]}))
    [reason] = _parts(refused).values()
    assert "'STRASSE' differs from 'straße' only in letter case" in reason


def test_a_measure_whose_column_holds_more_than_20_values_names_20_and_more():
    held = {("claim", "status"): [f"S{i:02d}" for i in range(25)]}
    refused = _plan(GOAL, FakePlanner(_won_plan("Won")), _tools(held=held))
    assert _parts(refused) == {"measure 'won_amount'": _reason(
        "claim.status", "Won", ", ".join(f"'S{i:02d}'" for i in range(20)) + " and more")}


def test_a_value_the_source_does_not_answer_for_refuses_nothing():
    """A probe Superset does not answer leaves the value unchecked, as on the ask path: nothing is refused for it."""
    tools = _tools(held={})
    tools.handlers["execute_sql"] = lambda request: (
        {"success": False, "error": "relation does not exist", "error_type": "GENERIC_DB_ENGINE_ERROR"}
        if _PROBE.fullmatch(request["sql"]) else _value_answer(request["sql"], {}) or EMPTY)
    assert isinstance(_plan(GOAL, FakePlanner(_won_plan("Won")), tools), BuildPlan)
    assert any(_PROBE.fullmatch(sql) for sql in tools.statements()), "the value was asked"
    assert not any(_HELD_READ.fullmatch(sql) for sql in tools.statements()), "and no values were read"


def test_a_failed_read_of_the_held_values_still_refuses():
    """The source said no row holds the value, so the part is refused even when the read of the values the column
    holds then fails; the reason says the source did not say which."""
    held = {("claim", "status"): ["WON"]}
    tools = _tools(held=held)
    tools.handlers["execute_sql"] = lambda request: (
        {"success": False, "error": "statement timeout", "error_type": "GENERIC_DB_ENGINE_ERROR"}
        if _HELD_READ.fullmatch(request["sql"]) else _value_answer(request["sql"], held) or EMPTY)
    refused = _plan(GOAL, FakePlanner(_won_plan("Won")), tools)
    assert _parts(refused) == {"measure 'won_amount'": "it filters claim.status on 'Won', which no row the builder "
                                                       "can read holds; the source did not say which values it holds"
                                                       + keep_it("Won")}


def test_a_value_compared_with_an_expression_is_never_probed():
    """`LOWER(status) = 'won'` is a value the grounding does not place, as on the ask path: nothing is asked or
    refused for it."""
    plan = _won_plan("x")
    lowered = dataclasses.replace(plan.measures[0], expression="SUM(CASE WHEN LOWER(status) = 'won' THEN amount END)")
    plan = _valid_plan(measures=(lowered,), charts=(_chart("won", Family.BIG_NUMBER, [WrittenRef(lowered)],
                                                           dataset_id=2),))
    tools = _tools(held={("claim", "status"): ["WON"]})
    assert isinstance(_plan(GOAL, FakePlanner(plan), tools), BuildPlan)
    assert not any(_PROBE.fullmatch(sql) or _HELD_READ.fullmatch(sql) for sql in tools.statements())


def test_a_chart_filter_on_a_value_no_row_holds_is_refused():
    """A chart's `NOT IN` filter on a value no row holds: the negation is vacuous, so it is refused as `=` and `IN`
    are, naming the values the column holds. A value the column holds and a non-string operand are not refused."""
    chart = _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                   filters=[Filter("region", "NOT IN", ["West", "east"]), Filter("premium", ">", 100),
                            Filter("region", "LIKE", "W%")])
    held = {("policy", "region"): ["west", "east", "central"]}
    planner = FakePlanner(_valid_plan(measures=(PREMIUM,), charts=(chart,)))
    refused = _plan(GOAL, planner, _tools(held=held))
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"chart 'premium_by_region'": _reason("policy.region", "West", "'central', 'east', 'west'",
                                                                                case=("west",))}
    assert len(planner.contexts) == 2 and planner.contexts[1].repair.refused[0].repairable
    for op, operand in (("=", "west"), ("!=", "east"), ("IN", ["west", "central"])):
        ok = _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                    filters=[Filter("region", op, operand)])
        assert isinstance(_plan(GOAL, FakePlanner(_valid_plan(measures=(PREMIUM,), charts=(ok,))),
                                _tools(held=held)), BuildPlan), (op, operand)


def test_a_chart_filter_on_a_joined_column_is_probed_on_its_own_table():
    """On a join, the dataset names a column two tables share after its table (`claim_created_at`); a filter on one
    the join names alone (`status`) is asked of the table it comes from, under that table's own name for it."""
    paid = _measure("paid", "Paid", "SUM(amount)", K.CURRENCY, dataset_id=-1)
    planned = PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN)
    chart = _chart("paid_by_region", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"],
                   filters=[Filter("status", "=", "Open"), Filter("claim_policy_id", "IN", ["7"])])
    plan = BuildPlan(charts=(chart,), measures=(paid,), reuse_candidates=(), datasets=(planned,))
    tools = _tools(held={("claim", "status"): ["open", "closed"], ("claim", "policy_id"): ["7"]})
    refused = _plan(GOAL, FakePlanner(plan), tools)
    assert _parts(refused) == {"chart 'paid_by_region'": _reason("claim.status", "Open", "'closed', 'open'", case=("open",))}
    probed = {(m["table"], m["column"]) for sql in tools.statements() if (m := _PROBE.fullmatch(sql))}
    assert probed == {("claim", "status"), ("claim", "policy_id")}


def test_a_filter_on_a_column_the_charts_rows_lack_is_never_probed():
    """A chart refused for a filter on a column its rows lack is not probed: the column has no table to ask."""
    chart = _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                   filters=[Filter("salary", "=", "x")])
    tools = _tools(held={})
    refused = _plan(GOAL, FakePlanner(_valid_plan(measures=(PREMIUM,), charts=(chart,))), tools)
    assert "not a column" in _parts(refused)["chart 'premium_by_region'"]
    assert not any(_PROBE.fullmatch(sql) or _HELD_READ.fullmatch(sql) for sql in tools.statements())


@pytest.mark.parametrize(("key", "title", "why"), [
    ("Bad Key", "Premium", "a chart's key is 1 to 40 of a-z, 0-9 and _"),
    ("premium_by_region", "", "a chart's title is 1 to"),
])
def test_a_chart_its_own_checks_refuse_is_never_probed(key, title, why):
    """A chart that fails one of its own checks is not probed, even for a filter on a value no row holds: only a chart
    whose own checks all pass costs the source a probe, and the refusal gives the chart's own reason alone."""
    chart = dataclasses.replace(_chart(key, Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                                       filters=[Filter("region", "=", "West")]), title=title)
    tools = _tools(held={("policy", "region"): ["central", "east", "west"]})
    refused = _plan(GOAL, FakePlanner(_valid_plan(measures=(PREMIUM,), charts=(chart,))), tools)
    [part] = refused.parts
    assert part.part == f"chart {key!r}" and part.reason.startswith(why), part
    assert not any(_PROBE.fullmatch(sql) or _HELD_READ.fullmatch(sql) for sql in tools.statements())


def test_a_percent_scaled_by_100_and_filtering_a_value_no_row_holds_is_told_both_in_one_round():
    """A measure the percent rule refuses is still checked for its filter values, so both reasons go back in its one
    repair rather than the second costing a round the plan no longer has."""
    rate = _measure("won_rate", "Won rate", "100.0 * AVG(CASE WHEN status = 'Won' THEN 1 ELSE 0 END)", K.PERCENT,
                    dataset_id=2)
    plan = _valid_plan(measures=(rate,), charts=(_chart("won_rate", Family.BIG_NUMBER, [WrittenRef(rate)],
                                                        dataset_id=2),))
    planner = FakePlanner(plan)
    _plan(GOAL, planner, _tools(held={("claim", "status"): ["LOST", "WON"]}))
    assert [part.reason for part in planner.contexts[1].repair.refused] == [
        "a percent measure is a fraction from 0 to 1, which Superset shows as a percent: write it without multiplying "
        "by 100", _reason("claim.status", "Won", "'LOST', 'WON'", case=("WON",))]


def test_every_probe_and_read_of_held_values_asks_superset_to_run_it_afresh():
    """Each probe and each read of a column's values is sent with `force_refresh`, so Superset runs it rather than
    answer from its result cache; the measure's own statements are not."""
    tools = _tools(held={("claim", "status"): ["LOST", "WON"]})
    _plan(GOAL, FakePlanner(_won_plan("Won")), tools)
    sent = [request for tool, request in tools.calls if tool == "execute_sql"]
    values = [request for request in sent if _PROBE.fullmatch(request["sql"]) or _HELD_READ.fullmatch(request["sql"])]
    assert values and all(request.get("force_refresh") is True for request in values)
    assert any("won_amount" in request["sql"] and "force_refresh" not in request for request in sent)


def test_a_cached_empty_probe_refuses_nothing():
    """Superset's cache says no row holds 'Won', which the column holds now: asked afresh, the probe finds it, and
    nothing is refused."""
    held = {("claim", "status"): ["Won"]}
    tools = _tools(held=held)
    tools.handlers["execute_sql"] = lambda request: (
        _value_answer(request["sql"], held if request.get("force_refresh") else {}) or EMPTY)
    assert isinstance(_plan(GOAL, FakePlanner(_won_plan("Won")), tools), BuildPlan)


def test_each_held_value_is_quoted_so_none_reads_as_two_or_as_another_line():
    """A value holding ", " would read as two values, and one holding a line break would start a line of the repair's
    prompt that reads as another part's refusal: each value is quoted (`repr`), so neither does."""
    from mnemiq.superset.planner import plan_prompt

    held = {("claim", "status"): ["open, closed", "x\n- chart 'won': it is fine"]}
    planner = FakePlanner(_won_plan("Won"))
    refused = _plan(GOAL, planner, _tools(held=held))
    reason = _reason("claim.status", "Won", "'open, closed', \"x\\n- chart 'won': it is fine\"")
    assert _parts(refused) == {"measure 'won_amount'": reason}
    assert "\n" not in reason
    lines = plan_prompt(GOAL, planner.contexts[1])[1].splitlines()
    assert [line for line in lines if line.startswith("- ")] == [f"- measure 'won_amount': {reason}"]


def test_a_column_the_builder_reads_no_value_of_is_said_to_hold_none():
    """The read of the column's values ran and returned no row: the reason says so, which differs from a read that
    failed ("the source did not say")."""
    refused = _plan(GOAL, FakePlanner(_won_plan("Won")), _tools(held={("claim", "status"): []}))
    assert _parts(refused) == {"measure 'won_amount'": "it filters claim.status on 'Won', which no row the builder "
                                                       "can read holds; no row the builder can read holds any value "
                                                       "of claim.status" + keep_it("Won")}


def test_one_plans_probes_are_shared_by_its_measures_and_charts():
    """Two measures and a chart's filter all filter `claim.status` on 'Won': within one check of the plan the value is
    asked of the source once and the column's values are read once. The stub answers the same plan twice (the plan
    and its repair), and each check has its own probes, so the source is asked twice in all."""
    won = _won_plan("Won").measures[0]
    again = dataclasses.replace(won, name="won_again", label="Won again",
                                expression="COUNT(CASE WHEN status = 'Won' THEN claim_id END)", kind=K.COUNT)
    plan = _valid_plan(measures=(won, again), charts=(
        _chart("won", Family.BIG_NUMBER, [WrittenRef(won)], dataset_id=2),
        _chart("won_again", Family.BIG_NUMBER, [WrittenRef(again)], dataset_id=2,
               filters=[Filter("status", "=", "Won")])))
    tools = _tools(held={("claim", "status"): ["WON", "LOST"]})
    refused = _plan(GOAL, FakePlanner(plan), tools)
    reason = _reason("claim.status", "Won", "'LOST', 'WON'", case=("WON",))
    assert _parts(refused) == {"measure 'won_amount'": reason, "measure 'won_again'": reason,
                               "chart 'won_again'": reason}
    probes = [m["value"] for sql in tools.statements() if (m := _PROBE.fullmatch(sql))]
    reads = [sql for sql in tools.statements() if _HELD_READ.fullmatch(sql)]
    assert probes == ["Won", "Won"] and len(reads) == 2, (probes, len(reads))


def test_once_a_plans_probe_time_is_spent_nothing_more_is_asked_or_refused(monkeypatch):
    """Each read takes 6 s on the plan's clock and the plan may spend `PROBE_BUDGET_S` (10 s): the first measure's
    probe and its column's read spend it, so the second measure's value is never asked, and only the first is
    refused."""
    from mnemiq.superset import planner as planner_module
    from mnemiq.superset.literals import Probes

    clock = [0.0]
    held = {("claim", "status"): ["WON"], ("policy", "region"): ["west"]}

    def execute(request):
        if _PROBE.fullmatch(request["sql"]) or _HELD_READ.fullmatch(request["sql"]):
            clock[0] += 6.0
        return _value_answer(request["sql"], held) or EMPTY

    monkeypatch.setattr(planner_module, "Probes", lambda: Probes(clock=lambda: clock[0]))
    won = _won_plan("Won").measures[0]
    west = _measure("west_premium", "West premium", "SUM(CASE WHEN region = 'West' THEN premium END)")
    plan = _valid_plan(measures=(won, west), charts=(
        _chart("won", Family.BIG_NUMBER, [WrittenRef(won)], dataset_id=2),
        _chart("west", Family.BIG_NUMBER, [WrittenRef(west)])))
    tools = _tools(held=held)
    tools.handlers["execute_sql"] = execute
    refused = _plan(GOAL, FakePlanner(plan), tools)
    assert set(_parts(refused)) == {"measure 'won_amount'"}
    assert not any((m := _PROBE.fullmatch(sql)) and m["value"] == "West" for sql in tools.statements())


# ---------------------------------------------------------------- a negated filter on a column that holds NULL


def _null_reason(column: str, op: str) -> str:
    """The refusal of a chart that filters `column` with the negated `op`, a row the builder can read holding NULL
    there."""
    return (f"it filters {column} with {op}, which leaves out every row where {column} is NULL, and rows the builder "
            f"can read hold NULL there; replace the filter with IS NULL if only those rows are meant, add {column} IS NOT "
            "NULL if leaving them out is meant, or filter on another column")


def _both_reason(column: str, op: str) -> str:
    """The refusal of a chart that filters `column` with the negated `op` and with IS NULL."""
    return (f"it filters {column} with both {op} and IS NULL, and no row can meet both; keep IS NULL alone if the rows "
            f"where {column} is NULL are meant, otherwise drop it")


# Each negated operator, with an operand the column holds, so no value of it is refused.
NEGATED = [("!=", "west"), ("NOT IN", ["west", "east"]), ("NOT LIKE", "w%")]
REGIONS_HELD = {("policy", "region"): ["central", "east", "west"]}


def _region_filtered(*filters: Filter, key: str = "premium_by_region", title: str | None = None) -> BuildPlan:
    chart = _chart(key, Family.BAR, [WrittenRef(PREMIUM)], dims=["region"], filters=filters)
    if title is not None:
        chart = dataclasses.replace(chart, title=title)
    return _valid_plan(measures=(PREMIUM,), charts=(chart,))


def _null_probes(tools: Tools) -> list[tuple[str, str]]:
    return [(m["table"], m["column"]) for sql in tools.statements() if (m := _NULL_PROBE.fullmatch(sql))]


def test_the_negated_operators_are_the_planners_own():
    from mnemiq.superset.literals import NEGATED_OPERATORS

    assert NEGATED_OPERATORS == {"!=", "NOT IN", "NOT LIKE"} and NEGATED_OPERATORS <= set(FILTER_OPERATORS)


@pytest.mark.parametrize(("op", "operand"), NEGATED)
def test_a_negated_filter_on_a_column_holding_null_is_refused(op, operand):
    """`region != 'west'` leaves out the rows where region is NULL as well as west's: SQL reads NULL != 'west' as
    unknown. A row the builder can read holds NULL there, so the chart is refused, repairably, with the three ways
    out; asked once per check of the plan, the plan and its repair."""
    from mnemiq.superset.planner import RefusedPart

    planner = FakePlanner(_region_filtered(Filter("region", op, operand)))
    tools = _tools(held=REGIONS_HELD, nulls={("policy", "region")})
    refused = _plan(GOAL, planner, tools)
    assert isinstance(refused, PlanRefused), refused
    assert refused.parts == (RefusedPart("chart 'premium_by_region'", _null_reason("region", op), True),)
    assert len(planner.contexts) == 2 and planner.contexts[1].repair.refused == refused.parts
    assert _null_probes(tools) == [("policy", "region")] * 2


@pytest.mark.parametrize(("op", "operand"), NEGATED)
def test_a_negated_filter_on_a_column_holding_no_null_is_built(op, operand):
    tools = _tools(held=REGIONS_HELD)
    assert isinstance(_plan(GOAL, FakePlanner(_region_filtered(Filter("region", op, operand))), tools), BuildPlan)
    assert _null_probes(tools) == [("policy", "region")]


def test_a_chart_that_states_its_null_handling_is_not_probed():
    """`region != 'west' AND region IS NOT NULL` says the NULL rows are meant to be left out: nothing to ask, nothing
    refused. Said of another column, `IS NOT NULL` or `IS NULL` says nothing of region's, which is asked."""
    tools = _tools(held=REGIONS_HELD, nulls={("policy", "region")})
    plan = _region_filtered(Filter("region", "!=", "west"), Filter("region", "IS NOT NULL", None))
    assert isinstance(_plan(GOAL, FakePlanner(plan), tools), BuildPlan)
    assert _null_probes(tools) == []
    for stated in ("IS NOT NULL", "IS NULL"):
        elsewhere = _region_filtered(Filter("region", "!=", "west"), Filter("premium", stated, None))
        refused = _plan(GOAL, FakePlanner(elsewhere), tools)
        assert _parts(refused) == {"chart 'premium_by_region'": _null_reason("region", "!=")}, stated


@pytest.mark.parametrize(("op", "operand"), NEGATED)
def test_a_negated_filter_beside_is_null_on_its_column_is_refused_without_a_probe(op, operand):
    """A chart's filters all apply, and a NULL never meets a negation, so `region != 'west' AND region IS NULL`
    selects no row. The chart is refused, repairably, from the plan alone: nothing is asked of the source, neither the
    column's NULLs nor the filter's values."""
    from mnemiq.superset.planner import RefusedPart

    planner = FakePlanner(_region_filtered(Filter("region", op, operand), Filter("region", "IS NULL", None)))
    tools = _tools(held=REGIONS_HELD, nulls={("policy", "region")})
    refused = _plan(GOAL, planner, tools)
    assert isinstance(refused, PlanRefused), refused
    assert refused.parts == (RefusedPart("chart 'premium_by_region'", _both_reason("region", op), True),)
    assert len(planner.contexts) == 2 and planner.contexts[1].repair.refused == refused.parts
    assert not any(_NULL_PROBE.fullmatch(sql) or _PROBE.fullmatch(sql) or _HELD_READ.fullmatch(sql)
                   for sql in tools.statements())


def test_one_plans_charts_share_one_null_probe_per_column():
    """Two charts negate region, which holds NULL: within one check of the plan the column is asked once. The stub
    answers the same plan twice (the plan and its repair), and each check has its own probes, so twice in all."""
    first = _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                   filters=[Filter("region", "!=", "west")])
    second = _chart("premium_outside_east", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                    filters=[Filter("region", "NOT IN", ["east"])])
    tools = _tools(held=REGIONS_HELD, nulls={("policy", "region")})
    refused = _plan(GOAL, FakePlanner(_valid_plan(measures=(PREMIUM,), charts=(first, second))), tools)
    assert _parts(refused) == {"chart 'premium_by_region'": _null_reason("region", "!="),
                               "chart 'premium_outside_east'": _null_reason("region", "NOT IN")}
    assert _null_probes(tools) == [("policy", "region")] * 2


def test_once_a_plans_probe_time_is_spent_no_null_probe_is_sent(monkeypatch):
    """Each read takes 6 s on the plan's clock and the plan may spend `PROBE_BUDGET_S` (10 s): the measure's probe
    and its column's read spend it, so the chart that then negates region is never asked whether region holds NULL,
    and is not refused for it."""
    from mnemiq.superset import planner as planner_module
    from mnemiq.superset.literals import Probes

    clock = [0.0]
    held = {("claim", "status"): ["WON"], **REGIONS_HELD}
    nulls = {("policy", "region")}

    def execute(request):
        sql = request["sql"]
        if _PROBE.fullmatch(sql) or _HELD_READ.fullmatch(sql) or _NULL_PROBE.fullmatch(sql):
            clock[0] += 6.0
        return _null_answer(sql, nulls) or _value_answer(sql, held) or EMPTY

    monkeypatch.setattr(planner_module, "Probes", lambda: Probes(clock=lambda: clock[0]))
    won = _won_plan("Won").measures[0]
    plan = _valid_plan(measures=(won, PREMIUM), charts=(
        _chart("won", Family.BIG_NUMBER, [WrittenRef(won)], dataset_id=2),
        _chart("premium_by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
               filters=[Filter("region", "!=", "west")])))
    tools = _tools()
    tools.handlers["execute_sql"] = execute
    refused = _plan(GOAL, FakePlanner(plan), tools)
    assert set(_parts(refused)) == {"measure 'won_amount'"}
    assert _null_probes(tools) == []


POSITIVE = {"=": "west", "IN": ["west"], ">": "m", "<": "m", ">=": "m", "<=": "m", "LIKE": "w%", "ILIKE": "W%",
            "IS NULL": None, "IS NOT NULL": None}


@pytest.mark.parametrize("op", list(POSITIVE))
def test_a_positive_filter_is_never_probed_for_null(op):
    """Every operator but the three negated ones selects no NULL row by asking for it, or asks for them by name."""
    from mnemiq.superset.literals import NEGATED_OPERATORS

    assert set(POSITIVE) == set(FILTER_OPERATORS) - NEGATED_OPERATORS
    tools = _tools(held=REGIONS_HELD, nulls={("policy", "region")})
    assert isinstance(_plan(GOAL, FakePlanner(_region_filtered(Filter("region", op, POSITIVE[op]))), tools),
                      BuildPlan)
    assert _null_probes(tools) == []


def test_a_null_probe_that_fails_refuses_nothing():
    """Superset does not answer the probe: the source said nothing, so nothing is refused."""
    tools = _tools(held=REGIONS_HELD)
    tools.handlers["execute_sql"] = lambda request: (
        {"success": False, "error": "statement timeout", "error_type": "GENERIC_DB_ENGINE_ERROR"}
        if _NULL_PROBE.fullmatch(request["sql"]) else _value_answer(request["sql"], REGIONS_HELD) or EMPTY)
    assert isinstance(_plan(GOAL, FakePlanner(_region_filtered(Filter("region", "!=", "west"))), tools), BuildPlan)
    assert _null_probes(tools) == [("policy", "region")]


@pytest.mark.parametrize(("key", "title", "why"), [
    ("Bad Key", "Premium", "a chart's key is 1 to 40 of a-z, 0-9 and _"),
    ("premium_by_region", "", "a chart's title is 1 to"),
])
def test_a_chart_its_own_checks_refuse_is_not_probed_for_null(key, title, why):
    """Only a chart whose own checks all pass costs the source a probe: one they refuse is refused for its own reason
    alone."""
    tools = _tools(held=REGIONS_HELD, nulls={("policy", "region")})
    refused = _plan(GOAL, FakePlanner(_region_filtered(Filter("region", "!=", "west"), key=key, title=title)), tools)
    [part] = refused.parts
    assert part.part == f"chart {key!r}" and part.reason.startswith(why), part
    assert _null_probes(tools) == []


def test_absent_values_and_a_null_column_give_both_reasons_in_one_refusal():
    """`region NOT IN ('Nope')`: no row holds 'Nope', so the filter excludes nothing it names, and the rows where
    region is NULL are left out. Both reasons go back on the one chart, in the one repair."""
    from mnemiq.superset.planner import RefusedPart

    planner = FakePlanner(_region_filtered(Filter("region", "NOT IN", ["Nope"])))
    refused = _plan(GOAL, planner, _tools(held=REGIONS_HELD, nulls={("policy", "region")}))
    part = "chart 'premium_by_region'"
    both = (RefusedPart(part, _reason("policy.region", "Nope", "'central', 'east', 'west'")),
            RefusedPart(part, _null_reason("region", "NOT IN")))
    assert refused.parts == both and planner.contexts[1].repair.refused == both


def test_a_negated_filter_on_a_joined_column_probes_the_table_it_reads():
    """On a join, `policy_created_at` is policy's `created_at` (both tables have one) and `status` is claim's alone:
    each is asked of the table and column it reads, and the refusal names the column as the chart wrote it."""
    paid = _measure("paid", "Paid", "SUM(amount)", K.CURRENCY, dataset_id=-1)
    planned = PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN)
    chart = _chart("paid_by_region", Family.BAR, [WrittenRef(paid)], dataset_id=-1, dims=["region"],
                   filters=[Filter("policy_created_at", "NOT LIKE", "2024%"), Filter("status", "!=", "open")])
    plan = BuildPlan(charts=(chart,), measures=(paid,), reuse_candidates=(), datasets=(planned,))
    tools = _tools(held={("claim", "status"): ["closed", "open"]}, nulls={("policy", "created_at")})
    refused = _plan(GOAL, FakePlanner(plan), tools)
    assert _parts(refused) == {"chart 'paid_by_region'": _null_reason("policy_created_at", "NOT LIKE")}
    assert _null_probes(tools) == [("policy", "created_at"), ("claim", "status")] * 2


# ---------------------------------------------------------------- the percent rule fails safe


def test_a_constant_past_decimals_range_is_not_a_scale_and_raises_nothing():
    for text in ("SUM(a) * 1e1000000", "SUM(a) * '1e1000000'", "AVG(CASE WHEN x THEN 1e1000000 END)",
                 "SUM(a) * -1e1000000", "SUM(a) / 1e1000000", "SUM(a) * (1e1000000 * 1e1000000)",
                 "SUM(a) / (SUM(b) / 1e1000000)", "SUM(a) * 1e99999999999999999999"):
        assert scales_by_100(text) is False, text
    # A real factor of 100 beside such a constant is still found.
    assert scales_by_100("SUM(a) * 100 + SUM(b) * 1e1000000")


def test_a_very_long_constant_chain_is_not_a_scale_and_raises_nothing():
    chain = " + ".join(["1"] * 3000)
    assert scales_by_100(f"SUM(a) * ({chain})") is False


# ---------------------------------------------------------------- a joined table's saved metric, carried onto a join

# The profile's counts for the joins' keys: policy_id holds each value once in policy and repeats in claim, so
# `policy JOIN claim` repeats policy's rows and leaves claim's at their own grain.
_KEY_COUNTS = {"policy.policy_id": (10, 10), "claim.policy_id": (30, 8), "claim.claim_id": (30, 30),
               "agent.agent_id": (5, 5)}
KEYED = SNAPSHOT.model_copy(update={"columns": [
    column.model_copy(update={"row_count": _KEY_COUNTS[column.id][0], "distinct_count": _KEY_COUNTS[column.id][1],
                              "null_count": 0}) if column.id in _KEY_COUNTS else column
    for column in SNAPSHOT.columns]})
GUARDED = Settings(guard_fanout=True, superset_currency="USD")

OPEN_PAID = "SUM(amount) FILTER (WHERE status = 'open' AND created_at >= TIMESTAMPTZ '2026-01-01 00:00:00+00')"
# That metric decided over the join, each column qualified by claim's place in it, and as the joined dataset names its
# columns: claim's created_at is `claim_created_at` there, since policy has a created_at too.
OPEN_PAID_DECIDED = ('SUM(claim."amount") FILTER(WHERE claim."status" = \'open\' AND claim."created_at" >= '
                     "CAST('2026-01-01 00:00:00+00' AS TIMESTAMPTZ))")
OPEN_PAID_SHOWN = ('SUM("amount") FILTER(WHERE "status" = \'open\' AND "claim_created_at" >= '
                   "CAST('2026-01-01 00:00:00+00' AS TIMESTAMPTZ))")
SAVED_ON_TABLES = {
    1: [_metric(1, "premium_total", "SUM(premium)", verbose_name="Premium"),
        _metric(1, "policies", "COUNT(policy_id)", verbose_name="Policies")],
    2: [_metric(2, "open_paid", OPEN_PAID, verbose_name="Open claims paid"),
        _metric(2, "paid", "SUM(claim.amount)"),
        _metric(2, "count", "COUNT(*)"),
        _metric(2, "open_share", "AVG(CASE WHEN status = 'open' THEN 100 ELSE 0 END)"),
        _metric(2, "beyond", "SUM(amount) + (SELECT MAX(premium) FROM policy)"),
        _metric(2, "each_amount", "amount"),
        _metric(2, "by_status", "COUNT(claim_id)", verbose_name="status"),
        _metric(2, "tally", "COUNT(claim_id)", verbose_name="COUNT"),
        _metric(2, "flagged", "COUNT(claim_id)", verbose_name="Open & flagged"),
        _metric(2, "marked", "COUNT(claim_id)", verbose_name="Claims {{ period }}"),
        _metric(2, "blank", "COUNT(claim_id)", verbose_name=""),
        _metric(2, "long_named", "COUNT(claim_id)", verbose_name="Claims " + "x" * 114)],
}
CERTIFIED = {2: [{"metric_name": "open_paid", "currency": {"symbol": "EUR", "symbolPosition": "prefix"},
                  "certified_by": "finance"}]}
OPEN_PAID_REF = SavedRef("open_paid", K.CURRENCY, table="claim", handle="open_paid_by_region")
CLOSED_ONLY = {("claim", "status"): ["closed"]}
CARRIED_DESCRIPTION = "The saved metric 'open_paid' of 'claim', carried onto the join."


def _carried_plan(*refs, join: str = JOIN, dims=("region",), filters=()) -> BuildPlan:
    """One bar on `join` plotting `refs`, the written ones among them the plan's measures."""
    return BuildPlan(charts=(_chart("carried", Family.BAR, refs, dataset_id=-1, dims=dims, filters=filters),),
                     measures=tuple(ref.measure for ref in refs if isinstance(ref, WrittenRef)), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))


def _carried(*refs, tools: Tools | None = None, settings: Settings = GUARDED, snapshot: Snapshot = KEYED, **plan):
    tools = tools or _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED)
    return _plan(GOAL, FakePlanner(_carried_plan(*refs, **plan)), tools, snapshot=snapshot, settings=settings)


def _seen(tools: Tools):
    """The context a plan made on `tools` hands its planner."""
    recorder = FakePlanner(_valid_plan())
    _plan(GOAL, recorder, tools)
    return recorder.contexts[0]


def _read_through(dataset_sql: str, expression: str) -> str:
    """`expression`, a measure on the joined dataset, with each column it names replaced by the joined column the
    dataset's SELECT projects under that name: what the measure computes over the join."""
    projected = {item.alias: item.this for item in sqlglot.parse_one(dataset_sql, read="postgres").expressions}
    return sqlglot.parse_one(f"SELECT {expression}", read="postgres").expressions[0].transform(
        lambda node: projected[node.name].copy() if isinstance(node, exp.Column) else node).sql(dialect="postgres")


def test_a_joined_tables_saved_metric_is_carried_onto_the_join():
    """claim's saved metric on `policy JOIN claim`, naming created_at (which policy has too) and filtering on a value:
    decided over the join with each column qualified by claim's place in it, under the fan-out guard, and carried to
    the chart as a written measure naming the joined dataset's columns. Its value is the data team's: never probed."""
    tools = _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED, held=CLOSED_ONLY)
    checked = _carried(OPEN_PAID_REF, tools=tools)
    assert isinstance(checked, BuildPlan), checked
    carried = Measure(name="open_paid_by_region", label="Open claims paid", expression=OPEN_PAID_SHOWN,
                      kind=K.CURRENCY, dataset_id=-1, description=CARRIED_DESCRIPTION, asked_by="", currency="EUR")
    assert checked.measures == (carried,)
    assert checked.charts[0].measures == (WrittenRef(carried),)
    (made,) = checked.datasets
    assert made.sql.endswith(" FROM " + JOIN) and 'claim."created_at" AS "claim_created_at"' in made.sql
    assert _read_through(made.sql, OPEN_PAID_SHOWN) == OPEN_PAID_DECIDED
    assert any(f"{OPEN_PAID_DECIDED} AS open_paid_by_region FROM {JOIN}" in sql for sql in tools.statements())
    assert not [sql for sql in tools.statements() if _PROBE.fullmatch(sql) or _HELD_READ.fullmatch(sql)]
    assert plan_from_json(json.loads(json.dumps(plan_to_json(checked)))) == checked
    # The control: written on the join, the same filter is probed, and refused for a value no row holds.
    written = _measure("open_total", "Open total", "SUM(amount) FILTER (WHERE status = 'open')", K.CURRENCY,
                       dataset_id=-1)
    refused = _carried(WrittenRef(written), tools=_tools(metrics=SAVED_ON_TABLES, held=CLOSED_ONLY))
    assert "which no row the builder can read holds" in _parts(refused)["measure 'open_total'"]


CARRIED_DRAFT = {"measures": [{"name": "open_paid_by_region", "saved_metric": "open_paid", "kind": "currency",
                               "table": "claim", "join": JOIN}],
                 "charts": [{"key": "carried", "family": "bar", "title": "Carried",
                             "measures": ["open_paid_by_region"], "dimensions": ["region"]}],
                 "reuse_dashboard": None}


def test_the_model_carries_a_joined_tables_saved_metric_by_its_table_and_join():
    """On the JSON channel a carried metric is a saved metric named with its table and the join: read as that
    reference under the model's name for it, and written back for the repair as the model wrote it."""
    model = ScriptedModel([json.dumps(CARRIED_DRAFT)])
    checked = _plan(GOAL, LLMPlanner(model), _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED), snapshot=KEYED,
                    settings=GUARDED)
    assert isinstance(checked, BuildPlan), checked
    assert len(model.sent) == 1 and [m.expression for m in checked.measures] == [OPEN_PAID_SHOWN]
    context = _seen(_tools(metrics=SAVED_ON_TABLES))
    plan = plan_from_draft(CARRIED_DRAFT, context)
    assert plan == _carried_plan(OPEN_PAID_REF)
    assert draft_of(plan, context) == CARRIED_DRAFT
    assert plan_from_draft(draft_of(plan, context), context) == plan


@pytest.mark.parametrize("change, why", [
    ({"table": "payroll"}, "'payroll' is not a table this builder can read"),
    ({"join": "policy JOIN"}, "the join 'policy JOIN' is not one SQL join"),
    # A written measure still names its rows with one of the two.
    ({"saved_metric": None, "label": "Paid", "expression": "SUM(amount)"},
     "name its rows with a table or a join, one of the two"),
])
def test_a_carried_metric_names_a_table_the_builder_reads_and_one_join(change, why):
    draft = copy.deepcopy(CARRIED_DRAFT)
    draft["measures"][0].update(change)
    if change.get("saved_metric", "") is None:
        del draft["measures"][0]["saved_metric"]
    refused = plan_from_draft(draft, _seen(_tools(metrics=SAVED_ON_TABLES)))
    assert isinstance(refused, PlanRefused), refused
    assert why in _parts(refused)["measure 'open_paid_by_region'"]


def _own_grain(metric: str, table: str, reason: str, *, may: bool = False) -> str:
    """The own-grain refusal: `may` where mnemiq cannot tell whether the join repeats the table's rows."""
    repeats = "may repeat" if may else "repeats"
    return (f"the saved metric {metric!r} is defined on {table}'s own rows, and this join {repeats} them: {reason}; join "
            f"{table} only to tables it meets at most once per row, or plan the chart on {table} alone")


# KEYED as enrichment leaves a snapshot whose partitions it profiled, where an unset guard is on.
PROFILED = KEYED.model_copy(update={"jobs": [Job(id=PARTITIONS_JOB, source_id="only", kind="profile:partitions",
                                                  status="done")]})


@pytest.mark.parametrize("metric", ["premium_total", "policies"])
def test_a_metric_on_a_join_that_repeats_its_tables_rows_is_refused_with_the_guard_on(metric):
    """policy's SUM(premium), and its COUNT(policy_id), on `policy JOIN claim`, which repeats each policy row once per
    claim: with the guard on, each is refused, the count as the sum, and so with it unset (the default) on a snapshot
    whose partitions were profiled. With the guard off, or unset on a snapshot whose partitions were never profiled,
    neither is checked for it, as no written measure is."""
    ref = SavedRef(metric, K.NUMBER, table="policy", handle="carried_metric")
    repeated = {"measure 'carried_metric'": _own_grain(
        metric, "policy", "each policy row is repeated once per matching claim row (claim.policy_id is not unique)")}
    refused = _carried(ref, dims=("status",))
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == repeated
    unset = Settings(guard_fanout=None, superset_currency="USD")
    refused = _carried(ref, dims=("status",), settings=unset, snapshot=PROFILED)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == repeated
    for settings in (_settings(), Settings(guard_fanout=None, superset_currency="USD")):
        carried = _carried(ref, dims=("status",), settings=settings)
        assert isinstance(carried, BuildPlan), carried
    assert [m.expression for m in carried.measures] == [{"premium_total": 'SUM("premium")',
                                                        "policies": 'COUNT("policy_policy_id")'}[metric]]


# KEYED with claim.policy_id's distinct count unmeasured, as profiling leaves a column it could not count.
CLAIM_KEY_UNMEASURED = KEYED.model_copy(update={"columns": [
    column.model_copy(update={"distinct_count": None}) if column.id == "claim.policy_id" else column
    for column in KEYED.columns]})


def test_a_metric_on_a_join_whose_key_has_no_uniqueness_profile_is_refused_with_the_guard_on():
    """policy's COUNT(policy_id) on `policy JOIN claim`, where claim.policy_id's uniqueness was never measured: nothing
    says each policy row meets one claim, so with the guard on it is refused. claim's own metric on the same join is
    carried: the key it meets, policy.policy_id, is known unique."""
    ref = SavedRef("policies", K.NUMBER, table="policy", handle="carried_metric")
    refused = _carried(ref, dims=("status",), snapshot=CLAIM_KEY_UNMEASURED)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'carried_metric'": _own_grain(
        "policies", "policy",
        "mnemiq cannot tell whether this join repeats policy's rows: claim.policy_id has no uniqueness profile",
        may=True)}
    checked = _carried(OPEN_PAID_REF, snapshot=CLAIM_KEY_UNMEASURED)
    assert isinstance(checked, BuildPlan), checked
    assert [m.expression for m in checked.measures] == [OPEN_PAID_SHOWN]


NULL_SIDE = ("claim is on a side the join may fill with NULL, where its saved metric 'open_paid' is not defined: put "
             "claim on the side the join keeps every row of, or join with JOIN")


@pytest.mark.parametrize("join", [
    "policy LEFT JOIN claim ON claim.policy_id = policy.policy_id",
    "claim RIGHT JOIN policy ON policy.policy_id = claim.policy_id",
    "policy FULL JOIN claim ON claim.policy_id = policy.policy_id",
    "claim FULL OUTER JOIN policy ON policy.policy_id = claim.policy_id",
    # A RIGHT JOIN makes every table before it nullable.
    "claim JOIN policy ON policy.policy_id = claim.policy_id RIGHT JOIN agent ON agent.agent_id = claim.claim_id",
])
def test_a_metric_on_a_side_an_outer_join_may_fill_with_null_is_refused(join):
    """Its rows there may be the NULLs the join fills in, where the metric is not defined. The join itself is not
    refused: the carried metric reads claim, and the chart policy."""
    refused = _carried(OPEN_PAID_REF, join=join)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'open_paid_by_region'": NULL_SIDE}


@pytest.mark.parametrize("join, qualifier", [
    ("claim LEFT JOIN policy ON policy.policy_id = claim.policy_id", "claim"),
    ("policy RIGHT JOIN claim ON claim.policy_id = policy.policy_id", "claim"),
    ("policy p JOIN claim c ON c.policy_id = p.policy_id", "c"),
    ('claim "C" LEFT OUTER JOIN policy ON policy.policy_id = "C".policy_id', '"C"'),
])
def test_a_metric_on_the_side_a_join_keeps_every_row_of_is_carried(join, qualifier):
    """Decided with each column qualified as the join names its table: by its alias, with its quoting."""
    tools = _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED)
    checked = _carried(OPEN_PAID_REF, join=join, tools=tools)
    assert isinstance(checked, BuildPlan), checked
    assert [m.expression for m in checked.measures] == [OPEN_PAID_SHOWN]
    decided = OPEN_PAID_DECIDED.replace("claim.", f"{qualifier}.")
    assert any(f"{decided} AS open_paid_by_region FROM " in sql for sql in tools.statements())
    assert _read_through(checked.datasets[0].sql, OPEN_PAID_SHOWN) == decided


def test_a_metric_naming_its_tables_columns_with_the_table_names_its_alias_on_the_join():
    tools = _tools(metrics=SAVED_ON_TABLES)
    join = "policy p JOIN claim c ON c.policy_id = p.policy_id"
    checked = _carried(SavedRef("paid", K.CURRENCY, table="claim", handle="paid_total"), join=join, tools=tools)
    assert isinstance(checked, BuildPlan), checked
    assert [(m.label, m.expression, m.currency) for m in checked.measures] == [("paid", 'SUM("amount")', None)]
    assert any('SUM(c."amount") AS paid_total FROM policy AS p JOIN claim AS c' in sql for sql in tools.statements())


@pytest.mark.parametrize("join", [
    "claim c JOIN policy p ON p.policy_id = c.policy_id JOIN agent a ON a.agent_id = p.policy_id",
    # The FULL JOIN may fill agent and policy with NULL; claim is joined after it with JOIN, which keeps its rows whole.
    "agent a FULL JOIN policy p ON p.policy_id = a.agent_id JOIN claim c ON c.policy_id = p.policy_id",
    # The LEFT JOIN's ON joins agent, the table it brings in, to policy on agent's key.
    "claim c JOIN policy p ON p.policy_id = c.policy_id LEFT JOIN agent a ON a.agent_id = p.policy_id",
    # Two LEFT JOINs: policy is reached from claim, and agent, joined after policy, through it.
    "claim c LEFT JOIN policy p ON p.policy_id = c.policy_id LEFT JOIN agent a ON a.agent_id = p.policy_id",
    # A conjunct on the table the LEFT JOIN brings in narrows its rows only.
    "claim c JOIN policy p ON p.policy_id = c.policy_id LEFT JOIN agent a ON a.agent_id = p.policy_id "
    "AND a.office = 'west'",
    # A RIGHT JOIN's ON filters the tables before it: claim, the table it keeps, meets one policy and one agent.
    "agent a JOIN policy p ON p.policy_id = a.agent_id RIGHT JOIN claim c ON c.policy_id = p.policy_id",
])
def test_a_metric_is_carried_onto_a_join_of_three_tables(join):
    """claim's metric, on a chart by agent's office: each claim row meets one policy row by policy's key, and each
    policy row one agent row by agent's, both keys known unique, so claim's rows are each read once. Decided with
    claim's columns qualified by its alias, and read back through the three-table dataset's SELECT as decided."""
    tools = _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED)
    checked = _carried(OPEN_PAID_REF, join=join, dims=("office",), tools=tools)
    assert isinstance(checked, BuildPlan), checked
    assert [m.expression for m in checked.measures] == [OPEN_PAID_SHOWN]
    decided = OPEN_PAID_DECIDED.replace("claim.", "c.")
    assert any(f"{decided} AS open_paid_by_region FROM " in sql for sql in tools.statements())
    (made,) = checked.datasets
    assert _read_through(made.sql, OPEN_PAID_SHOWN) == decided
    assert 'a."office" AS "office"' in made.sql


def test_a_metric_whose_tables_only_an_outer_joins_on_joins_is_refused_with_the_guard_on():
    """The LEFT JOIN keeps every row of claim and policy's product, matched or not, so its ON does not join claim to
    policy: claim is joined by no key to every other table, and its metric is not carried."""
    join = ("claim c JOIN policy p ON TRUE "
            "LEFT JOIN agent a ON c.policy_id = p.policy_id AND p.policy_id = a.agent_id")
    refused = _carried(OPEN_PAID_REF, join=join, dims=("office",))
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'open_paid_by_region'": _own_grain(
        "open_paid", "claim", "claim is joined by no key to every other table", may=True)}


@pytest.mark.parametrize("inner", ["TRUE", "a.office = 'west'"])
def test_a_metric_whose_table_meets_another_only_through_a_table_a_left_join_brings_in_is_refused(inner):
    """policy, brought in with LEFT JOIN on claim's key and on agent's, bounds how many policy rows each of them meets;
    the join still keeps every row of claim and agent's product, so policy does not join claim to agent, and claim is
    joined by no key to every other table."""
    join = (f"claim c JOIN agent a ON {inner} "
            "LEFT JOIN policy p ON p.policy_id = c.policy_id AND p.policy_id = a.agent_id")
    refused = _carried(OPEN_PAID_REF, join=join, dims=("office",))
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'open_paid_by_region'": _own_grain(
        "open_paid", "claim", "claim is joined by no key to every other table", may=True)}


NO_KEY = "policy JOIN claim ON claim.policy_id > policy.policy_id"
FOLDED = 'policy "P" JOIN claim p ON p.policy_id = "P".policy_id'


@pytest.mark.parametrize("ref, plan, reason", [
    (SavedRef("open_paid", K.CURRENCY, table="agent", handle="m"), {},
     f"agent is not one of the tables of the join {JOIN!r}"),
    (SavedRef("nope", K.NUMBER, table="claim", handle="m"), {},
     "claim has no saved metric named 'nope' this builder can reach"),
    # agent has no dataset, so no saved metric: not even the count its dataset would get.
    (SavedRef("count", K.COUNT, table="agent", handle="m"),
     {"join": "policy JOIN agent ON agent.agent_id = policy.policy_id"},
     "agent has no saved metric named 'count' this builder can reach"),
    (SavedRef("open_paid", K.CURRENCY, table="claim", handle="m"), {"join": NO_KEY},
     _own_grain("open_paid", "claim", "claim is joined by no key to every other table", may=True)),
    (SavedRef("open_paid", K.CURRENCY, table="claim", handle="m"), {"join": FOLDED},
     _own_grain("open_paid", "claim", "mnemiq cannot read how this join meets claim's rows", may=True)
     + "; give the joined tables aliases that differ by more than letter case"),
    (SavedRef("open_share", K.PERCENT, table="claim", handle="m"), {},
     "the saved metric 'open_share' is on a 0 to 100 scale, so it cannot be shown as a percent: plan it with kind "
     "number, or write the fraction"),
    (SavedRef("count", K.COUNT, table="claim", handle="m"), {},
     "it names no column of claim, so on a join it would count the join's rows"),
    (SavedRef("beyond", K.NUMBER, table="claim", handle="m"), {},
     "it reads more than its own table's rows: a subquery, EXISTS or another table"),
    (SavedRef("each_amount", K.NUMBER, table="claim", handle="m"), {},
     "the expression does not aggregate: it gives one value per row, not one per group"),
    (SavedRef("paid", K.CURRENCY, table="claim", handle="count"), {},
     "its name 'count' is also a saved metric's on its dataset (Superset gives every dataset it makes one named "
     "count); Superset would show two metrics of one name, so name it differently"),
    (SavedRef("paid", K.CURRENCY, table="claim", handle="Paid"), {},
     "the measure name 'Paid' must be lower case letters, digits and _, starting with a letter, at most 63 "
     "characters"),
], ids=["not-joined", "no-such-metric", "no-dataset", "no-key", "aliases-fold", "percent-100", "count", "subquery",
        "row-level", "name-is-count", "name-not-a-name"])
def test_a_saved_metric_that_cannot_be_carried_is_refused_with_mnemiqs_reason(ref, plan, reason):
    refused = _carried(ref, **plan)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused)[f"measure {ref.handle!r}"] == reason


def test_a_carried_metrics_name_is_held_apart_from_the_written_measures():
    written = _measure("open_paid_by_region", "Open total", "MAX(amount)", K.NUMBER, dataset_id=-1)
    refused = _carried(SavedRef("open_paid", K.NUMBER, table="claim", handle="open_paid_by_region"),
                       WrittenRef(written))
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'open_paid_by_region'": "two measures have this name"}


def test_a_joined_tables_saved_metric_on_a_tables_chart_is_refused():
    plan = BuildPlan(charts=(_chart("carried", Family.BAR, [OPEN_PAID_REF], dims=["region"]),), measures=(),
                     reuse_candidates=())
    refused = _plan(GOAL, FakePlanner(plan), _tools(metrics=SAVED_ON_TABLES), snapshot=KEYED, settings=GUARDED)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'open_paid_by_region'": (
        "a joined table's saved metric goes onto a join only, and its chart's rows are the table 'policy'")}


@pytest.mark.parametrize("metric, label", [
    ("open_paid", "Open claims paid"),
    # No verbose name: its name.
    ("paid", "paid"),
    # A verbose name Superset would save as other text, or read as a template: its name.
    ("flagged", "flagged"),
    ("marked", "marked"),
    # A blank verbose name, or one longer than a label: its name.
    ("blank", "blank"),
    ("long_named", "long_named"),
    # The joined dataset's own saved metric is count: its table is appended.
    ("tally", "COUNT (claim)"),
])
def test_a_carried_metrics_label_is_its_verbose_name_held_apart(metric, label):
    checked = _carried(SavedRef(metric, K.NUMBER, table="claim", handle="carried_metric"))
    assert isinstance(checked, BuildPlan), checked
    assert [m.label for m in checked.measures] == [label]


def test_a_carried_label_another_measure_has_takes_its_table_and_is_refused_when_that_is_taken_too():
    first = _measure("open_total", "Open claims paid", "MAX(amount)", K.NUMBER, dataset_id=-1)
    second = _measure("open_most", "Open claims paid (claim)", "MIN(amount)", K.NUMBER, dataset_id=-1)

    def plan(*written: Measure) -> BuildPlan:
        charts = [_chart("carried", Family.BAR, [OPEN_PAID_REF], dataset_id=-1, dims=["region"])]
        charts += [_chart(m.name, Family.BAR, [WrittenRef(m)], dataset_id=-1, dims=["region"]) for m in written]
        return BuildPlan(charts=tuple(charts), measures=written, reuse_candidates=(),
                         datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))

    tools = _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED)
    checked = _plan(GOAL, FakePlanner(plan(first)), tools, snapshot=KEYED, settings=GUARDED)
    assert isinstance(checked, BuildPlan), checked
    assert [m.label for m in checked.measures] == ["Open claims paid", "Open claims paid (claim)"]
    refused = _plan(GOAL, FakePlanner(plan(first, second)), tools, snapshot=KEYED, settings=GUARDED)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'open_paid_by_region'": (
        "its label 'Open claims paid' is another measure's, or a saved metric's on the join, and so is 'Open claims "
        "paid (claim)'; give the other measure a label of its own")}


def test_a_carried_metric_holds_its_label_and_name_apart_from_the_measures_the_dashboard_keeps():
    """A joined table's saved metric carried onto a build's join is a measure of the dashboard too: where its label is
    one the dashboard keeps, its table is appended, as for another measure of the plan; where the plan's name for it
    is one the dashboard keeps, it goes back for a name of its own (a carried measure is on a dataset the build makes,
    so it is never the kept one)."""
    def plan() -> BuildPlan:
        return BuildPlan(charts=(_chart("carried", Family.BAR, [OPEN_PAID_REF], dataset_id=-1, dims=["region"]),),
                         measures=(), reuse_candidates=(),
                         datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))

    tools = _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED)
    labelled = _measure("kept_paid", "Open claims paid", "SUM(amount)", K.CURRENCY, dataset_id=ELSEWHERE)
    checked = _plan(GOAL, FakePlanner(plan()), tools, snapshot=KEYED, settings=GUARDED, kept=(labelled,))
    assert isinstance(checked, BuildPlan), checked
    assert [m.label for m in checked.measures] == ["Open claims paid (claim)"]
    named = _measure("open_paid_by_region", "Paid elsewhere", "SUM(amount)", K.CURRENCY, dataset_id=ELSEWHERE)
    refused = _plan(GOAL, FakePlanner(plan()), tools, snapshot=KEYED, settings=GUARDED, kept=(named,))
    assert isinstance(refused, PlanRefused), refused
    reason = _parts(refused)["measure 'open_paid_by_region'"]
    assert "this dashboard keeps" in reason and "name" in reason, reason


def test_one_carried_metric_on_two_charts_is_one_measure_decided_against_both_charts_columns():
    def plan(ref: SavedRef) -> BuildPlan:
        return BuildPlan(charts=(_chart("by_region", Family.BAR, [ref], dataset_id=-1, dims=["region"]),
                                 _chart("by_status", Family.BAR, [ref], dataset_id=-1, dims=["status"])),
                         measures=(), reuse_candidates=(),
                         datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))

    tools = _tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED)
    checked = _plan(GOAL, FakePlanner(plan(OPEN_PAID_REF)), tools, snapshot=KEYED, settings=GUARDED)
    assert isinstance(checked, BuildPlan), checked
    (measure,) = checked.measures
    assert [chart.measures for chart in checked.charts] == [(WrittenRef(measure),)] * 2
    assert sum(f"AS open_paid_by_region FROM {JOIN}" in sql for sql in tools.statements()) == 1
    # Its label is held apart from the columns of every chart that plots it: 'status' is the second chart's.
    labelled = SavedRef("by_status", K.COUNT, table="claim", handle="m")
    checked = _plan(GOAL, FakePlanner(plan(labelled)), tools, snapshot=KEYED, settings=GUARDED)
    assert isinstance(checked, BuildPlan), checked
    assert [m.label for m in checked.measures] == ["status (claim)"]


def test_a_carried_label_a_charts_dimension_has_takes_its_table_and_is_refused_when_that_is_taken_too():
    """A carried metric's label is its metric's own, so where it is a dimension of a chart that plots it, the table is
    appended, as for a label another measure has; and where that is taken too, the reason says what the plan can
    change."""
    labelled = SavedRef("by_status", K.COUNT, table="claim", handle="m")
    checked = _carried(labelled, dims=("region", "status"))
    assert isinstance(checked, BuildPlan), checked
    assert [m.label for m in checked.measures] == ["status (claim)"]
    other = _measure("other", "status (claim)", "MAX(amount)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("carried", Family.BAR, [labelled], dataset_id=-1, dims=["region", "status"]),
                             _chart("other", Family.BAR, [WrittenRef(other)], dataset_id=-1, dims=["region"])),
                     measures=(other,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))
    refused = _plan(GOAL, FakePlanner(plan), _tools(metrics=SAVED_ON_TABLES), snapshot=KEYED, settings=GUARDED)
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'm'": (
        "the saved metric 'by_status''s label 'status' is also a dimension of a chart that plots it; drop that "
        "dimension or plan the chart on claim alone")}


@pytest.mark.parametrize("symbol", ["$", "eur", "EURO"])
def test_a_carried_metrics_currency_is_its_symbol_only_where_that_is_an_iso_code(symbol):
    certified = {2: [{"metric_name": "open_paid", "currency": {"symbol": symbol, "symbolPosition": "prefix"},
                      "certified_by": "finance"}]}
    checked = _carried(OPEN_PAID_REF, tools=_tools(metrics=SAVED_ON_TABLES, extension=certified))
    assert isinstance(checked, BuildPlan), checked
    assert [(m.expression, m.currency) for m in checked.measures] == [(OPEN_PAID_SHOWN, None)]


def test_only_aliases_postgres_folds_to_one_get_the_alias_advice():
    """`"ß"` and `ss` are two aliases to Postgres (which folds by lower case), so the own-grain reading reads the join,
    and policy's repeated rows are refused for what they are, with no word about aliases."""
    join = 'policy "ß" JOIN claim ss ON ss.policy_id = "ß".policy_id'
    refused = _carried(SavedRef("premium_total", K.NUMBER, table="policy", handle="m"), join=join, dims=("status",))
    assert isinstance(refused, PlanRefused), refused
    assert _parts(refused) == {"measure 'm'": _own_grain(
        "premium_total", "policy", "each policy row is repeated once per matching claim row (claim.policy_id is not "
        "unique)")}


def test_a_carried_metrics_chart_has_its_filters_checked_as_any_charts():
    refused = _carried(OPEN_PAID_REF, filters=[Filter("status", "=", "open")],
                       tools=_tools(metrics=SAVED_ON_TABLES, extension=CERTIFIED, held=CLOSED_ONLY))
    assert isinstance(refused, PlanRefused), refused
    assert list(_parts(refused)) == ["chart 'carried'"]
    assert "which no row the builder can read holds" in _parts(refused)["chart 'carried'"]


def test_a_join_its_carried_metric_and_chart_read_one_table_of_is_refused():
    refused = _carried(OPEN_PAID_REF, dims=("status",))
    assert isinstance(refused, PlanRefused), refused
    assert "read only claim" in _parts(refused)[f"the join {JOIN!r}"]


def test_a_carried_metric_counts_toward_a_plans_measures(monkeypatch):
    from mnemiq.superset import planner as planner_module

    ref = SavedRef("open_paid", K.NUMBER, table="claim", handle="open_paid_by_region")
    written = _measure("open_total", "Open total", "MAX(amount)", K.NUMBER, dataset_id=-1)
    assert isinstance(_carried(ref, WrittenRef(written)), BuildPlan)
    monkeypatch.setattr(planner_module, "MAX_MEASURES", 1)
    refused = _carried(ref, WrittenRef(written))
    assert _parts(refused)["the plan"] == "it writes 2 measures; a plan writes at most 1"


CARRYING_PROMPT = (
    '- a saved metric of a table\'s dataset: {"name": the plan\'s own name for it, "saved_metric": the saved metric\'s '
    'name, "kind": ..., "table": ...}, or, on a join, {"name", "saved_metric", "kind", "table": the joined table whose '
    'saved metric it is, "join": "a JOIN b ON ..."}. The same saved metric on two tables, or on two joins, is two '
    "measures, each with a name of its own.\n"
    "Use a saved metric where one says what a measure needs, on a join too: never write your own measure for what a "
    "joined table's saved metric says. Write a join only where the charts need columns of more than one table. A join "
    "is tables joined with JOIN, LEFT JOIN, RIGHT JOIN or FULL JOIN, each ON a condition on their columns, each table "
    "named alone or with an alias. On a join, a written measure names each column alone (amount, not claim.amount), "
    "and only a column one joined table alone has. A column two joined tables share is in the joined dataset as "
    "<table>_<column> (claim_created_at), for dimensions, filters, time columns, source and target only. A saved "
    "metric goes onto a join as its table defines it, mnemiq naming its columns as the join names them, and only where "
    "the join repeats none of its table's rows: its table joined to tables with at most one row for each of its rows, "
    "with JOIN (which leaves out its rows no other table matches) or on the side a LEFT or RIGHT JOIN keeps every row "
    "of (the left of a LEFT JOIN, the right of a RIGHT JOIN), never in a FULL JOIN; and only a saved metric that names "
    "a column of its own table, so never count (COUNT(*)). A written measure's name and label are never a saved "
    "metric's name on its table, and no two measures share a label. A table with no dataset gets one, with Superset's "
    "saved metric count (COUNT(*)).\n\nCharts: ")


def test_the_prompt_offers_a_joined_tables_saved_metric_on_a_join():
    system, _ = plan_prompt(GOAL, _seen(_tools()))
    assert CARRYING_PROMPT in system
    assert "The same saved metric on two tables is two measures" not in system
    assert "On a join, a measure names each column alone" not in system

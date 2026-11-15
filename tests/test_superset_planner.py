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

from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot
from mnemiq.superset import build_store
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.chart_form import _GRAINS
from mnemiq.superset.charts import ChartSpec, Family, Filter, SavedRef, WrittenRef
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import (
    COUNT_METRIC,
    FILTER_OPERATORS,
    GRAINS,
    BuildPlan,
    FakePlanner,
    LLMPlanner,
    PlannedDataset,
    PlanRefused,
    choose_dashboard,
    plan_build,
    plan_from_json,
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
          *(TOOL_PREFIX + name for name in ("create_dashboard", "claim_chart", "claim_dataset", "create_chart",
                                            "patch_chart", "add_saved_metric", "delete_dataset", "set_layout")))


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


def _tools(*, metrics=None, extension=None, dashboards=(), execute=None, datasets=None) -> Tools:
    """policy is dataset 1 and claim dataset 2; agent has none, and a virtual dataset over policy is never a table's."""
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

    def list_dashboards(request):
        word = request["search"].casefold()
        return {"dashboards": [dict(item) for item in dashboards if word in item["dashboard_title"].casefold()],
                "count": 0, "total_pages": 1}

    return Tools({
        "list_datasets": lambda request: {"datasets": datasets, "count": len(datasets)},
        "list_metrics": list_metrics,
        TOOL_PREFIX + "dataset_metrics": dataset_metrics,
        "list_dashboards": list_dashboards,
        "execute_sql": execute or (lambda request: EMPTY),
    })


def _adapter(tools: Tools) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(tools, database_id=7, schema="public")


def _settings() -> Settings:
    return Settings(guard_fanout=False, superset_currency="USD")


def _plan(goal: str, planner, tools: Tools, *, grants=GRANTS, snapshot=SNAPSHOT, **more):
    adapter = _adapter(tools)
    return plan_build(goal, planner=planner, client=adapter.client, adapter=adapter, snapshot=snapshot, grants=grants,
                      settings=_settings(), **more)


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


# ---------------------------------------------------------------- the brief's tests

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
    assert not set(tools.tools()) & set(WRITES), "nothing is built while the choice is open"
    # The choice survives the store, and only the next request's answer settles it.
    assert plan_from_json(json.loads(json.dumps(plan_to_json(plan)))) == plan
    reused, dashboard = choose_dashboard(plan, "reuse")
    assert dashboard == 12 and not reused.awaiting_choice and reused.charts == plan.charts
    fresh, dashboard = choose_dashboard(plan, "fresh")
    assert dashboard is None and not fresh.awaiting_choice
    with pytest.raises(ValueError):
        choose_dashboard(plan, "maybe")
    with pytest.raises(ValueError):
        choose_dashboard(fresh, "reuse")


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
    filters = [Filter("region", op, operand.get(op, "a")) for op in FILTER_OPERATORS]
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


# ---------------------------------------------------------------- fix round 1

COUNTS = {1: [_metric(1, "count", "COUNT(*)")], 2: [_metric(2, "count", "COUNT(*)")]}


def test_a_saved_metric_has_a_handle_of_its_own_in_the_plan():
    """I1: `count` on two tables. A saved measure's plan-local name is its handle; the metric's own name beside it."""
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
    """I2: the only tables are the FROM's and each JOIN's, and the only SELECT is the join's own."""
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
    """I3: rc2 reads a physical dataset FROM `table(self.table_name, schema=self.schema)`
    (`SqlaTable.get_sqla_table` in `superset/connectors/sqla/models.py`), which SQLAlchemy quotes for a mixed-case
    name: a dataset named `Agent` reads `"Agent"`, never the stored `agent`."""
    recorder = FakePlanner(_valid_plan())
    _plan(GOAL, recorder, _tools(datasets=[_dataset(1, "policy"), _dataset(2, "claim"), _dataset(5, "Agent")]))
    tables = {table.name: table.dataset_id for table in recorder.contexts[0].tables}
    assert tables == {"agent": None, "claim": 2, "policy": 1}


def test_a_shared_column_is_a_charts_under_its_tables_name():
    """I4: both tables carry created_at; the chart groups by claim's, as the joined dataset names it."""
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
    """I5: with `offer_reuse=False` no dashboard is listed, and a candidate the model names is refused."""
    tools = _tools(dashboards=[{"id": 12, "dashboard_title": "Premium by region"}])
    recorder = FakePlanner(_valid_plan())
    assert _plan(GOAL, recorder, tools, offer_reuse=False) == _valid_plan()
    assert "list_dashboards" not in tools.tools() and recorder.contexts[0].dashboards == ()
    refused = _plan(GOAL, FakePlanner(_valid_plan(reuse_candidates=(12,))), tools, offer_reuse=False)
    assert isinstance(refused, PlanRefused), refused
    assert "the dashboard to reuse" in _parts(refused)
    assert "list_dashboards" not in tools.tools()


def test_a_refusal_the_model_cannot_repair_stops_the_plan():
    """M1: Superset not answering a measure's proof (`source_unavailable`) is nobody's statement to repair."""
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
    """M2: by the type the snapshot records; a time of day or an unknown type is not one."""
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
    """M3: unique within a build by its key, and never longer than rc2's 250."""
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
    """M4: a planner's own sql, name or table_name never reaches the build."""
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
    """M5: decision 12 makes a virtual dataset only where a join is needed."""
    total = _measure("total", "Total", "SUM(premium)", K.NUMBER, dataset_id=-1)
    plan = BuildPlan(charts=(_chart("total", Family.BAR, [WrittenRef(total)], dataset_id=-1, dims=["region"]),),
                     measures=(total,), reuse_candidates=(),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=JOIN),))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "only" in _parts(refused)[f"the join {JOIN!r}"]


def test_the_planner_imports_no_private_name():
    """M6: what the planner reads of another module is that module's public interface. Parsed with `ast`."""
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
    """M7."""
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
    """M8: NaN and Infinity."""
    plan = _valid_plan(charts=(_chart("by_region", Family.BAR, [WrittenRef(PREMIUM)], dims=["region"],
                                      filters=[item]),), measures=(PREMIUM,))
    refused = _plan(GOAL, FakePlanner(plan), _tools())
    assert isinstance(refused, PlanRefused), refused
    assert "finite" in _parts(refused)["chart 'by_region'"]


def test_a_measures_description_is_bounded():
    """M8: at most 500 characters."""
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


# ---------------------------------------------------------------- fix round 2

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
    """Fix round 2: what the allow-list still let through. TABLESAMPLE would have made the dataset a sample."""
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
    """Fix round 2: the dataset's SQL is the parsed join's rendering, the text the dry run decided, not the raw
    text."""
    tools = _tools()
    checked = _plan(GOAL, FakePlanner(_joined_plan(written)), tools)
    assert isinstance(checked, BuildPlan), checked
    (made,) = checked.datasets
    assert made.join == written, "the plan keeps the join as the model wrote it"
    assert made.sql.endswith(" FROM " + rendered)
    assert any(f"AS paid FROM {rendered}" in sql for sql in tools.statements())


def test_virtual_dataset_names_cannot_meet():
    """Fix round 2: a key holds no `:`, so the name's key part is the key and nothing else."""
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

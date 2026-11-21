"""A dashboard build, one step per request: the dashboard, its datasets, its charts, the layout and the checks, each step
claimed with its intent before any Superset call, its lease renewed before each call, every object made by the
extension's creates with the step's marker, and a step whose outcome is uncertain run again first: its create, sent
again, is answered with the object carrying the marker, so nothing is made twice and nothing another made is adopted.

Every test runs against `FakeSuperset`, a real MCP server over streamable HTTP, through the real client and adapter,
with the extension's tools served by `World` (`tests/superset_build_world.py`): Superset's state in memory, numbers
from one DuckDB by two paths. The store is `MemoryBuildStore`, the Postgres store's own transitions on a clock the test
moves; `test_superset_build_store.py` holds the store to Postgres.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from mnemiq.config import Settings
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.build_store import DashboardBusy, StepState
from mnemiq.superset.builder import BuildBusy, BuildState, build_state, build_steps, next_step, start_build
from mnemiq.superset.chart_check import ChartStatus
from mnemiq.superset.charts import ChartSpec, Family, Filter, SavedRef, WrittenRef, extension_chart_request
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetRefusedCaller, SupersetToolError
from mnemiq.superset.dashboard_charts import LeftOut, datasets_built_on
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.marker import Marker, format_marker, parse_marker
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan, PlannedDataset
from tests.superset_build_world import (
    DATABASE_ID,
    FRESH_TOKEN,
    GRANTS,
    POLICY_DATASET,
    SCHEMA,
    SNAPSHOT,
    TOKEN,
    Clock,
    Hook,
    MemoryBuildStore,
    World,
    dataset_id_of,
)
from tests.superset_fake import FakeSuperset
from tests.superset_shared import BY_REGION, ONE_BAR, POLICIES, _chart

TENANT = "tenant-a"
PRINCIPAL = "superset:builder"
GOAL = "Premium and claims by region and month"
SETTINGS = Settings(guard_fanout=False, superset_currency="USD")

TOTAL = SavedRef("total_premium", MeasureKind.CURRENCY)
CLAIMED = Measure(name="claimed", label="Claimed", expression="SUM(amount)", kind=MeasureKind.CURRENCY, dataset_id=-1,
                  description="Amount claimed", asked_by="", currency="USD")
CLAIM_DATASET = PlannedDataset(key="dataset_1", placeholder=-1, table="claim", table_name="claim")


KPI_TOTAL = _chart("total_premium_kpi", Family.BIG_NUMBER, "Total premium", POLICY_DATASET, (TOTAL,))
KPI_POLICIES = _chart("policies_kpi", Family.BIG_NUMBER, "Policies", POLICY_DATASET, (WrittenRef(POLICIES),))
BY_MONTH = _chart("premium_by_month", Family.LINE, "Premium by month", POLICY_DATASET, (TOTAL,), time_column="issued",
                  time_grain="P1M")
BY_CHANNEL = _chart("premium_by_channel", Family.PIE, "Premium by channel", POLICY_DATASET, (TOTAL,),
                    dimensions=("channel",))
CLAIMS_TABLE = _chart("claimed_by_region", Family.TABLE, "Claimed by region", -1, (WrittenRef(CLAIMED),),
                      dimensions=("claim_region",))
FUNNEL = _chart("policies_by_status", Family.FUNNEL, "Policies by status", POLICY_DATASET, (WrittenRef(POLICIES),),
                dimensions=("status",))

# Six charts across five families, the planted saved metric reused on three, a dataset the build makes for the table.
SIX = BuildPlan(charts=(BY_MONTH, KPI_TOTAL, BY_REGION, KPI_POLICIES, BY_CHANNEL, CLAIMS_TABLE),
                measures=(POLICIES, CLAIMED), datasets=(CLAIM_DATASET,))
ONE_FUNNEL = BuildPlan(charts=(FUNNEL,), measures=(POLICIES,))


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        yield served


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock) -> MemoryBuildStore:
    return MemoryBuildStore(clock)


@pytest.fixture
def world(fake, clock) -> World:
    return World(fake, clock)


def _adapter(fake, token: str = TOKEN, *, timeout_s: float = 10.0) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken(token), timeout_s=timeout_s),
                              database_id=DATABASE_ID, schema=SCHEMA)


def _start(store, plan: BuildPlan = SIX, *, dashboard_id: int | None = None, goal: str = GOAL) -> BuildState:
    return start_build(store, tenant=TENANT, principal=PRINCIPAL, goal=goal, plan=plan, dashboard_id=dashboard_id)


def _next(store, build_id: str, adapter: SupersetSqlAdapter, settings: Settings = SETTINGS, **more) -> BuildState:
    build = store.load_build(TENANT, build_id, PRINCIPAL)
    return next_step(build, client=adapter.client, adapter=adapter, store=store, snapshot=SNAPSHOT, grants=GRANTS,
                     settings=settings, **more)


def _run(store, build_id: str, adapter: SupersetSqlAdapter, settings: Settings = SETTINGS, *,
         limit: int = 20) -> list[BuildState]:
    """`/next` until the build is done or stopped: each request's state."""
    states = []
    for _ in range(limit):
        state = _next(store, build_id, adapter, settings)
        states.append(state)
        if state.status != "building":
            return states
    raise AssertionError(f"the build did not end in {limit} requests: {states[-1]}")


def _step(state: BuildState, key: str) -> StepState:
    [found] = [step for step in state.steps if step.key == key]
    return found


def _tools(fake) -> list[str]:
    return [call.tool.rsplit(".", 1)[-1] for call in fake.calls]


def _chart_ids(state: BuildState, plan: BuildPlan = SIX) -> dict[str, int]:
    return {chart.key: _step(state, chart.key).object_id for chart in plan.charts}


def _raw_step(store, build_id: str, key: str) -> dict:
    [found] = [step for step in store.raw_steps(TENANT, build_id) if step["key"] == key]
    return found


def _once(hook):
    """A hook that acts the first time only, passing on what it returns (an `after` hook's answer in place of the
    tool's)."""
    done = []

    def run(*args):
        if not done:
            done.append(True)
            return hook(*args)
        return None

    return run


def _run_to(store, build_id: str, adapter, key: str) -> BuildState:
    """`/next` until step `key` is the next one to run."""
    state = None
    for _ in range(20):
        state = build_state(store, store.load_build(TENANT, build_id, PRINCIPAL))
        upcoming = next(step for step in state.steps if step.status != "done")
        if upcoming.key == key:
            return state
        _next(store, build_id, adapter)
    raise AssertionError(f"never reached {key}")


# --- the build, end to end


def test_build_runs_to_a_laid_out_checked_dashboard(fake, world, store):
    """Six charts across five families, the planted saved metric reused, a dataset made for the table: every step runs
    one per request in the build's order, every chart is checked, and the dashboard's layout puts the KPI row first."""
    started = _start(store)
    assert [step.key for step in started.steps] == ["dashboard", "dataset_1", *(c.key for c in SIX.charts), "layout",
                                                    "check"]
    assert started.status == "building" and all(step.status == "pending" for step in started.steps)
    states = _run(store, started.build_id, _adapter(fake))
    assert [state.ran for state in states] == [step.key for step in started.steps]
    final = states[-1]
    assert final.status == "done" and all(step.status == "done" for step in final.steps), final
    ids = _chart_ids(final)
    assert final.checks == {chart_id: ChartStatus.CHECKED for chart_id in ids.values()}, final.message
    dashboard = world.dashboards[final.dashboard_id]
    positions = json.loads(dashboard["position_json"])
    first_row = positions[positions["GRID_ID"]["children"][0]]
    assert [positions[c]["meta"]["chartId"] for c in first_row["children"]] == [ids["total_premium_kpi"],
                                                                                 ids["policies_kpi"]]
    assert sorted(dashboard["members"]) == sorted(ids.values())
    # The planted saved metric is reused by name, never written again.
    for key in ("total_premium_kpi", "premium_by_month", "premium_by_channel"):
        assert world.charts[ids[key]]["params"].get("metric", world.charts[ids[key]]["params"].get("metrics")) in (
            "total_premium", ["total_premium"])
    # The table's currency, which MCP's chart config cannot carry, rides on its create as format keys to merge.
    owed = {"column_config": {"Claimed": {"d3NumberFormat": ",.2f", "currencyFormat": {
        "symbol": "USD", "symbolPosition": "prefix"}}}}
    [table] = [call.request for call in world.calls("create_chart") if call.request["slice_name"] == "Claimed by region"]
    assert table["set"] == owed and "patch_chart" not in _tools(fake)
    assert world.charts[ids["claimed_by_region"]]["params"]["column_config"] == owed["column_config"]
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}


def test_the_made_dataset_is_the_one_its_charts_and_measures_use(fake, world, store):
    """`with_made_datasets` before every chart request and every measure kept: the table is generated, checked and kept
    on the dataset its step made, each measure is kept for the dashboard with its charts, the datasets made for it and
    the builder who asked."""
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]
    made = _step(final, "dataset_1").object_id
    assert made == dataset_id_of(world, "claim") and world.datasets[made]["sql"] is None
    ids = _chart_ids(final)
    [table_request] = [c.request for c in world.calls("create_chart") if c.request["slice_name"] == "Claimed by region"]
    assert table_request["dataset_id"] == made
    assert world.charts[ids["claimed_by_region"]]["datasource_id"] == made
    checked = [c.request["identifier"] for c in world.calls("get_dataset_info")]
    assert made in checked and all(isinstance(i, int) and i > 0 for i in checked)
    kept = {m.measure.name: m for m in store.measures_for(TENANT, final.dashboard_id) if m.saved_metric is None}
    assert set(kept) == {"policies", "claimed"}
    assert kept["claimed"].measure == dataclasses.replace(CLAIMED, dataset_id=made, asked_by=PRINCIPAL)
    assert kept["claimed"].chart_ids == (ids["claimed_by_region"],) and kept["claimed"].made_dataset_ids == (made,)
    assert kept["policies"].measure == dataclasses.replace(POLICIES, asked_by=PRINCIPAL)
    assert sorted(kept["policies"].chart_ids) == sorted((ids["policies_kpi"], ids["policies_by_region"]))
    assert kept["policies"].made_dataset_ids == ()
    assert kept["policies"].check_status == {ids["policies_kpi"]: ChartStatus.CHECKED,
                                             ids["policies_by_region"]: ChartStatus.CHECKED}
    # A dataset the build makes carries the build's marker and the dashboard's editors, both from its create.
    marker = parse_marker(world.datasets[made]["description"])
    assert marker == Marker(final.build_id, "dataset_1", final.dashboard_id)
    assert world.datasets[made]["editors"] == world.dashboards[final.dashboard_id]["editors"]


def test_written_measure_leaves_saved_metrics_unchanged(fake, world, store):
    """A measure mnemiq writes rides on its charts as SQL: no saved metric is added, and every dataset's saved metrics
    read the same before and after."""
    before = {i: json.dumps(d["metrics"], sort_keys=True) for i, d in world.datasets.items()}
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]
    assert final.status == "done"
    assert "add_saved_metric" not in _tools(fake)
    assert {i: json.dumps(world.datasets[i]["metrics"], sort_keys=True) for i in before} == before
    listed = _adapter(fake).client.call("list_metrics", {"dataset_id": POLICY_DATASET, "page_size": 25, "page": 1})
    assert sorted(m["name"] for m in listed["metrics"]) == ["count", "total_premium"]


def test_created_charts_get_the_dashboards_editors(fake, world, store):
    """`create_chart` is called once per chart, a funnel's among them, each after the dashboard shell exists and for
    that dashboard; every chart then has the dashboard's editors and the build's marker, both from its create."""
    plan = BuildPlan(charts=(*SIX.charts, FUNNEL), measures=SIX.measures, datasets=SIX.datasets)
    final = _run(store, _start(store, plan).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    tools = _tools(fake)
    shell = tools.index("create_dashboard")
    created = [i for i, tool in enumerate(tools) if tool == "create_chart"]
    assert len(created) == len(plan.charts) and min(created) > shell
    ids = _chart_ids(final, plan)
    assert {c.request["dashboard_id"] for c in world.calls("create_chart")} == {final.dashboard_id}
    editors = world.dashboards[final.dashboard_id]["editors"]
    for chart in plan.charts:
        row = world.charts[ids[chart.key]]
        assert row["editors"] == editors, chart.key
        assert parse_marker(row["description"]) == Marker(final.build_id, chart.key, final.dashboard_id)


def test_fresh_build_with_a_funnel(fake, world, store):
    """A new dashboard whose plan holds a funnel runs end to end: shell, the funnel through the extension, the layout
    and the check."""
    final = _run(store, _start(store, ONE_FUNNEL).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    funnel = _step(final, "policies_by_status").object_id
    assert final.checks == {funnel: ChartStatus.CHECKED}, final.message
    assert world.dashboards[final.dashboard_id]["members"] == [funnel]


def test_funnel_goes_through_the_extension(fake, world, store):
    """A funnel is not among MCP's chart types: the extension's `create_chart` makes it from mnemiq's own form data
    (`extension_chart_request`), never from a chart config, for the dashboard, with the marker written at create."""
    final = _run(store, _start(store, ONE_FUNNEL).build_id, _adapter(fake))[-1]
    assert "generate_chart" not in _tools(fake)
    [created] = [c.request for c in world.calls("create_chart")]
    expected = extension_chart_request(FUNNEL, {"policies": POLICIES}, {}, currency="USD")
    assert created == {**expected, "dashboard_id": final.dashboard_id,
                       "marker": format_marker(final.build_id, "policies_by_status", final.dashboard_id)}


def test_failed_check_is_reported_as_what_it_is(fake, world, store):
    """A chart whose numbers are off stays `MISMATCH`: the build still finishes, the others are checked, and the
    build says which chart did not check."""
    world.skewed["Policies by region"] = 1
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]
    ids = _chart_ids(final)
    assert final.status == "done"
    assert final.checks[ids["policies_by_region"]] is ChartStatus.MISMATCH
    assert all(status is ChartStatus.CHECKED for chart_id, status in final.checks.items()
               if chart_id != ids["policies_by_region"])
    assert "Policies by region" in final.message and "mismatch" in final.message
    assert "Premium by month" not in final.message
    kept = {m.measure.name: m for m in store.measures_for(TENANT, final.dashboard_id)}
    assert kept["policies"].check_status[ids["policies_by_region"]] is ChartStatus.MISMATCH


# A written measure whose label rc2's MCP saves as other text (`ColumnRef.sanitize_label` escapes `<`), on a currency
# table, whose per-column currency is set under the label mnemiq sent.
YOUNG = Measure(name="young_premium", label="Premium < 1y", expression="SUM(premium)", kind=MeasureKind.CURRENCY,
                dataset_id=POLICY_DATASET, description="", asked_by="", currency="USD")
YOUNG_TABLE = _chart("young_by_region", Family.TABLE, "Young premium by region", POLICY_DATASET, (WrittenRef(YOUNG),),
                     dimensions=("region",))


def test_a_label_superset_saved_as_other_text_is_not_checked(fake, world, store):
    """The plan check refuses a label rc2 would rewrite; past it, as with a plan made here by hand, rc2 saves
    `Premium < 1y` as `Premium &lt; 1y` while the table's currency is set under `Premium < 1y`. The chart's numbers
    match, but the format of the metric it shows was never judged, so it is not checked: never a false CHECKED."""
    final = _run(store, _start(store, BuildPlan(charts=(YOUNG_TABLE,), measures=(YOUNG,))).build_id,
                 _adapter(fake))[-1]
    chart_id = _step(final, "young_by_region").object_id
    params = world.charts[chart_id]["params"]
    assert [metric["label"] for metric in params["metrics"]] == ["Premium &lt; 1y"]  # as rc2 saves it
    assert list(params["column_config"]) == ["Premium < 1y"]
    assert final.status == "done" and final.checks == {chart_id: ChartStatus.NOT_CHECKED}, final.message
    assert "'Premium &lt; 1y'" in final.message and "could not be judged" in final.message


@pytest.mark.parametrize("sent, saved", [
    ("Premium < 1y", "Premium &lt; 1y"), ("Premium > 100", "Premium &gt; 100"), ("A &amp; B", "A & B"),
    ("A & B", "A & B"), ("Premium <b>by</b> region", "Premium by region"), ("Premium\u00a0by", "Premium&nbsp;by"),
    ("Pre\u200bmium", "Premium"), ("  Premium  ", "Premium"), ("Premium by region", "Premium by region"),
])
def test_the_world_saves_a_label_as_rc2_does(sent, saved):
    from tests.superset_build_world import rc2_saved

    assert rc2_saved(sent) == saved


@pytest.mark.parametrize("sent", ["javascript:alert", "monthly = target"])
def test_the_world_refuses_a_label_rc2_refuses(fake, world, store, sent):
    """A chart name or label rc2's validators refuse makes no chart: the extension refuses the chart config as a
    whole, and the step stops naming that."""
    from tests.superset_build_world import rc2_saved

    assert isinstance(rc2_saved(sent), ValueError)
    labelled = dataclasses.replace(YOUNG, label=sent)
    chart = dataclasses.replace(YOUNG_TABLE, measures=(WrittenRef(labelled),))
    build = _start(store, BuildPlan(charts=(chart,), measures=(labelled,))).build_id
    _run_to(store, build, _adapter(fake), "young_by_region")
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "young_by_region").reason == "extension_refused", stopped
    assert "(config)" in stopped.message
    assert world.charts == {} and len(world.calls("create_chart")) == 1


def test_no_dashboard_is_published(fake, world, store):
    """Nothing a build sends asks for a published dashboard, and the dashboard it made stays unpublished."""
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]

    def published(doc) -> bool:
        if isinstance(doc, dict):
            return doc.get("published") is True or any(published(v) for v in doc.values())
        return isinstance(doc, list) and any(published(v) for v in doc)

    assert not any(published(call.request) for call in fake.calls)
    assert world.dashboards[final.dashboard_id]["published"] is False


def _existing_dashboard(world, *, editors=("builder",), dashboard: int = 900) -> tuple[int, int]:
    """A dashboard made by hand, with one chart on it and a layout holding it: its id and the chart's."""
    existing = world.plant_chart(title="By hand", dataset_id=POLICY_DATASET, viz_type="table", user="analyst")
    world.dashboards[dashboard] = {
        "id": dashboard, "dashboard_title": "Sales", "published": True, "description": "Made by hand",
        "position_json": json.dumps({
            "DASHBOARD_VERSION_KEY": "v2", "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
            "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": ["ROW-x"], "parents": ["ROOT_ID"]},
            "ROW-x": {"type": "ROW", "id": "ROW-x", "children": ["CHART-x"], "parents": ["ROOT_ID", "GRID_ID"],
                      "meta": {}},
            "CHART-x": {"type": "CHART", "id": "CHART-x", "children": [], "parents": ["ROOT_ID", "GRID_ID", "ROW-x"],
                        "meta": {"chartId": existing, "width": 12, "height": 50}}}),
        "json_metadata": "{}", "editors": list(editors), "members": [existing], "created_by": "analyst",
        "created_on": world.superset_now(), "changed_on": world.superset_now()}
    return dashboard, existing


def test_an_existing_dashboard_is_read_then_added_to(fake, world, store):
    """For an existing dashboard, the first step reads whether the builder edits it, its layout and `changed_on`, and
    holds nothing; the charts are added to the layout it has, which is kept."""
    dashboard, existing = _existing_dashboard(world)
    state = _start(store, ONE_BAR, dashboard_id=dashboard)
    first = _next(store, state.build_id, _adapter(fake))
    assert first.ran == "dashboard" and _step(first, "dashboard").object_id == dashboard
    assert sorted(_tools(fake)) == ["editors_of", "get_layout"]
    [asked] = world.calls("editors_of")
    assert asked.request == {"dashboard_ids": [dashboard]}
    final = _run(store, state.build_id, _adapter(fake))[-1]
    assert final.status == "done" and final.dashboard_id == dashboard
    assert "create_dashboard" not in _tools(fake)
    positions = json.loads(world.dashboards[dashboard]["position_json"])
    assert positions["GRID_ID"]["children"][0] == "ROW-x"
    chart = _step(final, "policies_by_region").object_id
    assert sorted(world.dashboards[dashboard]["members"]) == sorted([existing, chart])
    assert world.dashboards[dashboard]["published"] is True  # the build changes no dashboard's publication


def test_an_existing_dashboard_the_builder_does_not_edit_is_refused_before_anything_is_made(fake, world, store):
    """The extension answers whether the builder edits the dashboard (`editors_of` with `dashboard_ids`): one who
    does not is refused at the first step, told its editors, and nothing is made."""
    dashboard, _ = _existing_dashboard(world, editors=("colleague", "analyst"))
    charts = set(world.charts)
    state = _start(store, ONE_BAR, dashboard_id=dashboard)
    stopped = _next(store, state.build_id, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "dashboard").status == "failed"
    assert _step(stopped, "dashboard").reason == "not_editor"
    assert "not an editor" in stopped.message and "analyst, colleague" in stopped.message
    assert set(world.charts) == charts and not any(t.startswith(("generate", "create", "claim")) for t in _tools(fake))
    later = build_state(store, store.load_build(TENANT, state.build_id, PRINCIPAL))
    assert "not an editor" in later.message and "colleague" not in later.message


# --- Building onto a dashboard plans only what the dashboard does not already show: each planned chart is compared
# with the dashboard's charts, read once as the builder, and one it shows exactly is left out and named.

THREE = BuildPlan(charts=(BY_REGION, KPI_TOTAL, BY_CHANNEL), measures=(POLICIES,))
BY_STATUS = _chart("policies_by_status_bar", Family.BAR, "Policies by status", POLICY_DATASET, (WrittenRef(POLICIES),),
                   dimensions=("status",))
DISTINCT = dataclasses.replace(POLICIES, name="distinct_policies", label="Distinct policies",
                               expression="COUNT(DISTINCT policy_id)")


def _built(fake, store, plan: BuildPlan) -> BuildState:
    final = _run(store, _start(store, plan).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    return final


def _plan_onto(fake, store, planned: BuildPlan, dashboard_id: int):
    from mnemiq.superset.planner import FakePlanner, plan_build

    adapter = _adapter(fake)
    return plan_build(GOAL, planner=FakePlanner(planned), client=adapter.client, adapter=adapter, snapshot=SNAPSHOT,
                      grants=GRANTS, settings=SETTINGS, offer_reuse=False, onto=dashboard_id,
                      built_on=datasets_built_on(store.measures_for(TENANT, dashboard_id)))


def test_a_planned_chart_the_dashboard_shows_is_left_out_and_named(fake, world, store):
    """Planned again onto the dashboard a build made, a chart it already shows is left out of the plan, under another
    key and title too, and named with the chart that shows it; the chart it lacks is planned and built, and nothing
    else is made. The dashboard's charts are read once: its layout, then each chart."""
    built = _built(fake, store, THREE)
    shown = _step(built, "policies_by_region").object_id
    again = dataclasses.replace(BY_REGION, key="region_policies", title="Policies per region")
    before = len(fake.calls)
    plan = _plan_onto(fake, store, BuildPlan(charts=(again, BY_STATUS), measures=(POLICIES,)), built.dashboard_id)
    assert isinstance(plan, BuildPlan), plan
    assert [chart.key for chart in plan.charts] == ["policies_by_status_bar"]
    assert plan.left_out == (LeftOut(part="Policies per region", chart_id=shown, title="Policies by region"),)
    read = [call.tool.rsplit(".", 1)[-1] for call in fake.calls[before:]]
    assert read.count("get_layout") == 1 and read.count("get_chart_info") == len(THREE.charts)
    final = _run(store, _start(store, plan, dashboard_id=built.dashboard_id).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    assert [step.key for step in final.steps if step.kind == "chart"] == ["policies_by_status_bar"]


@pytest.mark.parametrize("near", [
    pytest.param(dataclasses.replace(BY_MONTH, time_grain="P1W"), id="another grain"),
    pytest.param(dataclasses.replace(BY_REGION, filters=(Filter("premium", ">", 50),)), id="another filter"),
    pytest.param(dataclasses.replace(BY_REGION, measures=(WrittenRef(DISTINCT),)), id="another measure definition"),
    pytest.param(dataclasses.replace(BY_REGION, family=Family.LINE), id="another family"),
    pytest.param(dataclasses.replace(BY_MONTH, row_limit=5), id="another row limit"),
])
def test_a_near_match_is_not_left_out(fake, world, store, near):
    """A planned chart that differs from the dashboard's in one thing it shows is not one the dashboard shows."""
    built = _built(fake, store, BuildPlan(charts=(BY_REGION, BY_MONTH), measures=(POLICIES,)))
    measures = tuple(ref.measure for ref in near.measures if isinstance(ref, WrittenRef))
    plan = _plan_onto(fake, store, BuildPlan(charts=(dataclasses.replace(near, key="near", title="Near"),),
                                      measures=measures), built.dashboard_id)
    assert isinstance(plan, BuildPlan), plan
    assert [chart.key for chart in plan.charts] == ["near"] and plan.left_out == ()


def test_a_ranked_chart_is_left_out_only_when_it_ranks_the_same_way(fake, world, store):
    """A top-two bar is what the dashboard shows only while it ranks as the plan would: largest first. Flipped by
    hand to the smallest two, it shows other groups, and the planned one is built."""
    top_two = dataclasses.replace(BY_REGION, row_limit=2)
    built = _built(fake, store, BuildPlan(charts=(top_two,), measures=(POLICIES,)))
    planned = BuildPlan(charts=(dataclasses.replace(top_two, key="again", title="Again"),), measures=(POLICIES,))
    assert [left.chart_id for left in _plan_onto(fake, store, planned, built.dashboard_id).left_out] == [
        _step(built, top_two.key).object_id]
    chart = world.charts[_step(built, top_two.key).object_id]
    chart["params"] = {**chart["params"], "order_desc": False}
    plan = _plan_onto(fake, store, planned, built.dashboard_id)
    assert [c.key for c in plan.charts] == ["again"] and plan.left_out == ()


def test_a_goal_the_dashboard_shows_whole_builds_nothing(fake, world, store):
    """Every planned chart already on the dashboard: the plan keeps no chart and no measure, and names each part with
    the chart that shows it."""
    built = _built(fake, store, THREE)
    plan = _plan_onto(fake, store, THREE, built.dashboard_id)
    assert isinstance(plan, BuildPlan), plan
    assert plan.charts == () and plan.measures == () and plan.datasets == ()
    assert [left.part for left in plan.left_out] == [chart.title for chart in THREE.charts]
    assert [left.chart_id for left in plan.left_out] == [_step(built, chart.key).object_id for chart in THREE.charts]


def test_a_chart_on_a_dataset_the_build_makes_is_never_left_out(fake, world, store):
    """A planned chart on a dataset the build is to make reads no dataset the dashboard's charts can be on."""
    built = _built(fake, store, THREE)
    plan = _plan_onto(fake, store, BuildPlan(charts=(CLAIMS_TABLE,), measures=(CLAIMED,), datasets=(CLAIM_DATASET,)),
                      built.dashboard_id)
    assert isinstance(plan, BuildPlan), plan
    assert [chart.key for chart in plan.charts] == ["claimed_by_region"] and plan.left_out == ()
    assert plan.datasets and plan.measures


def _tabbed(chart_id: int) -> dict:
    """A layout whose charts sit in top-level tabs, as Superset's dashboard keeps one: the root holds the tabs, not
    the grid, which stays empty."""
    return {"DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["TABS-a"]},
            "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
            "TABS-a": {"type": "TABS", "id": "TABS-a", "children": ["TAB-a"], "parents": ["ROOT_ID"]},
            "TAB-a": {"type": "TAB", "id": "TAB-a", "children": ["ROW-x"], "parents": ["ROOT_ID", "TABS-a"],
                      "meta": {"text": "Overview"}},
            "ROW-x": {"type": "ROW", "id": "ROW-x", "children": ["CHART-x"], "meta": {},
                      "parents": ["ROOT_ID", "TABS-a", "TAB-a"]},
            "CHART-x": {"type": "CHART", "id": "CHART-x", "children": [], "meta": {"chartId": chart_id, "width": 12,
                                                                                 "height": 50},
                        "parents": ["ROOT_ID", "TABS-a", "TAB-a", "ROW-x"]}}


_MAKES = ("generate_chart", "create_dataset", "create_virtual_dataset", "create_chart", "patch_chart", "set_layout",
          "create_dashboard", "add_saved_metric")


@pytest.mark.parametrize("layout", ["tabs", "a node that is not an object", "not an object"])
def test_a_layout_no_chart_can_be_added_to_stops_the_build_at_its_first_step(fake, world, store, layout):
    """A dashboard whose layout mnemiq cannot add rows to (its charts in top-level tabs, or a layout that is not one
    it reads) is refused at the first step, which reads it, before any dataset or chart is made: never at the layout
    step, with every chart made and none placed."""
    dashboard, existing = _existing_dashboard(world)
    grid = json.loads(world.dashboards[dashboard]["position_json"])
    positions = {"tabs": _tabbed(existing), "a node that is not an object": {**grid, "ROW-x": []},
                 "not an object": [grid]}[layout]
    world.dashboards[dashboard]["position_json"] = json.dumps(positions)
    build = _start(store, SIX, dashboard_id=dashboard).build_id
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "dashboard").reason == "layout_unreadable", stopped.message
    assert f"dashboard {dashboard}" in stopped.message and "add charts to" in stopped.message
    assert not set(_tools(fake)) & set(_MAKES), _tools(fake)
    assert _run(store, build, _adapter(fake))[-1].status == "failed" and not set(_tools(fake)) & set(_MAKES)


def test_a_layout_raced_by_a_hand_edit_is_recomputed_once(fake, world, store):
    """The dashboard changed after the layout step read it: `set_layout` answers stale, and the layout is read and
    computed again once, from what the dashboard now holds."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "layout")

    def hand_edit(request, _world):
        dashboard = world.dashboards[request["dashboard_id"]]
        dashboard["changed_on"] = world.superset_now() + timedelta(seconds=1)

    world.hooks["set_layout"] = Hook(before=_once(hand_edit))
    state = _next(store, build, _adapter(fake))
    assert state.ran == "layout" and _step(state, "layout").status == "done", state
    assert len(world.calls("set_layout")) == 2 and len(world.calls("get_layout")) == 2


@pytest.mark.parametrize("tool", ["get_layout", "set_layout"])
def test_a_layout_call_that_goes_unanswered_is_sent_again_at_once(fake, world, store, tool):
    """A call of the layout step that Superset does not answer in time (the layout's read, or its save after Superset
    saved it) leaves the step uncertain at once, nothing waiting on its lease: the next request, with no clock advance,
    runs the step again to done, the layout saved once and the chart on the dashboard once."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "layout")
    ended = world.lose_answer(tool, hold_s=2)
    lost = _next(store, build, _adapter(fake, timeout_s=1))
    assert lost.ran == "layout" and _step(lost, "layout").status == "uncertain", lost
    assert _step(lost, "layout").reason == "superset_silent"
    again = _next(store, build, _adapter(fake))
    assert again.ran == "layout" and _step(again, "layout").status == "done", again
    chart = _step(again, "policies_by_region").object_id
    assert world.dashboards[again.dashboard_id]["members"] == [chart]
    assert len(world.calls("set_layout")) == 1 and ended.wait(10)


def test_an_unreadable_layout_answer_is_a_failed_read(fake, world, store):
    """`get_layout` answering something that is not a layout reads as a layout that could not be read, never as a
    save that failed."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "layout")
    world.hooks["get_layout"] = Hook(instead=lambda request, _w: {"layout": "elsewhere"})
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "layout").reason == "layout_unreadable"
    assert "could not be read" in stopped.message and "could not save" not in stopped.message


# --- the one-step rule, the claim and the lease


def test_concurrent_next_runs_one_step(fake, world, store):
    """Two `/next` at once: one runs the step, the other is refused (409) before it sends Superset anything."""
    build = _start(store, ONE_BAR).build_id
    holding, release = threading.Event(), threading.Event()

    def hold(request, _world):
        holding.set()
        assert release.wait(10)

    world.hooks["create_dashboard"] = Hook(before=hold)
    first: list = []
    worker = threading.Thread(target=lambda: first.append(_next(store, build, _adapter(fake))))
    worker.start()
    assert holding.wait(10)
    with pytest.raises(BuildBusy) as busy:
        _next(store, build, _adapter(fake, FRESH_TOKEN))
    assert busy.value.key == "dashboard" and "already running" in str(busy.value)
    release.set()
    worker.join(10)
    assert first and first[0].ran == "dashboard" and _step(first[0], "dashboard").status == "done"
    assert {c.authorization for c in fake.calls} == {f"Bearer {TOKEN}"}
    assert len(world.dashboards) == 1


# The most any one call may take (`TRANSPORT_MARGIN_S` past the 300 s Superset takes as a statement's limit): two such
# calls in one step run past the 10-minute lease, which only a renewal before the second keeps.
SLOWEST_CALL_S = 315


def test_lease_is_renewed_between_calls(fake, world, store, clock):
    """A step whose two calls take 10.5 minutes in all, past its 10-minute lease, keeps its claim: the layout step reads
    the layout and then saves it, and the lease is renewed before the save, so a second `/next` meanwhile is refused
    and runs nothing."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "layout")
    meanwhile: list = []

    def slow_read(request, answer, _world):
        clock.advance(SLOWEST_CALL_S)

    def slow_save(request, _world):
        clock.advance(SLOWEST_CALL_S - 15)  # past the first claim's lease, within the renewed one
        try:
            _next(store, build, _adapter(fake, FRESH_TOKEN))
        except BuildBusy as busy:
            meanwhile.append(busy)

    world.hooks["get_layout"] = Hook(after=_once(slow_read))
    world.hooks["set_layout"] = Hook(before=_once(slow_save))
    state = _next(store, build, _adapter(fake))
    assert len(meanwhile) == 1 and meanwhile[0].key == "layout"
    assert _step(state, "layout").status == "done"
    assert len(world.calls("get_layout")) == 1 and len(world.calls("set_layout")) == 1
    assert not any(c.authorization == f"Bearer {FRESH_TOKEN}" for c in fake.calls)


def test_worker_past_its_lease_does_not_start_another_call(fake, world, store, clock):
    """A worker whose lease passed during a call starts no further call: the chart's saved metrics were read, and its
    create is never sent. The step is left uncertain, and the next request runs it again, making the chart once."""
    build = _start(store, BuildPlan(charts=(KPI_TOTAL,), measures=())).build_id
    _run_to(store, build, _adapter(fake), "total_premium_kpi")
    world.hooks["dataset_metrics"] = Hook(after=_once(lambda request, answer, _w: clock.advance(601)))
    stalled = _next(store, build, _adapter(fake))
    assert "create_chart" not in _tools(fake) and world.charts == {}
    assert _step(stalled, "total_premium_kpi").status == "uncertain"
    assert _step(stalled, "total_premium_kpi").reason == "lease_passed"
    again = _next(store, build, _adapter(fake))
    assert again.ran == "total_premium_kpi" and _step(again, "total_premium_kpi").status == "done"
    assert len(world.calls("create_chart")) == 1 and len(world.charts) == 1


def test_a_lease_that_passes_during_the_checks_stops_them(fake, world, store, clock):
    """The check step renews its lease before each of its Superset calls, inside each chart's check too: once the
    lease has passed, no further call is started, and the next request checks again."""
    build = _start(store, BuildPlan(charts=(BY_REGION, KPI_POLICIES), measures=(POLICIES,))).build_id
    _run_to(store, build, _adapter(fake), "check")
    world.hooks["get_chart_data"] = Hook(after=_once(lambda request, answer, _w: clock.advance(601)))
    stalled = _next(store, build, _adapter(fake))
    assert _step(stalled, "check").status == "uncertain"
    assert len(world.calls("get_chart_data")) == 1 and "execute_sql" not in _tools(fake)
    final = _next(store, build, _adapter(fake))
    assert final.status == "done" and set(final.checks.values()) == {ChartStatus.CHECKED}


# --- a create whose answer was lost, sent again


# A dataset the build makes on a join, with a chart on it.
JOINED = PlannedDataset(key="dataset_1", placeholder=-1, join="policy JOIN claim ON policy.region = claim.claim_region",
                        sql='SELECT policy."premium" AS "premium", claim."amount" AS "amount" FROM policy JOIN claim '
                            "ON policy.region = claim.claim_region", name="mnemiq_policy_claim")
SHARE = Measure(name="claimed_share", label="Claimed share", expression="SUM(amount)", kind=MeasureKind.NUMBER,
                dataset_id=-1, description="", asked_by="")
ONE_VIRTUAL = BuildPlan(charts=(_chart("claimed_share_kpi", Family.BIG_NUMBER, "Claimed share", -1,
                                       (WrittenRef(SHARE),)),), measures=(SHARE,), datasets=(JOINED,))
ONE_TABLE = BuildPlan(charts=(CLAIMS_TABLE,), measures=(CLAIMED,), datasets=(CLAIM_DATASET,))
ONE_LINE = BuildPlan(charts=(BY_MONTH,), measures=())

# Each create a build sends: the plan, the step that sends it, and the extension's tool.
CREATES = {"dashboard": (ONE_BAR, "dashboard", "create_dashboard"),
           "physical dataset": (ONE_TABLE, "dataset_1", "create_dataset"),
           "virtual dataset": (ONE_VIRTUAL, "dataset_1", "create_dataset"),
           "line chart": (ONE_LINE, "premium_by_month", "create_chart"),
           "funnel": (ONE_FUNNEL, "policies_by_status", "create_chart")}


def _objects(world, tool: str) -> set[int]:
    """The ids of the objects of the kind `tool` makes, as Superset holds them now."""
    return set({"create_dashboard": world.dashboards, "create_dataset": world.datasets,
                "create_chart": world.charts}[tool])


@pytest.mark.parametrize(("plan", "key", "tool"), list(CREATES.values()), ids=list(CREATES))
def test_a_create_whose_answer_was_lost_is_sent_again_at_once(fake, world, store, plan, key, tool):
    """Superset made the object and committed it, and the call ended before its answer came back. The step is left
    uncertain at once, and the next request, with no wait for any lease, sends the same create: the extension answers
    with the object the first one made, found by the step's marker, so the step ends done with its id and Superset
    holds one."""
    build = _start(store, plan).build_id
    _run_to(store, build, _adapter(fake), key)
    before = _objects(world, tool)
    ended = world.lose_answer(tool, hold_s=2)
    lost = _next(store, build, _adapter(fake, timeout_s=1))
    assert _step(lost, key).status == "uncertain" and _step(lost, key).reason == "superset_silent", lost
    assert "while making the" in lost.message and "sends the same create" in lost.message, lost.message
    [made] = _objects(world, tool) - before
    again = _next(store, build, _adapter(fake))
    assert again.ran == key and _step(again, key).status == "done", again
    assert _step(again, key).object_id == made and _objects(world, tool) - before == {made}
    assert len(world.calls(tool)) == 2 and ended.wait(10)


def test_a_create_superset_runs_after_the_retry_makes_one_object(fake, world, store):
    """The first create's call ends unanswered before Superset ran it (held at its door). The step is left uncertain,
    and the next request sends the same create, which makes the chart. Then Superset runs the first call, late: it
    finds the chart the retry made by the step's marker and answers its id, making none."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    late = world.run_late("create_chart")
    try:
        held = _next(store, build, _adapter(fake, timeout_s=1))
        assert _step(held, "policies_by_region").status == "uncertain" and world.charts == {}, held
        made = _next(store, build, _adapter(fake))
        chart = _step(made, "policies_by_region").object_id
        assert _step(made, "policies_by_region").status == "done" and list(world.charts) == [chart], made
    finally:
        late.release.set()
    assert late.ran.wait(10)
    assert late.answers == [{"chart_id": chart}] and list(world.charts) == [chart]


def test_an_unmarked_object_of_the_planned_name_is_never_adopted(fake, world, store):
    """After the chart step began, the builder makes a chart by hand of the planned name on the planned dataset; the
    step's create then makes its own, and its answer is lost. The retry is answered with the chart carrying the
    step's marker, never the hand-made one, which the build leaves as it was and never places."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    by_hand: list[int] = []

    def made_by_hand(request, _world):
        if not by_hand:
            by_hand.append(world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET,
                                             viz_type="echarts_timeseries_bar"))

    world.lose_answer("create_chart", hold_s=2)
    world.hooks["create_chart"].before = made_by_hand
    _next(store, build, _adapter(fake, timeout_s=1))
    [hand] = by_hand
    before = dict(world.charts[hand])
    final = _run(store, build, _adapter(fake))[-1]
    assert final.status == "done", final
    ours = _step(final, "policies_by_region").object_id
    assert ours != hand and sorted(world.charts) == sorted([hand, ours])
    assert parse_marker(world.charts[ours]["description"]) == Marker(build, "policies_by_region", final.dashboard_id)
    assert world.charts[hand] == before and hand not in world.dashboards[final.dashboard_id]["members"]
    assert not [call for call in fake.calls if hand in (call.request.get("chart_id"), call.request.get("identifier"))]


def test_two_objects_carrying_the_steps_marker_stop_the_build_naming_them(fake, world, store):
    """Two charts already carry the step's marker (a person copied one, say): the extension makes nothing and answers
    both, and the step stops naming them, so the person can remove the one not wanted. A later read says why."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    marker = format_marker(build, "policies_by_region", store.load_build(TENANT, build, PRINCIPAL).dashboard_id)
    twins = [world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET,
                               viz_type="echarts_timeseries_bar", description=f"Copied by hand.\n{marker}")
             for _ in range(2)]
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").reason == "candidates", stopped
    assert "more than one object" in stopped.message, stopped.message
    assert all(str(chart_id) in stopped.message for chart_id in twins), stopped.message
    assert sorted(world.charts) == sorted(twins)
    later = build_state(store, store.load_build(TENANT, build, PRINCIPAL))
    assert later.ran is None and "cannot tell" in later.message


def test_candidates_this_builder_cannot_read_stop_the_build_saying_so(fake, world, store):
    """More than one object carries the step's marker, and none is one this builder may read: the extension makes
    nothing and names none, and the step stops saying so."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["create_chart"] = Hook(instead=lambda request, _w: {"candidates": []})
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").reason == "candidates", stopped
    assert "more than one" in stopped.message and "cannot read" in stopped.message, stopped.message
    assert world.charts == {}


def test_candidates_of_which_this_builder_reads_one_stop_the_build_naming_it(fake, world, store):
    """Two objects carry the step's marker and this builder may read one (3): the extension names that one alone.
    The step stops saying more than one object carries the marker, and naming the one it can read, never counting the
    list as the objects Superset holds."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["create_chart"] = Hook(instead=lambda request, _w: {"candidates": [3]})
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").reason == "candidates", stopped
    assert "more than one object" in stopped.message and "can read 3" in stopped.message, stopped.message
    assert "1 objects" not in stopped.message, stopped.message
    assert world.charts == {}


def test_a_busy_create_is_left_uncertain_and_sent_again(fake, world, store):
    """The extension answers busy (the marker's lock, or a row's, was held past its timeout): nothing was made, the
    step is left uncertain, and the next request sends the same create."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["create_chart"] = Hook(instead=_once_answer({"busy": True}))
    held = _next(store, build, _adapter(fake))
    assert _step(held, "policies_by_region").status == "uncertain" and _step(held, "policies_by_region").reason == "busy"
    assert world.charts == {}
    done = _next(store, build, _adapter(fake))
    assert _step(done, "policies_by_region").status == "done" and len(world.charts) == 1
    assert len(world.calls("create_chart")) == 2


def test_a_table_that_already_has_a_dataset_stops_the_build(fake, world, store):
    """The table the plan makes a dataset on already has one this build did not make: the extension answers that it
    exists, and the step stops saying so; nothing is made."""
    build = _start(store, ONE_TABLE).build_id
    _run_to(store, build, _adapter(fake), "dataset_1")
    world.datasets[60] = world._dataset_row(60, "claim", None, "Made by hand", "analyst", world.superset_now())
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "dataset_1").reason == "dataset_exists", stopped
    assert "already has a dataset named 'claim'" in stopped.message
    assert [d for d in world.datasets.values() if d["table_name"] == "claim"] == [world.datasets[60]]


def test_a_build_calls_no_mcp_create_claim_or_patch(fake, world, store):
    """A whole six-chart build, a dataset made for its table among its steps: every object is made by the extension's
    creates, and the build never calls MCP's `generate_chart`, `create_dataset` or `create_virtual_dataset`, a claim,
    or `patch_chart`."""
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    called = {call.tool for call in fake.calls}
    assert not called & {"generate_chart", "create_dataset", "create_virtual_dataset"}, called
    assert not called & {TOOL_PREFIX + name for name in ("claim_chart", "claim_dataset", "patch_chart")}, called
    assert {TOOL_PREFIX + name for name in ("create_dashboard", "create_dataset", "create_chart")} <= called


@pytest.mark.parametrize(("plan", "key", "tool"), [CREATES["dashboard"], CREATES["physical dataset"],
                                                   CREATES["line chart"]], ids=["dashboard", "dataset", "chart"])
def test_a_create_the_extension_refuses_stops_the_step_naming_it(fake, world, store, plan, key, tool):
    """The extension refuses a create (here because the metadata database's isolation is not one its marker lookup
    reads committed rows under): the step fails as refused by the extension, naming why, and nothing is made."""
    build = _start(store, plan).build_id
    _run_to(store, build, _adapter(fake), key)
    before = _objects(world, tool)
    world.hooks[tool] = Hook(instead=lambda request, _w: {"refused": "isolation"})
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, key).reason == "extension_refused", stopped
    assert "isolation" in stopped.message and _objects(world, tool) == before


# --- a step run again: found by its marker, whatever it is called, and never another's


def _made_once(world) -> dict:
    [made] = [c for c in world.charts.values() if c["slice_name"] == "Policies by region"]
    return made


class _Died(BaseException):
    """The worker's process ending: nothing in the builder catches it."""


def test_crash_after_commit_is_reconciled(fake, world, store, clock, monkeypatch):
    """Superset commits the chart's create, and the worker dies before it records the step. While the dead worker's
    lease holds, `/next` is refused; once it has passed, the next request sends the same create, which answers with
    the chart Superset made, and no second is made."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    finish = store.finish_step

    def dies_before_it_records(tenant, build_id, key, *args, **kwargs):
        if key == "policies_by_region" and not getattr(dies_before_it_records, "died", False):
            dies_before_it_records.died = True
            raise _Died()
        return finish(tenant, build_id, key, *args, **kwargs)

    monkeypatch.setattr(store, "finish_step", dies_before_it_records)
    with pytest.raises(_Died):
        _next(store, build, _adapter(fake))
    [made] = world.charts
    assert _raw_step(store, build, "policies_by_region")["status"] == "pending"  # nothing was finished
    with pytest.raises(BuildBusy):
        _next(store, build, _adapter(fake))  # the dead worker's lease still holds
    clock.advance(601)
    reconciled = _next(store, build, _adapter(fake))
    assert reconciled.ran == "policies_by_region" and _step(reconciled, "policies_by_region").status == "done"
    assert _step(reconciled, "policies_by_region").object_id == made
    assert list(world.charts) == [made] and len(world.calls("create_chart")) == 2


@pytest.mark.parametrize("whose", ["another build", "another step of this build", "another dashboard"])
def test_another_builds_chart_is_never_adopted(fake, world, store, whose):
    """A chart of the planned name, on the planned dataset, carrying a marker that is not this step's (compared whole,
    as its description's last line): never answered for this step, nor changed. The step's create makes its own."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    dashboard = store.load_build(TENANT, build, PRINCIPAL).dashboard_id
    marker = {"another build": format_marker("0" * 8 + "-0000-4000-8000-" + "0" * 12, "policies_by_region", dashboard),
              "another step of this build": format_marker(build, "policies_kpi", dashboard),
              "another dashboard": format_marker(build, "policies_by_region", dashboard + 1)}[whose]
    theirs = world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET, viz_type="echarts_timeseries_bar",
                               description=f"Theirs.\n{marker}")
    before = dict(world.charts[theirs])
    final = _run(store, build, _adapter(fake))[-1]
    assert final.status == "done", final
    ours = _step(final, "policies_by_region").object_id
    assert ours != theirs and world.charts[theirs] == before
    assert len(world.calls("create_chart")) == 1
    assert [c.request for c in fake.calls if c.request.get("chart_id") == theirs] == []


@pytest.mark.parametrize("made", ["dashboard", "funnel"])
def test_an_object_renamed_before_the_retry_is_found_by_its_marker(fake, world, store, made):
    """The create made the object and its answer was lost, and a person renamed the object before the retry: the
    retry finds it by the step's marker, whatever it is called by then, and makes no second."""
    plan, key, tool = CREATES[made]
    rows, title, answered = ((world.dashboards, "dashboard_title", "dashboard_id") if tool == "create_dashboard"
                             else (world.charts, "slice_name", "chart_id"))
    build = _start(store, plan).build_id
    _run_to(store, build, _adapter(fake), key)

    def renamed(request, answer, _world):
        rows[answer[answered]][title] = "Renamed by hand"
        time.sleep(2)

    world.hooks[tool] = Hook(after=_once(renamed))
    assert _step(_next(store, build, _adapter(fake, timeout_s=1)), key).status == "uncertain"
    state = _next(store, build, _adapter(fake))
    [row] = rows.values()
    assert _step(state, key).status == "done" and _step(state, key).object_id == row["id"], state
    assert row[title] == "Renamed by hand" and len(world.calls(tool)) == 2


def _too_many_editors(world, dashboard: int, *, is_editor: bool):
    """`editors_of` as the extension answers a dashboard whose editors' labels are more than it sends."""
    world.hooks["editors_of"] = Hook(instead=lambda request, _w: {"charts": [], "datasets": [], "dashboards": [
        {"dashboard_id": dashboard, "is_editor": is_editor, "too_large": True}]})


def test_an_editor_whose_dashboard_has_too_many_editors_to_send_is_an_editor(fake, world, store):
    """The extension cut the labels but kept `is_editor`: the builder edits the dashboard, so the build goes on."""
    dashboard, _ = _existing_dashboard(world)
    _too_many_editors(world, dashboard, is_editor=True)
    final = _run(store, _start(store, ONE_BAR, dashboard_id=dashboard).build_id, _adapter(fake))[-1]
    assert final.status == "done", final


def test_a_non_editor_of_a_dashboard_with_too_many_editors_is_refused(fake, world, store):
    dashboard, _ = _existing_dashboard(world, editors=("colleague",))
    _too_many_editors(world, dashboard, is_editor=False)
    stopped = _next(store, _start(store, ONE_BAR, dashboard_id=dashboard).build_id, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "dashboard").reason == "not_editor"
    assert "not an editor" in stopped.message


def test_the_lease_is_renewed_before_the_extensions_probe(fake, world, store, monkeypatch):
    """An extension call failing as a missing tool is followed by the probe that tells a missing extension from a
    failing tool: the probe is a call of the step too, so the lease is renewed before it."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    renewals: list[int] = []
    renew = store.renew_lease

    def counted(*args, **kwargs):
        renewals.append(len(fake.calls))
        return renew(*args, **kwargs)

    monkeypatch.setattr(store, "renew_lease", counted)

    # What rc2's MCP answers for a tool it does not have, and for an unexpected error in one it has: the fake sends an
    # exception a handler returns as Superset's error result.
    masked = RuntimeError(f"Internal error in {TOOL_PREFIX}create_chart: An unexpected error occurred. Error ID: 1a.")
    world.hooks["create_chart"] = Hook(instead=lambda request, _w: masked)
    with pytest.raises(SupersetToolError):
        _next(store, build, _adapter(fake))
    probe = next(i for i, c in enumerate(fake.calls) if c.tool == TOOL_PREFIX + "dataset_metrics"
                 and c.request == {"dataset_ids": []})
    create = next(i for i, c in enumerate(fake.calls) if c.tool == TOOL_PREFIX + "create_chart")
    assert create < probe and probe in renewals, (create, probe, renewals)


# --- a create that answers failure, and one that keeps failing


def test_a_create_that_answers_failure_and_made_nothing_fails(fake, world, store):
    """The extension's command failed, which it answers by its kind: nothing was made (the command's transaction
    rolled back), and the step stops, in mnemiq's words."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["create_chart"] = Hook(instead=lambda request, _w: {"failed": True})
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").reason == "not_saved"
    assert "could not save the chart 'Policies by region'" in stopped.message and world.charts == {}


def test_a_create_whose_tool_keeps_failing_stops_after_two(fake, world, store):
    """A create whose tool fails is left uncertain and raised; the next request sends it once more; failing the same
    way again, the step fails with its reason, so the build does not retry forever."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")

    def fails(request, _world):
        raise RuntimeError("Superset's chart tool broke")

    world.hooks["create_chart"] = Hook(before=fails)
    with pytest.raises(SupersetToolError):
        _next(store, build, _adapter(fake))
    first = build_state(store, store.load_build(TENANT, build, PRINCIPAL))
    assert _step(first, "policies_by_region").status == "uncertain"
    assert _step(first, "policies_by_region").reason == "tool_error"
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").reason == "tool_error"
    assert len(world.calls("create_chart")) == 2 and "broke" not in stopped.message


def test_an_abandoned_build_releases_its_dashboard(fake, world, store, clock):
    """A build left untouched for twice its lease no longer holds its dashboard: the next build of it fails the old
    build's next step as abandoned, in the same write as its own creation, and a later read of the old one says
    why. A build touched more recently than that still holds it."""
    dashboard, _ = _existing_dashboard(world)
    old = _start(store, ONE_BAR, dashboard_id=dashboard).build_id
    _next(store, old, _adapter(fake))
    clock.advance(2 * 600 - 1)
    with pytest.raises(DashboardBusy):
        _start(store, ONE_BAR, dashboard_id=dashboard)
    clock.advance(2)
    fresh = _start(store, ONE_BAR, dashboard_id=dashboard)
    assert fresh.status == "building"
    stale = build_state(store, store.load_build(TENANT, old, PRINCIPAL))
    assert stale.status == "failed" and _step(stale, "policies_by_region").reason == "abandoned"
    assert "untouched" in stale.message


def test_a_build_holding_a_dashboard_id_before_it_exists_stalls_a_new_dashboard_only_until_it_is_abandoned(
        fake, world, store, clock):
    """A build started on the id Superset will give the next dashboard made (the start route refuses that id, which
    no dashboard has yet; the store takes it from anyone) holds that id's one live build. Another person's new
    dashboard then gets the id, and its step meets the holder: refused while the holder may still be live. Once the
    holder has been left untouched for twice the lease, the step releases it, as starting a build of that dashboard
    would, and finishes. One dashboard is made throughout."""
    squatter = start_build(store, tenant=TENANT, principal="superset:squatter", goal=GOAL, plan=ONE_BAR,
                           dashboard_id=300).build_id
    assert not world.dashboards  # 300 is the id the World gives the next dashboard
    build = _start(store, ONE_BAR).build_id
    outcomes = []
    for _ in range(3):
        try:
            outcomes.append(_step(_next(store, build, _adapter(fake)), "dashboard").status)
        except DashboardBusy:
            outcomes.append("busy")
        clock.advance(601)
    assert outcomes == ["busy", "busy", "done"]
    assert sorted(world.dashboards) == [300]
    assert store.load_build(TENANT, build, PRINCIPAL).dashboard_id == 300
    stale = store.load_build(TENANT, squatter, "superset:squatter")
    assert stale.status == "failed" and stale.step("dashboard").reason == "abandoned"
    assert _run(store, build, _adapter(fake))[-1].status == "done"


# --- a token that expires between steps


def test_expired_token_between_steps_resumes(fake, world, store):
    """Superset stops accepting the builder's token once the chart is made, so the create's own answer is lost (the
    client's last request of the call is refused): the request fails as a refusal of the caller (sign in again), the
    build stays at its last completed step with this one uncertain, and the next request, with a fresh token, sends
    the same create, which answers with the chart that was made, never making a second."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["create_chart"] = Hook(after=_once(lambda request, answer, _w: fake.refuse_token(TOKEN)))
    with pytest.raises(SupersetRefusedCaller):
        _next(store, build, _adapter(fake))
    kept = build_state(store, store.load_build(TENANT, build, PRINCIPAL))
    assert [step.status for step in kept.steps] == ["done", "uncertain", "pending", "pending"]
    assert kept.status == "building" and _step(kept, "policies_by_region").object_id is None
    made = _made_once(world)
    final = _run(store, build, _adapter(fake, FRESH_TOKEN))[-1]
    assert final.status == "done", final
    assert _step(final, "policies_by_region").object_id == made["id"] and list(world.charts) == [made["id"]]
    assert [c.authorization for c in world.calls("create_chart")] == [f"Bearer {TOKEN}", f"Bearer {FRESH_TOKEN}"]


def _once_answer(answer: dict):
    """An `instead` hook answering `answer` the first time, then letting the tool run."""
    sent = []

    def instead(request, _world):
        if not sent:
            sent.append(True)
            return answer
        return None

    return instead


# --- the dashboard's measures


def test_a_chart_of_a_reused_saved_metric_keeps_its_check_status(fake, world, store):
    """A reused saved metric is kept as a measure of the dashboard too, naming the metric, so its charts' check
    statuses survive a later read and the dashboard's measure list holds it."""
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]
    ids = _chart_ids(final)
    reread = build_state(store, store.load_build(TENANT, final.build_id, PRINCIPAL))
    assert reread.checks == final.checks and set(reread.checks) == set(ids.values())
    [saved] = [m for m in store.measures_for(TENANT, final.dashboard_id) if m.saved_metric]
    assert saved.saved_metric == "total_premium" and saved.measure.expression == "SUM(premium)"
    assert saved.measure.kind is MeasureKind.CURRENCY and saved.measure.dataset_id == POLICY_DATASET
    assert sorted(saved.chart_ids) == sorted(ids[k] for k in ("total_premium_kpi", "premium_by_month",
                                                              "premium_by_channel"))
    assert saved.measure.asked_by == PRINCIPAL and saved.made_dataset_ids == ()


def _kept_measure(store, dashboard: int, measure: Measure, chart_ids: tuple) -> None:
    store.put_measure(TENANT, dashboard, dataclasses.replace(measure, asked_by="superset:someone"),
                      chart_ids=chart_ids, check_status={chart_ids[0]: ChartStatus.CHECKED})


def test_a_measure_the_dashboard_holds_with_another_definition_stops_the_build_at_its_first_step(fake, world,
                                                                                               store):
    """On a dashboard that already keeps a measure of a planned name with another definition, the build stops at its
    first step, which checks every measure the plan writes, before anything is made: here the conflict is the last
    chart's, after five that conflict with nothing, and the dataset the plan makes for it is not made either. The
    measure kept is unchanged."""
    dashboard, existing = _existing_dashboard(world)
    other = dataclasses.replace(CLAIMED, dataset_id=POLICY_DATASET, expression="SUM(premium)")
    _kept_measure(store, dashboard, other, (existing,))
    assert [chart.key for chart in SIX.charts][-1] == "claimed_by_region"
    build = _start(store, SIX, dashboard_id=dashboard).build_id
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "dashboard").reason == "measure_conflict", stopped.message
    assert "'claimed'" in stopped.message and "nothing" in stopped.message
    assert not set(_tools(fake)) & set(_MAKES), _tools(fake)
    [kept] = store.measures_for(TENANT, dashboard)
    assert kept.measure.expression == "SUM(premium)" and kept.chart_ids == (existing,)


def test_a_measure_kept_with_another_definition_after_the_first_step_still_stops_its_chart(fake, world, store):
    """A measure the dashboard comes to keep under a planned name after the first step (Superset mode keeps one build
    of a dashboard at a time, but a dashboard's measures can be changed apart from builds) still stops the chart that
    would show it, before the chart is made."""
    dashboard, existing = _existing_dashboard(world)
    build = _start(store, ONE_BAR, dashboard_id=dashboard).build_id
    assert _step(_next(store, build, _adapter(fake)), "dashboard").status == "done"
    _kept_measure(store, dashboard, dataclasses.replace(POLICIES, expression="COUNT(DISTINCT region)"), (existing,))
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").reason == "measure_conflict"
    assert "create_chart" not in _tools(fake) and "'policies'" in stopped.message


def test_the_same_definition_merges_its_charts(fake, world, store):
    """The same measure (its expression equal once normalized, its kind and dataset the same) gains the new chart
    beside those it had, and keeps their statuses."""
    dashboard, existing = _existing_dashboard(world)
    _kept_measure(store, dashboard, dataclasses.replace(POLICIES, expression="count( policy_id )"), (existing,))
    final = _run(store, _start(store, ONE_BAR, dashboard_id=dashboard).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    [kept] = store.measures_for(TENANT, dashboard)
    chart = _step(final, "policies_by_region").object_id
    assert kept.chart_ids == (existing, chart)
    assert kept.check_status == {existing: ChartStatus.CHECKED, chart: ChartStatus.CHECKED}


# --- what a build's state carries, and what it keeps


def test_a_build_state_never_carries_a_lease_id(fake, world, store):
    """The build routes answer with a `BuildState`: no lease id is in one, while a step runs or after. (A
    `not_editor` message names editors: it is for the HTTP answer only, never a model-visible field.)"""
    build = _start(store, ONE_BAR).build_id
    seen: list[str] = []

    def note_lease(request, _world):
        seen.extend(step["lease_id"] for step in store.raw_steps(TENANT, build) if step["lease_id"])
        mid = build_state(store, store.load_build(TENANT, build, PRINCIPAL))
        assert all(step.lease_id is None for step in mid.steps)
        assert not any(lease in json.dumps(dataclasses.asdict(mid), default=str) for lease in seen)

    world.hooks["create_dashboard"] = Hook(before=note_lease)
    states = _run(store, build, _adapter(fake))
    assert seen
    for state in states:
        assert all(step.lease_id is None for step in state.steps)
        assert not any(lease in json.dumps(dataclasses.asdict(state), default=str) for lease in seen)


def test_a_finish_whose_answer_was_lost_is_read_back(fake, world, store, monkeypatch):
    """The store's connection drops on `finish_step`, after the finish committed or before: the builder reads the
    build back before it decides, so a finish that committed is not mistaken for a lost lease, and one that did not
    is made again. The step ends done either way, once."""
    finish = store.finish_step
    dropped: list[str] = []

    def drops(tenant, build_id, key, status, object_id, **kept):
        if key == "dashboard" and not dropped:
            dropped.append("after")
            finish(tenant, build_id, key, status, object_id, **kept)
            raise psycopg.OperationalError("the connection dropped after the commit")
        if key == "policies_by_region" and len(dropped) == 1:
            dropped.append("before")
            raise psycopg.OperationalError("the connection dropped before the commit")
        return finish(tenant, build_id, key, status, object_id, **kept)

    monkeypatch.setattr(store, "finish_step", drops)
    final = _run(store, _start(store, ONE_BAR).build_id, _adapter(fake))[-1]
    assert dropped == ["after", "before"]
    assert final.status == "done" and len(world.dashboards) == 1 and len(world.calls("create_chart")) == 1


def test_one_live_build_per_dashboard(store):
    """A dashboard's measure list is replaced by its build's last write, so a second build of a dashboard is refused
    while one is building; another tenant's, or once the first has ended, is not."""
    first = _start(store, ONE_BAR, dashboard_id=900)
    with pytest.raises(DashboardBusy):
        _start(store, ONE_BAR, dashboard_id=900)
    start_build(store, tenant="tenant-b", principal=PRINCIPAL, goal=GOAL, plan=ONE_BAR, dashboard_id=900)
    _start(store, ONE_BAR, dashboard_id=901)
    raw = store._builds[(TENANT, first.build_id)]
    raw["status"] = "failed"
    _start(store, ONE_BAR, dashboard_id=900)


def test_a_plan_awaiting_its_dashboard_choice_is_not_built(store):
    with pytest.raises(ValueError):
        _start(store, BuildPlan(charts=ONE_BAR.charts, measures=ONE_BAR.measures, reuse_candidates=(5,)))


def test_the_steps_follow_the_plan():
    """The dashboard, each planned dataset, each chart, the layout, the checks: the store's kinds, in its order."""
    assert build_steps(SIX) == [("dashboard", "dashboard"), ("dataset_1", "dataset"),
                                *((chart.key, "chart") for chart in SIX.charts), ("layout", "layout"),
                                ("check", "check")]


@pytest.mark.parametrize("plan, virtual", [(ONE_VIRTUAL, True), (ONE_TABLE, False)], ids=["joined", "physical"])
def test_the_check_is_handed_each_virtual_dataset_the_build_made_with_its_sql(fake, world, store, monkeypatch, plan,
                                                                              virtual):
    """The check reads a virtual dataset with mnemiq's own SQL only when it knows the build made it and with which
    SQL: the build hands it exactly the virtual datasets it made, each with the SQL it sent, and no physical one."""
    import mnemiq.superset.builder as builder_module

    handed: list = []
    real = builder_module.check_chart

    def check(*args, **kwargs):
        handed.append(kwargs.get("made"))
        return real(*args, **kwargs)

    monkeypatch.setattr(builder_module, "check_chart", check)
    final = _run(store, _start(store, plan).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    made = _step(final, "dataset_1").object_id
    assert handed == [{made: plan.datasets[0].sql} if virtual else {}]
    if virtual:
        assert world.datasets[made]["sql"] == plan.datasets[0].sql


def test_a_virtual_dataset_is_named_for_its_build(fake, world, store):
    """Two builds that join the same tables make two datasets: each is named for its build, so neither collides with
    the other's name, and each carries its marker after a line saying mnemiq made it for the dashboard."""
    names = []
    for _ in range(2):
        state = _start(store, ONE_VIRTUAL)
        for _ in range(2):
            state = _next(store, state.build_id, _adapter(fake))
        made = world.datasets[_step(state, "dataset_1").object_id]
        assert made["table_name"].startswith("mnemiq_policy_claim_") and state.build_id[:8] in made["table_name"]
        lines = made["description"].splitlines()
        assert lines[-1] == format_marker(state.build_id, "dataset_1", state.dashboard_id)
        assert str(state.dashboard_id) in lines[0] and "mnemiq" in lines[0]
        names.append(made["table_name"])
    assert len(set(names)) == 2


def test_a_joined_tables_saved_metric_is_built_as_the_join_names_its_columns(fake, world, store):
    """The planner carries policy's saved metric onto a join with claim, and the build sends it as written: the chart's
    SQL metric is the metric's expression naming the joined dataset's columns, on the dataset the build made over the
    join, under the metric's verbose name, and the dashboard keeps it as one of its measures. Superset's numbers on
    the dataset the build made are checked against mnemiq's own SQL over the dataset's SQL."""
    from mnemiq.superset.planner import FakePlanner, plan_build

    world.datasets[POLICY_DATASET]["metrics"].append(
        {"metric_name": "active_premium", "expression": "SUM(CASE WHEN status = 'active' THEN premium END)",
         "verbose_name": "Active premium", "d3format": None, "currency": None, "certified_by": "finance",
         "certification_details": None, "warning_text": None, "description": None})
    join = "policy JOIN claim ON claim.claim_id = policy.policy_id"
    ref = SavedRef("active_premium", MeasureKind.CURRENCY, table="policy", handle="active_premium")
    planned = BuildPlan(charts=(_chart("active_by_claim_region", Family.BAR, "Active premium by claim region", -1, (ref,),
                                       dimensions=("claim_region",)),),
                        measures=(), datasets=(PlannedDataset(key="dataset_1", placeholder=-1, join=join),))
    adapter = _adapter(fake)
    plan = plan_build(GOAL, planner=FakePlanner(planned), client=adapter.client, adapter=adapter, snapshot=SNAPSHOT,
                      grants=GRANTS, settings=SETTINGS)
    assert isinstance(plan, BuildPlan), plan
    rendered = 'SUM(CASE WHEN "status" = \'active\' THEN "premium" END)'
    [measure] = plan.measures
    assert measure.expression == rendered and plan.charts[0].measures == (WrittenRef(measure),)
    final = _run(store, _start(store, plan).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    [request] = [call.request for call in world.calls("create_chart")]
    made = _step(final, "dataset_1").object_id
    assert request["dataset_id"] == made and world.datasets[made]["sql"] == plan.datasets[0].sql
    assert request["config"]["y"] == [{"sql_expression": rendered, "label": "Active premium"}]
    chart_id = _step(final, "active_by_claim_region").object_id
    assert final.checks == {chart_id: ChartStatus.CHECKED}, final.message
    [kept] = store.measures_for(TENANT, final.dashboard_id)
    assert kept.measure == dataclasses.replace(measure, dataset_id=made, asked_by=PRINCIPAL)
    assert kept.measure.description == "The saved metric 'active_premium' of 'policy', carried onto the join."


# --- The same build on the Postgres store: the builder joined to the store it runs on in a deployment, with
# Superset's clock the real one, so a step's window compares Postgres's `now()` with it as a deployment does. Gated
# as the store's own tests are (`MNEMIQ_TEST_BUILD_STORE_DSN`, and `MNEMIQ_TEST_BUILD_STORE_REQUIRED` in the job that
# gives them a server).


class _RealClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


@pytest.fixture
def pg_dsn():
    """A schema of its own in the store tests' database, as the DSN that makes it the search path."""
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from tests.test_superset_build_store import _database_dsn

    database = _database_dsn()
    schema = f"builder_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield make_conninfo(database, options=f"-c search_path={schema}")
    finally:
        with psycopg.connect(database, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def pg_store(pg_dsn):
    from mnemiq.superset.build_store import BuildStore

    store = BuildStore(pg_dsn)
    try:
        yield store
    finally:
        store.close()


def test_a_build_runs_on_the_postgres_store(fake, pg_store):
    """The six-chart build, end to end, on the store a deployment runs: every step one per request, every chart
    checked, the measures kept; then a second build of the dashboard is allowed, and a third refused while it builds."""
    world = World(fake, _RealClock())
    started = _start(pg_store)
    states = _run(pg_store, started.build_id, _adapter(fake))
    final = states[-1]
    assert [state.ran for state in states] == [step.key for step in started.steps]
    assert final.status == "done" and set(final.checks.values()) == {ChartStatus.CHECKED}, final.message
    kept = {m.measure.name: m for m in pg_store.measures_for(TENANT, final.dashboard_id)}
    assert {m.saved_metric for m in kept.values()} == {None, "total_premium"}
    assert kept["claimed"].measure.asked_by == PRINCIPAL
    # A later read has every chart's status, a chart of a reused saved metric's too: each is kept with a measure row.
    reread = build_state(pg_store, pg_store.load_build(TENANT, final.build_id, PRINCIPAL))
    assert reread.checks == final.checks and set(reread.checks) == set(_chart_ids(final).values())
    assert reread.ran is None
    again = _start(pg_store, ONE_BAR, dashboard_id=final.dashboard_id)
    with pytest.raises(DashboardBusy):
        _start(pg_store, ONE_BAR, dashboard_id=final.dashboard_id)
    assert again.status == "building" and world.dashboards[final.dashboard_id]["published"] is False


SENTINEL = 987654.321


def _with_sentinel(rows_key: str):
    """An `after` hook putting the sentinel in place of every number of the answer's rows, a column of whole numbers
    then typed as a float column (`execute_sql` names each column's pandas dtype)."""
    def after(request, answer, _world):
        if not isinstance(answer, dict) or not answer.get(rows_key):
            return answer
        answer = {**answer, rows_key: [dict(row) for row in answer[rows_key]]}
        for row in answer[rows_key]:
            row.update({key: SENTINEL for key, value in row.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)})
        if isinstance(answer.get("columns"), list):
            answer["columns"] = [{**column, "type": "float64"} if column.get("type") == "int64" else column
                                 for column in answer["columns"]]
        return answer

    return after


def test_sentinel_value_never_persisted(fake, pg_store, pg_dsn, monkeypatch):
    """After a full build on the Postgres store, its check step among its steps, with Superset answering both the
    chart's data (`get_chart_data`) and mnemiq's own statement (`execute_sql`) with `987654.321`: the check compared
    that number and the chart checked, and no stored row's text holds it, nor its rounding to cents. The probe is
    shown to read the rows: it finds the goal and the chart's id, and the number once it is planted past the store."""
    from psycopg import sql

    import mnemiq.superset.builder as builder_module
    from mnemiq.superset.build_store import BUILD_TABLE
    from tests.test_superset_build_store import STORE_COLUMNS

    world = World(fake, _RealClock())
    world.hooks["get_chart_data"] = Hook(after=_with_sentinel("data"))
    world.hooks["execute_sql"] = Hook(after=_with_sentinel("rows"))
    compared: list = []
    real = builder_module.check_chart

    def check(*args, **kwargs):
        found = real(*args, **kwargs)
        compared.append((found.chart_values, found.mnemiq_values))
        return found

    monkeypatch.setattr(builder_module, "check_chart", check)
    final = _run(pg_store, _start(pg_store, ONE_BAR).build_id, _adapter(fake))[-1]
    chart_id = _step(final, "policies_by_region").object_id
    assert final.status == "done" and final.checks == {chart_id: ChartStatus.CHECKED}, final.message
    [(chart, mine)] = compared
    assert chart == mine and all(row[-1] == SENTINEL for row in chart), chart  # the number reached the check

    def stored_text(con) -> str:
        return "\n".join(text for table in STORE_COLUMNS for (text,) in con.execute(
            sql.SQL("SELECT row_to_json(t)::text FROM {} AS t").format(sql.Identifier(table))))

    with psycopg.connect(pg_dsn) as con:
        text = stored_text(con)
        assert str(SENTINEL) not in text and "987654.32" not in text
        assert GOAL in text and f'"object_id": {chart_id}' in text  # the probe reads what the build kept
        con.execute(sql.SQL("UPDATE {} SET goal = goal || %s").format(sql.Identifier(BUILD_TABLE)),
                    (str(SENTINEL),))
        assert str(SENTINEL) in stored_text(con)  # and finds the number once it is there
        con.rollback()


def test_a_lost_create_answer_is_sent_again_on_the_postgres_store(fake, pg_store):
    """A chart made and its answer lost, on the Postgres store: the step is left uncertain at once, so the very next
    `/next` claims it, by Postgres's own clock, sends the same create and is answered with the chart; no second is
    made."""
    world = World(fake, _RealClock())
    build = _start(pg_store, ONE_BAR).build_id
    _run_to(pg_store, build, _adapter(fake), "policies_by_region")
    ended = world.lose_answer("create_chart", hold_s=2)
    lost = _next(pg_store, build, _adapter(fake, timeout_s=1))
    assert _step(lost, "policies_by_region").status == "uncertain", lost
    again = _next(pg_store, build, _adapter(fake))
    assert _step(again, "policies_by_region").status == "done", again
    assert list(world.charts) == [_step(again, "policies_by_region").object_id]
    assert len(world.calls("create_chart")) == 2 and ended.wait(10)


def test_two_next_at_once_on_the_postgres_store_run_one_step(fake, pg_store):
    """Two requests claim one step at once through the Postgres store's compare-and-set: one runs it, one is
    refused before it calls Superset."""
    world = World(fake, _RealClock())
    build = _start(pg_store, ONE_BAR).build_id
    holding, release = threading.Event(), threading.Event()
    world.hooks["create_dashboard"] = Hook(before=lambda request, _w: (holding.set(), release.wait(10)))
    first: list = []
    worker = threading.Thread(target=lambda: first.append(_next(pg_store, build, _adapter(fake))))
    worker.start()
    assert holding.wait(10)
    with pytest.raises(BuildBusy):
        _next(pg_store, build, _adapter(fake, FRESH_TOKEN))
    release.set()
    worker.join(10)
    assert first and _step(first[0], "dashboard").status == "done" and len(world.dashboards) == 1


MCP_2X = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "mcp-2x.yml"
STORE_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "superset-build-store.yml"


@pytest.mark.parametrize("workflow", [MCP_2X, STORE_WORKFLOW], ids=["mcp 2.x", "the store's Postgres"])
def test_the_workflows_run_this_module(workflow):
    """A build drives `SupersetMcp` over the wire to `FakeSuperset`, so this module runs where the client meets mcp
    2.x; and on the job with a Postgres of its own, where the builds on that store cannot skip. Each workflow's one
    pytest line names it."""
    runs = [line for line in workflow.read_text().splitlines() if re.search(r"\brun:.*\bpytest\b", line)]
    assert len(runs) == 1, f"expected one pytest line in {workflow.name}, found {runs}"
    assert f"tests/{Path(__file__).name}" in runs[0].split()



def _lapsed_bar(ending: str) -> tuple[Measure, ChartSpec]:
    lapsed = Measure(name="lapsed_premium", label="Lapsed premium",
                     expression=f"SUM(CASE WHEN status = 'lapsed' THEN premium {ending})", kind=MeasureKind.CURRENCY,
                     dataset_id=POLICY_DATASET, description="", asked_by="", currency="USD")
    chart = dataclasses.replace(_chart("lapsed_by_region", Family.BAR, "Lapsed premium by region", POLICY_DATASET,
                                       (WrittenRef(lapsed),), dimensions=("region",)), row_limit=2)
    return lapsed, chart


@pytest.mark.parametrize("ending", ["END", "ELSE 0 END"], ids=["no value is NULL", "no value is 0"])
def test_a_ranked_bar_whose_row_limit_cuts_groups_is_checked_as_a_ranking(fake, world, store, ending):
    """Lapsed premium by region, two bars of three, east having no lapsed policy. The chart is made with the HAVING
    that leaves out the groups with no value, so written with no ELSE east is left out and the two bars are the two
    regions that have one, checked; written to give 0 there, the two largest of three are kept, checked as a
    ranking."""
    lapsed, chart = _lapsed_bar(ending)
    final = _run(store, _start(store, BuildPlan(charts=(chart,), measures=(lapsed,))).build_id, _adapter(fake))[-1]
    chart_id = _step(final, "lapsed_by_region").object_id
    assert final.status == "done" and final.checks == {chart_id: ChartStatus.CHECKED}, final.message
    [having] = [f for f in world.charts[chart_id]["params"]["adhoc_filters"] if f.get("clause") == "HAVING"]
    assert having["sqlExpression"] == f"({lapsed.expression}) IS NOT NULL"


def test_an_extension_without_leave_out_no_value_stops_the_chart_step_and_makes_no_chart(fake, world, store):
    world.without_leave_out = True
    lapsed, chart = _lapsed_bar("END")
    states = _run(store, _start(store, BuildPlan(charts=(chart,), measures=(lapsed,))).build_id, _adapter(fake))
    final = states[-1]
    assert final.status == "failed", final
    assert "older than this mnemiq" in final.message and "update the extension" in final.message, final.message
    assert world.charts == {}


def test_a_carried_metric_ranked_on_a_join_the_build_made_carries_its_having_on_the_joins_columns(fake, world, store):
    """Policy's saved metric carried onto a join with claim, by claim region, one bar: the chart is made with the
    HAVING of the metric as the joined dataset names its columns, and the check over the dataset the build made reads
    the cut as a ranking."""
    from mnemiq.superset.planner import FakePlanner, plan_build

    world.datasets[POLICY_DATASET]["metrics"].append(
        {"metric_name": "active_premium", "expression": "SUM(CASE WHEN status = 'active' THEN premium END)",
         "verbose_name": "Active premium", "d3format": None, "currency": None, "certified_by": "finance",
         "certification_details": None, "warning_text": None, "description": None})
    join = "policy JOIN claim ON claim.claim_id = policy.policy_id"
    ref = SavedRef("active_premium", MeasureKind.CURRENCY, table="policy", handle="active_premium")
    chart = dataclasses.replace(_chart("active_by_claim_region", Family.BAR, "Active premium by claim region", -1,
                                       (ref,), dimensions=("claim_region",)), row_limit=1)
    planned = BuildPlan(charts=(chart,), measures=(), datasets=(PlannedDataset(key="dataset_1", placeholder=-1,
                                                                                join=join),))
    adapter = _adapter(fake)
    plan = plan_build(GOAL, planner=FakePlanner(planned), client=adapter.client, adapter=adapter, snapshot=SNAPSHOT,
                      grants=GRANTS, settings=SETTINGS)
    assert isinstance(plan, BuildPlan), plan
    [measure] = plan.measures
    final = _run(store, _start(store, plan).build_id, _adapter(fake))[-1]
    [request] = [call.request for call in world.calls("create_chart")]
    assert request["leave_out_no_value"] is True and '"status"' in measure.expression
    chart_id = _step(final, "active_by_claim_region").object_id
    [left_out] = [f for f in world.charts[chart_id]["params"]["adhoc_filters"] if f.get("clause") == "HAVING"]
    assert left_out["sqlExpression"] == f"({measure.expression}) IS NOT NULL"
    assert final.status == "done" and final.checks == {chart_id: ChartStatus.CHECKED}, final.message

"""A dashboard build, one step per request: the dashboard, its datasets, its charts, the layout and the checks, each step
claimed with its intent before any Superset call, its lease renewed before each call, and a step whose outcome is
uncertain reconciled by the build's marker before anything else, never repeated and never adopting another build's.

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
from zoneinfo import ZoneInfo

import psycopg
import pytest

from mnemiq.config import Settings
from mnemiq.superset import builder
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.build_store import DashboardBusy, StepState
from mnemiq.superset.builder import BuildBusy, BuildState, build_state, build_steps, next_step, start_build
from mnemiq.superset.chart_check import ChartStatus
from mnemiq.superset.charts import ChartSpec, Family, SavedRef, WrittenRef, extension_chart_request
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetRefusedCaller
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

TENANT = "tenant-a"
PRINCIPAL = "superset:builder"
GOAL = "Premium and claims by region and month"
SETTINGS = Settings(guard_fanout=False, superset_currency="USD")

TOTAL = SavedRef("total_premium", MeasureKind.CURRENCY)
POLICIES = Measure(name="policies", label="Policies", expression="COUNT(policy_id)", kind=MeasureKind.COUNT,
                   dataset_id=POLICY_DATASET, description="Policies written", asked_by="")
CLAIMED = Measure(name="claimed", label="Claimed", expression="SUM(amount)", kind=MeasureKind.CURRENCY, dataset_id=-1,
                  description="Amount claimed", asked_by="", currency="USD")
CLAIM_DATASET = PlannedDataset(key="dataset_1", placeholder=-1, table="claim", table_name="claim")


def _chart(key: str, family: Family, title: str, dataset_id: int, measures: tuple, *, dimensions: tuple = (),
           time_column: str | None = None, time_grain: str | None = None) -> ChartSpec:
    return ChartSpec(key=key, family=family, title=title, dataset_id=dataset_id, measures=measures,
                     dimensions=dimensions, time_column=time_column, time_grain=time_grain, filters=(),
                     row_limit=None, source=None, target=None)


KPI_TOTAL = _chart("total_premium_kpi", Family.BIG_NUMBER, "Total premium", POLICY_DATASET, (TOTAL,))
KPI_POLICIES = _chart("policies_kpi", Family.BIG_NUMBER, "Policies", POLICY_DATASET, (WrittenRef(POLICIES),))
BY_MONTH = _chart("premium_by_month", Family.LINE, "Premium by month", POLICY_DATASET, (TOTAL,), time_column="issued",
                  time_grain="P1M")
BY_REGION = _chart("policies_by_region", Family.BAR, "Policies by region", POLICY_DATASET, (WrittenRef(POLICIES),),
                   dimensions=("region",))
BY_CHANNEL = _chart("premium_by_channel", Family.PIE, "Premium by channel", POLICY_DATASET, (TOTAL,),
                    dimensions=("channel",))
CLAIMS_TABLE = _chart("claimed_by_region", Family.TABLE, "Claimed by region", -1, (WrittenRef(CLAIMED),),
                      dimensions=("claim_region",))
FUNNEL = _chart("policies_by_status", Family.FUNNEL, "Policies by status", POLICY_DATASET, (WrittenRef(POLICIES),),
                dimensions=("status",))

# Six charts across five families, the planted saved metric reused on three, a dataset the build makes for the table.
SIX = BuildPlan(charts=(BY_MONTH, KPI_TOTAL, BY_REGION, KPI_POLICIES, BY_CHANNEL, CLAIMS_TABLE),
                measures=(POLICIES, CLAIMED), datasets=(CLAIM_DATASET,))
ONE_BAR = BuildPlan(charts=(BY_REGION,), measures=(POLICIES,))
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
    """A hook that acts the first time only."""
    done = []

    def run(*args):
        if not done:
            done.append(True)
            hook(*args)

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
    # The table's currency, which MCP cannot send, is patched on after its claim.
    [patch] = [call.request for call in world.calls("patch_chart")]
    assert patch["chart_id"] == ids["claimed_by_region"]
    assert patch["set"] == {"column_config": {"Claimed": {"d3NumberFormat": ",.2f", "currencyFormat": {
        "symbol": "USD", "symbolPosition": "prefix"}}}}
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}


def test_the_made_dataset_is_the_one_its_charts_and_measures_use(fake, world, store):
    """`with_made_datasets` before every chart request and every measure kept: the table is generated, checked and kept
    on the dataset its step made, each measure is kept for the dashboard with its charts, the datasets made for it and
    the builder who asked."""
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]
    made = _step(final, "dataset_1").object_id
    assert made == dataset_id_of(world, "claim") and world.datasets[made]["sql"] is None
    ids = _chart_ids(final)
    [table_request] = [c.request for c in world.calls("generate_chart") if c.request["chart_name"] == "Claimed by region"]
    assert table_request["dataset_id"] == made
    assert world.charts[ids["claimed_by_region"]]["datasource_id"] == made
    checked = [c.request["identifier"] for c in world.calls("get_dataset_info")]
    assert made in checked and all(isinstance(i, int) and i > 0 for i in checked)
    kept = {m.measure.name: m for m in store.measures_for(TENANT, final.dashboard_id)}
    assert set(kept) == {"policies", "claimed"}
    assert kept["claimed"].measure == dataclasses.replace(CLAIMED, dataset_id=made, asked_by=PRINCIPAL)
    assert kept["claimed"].chart_ids == (ids["claimed_by_region"],) and kept["claimed"].made_dataset_ids == (made,)
    assert kept["policies"].measure == dataclasses.replace(POLICIES, asked_by=PRINCIPAL)
    assert sorted(kept["policies"].chart_ids) == sorted((ids["policies_kpi"], ids["policies_by_region"]))
    assert kept["policies"].made_dataset_ids == ()
    assert kept["policies"].check_status == {ids["policies_kpi"]: ChartStatus.CHECKED,
                                             ids["policies_by_region"]: ChartStatus.CHECKED}
    # A dataset the build makes carries the build's marker, set by its claim, and the dashboard's editors.
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
    """`claim_chart` is called once per chart MCP made, and `create_chart` once per funnel, each after the dashboard
    shell exists and for that dashboard; every chart then has the dashboard's editors and the build's marker."""
    plan = BuildPlan(charts=(*SIX.charts, FUNNEL), measures=SIX.measures, datasets=SIX.datasets)
    final = _run(store, _start(store, plan).build_id, _adapter(fake))[-1]
    assert final.status == "done", final
    tools = _tools(fake)
    shell = tools.index("create_dashboard")
    claims = [i for i, tool in enumerate(tools) if tool == "claim_chart"]
    created = [i for i, tool in enumerate(tools) if tool == "create_chart"]
    assert len(claims) == len(SIX.charts) and len(created) == 1 and min(claims + created) > shell
    ids = _chart_ids(final, plan)
    claimed = sorted(c.request["chart_id"] for c in world.calls("claim_chart"))
    assert claimed == sorted(ids[c.key] for c in SIX.charts)
    assert {c.request["dashboard_id"] for c in world.calls("claim_chart") + world.calls("create_chart")} == {
        final.dashboard_id}
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
    """A funnel is never sent to MCP's `generate_chart`: the extension's `create_chart` makes it from mnemiq's own
    request, for the dashboard, with the marker written at create, so no claim follows."""
    final = _run(store, _start(store, ONE_FUNNEL).build_id, _adapter(fake))[-1]
    assert "generate_chart" not in _tools(fake) and "claim_chart" not in _tools(fake)
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


def test_no_dashboard_is_published(fake, world, store):
    """Nothing a build sends asks for a published dashboard, and the dashboard it made stays unpublished."""
    final = _run(store, _start(store).build_id, _adapter(fake))[-1]

    def published(doc) -> bool:
        if isinstance(doc, dict):
            return doc.get("published") is True or any(published(v) for v in doc.values())
        return isinstance(doc, list) and any(published(v) for v in doc)

    assert not any(published(call.request) for call in fake.calls)
    assert world.dashboards[final.dashboard_id]["published"] is False


def test_an_existing_dashboard_is_read_then_added_to(fake, world, store):
    """For an existing dashboard, the first step reads its layout and `changed_on` and holds nothing; the charts are
    added to the layout it has, which is kept, tabs and all."""
    existing = world.plant_chart(title="By hand", dataset_id=POLICY_DATASET, viz_type="table")
    dashboard = 900
    world.dashboards[dashboard] = {
        "id": dashboard, "dashboard_title": "Sales", "published": True, "description": "Made by hand",
        "position_json": json.dumps({
            "DASHBOARD_VERSION_KEY": "v2", "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
            "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": ["ROW-x"], "parents": ["ROOT_ID"]},
            "ROW-x": {"type": "ROW", "id": "ROW-x", "children": ["CHART-x"], "parents": ["ROOT_ID", "GRID_ID"],
                      "meta": {}},
            "CHART-x": {"type": "CHART", "id": "CHART-x", "children": [], "parents": ["ROOT_ID", "GRID_ID", "ROW-x"],
                        "meta": {"chartId": existing, "width": 12, "height": 50}}}),
        "json_metadata": "{}", "editors": ["builder"], "members": [existing], "created_by": "analyst",
        "created_on": world.superset_now(), "changed_on": world.superset_now()}
    state = _start(store, ONE_BAR, dashboard_id=dashboard)
    first = _next(store, state.build_id, _adapter(fake))
    assert first.ran == "dashboard" and _step(first, "dashboard").object_id == dashboard
    assert [c.tool.rsplit(".", 1)[-1] for c in fake.calls] == ["get_layout"]
    final = _run(store, state.build_id, _adapter(fake))[-1]
    assert final.status == "done" and final.dashboard_id == dashboard
    assert "create_dashboard" not in _tools(fake)
    positions = json.loads(world.dashboards[dashboard]["position_json"])
    assert positions["GRID_ID"]["children"][0] == "ROW-x"
    chart = _step(final, "policies_by_region").object_id
    assert sorted(world.dashboards[dashboard]["members"]) == sorted([existing, chart])
    assert world.dashboards[dashboard]["published"] is True  # the build changes no dashboard's publication


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
    assert [c.authorization for c in fake.calls] == [f"Bearer {TOKEN}"]
    assert len(world.dashboards) == 1


# The most any one call may take (`TRANSPORT_MARGIN_S` past the 300 s Superset takes as a statement's limit): two such
# calls in one step run past the 10-minute lease, which only a renewal before the second keeps.
SLOWEST_CALL_S = 315


def test_lease_is_renewed_between_calls(fake, world, store, clock):
    """A step whose two calls take 10.5 minutes in all, past its 10-minute lease, keeps its claim: the lease is
    renewed before the second call, so a second `/next` meanwhile is refused and reconciles nothing."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    meanwhile: list = []

    def slow_create(request, answer, _world):
        clock.advance(SLOWEST_CALL_S)

    def slow_claim(request, _world):
        clock.advance(SLOWEST_CALL_S - 15)  # past the first claim's lease, within the renewed one
        try:
            _next(store, build, _adapter(fake, FRESH_TOKEN))
        except BuildBusy as busy:
            meanwhile.append(busy)

    world.hooks["generate_chart"] = Hook(after=_once(slow_create))
    world.hooks["claim_chart"] = Hook(before=_once(slow_claim))
    state = _next(store, build, _adapter(fake))
    assert len(meanwhile) == 1 and meanwhile[0].key == "policies_by_region"
    assert _step(state, "policies_by_region").status == "done"
    assert "list_charts" not in _tools(fake) and len(world.calls("generate_chart")) == 1
    assert not any(c.authorization == f"Bearer {FRESH_TOKEN}" for c in fake.calls)


def test_worker_past_its_lease_does_not_start_another_call(fake, world, store, clock):
    """A worker whose lease passed during a call starts no further call: the claim is left undone, and the next
    request reconciles the step, finds the chart by its window, and claims it once."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["generate_chart"] = Hook(after=_once(lambda request, answer, _w: clock.advance(601)))
    stalled = _next(store, build, _adapter(fake))
    assert "claim_chart" not in _tools(fake)
    assert _step(stalled, "policies_by_region").status == "uncertain"
    reconciled = _next(store, build, _adapter(fake))
    assert reconciled.ran == "policies_by_region" and _step(reconciled, "policies_by_region").status == "done"
    [made] = [c for c in world.charts.values() if c["slice_name"] == "Policies by region"]
    assert [c.request["chart_id"] for c in world.calls("claim_chart")] == [made["id"]]
    assert len(world.calls("generate_chart")) == 1


# --- reconciling a step whose outcome is uncertain


def test_timed_out_create_is_reconciled_not_repeated(fake, world, store):
    """Superset made the chart and the call timed out before it answered: the step is `uncertain`, and the next
    request finds the one unmarked chart made in the step's window and claims it. No second chart is made."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake, timeout_s=1), "policies_by_region")
    world.hooks["generate_chart"] = Hook(after=_once(lambda request, answer, _w: time.sleep(2)))
    timed_out = _next(store, build, _adapter(fake, timeout_s=1))
    assert _step(timed_out, "policies_by_region").status == "uncertain", timed_out
    assert timed_out.status == "building" and "claim_chart" not in _tools(fake)
    final = _run(store, build, _adapter(fake))[-1]
    assert final.status == "done", final
    [made] = [c for c in world.charts.values() if c["slice_name"] == "Policies by region"]
    assert len(world.calls("generate_chart")) == 1
    assert [c.request["chart_id"] for c in world.calls("claim_chart")] == [made["id"]]
    assert _step(final, "policies_by_region").object_id == made["id"]
    assert parse_marker(made["description"]) == Marker(build, "policies_by_region", final.dashboard_id)


class _Died(BaseException):
    """The worker's process ending: nothing in the builder catches it."""


def test_crash_after_commit_is_reconciled(fake, world, store, clock, monkeypatch):
    """Superset commits `generate_chart`, and the worker dies before it claims the chart or finishes the step. Once
    the lease passes, the next request claims the one unmarked candidate and makes no second chart."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    committed: list = []
    world.hooks["generate_chart"] = Hook(after=lambda request, answer, _w: committed.append(answer["chart"]["id"]))
    renew = store.renew_lease

    def dies_once_the_chart_is_made(*args, **kwargs):
        if committed and not getattr(dies_once_the_chart_is_made, "died", False):
            dies_once_the_chart_is_made.died = True
            raise _Died()
        return renew(*args, **kwargs)

    monkeypatch.setattr(store, "renew_lease", dies_once_the_chart_is_made)
    with pytest.raises(_Died):
        _next(store, build, _adapter(fake))
    assert committed and "claim_chart" not in _tools(fake)
    assert _raw_step(store, build, "policies_by_region")["status"] == "pending"  # nothing was finished
    with pytest.raises(BuildBusy):
        _next(store, build, _adapter(fake))  # the dead worker's lease still holds
    clock.advance(601)
    reconciled = _next(store, build, _adapter(fake))
    assert reconciled.ran == "policies_by_region" and _step(reconciled, "policies_by_region").status == "done"
    assert len(world.calls("generate_chart")) == 1
    assert [c.request["chart_id"] for c in world.calls("claim_chart")] == committed


def _times_out_making_nothing(request, _world):
    time.sleep(2)
    raise RuntimeError("Superset gave up before it saved anything")


def _uncertain_with_nothing_made(fake, world, store, plan: BuildPlan = ONE_BAR) -> str:
    """A build whose chart step timed out before Superset made anything: the step is `uncertain`."""
    build = _start(store, plan).build_id
    _run_to(store, build, _adapter(fake, timeout_s=1), "policies_by_region")
    world.hooks["generate_chart"] = Hook(before=_once(_times_out_making_nothing))
    charts = set(world.charts)
    state = _next(store, build, _adapter(fake, timeout_s=1))
    assert _step(state, "policies_by_region").status == "uncertain", state
    assert set(world.charts) == charts
    return build


@pytest.mark.parametrize("whose", ["another build", "another step of this build", "another dashboard"])
def test_another_builds_chart_is_never_adopted(fake, world, store, whose):
    """A chart of the planned name, on the planned dataset, made by the builder in the step's window, carrying a
    marker that is not this step's (compared field by field): never adopted, claimed or patched. The create is
    repeated, and the build's own chart is the one claimed."""
    build = _uncertain_with_nothing_made(fake, world, store)
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
    assert len(world.calls("generate_chart")) == 2
    touched = [c.request for c in fake.calls if c.request.get("chart_id") == theirs]
    assert touched == []
    assert [c.request["chart_id"] for c in world.calls("claim_chart")] == [ours]


def test_two_unmarked_candidates_stop_the_build(fake, world, store):
    """Superset made the chart and timed out, and another unmarked chart of the same name was made in the window: the
    build cannot tell which is its own, so it stops and names both, and claims neither."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake, timeout_s=1), "policies_by_region")

    def twin(request, answer, _world):
        world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET, viz_type="echarts_timeseries_bar")
        time.sleep(2)

    world.hooks["generate_chart"] = Hook(after=_once(twin))
    _next(store, build, _adapter(fake, timeout_s=1))
    stopped = _next(store, build, _adapter(fake))
    assert stopped.status == "failed" and _step(stopped, "policies_by_region").status == "failed"
    named = sorted(c["id"] for c in world.charts.values() if c["slice_name"] == "Policies by region")
    assert len(named) == 2 and all(str(chart_id) in stopped.message for chart_id in named), stopped.message
    assert "claim_chart" not in _tools(fake) and len(world.calls("generate_chart")) == 1
    again = _next(store, build, _adapter(fake))
    assert again.status == "failed" and again.ran is None


def test_a_chart_older_than_the_window_is_not_a_candidate(fake, world, store):
    """An unmarked chart of the planned name made before the step began (by hand, earlier) is not the step's: the
    create is repeated, and that chart is left alone."""
    old = world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET, viz_type="echarts_timeseries_bar",
                            created_on=world.superset_now() - timedelta(hours=3))
    build = _uncertain_with_nothing_made(fake, world, store)
    final = _run(store, build, _adapter(fake))[-1]
    assert final.status == "done" and _step(final, "policies_by_region").object_id != old
    assert world.charts[old]["description"] is None and len(world.calls("generate_chart")) == 2


@pytest.mark.parametrize(("before_start_s", "claimed"), [(120, True), (121, False)],
                         ids=["at the margin", "past the margin"])
def test_the_window_opens_the_margin_before_the_step_began(fake, world, store, before_start_s, claimed):
    """`created_on` is Superset's clock and `started_at` the control Postgres's, so the window opens
    `RECONCILE_MARGIN_S` (120 s) before the step began: a candidate made 120 s before is the step's, one made 121 s
    before is not, and the create is repeated."""
    assert builder.RECONCILE_MARGIN_S == 120
    build = _uncertain_with_nothing_made(fake, world, store)
    started = datetime.fromisoformat(_raw_step(store, build, "policies_by_region")["started_at"])
    made = started.astimezone(world.zone).replace(tzinfo=None) - timedelta(seconds=before_start_s)
    candidate = world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET,
                                  viz_type="echarts_timeseries_bar", created_on=made)
    final = _run(store, build, _adapter(fake))[-1]
    assert final.status == "done", final
    assert (_step(final, "policies_by_region").object_id == candidate) is claimed
    assert len(world.calls("generate_chart")) == (1 if claimed else 2)


@pytest.mark.parametrize(("setting", "claimed"), [("America/Los_Angeles", True), ("UTC", False)])
def test_supersets_wall_clock_is_read_in_its_zone(fake, clock, store, setting, claimed):
    """Superset writes `created_on` as its app server's wall clock, with no zone: read in `superset_timezone`, a
    chart made in the window by a Superset in Los Angeles is the step's; read as UTC, the same text is eight hours
    earlier than the step, so it is not."""
    world = World(fake, clock, zone=ZoneInfo("America/Los_Angeles"))
    settings = Settings(guard_fanout=False, superset_currency="USD", superset_timezone=setting)
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    world.hooks["generate_chart"] = Hook(before=_once(_times_out_making_nothing))
    assert _step(_next(store, build, _adapter(fake, timeout_s=1), settings), "policies_by_region").status == "uncertain"
    candidate = world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET,
                                  viz_type="echarts_timeseries_bar")
    final = _run(store, build, _adapter(fake), settings)[-1]
    assert (_step(final, "policies_by_region").object_id == candidate) is claimed


def test_a_chart_on_another_dataset_is_not_a_candidate(fake, world, store):
    """An unmarked chart of the planned name made in the window, but on another dataset, is not the step's."""
    build = _uncertain_with_nothing_made(fake, world, store)
    other = world.plant_chart(title="Policies by region", dataset_id=POLICY_DATASET,
                              viz_type="echarts_timeseries_bar")
    world.charts[other]["datasource_name"] = "public.somewhere_else"
    final = _run(store, build, _adapter(fake))[-1]
    assert _step(final, "policies_by_region").object_id != other and len(world.calls("generate_chart")) == 2


def test_a_dataset_whose_create_timed_out_is_reconciled(fake, world, store):
    """A physical dataset is made by `create_dataset` and marked by its claim: made and timed out, it is found in the
    window by its table name and claimed; no second is made."""
    build = _start(store).build_id
    _run_to(store, build, _adapter(fake, timeout_s=1), "dataset_1")
    world.hooks["create_dataset"] = Hook(after=_once(lambda request, answer, _w: time.sleep(2)))
    assert _step(_next(store, build, _adapter(fake, timeout_s=1)), "dataset_1").status == "uncertain"
    state = _next(store, build, _adapter(fake))
    assert state.ran == "dataset_1" and _step(state, "dataset_1").status == "done"
    assert len(world.calls("create_dataset")) == 1
    made = dataset_id_of(world, "claim")
    assert [c.request["dataset_id"] for c in world.calls("claim_dataset")] == [made]
    assert _step(state, "dataset_1").object_id == made


def test_a_dashboard_shell_whose_create_timed_out_is_adopted_by_its_marker(fake, world, store):
    """The shell carries its marker from its create: made and timed out, it is found by that marker and adopted."""
    build = _start(store, ONE_BAR).build_id
    world.hooks["create_dashboard"] = Hook(after=_once(lambda request, answer, _w: time.sleep(2)))
    assert _step(_next(store, build, _adapter(fake, timeout_s=1)), "dashboard").status == "uncertain"
    state = _next(store, build, _adapter(fake))
    assert _step(state, "dashboard").status == "done" and len(world.dashboards) == 1
    [shell] = world.dashboards.values()
    assert state.dashboard_id == shell["id"] and len(world.calls("create_dashboard")) == 1


# --- a token that expires between steps


def _refused_once_the_chart_is_made(fake, world, store, monkeypatch, when: str) -> None:
    """Superset stops accepting the builder's token once the chart is made: `during` the create (its own answer is
    lost, as the client's last request of the call is refused), or `between` the create and the claim."""
    if when == "during":
        world.hooks["generate_chart"] = Hook(after=_once(lambda request, answer, _w: fake.refuse_token(TOKEN)))
        return
    renew = store.renew_lease

    def refuse_once_made(*args, **kwargs):
        if world.calls("generate_chart"):
            fake.refuse_token(TOKEN)
        return renew(*args, **kwargs)

    monkeypatch.setattr(store, "renew_lease", refuse_once_made)


@pytest.mark.parametrize("when", ["during", "between"])
def test_expired_token_between_steps_resumes(fake, world, store, monkeypatch, when):
    """Superset refuses the token after the chart was made, during its create or before its claim: the request fails
    as a refusal of the caller (sign in again), the build stays at its last completed step with this one uncertain,
    and the next request, with a fresh token, claims the chart that was made, never making a second."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    _refused_once_the_chart_is_made(fake, world, store, monkeypatch, when)
    with pytest.raises(SupersetRefusedCaller):
        _next(store, build, _adapter(fake))
    kept = build_state(store, store.load_build(TENANT, build, PRINCIPAL))
    assert [step.status for step in kept.steps] == ["done", "uncertain", "pending", "pending"]
    assert kept.status == "building"
    [made] = [c for c in world.charts.values() if c["slice_name"] == "Policies by region"]
    # Between the calls, the step knew what it made and kept it; during the create, it did not.
    assert _step(kept, "policies_by_region").object_id == (made["id"] if when == "between" else None)
    final = _run(store, build, _adapter(fake, FRESH_TOKEN))[-1]
    assert final.status == "done", final
    assert len(world.calls("generate_chart")) == 1
    [claim] = world.calls("claim_chart")
    assert claim.request["chart_id"] == made["id"] and claim.authorization == f"Bearer {FRESH_TOKEN}"


def test_an_object_that_cannot_be_read_now_is_not_made_again(fake, world, store, monkeypatch):
    """The chart this step made cannot be read back for a moment (Superset answers an error that is not "not found"):
    the step stays uncertain and nothing is made; once it reads, the chart is claimed. Only "not found" means gone."""
    build = _start(store, ONE_BAR).build_id
    _run_to(store, build, _adapter(fake), "policies_by_region")
    _refused_once_the_chart_is_made(fake, world, store, monkeypatch, "between")
    with pytest.raises(SupersetRefusedCaller):
        _next(store, build, _adapter(fake))
    failing = {"error": "the metadata database is restarting", "error_type": "InternalError"}
    world.hooks["get_chart_info"] = Hook(instead=_once_answer(failing))
    held = _next(store, build, _adapter(fake, FRESH_TOKEN))
    assert held.ran == "policies_by_region" and _step(held, "policies_by_region").status == "uncertain"
    assert "restarting" not in held.message and len(world.calls("generate_chart")) == 1
    final = _run(store, build, _adapter(fake, FRESH_TOKEN))[-1]
    assert final.status == "done" and len(world.calls("generate_chart")) == 1


def _once_answer(answer: dict):
    """An `instead` hook answering `answer` the first time, then letting the tool run."""
    sent = []

    def instead(request, _world):
        if not sent:
            sent.append(True)
            return answer
        return None

    return instead


# --- what a build's state carries, and what it keeps


def test_a_build_state_never_carries_a_lease_id(fake, world, store):
    """The build routes answer with a `BuildState`: no lease id is in one, while a step runs or after."""
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

    def drops(tenant, build_id, key, status, object_id, *, lease_id):
        if key == "dashboard" and not dropped:
            dropped.append("after")
            finish(tenant, build_id, key, status, object_id, lease_id=lease_id)
            raise psycopg.OperationalError("the connection dropped after the commit")
        if key == "policies_by_region" and len(dropped) == 1:
            dropped.append("before")
            raise psycopg.OperationalError("the connection dropped before the commit")
        return finish(tenant, build_id, key, status, object_id, lease_id=lease_id)

    monkeypatch.setattr(store, "finish_step", drops)
    final = _run(store, _start(store, ONE_BAR).build_id, _adapter(fake))[-1]
    assert dropped == ["after", "before"]
    assert final.status == "done" and len(world.dashboards) == 1 and len(world.calls("generate_chart")) == 1


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


def test_a_virtual_dataset_is_named_for_its_build(fake, world, store):
    """Two builds that join the same tables make two datasets: each is named for its build, so neither collides with
    the other's name, and each carries its marker after a line saying mnemiq made it for the dashboard."""
    joined = PlannedDataset(key="dataset_1", placeholder=-1,
                            join="policy JOIN claim ON policy.region = claim.claim_region",
                            sql='SELECT policy."premium" AS "premium", claim."amount" AS "amount" FROM policy JOIN '
                                "claim ON policy.region = claim.claim_region", name="mnemiq_policy_claim")
    ratio = Measure(name="claimed_share", label="Claimed share", expression="SUM(amount)", kind=MeasureKind.NUMBER,
                    dataset_id=-1, description="", asked_by="")
    chart = _chart("claimed_share_kpi", Family.BIG_NUMBER, "Claimed share", -1, (WrittenRef(ratio),))
    plan = BuildPlan(charts=(chart,), measures=(ratio,), datasets=(joined,))
    names = []
    for _ in range(2):
        state = _start(store, plan)
        for _ in range(2):
            state = _next(store, state.build_id, _adapter(fake))
        made = world.datasets[_step(state, "dataset_1").object_id]
        assert made["table_name"].startswith("mnemiq_policy_claim_") and state.build_id[:8] in made["table_name"]
        lines = made["description"].splitlines()
        assert lines[-1] == format_marker(state.build_id, "dataset_1", state.dashboard_id)
        assert str(state.dashboard_id) in lines[0] and "mnemiq" in lines[0]
        names.append(made["table_name"])
    assert len(set(names)) == 2


# --- The same build on the Postgres store: the builder joined to the store it runs on in a deployment, with
# Superset's clock the real one, so a step's window compares Postgres's `now()` with it as a deployment does. Gated
# as the store's own tests are (`MNEMIQ_TEST_BUILD_STORE_DSN`, and `MNEMIQ_TEST_BUILD_STORE_REQUIRED` in the job that
# gives them a server).


class _RealClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


@pytest.fixture
def pg_store():
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from mnemiq.superset.build_store import BuildStore
    from tests.test_superset_build_store import _database_dsn

    database = _database_dsn()
    schema = f"builder_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    store = BuildStore(make_conninfo(database, options=f"-c search_path={schema}"))
    try:
        yield store
    finally:
        store.close()
        with psycopg.connect(database, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


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
    assert set(kept) == {"policies", "claimed"} and kept["claimed"].measure.asked_by == PRINCIPAL
    # A later read has each status kept with a written measure's row: a chart that plots only a reused saved metric
    # has no row, so its status rides in the check step's answer alone.
    ids = _chart_ids(final)
    written = {ids[c.key] for c in SIX.charts if any(isinstance(ref, WrittenRef) for ref in c.measures)}
    reread = build_state(pg_store, pg_store.load_build(TENANT, final.build_id, PRINCIPAL))
    assert reread.checks == {chart_id: ChartStatus.CHECKED for chart_id in written} and reread.ran is None
    again = _start(pg_store, ONE_BAR, dashboard_id=final.dashboard_id)
    with pytest.raises(DashboardBusy):
        _start(pg_store, ONE_BAR, dashboard_id=final.dashboard_id)
    assert again.status == "building" and world.dashboards[final.dashboard_id]["published"] is False


def test_a_timed_out_create_is_reconciled_on_the_postgres_store(fake, pg_store):
    """A chart made and timed out, on the Postgres store: the step is uncertain with Postgres's `started_at`, and the
    next request finds the chart in the window by Superset's own clock, and claims it; no second is made."""
    world = World(fake, _RealClock())
    build = _start(pg_store, ONE_BAR).build_id
    _run_to(pg_store, build, _adapter(fake), "policies_by_region")
    world.hooks["generate_chart"] = Hook(after=_once(lambda request, answer, _w: time.sleep(2)))
    assert _step(_next(pg_store, build, _adapter(fake, timeout_s=1)), "policies_by_region").status == "uncertain"
    final = _run(pg_store, build, _adapter(fake))[-1]
    assert final.status == "done", final
    assert len(world.calls("generate_chart")) == 1 and len(world.calls("claim_chart")) == 1


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

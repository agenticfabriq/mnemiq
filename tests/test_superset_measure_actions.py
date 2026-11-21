"""A dashboard's measures, reviewed and changed by an editor of the dashboard: listed with each value computed now as
the person asking and kept nowhere, edited on every copy that still matches, removed with their charts after the
person saw which, and saved to their dataset. Every change asks first whether the person edits each chart (and, to
save, the dataset), so one refusal changes nothing; a chart changed by hand, or one whose settings cannot be read
whole, is named and left as it is.

Every test builds its dashboard with the real builder against `FakeSuperset` and `World` (`tests/superset_build_world.py`),
then acts on it as one person or another through the real client and adapter.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from mnemiq.config import Settings
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.builder import next_step, start_build
from mnemiq.superset.chart_check import ChartStatus
from mnemiq.superset.charts import Family, WrittenRef
from mnemiq.superset.client import DelegatedToken, SupersetMcp
from mnemiq.superset.measure_actions import (
    VIRTUAL_NO_VALUE,
    ActionRefused,
    Asking,
    ChartsChanged,
    EditorsOf,
    edit_measure,
    list_measures,
    remove_measure,
    save_measure,
)
from mnemiq.superset.measures import MeasureKind
from mnemiq.superset.planner import BuildPlan, PlannedDataset
from tests.superset_build_world import (
    DATABASE_ID,
    GRANTS,
    POLICY_DATASET,
    SCHEMA,
    SNAPSHOT,
    TOKEN,
    Clock,
    Hook,
    MemoryBuildStore,
    World,
)
from tests.superset_fake import FakeSuperset
from tests.superset_shared import BY_REGION, POLICIES, _chart

TENANT = "tenant-a"
PRINCIPAL = "superset:builder"
SETTINGS = Settings(guard_fanout=False, superset_currency="USD")

BY_STATUS = _chart("policies_by_status", Family.BAR, "Policies by status", POLICY_DATASET, (WrittenRef(POLICIES),),
                   dimensions=("status",))
BY_CHANNEL = _chart("policies_by_channel", Family.PIE, "Policies by channel", POLICY_DATASET, (WrittenRef(POLICIES),),
                    dimensions=("channel",))
THREE_OF_POLICIES = BuildPlan(charts=(BY_REGION, BY_STATUS, BY_CHANNEL), measures=(POLICIES,))

CLAIMED = dataclasses.replace(POLICIES, name="claimed", label="Claimed", expression="SUM(amount)",
                              kind=MeasureKind.CURRENCY, dataset_id=-1, currency="USD")
CLAIM_DATASET = PlannedDataset(key="dataset_1", placeholder=-1, table="claim", table_name="claim")
CLAIMED_KPI = _chart("claimed_kpi", Family.BIG_NUMBER, "Claimed", -1, (WrittenRef(CLAIMED),))
CLAIMED_BY_REGION = _chart("claimed_by_region", Family.TABLE, "Claimed by region", -1, (WrittenRef(CLAIMED),),
                           dimensions=("claim_region",))
CLAIMS = BuildPlan(charts=(CLAIMED_KPI, CLAIMED_BY_REGION), measures=(CLAIMED,), datasets=(CLAIM_DATASET,))


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


def _adapter(fake, token: str = TOKEN) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken(token), timeout_s=10.0), database_id=DATABASE_ID,
                              schema=SCHEMA)


def _built(fake, store, plan: BuildPlan) -> dict:
    """The plan built to the end as the builder: each planned chart's id by its key, and the dashboard's id."""
    state = start_build(store, tenant=TENANT, principal=PRINCIPAL, goal="a goal", plan=plan)
    for _ in range(20):
        adapter = _adapter(fake)
        build = store.load_build(TENANT, state.build_id, PRINCIPAL)
        state = next_step(build, client=adapter.client, adapter=adapter, store=store, snapshot=SNAPSHOT,
                          grants=GRANTS, settings=SETTINGS)
        if state.status != "building":
            break
    assert state.status == "done", state
    ids = {step.key: step.object_id for step in state.steps if step.kind in ("chart", "dataset")}
    return {**ids, "dashboard": state.dashboard_id}


def _asking(fake, store, dashboard_id: int, token: str = TOKEN) -> Asking:
    return Asking(adapter=_adapter(fake, token), store=store, tenant=TENANT, principal=PRINCIPAL,
                  dashboard_id=dashboard_id, snapshot=SNAPSHOT, grants=GRANTS, settings=SETTINGS)


def _kept(store, dashboard_id: int) -> dict:
    return {row.measure.name: row for row in store.measures_for(TENANT, dashboard_id)}


def _metric(world, chart_id: int):
    params = world.charts[chart_id]["params"]
    return params["metric"] if "metric" in params and not params.get("metrics") else params["metrics"][0]


def _placed(world, dashboard_id: int) -> set[int]:
    layout = json.loads(world.dashboards[dashboard_id]["position_json"])
    return {node["meta"]["chartId"] for node in layout.values() if isinstance(node, dict)
            and node.get("type") == "CHART"}


def _tools(fake, since: int = 0) -> list[str]:
    return [call.tool.rsplit(".", 1)[-1] for call in fake.calls[since:]]


# --- the list


def test_values_are_computed_per_caller_never_stored(fake, world, store):
    """Two people with different row rules list the same dashboard's measure and get two values, each as Superset
    lets that person read; the store holds neither, before or after."""
    built = _built(fake, store, THREE_OF_POLICIES)
    world.user_rules = {"west": "region = 'west'", "east": "region = 'east'"}
    before = store.measures_for(TENANT, built["dashboard"])
    [west] = list_measures(_asking(fake, store, built["dashboard"], "tok-west"))
    [east] = list_measures(_asking(fake, store, built["dashboard"], "tok-east"))
    assert (west.value, east.value) == (2.0, 3.0), (west, east)
    assert west.measure == east.measure and west.value_note is None
    assert store.measures_for(TENANT, built["dashboard"]) == before
    assert "2.0" not in json.dumps([dataclasses.asdict(row) for row in before], default=str)
    [shown] = list_measures(_asking(fake, store, built["dashboard"]))
    assert [(chart.chart_id, chart.title, chart.check) for chart in shown.charts] == [
        (built["policies_by_region"], "Policies by region", ChartStatus.CHECKED),
        (built["policies_by_status"], "Policies by status", ChartStatus.CHECKED),
        (built["policies_by_channel"], "Policies by channel", ChartStatus.CHECKED)]


def test_virtual_measure_value_comes_through_the_chart(fake, world, store):
    """On a virtual dataset, the value is what Superset's chart of it shows as one number, read as the person, and
    mnemiq runs no statement of its own for it; with no such chart there is no value, and the note says why."""
    built = _built(fake, store, CLAIMS)
    made = built["dataset_1"]
    world.datasets[made]["sql"] = 'SELECT * FROM "claim"'
    since = len(fake.calls)
    [claimed] = list_measures(_asking(fake, store, built["dashboard"]))
    assert claimed.value == 1125.0 and claimed.value_note is None, claimed
    assert "execute_sql" not in _tools(fake, since) and "get_chart_data" in _tools(fake, since)
    world.charts[built["claimed_kpi"]]["params"]["metric"]["label"] = "Claimed, by hand"
    [claimed] = list_measures(_asking(fake, store, built["dashboard"]))
    assert claimed.value is None and claimed.value_note == VIRTUAL_NO_VALUE


def test_chart_removed_by_hand_drops_out_of_the_list(fake, world, store):
    """A chart someone took off the dashboard is no longer one of the measure's charts in the list."""
    built = _built(fake, store, THREE_OF_POLICIES)
    dashboard = world.dashboards[built["dashboard"]]
    layout = json.loads(dashboard["position_json"])
    gone = built["policies_by_status"]
    dashboard["position_json"] = json.dumps({key: node for key, node in layout.items() if not (
        isinstance(node, dict) and node.get("type") == "CHART" and node["meta"]["chartId"] == gone)})
    [shown] = list_measures(_asking(fake, store, built["dashboard"]))
    assert [chart.chart_id for chart in shown.charts] == [built["policies_by_region"], built["policies_by_channel"]]


# --- edit


def test_edit_leaves_hand_changed_copies(fake, world, store):
    """Edit writes the new definition into every copy that still matches and checks those charts again; a copy
    changed by hand is left as it is and named. The dashboard keeps the new definition."""
    built = _built(fake, store, THREE_OF_POLICIES)
    by_hand = built["policies_by_status"]
    world.charts[by_hand]["params"]["metrics"][0]["sqlExpression"] = "COUNT(*)"
    result = edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="Policies",
                          expression="COUNT(DISTINCT policy_id)", kind=MeasureKind.COUNT, currency=None,
                          description="Distinct policies")
    assert result.changed == (built["policies_by_region"], built["policies_by_channel"])
    assert result.left_by_hand == (by_hand,) and not result.not_editor and not result.could_not_read
    assert _metric(world, built["policies_by_region"])["sqlExpression"] == "COUNT(DISTINCT policy_id)"
    assert _metric(world, built["policies_by_channel"])["sqlExpression"] == "COUNT(DISTINCT policy_id)"
    assert _metric(world, by_hand)["sqlExpression"] == "COUNT(*)"
    kept = _kept(store, built["dashboard"])["policies"]
    assert kept.measure.expression == "COUNT(DISTINCT policy_id)" and kept.measure.description == "Distinct policies"
    assert kept.check_status[built["policies_by_region"]] is ChartStatus.CHECKED
    assert "left as changed by hand" in result.message


def test_non_editor_edit_changes_nothing(fake, world, store):
    """One chart the person does not edit: no copy changes anywhere, and that chart's editors are named."""
    built = _built(fake, store, THREE_OF_POLICIES)
    world.charts[built["policies_by_channel"]]["editors"] = ["someone"]
    before = {chart_id: json.dumps(world.charts[chart_id]["params"], sort_keys=True) for chart_id in world.charts}
    kept = _kept(store, built["dashboard"])["policies"]
    result = edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="Policies",
                          expression="COUNT(DISTINCT policy_id)", kind=MeasureKind.COUNT, currency=None,
                          description="")
    assert result.changed == () and result.not_editor == (EditorsOf(editors=("someone",),
                                                                    chart_id=built["policies_by_channel"],
                                                                    dataset_id=POLICY_DATASET),)
    assert {c: json.dumps(world.charts[c]["params"], sort_keys=True) for c in world.charts} == before
    assert _kept(store, built["dashboard"])["policies"] == kept
    assert "patch_chart" not in _tools(fake)


def test_stubbed_chart_is_named_not_edited(fake, world, store):
    """A chart whose settings Superset sends cut is named and left as it is; the others are edited."""
    built = _built(fake, store, THREE_OF_POLICIES)
    stubbed = built["policies_by_status"]

    def cut(request, answer, _world):
        if request.get("identifier") == stubbed and isinstance(answer.get("form_data"), dict):
            return {**answer, "form_data": {"_truncated": True}}
        return None

    world.hooks["get_chart_info"] = Hook(after=cut)
    result = edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="Policies",
                          expression="COUNT(DISTINCT policy_id)", kind=MeasureKind.COUNT, currency=None,
                          description="")
    assert stubbed in result.could_not_read and stubbed not in result.changed
    assert _metric(world, stubbed)["sqlExpression"] == "COUNT(policy_id)"
    assert set(result.changed) == {built["policies_by_region"], built["policies_by_channel"]}


def test_an_edit_the_deciders_refuse_changes_nothing(fake, world, store):
    """A new definition that does not aggregate is refused before any chart is read for editors or written."""
    built = _built(fake, store, THREE_OF_POLICIES)
    since = len(fake.calls)
    with pytest.raises(ActionRefused, match="refused"):
        edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="Policies", expression="policy_id",
                     kind=MeasureKind.COUNT, currency=None, description="")
    assert "patch_chart" not in _tools(fake) and "editors_of" not in _tools(fake, since)


def test_an_edit_of_the_kind_rewrites_the_charts_format(fake, world, store):
    """A kind changed to currency gives each chart the currency's format and its currency."""
    built = _built(fake, store, THREE_OF_POLICIES)
    edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="Policies", expression="SUM(premium)",
                 kind=MeasureKind.CURRENCY, currency="USD", description="")
    params = world.charts[built["policies_by_region"]]["params"]
    assert params["y_axis_format"] == ",.2f" and params["currency_format"] == {"symbol": "USD",
                                                                                "symbolPosition": "prefix"}
    assert world.charts[built["policies_by_channel"]]["params"]["number_format"] == ",.2f"


# --- remove


def test_remove_deletes_exactly_the_measures_charts(fake, world, store):
    """Remove moves the measure's charts to the trash and off the layout, and forgets the measure; nothing else."""
    built = _built(fake, store, THREE_OF_POLICIES)
    others = set(world.charts) - {built[key] for key in ("policies_by_region", "policies_by_status",
                                                         "policies_by_channel")}
    charts = (built["policies_by_region"], built["policies_by_status"], built["policies_by_channel"])
    result = remove_measure(_asking(fake, store, built["dashboard"]), "policies", chart_ids=charts)
    assert result.changed == charts and not result.not_editor and not result.blocked_by_report
    assert set(world.trashed) == set(charts) and set(world.charts) == others
    assert _placed(world, built["dashboard"]) == set()
    assert "policies" not in _kept(store, built["dashboard"])


def test_remove_deletes_made_datasets_once_unused(fake, world, store):
    """The dataset mnemiq made for the measure goes once its charts are gone."""
    built = _built(fake, store, CLAIMS)
    made = built["dataset_1"]
    remove_measure(_asking(fake, store, built["dashboard"]), "claimed",
                   chart_ids=(built["claimed_kpi"], built["claimed_by_region"]))
    assert made not in world.datasets and POLICY_DATASET in world.datasets


def test_non_editor_remove_takes_the_chart_off_and_names_it(fake, world, store):
    """A chart the person does not edit is not deleted: it is taken off the dashboard, and named with its editors."""
    built = _built(fake, store, THREE_OF_POLICIES)
    theirs = built["policies_by_channel"]
    world.charts[theirs]["editors"] = ["someone"]
    charts = (built["policies_by_region"], built["policies_by_status"], theirs)
    result = remove_measure(_asking(fake, store, built["dashboard"]), "policies", chart_ids=charts)
    assert result.not_editor == (EditorsOf(editors=("someone",), chart_id=theirs, dataset_id=POLICY_DATASET),)
    assert theirs in world.charts and theirs not in world.trashed and theirs not in _placed(world, built["dashboard"])
    assert "someone" in result.message


def test_chart_with_a_report_is_named(fake, world, store):
    """A chart a report holds cannot be deleted: it is named and stays on the dashboard, and the measure keeps it."""
    built = _built(fake, store, THREE_OF_POLICIES)
    held = built["policies_by_status"]
    world.reported.add(held)
    charts = (built["policies_by_region"], held, built["policies_by_channel"])
    result = remove_measure(_asking(fake, store, built["dashboard"]), "policies", chart_ids=charts)
    assert result.blocked_by_report == (held,) and held not in result.changed
    assert _placed(world, built["dashboard"]) == {held}
    assert _kept(store, built["dashboard"])["policies"].chart_ids == (held,)


def test_a_remove_sent_other_charts_than_the_measure_has_removes_nothing(fake, world, store):
    """The ids sent must be the measure's charts now: otherwise nothing is removed and the answer names the charts it
    has."""
    built = _built(fake, store, THREE_OF_POLICIES)
    with pytest.raises(ChartsChanged) as refused:
        remove_measure(_asking(fake, store, built["dashboard"]), "policies", chart_ids=(built["policies_by_region"],))
    assert [chart.chart_id for chart in refused.value.charts] == [
        built["policies_by_region"], built["policies_by_status"], built["policies_by_channel"]]
    assert not world.trashed and "delete_chart" not in _tools(fake)


# --- save


def test_save_switches_matching_copies_and_names_the_one_changed_by_hand(fake, world, store):
    """Saved to its dataset, the measure becomes a saved metric there with its definition and format; every copy
    switches to it, and a copy changed by hand is left and named."""
    built = _built(fake, store, THREE_OF_POLICIES)
    world.datasets[POLICY_DATASET]["editors"].append("builder")
    by_hand = built["policies_by_status"]
    world.charts[by_hand]["params"]["metrics"][0]["sqlExpression"] = "COUNT(*)"
    result = save_measure(_asking(fake, store, built["dashboard"]), "policies")
    [saved] = [m for m in world.datasets[POLICY_DATASET]["metrics"] if m["metric_name"] == "policies"]
    assert saved["expression"] == "COUNT(policy_id)" and saved["verbose_name"] == "Policies"
    assert saved["d3format"] == ",d"
    assert result.changed == (built["policies_by_region"], built["policies_by_channel"])
    assert result.left_by_hand == (by_hand,) and not result.not_editor
    assert _metric(world, built["policies_by_region"]) == "policies"
    assert _metric(world, by_hand)["sqlExpression"] == "COUNT(*)"
    kept = _kept(store, built["dashboard"])["policies"]
    assert kept.saved_metric == "policies"
    assert kept.check_status[built["policies_by_region"]] is ChartStatus.CHECKED


def test_save_by_a_non_editor_of_one_chart_changes_nothing(fake, world, store):
    """Save, like Edit, changes nothing when the person does not edit every chart holding a copy, and names that
    chart's editors: the copies never split between the saved metric and the written one."""
    built = _built(fake, store, THREE_OF_POLICIES)
    world.datasets[POLICY_DATASET]["editors"].append("builder")
    theirs = built["policies_by_channel"]
    world.charts[theirs]["editors"] = ["someone"]
    metrics = [dict(m) for m in world.datasets[POLICY_DATASET]["metrics"]]
    result = save_measure(_asking(fake, store, built["dashboard"]), "policies")
    assert result.changed == () and result.not_editor == (EditorsOf(editors=("someone",), chart_id=theirs,
                                                                    dataset_id=POLICY_DATASET),)
    assert world.datasets[POLICY_DATASET]["metrics"] == metrics
    assert "patch_chart" not in _tools(fake) and "add_saved_metric" not in _tools(fake)
    assert _kept(store, built["dashboard"])["policies"].saved_metric is None


def test_save_retried_after_the_metric_was_saved_carries_on(fake, world, store):
    """A save whose metric Superset already holds with the measure's own definition (an earlier save whose answer
    was lost) switches the copies; one of that name with another definition is refused, and nothing changes."""
    built = _built(fake, store, THREE_OF_POLICIES)
    world.datasets[POLICY_DATASET]["editors"].append("builder")
    world.datasets[POLICY_DATASET]["metrics"].append(
        {"metric_name": "policies", "expression": "COUNT(policy_id)", "verbose_name": "Policies", "d3format": ",d",
         "currency": None, "certified_by": None, "certification_details": None, "warning_text": None,
         "description": None})
    result = save_measure(_asking(fake, store, built["dashboard"]), "policies")
    assert set(result.changed) == {built["policies_by_region"], built["policies_by_status"],
                                   built["policies_by_channel"]}
    assert _kept(store, built["dashboard"])["policies"].saved_metric == "policies"


def test_save_onto_a_metric_of_its_name_with_another_definition_is_refused(fake, world, store):
    built = _built(fake, store, THREE_OF_POLICIES)
    world.datasets[POLICY_DATASET]["editors"].append("builder")
    world.datasets[POLICY_DATASET]["metrics"].append(
        {"metric_name": "policies", "expression": "COUNT(*)", "verbose_name": "Policies", "d3format": None,
         "currency": None, "certified_by": None, "certification_details": None, "warning_text": None,
         "description": None})
    with pytest.raises(ActionRefused, match="another definition"):
        save_measure(_asking(fake, store, built["dashboard"]), "policies")
    assert "patch_chart" not in _tools(fake)


def test_save_by_a_non_editor_changes_nothing(fake, world, store):
    """A person who does not edit the dataset is told its editors, and nothing is saved or switched."""
    built = _built(fake, store, THREE_OF_POLICIES)
    metrics = list(world.datasets[POLICY_DATASET]["metrics"])
    result = save_measure(_asking(fake, store, built["dashboard"]), "policies")
    assert result.not_editor == (EditorsOf(editors=("analyst",), dataset_id=POLICY_DATASET),)
    assert world.datasets[POLICY_DATASET]["metrics"] == metrics and result.changed == ()
    assert "patch_chart" not in _tools(fake) and "add_saved_metric" not in _tools(fake)
    assert _kept(store, built["dashboard"])["policies"].saved_metric is None


def test_the_mcp_2x_workflow_runs_this_module():
    """The actions drive `SupersetMcp` over the wire to `FakeSuperset`, so this module runs where the client meets
    mcp 2.x too: the workflow's one pytest line names it."""
    import re
    from pathlib import Path

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "mcp-2x.yml"
    runs = [line for line in workflow.read_text().splitlines() if re.search(r"\brun:.*\bpytest\b", line)]
    assert len(runs) == 1 and f"tests/{Path(__file__).name}" in runs[0].split()


def test_remove_leaves_a_chart_changed_by_hand_and_names_it(fake, world, store):
    """A chart whose copy someone changed by hand no longer shows the measure: Remove leaves it on the dashboard and
    names it, and deletes the others."""
    built = _built(fake, store, THREE_OF_POLICIES)
    by_hand = built["policies_by_status"]
    world.charts[by_hand]["params"]["metrics"][0]["sqlExpression"] = "COUNT(*)"
    charts = (built["policies_by_region"], by_hand, built["policies_by_channel"])
    result = remove_measure(_asking(fake, store, built["dashboard"]), "policies", chart_ids=charts)
    assert result.left_by_hand == (by_hand,) and by_hand not in result.changed
    assert by_hand in world.charts and _placed(world, built["dashboard"]) == {by_hand}
    assert "policies" not in _kept(store, built["dashboard"])


def test_remove_counts_a_chart_already_in_the_trash_as_gone(fake, world, store):
    """A chart already moved to the trash (an earlier remove whose answer was lost, or someone in Superset) drops out
    of the list; Remove of the rest takes it off the layout too, and forgets the measure."""
    built = _built(fake, store, THREE_OF_POLICIES)
    trashed = built["policies_by_status"]
    world.trashed[trashed] = world.charts.pop(trashed)
    [shown] = list_measures(_asking(fake, store, built["dashboard"]))
    listed = tuple(chart.chart_id for chart in shown.charts)
    assert trashed not in listed
    result = remove_measure(_asking(fake, store, built["dashboard"]), "policies", chart_ids=listed)
    assert set(result.changed) == set(listed)
    assert _placed(world, built["dashboard"]) == set()
    assert "policies" not in _kept(store, built["dashboard"])


def test_an_edit_whose_second_chart_keeps_changing_puts_the_first_back(fake, world, store):
    """The copies never split: when one chart keeps changing under the edit, the charts already changed are put
    back, the dashboard keeps the old definition, and the chart is named."""
    built = _built(fake, store, THREE_OF_POLICIES)
    restless = built["policies_by_channel"]

    def hand_edit(request, _world):
        if request.get("chart_id") == restless:
            world.charts[restless]["changed_on"] = world._stamp()

    world.hooks["patch_chart"] = Hook(before=hand_edit)
    result = edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="Policies",
                          expression="COUNT(DISTINCT policy_id)", kind=MeasureKind.COUNT, currency=None,
                          description="")
    assert result.changed == () and result.could_not_read == (restless,)
    for chart_id in (built["policies_by_region"], built["policies_by_status"], restless):
        assert _metric(world, chart_id)["sqlExpression"] == "COUNT(policy_id)", chart_id
    assert _kept(store, built["dashboard"])["policies"].measure.expression == "COUNT(policy_id)"


def test_an_edit_to_a_label_another_measure_has_is_refused(fake, world, store):
    """Two measures under one label could not be told apart on a chart: the edit is refused, and nothing changes."""
    built = _built(fake, store, BuildPlan(charts=(BY_REGION, dataclasses.replace(
        BY_STATUS, measures=(WrittenRef(dataclasses.replace(POLICIES, name="others", label="Others")),))),
        measures=(POLICIES, dataclasses.replace(POLICIES, name="others", label="Others"))))
    with pytest.raises(ActionRefused, match="label"):
        edit_measure(_asking(fake, store, built["dashboard"]), "policies", label="others",
                     expression="COUNT(policy_id)", kind=MeasureKind.COUNT, currency=None, description="")
    assert "patch_chart" not in _tools(fake)


def test_a_value_that_is_not_a_finite_number_is_no_value():
    """An infinity or a NaN Superset answers is no value, said so, never a number the list answers (which JSON could
    not carry)."""
    from mnemiq.superset.measure_actions import _finite

    assert _finite("3.5") == (3.5, None)
    for value in (float("nan"), float("inf"), "-inf"):
        number, note = _finite(value)
        assert number is None and note == "its value is not a finite number"
    assert _finite("n/a")[0] is None

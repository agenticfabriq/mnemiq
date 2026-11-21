"""Changing a dashboard by chat: the profile's model reads the person's request against what the dashboard's charts
show, and mnemiq writes only the settings the request names, through the extension's `patch_chart`, so a hand edit the
request did not touch stays as it is. A key outside what a chart's type may have set by chat is dropped and named; a
chart changed meanwhile is read again and changed once more, then named; a chart the person does not edit is named.

The charts are built by the real builder against `FakeSuperset` and `World`, and each change runs through the real
client and adapter, the model replaced by `FakeInterpreter`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mnemiq.config import Settings
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.builder import next_step, start_build
from mnemiq.superset.chart_change import (
    SETTABLE,
    ChartChange,
    FakeInterpreter,
    Interpretation,
    LLMChangeInterpreter,
    change_dashboard,
    change_prompt,
    dashboard_summary,
)
from mnemiq.superset.charts import Family, WrittenRef
from mnemiq.superset.client import DelegatedToken, SupersetMcp
from mnemiq.superset.measure_actions import Asking
from mnemiq.superset.planner import BuildPlan
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
BY_MONTH = _chart("policies_by_month", Family.LINE, "Policies by month", POLICY_DATASET, (WrittenRef(POLICIES),),
                  time_column="issued", time_grain="P1M")
TWO = BuildPlan(charts=(BY_MONTH, BY_REGION), measures=(POLICIES,))
DAILY = "show daily instead of monthly"


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


def _built(fake, store, plan: BuildPlan = TWO) -> dict:
    state = start_build(store, tenant=TENANT, principal=PRINCIPAL, goal="a goal", plan=plan)
    for _ in range(20):
        adapter = _adapter(fake)
        build = store.load_build(TENANT, state.build_id, PRINCIPAL)
        state = next_step(build, client=adapter.client, adapter=adapter, store=store, snapshot=SNAPSHOT,
                          grants=GRANTS, settings=SETTINGS)
        if state.status != "building":
            break
    assert state.status == "done", state
    return {**{step.key: step.object_id for step in state.steps if step.kind == "chart"},
            "dashboard": state.dashboard_id}


def _asking(fake, store, dashboard_id: int) -> Asking:
    return Asking(adapter=_adapter(fake), store=store, tenant=TENANT, principal=PRINCIPAL, dashboard_id=dashboard_id,
                  snapshot=SNAPSHOT, grants=GRANTS, settings=SETTINGS)


def _change(fake, store, built, request: str, interpretation: Interpretation):
    return change_dashboard(_asking(fake, store, built["dashboard"]), request,
                            interpreter=FakeInterpreter(interpretation))


def _patches(world) -> list[dict]:
    return [call.request for call in world.calls("patch_chart")]


def _by_hand_a_bar(world, chart_id: int) -> None:
    """The chart's type changed by hand in Explore, from a line to a bar."""
    chart = world.charts[chart_id]
    chart["viz_type"] = "echarts_timeseries_bar"
    chart["params"] = {**chart["params"], "viz_type": "echarts_timeseries_bar"}
    chart["changed_on"] = world._stamp()


def test_hand_changed_type_survives_a_grain_change(fake, world, store):
    """A chart built as a line and made a bar by hand: "show daily instead of monthly" writes only the grain, and
    the chart stays a bar."""
    built = _built(fake, store)
    chart = built["policies_by_month"]
    _by_hand_a_bar(world, chart)
    answer = _change(fake, store, built, DAILY, Interpretation(changes=(
        ChartChange(chart_id=chart, set={"time_grain_sqla": "P1D"}),)))
    [patch] = _patches(world)
    assert patch["chart_id"] == chart and patch["set"] == {"time_grain_sqla": "P1D"} and patch.get("unset", []) == []
    assert world.charts[chart]["viz_type"] == "echarts_timeseries_bar"
    assert world.charts[chart]["params"]["time_grain_sqla"] == "P1D"
    assert answer.changed == (chart,) and not answer.left and not answer.dropped


def test_change_names_only_settable_keys(fake, world, store):
    """A key the chart's type may not have set by chat is dropped, never sent, and named with its chart."""
    built = _built(fake, store)
    chart = built["policies_by_month"]
    answer = _change(fake, store, built, DAILY, Interpretation(changes=(ChartChange(
        chart_id=chart, set={"time_grain_sqla": "P1D", "query_mode": "raw", "datasource": "1__table"}),)))
    [patch] = _patches(world)
    assert patch["set"] == {"time_grain_sqla": "P1D"}
    assert answer.dropped == ((chart, "datasource"), (chart, "query_mode"))
    assert "query_mode" not in SETTABLE[Family.LINE] and "time_grain_sqla" in SETTABLE[Family.LINE]


def test_type_changes_only_when_asked(fake, world, store):
    """mnemiq never sends a chart's type itself: a grain change carries none, and a type the request names is sent,
    among the types a chart of that kind can take."""
    built = _built(fake, store)
    chart = built["policies_by_month"]
    _change(fake, store, built, DAILY, Interpretation(changes=(
        ChartChange(chart_id=chart, set={"time_grain_sqla": "P1D"}),)))
    assert "viz_type" not in _patches(world)[0]["set"]
    _change(fake, store, built, "make it a bar chart", Interpretation(changes=(
        ChartChange(chart_id=chart, set={"viz_type": "echarts_timeseries_bar"}),)))
    assert world.charts[chart]["viz_type"] == "echarts_timeseries_bar"
    answer = _change(fake, store, built, "make it a funnel", Interpretation(changes=(
        ChartChange(chart_id=chart, set={"viz_type": "funnel"}),)))
    assert answer.dropped == ((chart, "viz_type"),) and world.charts[chart]["viz_type"] == "echarts_timeseries_bar"


def test_concurrent_hand_edit_wins(fake, world, store):
    """A chart changed by hand between the read and the write: the write is refused as stale, the chart read again,
    and the change applied over the hand edit, which stays. Changed meanwhile twice, the chart is named and left."""
    built = _built(fake, store)
    chart = built["policies_by_month"]
    edits = {"n": 0}

    def hand_edit(request, _world):
        if edits["n"] < edits.get("limit", 1):
            edits["n"] += 1
            world.charts[chart]["params"] = {**world.charts[chart]["params"], "color_scheme": f"hand{edits['n']}"}
            world.charts[chart]["changed_on"] = world._stamp()

    world.hooks["patch_chart"] = Hook(before=hand_edit)
    answer = _change(fake, store, built, DAILY, Interpretation(changes=(
        ChartChange(chart_id=chart, set={"time_grain_sqla": "P1D"}),)))
    assert answer.changed == (chart,) and len(_patches(world)) == 2
    assert world.charts[chart]["params"]["color_scheme"] == "hand1"
    assert world.charts[chart]["params"]["time_grain_sqla"] == "P1D"
    edits.update(n=0, limit=2)
    answer = _change(fake, store, built, "show weekly", Interpretation(changes=(
        ChartChange(chart_id=chart, set={"time_grain_sqla": "P1W"}),)))
    assert answer.changed == () and [left[0] for left in answer.left] == [chart]
    assert world.charts[chart]["params"]["time_grain_sqla"] == "P1D"


def test_a_filter_change_keeps_the_charts_other_filters(fake, world, store):
    """A filter added by chat joins the chart's filters; a filter on another column and the builder's leave-out
    condition stay as they are."""
    built = _built(fake, store, BuildPlan(charts=(BY_REGION,), measures=(POLICIES,)))
    chart = built["policies_by_region"]
    world.charts[chart]["params"]["adhoc_filters"] = [
        {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "channel", "operator": "==", "comparator": "web"}]
    world.charts[chart]["changed_on"] = world._stamp()
    _change(fake, store, built, "only active policies", Interpretation(changes=(ChartChange(
        chart_id=chart, add_filters=({"column": "status", "op": "=", "value": "active"},)),)))
    filters = world.charts[chart]["params"]["adhoc_filters"]
    assert [(f["subject"], f["operator"], f["comparator"]) for f in filters] == [("channel", "==", "web"),
                                                                                 ("status", "==", "active")]
    _change(fake, store, built, "all channels", Interpretation(changes=(ChartChange(
        chart_id=chart, remove_filters=("channel",)),)))
    assert [f["subject"] for f in world.charts[chart]["params"]["adhoc_filters"]] == ["status"]


def test_a_change_on_a_chart_the_person_does_not_edit_is_named(fake, world, store):
    built = _built(fake, store)
    chart = built["policies_by_month"]
    world.charts[chart]["editors"] = ["someone"]
    answer = _change(fake, store, built, DAILY, Interpretation(changes=(
        ChartChange(chart_id=chart, set={"time_grain_sqla": "P1D"}),)))
    assert answer.changed == () and answer.not_editor[0].chart_id == chart and answer.not_editor[0].editors == (
        "someone",)
    assert _patches(world) == []


def test_a_chart_not_on_the_dashboard_is_never_changed(fake, world, store):
    """The model may name only a chart the dashboard places: any other id is left, unread and unwritten."""
    built = _built(fake, store)
    other = world.plant_chart(title="Elsewhere", dataset_id=POLICY_DATASET, viz_type="table")
    answer = _change(fake, store, built, DAILY, Interpretation(changes=(
        ChartChange(chart_id=other, set={"row_limit": 5}),)))
    assert answer.changed == () and [left[0] for left in answer.left] == [other] and _patches(world) == []


def test_a_removal_by_chat_is_proposed_and_nothing_is_deleted(fake, world, store):
    """The model's reading never deletes anything: a chart it reads the request as removing is answered as a
    proposal, by id and title, for the person to confirm."""
    built = _built(fake, store)
    chart = built["policies_by_region"]
    answer = _change(fake, store, built, "drop the region chart", Interpretation(remove_chart_ids=(chart,)))
    assert answer.would_remove == ((chart, "Policies by region"),) and answer.removed == ()
    assert chart in world.charts and "delete_chart" not in [c.tool for c in fake.calls]


def test_a_confirmed_removal_deletes_those_charts(fake, world, store):
    """Confirmed by the person, the charts go to Superset's trash and off the layout, and out of the measures they
    showed; the model is not asked."""
    built = _built(fake, store)
    chart = built["policies_by_region"]
    interpreter = FakeInterpreter(Interpretation())
    answer = change_dashboard(_asking(fake, store, built["dashboard"]), "", interpreter=interpreter,
                              confirmed_removal=(chart,))
    assert answer.removed == (chart,) and chart in world.trashed and interpreter.requests == []
    [kept] = store.measures_for(TENANT, built["dashboard"])
    assert chart not in kept.chart_ids


def test_a_confirmed_removal_superset_refuses_is_named(fake, world, store):
    """A chart a report holds is left on the dashboard and named, with the reason."""
    built = _built(fake, store)
    chart = built["policies_by_region"]
    world.reported.add(chart)
    answer = change_dashboard(_asking(fake, store, built["dashboard"]), "", interpreter=FakeInterpreter(
        Interpretation()), confirmed_removal=(chart,))
    assert answer.removed == () and answer.left == ((chart, "a report holds it"),)
    assert chart in world.charts


def test_a_filter_added_twice_is_added_once(fake, world, store):
    """A change sent again (its answer lost) adds no second copy of a filter the chart already has."""
    built = _built(fake, store, BuildPlan(charts=(BY_REGION,), measures=(POLICIES,)))
    chart = built["policies_by_region"]
    change = Interpretation(changes=(ChartChange(chart_id=chart, add_filters=(
        {"column": "status", "op": "=", "value": "active"},)),))
    _change(fake, store, built, "only active", change)
    _change(fake, store, built, "only active", change)
    subjects = [f["subject"] for f in world.charts[chart]["params"]["adhoc_filters"] if f.get("clause") == "WHERE"]
    assert subjects == ["status"]


def test_the_prompt_holds_the_charts_settings_and_never_an_editor(fake, world, store):
    """What the model sees: each chart's id, title, type, dataset, measures by label, grouping, grain, row limit and
    filters, and the columns it can group or filter by; never an editor."""
    built = _built(fake, store)
    world.charts[built["policies_by_month"]]["editors"] = ["a-person-named-only-to-the-panel"]
    summary = dashboard_summary(_asking(fake, store, built["dashboard"]))
    system, user = change_prompt(DAILY, summary)
    assert "a-person-named-only-to-the-panel" not in system + user
    shown = json.loads(user.split("\n\n", 1)[1].split(":", 1)[1])
    [month] = [c for c in shown["charts"] if c["id"] == built["policies_by_month"]]
    assert month["title"] == "Policies by month" and month["time_grain"] == "P1M" and month["measures"] == ["Policies"]
    assert "region" in month["columns"]


def test_the_llm_interpreter_reads_its_reply_strictly():
    """The model's reply is read into an interpretation; one that is not the JSON shape asked for is a refusal."""

    class Model:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def complete(self, system, user, **kwargs):
            return self.reply

    read = LLMChangeInterpreter(Model(json.dumps({"changes": [{"chart_id": 3, "set": {"row_limit": 5}}],
                                                  "add_chart": False, "remove_chart_ids": []})))
    found = read.interpret(DAILY, {"charts": []})
    assert found.changes == (ChartChange(chart_id=3, set={"row_limit": 5}),) and not found.add_chart
    refused = LLMChangeInterpreter(Model("no json here")).interpret(DAILY, {"charts": []})
    assert refused.reason and not refused.changes


def _calls_update_chart(source: str) -> list[int]:
    """The lines of `source` that call a tool named `update_chart` (a string constant passed to any call)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and any(isinstance(arg, ast.Constant) and arg.value == "update_chart"
                                              for arg in [*node.args, *(k.value for k in node.keywords)]):
            found.append(node.lineno)
    return found


def test_change_never_calls_update_chart():
    """MCP's `update_chart` replaces a chart's settings wholesale and loses hand edits: no change sends it. The
    guard is shown its planted twin."""
    package = Path(__file__).resolve().parents[1] / "src" / "mnemiq" / "superset"
    assert {path.name: found for path in sorted(package.glob("*.py"))
            if (found := _calls_update_chart(path.read_text()))} == {}
    assert _calls_update_chart('client.call("update_chart", {"identifier": 1})') == [1]
    assert _calls_update_chart('client.call(tool="update_chart", request={})') == [1]


def test_the_mcp_2x_workflow_runs_this_module():
    import re

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "mcp-2x.yml"
    runs = [line for line in workflow.read_text().splitlines() if re.search(r"\brun:.*\bpytest\b", line)]
    assert len(runs) == 1 and f"tests/{Path(__file__).name}" in runs[0].split()


# --- A change previewed, then applied exactly: what a conversation shows before "Apply", and what "Apply" writes

def _preview(fake, store, built, interpretation: Interpretation, request: str = DAILY):
    from mnemiq.superset.chart_change import preview_change

    return preview_change(_asking(fake, store, built["dashboard"]), request,
                          interpreter=FakeInterpreter(interpretation))


def _apply(fake, store, built, preview, *, strict: bool = True):
    from mnemiq.superset.chart_change import apply_change

    return apply_change(_asking(fake, store, built["dashboard"]), preview, strict=strict)


def _two_edits(built) -> Interpretation:
    return Interpretation(changes=(ChartChange(chart_id=built["policies_by_month"], set={"time_grain_sqla": "P1D"}),
                                   ChartChange(chart_id=built["policies_by_region"], set={"row_limit": 5})))


def _recorded(preview):
    from mnemiq.superset.chart_change import ChangePreview

    return ChangePreview.from_json(json.loads(json.dumps(preview.to_json())))


def test_a_preview_writes_nothing(fake, world, store):
    """A preview reads the dashboard and its charts, as the change does, and names each edit as it will be sent; it
    writes nothing."""
    built = _built(fake, store)
    month = built["policies_by_month"]
    world.charts[month]["params"]["color_scheme"] = "by hand"
    preview = _preview(fake, store, built, _two_edits(built))
    assert _patches(world) == []
    first, second = preview.edits
    assert (first.chart_id, dict(first.to_set), first.to_unset) == (month, {"time_grain_sqla": "P1D"}, ())
    assert (second.chart_id, dict(second.to_set)) == (built["policies_by_region"], {"row_limit": 5})
    assert first.title == "Policies by month" and first.said.startswith("Policies by month: ")
    assert first.changed_on == world._iso(world.charts[month]["changed_on"])
    assert preview.dashboard_changed_on == world._iso(world.dashboards[built["dashboard"]]["changed_on"])
    assert preview.message.startswith("To change: Policies by month: ")


def test_apply_sends_exactly_the_previewed_edits_and_asks_no_model(fake, world, store):
    """Applied from its record, as a confirm applies it, each edit is sent as the preview names it, against the
    `changed_on` the preview read. `apply_change` takes no model: nothing is read again by one."""
    import inspect

    from mnemiq.superset.chart_change import apply_change

    assert "interpreter" not in inspect.signature(apply_change).parameters
    built = _built(fake, store)
    preview = _preview(fake, store, built, _two_edits(built))
    answer = _apply(fake, store, built, _recorded(preview))
    patches = _patches(world)
    assert [(p["chart_id"], p["set"], p.get("unset", []), p["expect_changed_on"]) for p in patches] == [
        (edit.chart_id, dict(edit.to_set), list(edit.to_unset), edit.changed_on) for edit in preview.edits]
    assert answer.changed == tuple(edit.chart_id for edit in preview.edits) and answer.left == ()
    assert world.charts[built["policies_by_month"]]["params"]["time_grain_sqla"] == "P1D"


def test_a_previewed_filter_is_applied_from_its_record(fake, world, store):
    """A filter the person typed is kept in the record under `operand` and sent as the preview planned it."""
    built = _built(fake, store, BuildPlan(charts=(BY_REGION,), measures=(POLICIES,)))
    chart = built["policies_by_region"]
    preview = _preview(fake, store, built, Interpretation(changes=(ChartChange(
        chart_id=chart, add_filters=({"column": "status", "op": "=", "value": "active"},)),)), "only active")
    doc = preview.to_json()
    assert doc["edits"][0]["change"]["add_filters"] == [{"column": "status", "op": "=", "operand": "active"}]
    _apply(fake, store, built, _recorded(preview))
    [patch] = _patches(world)
    assert patch["set"] == dict(preview.edits[0].to_set)
    assert any(f.get("subject") == "status" and f.get("comparator") == "active"
               for f in world.charts[chart]["params"]["adhoc_filters"])


def test_a_strict_apply_refuses_when_a_previewed_chart_changed(fake, world, store):
    """A chart edited by hand since the preview: nothing at all is written, every edit left with the same reason, so
    the person asks again against the dashboard as it is."""
    from mnemiq.superset.chart_change import STALE_SINCE_PREVIEW

    built = _built(fake, store)
    preview = _recorded(_preview(fake, store, built, _two_edits(built)))
    region = built["policies_by_region"]
    world.charts[region]["params"] = {**world.charts[region]["params"], "color_scheme": "by hand"}
    world.charts[region]["changed_on"] = world._stamp()
    answer = _apply(fake, store, built, preview)
    assert _patches(world) == []
    assert answer.changed == ()
    assert answer.left == tuple((edit.chart_id, STALE_SINCE_PREVIEW) for edit in preview.edits)
    assert STALE_SINCE_PREVIEW == "it changed since the preview; ask again"


def test_a_strict_apply_refuses_when_the_dashboard_changed(fake, world, store):
    from mnemiq.superset.chart_change import STALE_SINCE_PREVIEW

    built = _built(fake, store)
    preview = _recorded(_preview(fake, store, built, _two_edits(built)))
    world.dashboards[built["dashboard"]]["changed_on"] = world._stamp()
    answer = _apply(fake, store, built, preview)
    assert _patches(world) == [] and answer.changed == ()
    assert answer.left == tuple((edit.chart_id, STALE_SINCE_PREVIEW) for edit in preview.edits)


def test_a_strict_apply_refuses_a_record_whose_plan_would_differ(fake, world, store):
    """A record whose edit would no longer plan to what was previewed (its digest) writes nothing."""
    from mnemiq.superset.chart_change import STALE_SINCE_PREVIEW, ChangePreview

    built = _built(fake, store)
    doc = _preview(fake, store, built, _two_edits(built)).to_json()
    doc["edits"][1]["change"]["set"] = {"row_limit": 50}
    answer = _apply(fake, store, built, ChangePreview.from_json(doc))
    assert _patches(world) == [] and answer.changed == ()
    assert [chart for chart, why in answer.left if why == STALE_SINCE_PREVIEW] == [
        built["policies_by_month"], built["policies_by_region"]]


def test_a_strict_apply_refuses_a_stale_write_and_does_not_retry(fake, world, store):
    """A chart edited by hand between the check and its write: the apply stops there. What was written stays, and
    the answer names it changed and the rest left; the stale chart is not read again or written."""
    from mnemiq.superset.chart_change import STALE_SINCE_PREVIEW

    built = _built(fake, store)
    month, region = built["policies_by_month"], built["policies_by_region"]
    preview = _recorded(_preview(fake, store, built, _two_edits(built)))
    seen = {"n": 0}

    def hand_edit_the_second(request, _world):
        seen["n"] += 1
        if request["chart_id"] == region:
            world.charts[region]["params"] = {**world.charts[region]["params"], "color_scheme": "by hand"}
            world.charts[region]["changed_on"] = world._stamp()

    world.hooks["patch_chart"] = Hook(before=hand_edit_the_second)
    answer = _apply(fake, store, built, preview)
    assert [p["chart_id"] for p in _patches(world)] == [month, region]
    assert answer.changed == (month,) and answer.left == ((region, STALE_SINCE_PREVIEW),)
    assert world.charts[region]["params"]["color_scheme"] == "by hand"
    assert world.charts[region]["params"].get("row_limit") != 5


def test_a_non_strict_apply_needs_the_preview_it_made(fake, world, store):
    """The retry of today's change plans again from what the model read, which a record does not hold."""
    built = _built(fake, store)
    preview = _recorded(_preview(fake, store, built, _two_edits(built)))
    with pytest.raises(ValueError):
        _apply(fake, store, built, preview, strict=False)


def test_a_preview_round_trips_through_json():
    """A preview that sets every key a chart may have set by chat, unsets one and adds and drops filters: its record
    is one a conversation keeps (`message_shape`), names no value-shaped key, and reads back as it was written."""
    import dataclasses

    from mnemiq.superset.chart_change import ChangePreview, PlannedEdit
    from mnemiq.superset.conversations import message_shape
    from mnemiq.superset.measure_actions import EditorsOf

    lists = {"groupby", "groupbyRows", "groupbyColumns"}
    every = {key: (["region"] if key in lists else {"symbol": "USD", "symbolPosition": "prefix"}
                   if key == "currency_format" else True if key == "order_desc" else 10 if key == "row_limit"
                   else "x") for key in sorted(set().union(*SETTABLE.values()))}
    recorded = ChartChange(chart_id=7, set=every, unset=("row_limit",),
                           add_filters=({"column": "region", "op": "IN", "value": ["west", "east"]},
                                        {"column": "status", "op": "IS NULL", "value": None}),
                           remove_filters=("stage",))
    edit = PlannedEdit(chart_id=7, title="Sales", changed_on="2026-11-21T10:00:00", to_set={"row_limit": 10},
                       to_unset=("row_limit",), said="Sales: at most 10 rows", digest="ab" * 32, recorded=recorded)
    preview = ChangePreview(dashboard_id=3, dashboard_changed_on="2026-11-21T09:00:00", edits=(edit,),
                            would_remove=((9, "Old"), (10, None)), add_chart=True, left=((8, "why"),),
                            dropped=((7, "query_mode"),), not_editor=(EditorsOf(editors=("ana",), chart_id=5),),
                            message="To change: Sales: at most 10 rows.", reason=None)
    doc = json.loads(json.dumps(preview.to_json()))
    message_shape({"role": "mnemiq", "kind": "change", "text": preview.message, "proposal": doc})
    from mnemiq.superset.chart_change import PREVIEW_JSON_SHAPE

    # The declared shape is the writer's own, both ways: every key written is admitted, every key admitted written.
    assert set(doc) == set(PREVIEW_JSON_SHAPE)
    assert set(doc["edits"][0]) == set(PREVIEW_JSON_SHAPE["edits"][0])
    assert set(doc["edits"][0]["change"]) == set(PREVIEW_JSON_SHAPE["edits"][0]["change"])
    assert set(doc["edits"][0]["change"]["set"]) == set(PREVIEW_JSON_SHAPE["edits"][0]["change"]["set"])
    assert doc["edits"][0]["change"]["add_filters"][0] == {"column": "region", "op": "IN", "operand": ["west", "east"]}
    back = ChangePreview.from_json(doc)
    assert back == dataclasses.replace(preview, edits=(dataclasses.replace(edit, to_set={}, to_unset=()),))
    assert back.edits[0].recorded == recorded and back.edits[0].change is None

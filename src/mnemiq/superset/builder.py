"""A dashboard build, run one step per request: the dashboard, then each dataset the plan makes, each chart, the
layout, and the value checks, in that order.

**One step, claimed first.** `next_step` claims the build's next step in the build store before it calls Superset
(`BuildStore.claim_next_step`), recording the step's intent: its kind, the name it will give what it makes, the dataset,
and, kept by the store, when it began. A second request while the claim's lease holds is refused (`BuildBusy`, a
409). The step renews its lease before each of its calls to Superset and starts no call once the lease has passed:
it then stops and leaves the step to be reconciled. It records its outcome before it returns.

**Every object a build makes carries its marker** (`mnemiq.superset.marker`), the last line of its description: the
dashboard shell, virtual datasets, funnels and sankeys from the create itself; charts MCP's `generate_chart` makes,
and physical datasets, from the extension's claim right after.

**A step whose outcome is uncertain is reconciled before anything else.** A call that timed out, a worker that died
or ran past its lease, a refusal of the caller midway: the step is `uncertain`, and the next request looks for what it
may have made, never making a second and never taking another build's.

- The object this step's own create answered with, when the step recorded it, is the step's.
- Otherwise, one object carrying this step's marker, compared field by field (build, step, dashboard), is adopted.
- Otherwise, for the moment between an MCP create and its claim: the objects with no marker that have the planned
  name, are on the planned dataset, were made by the builder (`created_by_me`), and were made after the step began.
  Exactly one is claimed; none, and the create is made again; more than one, and the build stops and names them.
- An object carrying any other marker is never touched.

"Made after the step began" compares two clocks: the step's `started_at` is the control Postgres's, in UTC, and an
object's `created_on` is Superset's app server's (see `RECONCILE_MARGIN_S` and `superset_timezone`).

**What a build keeps** is the plan, the step states, the ids of what it made, and each measure of the dashboard with
the charts that show it and each chart's check status. Never a value: a check's numbers are compared and dropped.

Every Superset call goes through the request's adapter's client, carrying the request's own token.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any

import psycopg

from mnemiq.superset.build_store import (
    DEFAULT_LEASE_S,
    DONE,
    FAILED,
    UNCERTAIN,
    BuildNotFound,
    LeaseLost,
    StepBusy,
    StepState,
    StoredBuild,
)
from mnemiq.superset.chart_check import ChartStatus, check_chart
from mnemiq.superset.charts import (
    EXTENSION_FAMILIES,
    ChartSpec,
    SavedRef,
    WrittenRef,
    extension_chart_request,
    generate_chart_request,
    owed_format_params,
)
from mnemiq.superset.client import SupersetUnavailable
from mnemiq.superset.extension import call_tool
from mnemiq.superset.layout import layout_chart_ids, layout_positions
from mnemiq.superset.marker import Marker, format_marker, parse_marker
from mnemiq.superset.planner import BuildPlan, PlannedDataset, plan_from_json, plan_to_json, with_made_datasets
from mnemiq.superset.saved_metrics import SavedMetric, read_saved_metrics

__all__ = ["DASHBOARD_TITLE_MAX", "RECONCILE_MARGIN_S", "BuildBusy", "BuildState", "build_state", "build_steps",
           "next_step", "start_build"]

logger = logging.getLogger(__name__)

# How far apart the control Postgres's clock and Superset's app server's may be. An object counts as made after its
# step began when Superset's `created_on` for it is no earlier than this before the step's `started_at`. Superset
# writes `created_on` with its server's `datetime.now()` (`AuditMixinNullable` in `superset/models/helpers.py`): the
# wall clock in that server's own zone, with no zone written, which MCP's list tools send as naive ISO 8601 text
# (`ChartInfo.created_on`, `DatasetInfo.created_on`, `DashboardInfo.created_on`). So a naive `created_on` is read in
# `superset_timezone` (UTC unless set; Superset's images run in UTC); in an hour a zone's clock repeats, as the later of
# its two readings, so a time that could fall in the window counts as in it. An aware one is read as written.
RECONCILE_MARGIN_S = 120

# rc2's MCP caps a dashboard title it makes up at this (`_MAX_TITLE_LENGTH`, `generate_dashboard`).
DASHBOARD_TITLE_MAX = 150
# rc2 keeps a dataset's name in 250 characters (`SqlaTable.table_name`); a virtual dataset's ends with its build's.
_DATASET_NAME_MAX = 250
_BUILD_SUFFIX = 8
# One page of the objects of a planned name the builder made: more than this is more than reconciliation can read.
_CANDIDATE_PAGE = 100

BUILDING = "building"


class BuildBusy(Exception):
    """No step was run: another request's claim on step `key` still holds, or (`key` None) the build moved on since
    the caller read it. Either way, read the build again."""

    def __init__(self, key: str | None) -> None:
        super().__init__("a step of this build is already running" if key is not None
                         else "the build moved on since it was read; read it again")
        self.key = key


@dataclass(frozen=True)
class BuildState:
    """A build as a request answers it: its status (`building`, `done` or `failed`), its dashboard once there is one,
    its steps (each `pending`, `done`, `uncertain` or `failed`, and never with its lease id), and each chart's check
    status. `ran` is the step this request ran or reconciled, and `message` mnemiq's own words on it: why it stopped,
    or which charts did not check. Neither is kept: a later read has neither."""

    build_id: str
    status: str
    dashboard_id: int | None
    steps: list[StepState]
    checks: dict[int, ChartStatus]
    ran: str | None = None
    message: str | None = None


# ---------------------------------------------------------------- starting and reading a build


def build_steps(plan: BuildPlan) -> list[tuple[str, str]]:
    """The build's steps, each `(key, kind)`: the dashboard, each dataset the plan makes, each chart, the layout, the
    checks."""
    return [("dashboard", "dashboard"), *((dataset.key, "dataset") for dataset in plan.datasets),
            *((chart.key, "chart") for chart in plan.charts), ("layout", "layout"), ("check", "check")]


def start_build(store, *, tenant: str, principal: str, goal: str, plan: BuildPlan,
                dashboard_id: int | None = None) -> BuildState:
    """A new build of `plan`, on the existing dashboard `dashboard_id` or a new one. Refused (ValueError) while the
    plan still offers a dashboard to reuse, which the builder answers first; `DashboardBusy` while another build of the
    dashboard is building."""
    if plan.awaiting_choice:
        raise ValueError("this plan offers a dashboard to reuse: answer reuse or fresh before it is built")
    stored = store.create_build(tenant, principal, goal, plan_to_json(plan), build_steps(plan),
                                dashboard_id=dashboard_id)
    return build_state(store, stored)


def build_state(store, build: StoredBuild, *, ran: str | None = None, message: str | None = None,
                checks: Mapping[int, ChartStatus] | None = None) -> BuildState:
    """`build` as a request answers it. Its charts' check statuses are read from the dashboard's measures unless
    `checks` gives them (the check step's own, which also covers a chart that plots only saved metrics)."""
    charts = {step.object_id for step in build.steps if step.kind == "chart" and step.object_id is not None}
    found: dict[int, ChartStatus] = {}
    if checks is not None:
        found = dict(checks)
    elif build.dashboard_id is not None and charts:
        for kept in store.measures_for(build.tenant_id, build.dashboard_id):
            found.update({chart: status for chart, status in kept.check_status.items() if chart in charts})
    return BuildState(build_id=build.build_id, status=build.status, dashboard_id=build.dashboard_id,
                      steps=[dataclasses.replace(step, lease_id=None) for step in build.steps], checks=found, ran=ran,
                      message=message)


# ---------------------------------------------------------------- one step


@dataclass(frozen=True)
class _Outcome:
    status: str
    object_id: int | None
    message: str | None = None
    checks: dict[int, ChartStatus] | None = None


class _Stop(Exception):
    """The step fails: mnemiq's own reason, which names what it found and nothing Superset said."""


class _Unsure(Exception):
    """The step's outcome cannot be settled now (Superset busy, a listing it could not read): it is left uncertain,
    to be reconciled by the next request."""


class _LeaseGone(Exception):
    """The step's lease passed, or another claim holds the step: it starts no further call."""


@dataclass
class _Run:
    """The step being run: the build as read, the claim, the plan with each made dataset's id, and what the step
    has made so far (`made`), recorded with any outcome but done."""

    store: Any
    build: StoredBuild
    claimed: StepState
    plan: BuildPlan
    adapter: Any
    snapshot: Any
    grants: Any
    settings: Any
    lease_s: int
    made: int | None = None
    _saved: dict[int, dict[str, SavedMetric]] = field(default_factory=dict)

    @property
    def client(self):
        return self.adapter.client

    @property
    def key(self) -> str:
        return self.claimed.key

    @property
    def uncertain(self) -> bool:
        return self.claimed.status == UNCERTAIN

    def renew(self) -> None:
        """Renew the lease before a call; once it cannot be, no further call is started."""
        if not self.store.renew_lease(self.build.tenant_id, self.build.build_id, self.key, self.claimed.lease_id,
                                      self.lease_s):
            raise _LeaseGone()

    def call(self, tool: str, request: dict) -> dict:
        self.renew()
        return self.client.call(tool, request)

    def extension(self, base: str, request: dict) -> dict:
        self.renew()
        return call_tool(self.client, base, request)

    def saved(self, dataset_id: int) -> dict[str, SavedMetric]:
        """The dataset's saved metrics, as the builder reads them, by name; read once per step."""
        if dataset_id not in self._saved:
            self.renew()
            self._saved[dataset_id] = {m.name: m for m in read_saved_metrics(self.client, dataset_ids=(dataset_id,))}
        return self._saved[dataset_id]

    def finish(self, status: str, object_id: int | None) -> StoredBuild:
        """Record the outcome. When the store's connection drops on the way, the build is read again before anything
        is decided: a finish that committed is not taken for a lost lease, and one that did not is made again."""
        tenant, build_id = self.build.tenant_id, self.build.build_id
        try:
            return self.store.finish_step(tenant, build_id, self.key, status, object_id,
                                          lease_id=self.claimed.lease_id)
        except psycopg.OperationalError:
            logger.warning("a build step's finish lost its answer from the store; reading the build again")
            reread = self.store.load_build(tenant, build_id, self.build.principal_id)
            if reread is None:
                raise BuildNotFound() from None
            now = reread.step(self.key)
            if now.lease_id == self.claimed.lease_id:
                return self.store.finish_step(tenant, build_id, self.key, status, object_id,
                                              lease_id=self.claimed.lease_id)
            if now.lease_id is None and now.status == status and now.object_id == object_id:
                return reread
            raise LeaseLost() from None

    def finish_quietly(self, status: str, object_id: int | None) -> StoredBuild | None:
        """`finish`, on a path already failing: a lease taken over, or a store that cannot answer, leaves the step for
        reconciliation and is logged, not raised over the failure."""
        try:
            return self.finish(status, object_id)
        except (LeaseLost, BuildNotFound, psycopg.Error) as exc:
            logger.warning("a build step could not record that it is %s (%s); reconciliation finds it", status,
                           type(exc).__name__)
            return None

    def chart_ids(self) -> dict[str, int]:
        return {step.key: step.object_id for step in self.build.steps if step.kind == "chart"
                and step.object_id is not None}

    def made_dataset_ids(self) -> set[int]:
        return {step.object_id for step in self.build.steps if step.kind == "dataset" and step.object_id is not None}


def next_step(build: StoredBuild, *, client, adapter, store, snapshot, grants, settings,
              lease_s: int = DEFAULT_LEASE_S) -> BuildState:
    """Claim and run `build`'s next step, as the builder `adapter` acts for: `client` is the adapter's own client, and
    every Superset call goes through it. A step left uncertain is reconciled first. Raises `BuildBusy` while another
    request's step holds the build; a refusal of the caller (`SupersetRefusal`) and any failure mnemiq cannot read as an
    outcome are raised once the step is recorded uncertain. Superset not answering leaves the step uncertain, said so in
    the state returned."""
    if client is not getattr(adapter, "client", None):
        raise ValueError("a build calls Superset only through the client of the request's adapter")
    target = next((step for step in build.steps if step.status != DONE), None)
    if target is None or build.status != BUILDING:
        stopped = next((step.key for step in build.steps if step.status == FAILED), None)
        return build_state(store, build, message=None if stopped is None else
                           f"this build stopped at its step {stopped!r}; nothing more is built")
    plan = _plan_of(build)
    claimed = store.claim_next_step(build.tenant_id, build.build_id, build.principal_id, _intent(target, build, plan),
                                    lease_s)
    if isinstance(claimed, StepBusy):
        raise BuildBusy(claimed.key)
    if claimed is None:
        reread = store.load_build(build.tenant_id, build.build_id, build.principal_id)
        return build_state(store, reread or build)
    run = _Run(store=store, build=build, claimed=claimed, plan=plan, adapter=adapter, snapshot=snapshot,
               grants=grants, settings=settings, lease_s=lease_s, made=claimed.object_id)
    try:
        outcome = _RUNNERS[claimed.kind](run)
    except _LeaseGone:
        logger.warning("a build step ran past its lease; it started no further call")
        stored = run.finish_quietly(UNCERTAIN, run.made)
        return _answer(run, stored, "this step ran past its lease, so it started no further call; the next request "
                                    "reconciles it")
    except _Stop as stop:
        return _recorded(run, FAILED, run.made, str(stop))
    except _Unsure as unsure:
        return _recorded(run, UNCERTAIN, run.made, str(unsure))
    except SupersetUnavailable as exc:
        logger.warning("Superset gave no answer during a build step (%s); the step is left uncertain",
                       type(exc).__name__)
        return _recorded(run, UNCERTAIN, run.made, "Superset did not answer this step in time, so whether it made "
                                                   "anything is not known; the next request finds out first")
    except Exception:
        run.finish_quietly(UNCERTAIN, run.made)
        raise
    return _recorded(run, outcome.status, outcome.object_id, outcome.message, outcome.checks)


_TAKEN_OVER = "this step ran past its lease and another request took it over; that request records it"


def _recorded(run: _Run, status: str, object_id: int | None, message: str | None,
              checks: dict[int, ChartStatus] | None = None) -> BuildState:
    """The step recorded as `status`, and the build as it then is. A step another request took over meanwhile is
    that request's to record: the answer says so."""
    try:
        stored = run.finish(status, object_id)
    except LeaseLost:
        return _answer(run, None, _TAKEN_OVER)
    return build_state(run.store, stored, ran=run.key, message=message, checks=checks)


def _answer(run: _Run, stored: StoredBuild | None, message: str) -> BuildState:
    build = run.build
    if stored is None:
        stored = run.store.load_build(build.tenant_id, build.build_id, build.principal_id) or build
    return build_state(run.store, stored, ran=run.key, message=message)


def _plan_of(build: StoredBuild) -> BuildPlan:
    """The build's plan, each dataset it has made in place of its placeholder."""
    made = {step.key: step.object_id for step in build.steps
            if step.kind == "dataset" and step.status == DONE and step.object_id is not None}
    return with_made_datasets(plan_from_json(build.plan), made)


def _title(goal: str) -> str:
    return " ".join(goal.split())[:DASHBOARD_TITLE_MAX]


def _planned(plan: BuildPlan, key: str) -> PlannedDataset:
    return next(dataset for dataset in plan.datasets if dataset.key == key)


def _chart_spec(plan: BuildPlan, key: str) -> ChartSpec:
    return next(chart for chart in plan.charts if chart.key == key)


def _dataset_name(build: StoredBuild, planned: PlannedDataset) -> str:
    """A physical dataset is named as its table is stored; a virtual one by the plan, then its build's id, so two
    builds that join the same tables make two datasets of two names."""
    if not planned.virtual:
        return planned.table_name
    suffix = build.build_id[:_BUILD_SUFFIX]
    return f"{planned.name[:_DATASET_NAME_MAX - len(suffix) - 1]}_{suffix}"


def _intent(target: StepState, build: StoredBuild, plan: BuildPlan) -> dict:
    """What the claim records before any call: the step, its kind, the name it gives what it makes, its dataset."""
    name, dataset_id = None, None
    if target.kind == "dashboard":
        name = None if build.dashboard_id is not None else _title(build.goal)
    elif target.kind == "dataset":
        name = _dataset_name(build, _planned(plan, target.key))
    elif target.kind == "chart":
        spec = _chart_spec(plan, target.key)
        name, dataset_id = spec.title, spec.dataset_id if spec.dataset_id > 0 else None
    return {"key": target.key, "kind": target.kind, "name": name, "dataset_id": dataset_id}


# ---------------------------------------------------------------- what the extension answers, in mnemiq's words


def _refused(answer: Mapping, what: str) -> _Stop:
    """The step's failure for an extension's answer that is not success: what it says, never Superset's text."""
    if answer.get("not_editor"):
        editors = ", ".join(str(label) for label in answer.get("editors") or [])
        return _Stop(f"this builder is not an editor of {what}" + (f"; its editors are {editors}" if editors else ""))
    if "not_found" in answer:
        return _Stop(f"{what}: Superset has no such {answer['not_found']} that this builder can read")
    if answer.get("claimed_by_another"):
        return _Stop(f"{what} carries another build's marker, so this build leaves it alone")
    if answer.get("too_large"):
        return _Stop(f"{what} is larger than the extension sends")
    if "refused" in answer:
        return _Stop(f"the extension refused the request for {what} ({answer['refused']})")
    return _Stop(f"Superset could not save {what}; its log says why")


def _check_answer(answer: Mapping, what: str) -> None:
    """Return on success; leave the step uncertain on a busy row (the same request may be sent again); else stop."""
    if answer.get("ok") is True:
        return
    if answer.get("busy"):
        raise _Unsure(f"Superset was busy with {what}; the next request carries on")
    raise _refused(answer, what)


# ---------------------------------------------------------------- reconciling an uncertain step


def _zone(settings) -> tzinfo:
    name = getattr(settings, "superset_timezone", "UTC") or "UTC"
    if name == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def _made_in_window(created_on: Any, started_at: str, zone: tzinfo) -> bool | None:
    """Whether Superset made the object no earlier than `RECONCILE_MARGIN_S` before the step began; None when its
    `created_on` cannot be read."""
    if not isinstance(created_on, str):
        return None
    try:
        moment = datetime.fromisoformat(created_on)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = max(moment.replace(tzinfo=zone, fold=fold).astimezone(UTC) for fold in (0, 1))
    return moment >= datetime.fromisoformat(started_at) - timedelta(seconds=RECONCILE_MARGIN_S)


def _listed(answer: Any, kind: str) -> list[dict]:
    """The objects a list tool answered, or `_Unsure` when the answer is not a whole page of them."""
    items = answer.get(kind) if isinstance(answer, Mapping) else None
    if not isinstance(items, list):
        raise _Unsure(f"Superset's list of {kind} could not be read, so what this step made cannot be told yet")
    found = [item for item in items if isinstance(item, Mapping) and type(item.get("id")) is int]
    total = answer.get("total_count")
    if len(found) != len(items) or (type(total) is int and total > len(found)):
        raise _Unsure(f"Superset listed more {kind} of this name than mnemiq reads at once, so what this step made "
                      "cannot be told")
    return found


def _settle(run: _Run, found: list[dict], *, ours: Marker, matches: Callable[[Mapping], bool], unmarked: bool,
            what: str) -> tuple[int | None, bool]:
    """What an uncertain step made, among `found`: `(id, marked)` for the one object carrying `ours`, compared field
    by field, or, when the step's create leaves its object unmarked until a claim (`unmarked`), for the one object
    with no marker that `matches` the plan and was made in the step's window. `(None, False)`: the step made nothing,
    and makes it again. An object carrying any other marker is passed over. More than one of either kind stops the
    build, naming them."""
    mine, candidates = [], []
    zone = _zone(run.settings)
    for item in found:
        if not matches(item):
            continue
        marker = parse_marker(item.get("description"))
        if marker == ours:
            mine.append(item)
        elif marker is None and unmarked:
            window = _made_in_window(item.get("created_on"), run.claimed.started_at, zone)
            if window is not False:
                candidates.append((item, window))
    if len(mine) == 1:
        return mine[0]["id"], True
    if mine:
        raise _Stop(f"Superset holds {len(mine)} {what}s carrying this step's marker ("
                    f"{', '.join(str(item['id']) for item in mine)}), so this build cannot tell which is its own")
    if len(candidates) == 1 and candidates[0][1] is True:
        return candidates[0][0]["id"], False
    if candidates:
        raise _Stop(f"Superset holds {len(candidates)} unmarked {what}s of this name, made by this builder since this "
                    f"step began ({', '.join(str(item['id']) for item, _ in candidates)}), so this build cannot tell "
                    "which it made; remove the ones that are not wanted and start the build again")
    return None, False


def _the_one_made(run: _Run, item: Mapping | None, ours: Marker, what: str) -> tuple[int | None, bool]:
    """The object the step's own create answered with, read again: `(id, marked)`, or `(None, False)` when it is gone
    and is made again. One carrying another marker since stops the build."""
    if item is None or type(item.get("id")) is not int:
        return None, False
    marker = parse_marker(item.get("description"))
    if marker is None:
        return item["id"], False
    if marker == ours:
        return item["id"], True
    raise _Stop(f"the {what} this step made ({item['id']}) now carries another build's marker, so this build leaves "
                "it alone")


# What MCP's info tools answer for an object that does not exist or that the caller cannot read
# (`ModelGetInfoCore.run_tool` in `superset/mcp_service/mcp_core.py`).
_NOT_FOUND = "not_found"


def _info(run: _Run, tool: str, object_id: int, columns: list[str]) -> dict | None:
    """The object as an info tool answers it, or None when Superset says it does not exist. Any other error is no
    answer at all (`_Unsure`): an object that cannot be read now is never taken for one that is gone, which would
    make it again."""
    answer = run.call(tool, {"identifier": object_id, "select_columns": columns})
    if not isinstance(answer, Mapping):
        raise _Unsure("Superset's answer about an object this step made could not be read; the next request asks again")
    if answer.get("error_type") == _NOT_FOUND:
        return None
    if answer.get("error_type") or answer.get("error"):
        logger.warning("%s answered a build's read with an error (%s)", tool, answer.get("error_type"))
        raise _Unsure("Superset answered a read of an object this step made with an error; the next request asks "
                      "again")
    return dict(answer)


# ---------------------------------------------------------------- the steps


def _run_dashboard(run: _Run) -> _Outcome:
    build = run.build
    if build.dashboard_id is not None:
        # An existing dashboard: its layout and `changed_on` are read now, and nothing is held across steps; each later
        # write takes the dashboard's row lock itself, and the layout step reads it afresh.
        answer = run.extension("get_layout", {"dashboard_id": build.dashboard_id})
        if answer.get("dashboard_id") != build.dashboard_id or not isinstance(answer.get("changed_on"), str):
            raise _refused(answer, f"dashboard {build.dashboard_id}")
        return _Outcome(DONE, build.dashboard_id)
    title = _title(build.goal)
    ours = Marker(build.build_id, "dashboard", None)
    if run.uncertain:
        listed = _listed(run.call("list_dashboards", {
            "filters": [{"col": "dashboard_title", "opr": "eq", "value": title}], "created_by_me": True,
            "select_columns": ["id", "dashboard_title", "description", "created_on"], "page": 1,
            "page_size": _CANDIDATE_PAGE}), "dashboards")
        found, _ = _settle(run, listed, ours=ours, matches=lambda item: item.get("dashboard_title") == title,
                           unmarked=False, what="dashboard")
        if found is not None:
            return _Outcome(DONE, found)
    answer = run.extension("create_dashboard", {"title": title, "marker": format_marker(*dataclasses.astuple(ours))})
    if type(answer.get("dashboard_id")) is int and answer["dashboard_id"] > 0:
        return _Outcome(DONE, answer["dashboard_id"])
    raise _refused(answer, "the dashboard")


def _run_dataset(run: _Run) -> _Outcome:
    build, planned = run.build, _planned(run.plan, run.key)
    name, dashboard_id = _dataset_name(build, planned), build.dashboard_id
    ours = Marker(build.build_id, planned.key, dashboard_id)
    marker = format_marker(*dataclasses.astuple(ours))
    dataset_id = None
    if run.uncertain:
        dataset_id, _ = _reconciled_dataset(run, planned, name, ours)
    if dataset_id is None:
        dataset_id = _made_dataset(run, planned, name, marker)
    run.made = dataset_id
    # The claim writes the editors (and a physical dataset's marker); made twice, it answers as the first did.
    _check_answer(run.extension("claim_dataset", {"dataset_id": dataset_id, "dashboard_id": dashboard_id,
                                                  "marker": marker}), f"the dataset {name!r}")
    return _Outcome(DONE, dataset_id)


def _reconciled_dataset(run: _Run, planned: PlannedDataset, name: str, ours: Marker) -> tuple[int | None, bool]:
    columns = ["id", "table_name", "schema", "database_id", "sql", "description", "created_on"]
    if run.made is not None:
        return _the_one_made(run, _info(run, "get_dataset_info", run.made, columns), ours, "dataset")
    adapter = run.adapter
    listed = _listed(run.call("list_datasets", {
        "filters": [{"col": "table_name", "opr": "eq", "value": name},
                    {"col": "schema", "opr": "eq", "value": adapter.schema}],
        "created_by_me": True, "select_columns": columns, "page": 1, "page_size": _CANDIDATE_PAGE}), "datasets")

    def planned_here(item: Mapping) -> bool:
        return (item.get("table_name") == name and item.get("schema") == adapter.schema
                and item.get("database_id") == adapter.database_id and bool(item.get("sql")) == planned.virtual)

    # A virtual dataset carries its marker from its create, so an unmarked one is never this step's.
    return _settle(run, listed, ours=ours, matches=planned_here, unmarked=not planned.virtual, what="dataset")


def _made_dataset(run: _Run, planned: PlannedDataset, name: str, marker: str) -> int:
    adapter = run.adapter
    if planned.virtual:
        prose = f"Made by mnemiq for dashboard {run.build.dashboard_id}."
        answer = run.call("create_virtual_dataset", {"database_id": adapter.database_id, "sql": planned.sql,
                                                     "dataset_name": name, "schema_name": adapter.schema,
                                                     "description": f"{prose}\n{marker}"})
    else:
        answer = run.call("create_dataset", {"database_id": adapter.database_id, "schema": adapter.schema,
                                             "table_name": planned.table_name})
    made = answer.get("id") if isinstance(answer, Mapping) else None
    if type(made) is int and made > 0:
        return made
    logger.warning("Superset did not make a build's dataset (%s): %s", answer.get("error_type"), answer.get("error"))
    if answer.get("error_type") == "DatasetExistsError":
        raise _Stop(f"Superset already has a dataset named {name!r} on this schema, which this builder cannot use; "
                    "the build uses only datasets it can read or makes")
    raise _Stop(f"Superset did not make the dataset {name!r}; mnemiq's log records why")


def _run_chart(run: _Run) -> _Outcome:
    build, spec = run.build, _chart_spec(run.plan, run.key)
    dashboard_id = build.dashboard_id
    ours = Marker(build.build_id, spec.key, dashboard_id)
    marker = format_marker(*dataclasses.astuple(ours))
    measures = {measure.name: measure for measure in run.plan.measures}
    saved = run.saved(spec.dataset_id) if any(isinstance(ref, SavedRef) for ref in spec.measures) else {}
    currency = run.settings.superset_currency
    by_extension = spec.family in EXTENSION_FAMILIES
    chart_id, marked = None, False
    if run.uncertain:
        chart_id, marked = _reconciled_chart(run, spec, ours, by_extension)
    if chart_id is None:
        if by_extension:
            answer = run.extension("create_chart", {**extension_chart_request(spec, measures, saved,
                                                                              currency=currency),
                                                    "dashboard_id": dashboard_id, "marker": marker})
            chart_id = answer.get("chart_id")
            if type(chart_id) is not int or chart_id <= 0:
                raise _refused(answer, f"the chart {spec.title!r}")
            marked = True
        else:
            chart_id = _generated_chart(run, generate_chart_request(spec, measures, saved, currency=currency),
                                        spec.title)
    run.made = chart_id
    if not marked:
        # The claim writes the dashboard's editors, the marker and the chart's query context, under the chart's lock.
        _check_answer(run.extension("claim_chart", {"chart_id": chart_id, "dashboard_id": dashboard_id,
                                                    "marker": marker}), f"the chart {spec.title!r}")
    if not by_extension:
        owed = owed_format_params(spec, measures, saved, currency=currency)
        if owed:
            _patched(run, chart_id, owed, spec.title)
    _keep_measures(run, spec, chart_id)
    return _Outcome(DONE, chart_id)


def _generated_chart(run: _Run, request: dict, title: str) -> int:
    answer = run.call("generate_chart", request)
    chart = answer.get("chart") if isinstance(answer, Mapping) and answer.get("success") is not False else None
    chart_id = chart.get("id") if isinstance(chart, Mapping) else None
    if type(chart_id) is int and chart_id > 0:
        return chart_id
    logger.warning("Superset did not make a build's chart: %s", answer.get("error") if isinstance(answer, Mapping)
                   else type(answer).__name__)
    raise _Stop(f"Superset did not make the chart {title!r}; mnemiq's log records why")


def _reconciled_chart(run: _Run, spec: ChartSpec, ours: Marker, by_extension: bool) -> tuple[int | None, bool]:
    if run.made is not None:
        item = _info(run, "get_chart_info", run.made, ["id", "slice_name", "description"])
        return _the_one_made(run, item, ours, "chart")
    dataset = _info(run, "get_dataset_info", spec.dataset_id, ["id", "table_name", "schema"])
    if dataset is None or not isinstance(dataset.get("table_name"), str):
        raise _Stop(f"the chart's dataset {spec.dataset_id} is not one this builder can read")
    # A chart's dataset, as `list_charts` names it: `SqlaTable.name`, the schema and the table's name.
    named = f"{dataset['schema']}.{dataset['table_name']}" if dataset.get("schema") else dataset["table_name"]
    listed = _listed(run.call("list_charts", {
        "filters": [{"col": "slice_name", "opr": "eq", "value": spec.title}], "created_by_me": True,
        "select_columns": ["id", "slice_name", "description", "datasource_name", "created_on"], "page": 1,
        "page_size": _CANDIDATE_PAGE}), "charts")
    # A funnel or a sankey carries its marker from its create, so an unmarked one is never this step's.
    return _settle(run, listed, ours=ours,
                   matches=lambda item: item.get("slice_name") == spec.title and item.get("datasource_name") == named,
                   unmarked=not by_extension, what="chart")


def _patched(run: _Run, chart_id: int, owed: dict, title: str) -> None:
    """Set the format keys MCP could not send (`owed_format_params`), keeping every other key, once the chart is as
    read: a chart changed between the read and the patch is read again, once."""
    for _ in range(2):
        info = _info(run, "get_chart_info", chart_id, ["id", "changed_on"])
        if info is None or not isinstance(info.get("changed_on"), str):
            raise _Stop(f"mnemiq could not read when the chart {title!r} last changed, so its format was not set")
        answer = run.extension("patch_chart", {"chart_id": chart_id, "set": owed, "unset": [],
                                               "expect_changed_on": info["changed_on"]})
        if not answer.get("stale"):
            _check_answer(answer, f"the chart {title!r}")
            return
    raise _Stop(f"the chart {title!r} kept changing while its format was set; set its number format by hand")


def _keep_measures(run: _Run, spec: ChartSpec, chart_id: int) -> None:
    """Keep each measure the chart writes on the dashboard, with the charts that show it and the datasets the build
    made for it, as asked by the build's principal. A measure the dashboard already has keeps its charts, datasets
    and check statuses, and the chart joins them."""
    build = run.build
    kept = {row.measure.name: row for row in run.store.measures_for(build.tenant_id, build.dashboard_id)}
    made = run.made_dataset_ids()
    for ref in spec.measures:
        if not isinstance(ref, WrittenRef):
            continue
        measure = dataclasses.replace(ref.measure, asked_by=build.principal_id)
        before = kept.get(measure.name)
        charts = list(before.chart_ids) if before else []
        datasets = list(before.made_dataset_ids) if before else []
        if chart_id not in charts:
            charts.append(chart_id)
        if measure.dataset_id in made and measure.dataset_id not in datasets:
            datasets.append(measure.dataset_id)
        run.store.put_measure(build.tenant_id, build.dashboard_id, measure, chart_ids=charts,
                              made_dataset_ids=datasets, check_status=dict(before.check_status) if before else {})


def _run_layout(run: _Run) -> _Outcome:
    """Add the build's charts to the dashboard's layout. A layout raced by another write (`stale`) is read and computed
    again, once."""
    dashboard_id = run.build.dashboard_id
    ids = run.chart_ids()
    charts = [(ids[spec.key], spec.family) for spec in run.plan.charts]
    for _ in range(2):
        answer = run.extension("get_layout", {"dashboard_id": dashboard_id})
        if answer.get("dashboard_id") != dashboard_id or not isinstance(answer.get("changed_on"), str):
            raise _refused(answer, f"the layout of dashboard {dashboard_id}")
        try:
            current = json.loads(answer["position_json"]) if answer.get("position_json") else {}
            if not isinstance(current, dict):
                raise ValueError("not an object")
            to_add = [(chart_id, family) for chart_id, family in charts if chart_id not in layout_chart_ids(current)]
            if not to_add:
                return _Outcome(DONE, None)
            positions = layout_positions(to_add, onto=current)
        except ValueError as exc:
            raise _Stop(f"the layout dashboard {dashboard_id} has is not one mnemiq can add charts to ({exc})") from None
        answer = run.extension("set_layout", {"dashboard_id": dashboard_id, "positions": positions,
                                              "add_chart_ids": [chart_id for chart_id, _ in to_add],
                                              "remove_chart_ids": [], "expect_changed_on": answer["changed_on"]})
        if not answer.get("stale"):
            _check_answer(answer, f"the layout of dashboard {dashboard_id}")
            return _Outcome(DONE, None)
    raise _Stop(f"dashboard {dashboard_id} kept changing while its layout was saved; save it again later")


def _run_check(run: _Run) -> _Outcome:
    """Check each chart's numbers and format (`check_chart`), the lease renewed before each chart, and keep each
    chart's status with the dashboard's measures. A chart that does not check is reported as what it is; the build
    still ends."""
    ids = run.chart_ids()
    statuses: dict[int, ChartStatus] = {}
    off: list[str] = []
    for spec in run.plan.charts:
        chart_id = ids[spec.key]
        saved = run.saved(spec.dataset_id)
        kinds = {(ref.name if isinstance(ref, SavedRef) else ref.measure.label): ref.kind
                 if isinstance(ref, SavedRef) else ref.measure.kind for ref in spec.measures}
        run.renew()
        found = check_chart(run.adapter, chart_id=chart_id, dataset_id=spec.dataset_id, kinds=kinds, saved=saved,
                            snapshot=run.snapshot, grants=run.grants, guard_fanout=run.settings.guard_fanout)
        statuses[chart_id] = found.status
        if found.status is not ChartStatus.CHECKED:
            off.append(f"{spec.title!r} ({found.status.value}: {found.reason})")
    build = run.build
    for row in run.store.measures_for(build.tenant_id, build.dashboard_id):
        mine = {chart: statuses[chart] for chart in row.chart_ids if chart in statuses}
        if mine:
            run.store.put_measure(build.tenant_id, build.dashboard_id, row.measure, chart_ids=row.chart_ids,
                                  made_dataset_ids=row.made_dataset_ids, check_status={**row.check_status, **mine})
    total = len(run.plan.charts)
    message = (f"every chart checked ({total} of {total})" if not off else
               f"{total - len(off)} of {total} charts checked; not checked: " + "; ".join(off))
    return _Outcome(DONE, None, message=message, checks=statuses)


_RUNNERS: dict[str, Callable[[_Run], _Outcome]] = {"dashboard": _run_dashboard, "dataset": _run_dataset,
                                                   "chart": _run_chart, "layout": _run_layout, "check": _run_check}

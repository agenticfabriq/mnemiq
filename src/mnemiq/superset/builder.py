"""A dashboard build, run one step per request: the dashboard, then each dataset the plan makes, each chart, the
layout, and the value checks, in that order.

**One step, claimed first.** `next_step` claims the build's next step in the build store before it calls Superset
(`BuildStore.claim_next_step`), recording the step's intent: its kind, the name it will give what it makes, the dataset,
and, kept by the store, when it began. A second request while the claim's lease holds is refused (`BuildBusy`, a
409). The step renews its lease before each of its calls to Superset, inside a chart's value check too, and starts no
call once the lease has passed: it then stops and leaves the step to be reconciled. It records its outcome, with a
reason code from the store's set when it fails, before it returns.

**Every object a build makes carries its marker** (`mnemiq.superset.marker`), the last line of its description: the
dashboard shell, virtual datasets, funnels and sankeys from the create itself; charts MCP's `generate_chart` makes,
and physical datasets, from the extension's claim right after.

**Before each create** the step reads the highest id among the objects of that kind the builder made
(`created_by_me`, by id, descending: Superset's ids only grow) and records it in its intent as `id_floor`. Whatever
the create makes has a higher id, whatever existed before does not, so no clock is compared. No create is sent
without a floor recorded, so a step with none made nothing.

**A create Superset may still be running is left to its lease.** A create whose call ends without an answer
(`SupersetUnavailable`, a timeout among them) is not finished: its claim runs out (the renewal just before the call
put the lease's end at least `DEFAULT_LEASE_S` after it began, past the longest a call runs), so by then Superset has
made the object or never will. A claim, a patch or a layout save is made under the object's row lock and answers the
same when made twice, so one that timed out leaves its step uncertain at once.

**A step whose outcome is uncertain is reconciled before anything else**, never making a second object and never
taking another build's:

- the object the step's own create answered with, when the step recorded it, read back by its id ("not found" means
  it is gone; any other error, that it cannot be told yet);
- otherwise, among the builder's objects of that kind above the floor, read newest first page by page down to it:
  the one carrying this step's marker, compared field by field (build, step, dashboard), whatever it is called by
  now; or, for the moment between an MCP create and its claim, the one with no marker of the planned name on the
  planned dataset. Exactly one is claimed; none, and the create is made again; more than one, and the build stops and
  names them.
- An object carrying any other marker is never touched.

A create that answers failure is reconciled the same way before it is believed, since rc2's `generate_chart` and
`create_dataset` can answer failure after Superset committed. A create whose tool fails twice fails its step.

**What a build keeps** is the plan, the step states, the ids of what it made, and each measure of the dashboard (a
saved metric a chart reuses among them) with the charts that show it and each chart's check status. Never a value: a
check's numbers are compared and dropped.

Every Superset call goes through the request's adapter's client, carrying the request's own token.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import psycopg
from sqlglot.errors import SqlglotError

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
from mnemiq.superset.client import SupersetToolError, SupersetUnavailable
from mnemiq.superset.extension import call_tool, dashboard_editorship
from mnemiq.superset.layout import layout_chart_ids, layout_positions
from mnemiq.superset.marker import Marker, format_marker, parse_marker
from mnemiq.superset.measures import Measure
from mnemiq.superset.planner import BuildPlan, PlannedDataset, plan_from_json, plan_to_json, with_made_datasets
from mnemiq.superset.saved_metrics import SavedMetric, normalized, read_saved_metrics

__all__ = ["DASHBOARD_TITLE_MAX", "BuildBusy", "BuildState", "build_state", "build_steps", "next_step",
           "start_build"]

logger = logging.getLogger(__name__)

# rc2's MCP caps a dashboard title it makes up at this (`_MAX_TITLE_LENGTH`, `generate_dashboard`).
DASHBOARD_TITLE_MAX = 150
# rc2 keeps a dataset's name in 250 characters (`SqlaTable.table_name`); a virtual dataset's ends with its build's.
_DATASET_NAME_MAX = 250
_BUILD_SUFFIX = 8
# One page of the builder's objects read back to a floor, and the most pages read before the step is left uncertain.
_CANDIDATE_PAGE = 100
_MAX_PAGES = 20

BUILDING = "building"

# What a later read of a stopped build says for each reason code its step was left with (`build_store.REASONS`).
_REASON_SAID = {
    "not_editor": "this builder is not an editor of an object the build changes",
    "not_found": "an object the build needs is not one this builder can read",
    "claimed_by_another": "an object the build was to use carries another build's marker",
    "extension_refused": "the extension refused a request of the build",
    "not_saved": "Superset could not save an object of the build",
    "too_large": "an object the build reads is larger than the extension sends",
    "candidates": "Superset holds more than one object this step may have made, and the build cannot tell which",
    "not_made": "Superset did not make an object of the build",
    "dataset_exists": "Superset already has a dataset of the planned name that this builder cannot use",
    "layout_unreadable": "the dashboard's layout could not be read, or is not one mnemiq can add charts to",
    "kept_changing": "an object kept changing while the build wrote it",
    "measure_conflict": "the dashboard keeps a measure of a planned name with another definition",
    "tool_error": "Superset's tool failed on the same step twice",
    "abandoned": ("the build was left untouched for longer than twice its lease, so another build of the dashboard "
                  "took the dashboard over"),
    "unreadable": "Superset's answer about what the step made could not be read",
    "busy": "Superset was busy with an object of the step",
    "superset_silent": "Superset did not answer a call of the step in time",
    "lease_passed": "the step ran past its lease",
}


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
    its steps (each `pending`, `done`, `uncertain` or `failed`, with its reason code, and never with its lease id), and
    each chart's check status. `ran` is the step this request ran or reconciled, and `message` mnemiq's own words on
    it: why it stopped (from its reason code on a later read), or which charts did not check. A `not_editor` message
    names the editors, for the HTTP answer only: never a field a model sees."""

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
    dashboard is building and has been touched within `build_store.ABANDONED_AFTER_S`."""
    if plan.awaiting_choice:
        raise ValueError("this plan offers a dashboard to reuse: answer reuse or fresh before it is built")
    stored = store.create_build(tenant, principal, goal, plan_to_json(plan), build_steps(plan),
                                dashboard_id=dashboard_id)
    return build_state(store, stored)


def build_state(store, build: StoredBuild, *, ran: str | None = None, message: str | None = None,
                checks: Mapping[int, ChartStatus] | None = None) -> BuildState:
    """`build` as a request answers it. Its charts' check statuses are read from the dashboard's measures unless
    `checks` gives them (the check step's own). A stopped build with no `message` says why from its step's reason."""
    charts = {step.object_id for step in build.steps if step.kind == "chart" and step.object_id is not None}
    found: dict[int, ChartStatus] = {}
    if checks is not None:
        found = dict(checks)
    elif build.dashboard_id is not None and charts:
        for kept in store.measures_for(build.tenant_id, build.dashboard_id):
            found.update({chart: status for chart, status in kept.check_status.items() if chart in charts})
    if message is None and build.status == FAILED:
        stopped = next((step for step in build.steps if step.status == FAILED), None)
        if stopped is not None:
            said = _REASON_SAID.get(stopped.reason or "", "its log says why")
            message = f"this build stopped at its step {stopped.key!r}: {said}"
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
    """The step fails: a reason code of the store's set, and mnemiq's own sentence, which names what it found and
    nothing Superset said."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class _Unsure(_Stop):
    """The step's outcome cannot be settled now (Superset busy, an answer it could not read): it is left uncertain, to
    be reconciled by the next request."""


class _LeaseGone(Exception):
    """The step's lease passed, or another claim holds the step: it starts no further call."""


@dataclass
class _Run:
    """The step being run: the build as read, the claim, the plan with each made dataset's id, the floor its create
    is judged by, and what the step has made so far (`made`), recorded with any outcome but done."""

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
    floor: int | None = None
    # What a create was making when its call ended unanswered: the step is then left to its lease.
    in_flight: str | None = None
    create_failed: bool = False
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

    def ensure_floor(self, tool: str, kind: str) -> int:
        """The step's floor: the one it recorded, or the highest id among the builder's objects of this kind, read now
        by id, descending, and recorded in its intent before any create is sent."""
        if self.floor is not None:
            return self.floor
        answer = self.call(tool, {"created_by_me": True, "select_columns": ["id"], "order_column": "id",
                                  "order_direction": "desc", "page": 1, "page_size": 1})
        items = answer.get(kind) if isinstance(answer, Mapping) else None
        if not isinstance(items, list):
            raise _Unsure("unreadable", f"Superset's list of the builder's {kind} could not be read, so no create was "
                                        "sent; the next request tries again")
        floor = max((item["id"] for item in items if isinstance(item, Mapping) and type(item.get("id")) is int),
                    default=0)
        if not self.store.record_floor(self.build.tenant_id, self.build.build_id, self.key, self.claimed.lease_id,
                                       floor):
            raise _LeaseGone()
        self.floor = floor
        return floor

    def finish(self, status: str, object_id: int | None, reason: str | None = None) -> StoredBuild:
        """Record the outcome. When the store's connection drops on the way, the build is read again before anything
        is decided: a finish that committed is not taken for a lost lease, and one that did not is made again."""
        tenant, build_id = self.build.tenant_id, self.build.build_id
        try:
            return self.store.finish_step(tenant, build_id, self.key, status, object_id,
                                          lease_id=self.claimed.lease_id, reason=reason)
        except psycopg.OperationalError:
            logger.warning("a build step's finish lost its answer from the store; reading the build again")
            reread = self.store.load_build(tenant, build_id, self.build.principal_id)
            if reread is None:
                raise BuildNotFound() from None
            now = reread.step(self.key)
            if now.lease_id == self.claimed.lease_id:
                return self.store.finish_step(tenant, build_id, self.key, status, object_id,
                                              lease_id=self.claimed.lease_id, reason=reason)
            if now.lease_id is None and now.status == status and now.object_id == object_id:
                return reread
            raise LeaseLost() from None

    def finish_quietly(self, status: str, object_id: int | None, reason: str | None = None) -> StoredBuild | None:
        """`finish`, on a path already failing: a lease taken over, or a store that cannot answer, leaves the step for
        reconciliation and is logged, not raised over the failure."""
        try:
            return self.finish(status, object_id, reason)
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
    outcome are raised once the step is recorded uncertain. A create Superset may still be making leaves the step to
    its lease; any other call Superset did not answer leaves it uncertain; either is said in the state returned."""
    if client is not getattr(adapter, "client", None):
        raise ValueError("a build calls Superset only through the client of the request's adapter")
    target = next((step for step in build.steps if step.status != DONE), None)
    if target is None or build.status != BUILDING:
        return build_state(store, build)
    plan = _plan_of(build)
    claimed = store.claim_next_step(build.tenant_id, build.build_id, build.principal_id, _intent(target, build, plan),
                                    lease_s)
    if isinstance(claimed, StepBusy):
        raise BuildBusy(claimed.key)
    if claimed is None:
        reread = store.load_build(build.tenant_id, build.build_id, build.principal_id)
        return build_state(store, reread or build)
    run = _Run(store=store, build=build, claimed=claimed, plan=plan, adapter=adapter, snapshot=snapshot,
               grants=grants, settings=settings, lease_s=lease_s, made=claimed.object_id,
               floor=(claimed.intent or {}).get("id_floor"))
    try:
        outcome = _RUNNERS[claimed.kind](run)
    except _LeaseGone:
        logger.warning("a build step ran past its lease; it started no further call")
        stored = run.finish_quietly(UNCERTAIN, run.made, "lease_passed")
        return _answer(run, stored, "this step ran past its lease, so it started no further call; the next request "
                                    "reconciles it")
    except _Unsure as unsure:
        return _recorded(run, UNCERTAIN, run.made, str(unsure), reason=unsure.reason)
    except _Stop as stop:
        return _recorded(run, FAILED, run.made, str(stop), reason=stop.reason)
    except SupersetUnavailable as exc:
        logger.warning("Superset gave no answer during a build step (%s)", type(exc).__name__)
        if run.in_flight is not None:
            return _left_to_its_lease(run)
        return _recorded(run, UNCERTAIN, run.made, "Superset did not answer this step in time, so whether it changed "
                                                   "anything is not known; the next request finds out first",
                         reason="superset_silent")
    except Exception:
        run.finish_quietly(UNCERTAIN, run.made, "tool_error" if run.create_failed else None)
        raise
    return _recorded(run, outcome.status, outcome.object_id, outcome.message, outcome.checks)


_TAKEN_OVER = "this step ran past its lease and another request took it over; that request records it"


def _recorded(run: _Run, status: str, object_id: int | None, message: str | None,
              checks: dict[int, ChartStatus] | None = None, *, reason: str | None = None) -> BuildState:
    """The step recorded as `status`, and the build as it then is. A step another request took over meanwhile is
    that request's to record: the answer says so."""
    try:
        stored = run.finish(status, object_id, reason)
    except LeaseLost:
        return _answer(run, None, _TAKEN_OVER)
    return build_state(run.store, stored, ran=run.key, message=message, checks=checks)


def _left_to_its_lease(run: _Run) -> BuildState:
    """A create Superset may still be running: nothing is recorded, and the claim runs out, so no request can make
    the object again while Superset might yet make it."""
    reread = run.store.load_build(run.build.tenant_id, run.build.build_id, run.build.principal_id) or run.build
    until = next((step.lease_until for step in reread.steps if step.key == run.key), None)
    when = f" (at {until})" if until else ""
    return build_state(run.store, reread, ran=run.key,
                       message=f"Superset did not answer while making {run.in_flight}, and may still be making it: this "
                               f"step's claim is left to run out{when}, and the first request after that finds what "
                               "was made")


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
    """What the claim records before any call: the step, its kind, the name it gives what it makes, its dataset. Its
    floor is recorded later, before its create."""
    name, dataset_id = None, None
    if target.kind == "dashboard":
        name = None if build.dashboard_id is not None else _title(build.goal)
    elif target.kind == "dataset":
        name = _dataset_name(build, _planned(plan, target.key))
    elif target.kind == "chart":
        spec = _chart_spec(plan, target.key)
        name, dataset_id = spec.title, spec.dataset_id if spec.dataset_id > 0 else None
    return {"key": target.key, "kind": target.kind, "name": name, "dataset_id": dataset_id, "id_floor": None}


# ---------------------------------------------------------------- what the extension answers, in mnemiq's words


def _editors(answer: Mapping) -> str:
    return ", ".join(str(label) for label in answer.get("editors") or [])


def _refused(answer: Mapping, what: str) -> _Stop:
    """The step's failure for an extension's answer that is not success: what it says, never Superset's text."""
    if answer.get("not_editor"):
        editors = _editors(answer)
        return _Stop("not_editor", f"this builder is not an editor of {what}"
                                   + (f"; its editors are {editors}" if editors else ""))
    if "not_found" in answer:
        return _Stop("not_found", f"{what}: Superset has no such {answer['not_found']} that this builder can read")
    if answer.get("claimed_by_another"):
        return _Stop("claimed_by_another", f"{what} carries another build's marker, so this build leaves it alone")
    if answer.get("too_large"):
        return _Stop("too_large", f"{what} is larger than the extension sends")
    if "refused" in answer:
        return _Stop("extension_refused", f"the extension refused the request for {what} ({answer['refused']})")
    return _Stop("not_saved", f"Superset could not save {what}; its log says why")


def _check_answer(answer: Mapping, what: str) -> None:
    """Return on success; leave the step uncertain on a busy row (the same request may be sent again); else stop."""
    if answer.get("ok") is True:
        return
    if answer.get("busy"):
        raise _Unsure("busy", f"Superset was busy with {what}; the next request carries on")
    raise _refused(answer, what)


def _read_refusal(answer: Mapping, what: str) -> _Stop:
    """A read's answer that is not the object: the extension's refusal for it, or a read that failed."""
    if any(key in answer for key in ("not_found", "not_editor", "too_large", "refused", "claimed_by_another")):
        return _refused(answer, what)
    return _Stop("layout_unreadable", f"{what} could not be read")


@contextmanager
def _creating(run: _Run, what: str) -> Iterator[None]:
    """A create's call: one Superset may still be running when the call ends unanswered marks the step to be left to
    its lease, and a tool that fails on a step it failed on before fails the step."""
    try:
        yield
    except SupersetUnavailable:
        run.in_flight = what
        raise
    except SupersetToolError:
        if run.claimed.reason == "tool_error":
            raise _Stop("tool_error", f"Superset's tool failed again making {what}; its log says why") from None
        run.create_failed = True
        raise


# ---------------------------------------------------------------- reconciling an uncertain step


# What MCP's info tools answer for an object that does not exist or that the caller cannot read
# (`ModelGetInfoCore.run_tool` in `superset/mcp_service/mcp_core.py`).
_NOT_FOUND = "not_found"


def _info(run: _Run, tool: str, object_id: int, columns: list[str]) -> dict | None:
    """The object as an info tool answers it, or None when Superset says it does not exist. Any other error is no
    answer at all (`_Unsure`): an object that cannot be read now is never taken for one that is gone, which would
    make it again."""
    answer = run.call(tool, {"identifier": object_id, "select_columns": columns})
    if not isinstance(answer, Mapping):
        raise _Unsure("unreadable", "Superset's answer about an object this step made could not be read; the next "
                                    "request asks again")
    if answer.get("error_type") == _NOT_FOUND:
        return None
    if answer.get("error_type") or answer.get("error"):
        logger.warning("%s answered a build's read with an error (%s)", tool, answer.get("error_type"))
        raise _Unsure("unreadable", "Superset answered a read of an object this step made with an error; the next "
                                    "request asks again")
    return dict(answer)


def _recent(run: _Run, tool: str, kind: str, columns: list[str]) -> list[dict]:
    """The builder's objects of this kind above the step's floor, newest first: pages by id, descending, read until
    one reaches the floor. Superset's ids only grow, so these are what was made since the floor was read."""
    floor = run.floor
    found: dict[int, dict] = {}
    for page in range(1, _MAX_PAGES + 1):
        answer = run.call(tool, {"created_by_me": True, "select_columns": columns, "order_column": "id",
                                 "order_direction": "desc", "page": page, "page_size": _CANDIDATE_PAGE})
        items = answer.get(kind) if isinstance(answer, Mapping) else None
        if not isinstance(items, list):
            raise _Unsure("unreadable", f"Superset's list of the builder's {kind} could not be read, so what this step "
                                        "made cannot be told yet")
        ids = [item["id"] for item in items if isinstance(item, Mapping) and type(item.get("id")) is int]
        found.update({item["id"]: dict(item) for item in items
                      if isinstance(item, Mapping) and type(item.get("id")) is int and item["id"] > floor})
        if len(items) < _CANDIDATE_PAGE or not ids or min(ids) <= floor:
            return sorted(found.values(), key=lambda item: -item["id"])
    raise _Unsure("unreadable", f"the builder made more {kind} since this step began than mnemiq reads back, so what "
                                "this step made cannot be told")


def _settle(found: list[dict], *, ours: Marker, matches: Callable[[Mapping], bool], unmarked: bool,
            what: str) -> tuple[int | None, bool]:
    """What an uncertain step made, among the builder's objects above its floor: `(id, marked)` for the one carrying
    `ours`, compared field by field, whatever it is called by now; or, when the step's create leaves its object
    unmarked until a claim (`unmarked`), for the one with no marker that `matches` the plan. `(None, False)`: the step
    made nothing, and makes it again. An object carrying any other marker is passed over. More than one of either
    kind stops the build, naming them."""
    mine = [item for item in found if parse_marker(item.get("description")) == ours]
    candidates = [item for item in found if unmarked and parse_marker(item.get("description")) is None
                  and matches(item)]
    for group, said in ((mine, f"{what}s carrying this step's marker"),
                        (candidates, f"unmarked {what}s of this name made by this builder since this step began")):
        if len(group) > 1:
            raise _Stop("candidates", f"Superset holds {len(group)} {said} ("
                                      f"{', '.join(str(item['id']) for item in group)}), so this build cannot tell which "
                                      "is its own; remove the ones that are not wanted and start the build again")
    if mine:
        return mine[0]["id"], True
    if candidates:
        return candidates[0]["id"], False
    return None, False


def _the_one_made(item: Mapping | None, ours: Marker, what: str) -> tuple[int | None, bool]:
    """The object the step's own create answered with, read again: `(id, marked)`, or `(None, False)` when it is gone
    and is made again. One carrying another marker since stops the build."""
    if item is None or type(item.get("id")) is not int:
        return None, False
    marker = parse_marker(item.get("description"))
    if marker is None:
        return item["id"], False
    if marker == ours:
        return item["id"], True
    raise _Stop("claimed_by_another", f"the {what} this step made ({item['id']}) now carries another build's marker, "
                                      "so this build leaves it alone")


# ---------------------------------------------------------------- the steps


def _run_dashboard(run: _Run) -> _Outcome:
    build = run.build
    if build.dashboard_id is not None:
        return _existing_dashboard(run, build.dashboard_id)
    title = _title(build.goal)
    ours = Marker(build.build_id, "dashboard", None)
    if run.uncertain and run.floor is not None:
        # The shell carries its marker from its create: found by it, whatever it is called by now.
        found, _ = _settle(_recent(run, "list_dashboards", "dashboards", ["id", "dashboard_title", "description"]),
                           ours=ours, matches=lambda item: False, unmarked=False, what="dashboard")
        if found is not None:
            return _Outcome(DONE, found)
    run.ensure_floor("list_dashboards", "dashboards")
    with _creating(run, "the dashboard"):
        answer = run.extension("create_dashboard", {"title": title, "marker": format_marker(*dataclasses.astuple(ours))})
    if type(answer.get("dashboard_id")) is int and answer["dashboard_id"] > 0:
        return _Outcome(DONE, answer["dashboard_id"])
    raise _refused(answer, "the dashboard")


def _existing_dashboard(run: _Run, dashboard_id: int) -> _Outcome:
    """An existing dashboard: whether the builder edits it, its layout and its `changed_on`, read now, before anything
    is made, and nothing held across steps; each later write takes the dashboard's row lock itself, and the layout step
    reads it afresh."""
    what = f"dashboard {dashboard_id}"
    run.renew()
    entry = dashboard_editorship(run.client, dashboard_id)
    if entry is None:
        raise _Stop("not_found", f"{what}: Superset has no such dashboard that this builder can read")
    if entry.get("too_large"):
        raise _Stop("too_large", f"the editors of {what} are more than the extension sends")
    if entry.get("is_editor") is not True:
        raise _refused({"not_editor": True, "editors": entry.get("editors")}, what)
    answer = run.extension("get_layout", {"dashboard_id": dashboard_id})
    if answer.get("dashboard_id") != dashboard_id or not isinstance(answer.get("changed_on"), str):
        raise _read_refusal(answer, f"the layout of {what}")
    return _Outcome(DONE, dashboard_id)


def _run_dataset(run: _Run) -> _Outcome:
    build, planned = run.build, _planned(run.plan, run.key)
    name, dashboard_id = _dataset_name(build, planned), build.dashboard_id
    ours = Marker(build.build_id, planned.key, dashboard_id)
    marker = format_marker(*dataclasses.astuple(ours))
    dataset_id = None
    if run.uncertain:
        dataset_id, _ = _reconciled_dataset(run, planned, name, ours)
    if dataset_id is None:
        dataset_id = _made_dataset(run, planned, name, marker, ours)
    run.made = dataset_id
    # The claim writes the editors (and a physical dataset's marker); made twice, it answers as the first did.
    _check_answer(run.extension("claim_dataset", {"dataset_id": dataset_id, "dashboard_id": dashboard_id,
                                                  "marker": marker}), f"the dataset {name!r}")
    return _Outcome(DONE, dataset_id)


_DATASET_COLUMNS = ["id", "table_name", "schema", "database_id", "sql", "description"]


def _reconciled_dataset(run: _Run, planned: PlannedDataset, name: str, ours: Marker) -> tuple[int | None, bool]:
    if run.made is not None:
        return _the_one_made(_info(run, "get_dataset_info", run.made, _DATASET_COLUMNS), ours, "dataset")
    if run.floor is None:
        return None, False  # no create is sent before a floor is recorded: nothing was made
    adapter = run.adapter

    def planned_here(item: Mapping) -> bool:
        return (item.get("table_name") == name and item.get("schema") == adapter.schema
                and item.get("database_id") == adapter.database_id and bool(item.get("sql")) == planned.virtual)

    # A virtual dataset carries its marker from its create, so an unmarked one is never this step's.
    return _settle(_recent(run, "list_datasets", "datasets", _DATASET_COLUMNS), ours=ours, matches=planned_here,
                   unmarked=not planned.virtual, what="dataset")


def _made_dataset(run: _Run, planned: PlannedDataset, name: str, marker: str, ours: Marker) -> int:
    adapter = run.adapter
    run.ensure_floor("list_datasets", "datasets")
    with _creating(run, f"the dataset {name!r}"):
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
    logger.warning("Superset answered a build's dataset create with a failure (%s): %s", answer.get("error_type"),
                   answer.get("error"))
    if answer.get("error_type") == "DatasetExistsError":
        raise _Stop("dataset_exists", f"Superset already has a dataset named {name!r} on this schema, which this "
                                      "builder cannot use; the build uses only datasets it can read or makes")
    # Superset may have committed it before the failure (rc2's `create_dataset` answers "created but could not be
    # serialized"): what was made since the floor is read before the failure is believed.
    found, _ = _reconciled_dataset(run, planned, name, ours)
    if found is not None:
        return found
    raise _Stop("not_made", f"Superset did not make the dataset {name!r}; mnemiq's log records why")


def _run_chart(run: _Run) -> _Outcome:
    build, spec = run.build, _chart_spec(run.plan, run.key)
    dashboard_id = build.dashboard_id
    ours = Marker(build.build_id, spec.key, dashboard_id)
    marker = format_marker(*dataclasses.astuple(ours))
    measures = {measure.name: measure for measure in run.plan.measures}
    saved = run.saved(spec.dataset_id) if any(isinstance(ref, SavedRef) for ref in spec.measures) else {}
    currency = run.settings.superset_currency
    by_extension = spec.family in EXTENSION_FAMILIES
    _held_to_the_dashboards_measures(run, spec)
    chart_id, marked = None, False
    if run.uncertain:
        chart_id, marked = _reconciled_chart(run, spec, ours, by_extension)
    if chart_id is None:
        run.ensure_floor("list_charts", "charts")
        if by_extension:
            with _creating(run, f"the chart {spec.title!r}"):
                answer = run.extension("create_chart", {**extension_chart_request(spec, measures, saved,
                                                                                  currency=currency),
                                                        "dashboard_id": dashboard_id, "marker": marker})
            chart_id = answer.get("chart_id")
            if type(chart_id) is not int or chart_id <= 0:
                raise _refused(answer, f"the chart {spec.title!r}")
            marked = True
        else:
            chart_id, marked = _generated_chart(run, spec, ours, generate_chart_request(spec, measures, saved,
                                                                                        currency=currency))
    run.made = chart_id
    if not marked:
        # The claim writes the dashboard's editors, the marker and the chart's query context, under the chart's lock.
        _check_answer(run.extension("claim_chart", {"chart_id": chart_id, "dashboard_id": dashboard_id,
                                                    "marker": marker}), f"the chart {spec.title!r}")
    if not by_extension:
        owed = owed_format_params(spec, measures, saved, currency=currency)
        if owed:
            _patched(run, chart_id, owed, spec.title)
    _keep_measures(run, spec, chart_id, saved)
    return _Outcome(DONE, chart_id)


def _generated_chart(run: _Run, spec: ChartSpec, ours: Marker, request: dict) -> tuple[int, bool]:
    with _creating(run, f"the chart {spec.title!r}"):
        answer = run.call("generate_chart", request)
    chart = answer.get("chart") if isinstance(answer, Mapping) and answer.get("success") is not False else None
    chart_id = chart.get("id") if isinstance(chart, Mapping) else None
    if type(chart_id) is int and chart_id > 0:
        return chart_id, False
    logger.warning("Superset answered a build's chart create with a failure: %s",
                   answer.get("error") if isinstance(answer, Mapping) else type(answer).__name__)
    # rc2's `generate_chart` can answer failure after `CreateChartCommand` committed: what was made since the floor is
    # read before the failure is believed.
    found, marked = _reconciled_chart(run, spec, ours, False)
    if found is not None:
        return found, marked
    raise _Stop("not_made", f"Superset did not make the chart {spec.title!r}; mnemiq's log records why")


def _reconciled_chart(run: _Run, spec: ChartSpec, ours: Marker, by_extension: bool) -> tuple[int | None, bool]:
    if run.made is not None:
        return _the_one_made(_info(run, "get_chart_info", run.made, ["id", "slice_name", "description"]), ours,
                             "chart")
    if run.floor is None:
        return None, False  # no create is sent before a floor is recorded: nothing was made
    found = _recent(run, "list_charts", "charts", ["id", "slice_name", "description", "datasource_name"])
    named: list[str] = []

    def planned_here(item: Mapping) -> bool:
        if item.get("slice_name") != spec.title:
            return False
        if not named:
            # A chart's dataset, as `list_charts` names it: `SqlaTable.name`, the schema and the table's name.
            dataset = _info(run, "get_dataset_info", spec.dataset_id, ["id", "table_name", "schema"])
            if dataset is None or not isinstance(dataset.get("table_name"), str):
                raise _Stop("not_found", f"the chart's dataset {spec.dataset_id} is not one this builder can read")
            named.append(f"{dataset['schema']}.{dataset['table_name']}" if dataset.get("schema")
                         else dataset["table_name"])
        return item.get("datasource_name") == named[0]

    # A funnel or a sankey carries its marker from its create, so an unmarked one is never this step's.
    return _settle(found, ours=ours, matches=planned_here, unmarked=not by_extension, what="chart")


def _patched(run: _Run, chart_id: int, owed: dict, title: str) -> None:
    """Set the format keys MCP could not send (`owed_format_params`), keeping every other key, once the chart is as
    read: a chart changed between the read and the patch is read again, once."""
    for _ in range(2):
        info = _info(run, "get_chart_info", chart_id, ["id", "changed_on"])
        if info is None or not isinstance(info.get("changed_on"), str):
            raise _Stop("not_found", f"mnemiq could not read when the chart {title!r} last changed, so its format was "
                                     "not set")
        answer = run.extension("patch_chart", {"chart_id": chart_id, "set": owed, "unset": [],
                                               "expect_changed_on": info["changed_on"]})
        if not answer.get("stale"):
            _check_answer(answer, f"the chart {title!r}")
            return
    raise _Stop("kept_changing", f"the chart {title!r} kept changing while its format was set; set its number format "
                                 "by hand")


def _same_definition(kept: Measure, planned: Measure) -> bool:
    """One measure: its kind and dataset the same, its expression the same once normalized (as written, when it does
    not parse)."""
    if kept.kind != planned.kind or kept.dataset_id != planned.dataset_id:
        return False
    try:
        return normalized(kept.expression) == normalized(planned.expression)
    except SqlglotError:
        return kept.expression.strip() == planned.expression.strip()


def _held_to_the_dashboards_measures(run: _Run, spec: ChartSpec) -> None:
    """Before the chart is made: a measure it writes that the dashboard already keeps under its name with another
    definition stops the step, since one name would then mean two measures on one dashboard."""
    build = run.build
    kept = {row.measure.name: row for row in run.store.measures_for(build.tenant_id, build.dashboard_id)}
    for ref in spec.measures:
        if isinstance(ref, WrittenRef) and (before := kept.get(ref.measure.name)) is not None \
                and not _same_definition(before.measure, ref.measure):
            raise _Stop("measure_conflict", f"the dashboard keeps a measure {ref.measure.name!r} with another "
                                            f"definition, so the chart {spec.title!r} is not made; give the planned "
                                            "measure another name, or change the one the dashboard keeps")


def _saved_measure(ref: SavedRef, metric: SavedMetric, dataset_id: int, principal: str) -> Measure:
    """A reused saved metric as a measure of the dashboard, named for the metric and its dataset, so two datasets'
    metrics of one name are two measures."""
    symbol = metric.currency.get("symbol") if isinstance(metric.currency, dict) else None
    return Measure(name=f"{ref.name}@{dataset_id}", label=metric.verbose_name or metric.name,
                   expression=metric.expression, kind=ref.kind, dataset_id=dataset_id, description="",
                   asked_by=principal, currency=symbol if isinstance(symbol, str) and symbol.isupper() else None)


def _keep_measures(run: _Run, spec: ChartSpec, chart_id: int, saved: Mapping[str, SavedMetric]) -> None:
    """Keep each measure the chart shows on the dashboard, a saved metric it reuses among them, with the charts that
    show it and the datasets the build made for it, as asked by the build's principal. A measure the dashboard
    already keeps keeps its charts, datasets and check statuses, and the chart joins them."""
    build = run.build
    kept = {row.measure.name: row for row in run.store.measures_for(build.tenant_id, build.dashboard_id)}
    made = run.made_dataset_ids()
    for ref in spec.measures:
        if isinstance(ref, WrittenRef):
            measure, name_of_saved = dataclasses.replace(ref.measure, asked_by=build.principal_id), None
        elif (metric := saved.get(ref.name)) is not None:
            measure, name_of_saved = _saved_measure(ref, metric, spec.dataset_id, build.principal_id), ref.name
        else:
            continue
        before = kept.get(measure.name)
        charts = list(before.chart_ids) if before else []
        datasets = list(before.made_dataset_ids) if before else []
        if chart_id not in charts:
            charts.append(chart_id)
        if measure.dataset_id in made and measure.dataset_id not in datasets:
            datasets.append(measure.dataset_id)
        run.store.put_measure(build.tenant_id, build.dashboard_id, measure, chart_ids=charts,
                              made_dataset_ids=datasets, check_status=dict(before.check_status) if before else {},
                              saved_metric=name_of_saved)


def _run_layout(run: _Run) -> _Outcome:
    """Add the build's charts to the dashboard's layout. A layout raced by another write (`stale`) is read and computed
    again, once."""
    dashboard_id = run.build.dashboard_id
    what = f"the layout of dashboard {dashboard_id}"
    ids = run.chart_ids()
    charts = [(ids[spec.key], spec.family) for spec in run.plan.charts]
    for _ in range(2):
        answer = run.extension("get_layout", {"dashboard_id": dashboard_id})
        if answer.get("dashboard_id") != dashboard_id or not isinstance(answer.get("changed_on"), str):
            raise _read_refusal(answer, what)
        try:
            current = json.loads(answer["position_json"]) if answer.get("position_json") else {}
            if not isinstance(current, dict):
                raise ValueError("not an object")
            to_add = [(chart_id, family) for chart_id, family in charts if chart_id not in layout_chart_ids(current)]
            if not to_add:
                return _Outcome(DONE, None)
            positions = layout_positions(to_add, onto=current)
        except ValueError as exc:
            raise _Stop("layout_unreadable", f"{what} is not one mnemiq can add charts to ({exc})") from None
        answer = run.extension("set_layout", {"dashboard_id": dashboard_id, "positions": positions,
                                              "add_chart_ids": [chart_id for chart_id, _ in to_add],
                                              "remove_chart_ids": [], "expect_changed_on": answer["changed_on"]})
        if not answer.get("stale"):
            _check_answer(answer, what)
            return _Outcome(DONE, None)
    raise _Stop("kept_changing", f"dashboard {dashboard_id} kept changing while its layout was saved; save it again "
                                 "later")


def _kinds(spec: ChartSpec) -> dict:
    """Each metric's kind by the label Superset names its values by: a saved metric's name, a written measure's
    label."""
    return {(ref.name if isinstance(ref, SavedRef) else ref.measure.label):
            (ref.kind if isinstance(ref, SavedRef) else ref.measure.kind) for ref in spec.measures}


def _run_check(run: _Run) -> _Outcome:
    """Check each chart's numbers and format (`check_chart`), the lease renewed before each of its calls, and keep each
    chart's status with the dashboard's measures. A chart that does not check is reported as what it is; the build
    still ends."""
    ids = run.chart_ids()
    statuses: dict[int, ChartStatus] = {}
    off: list[str] = []
    for spec in run.plan.charts:
        chart_id = ids[spec.key]
        found = check_chart(run.adapter, chart_id=chart_id, dataset_id=spec.dataset_id, kinds=_kinds(spec),
                            saved=run.saved(spec.dataset_id), snapshot=run.snapshot, grants=run.grants,
                            guard_fanout=run.settings.guard_fanout, before_call=run.renew)
        statuses[chart_id] = found.status
        if found.status is not ChartStatus.CHECKED:
            off.append(f"{spec.title!r} ({found.status.value}: {found.reason})")
    build = run.build
    for row in run.store.measures_for(build.tenant_id, build.dashboard_id):
        mine = {chart: statuses[chart] for chart in row.chart_ids if chart in statuses}
        if mine:
            run.store.put_measure(build.tenant_id, build.dashboard_id, row.measure, chart_ids=row.chart_ids,
                                  made_dataset_ids=row.made_dataset_ids, check_status={**row.check_status, **mine},
                                  saved_metric=row.saved_metric)
    total = len(run.plan.charts)
    message = (f"every chart checked ({total} of {total})" if not off else
               f"{total - len(off)} of {total} charts checked; not checked: " + "; ".join(off))
    return _Outcome(DONE, None, message=message, checks=statuses)


_RUNNERS: dict[str, Callable[[_Run], _Outcome]] = {"dashboard": _run_dashboard, "dataset": _run_dataset,
                                                   "chart": _run_chart, "layout": _run_layout, "check": _run_check}

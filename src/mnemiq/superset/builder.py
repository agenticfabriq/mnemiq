"""A dashboard build, run one step per request: the dashboard, then each dataset the plan makes, each chart, the
layout, and the value checks, in that order.

**One step, claimed first.** `next_step` claims the build's next step in the build store before it calls Superset
(`BuildStore.claim_next_step`), recording the step's intent: its kind, the name it will give what it makes, the dataset,
and, kept by the store, when it began. A second request while the claim's lease holds is refused (`BuildBusy`, a
409), so one request at a time runs a step. The step renews its lease before each of its calls to Superset, inside a
chart's value check too, and starts no call once the lease has passed: it then stops and leaves the step to the next
request. It records its outcome, with a reason code from the store's set when it fails, before it returns.

**Every object a build makes is made by the extension, with its marker** (`mnemiq.superset.marker`) as the last line
of its description, written in the create's own transaction: the dashboard shell (`create_dashboard`), each dataset,
physical or virtual (`create_dataset`), and each chart (`create_chart`, from MCP's own chart config, or a funnel's or a
sankey's form data). A create sent again with the same marker answers with the object the first one made, and makes
none. More than one object carrying the marker stops the step (`candidates`), naming those this builder may read.
Nothing is looked for by its name: an object without the step's marker, or with another, is never the step's.

**A lost answer is sent again at once.** A create whose call ends unanswered (`SupersetUnavailable`, a timeout among
them) leaves its step uncertain, and the next request sends the same create: if Superset made the object, the create
answers with it; if not, it makes it. A first call that Superset runs only after the retry waits for the marker's lock
and then answers with the object the retry made. Every other call a step makes may be sent again too, so a step
whose call went unanswered is run again by the next request, without waiting for its lease to pass. The lease makes
one request at a time run a step; it does not stand for how long Superset may take. A step whose worker ended before
it recorded its outcome still holds its claim, so `next_step` raises `BuildBusy` until that lease passes.

**What a build keeps** is the plan, the step states, the ids of what it made, and each measure of the dashboard (a
saved metric a chart reuses among them) with the charts that show it and each chart's check status. Never a value: a
check's numbers are compared and dropped.

Every Superset call goes through the request's adapter's client, carrying the request's own token.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    config_chart_request,
    extension_chart_request,
)
from mnemiq.superset.client import SupersetToolError, SupersetUnavailable
from mnemiq.superset.extension import call_tool, dashboard_editorship
from mnemiq.superset.layout import layout_chart_ids, layout_positions
from mnemiq.superset.marker import format_marker
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

BUILDING = "building"

# What a later read of a stopped build says for each reason code its step was left with (`build_store.REASONS`).
# `claimed_by_another`, `not_made` and `unreadable` are no longer given; a build that stopped on one before still reads.
_REASON_SAID = {
    "not_editor": "this builder is not an editor of an object the build changes",
    "not_found": "an object the build needs is not one this builder can read",
    "claimed_by_another": "an object the build was to use carries another build's marker",
    "extension_refused": "the extension refused a request of the build",
    "not_saved": "Superset could not save an object of the build",
    "too_large": "an object the build reads is larger than the extension sends",
    "candidates": "Superset holds more than one object carrying the step's marker, and the build cannot tell which",
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
                dashboard_id: int | None = None, profile: str | None = None) -> BuildState:
    """A new build of `plan`, on the existing dashboard `dashboard_id` or a new one, kept with the profile it was
    started under (None: the deployment's default). Refused (ValueError) while the plan still offers a dashboard to
    reuse, which the builder answers first; `DashboardBusy` while another build of the dashboard is building and has
    been touched within `build_store.ABANDONED_AFTER_S`."""
    if plan.awaiting_choice:
        raise ValueError("this plan offers a dashboard to reuse: answer reuse or fresh before it is built")
    stored = store.create_build(tenant, principal, goal, plan_to_json(plan), build_steps(plan),
                                dashboard_id=dashboard_id, profile=profile)
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
    """The step's outcome cannot be settled now (Superset busy, or silent): it is left uncertain, for the next request
    to run again."""


class _LeaseGone(Exception):
    """The step's lease passed, or another claim holds the step: it starts no further call."""


@dataclass
class _Run:
    """The step being run: the build as read, the claim, the plan with each made dataset's id, and what the step has
    made so far (`made`), recorded with any outcome but done."""

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
    create_failed: bool = False
    _saved: dict[int, dict[str, SavedMetric]] = field(default_factory=dict)

    @property
    def client(self):
        return self.adapter.client

    @property
    def key(self) -> str:
        return self.claimed.key

    def renew(self) -> None:
        """Renew the lease before a call; once it cannot be, no further call is started."""
        if not self.store.renew_lease(self.build.tenant_id, self.build.build_id, self.key, self.claimed.lease_id,
                                      self.lease_s):
            raise _LeaseGone()

    def extension(self, base: str, request: dict) -> dict:
        """An extension tool's call, the lease renewed before it and before the probe that may follow it."""
        return call_tool(self.client, base, request, before_call=self.renew)

    def saved(self, dataset_id: int) -> dict[str, SavedMetric]:
        """The dataset's saved metrics, as the builder reads them, by name; read once per step."""
        if dataset_id not in self._saved:
            self.renew()
            self._saved[dataset_id] = {m.name: m for m in read_saved_metrics(self.client, dataset_ids=(dataset_id,))}
        return self._saved[dataset_id]

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
    every Superset call goes through it. A step left uncertain is run again first, its create answering with what an
    earlier send made. Raises `BuildBusy` while another request's step holds the build; a refusal of the caller
    (`SupersetRefusal`) and any failure mnemiq cannot read as an outcome are raised once the step is recorded uncertain.
    A call Superset did not answer leaves the step uncertain, for the next request to run again, as the state returned
    says."""
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
               grants=grants, settings=settings, lease_s=lease_s, made=claimed.object_id)
    try:
        outcome = _RUNNERS[claimed.kind](run)
    except _LeaseGone:
        logger.warning("a build step ran past its lease; it started no further call")
        stored = run.finish_quietly(UNCERTAIN, run.made, "lease_passed")
        return _answer(run, stored, "this step ran past its lease, so it started no further call; the next request "
                                    "runs it again")
    except _Unsure as unsure:
        return _recorded(run, UNCERTAIN, run.made, str(unsure), reason=unsure.reason)
    except _Stop as stop:
        return _recorded(run, FAILED, run.made, str(stop), reason=stop.reason)
    except SupersetUnavailable as exc:
        logger.warning("Superset gave no answer during a build step (%s)", type(exc).__name__)
        return _recorded(run, UNCERTAIN, run.made, "Superset did not answer this step in time, so whether it changed "
                                                   "anything is not known; the next request runs it again",
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
    if any(key in answer for key in ("not_found", "not_editor", "too_large", "refused")):
        return _refused(answer, what)
    return _Stop("layout_unreadable", f"{what} could not be read")


@contextmanager
def _creating(run: _Run, what: str) -> Iterator[None]:
    """A create's call. One that ends unanswered leaves the step uncertain at once: the next request sends the same
    create, which answers with the object if this one made it. A tool that fails on a step it failed on before fails
    the step."""
    try:
        yield
    except SupersetUnavailable as exc:
        logger.warning("Superset gave no answer to a build's create (%s); the next request sends it again",
                       type(exc).__name__)
        raise _Unsure("superset_silent", f"Superset did not answer while making {what}, so whether it made it is not "
                                         "known; the next request sends the same create, which answers with what "
                                         "this one made, if it made anything") from None
    except SupersetToolError:
        if run.claimed.reason == "tool_error":
            raise _Stop("tool_error", f"Superset's tool failed again making {what}; its log says why") from None
        run.create_failed = True
        raise


def _created(run: _Run, tool: str, request: dict, *, made: str, what: str, exists: str | None = None) -> int:
    """The id the extension's create `tool` answers under `made`: the object it made for this step, or the one
    carrying the step's marker that an earlier send of it made. Busy leaves the step uncertain, for the same create to
    be sent again. More than one object carrying the marker stops it (`_candidates`), and so does, for a dataset, a
    table that already has one (the sentence `exists`); any other answer stops it as `_refused` reads it."""
    with _creating(run, what):
        answer = run.extension(tool, request)
    found = answer.get(made)
    if type(found) is int and found > 0:
        run.made = found
        return found
    if answer.get("busy"):
        raise _Unsure("busy", f"Superset was busy with {what}; the next request sends the same create again")
    if "candidates" in answer:
        raise _candidates(answer["candidates"], what)
    if answer.get("exists") and exists is not None:
        raise _Stop("dataset_exists", exists)
    raise _refused(answer, what)


def _candidates(ids: object, what: str) -> _Stop:
    """More than one object carries the step's marker, so the build cannot tell which is its own: the ones this
    builder may read are named, and none is, when it may read none of them."""
    named = [str(item) for item in ids] if isinstance(ids, list) else []
    if not named:
        return _Stop("candidates", f"Superset holds more than one object carrying the marker of {what}, and this "
                                   "builder cannot read any of them, so this build cannot tell which is its own; ask "
                                   "an editor of them to remove the ones that are not wanted, and start the build "
                                   "again")
    return _Stop("candidates", f"Superset holds more than one object carrying the marker of {what}, so this build "
                               f"cannot tell which is its own; of them, this builder can read {', '.join(named)}; "
                               "remove the ones that are not wanted and start the build again")


def _marker(build: StoredBuild, key: str, dashboard_id: int | None) -> str:
    return format_marker(build.build_id, key, dashboard_id)


# ---------------------------------------------------------------- the steps


def _run_dashboard(run: _Run) -> _Outcome:
    build = run.build
    if build.dashboard_id is not None:
        return _existing_dashboard(run, build.dashboard_id)
    made = _created(run, "create_dashboard", {"title": _title(build.goal), "marker": _marker(build, "dashboard", None)},
                    made="dashboard_id", what="the dashboard")
    return _Outcome(DONE, made)


def _existing_dashboard(run: _Run, dashboard_id: int) -> _Outcome:
    """An existing dashboard: whether the builder edits it, its layout and its `changed_on`, read now, before anything
    is made, and nothing held across steps; each later write takes the dashboard's row lock itself, and the layout step
    reads it afresh. What this read alone can refuse is refused here, so no dataset or chart is made that the build
    then cannot place or keep: a layout mnemiq cannot add rows to (one whose charts sit in top-level tabs among them),
    and a measure the plan writes that the dashboard keeps under its name with another definition."""
    what = f"dashboard {dashboard_id}"
    entry = dashboard_editorship(run.client, dashboard_id, before_call=run.renew)
    if entry is None:
        raise _Stop("not_found", f"{what}: Superset has no such dashboard that this builder can read")
    # `is_editor` is kept when only the editors' labels were too many to send.
    if entry.get("is_editor") is not True:
        if entry.get("is_editor") is not False:
            raise _Stop("too_large", f"whether this builder edits {what} was not answered whole")
        raise _refused({"not_editor": True, "editors": entry.get("editors")}, what)
    answer = run.extension("get_layout", {"dashboard_id": dashboard_id})
    if answer.get("dashboard_id") != dashboard_id or not isinstance(answer.get("changed_on"), str):
        raise _read_refusal(answer, f"the layout of {what}")
    try:
        layout_positions([], onto=_layout_of(answer))
    except ValueError as exc:
        raise _Stop("layout_unreadable", f"the layout of {what} is not one mnemiq can add charts to ({exc})") from None
    _held_to_the_dashboards_measures(run, run.plan.charts, stops="nothing of this build is made")
    return _Outcome(DONE, dashboard_id)


def _layout_of(answer: Mapping) -> dict:
    """The layout `get_layout` answered, as an object (an empty one for a dashboard that has none); ValueError when
    it is not one."""
    current = json.loads(answer["position_json"]) if answer.get("position_json") else {}
    if not isinstance(current, dict):
        raise ValueError("not an object")
    return current


def _run_dataset(run: _Run) -> _Outcome:
    """The dataset the plan makes: over its table, or, for a virtual one, over its SQL, named for the build; its
    editors set by the same create."""
    build, planned, adapter = run.build, _planned(run.plan, run.key), run.adapter
    name, dashboard_id = _dataset_name(build, planned), build.dashboard_id
    request = {"database_id": adapter.database_id, "schema": adapter.schema, "table_name": name,
               "dashboard_id": dashboard_id, "marker": _marker(build, planned.key, dashboard_id)}
    if planned.virtual:
        request.update(sql=planned.sql, description=f"Made by mnemiq for dashboard {dashboard_id}.")
    made = _created(run, "create_dataset", request, made="dataset_id", what=f"the dataset {name!r}",
                    exists=f"Superset already has a dataset named {name!r} on this schema, which this builder cannot "
                           "use; the build uses only datasets it can read or makes")
    return _Outcome(DONE, made)


def _run_chart(run: _Run) -> _Outcome:
    """The chart, made by the extension with its editors, its marker, its query context and its whole format: a funnel
    or a sankey from mnemiq's own form data, any other family from MCP's chart config."""
    build, spec = run.build, _chart_spec(run.plan, run.key)
    dashboard_id = build.dashboard_id
    measures = {measure.name: measure for measure in run.plan.measures}
    saved = run.saved(spec.dataset_id) if any(isinstance(ref, SavedRef) for ref in spec.measures) else {}
    currency = run.settings.superset_currency
    _held_to_the_dashboards_measures(run, (spec,), stops=f"the chart {spec.title!r} is not made")
    requested = extension_chart_request if spec.family in EXTENSION_FAMILIES else config_chart_request
    request = {**requested(spec, measures, saved, currency=currency), "dashboard_id": dashboard_id,
               "marker": _marker(build, spec.key, dashboard_id)}
    chart_id = _created(run, "create_chart", request, made="chart_id", what=f"the chart {spec.title!r}")
    _keep_measures(run, spec, chart_id, saved)
    return _Outcome(DONE, chart_id)


def _same_definition(kept: Measure, planned: Measure) -> bool:
    """One measure: its kind and dataset the same, its expression the same once normalized (as written, when it does
    not parse)."""
    if kept.kind != planned.kind or kept.dataset_id != planned.dataset_id:
        return False
    try:
        return normalized(kept.expression) == normalized(planned.expression)
    except SqlglotError:
        return kept.expression.strip() == planned.expression.strip()


def _held_to_the_dashboards_measures(run: _Run, specs: Sequence[ChartSpec], *, stops: str) -> None:
    """Before what `stops` names is made (every chart, at the first step; one chart, at its own): a measure the charts
    write that the dashboard already keeps under its name with another definition stops the step, since one name
    would then mean two measures on one dashboard. At the first step a measure on a dataset the build has yet to make
    carries the dataset's placeholder id, which is no kept measure's dataset; nor will the dataset's id be, once it is
    made, since Superset gives a new dataset an id no dataset had before."""
    build = run.build
    kept = {row.measure.name: row for row in run.store.measures_for(build.tenant_id, build.dashboard_id)}
    for spec in specs:
        for ref in spec.measures:
            if isinstance(ref, WrittenRef) and (before := kept.get(ref.measure.name)) is not None \
                    and not _same_definition(before.measure, ref.measure):
                raise _Stop("measure_conflict", f"the dashboard keeps a measure {ref.measure.name!r} with another "
                                                f"definition, so {stops}; give the planned measure another name, or "
                                                "change the one the dashboard keeps")


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
            current = _layout_of(answer)
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
        # The build made the chart with these measures alone: a metric of any other label is one whose format
        # nobody judged (Superset saved its label as other text, or someone added it since), never one to pass over.
        found = check_chart(run.adapter, chart_id=chart_id, dataset_id=spec.dataset_id, kinds=_kinds(spec),
                            saved=run.saved(spec.dataset_id), snapshot=run.snapshot, grants=run.grants,
                            guard_fanout=run.settings.guard_fanout, before_call=run.renew, every_kind_known=True)
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

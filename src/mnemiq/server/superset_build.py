"""Superset mode's build routes: a builder's goal to a checked plan and a build of it, the build run one step per
request, and the build as it stands; and a dashboard's measures, listed with each value computed for the person
asking, and edited, removed or saved to their dataset (`measure_actions`). Mounted only in Superset mode, and no other
route under `/v1/superset/`.

- `POST /v1/superset/builds` takes `{goal, dashboard_id?, fresh?, profile?}`. Asked with neither `dashboard_id` nor
  `fresh`, the plan may offer a dashboard the builder can see that already covers the goal: the answer is then the
  offer, its id, title and reason, with mnemiq's words on each answer to it, and nothing is built or kept. Reusing
  the dashboard is opening it, which needs no build. Asking again with its `dashboard_id` adds this goal's planned
  charts to it (a `dashboard_id` always does that, to the dashboard it names), and with `fresh: true` builds a new
  dashboard; asked either way, no dashboard is looked for. Onto a dashboard, the plan leaves out each chart the
  dashboard already shows (`dashboard_charts.leave_out_shown`), and the answer names each one beside the build
  (`left_out`, each `part`, `chart_id` and `title`); a goal it shows whole builds nothing and keeps nothing, and is
  answered as `shown` (200) with those parts and mnemiq's words. Otherwise the answer is the new build (201), or the
  plan's refusal, which is an answer too. The build keeps the profile it was asked under (`profile`, None for the
  deployment's default). Before anything is planned or kept, Superset is asked whether the person may build (the
  grants; a non-builder is refused as one, `NOT_A_BUILDER`, and nothing about a dashboard they named is read), then
  whether the extension is there and offers every create a build sends (`require_build_tools`: a missing one and an
  older one are each refused with a sentence of their own), then, for a `dashboard_id`, whether they edit that
  dashboard (the extension's `editors_of`): one they cannot read, or that does not exist, is a 404 that does not say
  which, and one they read but do not edit a 403 naming its editors. A model that does not answer the planning prompt
  is a 503 saying to try again; a prompt its server refuses or cuts as too long is a refused plan, since the goal and
  the schema are too large to plan in one prompt.
- `GET /v1/superset/builds/{build_id}`: the build as it stands, read from the build store alone.
- `GET /v1/superset/builds?dashboard_id=`: the person's builds still building, newest first (with `dashboard_id`,
  only that dashboard's), each `build_id`, `goal`, `dashboard_id`, `status`, `created_at`, `updated_at`, `steps_done`
  and `steps_total`, read from the build store alone: what a client offers to finish.
- `POST /v1/superset/builds/{build_id}/next`: the build's next step, claimed, run and recorded (`builder.next_step`);
  409 while another request's step holds the build.

**The door.** Every route takes the caller the `/v1` door verified (`asker`), and so needs the token's `db.read`. The
two `POST`s also need Superset's write scopes (`WRITE_SCOPES`), which a token Superset's mint route makes for building
carries and one it makes for asking does not (`NotForBuilding`, a 403). These routes write Superset's metadata,
through Superset as the builder, never the source database, so they need no mnemiq scope of their own.

**As the builder.** A route that calls Superset builds its own adapter with `superset_executor(settings, delegation)`
from the deployment's settings, which name Superset, and the request's verified token; the planner and the builder
call Superset only through that adapter's client. What the builder may read is what Superset says when asked as them,
read once per request. The build `GET`s call nothing.

**Under its profile.** A build's plan, and each of its steps, run under the settings of the profile it was started
under, picked again at every step as `/v1/ask` picks one: a profile gone or no longer offered to the builder refuses
the step (422) before Superset is asked anything.

**A dashboard's measures.** `GET /v1/superset/dashboards/{id}/measures` needs `db.read` and Superset's read scopes
for the dashboard's layout and charts (`READ_SCOPES`), which a token minted for building carries; each action
(`POST .../measures/{name}/edit`, `remove`, `save`) needs the write scopes too, and is asked only once Superset said the
person edits the dashboard, as a build onto it is. A measure as `_measure` writes it, an action's outcome as `_result`
writes it.

**What is answered.** A build as `_build` writes it, and nothing else: `build_id`, `status` and `dashboard_id`; its
`steps`, each `key`, `kind`, `status`, `object_id` (what the step made), `started_at` and `lease_until` (when a running
step's lease passes); `checks`, each chart's check status by chart id; `ran`, the step this request ran, and `message`,
mnemiq's own words on it; `plan`, the build's plan as the person is shown it (`builder.plan_summary`: each chart's
key, title, family and measures, each written measure's definition); and `check_reasons`, why each chart the check
step did not check did not, in that step's own answer only. Never a lease id, a step's intent, or a value. Every
failure is a status and mnemiq's own sentence (`server/superset_refusals.py`, and `_refusal` here for the build's
own).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mnemiq.llm.client import ModelUnavailable
from mnemiq.profiles import UnknownProfile
from mnemiq.server.superset_refusals import NotForBuilding
from mnemiq.server.v1_refusals import NotPermitted, refused, unknown_profile
from mnemiq.superset.build_store import (
    MAX_ID,
    PROPOSED,
    BuildNotFound,
    BuildStore,
    BuildStoreMissing,
    DashboardBusy,
    StoredBuild,
)
from mnemiq.superset.builder import BuildBusy, BuildState, build_state, next_step, start_build
from mnemiq.superset.chart_change import ChangeAnswer, change_dashboard
from mnemiq.superset.client import SupersetToolError
from mnemiq.superset.dashboard_charts import LeftOut, datasets_built_on
from mnemiq.superset.executor import SupersetCredentialMissing, superset_executor
from mnemiq.superset.extension import ExtensionMissing, call_tool, dashboard_editorship, require_build_tools
from mnemiq.superset.measure_actions import (
    ActionRefused,
    ActionResult,
    Asking,
    ChartsChanged,
    MeasureNotFound,
    MeasureView,
    edit_measure,
    list_measures,
    remove_measure,
    save_measure,
)
from mnemiq.superset.measure_actions import DashboardUnreadable as _MeasuresUnreadable
from mnemiq.superset.measures import MAX_LABEL, MeasureKind
from mnemiq.superset.planner import MAX_DESCRIPTION, MAX_GOAL, BuildPlan, PlanContext, PlanRefused, plan_build

__all__ = ["READ_SCOPES", "WRITE_SCOPES", "BuildAlreadyStarted", "BuildKit", "PlanStale", "confirm_proposed_build",
           "mount_superset_build"]

logger = logging.getLogger(__name__)

# What a build route needs beside `db.read`: Superset's own scopes for the writes a build makes, which are
# Superset's metadata, never the source database, so no mnemiq scope of their own.
WRITE_SCOPES = ("superset:chart:write", "superset:dashboard:write", "superset:dataset:write")
_NEEDED = ("db.read", *WRITE_SCOPES)
# What reading a dashboard's measures needs beside `db.read`: Superset's own scopes for its layout and its charts,
# which a token Superset mints for building carries and one minted for asking does not.
READ_SCOPES = ("superset:chart:read", "superset:dashboard:read")
_NEEDED_TO_READ = ("db.read", *READ_SCOPES)

NO_DELEGATION = ("Superset mode builds as the person whose verified token each request carries, and this request "
                 "carries none")
STORE_UNAVAILABLE = "mnemiq's control Postgres, where builds are kept, did not answer; try this request again."
# A dashboard id that names none this builder can read: whether it exists is not said.
NO_SUCH_DASHBOARD = "This builder can read no dashboard of that id, so nothing was planned."
# The model did not answer the planning prompt: an outage, answered as the store's and the extension's are, so a client
# tells it from a goal that cannot be planned (a refused plan) by its status alone.
MODEL_UNAVAILABLE = ("The model did not answer, so no plan was made and nothing was kept; this is an outage, not a "
                     "judgement about the goal, so try this request again.")


class DashboardUnreadable(Exception):
    """`dashboard_id` names no dashboard the builder can read: none of that id, or one they may not read."""


class PlanStale(Exception):
    """The dashboard a proposed build was planned onto changed since its plan, so the plan is not confirmed."""

    def __init__(self) -> None:
        super().__init__("The dashboard changed since this plan; plan it again.")


class BuildAlreadyStarted(Exception):
    """A confirm of a build that is no longer a proposal: confirmed already, from another tab or a second click."""

    def __init__(self) -> None:
        super().__init__("This build was already started.")


class NotAnEditor(Exception):
    """The builder reads the dashboard but does not edit it. `editors` are their labels, for the HTTP answer only."""

    def __init__(self, dashboard_id: int, editors: list) -> None:
        named = f"; its editors are {', '.join(str(label) for label in editors)}" if editors else ""
        super().__init__(f"This builder is not an editor of dashboard {dashboard_id}{named}.")


def _changed_on(client, dashboard_id: int) -> str:
    """The dashboard's `changed_on`, as the extension's `get_layout` answers it to the builder: what a proposal is
    planned against and read again at its confirm. DashboardUnreadable when they cannot read it."""
    answer = call_tool(client, "get_layout", {"dashboard_id": dashboard_id})
    if answer.get("dashboard_id") != dashboard_id or not isinstance(answer.get("changed_on"), str):
        raise DashboardUnreadable()
    return answer["changed_on"]


def confirm_proposed_build(asking, builds, build_id: str, *, runtime) -> StoredBuild:
    """Start the person's proposed build: the one confirm the confirm route and a conversation's confirm make. A build
    onto a dashboard is confirmed only while the dashboard's `changed_on` is the one its plan was made against
    (`PlanStale` otherwise), read once the builder's session is shown bound on Oracle (`Runtime.require_bound`);
    `DashboardBusy` while another build of it is building, the proposal kept; `BuildAlreadyStarted` once it is no
    longer a proposal; BuildNotFound unless it is the person's."""
    who, delegation = _builder(asking, building=True)
    build = builds.load_build(who.tenant_id, build_id, who.principal_id)
    if build is None:
        raise BuildNotFound()
    if build.status != PROPOSED:
        raise BuildAlreadyStarted()
    if build.dashboard_id is not None:
        adapter = superset_executor(runtime.settings, delegation)
        runtime.require_bound(adapter)
        if _changed_on(adapter.client, build.dashboard_id) != build.planned_against:
            raise PlanStale()
    try:
        return builds.confirm_build(who.tenant_id, build.build_id, who.principal_id)
    except BuildNotFound:
        # Confirmed meanwhile, from another tab: said as a second confirm is.
        again = builds.load_build(who.tenant_id, build.build_id, who.principal_id)
        if again is not None and again.status != PROPOSED:
            raise BuildAlreadyStarted() from None
        raise


def _held_to_its_editors(client, dashboard_id: int) -> None:
    """Return when the builder edits the dashboard, as the extension's `editors_of` answers it (the read the build's
    first step makes again, since its editors can change between steps); raise when they cannot read it or do not
    edit it. Asked once Superset has said the person may build, and before anything is planned or kept, so an id no
    dashboard has yet, or one the builder may not change, takes no dashboard's one live build, and the answer says
    nothing of a build running on it."""
    entry = dashboard_editorship(client, dashboard_id)
    if entry is None:
        raise DashboardUnreadable()
    if entry.get("is_editor") is True:
        return
    if entry.get("is_editor") is False:
        raise NotAnEditor(dashboard_id, entry.get("editors") or [])
    raise SupersetToolError("the extension's editors_of did not say whether this builder edits the dashboard",
                            detail=str(entry)[:200])


class StartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: Annotated[StrictStr, Field(min_length=1, max_length=MAX_GOAL)]
    # The dashboard to add this goal's planned charts to (the one a plan offered, or another the builder names); an id
    # the store can keep.
    dashboard_id: Annotated[StrictInt, Field(gt=0, le=MAX_ID)] | None = None
    # A new dashboard, whatever a plan would offer.
    fresh: StrictBool = False
    profile: StrictStr | None = None  # None: the deployment's default
    # Plan it and keep it as a proposal, writing nothing to Superset until the person confirms it.
    propose: StrictBool = False

    @model_validator(mode="after")
    def _one_answer(self) -> StartBody:
        if self.fresh and self.dashboard_id is not None:
            raise ValueError("name a dashboard to add charts to, or ask for a fresh one, not both")
        return self


class EditBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Annotated[StrictStr, Field(min_length=1, max_length=MAX_LABEL)]
    expression: Annotated[StrictStr, Field(min_length=1, max_length=4000)]
    kind: MeasureKind
    currency: Annotated[StrictStr, Field(pattern=r"^[A-Z]{3}$")] | None = None
    description: Annotated[StrictStr, Field(max_length=MAX_DESCRIPTION)] = ""


class RemoveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The charts the person was shown, which must be the measure's charts now.
    chart_ids: Annotated[list[Annotated[StrictInt, Field(gt=0, le=MAX_ID)]], Field(max_length=250)]


class SaveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


# A measure's name in a path, as the store keeps it: a written measure's name, or a reused saved metric's
# `<metric>@<dataset id>`.
_MEASURE_NAME = r"^[^/]{1,300}$"


def _measure(view: MeasureView) -> dict:
    """A measure as the measures route answers it, every field named here."""
    m = view.measure
    return {"name": m.name, "label": m.label, "expression": m.expression, "kind": str(m.kind),
            "currency": m.currency, "description": m.description, "asked_by": m.asked_by, "dataset_id": m.dataset_id,
            "saved_metric": view.saved_metric, "value": view.value, "value_note": view.value_note,
            "value_basis": str(view.value_basis) if view.value_basis else None, "value_chart_id": view.value_chart_id,
            "charts": [{"chart_id": c.chart_id, "title": c.title, "check": str(c.check) if c.check else None}
                       for c in view.charts]}


def _result(result: ActionResult) -> dict:
    return {"changed": list(result.changed), "left_by_hand": list(result.left_by_hand),
            "not_editor": [{**({"chart_id": e.chart_id} if e.chart_id is not None else {}),
                            **({"dataset_id": e.dataset_id} if e.dataset_id is not None else {}),
                            "editors": list(e.editors)} for e in result.not_editor],
            "blocked_by_report": list(result.blocked_by_report), "could_not_read": list(result.could_not_read),
            "message": result.message}


class ChangeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: Annotated[StrictStr, Field(min_length=1, max_length=MAX_GOAL)] | None = None
    # The charts the person confirmed removing, from a change's `would_remove`: removed, and nothing else done.
    remove_chart_ids: Annotated[list[Annotated[StrictInt, Field(gt=0, le=MAX_ID)]],
                                Field(min_length=1, max_length=250)] | None = None

    @model_validator(mode="after")
    def _one_ask(self) -> ChangeBody:
        if (self.request is None) == (self.remove_chart_ids is None):
            raise ValueError("send a request, or the charts confirmed for removal, not both")
        return self


def _changed(answer: ChangeAnswer) -> dict:
    """A change's outcome as the changes route answers it, every field named here."""
    return {"changed": list(answer.changed),
            "left": [{"chart_id": chart_id, "reason": reason} for chart_id, reason in answer.left],
            "dropped": [{"chart_id": chart_id, "key": key} for chart_id, key in answer.dropped],
            "not_editor": [{"chart_id": e.chart_id, "editors": list(e.editors)} for e in answer.not_editor],
            "removed": list(answer.removed),
            "would_remove": [{"chart_id": chart_id, "title": title} for chart_id, title in answer.would_remove],
            "added": None, "message": answer.message}


class _Offering:
    """The profile's planner, keeping each dashboard it was shown by id and title, so an offer is answered with the
    title the planner chose it by."""

    def __init__(self, planner) -> None:
        self._planner = planner
        self.titles: dict[int, str] = {}

    def plan(self, goal: str, context: PlanContext):
        self.titles.update({dashboard.id: dashboard.title for dashboard in context.dashboards})
        return self._planner.plan(goal, context)


def _open_store(settings) -> BuildStore:  # seam for tests
    return BuildStore.for_settings(settings)


def _build(state: BuildState) -> dict:
    """A build as a route answers it. Each step's fields are named here, so no other field of the store's, its lease
    id or its intent, can reach an answer."""
    return {
        "build_id": state.build_id,
        "status": state.status,
        "dashboard_id": state.dashboard_id,
        "steps": [{"key": step.key, "kind": step.kind, "status": step.status, "object_id": step.object_id,
                   "started_at": step.started_at, "lease_until": step.lease_until} for step in state.steps],
        "checks": {str(chart_id): str(status) for chart_id, status in state.checks.items()},
        "ran": state.ran,
        "message": state.message,
        "plan": state.plan,
        "check_reasons": {str(chart_id): reason for chart_id, reason in state.check_reasons.items()},
    }


def _offer(plan: BuildPlan, titles: dict[int, str]) -> dict:
    """The offer of a dashboard that already covers the goal, and mnemiq's words on each answer to it: reusing it is
    opening it, which builds nothing; `dashboard_id` adds this goal's charts to it; `fresh` builds a new one."""
    [dashboard_id] = plan.reuse_candidates  # the plan check holds an offer to one dashboard, one the planner was shown
    message = (f"Dashboard {dashboard_id} already covers this goal, so nothing was built: to reuse it, open it. To add "
               f"this goal's charts to it anyway, ask again with dashboard_id {dashboard_id}; for a new dashboard, ask "
               "again with fresh true.")
    return {"offer": {"dashboard_id": dashboard_id, "title": titles.get(dashboard_id), "reason": plan.reuse_reason,
                      "message": message}}


def _left_out(left: tuple[LeftOut, ...]) -> list[dict]:
    return [{"part": item.part, "chart_id": item.chart_id, "title": item.title} for item in left]


def _shown(dashboard_id: int, left: tuple[LeftOut, ...]) -> dict:
    """A goal the dashboard already shows whole: each part and the chart that shows it, and mnemiq's words; nothing
    was built or kept."""
    parts = "; ".join(f"{item.part!r} is its chart {item.chart_id}" for item in left)
    return {"shown": {"dashboard_id": dashboard_id, "left_out": _left_out(left),
                      "message": f"Dashboard {dashboard_id} already shows every chart this goal plans, so nothing was "
                                 f"built: {parts}."}}


def _refused_plan(plan: PlanRefused) -> dict:
    # mnemiq's reason and each part as the model named it; never the model's draft.
    return {"refused": {"reason": plan.reason, "parts": [{"part": part.part, "reason": part.reason}
                                                         for part in plan.parts]}}


def _builder(asking, *, building: bool, reading: bool = False):
    """The verified caller and their delegated token, or the refusal: no token (401), or, for a route that builds, a
    token that lacks a write scope (403), and for one that reads a dashboard's charts, one that lacks a read scope."""
    delegation = asking.handed.get("delegation")
    if delegation is None:
        raise SupersetCredentialMissing(NO_DELEGATION)
    if building and not set(WRITE_SCOPES) <= asking.scopes:
        raise NotForBuilding(_NEEDED)
    if reading and not set(READ_SCOPES) <= asking.scopes:
        raise NotForBuilding(_NEEDED_TO_READ)
    return asking.identity, delegation


def _refusal(exc: Exception, doors) -> HTTPException | None:
    """The response a failure of a build route becomes: the build's own here, Superset's and Superset mode's as every
    `/v1` route answers them (`v1_refusals.refused`). None for anything else, which stays a 500."""
    if isinstance(exc, (BuildBusy, DashboardBusy, PlanStale, BuildAlreadyStarted)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (BuildNotFound, MeasureNotFound)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, _MeasuresUnreadable):
        return HTTPException(status_code=404, detail=NO_SUCH_DASHBOARD)
    if isinstance(exc, ChartsChanged):
        return HTTPException(status_code=409, detail={
            "message": str(exc), "charts": [{"chart_id": c.chart_id, "title": c.title,
                                             "check": str(c.check) if c.check else None} for c in exc.charts]})
    if isinstance(exc, ActionRefused):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, DashboardUnreadable):
        return HTTPException(status_code=404, detail=NO_SUCH_DASHBOARD)
    if isinstance(exc, NotAnEditor):  # Superset's answer, in mnemiq's words: a 403 like every `/v1` one
        return NotPermitted(str(exc), None, doors.end_session_url if doors is not None else None)
    if isinstance(exc, psycopg.OperationalError):
        logger.warning("the build store did not answer a build route: %s", exc)
        return HTTPException(status_code=503, detail=STORE_UNAVAILABLE)
    if isinstance(exc, ModelUnavailable):  # a prompt too long to plan is a refused plan: only an outage reaches here
        logger.warning("the model did not answer a build's planning prompt (%s): %s", exc.kind, exc)
        return HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)
    if isinstance(exc, UnknownProfile):  # as `/v1/ask` answers it
        return unknown_profile(exc)
    if isinstance(exc, ExtensionMissing):
        logger.warning("%s", exc)
    return refused(exc, doors)


@dataclass(frozen=True)
class BuildKit:
    """What the panel's conversation routes share with the build routes, so a build or a change asked in a
    conversation is planned, checked and refused exactly as the build routes do it: the build store (`kept`), a goal
    planned and started or proposed (`started`), who asks about a dashboard (`asking`), and a failure answered as the
    build routes answer it (`answered`)."""

    kept: Callable[[], Any]
    started: Callable[..., tuple[int, dict]]
    asking: Callable[..., Asking]
    answered: Callable[[Callable[[], Any]], Any]
    deployment: Any


def mount_superset_build(app: FastAPI, runtime, doors, asker) -> BuildKit:
    """Mount the build, measures and changes routes on `app`, each behind `asker`, the `/v1` door's dependency. The build store is
    the deployment's control Postgres (`MNEMIQ_CONTROL_DSN`); without one, every build route answers 503 naming it."""
    # The deployment's settings name Superset (its MCP URL, database and schema), so every adapter is built from them;
    # a build's plan and steps run under the settings of the profile it was started under.
    deployment = runtime.settings
    try:
        store = _open_store(deployment)
    except BuildStoreMissing:
        logger.warning("MNEMIQ_CONTROL_DSN is not set, so every build route answers that building needs it")
        store = None

    def kept() -> Any:
        if store is None:
            raise BuildStoreMissing()
        return store

    def answered(work):
        try:
            return work()
        except Exception as exc:
            if (answer := _refusal(exc, doors)) is None:
                raise
            raise answer from None

    def started(asking_, goal: str, *, dashboard_id: int | None, fresh: bool, profile_name: str | None,
                propose: bool = False) -> tuple[int, dict]:
        """A goal planned and, when its plan is one, started as a build: the status and the answer, as the start route
        gives them. A change asked by chat that adds a chart runs it too, onto its dashboard. Proposed, the build is
        kept waiting for its confirm and nothing is written to Superset: a 200 naming it and its plan."""
        who, delegation = _builder(asking_, building=True)
        builds = kept()
        profile = runtime.build_profile(profile_name, who)
        planner = _Offering(profile.planner)
        adapter = superset_executor(deployment, delegation)
        # Whether Superset lets this person build at all, first: a non-builder is refused as one (NOT_A_BUILDER)
        # and learns nothing of a dashboard they named, its editors included.
        grants, served = runtime.builder_grants(adapter)
        # Before anything is planned or kept: every write of a build goes through the extension, which must be
        # one that makes every object a build makes.
        require_build_tools(adapter.client)
        planned_against = None
        if dashboard_id is not None:
            _held_to_its_editors(adapter.client, dashboard_id)
            if propose:  # read before planning, so a change made while it plans is one its confirm sees
                planned_against = _changed_on(adapter.client, dashboard_id)
        # The measures the dashboard keeps, read once: the datasets its charts are on, and the names and labels a
        # plan onto it leaves to them.
        on_it = builds.measures_for(who.tenant_id, dashboard_id) if dashboard_id is not None else []
        built_on = datasets_built_on(on_it) if dashboard_id is not None else None
        plan = plan_build(goal, planner=planner, client=adapter.client, adapter=adapter,
                          snapshot=served, grants=grants, settings=profile.settings,
                          offer_reuse=dashboard_id is None and not fresh, onto=dashboard_id, built_on=built_on,
                          kept=tuple(row.measure for row in on_it))
        if isinstance(plan, PlanRefused):
            return 200, _refused_plan(plan)
        if plan.awaiting_choice:
            return 200, _offer(plan, planner.titles)
        if dashboard_id is not None and not plan.charts:
            return 200, _shown(dashboard_id, plan.left_out)
        state = start_build(builds, tenant=who.tenant_id, principal=who.principal_id, goal=goal, plan=plan,
                            dashboard_id=dashboard_id, profile=profile_name, proposed=propose,
                            planned_against=planned_against)
        if propose:
            proposal = {"build_id": state.build_id, "goal": goal, "dashboard_id": dashboard_id, "plan": state.plan}
            if plan.left_out:
                proposal["left_out"] = _left_out(plan.left_out)
            return 200, {"proposal": proposal}
        answer = {"build": _build(state)}
        if plan.left_out:
            answer["left_out"] = _left_out(plan.left_out)
        return 201, answer

    @app.post("/v1/superset/builds", status_code=201)
    def start(body: StartBody, response: Response, asking=Depends(asker)) -> dict:
        def work() -> dict:
            status, answer = started(asking, body.goal, dashboard_id=body.dashboard_id, fresh=body.fresh,
                                     profile_name=body.profile, propose=body.propose)
            response.status_code = status
            return answer

        return answered(work)

    @app.get("/v1/superset/builds")
    def unfinished(dashboard_id: Annotated[int | None, Query(gt=0, le=MAX_ID)] = None,
                   asking=Depends(asker)) -> dict:
        """The person's builds still building, newest touched first, from the build store alone: what a client
        offers to finish. With `dashboard_id`, only that dashboard's. Calls no Superset tool."""

        def work() -> dict:
            who, _ = _builder(asking, building=False)
            listed = kept().unfinished_builds(who.tenant_id, who.principal_id, dashboard_id=dashboard_id)
            return {"builds": [{"build_id": build.build_id, "goal": build.goal, "dashboard_id": build.dashboard_id,
                                "status": build.status, "created_at": build.created_at,
                                "updated_at": build.updated_at,
                                "steps_done": sum(1 for step in build.steps if step.status == "done"),
                                "steps_total": len(build.steps)} for build in listed]}

        return answered(work)

    @app.get("/v1/superset/builds/{build_id}")
    def read(build_id: str, asking=Depends(asker)) -> dict:
        def work() -> dict:
            who, _delegation = _builder(asking, building=False)
            builds = kept()
            build = builds.load_build(who.tenant_id, build_id, who.principal_id)
            if build is None:
                raise BuildNotFound()
            return {"build": _build(build_state(builds, build))}

        return answered(work)

    @app.post("/v1/superset/builds/{build_id}/next")
    def step(build_id: str, asking=Depends(asker)) -> dict:
        """The build's next step. The builder's grants are read before `next_step` claims the step, so a person
        Superset refuses leaves no claimed step behind; those reads change nothing, and the claim still comes before
        every write."""

        def work() -> dict:
            who, delegation = _builder(asking, building=True)
            builds = kept()
            build = builds.load_build(who.tenant_id, build_id, who.principal_id)
            if build is None:
                raise BuildNotFound()
            profile = runtime.build_profile(build.profile, who)
            adapter = superset_executor(deployment, delegation)
            grants, served = runtime.builder_grants(adapter)
            return {"build": _build(next_step(build, client=adapter.client, adapter=adapter, store=builds,
                                              snapshot=served, grants=grants, settings=profile.settings))}

        return answered(work)

    @app.post("/v1/superset/builds/{build_id}/confirm")
    def confirm(build_id: str, asking=Depends(asker)) -> dict:
        """Start the person's proposed build, the person's "Build it", answered as `/next` answers a build. The panel
        then runs its steps with `/next`, as for any build."""

        def work() -> dict:
            _builder(asking, building=True)
            builds = kept()
            stored = confirm_proposed_build(asking, builds, build_id, runtime=runtime)
            return {"build": _build(build_state(builds, stored))}

        return answered(work)

    def asking(asking_, dashboard_id: int, *, changing: bool) -> Asking:
        """Who asks about dashboard `dashboard_id`, once Superset said they may build and, for a change, that they
        edit the dashboard (its editors named when they do not)."""
        who, delegation = _builder(asking_, building=changing, reading=True)
        builds = kept()
        profile = runtime.build_profile(None, who)
        adapter = superset_executor(deployment, delegation)
        grants, served = runtime.builder_grants(adapter)
        if changing:
            require_build_tools(adapter.client)
            _held_to_its_editors(adapter.client, dashboard_id)
            # A build keeps the dashboard's measures as it last writes them: none is changed while one builds on it.
            if builds.dashboard_building(who.tenant_id, dashboard_id):
                raise DashboardBusy()
        return Asking(adapter=adapter, store=builds, tenant=who.tenant_id, principal=who.principal_id,
                      dashboard_id=dashboard_id, snapshot=served, grants=grants, settings=profile.settings)

    @app.get("/v1/superset/dashboards/{dashboard_id}/measures")
    def measures(dashboard_id: Annotated[int, Path(gt=0, le=MAX_ID)], asking_=Depends(asker)) -> dict:
        """The dashboard's measures, each value computed now as the person asking and kept nowhere."""
        return answered(lambda: {"measures": [_measure(view) for view in
                                              list_measures(asking(asking_, dashboard_id, changing=False))]})

    @app.post("/v1/superset/dashboards/{dashboard_id}/measures/{name}/edit")
    def edit(dashboard_id: Annotated[int, Path(gt=0, le=MAX_ID)],
             name: Annotated[str, Path(pattern=_MEASURE_NAME)], body: EditBody, asking_=Depends(asker)) -> dict:
        return answered(lambda: _result(edit_measure(asking(asking_, dashboard_id, changing=True), name,
                                                     label=body.label, expression=body.expression, kind=body.kind,
                                                     currency=body.currency, description=body.description)))

    @app.post("/v1/superset/dashboards/{dashboard_id}/measures/{name}/remove")
    def remove(dashboard_id: Annotated[int, Path(gt=0, le=MAX_ID)],
               name: Annotated[str, Path(pattern=_MEASURE_NAME)], body: RemoveBody, asking_=Depends(asker)) -> dict:
        return answered(lambda: _result(remove_measure(asking(asking_, dashboard_id, changing=True), name,
                                                       chart_ids=body.chart_ids)))

    @app.post("/v1/superset/dashboards/{dashboard_id}/measures/{name}/save")
    def save(dashboard_id: Annotated[int, Path(gt=0, le=MAX_ID)],
             name: Annotated[str, Path(pattern=_MEASURE_NAME)], body: SaveBody, asking_=Depends(asker)) -> dict:
        return answered(lambda: _result(save_measure(asking(asking_, dashboard_id, changing=True), name)))

    @app.post("/v1/superset/dashboards/{dashboard_id}/changes")
    def changes(dashboard_id: Annotated[int, Path(gt=0, le=MAX_ID)], body: ChangeBody,
                asking_=Depends(asker)) -> dict:
        """A change asked of the dashboard by chat, made as the person: each chart setting the profile's model reads
        the request as naming, written alone; a chart to add planned and started as a build onto the dashboard
        (`added`, as the start route answers it)."""

        def work() -> dict:
            ask = asking(asking_, dashboard_id, changing=True)
            who, _ = _builder(asking_, building=True)
            interpreter = runtime.build_profile(None, who).interpreter
            if interpreter is None:
                raise RuntimeError("this runtime was built with no change interpreter, so it changes no dashboard")
            found = change_dashboard(ask, body.request or "", interpreter=interpreter,
                                     confirmed_removal=tuple(body.remove_chart_ids or ()))
            answer = _changed(found)
            if found.add_chart:
                # What the request changed stays answered whatever the planning of its chart meets.
                try:
                    answer["added"] = started(asking_, body.request, dashboard_id=dashboard_id, fresh=False,
                                              profile_name=None)[1]
                except Exception as exc:
                    refusal = _refusal(exc, doors)
                    if refusal is None:
                        raise
                    answer["added"] = {"error": {"status": refusal.status_code, "detail": refusal.detail}}
            return answer

        return answered(work)

    return BuildKit(kept=kept, started=started, asking=asking, answered=answered, deployment=deployment)

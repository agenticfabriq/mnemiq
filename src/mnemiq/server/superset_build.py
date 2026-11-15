"""Superset mode's build routes: a builder's goal to a checked plan and a build of it, the build run one step per
request, and the build as it stands. Mounted only in Superset mode, and no other route under `/v1/superset/`.

- `POST /v1/superset/builds` takes `{goal, dashboard_id?, fresh?, profile?}`. Asked with neither `dashboard_id` nor
  `fresh`, the plan may offer a dashboard the builder can see that already covers the goal: the answer is then the
  offer, its id, title and reason, with mnemiq's words on each answer to it, and nothing is built or kept. Reusing
  the dashboard is opening it, which needs no build. Asking again with its `dashboard_id` adds this goal's planned
  charts to it (a `dashboard_id` always does that, to the dashboard it names), and with `fresh: true` builds a new
  dashboard; asked either way, no dashboard is looked for. Otherwise the answer is the new build (201), or the plan's
  refusal, which is an answer too. The build keeps the profile it was asked under (`profile`, None for the
  deployment's default). Before anything is planned or kept, Superset is asked whether the person may build (the
  grants; a non-builder is refused as one, `NOT_A_BUILDER`, and nothing about a dashboard they named is read), then
  whether the extension is there, then, for a `dashboard_id`, whether they edit that dashboard (the extension's
  `editors_of`): one they cannot read, or that does not exist, is a 404 that does not say which, and one they read
  but do not edit a 403 naming its editors. A model that does not answer the planning prompt is a 503 saying to try
  again; a prompt its server refuses or cuts as too long is a refused plan, since the goal and the schema are too
  large to plan in one prompt.
- `GET /v1/superset/builds/{build_id}`: the build as it stands, read from the build store alone.
- `POST /v1/superset/builds/{build_id}/next`: the build's next step, claimed, run and recorded (`builder.next_step`);
  409 while another request's step holds the build.

**The door.** Every route takes the caller the `/v1` door verified (`asker`), and so needs the token's `db.read`. The
two `POST`s also need Superset's write scopes (`WRITE_SCOPES`), which a token Superset's mint route makes for building
carries and one it makes for asking does not (`NotForBuilding`, a 403). These routes write Superset's metadata,
through Superset as the builder, never the source database, so they need no mnemiq scope of their own.

**As the builder.** A route that calls Superset builds its own adapter with `superset_executor(settings, delegation)`
from the deployment's settings, which name Superset, and the request's verified token; the planner and the builder
call Superset only through that adapter's client. What the builder may read is what Superset says when asked as them,
read once per request. The `GET` calls nothing.

**Under its profile.** A build's plan, and each of its steps, run under the settings of the profile it was started
under, picked again at every step as `/v1/ask` picks one: a profile gone or no longer offered to the builder refuses
the step (422) before Superset is asked anything.

**What is answered.** A build as `_build` writes it, and nothing else: `build_id`, `status` and `dashboard_id`; its
`steps`, each `key`, `kind`, `status`, `object_id` (what the step made), `started_at` and `lease_until` (when a running
step's lease passes); `checks`, each chart's check status by chart id; `ran`, the step this request ran, and `message`,
mnemiq's own words on it. Never a lease id, a step's intent, or a value. Every failure is a status and mnemiq's own
sentence (`server/superset_refusals.py`, and `_refusal` here for the build's own).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mnemiq.llm.client import ModelUnavailable
from mnemiq.profiles import UnknownProfile
from mnemiq.server.superset_refusals import NotForBuilding
from mnemiq.server.v1_refusals import NotPermitted, refused, unknown_profile
from mnemiq.superset.build_store import MAX_ID, BuildNotFound, BuildStore, BuildStoreMissing, DashboardBusy
from mnemiq.superset.builder import BuildBusy, BuildState, build_state, next_step, start_build
from mnemiq.superset.client import SupersetToolError
from mnemiq.superset.executor import SupersetCredentialMissing, superset_executor
from mnemiq.superset.extension import ExtensionMissing, dashboard_editorship, require_extension
from mnemiq.superset.planner import MAX_GOAL, BuildPlan, PlanContext, PlanRefused, plan_build

__all__ = ["WRITE_SCOPES", "mount_superset_build"]

logger = logging.getLogger(__name__)

# What a build route needs beside `db.read`: Superset's own scopes for the writes a build makes, which are
# Superset's metadata, never the source database, so no mnemiq scope of their own.
WRITE_SCOPES = ("superset:chart:write", "superset:dashboard:write", "superset:dataset:write")
_NEEDED = ("db.read", *WRITE_SCOPES)

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


class NotAnEditor(Exception):
    """The builder reads the dashboard but does not edit it. `editors` are their labels, for the HTTP answer only."""

    def __init__(self, dashboard_id: int, editors: list) -> None:
        named = f"; its editors are {', '.join(str(label) for label in editors)}" if editors else ""
        super().__init__(f"This builder is not an editor of dashboard {dashboard_id}{named}.")


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

    @model_validator(mode="after")
    def _one_answer(self) -> StartBody:
        if self.fresh and self.dashboard_id is not None:
            raise ValueError("name a dashboard to add charts to, or ask for a fresh one, not both")
        return self


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


def _refused_plan(plan: PlanRefused) -> dict:
    # mnemiq's reason and each part as the model named it; never the model's draft.
    return {"refused": {"reason": plan.reason, "parts": [{"part": part.part, "reason": part.reason}
                                                         for part in plan.parts]}}


def _builder(asking, *, building: bool):
    """The verified caller and their delegated token, or the refusal: no token (401), or, for a route that builds, a
    token that lacks a write scope (403)."""
    delegation = asking.handed.get("delegation")
    if delegation is None:
        raise SupersetCredentialMissing(NO_DELEGATION)
    if building and not set(WRITE_SCOPES) <= asking.scopes:
        raise NotForBuilding(_NEEDED)
    return asking.identity, delegation


def _refusal(exc: Exception, doors) -> HTTPException | None:
    """The response a failure of a build route becomes: the build's own here, Superset's and Superset mode's as every
    `/v1` route answers them (`v1_refusals.refused`). None for anything else, which stays a 500."""
    if isinstance(exc, (BuildBusy, DashboardBusy)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, BuildNotFound):
        return HTTPException(status_code=404, detail=str(exc))
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


def mount_superset_build(app: FastAPI, runtime, doors, asker) -> None:
    """Mount the three build routes on `app`, each behind `asker`, the `/v1` door's dependency. The build store is
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

    @app.post("/v1/superset/builds", status_code=201)
    def start(body: StartBody, response: Response, asking=Depends(asker)) -> dict:
        def work() -> dict:
            who, delegation = _builder(asking, building=True)
            builds = kept()
            profile = runtime.build_profile(body.profile, who)
            planner = _Offering(profile.planner)
            adapter = superset_executor(deployment, delegation)
            # Whether Superset lets this person build at all, first: a non-builder is refused as one (NOT_A_BUILDER)
            # and learns nothing of a dashboard they named, its editors included.
            grants = runtime.builder_grants(adapter)
            # Before anything is planned or kept: every write of a build goes through the extension.
            require_extension(adapter.client)
            if body.dashboard_id is not None:
                _held_to_its_editors(adapter.client, body.dashboard_id)
            plan = plan_build(body.goal, planner=planner, client=adapter.client, adapter=adapter,
                              snapshot=runtime.snapshot, grants=grants, settings=profile.settings,
                              offer_reuse=body.dashboard_id is None and not body.fresh)
            if isinstance(plan, PlanRefused):
                response.status_code = 200
                return _refused_plan(plan)
            if plan.awaiting_choice:
                response.status_code = 200
                return _offer(plan, planner.titles)
            return {"build": _build(start_build(builds, tenant=who.tenant_id, principal=who.principal_id,
                                                goal=body.goal, plan=plan, dashboard_id=body.dashboard_id,
                                                profile=body.profile))}

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
            grants = runtime.builder_grants(adapter)
            return {"build": _build(next_step(build, client=adapter.client, adapter=adapter, store=builds,
                                              snapshot=runtime.snapshot, grants=grants, settings=profile.settings))}

        return answered(work)

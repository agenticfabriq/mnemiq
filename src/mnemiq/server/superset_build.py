"""Superset mode's build routes: a builder's goal to a checked plan and a build of it, the build run one step per
request, and the build as it stands. Mounted only in Superset mode, and no other route under `/v1/superset/`.

- `POST /v1/superset/builds` takes `{goal, dashboard_id?, fresh?, profile?}`. Asked with neither `dashboard_id` nor
  `fresh`, the plan may offer a dashboard the builder can see that already covers the goal: the answer is then the
  offer, its id, title and reason, and nothing is kept. The builder answers by asking again, with that `dashboard_id`
  to build on it or with `fresh: true` for a new one; asked either way, no dashboard is looked for. Otherwise the
  answer is the new build (201), or the plan's refusal, which is an answer too.
- `GET /v1/superset/builds/{build_id}`: the build as it stands, read from the build store alone.
- `POST /v1/superset/builds/{build_id}/next`: the build's next step, claimed, run and recorded (`builder.next_step`);
  409 while another request's step holds the build.

**The door.** Every route takes the caller the `/v1` door verified (`asker`), and so needs the token's `db.read`. The
two `POST`s also need Superset's write scopes (`WRITE_SCOPES`), which a token Superset's mint route makes for building
carries and one it makes for asking does not (`NotForBuilding`, a 403). These routes write Superset's metadata,
through Superset as the builder, never the source database, so they need no mnemiq scope of their own.

**As the builder.** A route that calls Superset builds its own adapter with `superset_executor(settings, delegation)`
from the request's verified token, and the planner and the builder call Superset only through that adapter's client.
What the builder may read is what Superset says when asked as them, read once per request. The `GET` calls nothing.

**What is answered.** A build as `_build` writes it: its id, status and dashboard, each step's key, kind, status, the
id of what it made and when it began, and each chart's check status. Never a lease id, never a value. Every failure is
a status and mnemiq's own sentence (`server/superset_refusals.py`, and `_refusal` here for the build's own).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mnemiq.profiles import UnknownProfile
from mnemiq.server.superset_refusals import NotForBuilding
from mnemiq.superset.build_store import MAX_ID, BuildNotFound, BuildStore, BuildStoreMissing, DashboardBusy
from mnemiq.superset.builder import BuildBusy, BuildState, build_state, next_step, start_build
from mnemiq.superset.executor import SupersetCredentialMissing, superset_executor
from mnemiq.superset.extension import ExtensionMissing, require_extension
from mnemiq.superset.planner import MAX_GOAL, BuildPlan, PlanContext, PlanRefused, plan_build

__all__ = ["WRITE_SCOPES", "mount_superset_build"]

logger = logging.getLogger(__name__)

# What a build route needs beside `db.read` (decision 6): Superset's own scopes for the writes a build makes.
WRITE_SCOPES = ("superset:chart:write", "superset:dashboard:write", "superset:dataset:write")
_NEEDED = ("db.read", *WRITE_SCOPES)

NO_DELEGATION = ("Superset mode builds as the person whose verified token each request carries, and this request "
                 "carries none")
STORE_UNAVAILABLE = "mnemiq's control Postgres, where builds are kept, did not answer; try this request again."


class StartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: Annotated[StrictStr, Field(min_length=1, max_length=MAX_GOAL)]
    # The dashboard to build on: the one a plan offered, or another the builder names; an id the store can keep.
    dashboard_id: Annotated[StrictInt, Field(gt=0, le=MAX_ID)] | None = None
    # A new dashboard, the answer to an offer declined.
    fresh: StrictBool = False
    profile: StrictStr | None = None  # None: the deployment's default

    @model_validator(mode="after")
    def _one_answer(self) -> StartBody:
        if self.fresh and self.dashboard_id is not None:
            raise ValueError("name a dashboard to build on, or ask for a fresh one, not both")
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
    [dashboard_id] = plan.reuse_candidates  # the plan check holds an offer to one dashboard, one the planner was shown
    return {"offer": {"dashboard_id": dashboard_id, "title": titles.get(dashboard_id), "reason": plan.reuse_reason}}


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


def _refusal(exc: Exception, doors, refused) -> HTTPException | None:
    """The response a failure of a build route becomes: the build's own here, Superset's and Superset mode's by
    `refused` (`app._refused`). None for anything else, which stays a 500."""
    if isinstance(exc, (BuildBusy, DashboardBusy)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, BuildNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, psycopg.OperationalError):
        logger.warning("the build store did not answer a build route: %s", exc)
        return HTTPException(status_code=503, detail=STORE_UNAVAILABLE)
    if isinstance(exc, UnknownProfile):
        # As `/v1/ask` composes it: what this caller may pick, never the name asked for.
        return HTTPException(status_code=422, detail=f"unknown profile; you may pick: {exc.offered}")
    if isinstance(exc, ExtensionMissing):
        logger.warning("%s", exc)
    return refused(exc, doors)


def mount_superset_build(app: FastAPI, runtime, doors, asker) -> None:
    """Mount the three build routes on `app`, each behind `asker`, the `/v1` door's dependency. The build store is
    the deployment's control Postgres (`MNEMIQ_CONTROL_DSN`); without one, every build route answers 503 naming it."""
    from mnemiq.server.app import _refused  # here: app.py imports this module

    settings = runtime.settings
    try:
        store = _open_store(settings)
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
            if (refused := _refusal(exc, doors, _refused)) is None:
                raise
            raise refused from None

    @app.post("/v1/superset/builds", status_code=201)
    def start(body: StartBody, response: Response, asking=Depends(asker)) -> dict:
        def work() -> dict:
            who, delegation = _builder(asking, building=True)
            builds = kept()
            planner = _Offering(runtime.build_planner(body.profile, who))
            adapter = superset_executor(settings, delegation)
            grants = runtime.builder_grants(adapter)
            # Before anything is planned or kept: every write of a build goes through the extension.
            require_extension(adapter.client)
            plan = plan_build(body.goal, planner=planner, client=adapter.client, adapter=adapter,
                              snapshot=runtime.snapshot, grants=grants, settings=settings,
                              offer_reuse=body.dashboard_id is None and not body.fresh)
            if isinstance(plan, PlanRefused):
                response.status_code = 200
                return _refused_plan(plan)
            if plan.awaiting_choice:
                response.status_code = 200
                return _offer(plan, planner.titles)
            return {"build": _build(start_build(builds, tenant=who.tenant_id, principal=who.principal_id,
                                                goal=body.goal, plan=plan, dashboard_id=body.dashboard_id))}

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
        def work() -> dict:
            who, delegation = _builder(asking, building=True)
            builds = kept()
            build = builds.load_build(who.tenant_id, build_id, who.principal_id)
            if build is None:
                raise BuildNotFound()
            adapter = superset_executor(settings, delegation)
            grants = runtime.builder_grants(adapter)
            return {"build": _build(next_step(build, client=adapter.client, adapter=adapter, store=builds,
                                              snapshot=runtime.snapshot, grants=grants, settings=settings))}

        return answered(work)

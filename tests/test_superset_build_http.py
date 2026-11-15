"""Superset mode's build routes at the HTTP door: `POST /v1/superset/builds`, `GET /v1/superset/builds/{build_id}` and
`POST /v1/superset/builds/{build_id}/next`, and nothing else under `/v1/superset/`.

Each route takes the caller the `/v1` door verified, needs `db.read` as every `/v1` route does, and, but for the `GET`,
Superset's write scopes too (`superset:chart:write superset:dashboard:write superset:dataset:write`): a token minted for
asking is refused for building. Each builds its adapter with `superset_executor(settings, delegation)`, so every call to
Superset carries the request's own token. Every refusal reaches the person as a status and mnemiq's own sentence.

The deployment is served as `mnemiq serve --http` serves it, with uvicorn replaced. Superset is `FakeSuperset`, a real
MCP server on loopback, its tools answered by `World` (`tests/superset_build_world.py`), the extension's included; the
build store is `MemoryBuildStore`, the Postgres store's own transitions over a dict; the planner is `FakePlanner`, so no
model is asked anything.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import os
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient

import mnemiq.adapters.resolve as resolve_module
import mnemiq.runtime as runtime_module
import mnemiq.server.app as app_module
from mnemiq.config import Settings
from mnemiq.llm.client import ContextTooLong, ModelUnavailable, PromptCut
from mnemiq.contract import Snapshot, SourceBinding
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot
from mnemiq.superset.build_store import STORE_MISSING
from mnemiq.superset.builder import start_build
from mnemiq.superset.client import DelegatedToken, SupersetMcp
from mnemiq.superset.extension import EXTENSION_TOOLS, TOOL_PREFIX
from mnemiq.superset.grants import NOT_A_BUILDER
from mnemiq.superset.planner import FakePlanner, LLMPlanner, PlanRefused, RefusedPart
from tests.auth_keys import AUDIENCE, ISSUER, claims, jwks_for, mint, rsa_key
from tests.superset_build_world import SNAPSHOT as WORLD_SNAPSHOT
from tests.superset_build_world import TABLES, Clock, Hook, MemoryBuildStore, World
from tests.superset_fake import FakeSuperset
from tests.sweep.stub_model import StubModel
from tests.superset_shared import (
    ONE_BAR,
    ROLE_DENIAL,
    SIGN_IN,
    SNAPSHOT_REFUSED,
    SOURCE_ID,
    _source_job,
    superset_deployment,
)

GOAL = "Policies by region"
# The sentences the build routes answer with, as the person reads them.
NOT_FOR_BUILDING = "This sign-in was made for asking, not building; open the builder from Superset again."
EXTENSION_MISSING = "This Superset does not have the mnemiq extension installed."
STEP_RUNNING = "a step of this build is already running"
DASHBOARD_BUSY = "another build of this dashboard is still building; it must finish or stop first"
NO_SUCH_DASHBOARD = "This builder can read no dashboard of that id, so nothing was planned."

# What a token minted for asking carries, and what one minted for building adds (decision 7).
ASK_SCOPES = "db.read superset:sqllab:write superset:dataset:read"
BUILD_SCOPES = (f"{ASK_SCOPES} superset:chart:read superset:chart:write superset:dashboard:read "
                "superset:dashboard:write superset:dataset:write")
# What a build route needs beside `db.read` (decision 6).
WRITE_SCOPES = ("superset:chart:write", "superset:dashboard:write", "superset:dataset:write")
# Two profiles, `a` the default; anyone may pick `b`, which guards fan-out where the deployment does not.
PROFILES = """
default = "a"

[profiles.a]
llm_model = "model-a"

[profiles.b]
llm_model = "model-b"
roles = ["*"]
guard_fanout = true
"""

BUILD_ROUTES = {("POST", "/v1/superset/builds"), ("GET", "/v1/superset/builds/{build_id}"),
                ("POST", "/v1/superset/builds/{build_id}/next")}
# The calls that change something in Superset, which a refused request never makes.
WRITES = {"generate_chart", "create_dataset", "create_virtual_dataset",
          *(TOOL_PREFIX + base for base in ("create_dashboard", "claim_chart", "claim_dataset", "create_chart",
                                            "patch_chart", "add_saved_metric", "delete_dataset", "set_layout"))}


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


@pytest.fixture
def fake(request):
    """Superset, with mnemiq's extension unless the test asks for one without (`indirect` parametrization)."""
    with FakeSuperset(extension=getattr(request, "param", True)) as served:
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


@pytest.fixture
def model():
    """The model the deployment is configured with. A build's plan comes from `FakePlanner`, so it is never asked."""
    with StubModel(sql="SELECT 1") as stub:
        yield stub
    assert stub.calls == 0


@pytest.fixture
def jwks(tmp_path) -> str:
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps(jwks_for((rsa_key(), "k1"))))
    return str(path)


@pytest.fixture
def no_direct_source(monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("Superset mode reached a direct source connection")

    monkeypatch.setattr(runtime_module, "source_adapter", reached)
    monkeypatch.setattr(resolve_module, "adapter_for", reached)


def _snapshot() -> Snapshot:
    """The World's tables, as enrichment through Superset records them: each bound to the name it is stored under, and
    the job naming the database and schema they were read from."""
    return Snapshot(version="v1", source_id=SOURCE_ID, created_at="t", columns=list(WORLD_SNAPSHOT.columns),
                    source_bindings=[SourceBinding(id=f"sb:{table}", source_id=SOURCE_ID, object_id=table,
                                                   source_object=table, binding_type="table") for table in TABLES],
                    jobs=[_source_job()])


def _settings(tmp_path, model, fake, jwks, **over) -> Settings:
    """A Superset-mode deployment over the World's tables, whose doors verify tokens from `ISSUER` for `AUDIENCE`."""
    settings = superset_deployment(tmp_path, model.url, fake.url, snapshot=False, oidc_issuer=ISSUER,
                                   oidc_audience=AUDIENCE, oidc_jwks_file=jwks, tenant="acme", guard_fanout=False,
                                   **over)
    con = init_store(settings.store_path)
    save_snapshot(con, _snapshot())
    con.close()
    return settings


def _build_module():
    return importlib.import_module("mnemiq.server.superset_build")


def _serve(settings: Settings, monkeypatch, store=None):
    """What `serve_http` serves for `settings`, and the runtime it built, with `store` as the build store (None: the
    store the settings name). uvicorn is replaced, nothing else."""
    if store is not None:
        monkeypatch.setattr(_build_module(), "_open_store", lambda _settings: store)
    built: dict = {}
    real = runtime_module.build_runtime
    monkeypatch.setattr(runtime_module, "build_runtime", lambda s: built.setdefault("runtime", real(s)))
    monkeypatch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
    app_module.serve_http(settings, host="127.0.0.1", port=8080)
    return TestClient(built["app"]), built["runtime"]


def _planned(runtime, planner, profile: str | None = None):
    runtime.profiles[profile or runtime.default_profile].planner = planner
    return planner


@pytest.fixture
def served(tmp_path, model, fake, world, store, jwks, monkeypatch, no_direct_source):
    """The deployment, its build store the memory one, and its default profile planning `ONE_BAR`."""
    client, runtime = _serve(_settings(tmp_path, model, fake, jwks), monkeypatch, store)
    _planned(runtime, FakePlanner(ONE_BAR))
    return client, runtime


def _claims(**over) -> dict:
    return claims(**{"sub": "west", "preferred_username": "west", "scope": BUILD_SCOPES, **over})


def _token(**over) -> str:
    return mint(_claims(**over))


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _whoami(client, token: str) -> tuple[str, str]:
    found = client.get("/v1/whoami", headers=_bearer(token)).json()
    return found["tenant_id"], found["principal_id"]


def _start(client, token: str | None = None, **body):
    return client.post("/v1/superset/builds", json={"goal": GOAL, **body}, headers=_bearer(token or _token()))


def _next(client, build_id: str, token: str | None = None):
    return client.post(f"/v1/superset/builds/{build_id}/next", headers=_bearer(token or _token()))


def _get(client, build_id: str, token: str | None = None):
    return client.get(f"/v1/superset/builds/{build_id}", headers=_bearer(token or _token()))


def _stored(store, client, token: str | None = None, plan=ONE_BAR, **more):
    """A build of `plan` started in the store directly, for the person `token` names, as the start route would."""
    tenant, principal = _whoami(client, token or _token())
    started = start_build(store, tenant=tenant, principal=principal, goal=GOAL, plan=plan, **more)
    return store.load_build(tenant, started.build_id, principal)


def _writes(fake) -> list[str]:
    return [call.tool for call in fake.calls if call.tool in WRITES]


def _routes(app) -> set[tuple[str, str]]:
    return {(method, route.path) for route in app.routes if getattr(route, "path", "").startswith("/v1/superset")
            for method in getattr(route, "methods", None) or ()}


def _call(client, method: str, path: str, token: str | None):
    path = path.replace("{build_id}", str(uuid.uuid4()))
    headers = _bearer(token) if token is not None else {}
    return client.post(path, json={"goal": GOAL}, headers=headers) if method == "POST" else client.get(path,
                                                                                                    headers=headers)


# --- the routes there are, and where


def test_only_build_routes_exist_after_pr_c(served):
    client, _ = served
    assert _routes(client.app) == BUILD_ROUTES


def test_no_route_mounted_in_direct_mode(tmp_path, model):
    """With no Superset URL, no route is mounted under `/v1/superset/`, and a request there finds nothing."""
    from tests.scripted_deployment import IDENTITY, deployment

    settings = deployment(tmp_path, model.url)
    assert not settings.superset_mode
    app = app_module.build_app(runtime_module.build_runtime(settings), IDENTITY, static_dir=tmp_path)
    assert _routes(app) == set()
    client = TestClient(app)
    for method, path in BUILD_ROUTES:
        assert _call(client, method, path, None).status_code == 404


def test_superset_mode_without_doors_builds_for_nobody(tmp_path, model, fake, world, store, monkeypatch,
                                                      no_direct_source):
    """Built without an issuer, which startup refuses in Superset mode, the routes are there and answer nobody: no
    token to hand on, so no Superset call and no build."""
    from mnemiq.contract import IdentityContext

    monkeypatch.setattr(_build_module(), "_open_store", lambda _settings: store)
    settings = superset_deployment(tmp_path, model.url, fake.url, snapshot=False)
    con = init_store(settings.store_path)
    save_snapshot(con, _snapshot())
    con.close()
    runtime = runtime_module.build_runtime(settings)
    app = app_module.build_app(runtime, IdentityContext(tenant_id="acme", principal_id="west"), static_dir=tmp_path)
    assert _routes(app) == BUILD_ROUTES
    client = TestClient(app)
    for method, path in BUILD_ROUTES:
        r = _call(client, method, path, None)
        assert r.status_code == 401, (path, r.text)
    assert fake.calls == [] and store.claims == []


# --- the door


def _expired() -> str:
    now = int(time.time())
    return _token(iat=now - 900, nbf=now - 900, exp=now - 600)


BAD_TOKENS = {
    "expired": _expired,
    "wrong-audience": lambda: _token(aud="another-service"),
    "wrong-issuer": lambda: _token(iss="https://elsewhere.example/realms/r"),
    "foreign-key": lambda: mint(_claims(), rsa_key("elsewhere"), kid="k9"),
    "no-token": lambda: None,
}


@pytest.mark.parametrize("make", BAD_TOKENS.values(), ids=BAD_TOKENS.keys())
@pytest.mark.parametrize("method, path", sorted(BUILD_ROUTES))
def test_build_routes_refuse_bad_tokens(served, fake, store, make, method, path):
    """Each refused at the door, before the store or Superset is asked anything."""
    client, _ = served
    r = _call(client, method, path, make())
    assert r.status_code == 401, r.text
    assert r.headers["www-authenticate"] == "Bearer"
    assert fake.calls == [] and store.claims == []


@pytest.mark.parametrize("method, path", sorted(BUILD_ROUTES))
def test_a_token_without_db_read_is_refused(served, fake, store, method, path):
    client, _ = served
    r = _call(client, method, path, _token(scope=" ".join(WRITE_SCOPES)))
    assert r.status_code == 403, r.text
    assert "db.read" in r.json()["detail"]
    assert fake.calls == [] and store.claims == []


def test_ask_token_cannot_build(served, fake, world, store):
    """A token minted for asking starts no build and runs no step: 403, mnemiq's sentence, an insufficient-scope
    challenge naming what building needs, and nothing asked of Superset. It may still read a build of its person's."""
    client, _ = served
    build = _stored(store, client)
    ask = _token(scope=ASK_SCOPES)
    for r in (_start(client, ask), _next(client, build.build_id, ask)):
        assert r.status_code == 403, r.text
        assert r.json() == {"detail": NOT_FOR_BUILDING, "end_session_url": None}
        challenge = r.headers["www-authenticate"]
        assert 'error="insufficient_scope"' in challenge and all(scope in challenge for scope in WRITE_SCOPES)
    assert fake.calls == [] and store.claims == []
    read = _get(client, build.build_id, ask)
    assert read.status_code == 200, read.text
    assert read.json()["build"]["build_id"] == build.build_id


@pytest.mark.parametrize("missing", WRITE_SCOPES)
def test_each_write_scope_is_needed(served, fake, store, missing):
    client, _ = served
    build = _stored(store, client)
    short = _token(scope=" ".join(scope for scope in BUILD_SCOPES.split() if scope != missing))
    for r in (_start(client, short), _next(client, build.build_id, short)):
        assert r.status_code == 403 and r.json()["detail"] == NOT_FOR_BUILDING, r.text
    assert fake.calls == [] and store.claims == []


def test_the_routes_need_exactly_the_write_scopes_a_build_token_carries():
    """mnemiq's list of what building needs is within what Superset's mint route puts in a build token and outside what
    it puts in an ask token, so neither can drift from the other."""
    from tests.superset_extension_path import use_extension_source

    use_extension_source()
    from agenticfabriq.mnemiq import token as minted

    needed = set(_build_module().WRITE_SCOPES)
    assert needed == set(WRITE_SCOPES)
    assert needed <= set(minted.BUILD_SCOPES) and not needed & set(minted.ASK_SCOPES)
    assert set(BUILD_SCOPES.split()) == set(minted.BUILD_SCOPES) and set(ASK_SCOPES.split()) == set(minted.ASK_SCOPES)


def _denying_queries(world: World) -> None:
    """Superset's roles for this person do not permit SQL, as a Gamma user's do not: its refusal, sent as the tool's
    error."""
    world.hooks["execute_sql"] = Hook(instead=lambda request, _world: RuntimeError(ROLE_DENIAL))


@pytest.mark.parametrize("route", ["start", "next"])
def test_build_routes_refuse_a_non_builder(served, fake, world, store, route):
    """A person Superset's roles do not let build is told the roles, and nothing else: no build, no claim, no write."""
    client, _ = served
    build = _stored(store, client)
    _denying_queries(world)
    r = _start(client) if route == "start" else _next(client, build.build_id)
    assert r.status_code == 403, r.text
    assert r.json() == {"detail": NOT_A_BUILDER, "end_session_url": None}
    assert store.claims == [] and _writes(fake) == []
    assert list(store._builds) == [(build.tenant_id, build.build_id)]


@pytest.mark.parametrize("route", ["start", "next"])
def test_a_sign_in_superset_refuses_is_told_to_sign_in_again(served, fake, store, route):
    client, _ = served
    token = _token(jti="refused")
    build = _stored(store, client, token)
    fake.refuse_token(token)
    r = _start(client, token) if route == "start" else _next(client, build.build_id, token)
    assert r.status_code == 401, r.text
    assert r.json()["detail"] == SIGN_IN and r.headers["www-authenticate"] == "Bearer"
    assert store.claims == []


# --- what a build needs


def test_missing_store_is_named(tmp_path, model, fake, world, jwks, monkeypatch, no_direct_source):
    """Without `MNEMIQ_CONTROL_DSN` every build route answers 503 with the sentence that names it, before Superset is
    asked anything; asking is untouched."""
    client, runtime = _serve(_settings(tmp_path, model, fake, jwks), monkeypatch)
    assert runtime.settings.control_dsn is None
    for method, path in sorted(BUILD_ROUTES):
        r = _call(client, method, path, _token())
        assert r.status_code == 503, (path, r.text)
        assert r.json()["detail"] == STORE_MISSING and "MNEMIQ_CONTROL_DSN" in r.json()["detail"]
    assert fake.calls == []


def test_a_snapshot_superset_mode_will_not_serve_refuses_building(served, fake, store, monkeypatch):
    """A newer snapshot Superset mode will not serve refuses a build as it refuses an answer: the snapshot is read
    again before anything is asked of Superset, so a plan and a check are never made from the snapshot it replaced."""
    client, runtime = served
    build = _stored(store, client)
    serving = runtime.snapshot
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    monkeypatch.setattr(runtime_module, "load_current_snapshot",
                        lambda settings, con: (_snapshot().model_copy(update={"version": "v2", "jobs": []}),
                                               {SOURCE_ID: "v2"}))
    for r in (_start(client), _next(client, build.build_id)):
        assert r.status_code == 503, r.text
        assert r.json()["detail"] == SNAPSHOT_REFUSED
    assert runtime.snapshot is serving
    assert fake.calls == [] and store.claims == []


def _masked(base: str):
    """What rc2's MCP service answers for a tool it does not have (`GlobalErrorHandlerMiddleware`), as the tool's
    error."""
    def answer(request, _world):
        return RuntimeError(f"Internal error in {TOOL_PREFIX}{base}: An unexpected error occurred. Error ID: 3f9c0a1b. "
                            "Please contact support if this persists.")

    return answer


@pytest.mark.parametrize("fake", [False], indirect=True)
def test_missing_extension_is_named(served, fake, world, store):
    """A Superset without mnemiq's extension: a build is refused before it is planned or kept, with the sentence that
    names the extension and none of Superset's words."""
    client, runtime = served
    r = _start(client)
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == EXTENSION_MISSING
    assert "Error ID" not in r.text and "unexpected" not in r.text
    assert store._builds == {} and _writes(fake) == []
    assert runtime.profiles[runtime.default_profile].planner.goals == []
    # Settled by one call of the extension's probe, not the probe asked twice.
    assert [call.tool for call in fake.calls if call.tool.startswith("extensions.")] == [
        TOOL_PREFIX + "dataset_metrics"]


def test_an_extension_gone_mid_build_is_named(served, fake, world, store):
    """The extension removed between steps: the next step's first extension call finds it missing, and the step is
    left to be reconciled once it is back."""
    client, _ = served
    build = _stored(store, client)
    for base in EXTENSION_TOOLS:
        world.hooks[base] = Hook(instead=_masked(base))
    r = _next(client, build.build_id)
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == EXTENSION_MISSING and "Error ID" not in r.text
    assert [step["status"] for step in store.raw_steps(build.tenant_id, build.build_id)][0] == "uncertain"
    assert world.dashboards == {}


# --- a build, through the routes


def test_a_build_runs_through_the_routes_and_no_answer_holds_a_lease(served, fake, world, store):
    """Started, run one step per `/next` to a laid-out, checked dashboard, and read back: each answer is the build as
    the builder sees it, and none holds a lease id, though one was held during every step."""
    client, runtime = served
    leases: set[str] = set()

    def keep_leases(request, answer, _world):
        leases.update(step["lease_id"] for build in store._builds.values() for step in build["steps"]
                      if step["lease_id"])

    for tool in ("create_dashboard", "generate_chart", "get_layout", "get_chart_data"):
        world.hooks[tool] = Hook(after=keep_leases)
    started = _start(client)
    assert started.status_code == 201, started.text
    build = started.json()["build"]
    assert build["status"] == "building" and build["dashboard_id"] is None
    assert [step["key"] for step in build["steps"]] == ["dashboard", "policies_by_region", "layout", "check"]
    assert all(step["status"] == "pending" for step in build["steps"])
    planner = runtime.profiles[runtime.default_profile].planner
    assert planner.goals == [GOAL] and len(planner.contexts) == 1
    answers = [started]
    for _ in range(10):
        answers.append(_next(client, build["build_id"]))
        assert answers[-1].status_code == 200, answers[-1].text
        if answers[-1].json()["build"]["status"] != "building":
            break
    final = answers[-1].json()["build"]
    assert [a.json()["build"]["ran"] for a in answers[1:]] == ["dashboard", "policies_by_region", "layout", "check"]
    assert final["status"] == "done" and final["dashboard_id"] in world.dashboards
    chart = next(step["object_id"] for step in final["steps"] if step["key"] == "policies_by_region")
    assert final["checks"] == {str(chart): "checked"}, final
    read = _get(client, build["build_id"])
    assert read.status_code == 200
    assert read.json()["build"]["steps"] == final["steps"] and read.json()["build"]["checks"] == final["checks"]
    answers.append(read)
    assert leases, "no step held a lease while it ran"
    for answer in answers:
        assert "lease_id" not in answer.text and not any(lease in answer.text for lease in leases)
        assert set(answer.json()) == {"build"}
        assert all(set(step) <= {"key", "kind", "status", "object_id", "started_at", "lease_until"}
                   for step in answer.json()["build"]["steps"])


def test_a_dashboard_is_offered_before_anything_is_built(served, fake, world, store):
    """The planner offers a dashboard the builder can see: the answer is the offer, its id, title and reason, and
    mnemiq's words on what each answer to it does, and no build. Reusing it is opening it, which builds nothing.
    Asking again with its `dashboard_id` adds this goal's charts to it, and with `fresh` builds a new dashboard;
    asked either way, no dashboard is looked for, and the build starts."""
    client, runtime = served
    world.dashboards[900] = {"id": 900, "dashboard_title": "Policies at a glance", "published": True,
                             "description": None, "position_json": None, "json_metadata": "{}",
                             "editors": ["someone"], "members": [], "created_by": "someone",
                             "created_on": world.superset_now(), "changed_on": world.superset_now()}
    offer = dataclasses.replace(ONE_BAR, reuse_candidates=(900,), reuse_reason="It already shows policies.")
    planner = _planned(runtime, FakePlanner([offer, ONE_BAR, ONE_BAR]))
    r = _start(client)
    assert r.status_code == 200, r.text
    assert r.json() == {"offer": {
        "dashboard_id": 900, "title": "Policies at a glance", "reason": "It already shows policies.",
        "message": ("Dashboard 900 already covers this goal, so nothing was built: to reuse it, open it. To add this "
                    "goal's charts to it anyway, ask again with dashboard_id 900; for a new dashboard, ask again with "
                    "fresh true.")}}
    assert store._builds == {} and _writes(fake) == []
    assert [d.id for d in planner.contexts[0].dashboards] == [900]

    searched = len([call for call in fake.calls if call.tool == "list_dashboards"])
    reused = _start(client, dashboard_id=900)
    assert reused.status_code == 201, reused.text
    assert reused.json()["build"]["dashboard_id"] == 900
    fresh = _start(client, fresh=True)
    assert fresh.status_code == 201, fresh.text
    assert fresh.json()["build"]["dashboard_id"] is None
    assert planner.contexts[1].dashboards == () and planner.contexts[2].dashboards == ()
    assert len([call for call in fake.calls if call.tool == "list_dashboards"]) == searched
    assert len(store._builds) == 2 and _writes(fake) == []


@pytest.mark.parametrize("body", [{"dashboard_id": 900, "fresh": True}, {"dashboard_id": 0}, {"dashboard_id": "900"},
                                  {"dashboard_id": 2**31}, {"fresh": "yes"}, {"goal": 7}, {"surprise": 1}],
                         ids=["both", "zero", "text-id", "past-the-store", "text-fresh", "goal-not-text",
                              "unknown-key"])
def test_a_body_the_route_cannot_read_is_422(served, fake, store, body):
    """Refused before anything is planned or asked of Superset, an id the store could not keep included."""
    client, runtime = served
    r = client.post("/v1/superset/builds", json={"goal": GOAL, **body}, headers=_bearer(_token()))
    assert r.status_code == 422, r.text
    assert fake.calls == [] and store._builds == {}
    assert runtime.profiles[runtime.default_profile].planner.goals == []


def _dashboard(world, dashboard_id: int, *, editors=("someone",), title: str = "Sales") -> int:
    """A dashboard made by hand. The person each test's token names is `someone` to the World."""
    world.dashboards[dashboard_id] = {"id": dashboard_id, "dashboard_title": title, "published": True,
                                      "description": None, "position_json": None, "json_metadata": "{}",
                                      "editors": list(editors), "members": [], "created_by": "analyst",
                                      "created_on": world.superset_now(), "changed_on": world.superset_now()}
    return dashboard_id


def test_the_largest_dashboard_id_the_store_keeps_is_built_on(served, fake, world, store):
    """The control for the bound: Postgres's largest `integer` is read, planned and kept."""
    client, _ = served
    _dashboard(world, 2**31 - 1)
    r = _start(client, dashboard_id=2**31 - 1)
    assert r.status_code == 201, r.text
    assert r.json()["build"]["dashboard_id"] == 2**31 - 1


def _asked_only_whether_the_builder_edits(fake, dashboard_id: int) -> None:
    """The one call Superset was asked: the extension's read of whether the builder edits the dashboard."""
    assert [(call.tool, call.request) for call in fake.calls] == [
        (TOOL_PREFIX + "editors_of", {"dashboard_ids": [dashboard_id]})]


def test_a_dashboard_the_builder_cannot_read_is_not_found_before_anything_is_planned(served, fake, world, store):
    """A dashboard id Superset has no dashboard of that the builder can read, whether none exists yet (the id it will
    give the next dashboard made) or they may not read it, is refused before anything is planned or kept, in words
    that do not say which. So it takes no dashboard's one live build, and says nothing of a build running on it."""
    client, runtime = served
    running = _stored(store, client, dashboard_id=900)  # someone who can read it is building on dashboard 900
    fake.calls.clear()
    r = _start(client, dashboard_id=900)
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == NO_SUCH_DASHBOARD and "900" not in r.text
    _asked_only_whether_the_builder_edits(fake, 900)
    assert list(store._builds) == [(running.tenant_id, running.build_id)]
    assert runtime.profiles[runtime.default_profile].planner.goals == []


def test_a_dashboard_the_builder_does_not_edit_is_refused_naming_its_editors(served, fake, world, store):
    """One the builder reads but does not edit is a 403 naming its editors, in the HTTP answer only, with where Sign out
    ends as every `/v1` 403 carries it; nothing is planned or kept."""
    client, runtime = served
    _dashboard(world, 900, editors=("colleague", "analyst"))
    r = _start(client, dashboard_id=900)
    assert r.status_code == 403, r.text
    assert r.json() == {"detail": "This builder is not an editor of dashboard 900; its editors are analyst, colleague.",
                        "end_session_url": None}
    _asked_only_whether_the_builder_edits(fake, 900)
    assert store._builds == {} and runtime.profiles[runtime.default_profile].planner.goals == []


@pytest.mark.parametrize("refusal", [
    PlanRefused("the model's reply could not be read as a plan"),
    PlanRefused("measure 'x' does not aggregate", (RefusedPart("measure 'x'", "does not aggregate"),),
                draft='{"draft-only": true}'),
], ids=["unreadable", "a-part-refused-twice"])
def test_a_refused_plan_is_an_answer(served, fake, store, refusal):
    """No plan is an answer, not a failure: the reason and each refused part as the plan step gave them, never the
    model's draft, and nothing kept or written."""
    client, runtime = served
    _planned(runtime, FakePlanner(refusal))
    r = _start(client)
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"refused"} and "draft-only" not in r.text
    told = r.json()["refused"]
    assert told["parts"] == [{"part": part.part, "reason": part.reason} for part in refusal.parts]
    if refusal.parts:  # refused again after the repair: the plan step's own reason, naming each part
        assert all(part.part in told["reason"] for part in refusal.parts)
    else:  # the first reply could not be read: its reason as it is
        assert told["reason"] == refusal.reason
    assert store._builds == {} and _writes(fake) == []


class _FailingModel:
    """The profile's model client, failing every call as `LLMClient` raises a provider's failure."""

    def __init__(self, failure: Exception) -> None:
        self.failure, self.calls = failure, 0

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        raise self.failure


# Each failure, with the provider's own words in it (which no answer may carry), and a phrase of mnemiq's sentence.
MODEL_FAILURES = {
    "an outage": (ModelUnavailable("Connection refused by model-host.internal:8000", kind="connect"),
                  "did not respond"),
    "a context too long": (ContextTooLong("model-host.internal: maximum context length is 32768 tokens"),
                           "longer than its context window"),
    "a cut prompt": (PromptCut("model-host.internal read 8000 of 41000 prompt tokens"), "cut the planning prompt"),
}


@pytest.mark.parametrize("case", list(MODEL_FAILURES))
def test_a_model_that_fails_to_plan_is_an_answer_in_mnemiqs_words(served, fake, store, case):
    """The profile's model, asked for the plan, does not answer, refuses the prompt as longer than its window, or
    cuts the prompt to fit it. Each is an answer, never a 500, as `/v1/ask` answers the same failures inside a 200:
    the plan is refused with mnemiq's own sentence, the provider's words go to the log only, the model is asked once,
    and nothing is kept or written."""
    client, runtime = served
    failure, said = MODEL_FAILURES[case]
    model = _FailingModel(failure)
    _planned(runtime, LLMPlanner(model))
    r = _start(client)
    assert r.status_code == 200, r.text
    told = r.json()["refused"]
    assert said in told["reason"] and told["parts"] == [], told
    assert "model-host" not in r.text
    assert model.calls == 1
    assert store._builds == {} and _writes(fake) == []


def test_a_build_is_planned_by_the_profile_it_was_asked_under(tmp_path, model, fake, world, store, jwks,
                                                             monkeypatch, no_direct_source):
    """Each profile carries the planner assembly built on its own model client, and a build asked under a profile is
    planned by that profile's planner alone; a profile the person may not pick is refused before any work."""
    client, runtime = _serve(_settings(tmp_path, model, fake, jwks, profiles=PROFILES), monkeypatch, store)
    for name in ("a", "b"):
        loaded = runtime.profiles[name]
        assert isinstance(loaded.planner, LLMPlanner) and loaded.planner._client is loaded.clients[0]
    assert runtime.profiles["a"].planner._client is not runtime.profiles["b"].planner._client
    a, b = _planned(runtime, FakePlanner(ONE_BAR), "a"), _planned(runtime, FakePlanner(ONE_BAR), "b")
    assert _start(client, profile="b").status_code == 201
    assert (a.goals, b.goals) == ([], [GOAL])
    assert _start(client).status_code == 201
    assert (a.goals, b.goals) == ([GOAL], [GOAL])
    fake.calls.clear()
    r = _start(client, profile="nope")
    assert r.status_code == 422 and "nope" not in r.text
    assert fake.calls == []


# --- a build keeps its profile


@pytest.fixture
def profiled(tmp_path, model, fake, world, store, jwks, monkeypatch, no_direct_source):
    """A deployment of two profiles, each planning `ONE_BAR`."""
    client, runtime = _serve(_settings(tmp_path, model, fake, jwks, profiles=PROFILES), monkeypatch, store)
    for name in ("a", "b"):
        _planned(runtime, FakePlanner(ONE_BAR), name)
    return client, runtime


def _settings_each_call_receives(monkeypatch) -> dict[str, list]:
    """The settings the plan, each step and each adapter were given, call by call, through the routes' own names for
    them."""
    module = _build_module()
    seen: dict[str, list] = {"plan": [], "step": [], "adapter": []}
    real_plan, real_step, real_executor = module.plan_build, module.next_step, module.superset_executor

    def plan(*args, **kwargs):
        seen["plan"].append(kwargs["settings"])
        return real_plan(*args, **kwargs)

    def step(*args, **kwargs):
        seen["step"].append(kwargs["settings"])
        return real_step(*args, **kwargs)

    def executor(settings, delegation):
        seen["adapter"].append(settings)
        return real_executor(settings, delegation)

    monkeypatch.setattr(module, "plan_build", plan)
    monkeypatch.setattr(module, "next_step", step)
    monkeypatch.setattr(module, "superset_executor", executor)
    return seen


def _run_to_the_end(client, build_id: str) -> list:
    answers = []
    for _ in range(10):
        answers.append(_next(client, build_id))
        assert answers[-1].status_code == 200, answers[-1].text
        if answers[-1].json()["build"]["status"] != "building":
            return answers
    raise AssertionError("the build did not end")


def test_a_build_keeps_the_profile_it_started_with(profiled, fake, world, store, monkeypatch):
    """Started under profile `b`, whose `guard_fanout` is not the deployment's, the build keeps `b` on its row: its plan
    and every one of its steps, the check included, run under `b`'s settings, while every adapter is built from the
    deployment's, which name Superset."""
    client, runtime = profiled
    seen = _settings_each_call_receives(monkeypatch)
    b = runtime.profiles["b"].settings
    assert b.guard_fanout is True and runtime.settings.guard_fanout is False
    started = _start(client, profile="b")
    assert started.status_code == 201, started.text
    build_id = started.json()["build"]["build_id"]
    [row] = [row for (_, kept), row in store._builds.items() if kept == build_id]
    assert row["profile"] == "b"
    answers = _run_to_the_end(client, build_id)
    assert [a.json()["build"]["ran"] for a in answers] == ["dashboard", "policies_by_region", "layout", "check"]
    assert seen["plan"] == [b] and seen["plan"][0] is b
    assert len(seen["step"]) == 4 and all(settings is b for settings in seen["step"])
    assert len(seen["adapter"]) == 5 and all(settings is runtime.settings for settings in seen["adapter"])


def test_a_build_started_under_no_profile_runs_under_the_deployments(served, fake, world, store, monkeypatch):
    client, runtime = served
    seen = _settings_each_call_receives(monkeypatch)
    started = _start(client)
    assert started.status_code == 201, started.text
    build_id = started.json()["build"]["build_id"]
    [row] = [row for (_, kept), row in store._builds.items() if kept == build_id]
    assert row["profile"] is None
    _run_to_the_end(client, build_id)
    assert seen["plan"] and seen["step"] and seen["adapter"]
    assert all(settings is runtime.settings for settings in seen["plan"] + seen["step"] + seen["adapter"])


def test_a_profile_gone_at_a_later_step_refuses_the_step(profiled, fake, world, store, monkeypatch):
    """The profile a build was started under no longer offered: its next step is refused with the sentence the start
    and `/v1/ask` give an unknown profile, before Superset is asked anything or a step is claimed."""
    client, runtime = profiled
    started = _start(client, profile="b")
    assert started.status_code == 201, started.text
    monkeypatch.delitem(runtime.profiles, "b")
    fake.calls.clear()
    claims = list(store.claims)
    r = _next(client, started.json()["build"]["build_id"])
    assert r.status_code == 422, r.text
    assert fake.calls == [] and store.claims == claims
    asked = client.post("/v1/ask", json={"question": "q", "profile": "b"}, headers=_bearer(_token()))
    again = _start(client, profile="b")
    assert asked.status_code == again.status_code == 422
    assert r.json()["detail"] == asked.json()["detail"] == again.json()["detail"]
    assert r.json()["detail"].startswith("unknown profile; you may pick:") and "'a'" in r.json()["detail"]
    assert fake.calls == []


# --- one step at a time, and one build per dashboard


def test_next_while_a_step_runs_is_409(served, fake, world, store):
    """While one request's step holds its lease, a second `/next` runs nothing and answers 409, and a read of the build
    then shows the step claimed without its lease id."""
    client, _ = served
    build = _stored(store, client)
    entered, release = threading.Event(), threading.Event()
    leases: list[str | None] = []

    def hold(request, _world):
        leases.extend(step["lease_id"] for step in store.raw_steps(build.tenant_id, build.build_id)
                      if step["key"] == "dashboard")
        entered.set()
        assert release.wait(20)

    world.hooks["create_dashboard"] = Hook(before=hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_next, client, build.build_id)
        assert entered.wait(20)
        try:
            second = _next(client, build.build_id)
            read = _get(client, build.build_id)
        finally:
            release.set()
        ran = first.result(timeout=30)
    assert second.status_code == 409, second.text
    assert second.json()["detail"] == STEP_RUNNING
    [held] = leases
    assert held  # the step held a lease while the read was taken
    assert read.status_code == 200 and "lease_id" not in read.text and held not in read.text
    assert [step["key"] for step in read.json()["build"]["steps"] if step["lease_until"]] == ["dashboard"]
    assert ran.status_code == 200 and ran.json()["build"]["ran"] == "dashboard"
    assert len([c for c in fake.calls if c.tool == TOOL_PREFIX + "create_dashboard"]) == 1


def test_a_dashboard_with_a_build_building_is_409(served, fake, world, store):
    client, _ = served
    world.dashboards[900] = {"id": 900, "dashboard_title": "Sales", "published": True, "description": None,
                             "position_json": None, "json_metadata": "{}", "editors": ["someone"], "members": [],
                             "created_by": "someone", "created_on": world.superset_now(),
                             "changed_on": world.superset_now()}
    assert _start(client, dashboard_id=900).status_code == 201
    r = _start(client, dashboard_id=900)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == DASHBOARD_BUSY
    assert len(store._builds) == 1


def test_another_persons_build_is_not_found(served, fake, store):
    """A build is its person's: another person's token reads and runs it exactly as one that does not exist, and asks
    Superset nothing."""
    client, _ = served
    build = _stored(store, client)
    east = _token(sub="east", preferred_username="east")
    fake.calls.clear()
    for r in (_get(client, build.build_id, east), _next(client, build.build_id, east),
              _get(client, str(uuid.uuid4())), _get(client, "not-a-build")):
        assert r.status_code == 404, r.text
    assert fake.calls == [] and store.claims == []


def test_a_store_that_does_not_answer_is_503_in_mnemiqs_words(served, fake, store, monkeypatch):
    client, _ = served
    build = _stored(store, client)

    def lost(*args, **kwargs):
        raise psycopg.OperationalError("server closed the connection unexpectedly: secret-host:5432")

    monkeypatch.setattr(store, "load_build", lost)
    monkeypatch.setattr(store, "create_build", lost)
    for r in (_get(client, build.build_id), _next(client, build.build_id), _start(client)):
        assert r.status_code == 503, r.text
        assert "secret-host" not in r.text and "MNEMIQ_CONTROL_DSN" not in r.text
        assert "control Postgres" in r.json()["detail"]


# --- every call as the person asking


def test_every_call_carries_the_requests_token(served, fake, world, store, monkeypatch):
    """Two people start a build at once and then run its first step at once. Every call each request makes reaches
    Superset with that request's own token: the calls that name a request's goal carry its token, at the client and
    on the wire, and each request built its own adapter with `superset_executor` from its own delegated token."""
    client, runtime = served
    module = _build_module()
    tokens = {"westfield": _token(sub="west", preferred_username="west", jti="w"),
              "eastbrook": _token(sub="east", preferred_username="east", jti="e")}
    held: list[tuple[str, str, dict]] = []  # (the token the client carried, tool, request), one per call
    built: list[tuple[str, object]] = []  # (the token an adapter was built with, the settings it was built from)
    lock = threading.Lock()
    barrier = {"now": threading.Barrier(2, timeout=20), "first": set()}
    alone: list[str] = []  # a request whose first call found no other in flight
    real_call = SupersetMcp.call
    real_executor = module.superset_executor

    def recording_call(self, tool, request, **kwargs):
        token = self.credential.value
        with lock:
            held.append((token, tool, copy.deepcopy(request)))
            first = token not in barrier["first"]
            barrier["first"].add(token)
        if first:  # each request's first call waits for the other's: the two are in flight at once
            try:
                barrier["now"].wait()
            except threading.BrokenBarrierError:
                alone.append(tool)
        return real_call(self, tool, request, **kwargs)

    def recording_executor(settings, delegation):
        with lock:
            built.append((delegation.value, settings))
        return real_executor(settings, delegation)

    monkeypatch.setattr(SupersetMcp, "call", recording_call)
    monkeypatch.setattr(module, "superset_executor", recording_executor)

    def start(word: str):
        return client.post("/v1/superset/builds", json={"goal": f"Policies by region for {word}"},
                           headers=_bearer(tokens[word]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        started = dict(zip(tokens, pool.map(start, tokens)))
    assert all(r.status_code == 201 for r in started.values()), {w: r.text for w, r in started.items()}
    builds = {word: r.json()["build"]["build_id"] for word, r in started.items()}

    barrier.update(now=threading.Barrier(2, timeout=20), first=set())
    with ThreadPoolExecutor(max_workers=2) as pool:
        stepped = dict(zip(tokens, pool.map(lambda word: _next(client, builds[word], tokens[word]), tokens)))
    assert all(r.status_code == 200 and r.json()["build"]["ran"] == "dashboard" for r in stepped.values()), stepped

    # Each request built one adapter, from its own token and the deployment's settings.
    assert Counter(token for token, _ in built) == Counter({tokens["westfield"]: 2, tokens["eastbrook"]: 2})
    assert all(settings is runtime.settings for _, settings in built)
    # Whatever names a request's goal was sent with that request's token: the dashboards its plan looked for, the
    # dashboard its first step made.
    for word, token in tokens.items():
        named = [(t, tool) for t, tool, request in held if word in json.dumps(request)]
        assert {tool for _, tool in named} >= {"list_dashboards", TOOL_PREFIX + "create_dashboard"}, named
        assert {t for t, _ in named} == {token}
        on_the_wire = [call.authorization for call in fake.calls if word in json.dumps(call.request)]
        assert on_the_wire and set(on_the_wire) == {f"Bearer {token}"}
    # And the wire carried exactly the tokens the clients did, call for call.
    assert Counter(call.authorization for call in fake.calls) == Counter(f"Bearer {t}" for t, _, _ in held)
    assert {t for t, _, _ in held} == set(tokens.values())
    assert {DelegatedToken(t) for t, _ in built} == {DelegatedToken(t) for t in tokens.values()}
    assert alone == []  # and the two requests of each pair were in flight together

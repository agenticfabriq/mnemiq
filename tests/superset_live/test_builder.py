"""Superset mode's dashboard builder, live: builds driven through mnemiq's own build routes in process, against the
suite's Superset 7.0.0rc2 with mnemiq's extension loaded, each request carrying a token Superset's mint route made for
building (`Live.minted(username, "build")`), and the build store on the live Postgres, in a database this module
makes and drops.

A build's plan comes from a stub planner (`FakePlanner`), which returns a fixed plan and asks no model: no model client
is made or called while a build runs, and the stub model is asked nothing. Everything after the planner is mnemiq's
own: the plan check (a written measure a saved metric says the same as is replaced by it), the build's steps, each
made by the extension's create carrying its step's marker, the layout and the value check.

The module makes a table of its own in the live source database, of a name no run has used, before it enriches: so
the snapshot holds a table no Superset dataset exists for, which a build makes one for. The module removes what that
build made, and the table, once it is done.

Each test states its expectation in its docstring. A build's record (each request's step, its key, kind and status,
the Superset calls it made and its latency; each step's id and whether what it made carries its marker; each chart's
id, type and check status) is printed as an `OBSERVED` line, which `pytest -rA` shows for passing tests too, and
written as JSON to the directory `MNEMIQ_TEST_SUPERSET_RECORD_DIR` names when it is set. No token, key, password or
value from the data goes into it: a comparison of numbers is asserted as a bool, so a failure does not print them
either.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from mnemiq.server.superset_refusals import SIGN_IN_AGAIN
from mnemiq.superset.build_store import BUILD_TABLE, MEASURE_TABLE
from mnemiq.superset.builder import start_build
from mnemiq.superset.chart_form import _NOT_CHECKED_TYPES
from mnemiq.superset.charts import ChartSpec, Family, WrittenRef, generate_chart_request
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetRefusedCaller, SupersetUnavailable
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.marker import Marker, format_marker, parse_marker
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan, FakePlanner, PlannedDataset
from mnemiq.superset.saved_metrics import normalized
from tests.superset_extension_path import use_extension_source
from tests.superset_live.live import (
    Call,
    Recorder,
    carrying,
    enrich_and_build,
    fingerprint_of,
    ground_truth,
    metadata,
    record_calls,
    run_setup,
    superset_settings,
)
from tests.test_superset_build_store import STORE_COLUMNS, value_shaped_names

use_extension_source()

from agenticfabriq.mnemiq.token import BUILD_SCOPES  # noqa: E402

pytestmark = pytest.mark.integration

BUILD = " ".join(BUILD_SCOPES)
# The builder of the six-chart build and the funnel: Alpha and sql_lab, narrowed by no row rule.
BUILDER = "analyst"
GOAL = "sales by region and month"
# The build store's own database on the live Postgres, made by this module and dropped after it.
STORE_DATABASE = "mnemiq_build_store_live"
# The saved metric planted on the `policy` dataset before the six-chart build, and removed after the module: what
# that build's `sales` measure says, so the plan check reuses it.
PLANTED = ("total_sales", "SUM(premium)")
RECORD_DIR = "MNEMIQ_TEST_SUPERSET_RECORD_DIR"
# What a build answer's step may hold (`superset_build._build`).
STEP_FIELDS = {"key", "kind", "status", "object_id", "started_at", "lease_until"}
BUILD_FIELDS = {"build_id", "status", "dashboard_id", "steps", "checks", "ran", "message"}
# The calls that change something in Superset: MCP's own creates and the extension's write tools.
WRITES = {"generate_chart", "create_dataset", "create_virtual_dataset",
          *(TOOL_PREFIX + base for base in ("create_dashboard", "create_dataset", "create_chart", "patch_chart",
                                            "add_saved_metric", "delete_dataset", "set_layout"))}
CREATE_DATASET = TOOL_PREFIX + "create_dataset"
CREATE_CHART = TOOL_PREFIX + "create_chart"
# The probe a build's start sends (`require_build_tools`): the extension's `create_dataset` with an empty request,
# which it refuses without touching Superset.
PROBE = CREATE_DATASET
# What a build never calls: MCP's own creates, the claims an older extension offered, and a patch after a create.
NEVER_CALLED = {"generate_chart", "create_dataset", "create_virtual_dataset",
                *(TOOL_PREFIX + base for base in ("claim_chart", "claim_dataset", "patch_chart"))}
# The table each kind of object a build makes is kept in, in Superset's metadata database.
TABLES = {"dashboard": "dashboards", "dataset": "tables", "chart": "slices"}


def observe(test: str, **facts) -> None:
    print(f"OBSERVED {test} {json.dumps(facts, default=str, sort_keys=True)}")


def record(name: str, summary: dict) -> None:
    """A build's record, printed, and written as `<name>.json` to the directory `MNEMIQ_TEST_SUPERSET_RECORD_DIR`
    names when it is set."""
    observe(name, **summary)
    folder = os.environ.get(RECORD_DIR, "").strip()
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n")


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def as_user(live, username: str) -> SupersetMcp:
    """An MCP client carrying a token the test identity provider signed for `username` with a build token's scopes:
    the suite's own reads and fixtures, never a build's calls."""
    return SupersetMcp(live.mcp_url, DelegatedToken(live.token(username, scope=BUILD)), timeout_s=90)


def user_id(username: str) -> int:
    (row,) = metadata("SELECT id FROM ab_user WHERE username = %s", (username,))
    return row[0]


def label_of(username: str) -> str:
    (row,) = metadata("SELECT s.label FROM subjects s JOIN ab_user u ON u.id = s.user_id WHERE u.username = %s",
                      (username,))
    return row[0]


def made_by(username: str) -> dict[str, set[int]]:
    """The ids of every dashboard, chart and dataset `username` created, in the trash or not."""
    uid = user_id(username)
    return {kind: {row[0] for row in metadata(f"SELECT id FROM {table} WHERE created_by_fk = %s", (uid,))}
            for kind, table in (("dashboards", "dashboards"), ("charts", "slices"), ("datasets", "tables"))}


def saved_metrics(dataset_id: int) -> dict[str, str]:
    return {name: expression for name, expression in
            metadata("SELECT metric_name, expression FROM sql_metrics WHERE table_id = %s", (dataset_id,))}


def stored_chart(chart_id: int) -> dict:
    (row,) = metadata("SELECT params, description, viz_type, created_by_fk, deleted_at FROM slices WHERE id = %s",
                      (chart_id,))
    return {"params": json.loads(row[0] or "{}"), "description": row[1], "viz_type": row[2], "created_by": row[3],
            "deleted_at": row[4]}


def editor_ids(table: str, column: str, object_id: int) -> list[int]:
    return [row[0] for row in metadata(f"SELECT subject_id FROM {table} WHERE {column} = %s ORDER BY 1",
                                       (object_id,))]


def user_subject(username: str) -> int:
    (row,) = metadata("SELECT s.id FROM subjects s JOIN ab_user u ON u.id = s.user_id WHERE u.username = %s "
                      "AND s.type = 1", (username,))
    return row[0]


def chart_metrics(params: dict) -> list:
    """The metrics a saved chart's params name: `metric` for a one-metric chart, else `metrics`."""
    return [params["metric"]] if params.get("metric") else list(params.get("metrics") or [])


def short(tool: str) -> str:
    return tool.replace(TOOL_PREFIX, "ext.")


def writes(calls: list[Call]) -> list[str]:
    """The tools among `calls` that change something in Superset: every write tool's call but the start's probe, an
    empty `create_dataset` that the extension refuses."""
    return [call.tool for call in calls if call.tool in WRITES and not (call.tool == PROBE and call.request == {})]


def step_marker(build_id: str, key: str, kind: str, dashboard_id: int | None) -> str:
    """The marker a build's step gives what it makes: the dashboard shell's own (`new`), else its step's for the
    dashboard."""
    return format_marker(build_id, key, None if kind == "dashboard" else dashboard_id)


# --- the deployment ------------------------------------------------------------------------------------------------


@dataclass
class Served:
    client: object
    runtime: object
    store: object
    store_dsn: str

    def plan_with(self, plan: BuildPlan) -> FakePlanner:
        """The default profile planning `plan`, every time it is asked, with no model."""
        planner = FakePlanner(plan)
        self.runtime.profiles[self.runtime.default_profile].planner = planner
        return planner

    def raw_steps(self, build_id: str) -> list[dict]:
        """The build's steps as the store's row holds them, read on a connection of the test's own."""
        import psycopg

        with psycopg.connect(self.store_dsn, autocommit=True, connect_timeout=10) as con:
            row = con.execute(f"SELECT steps FROM {BUILD_TABLE} WHERE build_id = %s", (build_id,)).fetchone()
        assert row is not None, f"the store holds no build {build_id}"
        return row[0]

    def raw_plan(self, build_id: str) -> dict:
        import psycopg

        with psycopg.connect(self.store_dsn, autocommit=True, connect_timeout=10) as con:
            (plan,) = con.execute(f"SELECT plan FROM {BUILD_TABLE} WHERE build_id = %s", (build_id,)).fetchone()
        return plan

    def builds(self) -> set[str]:
        """The ids of every build the store keeps: none before the store first connects and makes its tables."""
        import psycopg

        with psycopg.connect(self.store_dsn, autocommit=True, connect_timeout=10) as con:
            if con.execute("SELECT to_regclass(%s)", (BUILD_TABLE,)).fetchone()[0] is None:
                return set()
            return {str(row[0]) for row in con.execute(f"SELECT build_id FROM {BUILD_TABLE}")}


@pytest.fixture(scope="module")
def store_dsn(live) -> Iterator[str]:
    """A database of the build store's own on the live Postgres, made here and dropped once the module is done."""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    def admin(statement) -> None:
        with psycopg.connect(live.pg_dsn, autocommit=True, connect_timeout=10) as con:
            con.execute(statement)

    name = sql.Identifier(STORE_DATABASE)
    admin(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(name))
    admin(sql.SQL("CREATE DATABASE {}").format(name))
    try:
        yield make_conninfo(live.pg_dsn, dbname=STORE_DATABASE)
    finally:
        admin(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(name))


@dataclass
class SalesTable:
    """The module's own table in the live source database, and what a build made on it, which teardown removes."""

    name: str
    dataset_id: int | None = None
    dashboard_id: int | None = None
    chart_ids: list[int] = field(default_factory=list)


@pytest.fixture(scope="module")
def sales_table(live) -> Iterator[SalesTable]:
    """A table of a name no run has used (`live_sales_<random>`) in the live source database's `public` schema: twelve
    synthetic sales, each a region and an amount with cents. Made before the module enriches, so its snapshot holds
    it, and no Superset dataset exists for it. Once the module is done, what a build made on it is removed as its
    builder removes it (each chart through MCP's `delete_chart`, then the dataset through the extension's
    `delete_dataset`, which moves it to Superset's trash), and the table is dropped."""
    import psycopg
    from psycopg import sql

    table = SalesTable(f"live_sales_{uuid.uuid4().hex[:8]}")
    name = sql.Identifier("public", table.name)
    with psycopg.connect(live.pg_dsn, autocommit=True, connect_timeout=10) as con:
        con.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY, region text NOT NULL, "
                            "amount numeric(10, 2) NOT NULL)").format(name))
        con.execute(sql.SQL("INSERT INTO {} (id, region, amount) SELECT g, (ARRAY['west', 'east', 'central'])"
                            "[1 + g % 3], round(12.5 + g * 7.31, 2) FROM generate_series(1, 12) AS g").format(name))
        con.execute(sql.SQL("ANALYZE {}").format(name))
    problems: list = []
    try:
        yield table
    finally:
        builder = as_user(live, BUILDER)
        for chart_id in table.chart_ids:
            removed = builder.call("delete_chart", {"identifier": chart_id})
            if removed.get("success") is False:
                problems.append(("delete_chart", chart_id, removed.get("error_type")))
        if table.dataset_id is not None:
            answer = builder.call(TOOL_PREFIX + "delete_dataset", {"dataset_id": table.dataset_id,
                                                                   "dashboard_id": table.dashboard_id})
            if answer != {"ok": True, "soft_deleted": True}:
                problems.append(("delete_dataset", table.dataset_id, answer))
        with psycopg.connect(live.pg_dsn, autocommit=True, connect_timeout=10) as con:
            con.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(name))
        assert not problems, problems


@pytest.fixture(scope="module")
def enriched(live, model, tmp_path_factory, sales_table):
    """The suite's `enriched` (`conftest.py`), with the module's own table made first, so the snapshot holds it: the
    store every test here answers from, enriched through Superset as the service user and built."""
    settings = superset_settings(live, model, tmp_path_factory.mktemp("superset-store") / "store.duckdb")
    return settings, enrich_and_build(settings, live.service_key_file)


@pytest.fixture(scope="module")
def served(live, model, enriched, store_dsn) -> Iterator[Served]:
    """What `mnemiq serve --http` serves in Superset mode for the enriched store, its control Postgres the module's
    build store: `serve_http` checks the settings and builds the runtime and the doors; uvicorn alone is replaced. The
    runtime and the build store it opened are kept, so a test can give the default profile its stub planner."""
    from fastapi.testclient import TestClient

    import mnemiq.runtime as runtime_module
    import mnemiq.server.app as app_module
    import mnemiq.server.superset_build as build_module

    settings, _ = enriched
    settings = settings.model_copy(update={"control_dsn": store_dsn})
    built: dict = {}
    with pytest.MonkeyPatch.context() as patch:
        real_runtime, real_store = runtime_module.build_runtime, build_module._open_store
        patch.setattr(runtime_module, "build_runtime", lambda s: built.setdefault("runtime", real_runtime(s)))
        patch.setattr(build_module, "_open_store", lambda s: built.setdefault("store", real_store(s)))
        patch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
        app_module.serve_http(settings, host="127.0.0.1", port=8080)
    try:
        yield Served(TestClient(built["app"]), built["runtime"], built["store"], store_dsn)
    finally:
        built["store"].close()


@pytest.fixture(scope="module")
def policy_id(live) -> int:
    rows = metadata("SELECT id FROM tables WHERE table_name = 'policy' AND schema = 'public' AND deleted_at IS NULL "
                    "AND (sql IS NULL OR sql = '')")
    assert len(rows) == 1, rows
    return rows[0][0]


@pytest.fixture(scope="module")
def planted(live, policy_id) -> Iterator[str]:
    """`total_sales`, `SUM(premium)`, planted on the `policy` dataset by an admin through the extension's
    `add_saved_metric`, and removed by Superset's own command once the module is done (`setup_superset.py
    drop-metric`). One left by an earlier run is removed first; any other saved metric of that definition fails here,
    since the plan check would reuse it instead (`reusable`: certified first, then by name)."""
    name, expression = PLANTED
    run_setup("drop-metric", str(policy_id), name)
    others = [n for n, e in saved_metrics(policy_id).items() if normalized(e) == normalized(expression)]
    assert not others, f"the policy dataset already has saved metrics of {expression}: {others}; remove them first"
    admin = as_user(live, "admin")
    changed_on = admin.call("get_dataset_info", {"identifier": policy_id, "select_columns": ["id", "changed_on"]})
    answer = admin.call(TOOL_PREFIX + "add_saved_metric", {
        "dataset_id": policy_id, "metric": {"metric_name": name, "expression": expression},
        "expect_changed_on": changed_on["changed_on"]})
    assert answer == {"ok": True}, answer
    assert saved_metrics(policy_id).get(name) == expression
    try:
        yield name
    finally:
        assert run_setup("drop-metric", str(policy_id), name) == "dropped 1"
        assert name not in saved_metrics(policy_id)


# --- a build, run through the routes -------------------------------------------------------------------------------


@dataclass
class Request:
    """One request of a build: what it was, its answer, how long it took, the token it carried (held in memory only,
    to compare each call's fingerprint with) and every Superset call it made."""

    label: str
    status_code: int
    body: dict
    text: str
    seconds: float
    token: str
    calls: list[Call]

    @property
    def build(self) -> dict:
        return self.body.get("build") or {}


@dataclass
class BuildRun:
    name: str
    started: str
    requests: list[Request] = field(default_factory=list)
    ended: str | None = None
    stopped: str | None = None  # why the loop ended before the build did
    model_before: tuple = ()
    model_after: tuple = ()
    llm_made: int = 0
    llm_asked: int = 0
    # Every lease id a step held while the build ran, as its claim answered it and as each renewal named it.
    leases: set[str] = field(default_factory=set)
    planner: FakePlanner | None = None

    @property
    def final(self) -> dict:
        built = [r.build for r in self.requests if r.status_code in (200, 201) and r.build]
        return built[-1] if built else {}

    @property
    def last_step(self) -> dict:
        """The answer of the last `/next` that ran a step: the one that says how the build ended."""
        ran = [r.build for r in self.requests if r.label == "next" and r.status_code == 200 and r.build.get("ran")]
        return ran[-1] if ran else {}

    @property
    def build_id(self) -> str | None:
        return self.final.get("build_id")


def _send(served: Served, recorder: Recorder, run: BuildRun, label: str, method: str, path: str, token: str,
          body: dict | None = None) -> Request:
    mark = recorder.mark()
    started = time.perf_counter()
    if method == "POST":
        answer = served.client.post(path, json=body, headers=bearer(token))
    else:
        answer = served.client.get(path, headers=bearer(token))
    seconds = time.perf_counter() - started
    try:
        parsed = answer.json()
    except ValueError:
        parsed = {}
    request = Request(label, answer.status_code, parsed if isinstance(parsed, dict) else {}, answer.text, seconds,
                      token, recorder.since(mark))
    run.requests.append(request)
    return request


def run_build(live, served: Served, model, plan: BuildPlan, name: str, *, builder: str = BUILDER, goal: str = GOAL,
              stop_after: int | None = None, before_next=None, **body) -> BuildRun:
    """The build of `plan` started and run one step per `/next` until it ends (or `stop_after` steps), each request
    with a token minted for it, every Superset call recorded, and the model's clients counted. `before_next`, when
    given, runs once the start has answered 201 and before the first `/next`. Bounded: one `/next` per step and three
    more; a step left to its lease (409 on the next request) stops the loop and says so."""
    from mnemiq.llm.client import LLMClient

    run = BuildRun(name=name, started=datetime.now(UTC).isoformat(timespec="seconds"))
    run.planner = served.plan_with(plan)
    run.model_before = (model.calls, model.embedding_calls, served.runtime.usage().calls)
    made, asked = [], []
    with pytest.MonkeyPatch.context() as patch:
        recorder = record_calls(patch)
        real_init, real_complete = LLMClient.__init__, LLMClient.complete

        def init(self, *args, **kwargs):
            made.append(1)
            real_init(self, *args, **kwargs)

        def complete(self, *args, **kwargs):
            asked.append(1)
            return real_complete(self, *args, **kwargs)

        patch.setattr(LLMClient, "__init__", init)
        patch.setattr(LLMClient, "complete", complete)
        store_type = type(served.store)
        real_claim, real_renew = store_type.claim_next_step, store_type.renew_lease

        def claim(self, *args, **kwargs):
            claimed = real_claim(self, *args, **kwargs)
            if getattr(claimed, "lease_id", None):
                run.leases.add(claimed.lease_id)
            return claimed

        def renew(self, tenant, build_id, key, lease_id, *args, **kwargs):
            run.leases.add(lease_id)
            return real_renew(self, tenant, build_id, key, lease_id, *args, **kwargs)

        patch.setattr(store_type, "claim_next_step", claim)
        patch.setattr(store_type, "renew_lease", renew)
        start = _send(served, recorder, run, "start", "POST", "/v1/superset/builds", live.minted(builder, "build"),
                      {"goal": goal, **body})
        if start.status_code == 201:
            build_id = start.build["build_id"]
            if before_next is not None:
                before_next()
            limit = len(start.build["steps"]) + 3 if stop_after is None else stop_after
            for _ in range(limit):
                step = _send(served, recorder, run, "next", "POST", f"/v1/superset/builds/{build_id}/next",
                             live.minted(builder, "build"))
                if step.status_code == 409:
                    run.stopped = f"409 after {run.requests[-2].build.get('ran')!r}: {step.body.get('detail')}"
                    break
                if step.status_code != 200 or step.build.get("status") != "building":
                    break
                if "left to run out" in (step.build.get("message") or ""):
                    run.stopped = f"step {step.build.get('ran')!r} left to its lease: {step.build.get('message')}"
                    break
            else:
                if stop_after is None:
                    run.stopped = f"the build had not ended after {limit} requests"
            _send(served, recorder, run, "read", "GET", f"/v1/superset/builds/{build_id}",
                  live.minted(builder, "build"))
    run.ended = datetime.now(UTC).isoformat(timespec="seconds")
    run.model_after = (model.calls, model.embedding_calls, served.runtime.usage().calls)
    run.llm_made, run.llm_asked = len(made), len(asked)
    return run


def summary(run: BuildRun, served: Served, plan: BuildPlan | None = None) -> dict:
    """The build's record: each request (its step's key, kind and status, its HTTP status, latency, and the Superset
    calls it made by tool, with their JSON-RPC requests), each step's id and whether what it made carries its marker
    (None for a step that makes nothing: a layout, a check, or the dashboard step of a build on an existing dashboard,
    whose intent names no title), and each chart's id, type and check status. Ids, names, statuses and timings only:
    never a value, a token or a lease id."""
    final = run.final
    kinds = {step["key"]: step["kind"] for step in final.get("steps", [])}
    raw = {step["key"]: step for step in served.raw_steps(run.build_id)} if run.build_id else {}
    requests = []
    for request in run.requests:
        ran = request.build.get("ran")
        step = next((s for s in request.build.get("steps", []) if s["key"] == ran), None)
        requests.append({
            "request": request.label, "http": request.status_code, "ran": ran, "kind": kinds.get(ran),
            "step_status": step["status"] if step else None, "seconds": round(request.seconds, 3),
            "superset_calls": len(request.calls),
            "superset_rpc_requests": sum(len(call.rpc) for call in request.calls),
            "superset_seconds": round(sum(call.seconds for call in request.calls), 3),
            "tools": dict(sorted(Counter(short(call.tool) for call in request.calls).items())),
            **({"detail": request.body["detail"]} if "detail" in request.body else {}),
            **({"refused": request.body["refused"]} if "refused" in request.body else {}),
        })
    dashboard_id = final.get("dashboard_id")

    def marked(key: str, step: dict) -> bool | None:
        if step["kind"] not in TABLES or step["object_id"] is None or dashboard_id is None:
            return None
        if step["kind"] == "dashboard" and (step.get("intent") or {}).get("name") is None:
            return None
        marker = step_marker(run.build_id, key, step["kind"], dashboard_id)
        return [found for found, _ in carrying(TABLES[step["kind"]], marker)] == [step["object_id"]]

    steps = [{"key": key, "kind": step["kind"], "status": step["status"], "object_id": step["object_id"],
              "marked": marked(key, step), "reason": step.get("reason")} for key, step in raw.items()]
    families = {chart.key: chart.family.value for chart in plan.charts} if plan else {}
    charts = []
    for key, step in raw.items():
        if step["kind"] == "chart" and step["object_id"] is not None:
            chart_id = step["object_id"]
            charts.append({"key": key, "chart_id": chart_id, "planned_type": families.get(key),
                           "type": stored_chart(chart_id)["viz_type"],
                           "check": final.get("checks", {}).get(str(chart_id))})
    return {"build_id": run.build_id, "dashboard_id": final.get("dashboard_id"), "status": final.get("status"),
            "planner_asked": len(run.planner.goals) if run.planner else None, "started": run.started, "ended": run.ended, "stopped": run.stopped,
            "message": run.last_step.get("message"), "leases_held": len(run.leases),
            "requests": requests, "steps": steps, "charts": charts,
            "model_calls": {"chat": run.model_after[0] - run.model_before[0],
                            "embeddings": run.model_after[1] - run.model_before[1],
                            "profile_usage": run.model_after[2] - run.model_before[2],
                            "llm_clients_made": run.llm_made, "llm_completions": run.llm_asked}}


def never_called(run: BuildRun) -> list[tuple[str, str]]:
    """Each call of the build to a tool it never calls, with its request's label: none, for a build that made every
    object through the extension's creates."""
    return [(request.label, call.tool) for request in run.requests for call in request.calls
            if call.tool in NEVER_CALLED]


def assert_no_model_call(run: BuildRun) -> None:
    assert run.llm_made == 0 and run.llm_asked == 0, (run.llm_made, run.llm_asked)
    assert run.model_after == run.model_before, (run.model_before, run.model_after)


def assert_ran_to_the_end(run: BuildRun) -> None:
    assert run.stopped is None, run.stopped
    failed = [(r.label, r.status_code, r.body.get("detail")) for r in run.requests if r.status_code not in (200, 201)]
    assert not failed, failed
    final = run.final
    assert final.get("status") == "done", (final.get("status"), final.get("message"))
    assert all(step["status"] == "done" for step in final["steps"]), [(s["key"], s["status"]) for s in final["steps"]]


# --- the numbers, read from the source -----------------------------------------------------------------------------

# What the six-chart build's charts show, read from the source directly: every one a sum of `premium`, which has cents.
SALES_TRUTH = (
    "SELECT sum(premium) FROM policy",
    "SELECT region, sum(premium) FROM policy GROUP BY 1",
    "SELECT status, sum(premium) FROM policy GROUP BY 1",
    "SELECT date_trunc('month', issued), sum(premium) FROM policy GROUP BY 1",
    "SELECT region, status, sum(premium) FROM policy GROUP BY 1, 2",
)


def sales_values(live) -> list[Decimal]:
    """Every sum the six-chart build's charts show that has a fractional part: a number of the data no id, count or
    time spells."""
    found = {row[-1] for sql in SALES_TRUTH for row in ground_truth(live, sql)}
    return sorted(value for value in found if isinstance(value, Decimal) and value != value.to_integral_value())


def spellings(value: Decimal) -> set[str]:
    """How a sum may be written in JSON or text: to the cent, as a float writes it, and with trailing zeros cut."""
    cents = f"{value:.2f}"
    return {cents, repr(float(value)), cents.rstrip("0").rstrip(".")}


def found_values(text: str, values: list[Decimal]) -> int:
    """How many of `values` appear in `text` as a number of their own (not inside a longer one)."""
    return sum(1 for value in values
               if any(re.search(rf"(?<![\d.]){re.escape(s)}(?!\d)", text) for s in spellings(value)))


def numbers_outside_ids(doc, path: str = "") -> list[str]:
    """Every number in an answer that is not under an id field: where a value would sit."""
    if isinstance(doc, dict):
        return [p for key, item in doc.items() for p in numbers_outside_ids(item, f"{path}.{key}")
                if not (key in ("dashboard_id", "object_id") and isinstance(item, int))]
    if isinstance(doc, list):
        return [p for index, item in enumerate(doc) for p in numbers_outside_ids(item, f"{path}[{index}]")]
    if isinstance(doc, (int, float)) and not isinstance(doc, bool):
        return [path]
    return []


# --- six charts, the planted saved metric reused, every chart checked ----------------------------------------------


def six_chart_plan(policy_id: int) -> BuildPlan:
    """Six charts across five families on the `policy` dataset: two KPI tiles, a monthly line, a bar by region, a pie
    by status and a table by region and status. `sales` is written as the planted metric's own expression, so the plan
    check replaces it with `total_sales`; `policies` is a measure of mnemiq's own. No family of `_NOT_CHECKED_TYPES`."""
    sales = Measure(name="sales", label="Sales", expression="SUM(premium)", kind=MeasureKind.CURRENCY,
                    dataset_id=policy_id, description="Premium written", asked_by="")
    policies = Measure(name="policies", label="Policies", expression="COUNT(id)", kind=MeasureKind.COUNT,
                       dataset_id=policy_id, description="Policies written", asked_by="")

    def chart(key: str, family: Family, title: str, measures: tuple, **more) -> ChartSpec:
        fields = {"dimensions": (), "time_column": None, "time_grain": None, "filters": (), "row_limit": None,
                  "source": None, "target": None, **more}
        return ChartSpec(key=key, family=family, title=title, dataset_id=policy_id,
                         measures=tuple(WrittenRef(m) for m in measures), **fields)

    return BuildPlan(charts=(
        chart("sales_by_month", Family.LINE, "Sales by month", (sales,), time_column="issued", time_grain="P1M"),
        chart("total_sales", Family.BIG_NUMBER, "Total sales", (sales,)),
        chart("sales_by_region", Family.BAR, "Sales by region", (sales,), dimensions=("region",)),
        chart("policies_written", Family.BIG_NUMBER, "Policies written", (policies,)),
        chart("sales_by_status", Family.PIE, "Sales by status", (sales,), dimensions=("status",)),
        chart("sales_by_region_and_status", Family.TABLE, "Sales and policies by region and status", (sales, policies),
              dimensions=("region", "status")),
    ), measures=(sales, policies))


KPI_KEYS = ("total_sales", "policies_written")


@dataclass
class SixCharts:
    run: BuildRun
    plan: BuildPlan
    metrics_after_planting: dict[str, str]
    metrics_after_build: dict[str, str]
    kept_plan: dict
    raw_steps: list[dict]
    summary: dict


@pytest.fixture(scope="module")
def six_charts(live, served, model, policy_id, planted) -> SixCharts:
    """The six-chart build, once, for every test that reads it: the goal "sales by region and month", run to its end as
    `analyst`, and its record."""
    plan = six_chart_plan(policy_id)
    metrics_after_planting = saved_metrics(policy_id)
    run = run_build(live, served, model, plan, "sales-by-region-and-month")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped,
                                                             "requests": [(r.label, r.status_code) for r in run.requests]}
    record("sales-by-region-and-month", found)
    kept_plan = served.raw_plan(run.build_id) if run.build_id else {}
    raw_steps = served.raw_steps(run.build_id) if run.build_id else []
    return SixCharts(run, plan, metrics_after_planting, saved_metrics(policy_id), kept_plan, raw_steps, found)


def test_six_charts_reuse_the_planted_metric_and_every_one_checks(live, six_charts, policy_id, planted):
    """The goal "sales by region and month", planned by the stub as six charts across five families (two KPI
    tiles, a monthly line, a bar, a pie, a table), none of a type mnemiq does not check by rule. Expected:
    - the start answers 201 and each `/next` runs the next step (dashboard, six charts, layout, check) to `done`;
    - the plan kept holds `total_sales` (the planted saved metric) where the plan wrote `sales`, and no written
      `sales`; every chart that shows sales names `total_sales` in its saved params; the policy dataset's saved
      metrics after the build are exactly those after planting: no new metric of that definition, none at all;
    - every chart's check status is `CHECKED` (else: which chart, its type, its status and mnemiq's reason);
    - Superset's own `get_dashboard_layout` places the two KPI tiles first, and all six charts;
    - the bar, the pie and the total show the source's sums, read as the builder through `get_chart_data`;
    - no answer holds a lease id, a field beyond the build's own, or a number outside an id field, and none holds
      any of the sums the charts show; no model client is made or called."""
    run, plan = six_charts.run, six_charts.plan
    assert {chart.family.value for chart in plan.charts}.isdisjoint(_NOT_CHECKED_TYPES)
    assert len(plan.charts) >= 6 and len({chart.family for chart in plan.charts}) >= 4
    assert_ran_to_the_end(run)
    final = run.final
    keys = [step["key"] for step in final["steps"]]
    assert keys == ["dashboard", *(chart.key for chart in plan.charts), "layout", "check"]
    assert [r.build.get("ran") for r in run.requests if r.label == "next"] == keys
    assert run.requests[0].status_code == 201
    ids = {step["key"]: step["object_id"] for step in final["steps"] if step["kind"] == "chart"}
    dashboard_id = final["dashboard_id"]

    # The planted saved metric is the one the charts use.
    kept = six_charts.kept_plan
    for chart in kept["charts"]:
        refs = chart["measures"]
        shows_sales = chart["key"] != "policies_written"
        assert ({"saved": planted, "kind": "currency"} in refs) == shows_sales, (chart["key"], refs)
        assert {"written": "sales"} not in refs
    assert [m["name"] for m in kept["measures"]] == ["policies"]
    for key, chart_id in ids.items():
        metrics = chart_metrics(stored_chart(chart_id)["params"])
        assert (planted in metrics) == (key != "policies_written"), (key, metrics)
    assert six_charts.metrics_after_build == six_charts.metrics_after_planting
    assert [n for n, e in six_charts.metrics_after_build.items()
            if normalized(e) == normalized(PLANTED[1])] == [planted]

    # Every chart checked; a chart that is not is named with its type, status and mnemiq's reason.
    not_checked = [(key, stored_chart(chart_id)["viz_type"], final["checks"].get(str(chart_id)))
                   for key, chart_id in ids.items() if final["checks"].get(str(chart_id)) != "checked"]
    assert not not_checked, (not_checked, run.last_step.get("message"))
    assert run.last_step["ran"] == "check" and run.last_step["message"] == (
        f"every chart checked ({len(ids)} of {len(ids)})")
    read = run.requests[-1]
    assert read.label == "read" and read.build["steps"] == run.last_step["steps"]
    assert read.build["checks"] == run.last_step["checks"]
    assert set(final["checks"]) == {str(chart_id) for chart_id in ids.values()}

    # The layout as Superset's own tool reads it.
    builder = as_user(live, BUILDER)
    layout = builder.call("get_dashboard_layout", {"identifier": dashboard_id})
    placed = [chart["chart_id"] for chart in layout.get("charts") or []]
    assert placed[:len(KPI_KEYS)] == [ids[key] for key in KPI_KEYS], placed
    assert sorted(placed) == sorted(ids.values())

    # Three charts' numbers held to the source, read as the builder.
    def rows(chart_id: int, *columns: str) -> list[tuple]:
        data = builder.call("get_chart_data", {"identifier": chart_id}).get("data") or []
        return sorted(tuple(row.get(c) for c in columns) + (round(float(row[planted]), 2),) for row in data)

    def truth(sql: str) -> list[tuple]:
        return sorted(tuple(row[:-1]) + (round(float(row[-1]), 2),) for row in ground_truth(live, sql))

    same = {"total": rows(ids["total_sales"]) == truth(SALES_TRUTH[0]),
            "by region": rows(ids["sales_by_region"], "region") == truth(SALES_TRUTH[1]),
            "by status": rows(ids["sales_by_status"], "status") == truth(SALES_TRUTH[2])}
    assert all(same.values()), same

    # No answer holds a lease id or a value.
    values = sales_values(live)
    assert len(values) > 20, len(values)
    assert len(run.leases) >= len(keys), len(run.leases)  # a lease was held for every step
    for request in run.requests:
        assert "lease_id" not in request.text and not any(lease in request.text for lease in run.leases)
        assert set(request.body) == {"build"}, request.body.keys()
        assert set(request.build) == BUILD_FIELDS
        assert all(set(step) == STEP_FIELDS for step in request.build["steps"])
        assert numbers_outside_ids(request.body) == [], numbers_outside_ids(request.body)
        assert found_values(request.text, values) == 0, request.label
    assert_no_model_call(run)
    observe("six charts", charts={key: (chart_id, final["checks"][str(chart_id)]) for key, chart_id in ids.items()},
            layout=placed, message=run.last_step["message"])


def test_every_object_a_build_made_carries_its_marker(six_charts):
    """The marker, live. Every object the six-chart build made is the one object in Superset whose description's last
    line is its step's marker, in the trash or not: the dashboard its shell's (`new`), each chart its step's for the
    dashboard. Each was made by the step's one create through the extension (`create_dashboard`, `create_chart`),
    which carried that marker and was answered that id. No request of the build called MCP's `generate_chart`,
    `create_dataset` or `create_virtual_dataset`, a claim, or `patch_chart`: nothing was made another way, taken over
    after it was made, or changed after its create."""
    run = six_charts.run
    assert_ran_to_the_end(run)
    dashboard_id = run.final["dashboard_id"]
    made = {step["key"]: step for step in six_charts.raw_steps if step["kind"] in ("dashboard", "chart")}
    assert len(made) == 1 + len(six_charts.plan.charts)
    for key, step in made.items():
        kind, made_id = step["kind"], step["object_id"]
        marker = step_marker(run.build_id, key, kind, dashboard_id)
        assert carrying(TABLES[kind], marker) == [(made_id, None)], key
        [request] = [r for r in run.requests if r.label == "next" and r.build.get("ran") == key]
        creates = [call for call in request.calls if call.tool == TOOL_PREFIX + f"create_{kind}"]
        assert len(creates) == 1, (key, [call.tool for call in request.calls])
        assert creates[0].request["marker"] == marker and creates[0].payload == {f"{kind}_id": made_id}, key
    assert never_called(run) == []
    observe("markers", made={key: step["object_id"] for key, step in made.items()})


def test_every_superset_call_of_a_build_carries_its_requests_token(six_charts):
    """Across the whole six-chart build, each request carries a token minted for it alone, and every
    Superset call the request made carries that request's token and no other's (the Recorder's fingerprint of the
    client's credential, call by call). Every request but a read asks Superset something, and the read asks
    nothing."""
    run = six_charts.run
    assert_ran_to_the_end(run)
    tokens = [request.token for request in run.requests]
    assert len(set(tokens)) == len(tokens)
    for request in run.requests:
        held = {call.credential for call in request.calls}
        if request.label == "read":
            assert held == set(), held
        else:
            assert held == {f"DelegatedToken:{fingerprint_of(request.token)}"}, (request.label, held)
    observe("tokens", requests=len(run.requests), calls=sum(len(r.calls) for r in run.requests))


def test_the_live_build_store_holds_no_value(live, six_charts, served):
    """After the six-chart build, the store's no-value guard (`value_shaped_names`) read over the live store finds no
    column and no jsonb key at any depth named as a value, a row or a sample is kept, having read every column of both
    tables and every key of their jsonb documents; and no stored row's text holds any sum those charts show, read from
    the source.
    The controls, each in a transaction rolled back: a value-shaped key planted in a stored check status is flagged,
    and one of those sums planted in a stored goal is found."""
    import psycopg
    from psycopg import sql

    assert_ran_to_the_end(six_charts.run)
    values = sales_values(live)

    def stored_text(con) -> str:
        return "\n".join(text for table in STORE_COLUMNS for (text,) in con.execute(
            sql.SQL("SELECT row_to_json(t)::text FROM {} AS t").format(sql.Identifier(table))))

    with psycopg.connect(served.store_dsn, connect_timeout=10) as con:
        flagged, read, keys = value_shaped_names(con)
        text = stored_text(con)
        assert flagged == [], flagged
        assert read == STORE_COLUMNS, read
        assert keys > 100, keys
        assert six_charts.run.build_id in text and GOAL in text  # the probe reads what the build kept
        assert found_values(text, values) == 0
        con.execute(sql.SQL("UPDATE {} SET check_status = check_status || '{{\"1\": {{\"value\": 1}}}}'").format(
            sql.Identifier(MEASURE_TABLE)))
        con.execute(sql.SQL("UPDATE {} SET goal = goal || %s").format(sql.Identifier(BUILD_TABLE)),
                    (f" {values[0]:.2f}",))
        planted_flags = value_shaped_names(con)[0]
        planted_found = found_values(stored_text(con), values)
        con.rollback()
        assert value_shaped_names(con)[0] == [] and found_values(stored_text(con), values) == 0
    assert any(flag.startswith(f"{MEASURE_TABLE}.check_status.1.value") for flag in planted_flags), planted_flags
    assert planted_found == 1, planted_found
    observe("store", keys_read=keys, sums_looked_for=len(values), columns=read)


def _expired(live, token: str) -> str:
    """A minted token's own claims with its times moved an hour back, signed with the key Superset's mint route signs
    with (`MNEMIQ_TOKEN_KEY_FILE` in `superset_config.py`, the test identity provider's): the token the route made an
    hour ago."""
    import jwt

    claims = jwt.decode(token, (live.keys / "public.pem").read_bytes(), algorithms=["RS256"],
                        audience=live.audience, issuer=live.issuer)
    now = int(time.time())
    moved = {**claims, "iat": now - 3600, "nbf": now - 3600, "exp": now - 1800}
    return jwt.encode(moved, (live.keys / "private.pem").read_bytes(), algorithm="RS256",
                      headers={"kid": jwt.get_unverified_header(token)["kid"]})


def one_bar(policy_id: int) -> BuildPlan:
    policies = Measure(name="policies", label="Policies", expression="COUNT(id)", kind=MeasureKind.COUNT,
                       dataset_id=policy_id, description="Policies written", asked_by="")
    return BuildPlan(charts=(ChartSpec(key="policies_by_region", family=Family.BAR, title="Policies by region",
                                       dataset_id=policy_id, measures=(WrittenRef(policies),), dimensions=("region",),
                                       time_column=None, time_grain=None, filters=(), row_limit=None, source=None,
                                       target=None),), measures=(policies,))


def test_an_expired_minted_token_is_refused_at_mnemiqs_door(live, served, policy_id, recorder):
    """A token Superset's mint route made for building starts a build (201: the control); the
    same token expired is refused at mnemiq's door, 401 with a Bearer challenge, on `POST /v1/superset/builds` and on
    that build's `/next`, and neither asks Superset anything: no call, no step claimed, no second build."""
    served.plan_with(one_bar(policy_id))
    minted = live.minted(BUILDER, "build")
    started = served.client.post("/v1/superset/builds", json={"goal": "policies by region"}, headers=bearer(minted))
    assert started.status_code == 201, started.text
    build_id = started.json()["build"]["build_id"]
    before = served.raw_steps(build_id)
    expired = _expired(live, minted)
    mark = recorder.mark()
    refused = {"start": served.client.post("/v1/superset/builds", json={"goal": "policies by region"},
                                           headers=bearer(expired)),
               "next": served.client.post(f"/v1/superset/builds/{build_id}/next", headers=bearer(expired))}
    for route, answer in refused.items():
        assert answer.status_code == 401, (route, answer.status_code)
        assert answer.headers["www-authenticate"] == "Bearer", route
    assert recorder.since(mark) == []
    assert served.raw_steps(build_id) == before
    assert all(step["status"] == "pending" and step["lease_id"] is None for step in before)
    observe("expired token", **{route: (a.status_code, a.json().get("detail")) for route, a in refused.items()})


def test_an_unknown_users_token_is_refused_by_superset_in_mnemiqs_words(live, served, policy_id, recorder):
    """A token the identity provider validly signed, with a build token's scopes, for `nobody`, a
    user Superset has no account for, passes mnemiq's door and is refused by Superset. Expected, on
    `POST /v1/superset/builds` and on `/next` of a build kept for nobody's principal: 401, mnemiq's sign-in sentence
    (`SIGN_IN_AGAIN`) and a Bearer challenge, and none of Superset's text. The client read Superset's answer as
    `SupersetRefusedCaller`; the kept build's steps are untouched."""
    served.plan_with(one_bar(policy_id))
    token = live.token("nobody", scope=BUILD)
    whoami = served.client.get("/v1/whoami", headers=bearer(token)).json()
    kept = start_build(served.store, tenant=whoami["tenant_id"], principal=whoami["principal_id"],
                       goal="policies by region", plan=one_bar(policy_id))
    before = served.raw_steps(kept.build_id)
    mark = recorder.mark()
    answers = {"start": served.client.post("/v1/superset/builds", json={"goal": "policies by region"},
                                           headers=bearer(token)),
               "next": served.client.post(f"/v1/superset/builds/{kept.build_id}/next", headers=bearer(token))}
    calls = recorder.since(mark)
    refusals = [call.raised for call in calls if call.raised is not None]
    assert refusals and all(isinstance(raised, SupersetRefusedCaller) for raised in refusals), refusals
    said = {raised.detail for raised in refusals}
    for route, answer in answers.items():
        assert answer.status_code == 401, (route, answer.status_code, answer.text)
        assert answer.json()["detail"] == SIGN_IN_AGAIN, route
        assert answer.headers["www-authenticate"] == "Bearer"
        assert not any(text and text in answer.text for text in said), route
        assert "nobody" not in answer.text
    assert writes(calls) == []
    assert served.raw_steps(kept.build_id) == before
    observe("unknown user", status={route: a.status_code for route, a in answers.items()},
            detail=SIGN_IN_AGAIN, superset_said=sorted(said), calls=[c.tool for c in calls])


# --- a funnel, made through the extension --------------------------------------------------------------------------


def test_a_funnel_is_made_through_the_extension_and_its_data_read(live, served, model, policy_id):
    """A funnel by chat: the stub's plan holds one funnel (`COUNT(id)` of policies by status). Expected: the build
    runs to `done` (dashboard, funnel, layout, check); the funnel was made by the extension's `create_chart`, never
    MCP's `generate_chart`, a claim or `patch_chart`; Superset holds it as a `funnel` chart created by analyst, analyst
    among its editors and its editors the dashboard's, its description ending in this build's marker; Superset's
    `get_chart_data`, asked as analyst with a minted token, answers one row per status, each count the source's. No
    model client is made or called. Its check status is recorded (rendering is not this PR's)."""
    policies = Measure(name="policies", label="Policies", expression="COUNT(id)", kind=MeasureKind.COUNT,
                       dataset_id=policy_id, description="Policies written", asked_by="")
    funnel = ChartSpec(key="policies_by_status", family=Family.FUNNEL, title="Policies by status",
                       dataset_id=policy_id, measures=(WrittenRef(policies),), dimensions=("status",),
                       time_column=None, time_grain=None, filters=(), row_limit=None, source=None, target=None)
    plan = BuildPlan(charts=(funnel,), measures=(policies,))
    run = run_build(live, served, model, plan, "funnel", goal="policies by status as a funnel")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped}
    record("funnel", found)
    assert_ran_to_the_end(run)
    final = run.final
    [chart_id] = [step["object_id"] for step in final["steps"] if step["key"] == funnel.key]
    dashboard_id = final["dashboard_id"]
    [step_request] = [r for r in run.requests if r.build.get("ran") == funnel.key]
    tools = [call.tool for call in step_request.calls]
    assert tools.count(CREATE_CHART) == 1, tools
    assert never_called(run) == []
    stored = stored_chart(chart_id)
    assert stored["viz_type"] == "funnel" and stored["deleted_at"] is None
    assert stored["created_by"] == user_id(BUILDER)
    assert parse_marker(stored["description"]) == Marker(run.build_id, funnel.key, dashboard_id)
    editors = editor_ids("chart_editors", "chart_id", chart_id)
    assert user_subject(BUILDER) in editors
    assert editors == editor_ids("dashboard_editors", "dashboard_id", dashboard_id)
    data = SupersetMcp(live.mcp_url, DelegatedToken(live.minted(BUILDER, "build")), timeout_s=90).call(
        "get_chart_data", {"identifier": chart_id})
    rows = data.get("data") or []
    shown = sorted((row.get("status"), row.get("Policies")) for row in rows)
    truth = sorted(ground_truth(live, "SELECT status, count(id) FROM policy GROUP BY 1"))
    same = shown == truth
    assert len(rows) == 3 and same, (len(rows), same, sorted(rows[0]) if rows else None)
    assert_no_model_call(run)
    observe("funnel", chart_id=chart_id, dashboard_id=dashboard_id, rows=len(rows),
            check=final["checks"].get(str(chart_id)), message=run.last_step.get("message"))


def test_a_measure_that_is_null_everywhere_is_empty(live, served, model, policy_id):
    """A big number whose written measure is NULL on every row (`MAX` of a CASE no row satisfies). Expected: the premise
    holds (the source's answer is one NULL), the build runs to `done`, and the chart's check is `empty`, never
    `checked`: Superset shows it as "No data"."""
    expression = "MAX(CASE WHEN premium < 0 THEN premium END)"
    assert ground_truth(live, f"SELECT {expression} FROM policy") == [(None,)]
    worst = Measure(name="worst_refund", label="Worst refund", expression=expression, kind=MeasureKind.NUMBER,
                    dataset_id=policy_id, description="Largest negative premium", asked_by="")
    big = ChartSpec(key="worst_refund", family=Family.BIG_NUMBER, title="Worst refund", dataset_id=policy_id,
                    measures=(WrittenRef(worst),), dimensions=(), time_column=None, time_grain=None, filters=(),
                    row_limit=None, source=None, target=None)
    plan = BuildPlan(charts=(big,), measures=(worst,))
    run = run_build(live, served, model, plan, "all null", goal="the worst refund")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped}
    record("all null", found)
    assert_ran_to_the_end(run)
    [chart_id] = [step["object_id"] for step in run.final["steps"] if step["key"] == big.key]
    assert run.final["checks"].get(str(chart_id)) == "empty", run.final["checks"]
    assert_no_model_call(run)


def _percent_big_number(policy_id: int, key: str, expression: str) -> tuple[Measure, ChartSpec]:
    rate = Measure(name=key, label="Rate", expression=expression, kind=MeasureKind.PERCENT, dataset_id=policy_id,
                   description="A rate", asked_by="")
    big = ChartSpec(key=key, family=Family.BIG_NUMBER, title="Rate", dataset_id=policy_id,
                    measures=(WrittenRef(rate),), dimensions=(), time_column=None, time_grain=None, filters=(),
                    row_limit=None, source=None, target=None)
    return rate, big


def test_a_percent_on_a_0_to_100_scale_is_format_wrong(live, served, model, policy_id):
    """A big number whose written percent measure is `SUM(premium) / 1000.0`, which is above 1 on the seed. Expected:
    the premise holds (the source's answer is above 1), the build runs to `done`, and the chart's check is
    `format_wrong`, never `checked`: Superset would show it under the percent format as a hundred times too large. The
    build's message, as the store keeps it, says "above 100%". No model client is made or called."""
    expression = "SUM(premium) / 1000.0"
    [(truth,)] = ground_truth(live, f"SELECT {expression} FROM policy")
    assert truth is not None and abs(truth) > 1, truth
    rate, big = _percent_big_number(policy_id, "scaled_rate", expression)
    plan = BuildPlan(charts=(big,), measures=(rate,))
    run = run_build(live, served, model, plan, "scaled percent", goal="a rate")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped}
    record("scaled percent", found)
    assert_ran_to_the_end(run)
    [chart_id] = [step["object_id"] for step in run.final["steps"] if step["key"] == big.key]
    assert run.final["checks"].get(str(chart_id)) == "format_wrong", run.final["checks"]
    message = run.last_step.get("message") or ""
    assert "format_wrong" in message and "above 100%" in message, message
    assert_no_model_call(run)


def test_a_percent_multiplied_by_100_is_refused_before_anything_is_made(live, served, model, policy_id):
    """A written percent measure `100.0 * AVG(premium) / MAX(premium)`. Expected: the start answers 200 with the
    refused plan and nothing else: its one part is the measure, with mnemiq's reason (a percent is a fraction from 0 to
    1; write it without multiplying by 100). The planner was asked twice (the plan and its one repair). No build is
    kept, no write call is made, and analyst has made nothing new in Superset. No model client is made or called."""
    rate, big = _percent_big_number(policy_id, "scaled_rate", "100.0 * AVG(premium) / MAX(premium)")
    before, builds_before = made_by(BUILDER), served.builds()
    run = run_build(live, served, model, BuildPlan(charts=(big,), measures=(rate,)), "percent-refused", goal="a rate")
    record("percent-refused", summary(run, served))
    [start] = run.requests
    reason = ("a percent measure is a fraction from 0 to 1, which Superset shows as a percent: write it without "
              "multiplying by 100")
    assert start.status_code == 200, (start.status_code, start.body)
    assert start.body == {"refused": {
        "reason": f"mnemiq could not plan this dashboard: measure 'scaled_rate': {reason}",
        "parts": [{"part": "measure 'scaled_rate'", "reason": reason}]}}, start.body
    assert run.planner.goals == ["a rate"] * 2
    assert writes(start.calls) == []
    assert served.builds() == builds_before and made_by(BUILDER) == before
    assert_no_model_call(run)


# --- a histogram, made through the extension ------------------------------------------------------------------------


def test_a_build_with_a_histogram_makes_it(live, served, model, policy_id):
    """A histogram and a bar in one stub plan. Expected: the build runs to `done`; the histogram exists as a
    `histogram_v2` chart carrying this build's marker and is on the dashboard, saved with no query context; the value
    check leaves it NOT_CHECKED, and the build's message names it with the histogram rule's reason
    (`_NOT_CHECKED_TYPES`), while the bar is CHECKED (the one status the build keeps, for the charts of its measures).
    No model client is made or called."""
    policies = Measure(name="policies", label="Policies", expression="COUNT(id)", kind=MeasureKind.COUNT,
                       dataset_id=policy_id, description="Policies written", asked_by="")
    fields = {"time_column": None, "time_grain": None, "filters": (), "row_limit": None, "source": None, "target": None}
    histogram = ChartSpec(key="premium_spread", family=Family.HISTOGRAM, title="Spread of premiums",
                          dataset_id=policy_id, measures=(), dimensions=("premium",), **fields)
    bar = ChartSpec(key="policies_by_region", family=Family.BAR, title="Policies by region", dataset_id=policy_id,
                    measures=(WrittenRef(policies),), dimensions=("region",), **fields)
    plan = BuildPlan(charts=(histogram, bar), measures=(policies,))
    run = run_build(live, served, model, plan, "histogram", goal="the spread of premiums and policies by region")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped}
    record("histogram", found)
    assert_ran_to_the_end(run)
    final = run.final
    ids = {step["key"]: step["object_id"] for step in final["steps"] if step["kind"] == "chart"}
    assert set(ids) == {histogram.key, bar.key}, ids
    dashboard_id = final["dashboard_id"]
    stored = stored_chart(ids[histogram.key])
    assert stored["viz_type"] == "histogram_v2" and stored["deleted_at"] is None
    assert parse_marker(stored["description"]) == Marker(run.build_id, histogram.key, dashboard_id)
    assert ids[histogram.key] in members(dashboard_id) and ids[bar.key] in members(dashboard_id)
    assert metadata("SELECT query_context FROM slices WHERE id = %s", (ids[histogram.key],)) == [(None,)]
    assert final["checks"] == {str(ids[bar.key]): "checked"}, final["checks"]
    assert run.last_step["message"] == (
        f"1 of 2 charts checked; not checked: {histogram.title!r} (not_checked: {_NOT_CHECKED_TYPES['histogram_v2']})")
    assert_no_model_call(run)
    observe("histogram", chart_ids=ids, dashboard_id=dashboard_id, checks=final["checks"],
            message=run.last_step.get("message"))


# --- an existing dashboard, for its editors only -------------------------------------------------------------------


@pytest.fixture
def readable_dashboard(live) -> int:
    """A dashboard analyst made through the extension and published through Superset's own API, with no chart, so
    every Alpha may read it (`DashboardAccessFilter`: a published dashboard with no viewers and nothing to check
    access against); analyst alone edits it."""
    build_id = str(uuid.uuid4())
    made = as_user(live, BUILDER).call(TOOL_PREFIX + "create_dashboard", {
        "title": f"live editors {build_id[:8]}", "marker": format_marker(build_id, "dashboard", None)})
    dashboard_id = made["dashboard_id"]
    with live.signed_in(BUILDER) as (client, headers):
        csrf = live.csrf_token(client, headers)
        answer = client.put(f"/api/v1/dashboard/{dashboard_id}", json={"published": True},
                            headers={**headers, "X-CSRFToken": csrf})
    assert answer.status_code == 200, answer.status_code
    return dashboard_id


def set_editors(live, dashboard_id: int, usernames: list[str]) -> None:
    """As analyst, signed in, the dashboard's editors set to `usernames` through Superset's own REST API
    (`PUT /api/v1/dashboard/<id>`), as a person sets them in Superset."""
    with live.signed_in(BUILDER) as (client, headers):
        csrf = live.csrf_token(client, headers)
        answer = client.put(f"/api/v1/dashboard/{dashboard_id}",
                            json={"editors": [user_subject(name) for name in usernames]},
                            headers={**headers, "X-CSRFToken": csrf})
    assert answer.status_code == 200, answer.status_code
    assert editor_ids("dashboard_editors", "dashboard_id", dashboard_id) == sorted(user_subject(n) for n in usernames)


def test_a_dashboard_is_built_on_by_its_editors_only(live, served, model, policy_id, readable_dashboard):
    """Who may build on an existing dashboard (`dashboard_id`), asked of the extension's `editors_of` at the start,
    before anything is planned, and again at the build's first step. `west` (Alpha) reads analyst's published
    dashboard and does not edit it, as `editors_of` itself answers west. Expected:
    - west's start is refused, 403, with mnemiq's sentence naming the editors ("This builder is not an editor of
      dashboard N; its editors are ..."): the planner is not asked, no build is kept, the only Superset calls are the
      grants read, the extension's probe and `editors_of`, and west has made nothing in Superset;
    - analyst makes west an editor too, through Superset's API: west's start is then 201; analyst takes west off the
      editors before west's first `/next`, which is refused at the dashboard step (the build `failed`, its dashboard
      step `failed` as `not_editor`, the sentence naming analyst); west has still made nothing and no write call was
      made;
    - analyst, its editor, builds on the same dashboard: the start is 201 and the first `/next` passes the dashboard
      step, `done` with the dashboard's id.
    No model client is made or called."""
    dashboard_id = readable_dashboard
    west_reads = as_user(live, "west").call(TOOL_PREFIX + "editors_of", {"dashboard_ids": [dashboard_id]})
    assert west_reads["dashboards"] == [{"dashboard_id": dashboard_id, "is_editor": False,
                                         "editors": [label_of(BUILDER)]}], west_reads
    before, builds_before = made_by("west"), served.builds()

    refused = run_build(live, served, model, one_bar(policy_id), "reader-refused-at-start", builder="west",
                        goal="policies", stop_after=1, dashboard_id=dashboard_id)
    record("reader-refused-at-start", summary(refused, served))
    [start] = refused.requests
    assert start.status_code == 403, (start.status_code, start.body)
    assert start.body["detail"] == (f"This builder is not an editor of dashboard {dashboard_id}; its editors are "
                                    f"{label_of(BUILDER)}."), start.body
    assert refused.planner.goals == []
    tools = {short(call.tool) for call in start.calls}
    assert "ext.editors_of" in tools and tools <= {"execute_sql", short(PROBE), "ext.editors_of"}, tools
    assert [(call.request, "refused" in (call.payload or {})) for call in start.calls if call.tool == PROBE] == [
        ({}, True)]
    assert writes(start.calls) == []
    assert served.builds() == builds_before and made_by("west") == before
    assert_no_model_call(refused)

    set_editors(live, dashboard_id, [BUILDER, "west"])
    stopped = run_build(live, served, model, one_bar(policy_id), "reader-refused-at-step-one", builder="west",
                        goal="policies", stop_after=1, dashboard_id=dashboard_id,
                        before_next=lambda: set_editors(live, dashboard_id, [BUILDER]))
    record("reader-refused-at-step-one", summary(stopped, served))
    assert [r.status_code for r in stopped.requests] == [201, 200, 200], [(r.label, r.status_code)
                                                                        for r in stopped.requests]
    failed = stopped.requests[1].build
    assert failed["ran"] == "dashboard" and failed["status"] == "failed"
    assert failed["steps"][0]["status"] == "failed" and failed["steps"][0]["object_id"] is None
    assert failed["message"] == (f"this builder is not an editor of dashboard {dashboard_id}; its editors are "
                                 f"{label_of(BUILDER)}"), failed["message"]
    assert served.raw_steps(stopped.build_id)[0]["reason"] == "not_editor"
    assert made_by("west") == before
    assert writes([call for r in stopped.requests for call in r.calls]) == []
    assert_no_model_call(stopped)

    passed = run_build(live, served, model, one_bar(policy_id), "editor-passes-step-one", builder=BUILDER,
                       goal="policies", stop_after=1, dashboard_id=dashboard_id)
    record("editor-passes-step-one", summary(passed, served))
    assert [r.status_code for r in passed.requests] == [201, 200, 200]
    first = passed.requests[1].build
    assert first["ran"] == "dashboard" and first["status"] == "building"
    assert first["steps"][0]["status"] == "done" and first["steps"][0]["object_id"] == dashboard_id
    tools = [short(call.tool) for call in passed.requests[1].calls]
    assert {"ext.editors_of", "ext.get_layout"} <= set(tools), tools
    assert writes(passed.requests[1].calls) == [], tools
    assert_no_model_call(passed)
    observe("editors only", dashboard_id=dashboard_id, at_start=start.body["detail"], at_step_one=failed["message"],
            editor=first["steps"][0]["status"])


# --- a label Superset would save as other text ----------------------------------------------------------------------


def test_a_measure_label_superset_would_rewrite_is_refused_before_anything_is_made(live, served, model, policy_id):
    """The label rule, live. The stub's plan holds one bar chart of a currency measure labelled "Premium < 1y", which
    rc2's MCP would save as other text (nh3 escapes `<`). Expected: the start answers 200 with the refused plan and
    nothing else: its one part is the measure, with mnemiq's reason ("the measure's label holds <, >, & or a no-break
    space, which Superset saves as other text; write it without them"), and the plan's reason names that part. The
    planner was asked twice (the plan and its one repair, the same plan again). No build is kept, no write call is made,
    and analyst has made nothing new in Superset. No model client is made or called."""
    recent = Measure(name="recent_premium", label="Premium < 1y", expression="AVG(premium)",
                     kind=MeasureKind.CURRENCY, dataset_id=policy_id, description="Premium of the last year",
                     asked_by="")
    chart = ChartSpec(key="recent_premium_by_region", family=Family.BAR, title="Recent premium by region",
                      dataset_id=policy_id, measures=(WrittenRef(recent),), dimensions=("region",), time_column=None,
                      time_grain=None, filters=(), row_limit=None, source=None, target=None)
    before, builds_before = made_by(BUILDER), served.builds()
    run = run_build(live, served, model, BuildPlan(charts=(chart,), measures=(recent,)), "label-refused",
                    goal="recent premium by region")
    record("label-refused", summary(run, served))
    [start] = run.requests
    reason = ("the measure's label holds <, >, & or a no-break space, which Superset saves as other text; write it "
              "without them")
    assert start.status_code == 200, (start.status_code, start.body)
    assert start.body == {"refused": {
        "reason": f"mnemiq could not plan this dashboard: measure 'recent_premium': {reason}",
        "parts": [{"part": "measure 'recent_premium'", "reason": reason}]}}, start.body
    assert run.planner.goals == ["recent premium by region"] * 2
    assert writes(start.calls) == []
    assert served.builds() == builds_before and made_by(BUILDER) == before
    assert_no_model_call(run)
    observe("label refused", status=start.status_code, parts=start.body["refused"]["parts"],
            calls=dict(Counter(short(call.tool) for call in start.calls)))


# --- a dataset the build makes ---------------------------------------------------------------------------------------


def test_a_dataset_the_build_makes_carries_its_marker_and_its_chart_checks(live, served, model, sales_table):
    """The dataset step, live. The module's own table is in the snapshot, and Superset has no dataset for it. The
    stub's plan makes one (a physical dataset on the table) and puts one bar chart on it: `SUM(amount)` by region, a
    currency measure. Expected:
    - the build runs to `done`: dashboard, the dataset, the chart, layout, check;
    - the dataset step sent one create, the extension's `create_dataset`, for the table (no SQL) and the build's
      dashboard, carrying its step's marker, and was answered the dataset's id; no request called MCP's
      `create_dataset`, `create_virtual_dataset` or `generate_chart`, a claim, or `patch_chart`;
    - Superset holds the dataset on the table, physical, created by analyst, its description's last line the step's
      marker, the one dataset carrying it, analyst among its editors; the chart reads it, and its check status is
      `CHECKED`;
    - `get_chart_data`, asked as analyst with a minted token, shows the table's sums by region, the source's;
    - no model client is made or called.
    Teardown removes the chart and the dataset as analyst removes them, and drops the table."""
    name = sales_table.name
    assert name in {column.object_id for column in served.runtime.snapshot.columns}
    assert metadata("SELECT id FROM tables WHERE table_name = %s", (name,)) == []
    amount = Measure(name="amount_sold", label="Amount sold", expression="SUM(amount)", kind=MeasureKind.CURRENCY,
                     dataset_id=-1, description="Amount sold", asked_by="")
    chart = ChartSpec(key="amount_by_region", family=Family.BAR, title="Amount sold by region", dataset_id=-1,
                      measures=(WrittenRef(amount),), dimensions=("region",), time_column=None, time_grain=None,
                      filters=(), row_limit=None, source=None, target=None)
    plan = BuildPlan(charts=(chart,), measures=(amount,),
                     datasets=(PlannedDataset(key="dataset_1", placeholder=-1, table=name),))
    run = run_build(live, served, model, plan, "dataset-step", goal="amount sold by region")
    record("dataset-step", summary(run, served, plan))
    raw = {step["key"]: step for step in served.raw_steps(run.build_id)} if run.build_id else {}
    # What the build made, for the table's teardown, before anything is asserted.
    sales_table.dashboard_id = run.final.get("dashboard_id")
    sales_table.dataset_id = (raw.get("dataset_1") or {}).get("object_id")
    sales_table.chart_ids = [made for made in [(raw.get(chart.key) or {}).get("object_id")] if made is not None]
    assert_ran_to_the_end(run)
    assert [step["key"] for step in run.final["steps"]] == ["dashboard", "dataset_1", chart.key, "layout", "check"]
    dashboard_id, dataset_id = run.final["dashboard_id"], raw["dataset_1"]["object_id"]
    marker = format_marker(run.build_id, "dataset_1", dashboard_id)

    [made] = [r for r in run.requests if r.build.get("ran") == "dataset_1"]
    creates = [call for call in made.calls if call.tool == CREATE_DATASET]
    assert len(creates) == 1, [call.tool for call in made.calls]
    sent = creates[0].request
    assert (sent["table_name"], sent["schema"], sent["dashboard_id"], sent["marker"]) == (
        name, "public", dashboard_id, marker), sent
    assert "sql" not in sent and creates[0].payload == {"dataset_id": dataset_id}
    assert never_called(run) == []

    (row,) = metadata("SELECT table_name, schema, sql, created_by_fk, deleted_at, description FROM tables "
                      "WHERE id = %s", (dataset_id,))
    assert row[:5] == (name, "public", None, user_id(BUILDER), None), row[:5]
    assert parse_marker(row[5]) == Marker(run.build_id, "dataset_1", dashboard_id)
    assert carrying("tables", marker) == [(dataset_id, None)]
    assert user_subject(BUILDER) in editor_ids("sqlatable_editors", "table_id", dataset_id)
    chart_id = raw[chart.key]["object_id"]
    (on,) = metadata("SELECT datasource_id, datasource_type FROM slices WHERE id = %s", (chart_id,))
    assert on == (dataset_id, "table"), on
    assert run.final["checks"] == {str(chart_id): "checked"}, (run.final["checks"], run.last_step.get("message"))

    data = SupersetMcp(live.mcp_url, DelegatedToken(live.minted(BUILDER, "build")), timeout_s=90).call(
        "get_chart_data", {"identifier": chart_id}).get("data") or []
    shown = sorted((row.get("region"), round(float(row["Amount sold"]), 2)) for row in data)
    truth = sorted((region, round(float(total), 2)) for region, total in ground_truth(
        live, f"SELECT region, sum(amount) FROM public.{name} GROUP BY 1"))
    same = shown == truth
    assert len(data) == 3 and same, (len(data), same)
    assert_no_model_call(run)
    observe("dataset step", dataset_id=dataset_id, chart_id=chart_id, check=run.final["checks"],
            message=run.last_step.get("message"))


# --- a create whose answer is lost ---------------------------------------------------------------------------------


def members(dashboard_id: int) -> list[int]:
    return [row[0] for row in metadata("SELECT slice_id FROM dashboard_slices WHERE dashboard_id = %s ORDER BY 1",
                                       (dashboard_id,))]


@dataclass
class Removed:
    """Charts and dashboards a test made, removed when it ends as analyst removes them: each chart through MCP's
    `delete_chart`, then each dashboard through MCP's `delete_dashboard`. An id is recorded before anything about it is
    asserted."""

    charts: list[int] = field(default_factory=list)
    dashboards: list[int] = field(default_factory=list)


@pytest.fixture
def removed(live) -> Iterator[Removed]:
    found = Removed()
    yield found
    builder = as_user(live, BUILDER)
    problems: list = []
    for tool, ids in (("delete_chart", found.charts), ("delete_dashboard", found.dashboards)):
        for object_id in dict.fromkeys(ids):
            try:
                answer = builder.call(tool, {"identifier": object_id})
            except Exception as exc:  # noqa: BLE001 (reported, so the rest is still removed)
                problems.append((tool, object_id, type(exc).__name__))
                continue
            if answer.get("success") is False:
                problems.append((tool, object_id, answer.get("error_type")))
    assert not problems, problems


@dataclass
class Lost:
    """What `lost_first` saw: what Superset answered the first call (None when it was never sent), and any error the
    work run in between raised."""

    answers: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@contextmanager
def lost_first(tool: str, *, sent: bool, meanwhile=None) -> Iterator[Lost]:
    """Inside the block, the client every adapter holds is wrapped in process (`SupersetMcp.call`, beneath the recorder
    `run_build` puts over it): the first call of `tool` is sent and runs in Superset (`sent`), or never reaches it, and
    either way its caller is answered `SupersetUnavailable`, as when an answer is lost on its way back or the service
    cannot be reached. Every other call is the client's own. `meanwhile(call)`, when given, runs once in between, on a
    thread of its own, with the client's own `call`, so nothing it sends is recorded as the build's."""
    lost = Lost()
    with pytest.MonkeyPatch.context() as patch:
        real = SupersetMcp.call

        def in_between() -> None:
            try:
                meanwhile(real)
            except BaseException as exc:  # noqa: BLE001 (reported by the test)
                lost.errors.append(exc)

        def call(self, name, request, **kwargs):
            if name != tool or lost.answers:
                return real(self, name, request, **kwargs)
            lost.answers.append(real(self, name, request, **kwargs) if sent else None)
            if meanwhile is not None:
                worker = threading.Thread(target=in_between)
                worker.start()
                worker.join(120)
            raise SupersetUnavailable("the test lost this call's answer")

        patch.setattr(SupersetMcp, "call", call)
        yield lost


def test_a_build_whose_chart_answer_is_lost_makes_one_chart(live, served, model, policy_id, removed):
    """A lost answer, live. In a one-chart build (a bar of policies by region), the first `create_chart` the chart step
    sends runs in Superset and returns, and its answer is replaced in process by `SupersetUnavailable`, as when it is
    lost on its way back (`lost_first`). Expected:
    - that request answers 200 with the chart step `uncertain` and mnemiq's sentence that Superset did not answer while
      making the chart, the step holding no id; the next request sends the same create (the same request, the same
      marker) and is answered the chart the first one made;
    - the build runs to `done`; Superset holds exactly one chart carrying the step's marker, the one the build recorded
      and the dashboard's one chart; it is `CHECKED`;
    - no request called MCP's `generate_chart`, `create_dataset` or `create_virtual_dataset`, a claim, or `patch_chart`;
      no model client is made or called."""
    plan = one_bar(policy_id)
    chart = plan.charts[0]
    with lost_first(CREATE_CHART, sent=True) as lost:
        run = run_build(live, served, model, plan, "lost-chart-answer", goal="policies by region")
    record("lost-chart-answer", summary(run, served, plan))
    dashboard_id = run.final.get("dashboard_id")
    marker = format_marker(run.build_id, chart.key, dashboard_id) if run.build_id and dashboard_id else None
    removed.dashboards.extend([dashboard_id] if dashboard_id else [])
    removed.charts.extend(chart_id for chart_id, _ in (carrying("slices", marker) if marker else []))
    assert_ran_to_the_end(run)
    [made] = lost.answers
    assert set(made) == {"chart_id"}, made
    tries = [request for request in run.requests if request.build.get("ran") == chart.key]
    assert len(tries) == 2, [(r.label, r.build.get("ran")) for r in run.requests]
    silent, settled = tries
    [step] = [s for s in silent.build["steps"] if s["key"] == chart.key]
    assert (step["status"], step["object_id"]) == ("uncertain", None), step
    assert silent.build["message"].startswith(f"Superset did not answer while making the chart {chart.title!r}")
    sent = [[call for call in request.calls if call.tool == CREATE_CHART] for request in tries]
    assert [len(calls) for calls in sent] == [1, 1], sent
    (first,), (again,) = sent
    assert isinstance(first.raised, SupersetUnavailable) and first.payload is None
    assert again.request == first.request and first.request["marker"] == marker
    assert again.payload == made
    [built] = [s["object_id"] for s in run.final["steps"] if s["key"] == chart.key]
    assert built == made["chart_id"]
    assert carrying("slices", marker) == [(built, None)]
    assert members(dashboard_id) == [built]
    assert run.final["checks"] == {str(built): "checked"}, (run.final["checks"], run.last_step.get("message"))
    assert never_called(run) == []
    assert_no_model_call(run)
    observe("lost answer", chart_id=built, silent=silent.build["message"], check=run.final["checks"])


def test_a_hand_made_same_named_chart_is_never_adopted(live, served, model, policy_id, removed):
    """A chart made by hand is never a build's, whatever its name. In a one-chart build (a bar of policies by region),
    the chart step's first `create_chart` is answered `SupersetUnavailable` in process (`lost_first`), and in between,
    after the step began and before its retry, analyst, the builder, makes by hand with MCP's `generate_chart` a chart
    of the same name on the same dataset, which carries no marker. Twice: once with the first create run in Superset
    (its answer lost on the way back), once with it never reaching Superset. Expected, each time:
    - the build runs to `done`, and its chart step's object is the one chart carrying the step's marker, never the
      hand-made one: the chart the first create made in the first case; in the second, a chart the retry made, after
      the hand-made one, since none carried the marker;
    - the two creates the step sent are one request; the hand-made chart is as it was made (its name, dataset, params,
      description and `changed_on`), carries no marker, and is not among the dashboard's charts, which are the build's
      chart alone; the build's chart is `CHECKED`;
    - no request called MCP's creates, a claim or `patch_chart`; no model client is made or called.
    The hand-made charts are removed with the builds' own."""
    plan = one_bar(policy_id)
    chart = plan.charts[0]
    row = "SELECT slice_name, datasource_id, params, description, changed_on, deleted_at FROM slices WHERE id = %s"
    seen = {}
    for case, sent in (("answer-lost", True), ("never-sent", False)):
        hand: dict = {}

        def make_by_hand(real_call, hand=hand) -> None:
            answer = real_call(as_user(live, BUILDER), "generate_chart",
                               generate_chart_request(chart, {}, {}, currency="USD"))
            hand["id"] = (answer.get("chart") or {}).get("id")
            hand["row"] = metadata(row, (hand["id"],))

        with lost_first(CREATE_CHART, sent=sent, meanwhile=make_by_hand) as lost:
            run = run_build(live, served, model, plan, f"hand-made-{case}", goal="policies by region")
        record(f"hand-made-{case}", summary(run, served, plan))
        dashboard_id = run.final.get("dashboard_id")
        marker = format_marker(run.build_id, chart.key, dashboard_id) if run.build_id and dashboard_id else None
        removed.dashboards.extend([dashboard_id] if dashboard_id else [])
        removed.charts.extend([hand["id"]] if type(hand.get("id")) is int else [])
        removed.charts.extend(chart_id for chart_id, _ in (carrying("slices", marker) if marker else []))
        assert lost.errors == [], (case, lost.errors)
        assert_ran_to_the_end(run)
        hand_id = hand["id"]
        assert type(hand_id) is int, case
        [built] = [s["object_id"] for s in run.final["steps"] if s["key"] == chart.key]
        assert built != hand_id and carrying("slices", marker) == [(built, None)], (case, built, hand_id)
        assert lost.answers == ([{"chart_id": built}] if sent else [None]), (case, lost.answers)
        if not sent:
            assert built > hand_id, (case, built, hand_id)
        creates = [call for request in run.requests for call in request.calls if call.tool == CREATE_CHART]
        assert len(creates) == 2 and creates[0].request == creates[1].request, case
        assert creates[1].payload == {"chart_id": built}, case
        stored = metadata(row, (hand_id,))
        assert stored == hand["row"], case
        (name, dataset_id, _, description, _, deleted_at), = stored
        assert (name, dataset_id, deleted_at) == (chart.title, policy_id, None)
        assert parse_marker(description) is None, case
        assert metadata("SELECT slice_name, datasource_id FROM slices WHERE id = %s", (built,)) == [
            (chart.title, policy_id)]
        assert members(dashboard_id) == [built], case
        assert run.final["checks"] == {str(built): "checked"}, (case, run.final["checks"])
        assert never_called(run) == [], case
        assert_no_model_call(run)
        seen[case] = {"built": built, "hand_made": hand_id}
    observe("hand made", **seen)

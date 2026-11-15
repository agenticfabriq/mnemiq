"""Superset mode's dashboard builder, live: builds driven through mnemiq's own build routes in process, against the
suite's Superset 7.0.0rc2 with mnemiq's extension loaded, each request carrying a token Superset's mint route made for
building (`Live.minted(username, "build")`), and the build store on the live Postgres, in a database this module
makes and drops.

A build's plan comes from a stub planner (`FakePlanner`), which returns a fixed plan and asks no model: no model client
is made or called while a build runs, and the stub model is asked nothing. Everything after the planner is mnemiq's
own: the plan check (a written measure a saved metric says the same as is replaced by it), the build's steps, the id
floor each create is judged by, the layout and the value check.

Each test states its expectation in its docstring. A build's record (each request's step, its key, kind and status,
the Superset calls it made and its latency; each create's floor and the id it made; each chart's id, type and check
status) is printed as an `OBSERVED` line, which `pytest -rA` shows for passing tests too, and written as JSON to the
directory `MNEMIQ_TEST_SUPERSET_RECORD_DIR` names when it is set. No token, key, password or value from the data goes
into it: a comparison of numbers is asserted as a bool, so a failure does not print them either.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from mnemiq.server.superset_refusals import SIGN_IN_AGAIN
from mnemiq.superset.build_store import BUILD_TABLE, MEASURE_TABLE
from mnemiq.superset.builder import start_build
from mnemiq.superset.chart_form import _NOT_CHECKED_TYPES
from mnemiq.superset.charts import ChartSpec, Family, WrittenRef
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetRefusedCaller
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.marker import Marker, format_marker, parse_marker
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan, FakePlanner
from mnemiq.superset.saved_metrics import normalized
from tests.superset_extension_path import use_extension_source
from tests.superset_live.live import Call, Recorder, fingerprint_of, ground_truth, metadata, record_calls, run_setup
from tests.test_superset_build_store import STORE_COLUMNS, value_shaped_names

use_extension_source()

from agenticfabriq.mnemiq.token import BUILD_SCOPES  # noqa: E402

pytestmark = pytest.mark.integration

BUILD = " ".join(BUILD_SCOPES)
# The builder of B2 and B6: Alpha and sql_lab, narrowed by no row rule.
BUILDER = "analyst"
GOAL = "sales by region and month"
# The build store's own database on the live Postgres, made by this module and dropped after it.
STORE_DATABASE = "mnemiq_build_store_live"
# The saved metric planted on the `policy` dataset before B2, and removed after the module: what B2's `sales` measure
# says, so the plan check reuses it.
PLANTED = ("total_sales", "SUM(premium)")
RECORD_DIR = "MNEMIQ_TEST_SUPERSET_RECORD_DIR"
# What a build answer's step may hold (`superset_build._build`).
STEP_FIELDS = {"key", "kind", "status", "object_id", "started_at", "lease_until"}
BUILD_FIELDS = {"build_id", "status", "dashboard_id", "steps", "checks", "ran", "message"}
# The calls that change something in Superset.
WRITES = {"generate_chart", "create_dataset", "create_virtual_dataset",
          *(TOOL_PREFIX + base for base in ("create_dashboard", "claim_chart", "claim_dataset", "create_chart",
                                            "patch_chart", "add_saved_metric", "delete_dataset", "set_layout"))}
# The list each create's floor is read from, by the step's kind.
FLOOR_TOOLS = {"dashboard": "list_dashboards", "chart": "list_charts", "dataset": "list_datasets"}


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


def highest(username: str, table: str) -> int:
    """The highest id among `username`'s objects of `table` not in the trash, 0 for none: what a floor reads."""
    (row,) = metadata(f"SELECT coalesce(max(id), 0) FROM {table} WHERE created_by_fk = %s AND deleted_at IS NULL",
                      (user_id(username),))
    return row[0]


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
              stop_after: int | None = None, **body) -> BuildRun:
    """The build of `plan` started and run one step per `/next` until it ends (or `stop_after` steps), each request
    with a token minted for it, every Superset call recorded, and the model's clients counted. Bounded: one `/next`
    per step and three more; a step left to its lease (409 on the next request) stops the loop and says so."""
    from mnemiq.llm.client import LLMClient

    served.plan_with(plan)
    run = BuildRun(name=name, started=datetime.now(UTC).isoformat(timespec="seconds"))
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
    calls it made by tool, with their JSON-RPC requests), each step's floor and the id it made, and each chart's id,
    type and check status. Ids, names, statuses and timings only: never a value, a token or a lease id."""
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
        })
    steps = [{"key": key, "kind": step["kind"], "status": step["status"], "object_id": step["object_id"],
              "id_floor": (step.get("intent") or {}).get("id_floor"), "reason": step.get("reason")}
             for key, step in raw.items()]
    families = {chart.key: chart.family.value for chart in plan.charts} if plan else {}
    charts = []
    for key, step in raw.items():
        if step["kind"] == "chart" and step["object_id"] is not None:
            chart_id = step["object_id"]
            charts.append({"key": key, "chart_id": chart_id, "planned_type": families.get(key),
                           "type": stored_chart(chart_id)["viz_type"],
                           "check": final.get("checks", {}).get(str(chart_id))})
    return {"build_id": run.build_id, "dashboard_id": final.get("dashboard_id"), "status": final.get("status"),
            "started": run.started, "ended": run.ended, "stopped": run.stopped,
            "message": run.last_step.get("message"), "leases_held": len(run.leases),
            "requests": requests, "steps": steps, "charts": charts,
            "model_calls": {"chat": run.model_after[0] - run.model_before[0],
                            "embeddings": run.model_after[1] - run.model_before[1],
                            "profile_usage": run.model_after[2] - run.model_before[2],
                            "llm_clients_made": run.llm_made, "llm_completions": run.llm_asked}}


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

# What B2's charts show, read from the source directly: every one a sum of `premium`, which has cents.
SALES_TRUTH = (
    "SELECT sum(premium) FROM policy",
    "SELECT region, sum(premium) FROM policy GROUP BY 1",
    "SELECT status, sum(premium) FROM policy GROUP BY 1",
    "SELECT date_trunc('month', issued), sum(premium) FROM policy GROUP BY 1",
    "SELECT region, status, sum(premium) FROM policy GROUP BY 1, 2",
)


def sales_values(live) -> list[Decimal]:
    """Every sum B2's charts show that has a fractional part: a number of the data no id, count or time spells."""
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


# --- B2: six charts, the planted saved metric reused, every chart checked ------------------------------------------


def b2_plan(policy_id: int) -> BuildPlan:
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
class B2:
    run: BuildRun
    plan: BuildPlan
    metrics_after_planting: dict[str, str]
    metrics_after_build: dict[str, str]
    floors_before: dict[str, int]
    kept_plan: dict
    raw_steps: list[dict]
    summary: dict


@pytest.fixture(scope="module")
def b2(live, served, model, policy_id, planted) -> B2:
    """B2's build, once, for every test that reads it: the goal "sales by region and month", run to its end as
    `analyst`, and its record."""
    plan = b2_plan(policy_id)
    floors_before = {"dashboards": highest(BUILDER, "dashboards"), "slices": highest(BUILDER, "slices")}
    metrics_after_planting = saved_metrics(policy_id)
    run = run_build(live, served, model, plan, "b2-sales-by-region-and-month")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped,
                                                             "requests": [(r.label, r.status_code) for r in run.requests]}
    record("b2-sales-by-region-and-month", found)
    kept_plan = served.raw_plan(run.build_id) if run.build_id else {}
    raw_steps = served.raw_steps(run.build_id) if run.build_id else []
    return B2(run, plan, metrics_after_planting, saved_metrics(policy_id), floors_before, kept_plan, raw_steps, found)


def test_b2_six_charts_reuse_the_planted_metric_and_every_one_checks(live, b2, policy_id, planted):
    """B2. The goal "sales by region and month", planned by the stub as six charts across five families (two KPI
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
    run, plan = b2.run, b2.plan
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
    kept = b2.kept_plan
    for chart in kept["charts"]:
        refs = chart["measures"]
        shows_sales = chart["key"] != "policies_written"
        assert ({"saved": planted, "kind": "currency"} in refs) == shows_sales, (chart["key"], refs)
        assert {"written": "sales"} not in refs
    assert [m["name"] for m in kept["measures"]] == ["policies"]
    for key, chart_id in ids.items():
        metrics = chart_metrics(stored_chart(chart_id)["params"])
        assert (planted in metrics) == (key != "policies_written"), (key, metrics)
    assert b2.metrics_after_build == b2.metrics_after_planting
    assert [n for n, e in b2.metrics_after_build.items() if normalized(e) == normalized(PLANTED[1])] == [planted]

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
    observe("B2", charts={key: (chart_id, final["checks"][str(chart_id)]) for key, chart_id in ids.items()},
            layout=placed, message=run.last_step["message"])


def test_b2_every_made_id_is_above_its_floor(b2):
    """The id floor, live. Before each create, the step reads the highest id among the builder's objects of that kind
    (`created_by_me`, ordered by id, descending, one item) and records it in its intent. Expected: every create step
    (the dashboard, each chart) records a floor, its list call was ordered by id and descending, and the id it made
    is above its floor; the dashboard's floor is analyst's highest dashboard id before the build, the first chart's
    analyst's highest chart id before it, and each later chart's the chart made just before it."""
    run = b2.run
    assert_ran_to_the_end(run)
    made = [step for step in b2.raw_steps if step["kind"] in ("dashboard", "chart")]
    floors = [(step["key"], (step.get("intent") or {}).get("id_floor"), step["object_id"]) for step in made]
    assert all(type(floor) is int and type(made_id) is int and made_id > floor for _, floor, made_id in floors), floors
    by_key = {key: (floor, made_id) for key, floor, made_id in floors}
    assert by_key["dashboard"][0] == b2.floors_before["dashboards"], (by_key["dashboard"], b2.floors_before)
    charts = [key for key, _, _ in floors if key != "dashboard"]
    expected = [b2.floors_before["slices"], *(by_key[key][1] for key in charts[:-1])]
    assert [by_key[key][0] for key in charts] == expected, (floors, b2.floors_before)
    for request in run.requests:
        if request.label != "next" or request.build.get("ran") not in by_key:
            continue
        kind = "dashboard" if request.build["ran"] == "dashboard" else "chart"
        listed = [call.request for call in request.calls if call.tool == FLOOR_TOOLS[kind]]
        assert listed and listed[0] == {"created_by_me": True, "select_columns": ["id"], "order_column": "id",
                                        "order_direction": "desc", "page": 1, "page_size": 1}, listed
    observe("B2 floors", floors=floors)


def test_b8_every_superset_call_of_a_build_carries_its_requests_token(b2):
    """B8, the build half: across the whole of B2, each request carries a token minted for it alone, and every
    Superset call the request made carries that request's token and no other's (the Recorder's fingerprint of the
    client's credential, call by call). Every request but a read asks Superset something, and the read asks
    nothing."""
    run = b2.run
    assert_ran_to_the_end(run)
    tokens = [request.token for request in run.requests]
    assert len(set(tokens)) == len(tokens)
    for request in run.requests:
        held = {call.credential for call in request.calls}
        if request.label == "read":
            assert held == set(), held
        else:
            assert held == {f"DelegatedToken:{fingerprint_of(request.token)}"}, (request.label, held)
    observe("B8 tokens", requests=len(run.requests), calls=sum(len(r.calls) for r in run.requests))


def test_b8_the_live_build_store_holds_no_value(live, b2, served):
    """B8, the build half: after B2, Task 9's guard (`value_shaped_names`) read over the live store finds no column and
    no jsonb key at any depth named as a value, a row or a sample is kept, having read every column of both tables and
    every key of their jsonb documents; and no stored row's text holds any sum B2's charts show, read from the source.
    The controls, each in a transaction rolled back: a value-shaped key planted in a stored check status is flagged,
    and one of those sums planted in a stored goal is found."""
    import psycopg
    from psycopg import sql

    assert_ran_to_the_end(b2.run)
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
        assert b2.run.build_id in text and GOAL in text  # the probe reads what the build kept
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
    observe("B8 store", keys_read=keys, sums_looked_for=len(values), columns=read)


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


def test_b8_an_expired_minted_token_is_refused_at_mnemiqs_door(live, served, policy_id, recorder):
    """B8, the build half: a token Superset's mint route made for building starts a build (201: the control); the
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
    observe("B8 expired", **{route: (a.status_code, a.json().get("detail")) for route, a in refused.items()})


def test_b8_an_unknown_users_token_is_refused_by_superset_in_mnemiqs_words(live, served, policy_id, recorder):
    """B8, the build half: a token the identity provider validly signed, with a build token's scopes, for `nobody`, a
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
    assert not [call.tool for call in calls if call.tool in WRITES]
    assert served.raw_steps(kept.build_id) == before
    observe("B8 unknown user", status={route: a.status_code for route, a in answers.items()},
            detail=SIGN_IN_AGAIN, superset_said=sorted(said), calls=[c.tool for c in calls])


# --- B6: a funnel, made through the extension ----------------------------------------------------------------------


def test_b6_a_funnel_is_made_through_the_extension_and_its_data_read(live, served, model, policy_id):
    """B6. A funnel by chat: the stub's plan holds one funnel (`COUNT(id)` of policies by status). Expected: the build
    runs to `done` (dashboard, funnel, layout, check); the funnel was made by the extension's `create_chart`, never
    MCP's `generate_chart`, and needed no claim; Superset holds it as a `funnel` chart created by analyst, analyst
    among its editors and its editors the dashboard's, its description ending in this build's marker; Superset's
    `get_chart_data`, asked as analyst with a minted token, answers one row per status, each count the source's. No
    model client is made or called. Its check status is recorded (rendering is not this PR's)."""
    policies = Measure(name="policies", label="Policies", expression="COUNT(id)", kind=MeasureKind.COUNT,
                       dataset_id=policy_id, description="Policies written", asked_by="")
    funnel = ChartSpec(key="policies_by_status", family=Family.FUNNEL, title="Policies by status",
                       dataset_id=policy_id, measures=(WrittenRef(policies),), dimensions=("status",),
                       time_column=None, time_grain=None, filters=(), row_limit=None, source=None, target=None)
    plan = BuildPlan(charts=(funnel,), measures=(policies,))
    run = run_build(live, served, model, plan, "b6-funnel", goal="policies by status as a funnel")
    found = summary(run, served, plan) if run.build_id else {"stopped": run.stopped}
    record("b6-funnel", found)
    assert_ran_to_the_end(run)
    final = run.final
    [chart_id] = [step["object_id"] for step in final["steps"] if step["key"] == funnel.key]
    dashboard_id = final["dashboard_id"]
    [step_request] = [r for r in run.requests if r.build.get("ran") == funnel.key]
    tools = [call.tool for call in step_request.calls]
    assert TOOL_PREFIX + "create_chart" in tools
    assert "generate_chart" not in tools and TOOL_PREFIX + "claim_chart" not in tools, tools
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
    observe("B6", chart_id=chart_id, dashboard_id=dashboard_id, rows=len(rows),
            check=final["checks"].get(str(chart_id)), message=run.last_step.get("message"))


# --- step 1: an existing dashboard, for its editors only ------------------------------------------------------------


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


def test_step_one_refuses_a_reader_who_does_not_edit_and_passes_an_editor(live, served, model, policy_id,
                                                                          readable_dashboard):
    """The step-1 editor check (the extension's `editors_of` with `dashboard_ids`). `west` (Alpha) can read analyst's
    published dashboard and does not edit it, as `editors_of` itself answers west. Expected: west's build on it starts
    (201), and its first `/next` is refused at the dashboard step: the build `failed`, its dashboard step `failed`,
    and mnemiq's sentence naming the editors ("this builder is not an editor of dashboard N; its editors are ...");
    nothing is made in Superset (west's dashboards, charts and datasets are the same ids before and after, and no
    write call was made). analyst, its editor, then builds on the same dashboard: the first `/next` passes step 1,
    the dashboard step `done` with the dashboard's id. No model client is made or called."""
    dashboard_id = readable_dashboard
    west_reads = as_user(live, "west").call(TOOL_PREFIX + "editors_of", {"dashboard_ids": [dashboard_id]})
    assert west_reads["dashboards"] == [{"dashboard_id": dashboard_id, "is_editor": False,
                                         "editors": [label_of(BUILDER)]}], west_reads
    before = made_by("west")
    refused = run_build(live, served, model, one_bar(policy_id), "step1-reader", builder="west", goal="policies",
                        stop_after=1, dashboard_id=dashboard_id)
    record("step1-reader", summary(refused, served) if refused.build_id else {"stopped": refused.stopped})
    assert [r.status_code for r in refused.requests] == [201, 200, 200], [(r.label, r.status_code)
                                                                        for r in refused.requests]
    stopped = refused.requests[1].build
    assert stopped["ran"] == "dashboard" and stopped["status"] == "failed"
    assert stopped["steps"][0]["status"] == "failed" and stopped["steps"][0]["object_id"] is None
    assert stopped["message"] == (f"this builder is not an editor of dashboard {dashboard_id}; its editors are "
                                  f"{label_of(BUILDER)}"), stopped["message"]
    assert made_by("west") == before
    assert not [call.tool for r in refused.requests for call in r.calls if call.tool in WRITES]
    assert [step["reason"] for step in served.raw_steps(refused.build_id)][0] == "not_editor"
    assert_no_model_call(refused)

    passed = run_build(live, served, model, one_bar(policy_id), "step1-editor", builder=BUILDER, goal="policies",
                       stop_after=1, dashboard_id=dashboard_id)
    record("step1-editor", summary(passed, served) if passed.build_id else {"stopped": passed.stopped})
    assert [r.status_code for r in passed.requests] == [201, 200, 200]
    first = passed.requests[1].build
    assert first["ran"] == "dashboard" and first["status"] == "building"
    assert first["steps"][0]["status"] == "done" and first["steps"][0]["object_id"] == dashboard_id
    tools = [short(call.tool) for call in passed.requests[1].calls]
    assert {"ext.editors_of", "ext.get_layout"} <= set(tools), tools
    assert not [call.tool for call in passed.requests[1].calls if call.tool in WRITES], tools
    assert_no_model_call(passed)
    observe("step 1", dashboard_id=dashboard_id, reader=stopped["message"], editor=first["steps"][0]["status"])

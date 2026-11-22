"""The dashboard builder's store, in a real Postgres: each build's plan, steps and leases, and each dashboard's measure
list, kept in mnemiq's control Postgres, and never a value.

Each test that needs a server gets a schema of its own in the database MNEMIQ_TEST_BUILD_STORE_DSN names, made the
store's search path and dropped when the test is done. Without the variable each of those tests skips, naming it, or
fails where MNEMIQ_TEST_BUILD_STORE_REQUIRED is `1`, as in the CI job (`superset-build-store.yml`) that runs them
against a Postgres of its own. The tests of that switch, and of what the store refuses before it connects, need no
server.

Not marked `integration`, for the reason `test_pg_declared_keys.py` gives: the root conftest seeds ACME whenever an
integration test is collected, and these tests need nothing of it.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

import mnemiq.superset.build_store as build_store
from mnemiq.superset.build_store import (
    BUILD_TABLE,
    MEASURE_TABLE,
    BuildNotFound,
    BuildStore,
    BuildStoreMissing,
    DashboardBusy,
    DashboardMeasure,
    LeaseLost,
    StepBusy,
    StepState,
    StoredBuild,
)
from mnemiq.superset.chart_check import ChartCheck, ChartStatus
from mnemiq.superset.charts import ChartSpec, Family, Filter, SavedRef, WrittenRef
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan, PlannedDataset, plan_from_json, plan_to_json

DSN_VARIABLE = "MNEMIQ_TEST_BUILD_STORE_DSN"
REQUIRED_VARIABLE = "MNEMIQ_TEST_BUILD_STORE_REQUIRED"

TENANT = "tenant-a"
OTHER_TENANT = "tenant-b"
ANA = "superset:ana"
BO = "superset:bo"
GOAL = "Written premium by region, and the claims trend"

# One build's steps, in the order a build runs them: the dashboard, the datasets, the charts, the layout, the checks.
STEPS = [("dashboard", "dashboard"), ("policies", "dataset"), ("premium_by_region", "chart"),
         ("claims_trend", "chart"), ("layout", "layout"), ("checks", "check")]
# The kinds whose step makes an object, and so finishes `done` with its id.
MAKES_AN_OBJECT = {"dashboard", "dataset", "chart"}

MEASURE = Measure(name="premium", label="Premium", expression="SUM(premium)", kind=MeasureKind.CURRENCY,
                  dataset_id=42, description="Written premium", asked_by=ANA, currency="USD")
CLAIMS = Measure(name="claims", label="Claims", expression="COUNT(*)", kind=MeasureKind.COUNT, dataset_id=42,
                 description="Claims filed", asked_by=BO)


def _spec(key: str, family: Family, title: str, measures: tuple, **more) -> ChartSpec:
    fields = {"dataset_id": 42, "dimensions": (), "time_column": None, "time_grain": None, "filters": (),
              "row_limit": None, "source": None, "target": None, **more}
    return ChartSpec(key=key, family=family, title=title, measures=measures, **fields)


# A plan as the build keeps it: what the planner's `plan_to_json` writes.
PLAN = plan_to_json(BuildPlan(
    charts=(_spec("premium_by_region", Family.BAR, "Premium by region", (WrittenRef(MEASURE),),
                  dimensions=("region",)),
            _spec("claims_trend", Family.LINE, "Claims", (WrittenRef(CLAIMS),), time_column="filed",
                  time_grain="P1M")),
    measures=(MEASURE, CLAIMS)))
# A plan holding every part `plan_to_json` writes: a saved metric and a written one, filters with one operand, a list
# and none, a dataset the build makes on a table and one on a join, and an offer to reuse a dashboard.
FULL_PLAN = plan_to_json(BuildPlan(
    charts=(_spec("premium_by_region", Family.BAR, "Premium by region",
                  (SavedRef("total_premium", MeasureKind.CURRENCY), WrittenRef(MEASURE)), dimensions=("region",),
                  filters=(Filter("region", "IN", ["west", "east"]), Filter("premium", ">", 0.25),
                           Filter("region", "IS NOT NULL")), row_limit=100),
            _spec("flow", Family.SANKEY, "Flow", (WrittenRef(CLAIMS),), dataset_id=-2, source="region",
                  target="channel")),
    measures=(MEASURE, dataclasses.replace(CLAIMS, dataset_id=-2)), reuse_candidates=(12,), reuse_reason="it fits",
    datasets=(PlannedDataset(key="dataset_1", placeholder=-1, table="policy", table_name="policy"),
              PlannedDataset(key="dataset_2", placeholder=-2, join="policy JOIN claim ON policy.id = claim.policy_id",
                             sql="SELECT 1", name="mnemiq_policy_claim"))))

# What a value, a row or a sample is kept under. The guard flags a column or a jsonb key at any depth whose name, folded
# (`_fold`), is one of these. It keeps its own list rather than reading the store's: a guard that shared the store's list
# could not catch that list being short.
VALUE_SHAPED = frozenset({"value", "values", "row", "rows", "data", "result", "results", "sample", "samples"})


def _fold(name: str) -> str:
    """A name as the guard compares it: NFKC (a fullwidth or compatibility spelling is the plain one), stripped of
    surrounding space, casefolded."""
    return unicodedata.normalize("NFKC", name).strip().casefold()

# The store's tables and their columns, each in its order.
STORE_COLUMNS = {
    BUILD_TABLE: ["tenant_id", "build_id", "principal_id", "dashboard_id", "goal", "profile", "plan", "steps", "status",
                  "created_at", "updated_at", "planned_against"],
    MEASURE_TABLE: ["tenant_id", "dashboard_id", "name", "definition", "chart_ids", "made_dataset_ids",
                    "check_status", "updated_at"],
}


def _intent(key: str, kind: str, name: str | None = None, dataset_id: int | None = None) -> dict:
    """The intent a builder records when it claims a step: the step, its kind, the planned name and dataset."""
    return {"key": key, "kind": kind, "name": name or key.replace("_", " "), "dataset_id": dataset_id}


def _offline_store() -> BuildStore:
    """A store whose DSN names no server: anything it refuses before connecting, it refuses here, and anything that
    reached the server would fail to connect instead of raising the refusal."""
    return BuildStore("postgresql://nobody@127.0.0.1:9/none?connect_timeout=1")


# --- The gate, with no server. Its outcome is caught rather than let through, as `test_pg_declared_keys.py` does: a
# test of whether the gate skips must not be one the gate can skip.

def _database_dsn() -> str:
    """The database the tests make their schemas in, from MNEMIQ_TEST_BUILD_STORE_DSN. Without it they skip, unless
    MNEMIQ_TEST_BUILD_STORE_REQUIRED is `1`, as in the CI job that gives them a server: there they fail instead, so that
    job cannot pass by skipping them. A switch set to anything but `1` or `0` fails too, rather than reading as unset."""
    required = os.environ.get(REQUIRED_VARIABLE, "").strip()
    if required not in ("", "0", "1"):
        pytest.fail(f"{REQUIRED_VARIABLE} is {required!r}: set it to 1 to make these tests fail rather than skip "
                    f"without {DSN_VARIABLE}, or to 0 or nothing to let them skip", pytrace=False)
    dsn = os.environ.get(DSN_VARIABLE, "").strip()
    if not dsn:
        reason = f"{DSN_VARIABLE} is not set: these tests need a Postgres database to make their own schemas in"
        if required == "1":
            pytest.fail(f"{reason}, and {REQUIRED_VARIABLE}=1 says they must run here, not skip", pytrace=False)
        pytest.skip(reason)
    return dsn


def _gate(monkeypatch, dsn: str | None, switch: str | None) -> tuple[str, str]:
    for name, value in ((DSN_VARIABLE, dsn), (REQUIRED_VARIABLE, switch)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    try:
        return "returned", _database_dsn()
    except pytest.skip.Exception as exc:
        return "skipped", str(exc)
    except pytest.fail.Exception as exc:
        return "failed", str(exc)


@pytest.mark.parametrize("switch", [None, "", "0"])
def test_without_the_dsn_and_without_the_switch_the_tests_skip_naming_the_dsn(monkeypatch, switch):
    outcome, reason = _gate(monkeypatch, None, switch)
    assert outcome == "skipped", reason
    assert DSN_VARIABLE in reason


def test_with_the_switch_and_without_the_dsn_the_tests_fail_naming_both(monkeypatch):
    outcome, reason = _gate(monkeypatch, None, "1")
    assert outcome == "failed", reason
    assert DSN_VARIABLE in reason and REQUIRED_VARIABLE in reason, reason


@pytest.mark.parametrize("switch", [None, "1"])
def test_with_the_dsn_the_tests_run_whatever_the_switch(monkeypatch, switch):
    assert _gate(monkeypatch, "postgresql://store@127.0.0.1/postgres", switch) == (
        "returned", "postgresql://store@127.0.0.1/postgres")


@pytest.mark.parametrize("dsn", [None, "postgresql://store@127.0.0.1/postgres"])
@pytest.mark.parametrize("switch", ["true", "yes", "on"])
def test_a_switch_set_to_anything_but_1_or_0_fails_naming_it(monkeypatch, dsn, switch):
    outcome, reason = _gate(monkeypatch, dsn, switch)
    assert outcome == "failed", reason
    assert REQUIRED_VARIABLE in reason and repr(switch) in reason, reason


# The workflow that gives these tests a server. Read as text: no YAML parser is a dependency of the `dev` extra.
WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "superset-build-store.yml"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _scalar(raw: str) -> str:
    """A plain or quoted YAML scalar's text, without a trailing comment."""
    quoted = re.match(r"""(["'])(.*?)\1""", raw)
    return quoted.group(2) if quoted else raw.split(" #", 1)[0].strip()


def _store_step_env(text: str) -> dict[str, str]:
    """The `env` of the one step in `text` whose `- run:` line runs pytest on this module, name to value, read by
    indentation as YAML nests it (`test_pg_declared_keys.py` reads its own workflow the same way)."""
    lines = text.splitlines()
    module = f"tests/{Path(__file__).name}"
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("- run:") and {"pytest", module} <= set(line.split())]
    assert len(starts) == 1, f"expected one step running pytest on {module} in {WORKFLOW.name}, found {len(starts)}"
    dash = _indent(lines[starts[0]])
    body = []
    for line in lines[starts[0] + 1:]:
        if line.strip() and not line.lstrip().startswith("#"):
            if _indent(line) <= dash:
                break
            body.append(line)
    envs = [i for i, line in enumerate(body) if line.strip() == "env:" and _indent(line) == dash + 2]
    assert len(envs) == 1, f"the step running {module} has no env of its own in {WORKFLOW.name}"
    env = {}
    for line in body[envs[0] + 1:]:
        if _indent(line) <= dash + 2:
            break
        name, _, value = line.strip().partition(":")
        env[name] = _scalar(value.strip())
    return env


def test_the_ci_workflow_sets_what_the_gate_reads():
    """The job that gives these tests a server sets the DSN and the switch under the names `_database_dsn` reads, the
    switch to `1`, so it can neither pass by skipping nor fail for want of a DSN set under another name."""
    env = _store_step_env(WORKFLOW.read_text())
    assert env.get(DSN_VARIABLE), f"{WORKFLOW.name} does not set {DSN_VARIABLE} for the store tests: {env}"
    assert env.get(REQUIRED_VARIABLE) == "1", f"{WORKFLOW.name} does not set {REQUIRED_VARIABLE} to 1: {env}"


# --- What the store refuses before it connects. None of these needs a server: each refusal is raised before any
# statement is sent, so the offline store raises it rather than failing to connect.

@pytest.mark.parametrize("dsn", [None, "", "   "])
def test_missing_store_is_named(dsn):
    with pytest.raises(BuildStoreMissing) as raised:
        BuildStore(dsn)
    assert "MNEMIQ_CONTROL_DSN" in str(raised.value)
    with pytest.raises(BuildStoreMissing) as raised:
        BuildStore.for_settings(SimpleNamespace(control_dsn=dsn))
    assert "MNEMIQ_CONTROL_DSN" in str(raised.value)


@pytest.mark.parametrize(("steps", "why"), [
    ([], "no step"),
    ([("policies", "dataset"), ("dashboard", "dashboard")], "the dashboard is not first"),
    ([("dashboard", "dashboard"), ("trend", "chart"), ("policies", "dataset")], "a chart before a dataset"),
    ([("dashboard", "dashboard"), ("checks", "check"), ("layout", "layout")], "the checks before the layout"),
    ([("dashboard", "dashboard"), ("dashboard_two", "dashboard")], "two dashboard steps"),
    ([("dashboard", "dashboard"), ("trend", "chart"), ("trend", "chart")], "a key twice"),
    ([("dashboard", "dashboard"), ("Trend", "chart")], "a key the marker cannot carry"),
    ([("dashboard", "dashboard"), ("trend line", "chart")], "a key with a space"),
    ([("dashboard", "dashboard"), ("t" * 41, "chart")], "a key longer than a marker's"),
    ([("dashboard", "dashboard"), ("trend", "picture")], "an unknown kind"),
])
def test_a_build_runs_dashboard_first_then_datasets_charts_layout_and_checks(steps, why):
    with pytest.raises(ValueError):
        _offline_store().create_build(TENANT, ANA, GOAL, PLAN, steps)


@pytest.mark.parametrize(("tenant", "principal", "goal"), [
    ("", ANA, GOAL), ("  ", ANA, GOAL), (None, ANA, GOAL), (TENANT, "", GOAL), (TENANT, ANA, ""), (TENANT, ANA, 3),
])
def test_a_build_needs_a_tenant_a_principal_and_a_goal(tenant, principal, goal):
    with pytest.raises(ValueError):
        _offline_store().create_build(tenant, principal, goal, PLAN, STEPS)


@pytest.mark.parametrize("plan", [
    {"charts": [{"key": "trend", "rows": [["West", 1]]}]},
    {"measures": [{"name": "premium", "value": 12.5}]},
    {"charts": [{"filters": [{"column": "region", "op": "==", "value": "West"}]}]},
    {"result": {"chart": 1}},
    {"Data": []},
    {"charts": [{"values": []}]},
    {"row": {"region": "West"}},
    {"measures": [{"results": []}]},
    {"sample": "West"},
    {"Samples": ["West", "East"]},
    {"ｖａｌｕｅ": 1},
    {"charts": [{"value ": 1}]},
    {" DATA\t": []},
])
def test_a_plan_holding_a_value_shaped_key_is_refused(plan):
    """At any depth, in any letter case, in a fullwidth spelling or with space around it: a key that reads, folded, as
    one a value, a row or a sample is kept under is refused. A backstop: the guarantee is that the planner is given no
    value. `test_the_json_the_store_writes_is_jsonb_postgres_reads` keeps a `row_limit`, which only contains a word."""
    with pytest.raises(ValueError):
        _offline_store().create_build(TENANT, ANA, GOAL, plan, STEPS)


def _planted(path: tuple, key: str | None, item: object) -> dict:
    """`FULL_PLAN` with `item` put under `key` in the object at `path`, or, with no key, in place of what is at
    `path`."""
    plan = copy.deepcopy(FULL_PLAN)
    node = plan
    for step in path if key is not None else path[:-1]:
        node = node[step]
    node[key if key is not None else path[-1]] = item
    return plan


SENTINEL = 987654.321
# Where a value could hide in a kept plan, and what is planted there.
PLANTED = {
    "a key beside the plan's own": ((), "notes", SENTINEL),
    "a key in a measure": (("measures", 0), "note", SENTINEL),
    "a key in a chart": (("charts", 0), "colour", SENTINEL),
    "a key in a chart's measure": (("charts", 0, "measures", 0), "label", SENTINEL),
    "a key in a filter": (("charts", 0, "filters", 0), "comment", SENTINEL),
    "a key in a planned dataset": (("datasets", 1), "rows_estimate", SENTINEL),
    "a key in the reuse offer": (("reuse",), "score", SENTINEL),
    "an object for a filter's operand": (("charts", 0, "filters", 1, "operand"), None, {"planted": SENTINEL}),
    "an object in a filter's listed operand": (("charts", 0, "filters", 0, "operand", 0), None, {"planted": SENTINEL}),
    "an object among a chart's dimensions": (("charts", 0, "dimensions", 0), None, {"planted": SENTINEL}),
    "an object for the format": (("format",), None, {"planted": SENTINEL}),
    "an object among the offered dashboards": (("reuse", "candidates", 0), None, {"planted": SENTINEL}),
    "a list for a measure": (("measures", 0), None, [SENTINEL]),
    "a list for a chart's title": (("charts", 0, "title"), None, [SENTINEL]),
}


def test_the_plan_check_admits_every_part_the_planner_writes():
    """Known good: every key `plan_to_json` writes, at every depth, in a plan holding each kind of part, passes the
    check, and reads back as the plan it was."""
    kept = build_store._plan(FULL_PLAN)
    assert kept == FULL_PLAN
    assert plan_to_json(plan_from_json(kept)) == FULL_PLAN
    assert build_store._plan(PLAN) == PLAN


@pytest.mark.parametrize("case", list(PLANTED))
def test_the_plan_check_is_an_allow_list_of_what_the_planner_writes(case):
    """Planted: a key `plan_to_json` never writes, at any depth, or an object or a list where it writes a single
    value, is refused before anything is sent, whatever the key is called. A value kept under any name has nowhere to
    go."""
    plan = _planted(*PLANTED[case])
    with pytest.raises(ValueError):
        build_store._plan(plan)
    with pytest.raises(ValueError):
        _offline_store().create_build(TENANT, ANA, GOAL, plan, STEPS)


def test_the_store_holds_a_plan_to_the_shape_the_planner_declares(monkeypatch):
    """The allowed shape is the planner's own declaration beside its writer (`planner.PLAN_JSON_SHAPE`): a key the
    planner declares is admitted, with nothing in the store changed, and refused again once it no longer does."""
    import mnemiq.superset.planner as planner

    with_subtitle = copy.deepcopy(FULL_PLAN)
    with_subtitle["charts"][0]["subtitle"] = "by region"
    with pytest.raises(ValueError):
        build_store._plan(with_subtitle)
    widened = copy.deepcopy(planner.PLAN_JSON_SHAPE)
    widened["charts"][0]["subtitle"] = planner.SCALAR
    monkeypatch.setattr(planner, "PLAN_JSON_SHAPE", widened)
    assert build_store._plan(with_subtitle) == with_subtitle
    monkeypatch.undo()
    with pytest.raises(ValueError):
        build_store._plan(with_subtitle)


@pytest.mark.parametrize("plan", [{"charts": [object()]}, {1: "one"}, {"limit": math.nan}, ["not", "a", "mapping"]])
def test_a_plan_that_is_not_json_is_refused(plan):
    with pytest.raises(ValueError):
        _offline_store().create_build(TENANT, ANA, GOAL, plan, STEPS)


@pytest.mark.parametrize("intent", [
    {"kind": "dashboard", "name": "x"},
    {"key": "dashboard", "name": "x"},
    {"key": "dashboard", "kind": "dashboard", "value": 987654.321},
    {"key": "dashboard", "kind": "dashboard", "rows": [[1]]},
    {"key": "dashboard", "kind": "dashboard", "started_at": "2026-01-01T00:00:00+00:00"},
    {"key": "dashboard", "kind": "dashboard", "name": 987654.321},
    {"key": "dashboard", "kind": "dashboard", "dataset_id": 987654.321},
    {"key": "dashboard", "kind": "dashboard", "dataset_id": True},
    {"key": "dashboard", "kind": "picture"},
    {"key": "dashboard", "kind": "dashboard", "id_floor": 41},
    {"key": "dashboard", "kind": "dashboard", "id_floor": None},
])
def test_an_intent_holds_the_step_its_kind_its_name_and_its_dataset_only(intent):
    """The intent a claim records: the step, its kind, the planned name, the dataset. `started_at` is the store's own, read
    from Postgres's clock, and nothing else is accepted, so no value can ride in an intent. An `id_floor`, which an
    earlier mnemiq recorded before a create, is no longer written."""
    with pytest.raises(ValueError):
        _offline_store().claim_next_step(TENANT, str(uuid.uuid4()), ANA, intent)


@pytest.mark.parametrize("lease_s", [0, -1, 1.5, True, "600"])
def test_a_lease_is_a_positive_whole_number_of_seconds(lease_s):
    with pytest.raises(ValueError):
        _offline_store().claim_next_step(TENANT, str(uuid.uuid4()), ANA, _intent("dashboard", "dashboard"),
                                         lease_s=lease_s)


@pytest.mark.parametrize(("status", "object_id"), [
    ("running", 41), ("pending", 41), ("done", 987654.321), ("done", -3), ("done", True), ("failed", "41"),
])
def test_a_step_finishes_done_failed_or_uncertain_with_an_id(status, object_id):
    with pytest.raises(ValueError):
        _offline_store().finish_step(TENANT, str(uuid.uuid4()), "dashboard", status, object_id,
                                     lease_id=str(uuid.uuid4()))


@pytest.mark.parametrize("reason", ["the dataset vanished", "NOT_EDITOR", 3, "", "superset said: no"])
def test_a_step_keeps_only_a_reason_code_of_the_stores(reason):
    """A step's reason is one of the store's codes (`REASONS`), never free text, so Superset's words cannot be kept
    there."""
    assert "not_editor" in build_store.REASONS and "abandoned" in build_store.REASONS
    with pytest.raises(ValueError):
        _offline_store().finish_step(TENANT, str(uuid.uuid4()), "dashboard", "failed", None,
                                     lease_id=str(uuid.uuid4()), reason=reason)


@pytest.mark.parametrize("saved_metric", [3, "", ["total_premium"]])
def test_a_measure_names_its_saved_metric_as_text(saved_metric):
    with pytest.raises(ValueError):
        _offline_store().put_measure(TENANT, 7, MEASURE, saved_metric=saved_metric)


@pytest.mark.parametrize("change", [
    {"check_status": {43: 987654.321}},
    {"check_status": {43: "fine"}},
    {"check_status": {"forty-three": ChartStatus.CHECKED}},
    {"chart_ids": (43, 987654.321)},
    {"chart_ids": (0,)},
    {"made_dataset_ids": ("42",)},
])
def test_a_measure_row_holds_ids_and_statuses_only(change):
    """`chart_ids` and `made_dataset_ids` are ids, and `check_status` maps a chart id to a check status, so a number
    computed from the data has nowhere to go."""
    kept = {"chart_ids": (43,), "made_dataset_ids": (), "check_status": {}} | change
    with pytest.raises(ValueError):
        _offline_store().put_measure(TENANT, 7, MEASURE, **kept)


# --- Against a server.

@pytest.fixture
def store_dsn() -> Iterator[str]:
    """A DSN whose search path is a schema made for this test alone, dropped after it."""
    database = _database_dsn()
    schema = f"build_store_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield make_conninfo(database, options=f"-c search_path={schema}")
    finally:
        with psycopg.connect(database, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def stores(store_dsn) -> Iterator:
    """Stores on the test's schema, each with a connection of its own, closed when the test is done."""
    made: list[BuildStore] = []

    def make() -> BuildStore:
        made.append(BuildStore(store_dsn))
        return made[-1]

    try:
        yield make
    finally:
        for store in made:
            store.close()


@pytest.fixture
def store(stores) -> BuildStore:
    return stores()


def _raw_build(dsn: str, build_id: str) -> dict:
    """The build's row as Postgres holds it, read on a connection of its own."""
    with psycopg.connect(dsn, autocommit=True) as con:
        cur = con.execute(sql.SQL("SELECT * FROM {} WHERE build_id = %s").format(sql.Identifier(BUILD_TABLE)),
                          (build_id,))
        names = [column.name for column in cur.description]
        row = cur.fetchone()
    assert row is not None, f"no row for build {build_id}"
    return dict(zip(names, row, strict=True))


def _raw_step(dsn: str, build_id: str, key: str) -> dict:
    [step] = [step for step in _raw_build(dsn, build_id)["steps"] if step["key"] == key]
    return step


def _run_step(store: BuildStore, build: StoredBuild, key: str, kind: str, object_id: int | None,
              status: str = "done") -> StoredBuild:
    """Claim `key`, renew the lease once as a step does before its second call, and finish it."""
    claimed = store.claim_next_step(build.tenant_id, build.build_id, build.principal_id, _intent(key, kind))
    assert isinstance(claimed, StepState) and claimed.key == key, claimed
    assert store.renew_lease(build.tenant_id, build.build_id, key, claimed.lease_id)
    return store.finish_step(build.tenant_id, build.build_id, key, status, object_id, lease_id=claimed.lease_id)


def _run_build(store: BuildStore, tenant: str = TENANT, principal: str = ANA) -> StoredBuild:
    """A whole build, every step claimed and finished in order, and the dashboard's two measures put with the charts
    that show them and each chart's check status."""
    build = store.create_build(tenant, principal, GOAL, PLAN, STEPS)
    ids = iter(range(41, 100))
    for key, kind in STEPS:
        build = _run_step(store, build, key, kind, next(ids) if kind in MAKES_AN_OBJECT else None)
    store.put_measure(tenant, build.dashboard_id, MEASURE, chart_ids=(43,), made_dataset_ids=(42,),
                      check_status={43: ChartStatus.CHECKED})
    store.put_measure(tenant, build.dashboard_id, CLAIMS, chart_ids=(44,), made_dataset_ids=(42,),
                      check_status={44: ChartStatus.NOT_CHECKED})
    return build


def _keys(doc, path: str = "") -> Iterator[tuple[str, str]]:
    """Every key in a jsonb document at any depth, with the path to it."""
    if isinstance(doc, dict):
        for key, item in doc.items():
            yield f"{path}.{key}", key
            yield from _keys(item, f"{path}.{key}")
    elif isinstance(doc, list):
        for index, item in enumerate(doc):
            yield from _keys(item, f"{path}[{index}]")


def value_shaped_names(con: psycopg.Connection) -> tuple[list[str], dict[str, list[str]], int]:
    """The guard: every column of the store's tables, and every key at any depth of every row of their jsonb columns,
    whose name, folded, is one a value, a row or a sample is kept under. The tables and columns are read from Postgres's own
    `information_schema.columns`, in the connection's current schema, every table named as the store's are; so is
    which columns are jsonb. Returns what it flagged, the columns it read by table, and how many jsonb keys it read."""
    columns = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name LIKE %s ORDER BY table_name, ordinal_position",
        (r"mnemiq\_superset\_%",)).fetchall()
    flagged = [f"{table}.{column}" for table, column, _ in columns if _fold(column) in VALUE_SHAPED]
    read: dict[str, list[str]] = {}
    keys = 0
    for table, column, data_type in columns:
        read.setdefault(table, []).append(column)
        if data_type != "jsonb":
            continue
        for (doc,) in con.execute(sql.SQL("SELECT {} FROM {}").format(sql.Identifier(column), sql.Identifier(table))):
            for path, key in _keys(doc):
                keys += 1
                if _fold(key) in VALUE_SHAPED:
                    flagged.append(f"{table}.{column}{path}")
    return flagged, read, keys


def _stopped_midway(store: BuildStore) -> StoredBuild:
    """A second build whose dataset step timed out: its intent is kept, for reconciliation."""
    build = store.create_build(TENANT, BO, GOAL, PLAN, STEPS)
    build = _run_step(store, build, "dashboard", "dashboard", 61)
    claimed = store.claim_next_step(TENANT, build.build_id, BO, _intent("policies", "dataset", dataset_id=None))
    return store.finish_step(TENANT, build.build_id, "policies", "uncertain", None, lease_id=claimed.lease_id)


def test_store_holds_no_value(store, store_dsn):
    """The known-good case: after whole builds and their measures, no column and no jsonb key at any depth is named as
    a value is kept. The guard read every column of both tables, and walked into the nested documents
    (an intent's keys, a definition's), so it passed by reading, not by finding nothing to read."""
    _run_build(store)
    _stopped_midway(store)
    with psycopg.connect(store_dsn, autocommit=True) as con:
        flagged, read, keys = value_shaped_names(con)
        jsonb = {(table, column) for table, column in con.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND data_type = 'jsonb'")}
    assert flagged == []
    assert read == STORE_COLUMNS
    assert jsonb == {(BUILD_TABLE, "plan"), (BUILD_TABLE, "steps"), (MEASURE_TABLE, "definition"),
                     (MEASURE_TABLE, "check_status")}
    assert keys > 100, keys


@pytest.mark.parametrize(("plant", "caught"), [
    (sql.SQL("ALTER TABLE {} ADD COLUMN value numeric").format(sql.Identifier(MEASURE_TABLE)),
     f"{MEASURE_TABLE}.value"),
    (sql.SQL("UPDATE {} SET plan = jsonb_set(plan, '{{charts,0,filters}}', "
             "'[{{\"column\": \"region\", \"op\": \"==\", \"value\": \"West\"}}]')").format(sql.Identifier(BUILD_TABLE)),
     f"{BUILD_TABLE}.plan.charts[0].filters[0].value"),
    (sql.SQL("UPDATE {} SET check_status = check_status || '{{\"43\": {{\"result\": 1}}}}'").format(
        sql.Identifier(MEASURE_TABLE)),
     f"{MEASURE_TABLE}.check_status.43.result"),
    (sql.SQL('ALTER TABLE {} ADD COLUMN "Samples" text').format(sql.Identifier(MEASURE_TABLE)),
     f"{MEASURE_TABLE}.Samples"),
    (sql.SQL("UPDATE {} SET plan = plan || '{{\"ｖａｌｕｅ\": 1}}'").format(sql.Identifier(BUILD_TABLE)),
     f"{BUILD_TABLE}.plan.ｖａｌｕｅ"),
    (sql.SQL("UPDATE {} SET steps = jsonb_set(steps, ARRAY['0', 'intent', 'value '], '1')").format(
        sql.Identifier(BUILD_TABLE)),
     f"{BUILD_TABLE}.steps[0].intent.value "),
], ids=["a column", "a key deep in a plan", "a key in a check status", "a mixed-case column", "a fullwidth key",
        "a key with a trailing space"])
def test_store_holds_no_value_catches_a_planted_value(store, store_dsn, plant, caught):
    """The guard's planted-violation twin. The store refuses a value-shaped key itself, so the defect is planted past
    it, in a transaction that is rolled back: a `value` column, a filter's `value` deep in a stored plan, a `result`
    inside a check status, a quoted `"Samples"` column, a fullwidth `ｖａｌｕｅ` key and a `value ` key with a trailing
    space. Each is flagged, by its place, and once rolled back the store reads clean again."""
    _run_build(store)
    with psycopg.connect(store_dsn) as con:
        con.execute(plant)
        flagged, _, _ = value_shaped_names(con)
        con.rollback()
        assert value_shaped_names(con)[0] == []
    assert caught in flagged, flagged


SENTINEL = 987654.321


def test_sentinel_value_never_persisted_through_the_stores_doors(store, store_dsn):
    """A build step driven by hand, whose Superset answer and value check both hold `987654.321`, records all it may
    (the chart's id, the step's state, the measure, its charts and the check's status), and no stored row's text holds
    the number, nor its rounding to cents. Every door that takes free-form input refuses it as well. The probe is shown
    to read the rows: it finds the step's chart id and the goal, and finds the number once it is planted past the
    store. `test_superset_builder.py` holds a whole build, its check step included, to the same."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    build = _run_step(store, build, "dashboard", "dashboard", 41)
    build = _run_step(store, build, "policies", "dataset", 42)

    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("premium_by_region", "chart", dataset_id=42))
    answer = {"chart_id": 43, "rows": [{"region": "West", "premium": SENTINEL}]}  # what the fake Superset returns
    check = ChartCheck(chart_id=answer["chart_id"], status=ChartStatus.CHECKED, reason="",
                       chart_values=(("West", SENTINEL),), mnemiq_values=(("West", SENTINEL),))
    build = store.finish_step(TENANT, build.build_id, "premium_by_region", "done", answer["chart_id"],
                              lease_id=claimed.lease_id)
    store.put_measure(TENANT, build.dashboard_id, MEASURE, chart_ids=(check.chart_id,), made_dataset_ids=(42,),
                      check_status={check.chart_id: check.status})

    refused = [
        lambda: store.put_measure(TENANT, build.dashboard_id, MEASURE, chart_ids=(43,), made_dataset_ids=(),
                                  check_status={43: check.chart_values[0][1]}),
        lambda: store.claim_next_step(TENANT, build.build_id, ANA,
                                      _intent("claims_trend", "chart") | {"value": SENTINEL}),
        lambda: store.create_build(TENANT, ANA, GOAL, PLAN | {"rows": answer["rows"]}, STEPS),
        lambda: store.finish_step(TENANT, build.build_id, "claims_trend", "done", SENTINEL,
                                  lease_id=claimed.lease_id),
    ]
    for door in refused:
        with pytest.raises(ValueError):
            door()

    def stored_text(con) -> str:
        return "\n".join(text for table in STORE_COLUMNS for (text,) in con.execute(
            sql.SQL("SELECT row_to_json(t)::text FROM {} AS t").format(sql.Identifier(table))))

    with psycopg.connect(store_dsn) as con:
        text = stored_text(con)
        assert str(SENTINEL) not in text and "987654.32" not in text
        assert GOAL in text and '"object_id": 43' in text  # the probe reads what the step recorded
        con.execute(sql.SQL("UPDATE {} SET goal = goal || %s").format(sql.Identifier(BUILD_TABLE)), (str(SENTINEL),))
        assert str(SENTINEL) in stored_text(con)  # and finds the number once it is there
        con.rollback()


def test_the_tables_are_made_lazily_and_once(store_dsn, stores):
    """No table exists until a store first connects; then both do, and a second store on the same schema finds them
    and reads the first one's rows."""
    with psycopg.connect(store_dsn, autocommit=True) as con:
        def tables() -> set[str]:
            return {name for (name,) in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}

        assert tables() == set()
        first = stores()
        assert tables() == set()  # a store connects on its first call, not when it is made
        build = first.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
        assert tables() == {BUILD_TABLE, MEASURE_TABLE}
        assert stores().load_build(TENANT, build.build_id, ANA) == build


def test_a_build_is_stored_as_created(store, store_dsn):
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    assert uuid.UUID(build.build_id).hex == build.build_id.replace("-", "") and build.build_id == build.build_id.lower()
    assert (build.tenant_id, build.principal_id, build.goal, build.plan, build.dashboard_id, build.status) == (
        TENANT, ANA, GOAL, PLAN, 12, "building")
    assert [(step.key, step.kind, step.status, step.object_id, step.started_at, step.intent, step.lease_id)
            for step in build.steps] == [(key, kind, "pending", None, None, None, None) for key, kind in STEPS]
    assert store.load_build(TENANT, build.build_id, ANA) == build
    assert _raw_build(store_dsn, build.build_id)["status"] == "building"


def test_another_tenant_cannot_load(store, store_dsn):
    """Another tenant, with the build's own principal and id, neither loads nor claims the build, nor renews or
    finishes its step with the claim's own lease, nor reads or deletes the tenant's measures on a dashboard of the same
    id. The control: the build's own tenant does each."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    assert store.load_build(OTHER_TENANT, build.build_id, ANA) is None
    with pytest.raises(BuildNotFound):
        store.claim_next_step(OTHER_TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    assert _raw_step(store_dsn, build.build_id, "dashboard")["lease_id"] is None

    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    assert isinstance(claimed, StepState)
    assert store.renew_lease(OTHER_TENANT, build.build_id, "dashboard", claimed.lease_id) is False
    with pytest.raises((BuildNotFound, LeaseLost)):
        store.finish_step(OTHER_TENANT, build.build_id, "dashboard", "done", 41, lease_id=claimed.lease_id)
    assert _raw_step(store_dsn, build.build_id, "dashboard")["status"] == "pending"
    assert store.renew_lease(TENANT, build.build_id, "dashboard", claimed.lease_id) is True
    finished = store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=claimed.lease_id)
    assert finished.dashboard_id == 41

    store.put_measure(TENANT, 41, MEASURE, chart_ids=(43,), made_dataset_ids=(), check_status={})
    assert store.measures_for(OTHER_TENANT, 41) == []
    assert store.delete_measure(OTHER_TENANT, 41, MEASURE.name) is False
    assert [kept.measure for kept in store.measures_for(TENANT, 41)] == [MEASURE]


def test_a_build_keeps_the_profile_it_was_started_under(store, store_dsn):
    """The profile a build was started under is kept on its row and read back with it, by every read and write that
    returns the build; a build started under none keeps none (the deployment's default). A profile is a name: empty or
    not text, it is refused before anything is written."""
    named = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, profile="b")
    plain = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    assert (named.profile, plain.profile) == ("b", None)
    assert store.load_build(TENANT, named.build_id, ANA).profile == "b"
    assert store.load_build(TENANT, plain.build_id, ANA).profile is None
    claimed = store.claim_next_step(TENANT, named.build_id, ANA, _intent("dashboard", "dashboard"))
    finished = store.finish_step(TENANT, named.build_id, "dashboard", "done", 300, lease_id=claimed.lease_id)
    assert finished.profile == "b"
    with psycopg.connect(store_dsn, autocommit=True) as con:
        kept = dict(con.execute(sql.SQL("SELECT build_id::text, profile FROM {}").format(sql.Identifier(BUILD_TABLE)))
                    .fetchall())
    assert kept == {named.build_id: "b", plain.build_id: None}
    for bad in ("", "   ", 7):
        with pytest.raises(ValueError):
            store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, profile=bad)
    assert len(kept) == 2


def test_another_principal_cannot_load_a_build(store, store_dsn):
    """Another principal of the same tenant neither loads nor claims the build, and the build is left unclaimed. The
    control: its own principal loads and claims it."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    assert store.load_build(TENANT, build.build_id, BO) is None
    with pytest.raises(BuildNotFound):
        store.claim_next_step(TENANT, build.build_id, BO, _intent("dashboard", "dashboard"))
    assert _raw_step(store_dsn, build.build_id, "dashboard")["lease_id"] is None
    assert store.load_build(TENANT, build.build_id, ANA) == build
    assert isinstance(store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard")), StepState)


@pytest.mark.parametrize("build_id", ["not-a-uuid", "", "00000000-0000-0000-0000-000000000000"])
def test_a_build_id_that_names_no_build_loads_nothing(store, build_id):
    store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    assert store.load_build(TENANT, build_id, ANA) is None
    with pytest.raises(BuildNotFound):
        store.claim_next_step(TENANT, build_id, ANA, _intent("dashboard", "dashboard"))


def test_measures_are_the_dashboards(store):
    """A dashboard's measures are the dashboard's, not their writer's: any principal of the tenant reads all of them,
    since Superset checks editorship on each action. Each comes back as it was put, a second put replaces it, and a
    delete removes it alone. Another dashboard's measures are not listed."""
    store.put_measure(TENANT, 7, MEASURE, chart_ids=(43, 45), made_dataset_ids=(42,),
                      check_status={43: ChartStatus.CHECKED, 45: ChartStatus.MISMATCH})
    store.put_measure(TENANT, 7, CLAIMS, chart_ids=(44,), made_dataset_ids=(), check_status={44: "not_checked"})
    store.put_measure(TENANT, 8, CLAIMS, chart_ids=(50,), made_dataset_ids=(), check_status={})

    kept = store.measures_for(TENANT, 7)
    assert all(isinstance(item, DashboardMeasure) for item in kept)
    assert [(item.measure, item.chart_ids, item.made_dataset_ids, item.check_status) for item in kept] == [
        (CLAIMS, (44,), (), {44: ChartStatus.NOT_CHECKED}),
        (MEASURE, (43, 45), (42,), {43: ChartStatus.CHECKED, 45: ChartStatus.MISMATCH}),
    ]
    assert {item.measure.asked_by for item in kept} == {ANA, BO}

    relabelled = dataclasses.replace(MEASURE, label="Written premium", currency="EUR")
    store.put_measure(TENANT, 7, relabelled, chart_ids=(43,), made_dataset_ids=(42,), check_status={})
    assert [(item.measure, item.chart_ids, item.check_status) for item in store.measures_for(TENANT, 7)] == [
        (CLAIMS, (44,), {44: ChartStatus.NOT_CHECKED}), (relabelled, (43,), {})]

    assert store.delete_measure(TENANT, 7, MEASURE.name) is True
    assert store.delete_measure(TENANT, 7, MEASURE.name) is False
    assert [item.measure for item in store.measures_for(TENANT, 7)] == [CLAIMS]
    assert [item.measure for item in store.measures_for(TENANT, 8)] == [CLAIMS]


def test_a_measure_definition_is_kept_under_its_seven_keys(store, store_dsn):
    store.put_measure(TENANT, 7, MEASURE, chart_ids=(43,), made_dataset_ids=(42,), check_status={})
    with psycopg.connect(store_dsn, autocommit=True) as con:
        [(definition,)] = con.execute(sql.SQL("SELECT definition FROM {}").format(sql.Identifier(MEASURE_TABLE)))
    assert definition == {"expression": "SUM(premium)", "label": "Premium", "kind": "currency", "currency": "USD",
                          "description": "Written premium", "asked_by": ANA, "dataset_id": 42}


def test_steps_are_claimed_one_at_a_time_in_the_builds_order(store):
    """The next step is the first not done. An intent naming any other step claims nothing; a second claim while a
    lease holds answers busy, naming the step; once every step is done there is nothing to claim and the build is
    done. The dashboard step's id becomes the build's dashboard."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    moved = store.claim_next_step(TENANT, build.build_id, ANA, _intent("policies", "dataset"))
    assert moved == StepBusy(key=None)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    assert (claimed.key, claimed.kind, claimed.status) == ("dashboard", "dashboard", "pending")
    assert store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard")) == StepBusy(
        key="dashboard")
    assert store.claim_next_step(TENANT, build.build_id, ANA, _intent("policies", "dataset")) == StepBusy(
        key="dashboard")
    build = store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=claimed.lease_id)
    assert build.dashboard_id == 41 and build.step("dashboard").object_id == 41 and build.status == "building"
    assert build.step("dashboard").lease_id is None
    ids = iter(range(42, 100))
    for key, kind in STEPS[1:]:
        build = _run_step(store, build, key, kind, next(ids) if kind in MAKES_AN_OBJECT else None)
    assert build.status == "done" and {step.status for step in build.steps} == {"done"}
    assert store.claim_next_step(TENANT, build.build_id, ANA, _intent("checks", "check")) is None
    assert store.load_build(TENANT, build.build_id, ANA) == build


def test_a_failed_step_stops_the_build(store):
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    build = _run_step(store, build, "dashboard", "dashboard", 41)
    build = _run_step(store, build, "policies", "dataset", None, status="failed")
    assert build.status == "failed" and build.step("policies").status == "failed"
    assert store.claim_next_step(TENANT, build.build_id, ANA, _intent("premium_by_region", "chart")) is None
    assert store.claim_next_step(TENANT, build.build_id, ANA, _intent("policies", "dataset")) is None


def test_an_existing_dashboard_is_the_builds_from_the_start(store):
    """A build on an existing dashboard knows its id from the start; its dashboard step, the read of that dashboard,
    finishes with that id and no other."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    with pytest.raises(ValueError):
        store.finish_step(TENANT, build.build_id, "dashboard", "done", 13, lease_id=claimed.lease_id)
    assert store.load_build(TENANT, build.build_id, ANA).step("dashboard").status == "pending"
    build = store.finish_step(TENANT, build.build_id, "dashboard", "done", 12, lease_id=claimed.lease_id)
    assert build.dashboard_id == 12


@pytest.mark.parametrize(("key", "kind"), [("dashboard", "dashboard"), ("policies", "dataset"),
                                           ("premium_by_region", "chart")])
def test_a_step_that_makes_an_object_finishes_done_with_its_id(store, key, kind):
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    ids = iter(range(41, 100))
    for earlier, earlier_kind in STEPS[:[k for k, _ in STEPS].index(key)]:
        build = _run_step(store, build, earlier, earlier_kind, next(ids))
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent(key, kind))
    with pytest.raises(ValueError):
        store.finish_step(TENANT, build.build_id, key, "done", None, lease_id=claimed.lease_id)
    assert store.finish_step(TENANT, build.build_id, key, "done", 77, lease_id=claimed.lease_id).step(
        key).object_id == 77


def test_two_concurrent_claims_one_wins(store_dsn, stores):
    """Two stores, each on a connection of its own, claim the build's first step at once. A third connection holds the
    build's row until both have read it and are waiting to write it, so both decide on the same row; then it lets go.
    One gets the step and the other `StepBusy`, naming it, and the row holds the winner's lease alone."""
    build = stores().create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    first, second = stores(), stores()
    for racer in (first, second):  # each connects, and finds the tables made, before the race
        assert racer.load_build(TENANT, build.build_id, ANA) == build
    intent = _intent("dashboard", "dashboard")
    with psycopg.connect(store_dsn) as holder, psycopg.connect(store_dsn, autocommit=True) as watcher:
        holder.execute(sql.SQL("SELECT 1 FROM {} WHERE build_id = %s FOR UPDATE").format(
            sql.Identifier(BUILD_TABLE)), (build.build_id,))
        holder_pid = holder.info.backend_pid
        with ThreadPoolExecutor(max_workers=2) as pool:
            racing = [pool.submit(racer.claim_next_step, TENANT, build.build_id, ANA, intent)
                      for racer in (first, second)]
            deadline = time.monotonic() + 20
            waiting = 0
            while time.monotonic() < deadline and waiting < 2:
                [(waiting,)] = watcher.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock' AND pid <> %s", (holder_pid,)).fetchall()
                if waiting < 2:
                    time.sleep(0.02)
            holder.rollback()
            results = [future.result(timeout=30) for future in racing]
    assert waiting == 2, "the two claims did not both reach the write while the row was held"
    won = [result for result in results if isinstance(result, StepState)]
    lost = [result for result in results if isinstance(result, StepBusy)]
    assert len(won) == 1 and lost == [StepBusy(key="dashboard")], results
    stored = _raw_step(store_dsn, build.build_id, "dashboard")
    assert stored["lease_id"] == won[0].lease_id


def test_expired_lease_becomes_uncertain(store, store_dsn):
    """While the lease holds, a second claim is busy. Once it has passed, the step reads `uncertain`, and the next claim
    takes it to reconcile: still `uncertain`, with the first claim's intent and `started_at`, from which reconciliation
    looks for what the first worker made, and a lease of its own. The first worker can then neither renew nor finish."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    intent = _intent("dashboard", "dashboard", name="Premium")
    first = store.claim_next_step(TENANT, build.build_id, ANA, intent, lease_s=1)
    assert isinstance(first, StepState) and first.status == "pending"
    assert store.claim_next_step(TENANT, build.build_id, ANA, intent) == StepBusy(key="dashboard")
    time.sleep(1.5)
    assert store.load_build(TENANT, build.build_id, ANA).step("dashboard").status == "uncertain"

    again = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard", name="Other"))
    assert isinstance(again, StepState)
    assert (again.key, again.status, again.intent, again.started_at) == (
        "dashboard", "uncertain", intent, first.started_at)
    assert again.lease_id != first.lease_id
    assert store.renew_lease(TENANT, build.build_id, "dashboard", first.lease_id) is False
    with pytest.raises(LeaseLost):
        store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=first.lease_id)
    stored = _raw_step(store_dsn, build.build_id, "dashboard")
    assert (stored["status"], stored["lease_id"], stored["intent"]) == ("uncertain", again.lease_id, intent)
    assert store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=again.lease_id).step(
        "dashboard").status == "done"


def test_a_step_left_uncertain_is_reconciled_next(store):
    """A worker whose call timed out leaves its step `uncertain` at once, giving up its lease, so the next claim takes
    it to reconcile without waiting for the lease to pass."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    build = _run_step(store, build, "dashboard", "dashboard", 41)
    intent = _intent("policies", "dataset", name="policies", dataset_id=None)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, intent)
    build = store.finish_step(TENANT, build.build_id, "policies", "uncertain", None, lease_id=claimed.lease_id)
    assert build.step("policies").status == "uncertain" and build.step("policies").lease_id is None
    again = store.claim_next_step(TENANT, build.build_id, ANA, intent)
    assert (again.status, again.intent, again.started_at) == ("uncertain", intent, claimed.started_at)


def test_a_renewed_lease_runs_from_the_renewal(store, store_dsn):
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=60)
    before = _raw_step(store_dsn, build.build_id, "dashboard")["lease_until"]
    assert store.renew_lease(TENANT, build.build_id, "dashboard", claimed.lease_id, lease_s=600) is True
    after = _raw_step(store_dsn, build.build_id, "dashboard")["lease_until"]
    assert datetime.fromisoformat(after) > datetime.fromisoformat(before)
    assert store.renew_lease(TENANT, build.build_id, "dashboard", str(uuid.uuid4())) is False
    assert store.renew_lease(TENANT, build.build_id, "policies", claimed.lease_id) is False


def test_intent_is_saved_before_the_call(store, store_dsn, stores):
    """While the step's call runs, the build's row, read on another connection, already holds the claim's intent, its
    `started_at` and its lease: what reconciliation needs if the worker dies inside the call."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    build = _run_step(store, build, "dashboard", "dashboard", 41)
    intent = _intent("policies", "dataset", name="policies", dataset_id=None)
    seen: dict = {}

    def call() -> int:  # the fake Superset call: it looks at the store from elsewhere, as a second worker would
        seen["row"] = _raw_step(store_dsn, build.build_id, "policies")
        seen["loaded"] = stores().load_build(TENANT, build.build_id, ANA).step("policies")
        return 42

    claimed = store.claim_next_step(TENANT, build.build_id, ANA, intent)
    made = call()
    store.finish_step(TENANT, build.build_id, "policies", "done", made, lease_id=claimed.lease_id)

    row = seen["row"]
    assert row["intent"] == intent and row["status"] == "pending" and row["object_id"] is None
    assert row["started_at"] == claimed.started_at and row["lease_id"] == claimed.lease_id
    assert datetime.fromisoformat(row["lease_until"]) > datetime.fromisoformat(row["started_at"])
    assert seen["loaded"].intent == intent and seen["loaded"].lease_id == claimed.lease_id


class _AnHourAhead(datetime):
    """A Python clock an hour ahead of Postgres's. The store reads the time only from Postgres, so it never sees this;
    a store that read the worker's clock would put `started_at` and every lease an hour late."""

    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) + timedelta(hours=1)

    @classmethod
    def utcnow(cls):
        return datetime.now(tz=None) + timedelta(hours=1)


def test_a_lease_is_read_by_postgres_clock(store, store_dsn, monkeypatch):
    """`started_at` and the lease are the database's time, not the worker's: with the store's Python clock an hour
    ahead, they still agree with `now()` in Postgres, at the claim and at a renewal."""
    monkeypatch.setattr(build_store, "datetime", _AnHourAhead)
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=600)
    with psycopg.connect(store_dsn, autocommit=True) as con:
        def seconds(sql_text: str, *args) -> float:
            [(value,)] = con.execute(sql_text, args).fetchall()
            return float(value)

        assert 0 <= seconds("SELECT extract(epoch FROM now() - %s::timestamptz)", claimed.started_at) < 60
        assert seconds("SELECT extract(epoch FROM %s::timestamptz - %s::timestamptz)", claimed.lease_until,
                       claimed.started_at) == 600
        assert store.renew_lease(TENANT, build.build_id, "dashboard", claimed.lease_id, lease_s=300)
        renewed = _raw_step(store_dsn, build.build_id, "dashboard")["lease_until"]
        assert 240 < seconds("SELECT extract(epoch FROM %s::timestamptz - now())", renewed) <= 300


def test_a_store_reconnects_after_its_connection_drops(store_dsn):
    """The call that meets a dropped connection raises, and the next connects afresh. Only this store's own
    connection is ended, found by an application name of its own."""
    name = f"build-store-test-{uuid.uuid4().hex[:12]}"
    with BuildStore(make_conninfo(store_dsn, application_name=name)) as store:
        build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
        with psycopg.connect(store_dsn, autocommit=True) as con:
            [(ended,)] = con.execute("SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                                     "WHERE application_name = %s", (name,)).fetchall()
        assert ended == 1
        with pytest.raises(psycopg.OperationalError):
            store.load_build(TENANT, build.build_id, ANA)
        assert store.load_build(TENANT, build.build_id, ANA) == build


def test_the_json_the_store_writes_is_jsonb_postgres_reads(store, store_dsn):
    """A plan's own types survive the round trip: a float stays a float, a list a list, a null a null."""
    plan = FULL_PLAN
    build = store.create_build(TENANT, ANA, GOAL, plan, STEPS)
    assert store.load_build(TENANT, build.build_id, ANA).plan == plan
    with psycopg.connect(store_dsn, autocommit=True) as con:
        [(same,)] = con.execute(sql.SQL("SELECT plan = %s FROM {}").format(sql.Identifier(BUILD_TABLE)),
                                (Jsonb(plan),)).fetchall()
    assert same is True


def test_a_store_is_safe_to_share_between_threads(store):
    """The server shares one store across its request threads; each call holds the connection alone. Each build makes
    a dashboard of its own, since a dashboard has one build building at a time."""
    builds = [store.create_build(TENANT, ANA, GOAL, PLAN, STEPS) for _ in range(4)]
    errors: list[BaseException] = []

    def work(build: StoredBuild, made: int) -> None:
        try:
            for key, kind in STEPS[:3]:
                _run_step(store, store.load_build(TENANT, build.build_id, ANA), key, kind, made)
        except BaseException as exc:  # noqa: BLE001 -- re-raised below, in the test's own thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(build, 90 + n)) for n, build in enumerate(builds)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert [store.load_build(TENANT, build.build_id, ANA).step("premium_by_region").status for build in builds] == [
        "done"] * 4


def test_a_lease_that_has_passed_is_not_renewed(store, store_dsn):
    """Nobody has taken the step again, but its lease has passed: the worker's renewal fails and leaves the lease as it
    was, so the worker starts no further call and the step waits to be reconciled. The control is
    `test_a_renewed_lease_runs_from_the_renewal`, where the same renewal inside the lease succeeds."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=1)
    time.sleep(1.2)
    before = _raw_step(store_dsn, build.build_id, "dashboard")
    assert before["lease_id"] == claimed.lease_id  # still this worker's claim: nobody took the step
    assert store.renew_lease(TENANT, build.build_id, "dashboard", claimed.lease_id) is False
    after = _raw_step(store_dsn, build.build_id, "dashboard")
    assert after == before and after["lease_until"] == claimed.lease_until
    assert store.load_build(TENANT, build.build_id, ANA).step("dashboard").status == "uncertain"


def test_a_worker_past_its_lease_still_finishes_when_nobody_took_the_step(store):
    """A call that began inside the lease and returned after it: while no claim has taken the step, the worker records
    what it made, and the step needs no reconciling."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=1)
    time.sleep(1.2)
    assert store.load_build(TENANT, build.build_id, ANA).step("dashboard").status == "uncertain"
    finished = store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=claimed.lease_id)
    step = finished.step("dashboard")
    assert (step.status, step.object_id, step.lease_id, finished.dashboard_id) == ("done", 41, None, 41)


def test_finish_after_another_worker_took_the_step_is_lease_lost(store, store_dsn):
    """Once another claim has taken the step to reconcile it, the first worker's finish raises `LeaseLost` and writes
    nothing: the step stays the second claim's, which finishes it."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    first = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=1)
    time.sleep(1.2)
    second = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    assert isinstance(second, StepState) and second.status == "uncertain"
    before = _raw_build(store_dsn, build.build_id)
    with pytest.raises(LeaseLost):
        store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=first.lease_id)
    assert _raw_build(store_dsn, build.build_id) == before
    assert store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=second.lease_id).step(
        "dashboard").status == "done"


@pytest.mark.parametrize(("lock_ms", "statement_ms", "raised"), [
    (500, 30_000, psycopg.errors.LockNotAvailable),
    (30_000, 1_000, psycopg.errors.QueryCanceled),
], ids=["the lock timeout", "the statement timeout"])
def test_a_held_row_makes_a_claim_fail_promptly_not_hang(store_dsn, monkeypatch, lock_ms, statement_ms, raised):
    """A store's session waits a bounded time for a lock and for a statement, so one blocked statement cannot stall
    every build behind the store's one connection. With another connection holding the build's row, the claim raises
    within its timeout instead of waiting for the row, writes nothing, and once the row is let go the next claim takes
    the step. Each timeout is shown alone, the other set long."""
    monkeypatch.setattr(build_store, "LOCK_TIMEOUT_MS", lock_ms)
    monkeypatch.setattr(build_store, "STATEMENT_TIMEOUT_MS", statement_ms)
    with BuildStore(store_dsn) as store:
        build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
        with psycopg.connect(store_dsn) as holder, ThreadPoolExecutor(max_workers=1) as pool:
            holder.execute(sql.SQL("SELECT 1 FROM {} WHERE build_id = %s FOR UPDATE").format(
                sql.Identifier(BUILD_TABLE)), (build.build_id,))
            started = time.monotonic()
            claiming = pool.submit(store.claim_next_step, TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
            try:
                error = claiming.exception(timeout=10)
            except TimeoutError:
                holder.rollback()  # let the claim through, so the test ends, then fail it
                claiming.exception(timeout=30)
                pytest.fail("the claim waited on the held row for 10 s instead of failing within its timeout")
            elapsed = time.monotonic() - started
            holder.rollback()
        assert isinstance(error, raised), repr(error)
        assert elapsed < 5, elapsed
        assert _raw_step(store_dsn, build.build_id, "dashboard")["lease_id"] is None
        assert isinstance(store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard")),
                          StepState)


# --- A builder's unfinished builds: what a client offers to finish.

def test_unfinished_builds_are_the_callers_own(store):
    """A builder's unfinished builds are theirs, of their tenant, newest touched first: another principal's and
    another tenant's are not listed. Of three, the middle one touched last comes first, then the newest made: an
    order neither by when they were made nor oldest touched first."""
    a = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    b = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=13)
    c = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=16)
    store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=14)
    store.create_build(OTHER_TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=15)
    claimed = store.claim_next_step(TENANT, b.build_id, ANA, _intent("dashboard", "dashboard"))
    store.finish_step(TENANT, b.build_id, "dashboard", "done", 13, lease_id=claimed.lease_id)
    listed = store.unfinished_builds(TENANT, ANA)
    assert [build.build_id for build in listed] == [b.build_id, c.build_id, a.build_id]


def test_done_and_failed_builds_are_not_listed(store):
    done = _run_build(store)
    assert done.status == "done"
    failed = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    claimed = store.claim_next_step(TENANT, failed.build_id, ANA, _intent("dashboard", "dashboard"))
    store.finish_step(TENANT, failed.build_id, "dashboard", "failed", None, lease_id=claimed.lease_id)
    building = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=13)
    assert [build.build_id for build in store.unfinished_builds(TENANT, ANA)] == [building.build_id]


def test_unfinished_builds_filter_by_dashboard(store):
    """Asked for a dashboard, only builds of that dashboard are listed: one with no dashboard yet is listed only
    when none is asked for."""
    on_twelve = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    fresh = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    assert [b.build_id for b in store.unfinished_builds(TENANT, ANA, dashboard_id=12)] == [on_twelve.build_id]
    assert {b.build_id for b in store.unfinished_builds(TENANT, ANA)} == {on_twelve.build_id, fresh.build_id}
    assert store.unfinished_builds(TENANT, ANA, dashboard_id=99) == []
    assert len(store.unfinished_builds(TENANT, ANA, limit=1)) == 1


@pytest.mark.parametrize("call", [
    pytest.param(lambda store: store.unfinished_builds(TENANT, ANA, limit=0), id="limit 0"),
    pytest.param(lambda store: store.unfinished_builds(TENANT, ANA, limit=51), id="limit over the most"),
    pytest.param(lambda store: store.unfinished_builds(TENANT, ANA, limit=True), id="limit a bool"),
    pytest.param(lambda store: store.unfinished_builds("", ANA), id="no tenant"),
    pytest.param(lambda store: store.unfinished_builds(TENANT, ""), id="no principal"),
    pytest.param(lambda store: store.unfinished_builds(TENANT, ANA, dashboard_id=0), id="dashboard id 0"),
])
def test_unfinished_builds_refuse_a_bad_request_before_connecting(call):
    with pytest.raises(ValueError):
        call(_offline_store())


# --- One live build per dashboard: a dashboard's measures are replaced by whichever build writes them last.

def test_a_dashboard_with_a_build_building_is_said_to_be_building(store):
    """What an action on a dashboard's measures asks first: whether a build of the tenant's is building on it. Another
    tenant's and another dashboard's build do not count, nor one that ended."""
    assert store.dashboard_building(TENANT, 12) is False
    first = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    assert store.dashboard_building(TENANT, 12) is True
    assert store.dashboard_building(OTHER_TENANT, 12) is False and store.dashboard_building(TENANT, 13) is False
    claimed = store.claim_next_step(TENANT, first.build_id, ANA, _intent("dashboard", "dashboard"))
    store.finish_step(TENANT, first.build_id, "dashboard", "failed", None, lease_id=claimed.lease_id)
    assert store.dashboard_building(TENANT, 12) is False


def test_a_second_build_of_a_dashboard_is_refused_while_one_is_building(store):
    """While a build of a dashboard is building, another of the same tenant and dashboard is refused; another
    tenant's, another dashboard's, and one once the first has ended, are not. A new dashboard's build holds its
    dashboard from when its dashboard step names it."""
    first = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    with pytest.raises(DashboardBusy) as busy:
        store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=12)
    assert first.build_id not in str(busy.value)
    store.create_build(OTHER_TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=13)
    fresh = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    _run_step(store, fresh, "dashboard", "dashboard", 14)
    with pytest.raises(DashboardBusy):
        store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=14)
    claimed = store.claim_next_step(TENANT, first.build_id, ANA, _intent("dashboard", "dashboard"))
    store.finish_step(TENANT, first.build_id, "dashboard", "failed", None, lease_id=claimed.lease_id)
    again = store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=12)
    assert again.status == "building"


def test_two_builds_of_a_dashboard_started_at_once_leave_one(stores):
    """Two stores create a build of one dashboard at once, from two threads: exactly one is made, the other refused
    by Postgres itself (a unique index over the dashboards of builds still building), not by a read before the
    write."""
    barrier = threading.Barrier(2)
    made, refused = [], []

    def start(each: BuildStore) -> None:
        barrier.wait()
        try:
            made.append(each.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12))
        except DashboardBusy:
            refused.append(True)

    pair = [stores(), stores()]
    for each in pair:
        each.load_build(TENANT, str(uuid.uuid4()), ANA)  # connected, tables made
    threads = [threading.Thread(target=start, args=(each,)) for each in pair]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert (len(made), len(refused)) == (1, 1)


def test_the_live_dashboard_rule_is_postgres_own(store, store_dsn):
    """The rule is a partial unique index Postgres holds (read from its catalogue): its columns are the tenant and
    the dashboard, over builds still building with a dashboard."""
    store.load_build(TENANT, str(uuid.uuid4()), ANA)
    with psycopg.connect(store_dsn, autocommit=True) as con:
        [(definition,)] = con.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = %s "
            "AND indexdef LIKE 'CREATE UNIQUE INDEX%%' AND indexname <> %s",
            (BUILD_TABLE, f"{BUILD_TABLE}_pkey")).fetchall()
    assert "(tenant_id, dashboard_id)" in definition
    assert "status = 'building'" in definition and "dashboard_id IS NOT NULL" in definition


# --- An intent kept by an older build, a step's reason, a reused saved metric's row, and a build left untouched.

def test_a_stored_intent_with_an_id_floor_still_reads(store, store_dsn):
    """A step claimed by an earlier mnemiq holds an `id_floor` in its stored intent, the highest id it read before its
    create. It still reads, without it: from the build, from the reconciling claim that keeps the first claim's
    intent, and once finished."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=1)
    assert claimed.intent == _intent("dashboard", "dashboard")
    with psycopg.connect(store_dsn, autocommit=True) as con:
        con.execute(sql.SQL("UPDATE {} SET steps = jsonb_set(steps, ARRAY['0', 'intent', 'id_floor'], '41') "
                            "WHERE build_id = %s").format(sql.Identifier(BUILD_TABLE)), (build.build_id,))
    assert _raw_build(store_dsn, build.build_id)["steps"][0]["intent"]["id_floor"] == 41  # planted past the store
    planned = _intent("dashboard", "dashboard")
    assert store.load_build(TENANT, build.build_id, ANA).step("dashboard").intent == planned
    time.sleep(1.2)
    again = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    assert again.status == "uncertain" and again.intent == planned
    done = store.finish_step(TENANT, build.build_id, "dashboard", "done", 41, lease_id=again.lease_id)
    assert done.step("dashboard").intent == planned and done.step("dashboard").status == "done"


def test_a_step_keeps_its_reason(store):
    """A failed step's reason code is kept and read back; an uncertain step's too, through its reconciling claim; a
    done step has none."""
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    store.finish_step(TENANT, build.build_id, "dashboard", "uncertain", None, lease_id=claimed.lease_id,
                      reason="tool_error")
    again = store.claim_next_step(TENANT, build.build_id, ANA, _intent("dashboard", "dashboard"))
    assert again.status == "uncertain" and again.reason == "tool_error"
    stopped = store.finish_step(TENANT, build.build_id, "dashboard", "failed", None, lease_id=again.lease_id,
                                reason="not_editor")
    assert stopped.step("dashboard").reason == "not_editor" and stopped.status == "failed"
    assert store.load_build(TENANT, build.build_id, ANA).step("dashboard").reason == "not_editor"
    done = _run_step(store, store.create_build(TENANT, ANA, GOAL, PLAN, STEPS), "dashboard", "dashboard", 41)
    assert done.step("dashboard").reason is None


def test_a_reused_saved_metric_is_kept_as_a_measure_naming_it(store, store_dsn):
    """A reused saved metric keeps a measure row, its definition the seven keys and the metric's name."""
    saved = dataclasses.replace(MEASURE, name="total_premium@42", label="Total premium")
    kept = store.put_measure(TENANT, 7, saved, chart_ids=(43,), saved_metric="total_premium",
                             check_status={43: ChartStatus.CHECKED})
    assert kept.saved_metric == "total_premium" and kept.measure == saved
    [read] = store.measures_for(TENANT, 7)
    assert read == kept
    with psycopg.connect(store_dsn, autocommit=True) as con:
        [(definition,)] = con.execute(sql.SQL("SELECT definition FROM {}").format(sql.Identifier(MEASURE_TABLE)))
    assert set(definition) == {"expression", "label", "kind", "currency", "description", "asked_by", "dataset_id",
                               "saved_metric"}
    written = store.put_measure(TENANT, 7, MEASURE, chart_ids=(44,))
    assert written.saved_metric is None


def _untouched_for(store_dsn: str, build_id: str, seconds: int) -> None:
    """The build's row as if nothing had written it for `seconds`, planted past the store."""
    with psycopg.connect(store_dsn, autocommit=True) as con:
        con.execute(sql.SQL("UPDATE {} SET updated_at = now() - make_interval(secs => %s) WHERE build_id = %s")
                    .format(sql.Identifier(BUILD_TABLE)), (seconds, build_id))


def test_a_build_left_untouched_releases_its_dashboard(store, store_dsn):
    """A build of a dashboard untouched for more than twice the lease, its claims all passed, no longer holds the
    dashboard: the next build's creation fails its next step as abandoned in the same transaction. One touched more
    recently, or one whose claim still holds, keeps the dashboard."""
    old = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    _run_step(store, old, "dashboard", "dashboard", 12)
    _untouched_for(store_dsn, old.build_id, 2 * build_store.DEFAULT_LEASE_S - 30)
    with pytest.raises(DashboardBusy):
        store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=12)
    _untouched_for(store_dsn, old.build_id, 2 * build_store.DEFAULT_LEASE_S + 30)
    fresh = store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=12)
    stale = store.load_build(TENANT, old.build_id, ANA)
    assert stale.status == "failed" and stale.step("policies").status == "failed"
    assert stale.step("policies").reason == "abandoned" and stale.step("dashboard").status == "done"
    assert fresh.status == "building"
    held = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=13)
    store.claim_next_step(TENANT, held.build_id, ANA, _intent("dashboard", "dashboard"), lease_s=86_400)
    _untouched_for(store_dsn, held.build_id, 2 * build_store.DEFAULT_LEASE_S + 30)
    with pytest.raises(DashboardBusy):
        store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=13)


def test_a_new_dashboard_step_takes_its_dashboard_from_a_build_left_untouched(store, store_dsn):
    """A build can hold a dashboard id no dashboard has yet; a new dashboard that Superset then gives that id is
    another build's to finish. While the holder was touched within twice the lease, the finish is DashboardBusy and
    writes nothing; once it has been left untouched for longer, with no claim of it holding, the finish fails the
    holder's next step as abandoned, as starting a build of that dashboard does, and finishes."""
    holder = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    fresh = store.create_build(TENANT, BO, GOAL, PLAN, STEPS)
    claimed = store.claim_next_step(TENANT, fresh.build_id, BO, _intent("dashboard", "dashboard"))
    _untouched_for(store_dsn, holder.build_id, 2 * build_store.DEFAULT_LEASE_S - 30)
    with pytest.raises(DashboardBusy):
        store.finish_step(TENANT, fresh.build_id, "dashboard", "done", 12, lease_id=claimed.lease_id)
    assert store.load_build(TENANT, holder.build_id, ANA).status == "building"
    assert store.load_build(TENANT, fresh.build_id, BO).step("dashboard").lease_id == claimed.lease_id
    _untouched_for(store_dsn, holder.build_id, 2 * build_store.DEFAULT_LEASE_S + 30)
    done = store.finish_step(TENANT, fresh.build_id, "dashboard", "done", 12, lease_id=claimed.lease_id)
    assert done.dashboard_id == 12 and done.step("dashboard").status == "done"
    stale = store.load_build(TENANT, holder.build_id, ANA)
    assert stale.status == "failed" and stale.step("dashboard").reason == "abandoned"


# --- A build that waits as a proposal until the person confirms it

PLANNED_AGAINST = "2026-11-21T10:00:00.000000+00:00"


def test_a_proposal_holds_no_dashboard_and_runs_nothing(store):
    """A proposed build is kept with the dashboard's `changed_on` it was planned against, is no unfinished build, does
    not hold its dashboard's one live build, and has no step to claim until it is confirmed."""
    proposal = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12, proposed=True,
                                  planned_against=PLANNED_AGAINST)
    assert (proposal.status, proposal.planned_against, proposal.dashboard_id) == ("proposed", PLANNED_AGAINST, 12)
    assert store.load_build(TENANT, proposal.build_id, ANA) == proposal
    assert store.unfinished_builds(TENANT, ANA) == []
    assert store.dashboard_building(TENANT, 12) is False
    other = store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=12)
    assert other.status == "building"
    with pytest.raises(BuildNotFound):
        store.claim_next_step(TENANT, proposal.build_id, ANA, _intent("dashboard", "dashboard"))
    assert store.load_build(TENANT, proposal.build_id, ANA).steps == proposal.steps


def test_a_build_started_at_once_has_nothing_planned_against(store):
    build = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12)
    assert (build.status, build.planned_against) == ("building", None)


def test_confirm_starts_a_proposal_once(store):
    proposal = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12, proposed=True,
                                  planned_against=PLANNED_AGAINST)
    with pytest.raises(BuildNotFound):
        store.confirm_build(TENANT, proposal.build_id, BO)
    with pytest.raises(BuildNotFound):
        store.confirm_build(OTHER_TENANT, proposal.build_id, ANA)
    confirmed = store.confirm_build(TENANT, proposal.build_id, ANA)
    assert (confirmed.status, confirmed.planned_against) == ("building", PLANNED_AGAINST)
    assert store.unfinished_builds(TENANT, ANA) == [confirmed]
    assert store.dashboard_building(TENANT, 12) is True
    with pytest.raises(BuildNotFound):
        store.confirm_build(TENANT, proposal.build_id, ANA)
    claimed = store.claim_next_step(TENANT, proposal.build_id, ANA, _intent("dashboard", "dashboard"))
    assert claimed.key == "dashboard"


def test_confirm_on_a_busy_dashboard_is_refused_and_the_proposal_stays(store):
    proposal = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12, proposed=True)
    store.create_build(TENANT, BO, GOAL, PLAN, STEPS, dashboard_id=12)
    with pytest.raises(DashboardBusy):
        store.confirm_build(TENANT, proposal.build_id, ANA)
    assert store.load_build(TENANT, proposal.build_id, ANA).status == "proposed"


def test_two_confirms_at_once_confirm_once(stores):
    pair = [stores(), stores()]
    proposal = pair[0].create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12, proposed=True)
    pair[1].load_build(TENANT, proposal.build_id, ANA)  # connected
    barrier = threading.Barrier(2)
    confirmed, refused = [], []

    def confirm(each: BuildStore) -> None:
        barrier.wait()
        try:
            confirmed.append(each.confirm_build(TENANT, proposal.build_id, ANA))
        except BuildNotFound:
            refused.append(True)

    threads = [threading.Thread(target=confirm, args=(each,)) for each in pair]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert (len(confirmed), len(refused)) == (1, 1)


def test_withdraw_removes_only_a_proposal(store):
    proposal = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12, proposed=True)
    building = store.create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=13)
    assert store.withdraw_proposal(TENANT, building.build_id, ANA) is False
    assert store.withdraw_proposal(TENANT, proposal.build_id, BO) is False
    assert store.withdraw_proposal(TENANT, str(uuid.uuid4()), ANA) is False
    assert store.withdraw_proposal(TENANT, "not-an-id", ANA) is False
    assert store.withdraw_proposal(TENANT, proposal.build_id, ANA) is True
    assert store.load_build(TENANT, proposal.build_id, ANA) is None
    assert store.load_build(TENANT, building.build_id, ANA) == building


def test_an_existing_build_table_gains_planned_against(store_dsn, stores):
    """A deployment whose build table was made before proposals: the next store to connect adds the column, and the
    builds already kept read as started at once."""
    first = stores()
    kept = first.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    first.close()
    with psycopg.connect(store_dsn, autocommit=True) as con:
        con.execute(sql.SQL("ALTER TABLE {} DROP COLUMN planned_against").format(sql.Identifier(BUILD_TABLE)))
    later = stores()
    assert later.load_build(TENANT, kept.build_id, ANA).planned_against is None
    proposal = later.create_build(TENANT, ANA, GOAL, PLAN, STEPS, proposed=True, planned_against=PLANNED_AGAINST)
    assert later.load_build(TENANT, proposal.build_id, ANA).planned_against == PLANNED_AGAINST


@pytest.mark.parametrize("planned_against", ["", "   ", 7])
def test_a_proposal_planned_against_nothing_readable_is_refused(planned_against):
    with pytest.raises(ValueError):
        _offline_store().create_build(TENANT, ANA, GOAL, PLAN, STEPS, dashboard_id=12, proposed=True,
                                      planned_against=planned_against)


def test_a_store_connecting_while_a_transaction_reads_the_builds_does_not_wait(store_dsn, stores):
    """A store connecting (a server starting, or a broken connection made again) while another transaction holds the
    build table open, as a long report or a dump does: its tables are found as they are, and it answers at once,
    rather than waiting on a lock its own setup asks for."""
    first = stores()
    kept = first.create_build(TENANT, ANA, GOAL, PLAN, STEPS)
    with psycopg.connect(store_dsn) as reader:  # a transaction left open, reading the builds
        reader.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(BUILD_TABLE))).fetchone()
        started = time.monotonic()
        assert stores().load_build(TENANT, kept.build_id, ANA) == kept
        assert time.monotonic() - started < 5
        reader.rollback()

"""mnemiq's Superset extension, live: its token route on Superset's web server, and its write and read tools on Superset's MCP
service, in the suite's Superset 7.0.0rc2 with the extension loaded (`setup_superset.py extension`).

Every tool is called over MCP as a person whose token Superset verifies, with the scopes a build token carries, and
every chart and dataset a build would make is made as mnemiq makes it, carrying a build's marker: by the extension's
`create_chart`, from mnemiq's own `config_chart_request` (`extension_chart_request` for a funnel or a sankey), and by
its `create_dataset`. MCP's own `generate_chart` and `create_virtual_dataset` are called only where a test compares
with what MCP saves, stands for an object a person made by hand, or needs a dataset with saved metrics of its own,
which the extension's create does not take. Each test makes the dashboards, datasets and charts it needs on the seeded
`policy` table, so none depends on another; the tests of a create carrying a marker remove what they made (`made`).
What a tool wrote is read back from Superset's metadata database directly (`metadata`), and every number a chart
answers with is held to the source read directly (`ground_truth`).

Each test states its expectation in its docstring. Observations a run should keep are printed as `OBSERVED` lines,
which `pytest -rA` shows for passing tests too.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import jwt
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.chart_check import ChartStatus, check_chart
from mnemiq.superset.charts import (
    ChartSpec,
    Family,
    Filter,
    WrittenRef,
    config_chart_request,
    extension_chart_request,
    generate_chart_request,
    owed_format_params,
)
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetPermissionDenied
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.marker import Marker, format_marker, parse_marker
from mnemiq.superset.measures import Measure, MeasureKind
from tests.superset_extension_path import use_extension_source
from tests.superset_live.live import (
    TENANT,
    TOKEN_ROUTE,
    carrying,
    fingerprint_of,
    ground_truth,
    metadata,
    required,
    run_setup,
)

use_extension_source()

from agenticfabriq.mnemiq.token import ASK_SCOPES, BUILD_SCOPES  # noqa: E402
from agenticfabriq.mnemiq.writes import LOCK_TIMEOUT_S  # noqa: E402

pytestmark = pytest.mark.integration

BUILD = " ".join(BUILD_SCOPES)
ASK = " ".join(ASK_SCOPES)
WRITE_TOOLS = ("create_dashboard", "create_dataset", "create_chart", "patch_chart", "add_saved_metric",
               "delete_dataset", "set_layout")
READ_TOOLS = ("dataset_metrics", "editors_of", "get_layout", "row_rules", "table_rules", "tables_read",
              "session_binding")
# The builder in most tests: Alpha and sql_lab, narrowed by no row rule.
BUILDER = "analyst"
MONTHLY = ("SELECT date_trunc('month', issued)::date AS month, count(*) FROM policy GROUP BY 1 ORDER BY 1")
BY_REGION_AND_STATUS = "SELECT region, status, count(*) FROM policy GROUP BY 1, 2"
BY_STATUS = "SELECT status, count(*) FROM policy GROUP BY 1"
MONTHLY_BY_STATUS = "SELECT date_trunc('month', issued)::date, status, count(*) FROM policy GROUP BY 1, 2"
POLICY_COLUMNS = "SELECT id, region, status, premium, issued FROM policy"


def observe(test: str, **facts) -> None:
    print(f"OBSERVED {test} {json.dumps(facts, default=str, sort_keys=True)}")


def as_user(live, username: str, scope: str = BUILD) -> SupersetMcp:
    """An MCP client carrying a token the test identity provider signed for `username` with `scope`."""
    return SupersetMcp(live.mcp_url, DelegatedToken(live.token(username, scope=scope)), timeout_s=90)


def call(client: SupersetMcp, base: str, **request) -> dict:
    return client.call(TOOL_PREFIX + base, request)


def listed_tools(live, token: str) -> set[str]:
    """The tool names Superset's `tools/list` shows the bearer of `token`."""
    import anyio
    from mcp import ClientSession
    from mcp.shared._httpx_utils import create_mcp_http_client

    from mnemiq.superset.client import _opener

    async def listing() -> set[str]:
        http_client = create_mcp_http_client(headers={"Authorization": f"Bearer {token}"})
        async with _opener()(live.mcp_url, http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {tool.name for tool in (await session.list_tools()).tools}

    return anyio.run(listing)


@pytest.fixture(scope="module")
def extension(live) -> None:
    """Superset serves the extension's write tools; else every test here fails saying how to put them there."""
    listed = listed_tools(live, live.token(BUILDER, scope=BUILD))
    missing = [base for base in (*WRITE_TOOLS, *READ_TOOLS) if TOOL_PREFIX + base not in listed]
    assert not missing, (f"Superset's MCP service does not serve {missing}: build the extension with "
                         "`setup_superset.py extension`, set MNEMIQ_TEST_SUPERSET_EXTENSION_DIR for Superset's "
                         "processes, run `superset init`, and restart them")


@pytest.fixture(scope="module")
def policy_id(live) -> int:
    rows = metadata("SELECT id FROM tables WHERE table_name = 'policy' AND schema = 'public' AND deleted_at IS NULL")
    assert len(rows) == 1, rows
    return rows[0][0]


# --- what a build makes --------------------------------------------------------------------------------------------


@dataclass
class Build:
    """One build: its client (the builder's), its id and its dashboard."""

    client: SupersetMcp
    build_id: str
    dashboard_id: int

    def marker(self, step: str) -> str:
        return format_marker(self.build_id, step, self.dashboard_id)


def new_build(client: SupersetMcp, title: str) -> Build:
    build_id = str(uuid.uuid4())
    answer = call(client, "create_dashboard", title=f"{title} {build_id[:8]}",
                  marker=format_marker(build_id, "dashboard", None))
    assert set(answer) == {"dashboard_id"}, answer
    return Build(client, build_id, answer["dashboard_id"])


def policies(dataset_id: int) -> Measure:
    return Measure(name="policies", label="policies", expression="COUNT(*)", kind=MeasureKind.COUNT,
                   dataset_id=dataset_id, description="how many policies", asked_by=BUILDER)


def spec(family: Family, dataset_id: int, **over) -> ChartSpec:
    fields = {"key": "chart", "family": family, "title": f"{family.name.lower()} {uuid.uuid4().hex[:8]}",
              "dataset_id": dataset_id, "measures": (WrittenRef(policies(dataset_id)),), "dimensions": (),
              "time_column": None, "time_grain": None, "filters": (), "row_limit": None, "source": None,
              "target": None, **over}
    return ChartSpec(**fields)


def mcp_chart(client: SupersetMcp, chart: ChartSpec) -> int:
    """A chart MCP's `generate_chart` saves from mnemiq's own `generate_chart_request`: what MCP itself saves for the
    chart config a build sends the extension, for a test to compare with. A build no longer makes a chart this way."""
    answer = client.call("generate_chart", generate_chart_request(chart, {}, {}, currency="USD"))
    assert answer.get("success") and (answer.get("chart") or {}).get("id"), {k: answer.get(k) for k in
                                                                           ("success", "error")}
    return answer["chart"]["id"]


def extension_chart(build: Build, chart: ChartSpec, step: str = "chart") -> dict:
    """The extension's `create_chart` answer for mnemiq's own request for a funnel or a sankey."""
    request = extension_chart_request(chart, {}, {}, currency="USD")
    return call(build.client, "create_chart", **request, dashboard_id=build.dashboard_id, marker=build.marker(step))


def config_request(build: Build, chart: ChartSpec, marker: str) -> dict:
    """The extension's `create_chart` request a build sends for `chart`, of one of MCP's chart types: mnemiq's own
    `config_chart_request`, with the build's dashboard and `marker`."""
    return {**config_chart_request(chart, {}, {}, currency="USD"), "dashboard_id": build.dashboard_id,
            "marker": marker}


def config_chart(build: Build, chart: ChartSpec, step: str = "chart") -> dict:
    """The extension's `create_chart` answer for `chart` made as a build makes it (`config_request`), for `step`."""
    return call(build.client, "create_chart", **config_request(build, chart, build.marker(step)))


def made_chart(build: Build, chart: ChartSpec, step: str = "chart") -> int:
    """The id of `chart`, made by the extension as a build makes it."""
    answer = extension_chart(build, chart, step) if chart.family in (Family.FUNNEL, Family.SANKEY) else config_chart(
        build, chart, step)
    assert set(answer) == {"chart_id"}, answer
    return answer["chart_id"]


@dataclass
class Made:
    """What a test made, removed when it ends as analyst removes what it made: each chart through MCP's
    `delete_chart`, each dataset through the extension's `delete_dataset` (into Superset's trash), each dashboard
    through MCP's `delete_dashboard`, then each table the test made in the source database dropped. An id is recorded
    the moment an answer names it, before anything about it is asserted."""

    live: object
    charts: list[int] = field(default_factory=list)
    datasets: list[tuple[int, int]] = field(default_factory=list)
    dashboards: list[int] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def build(self, client: SupersetMcp, title: str) -> Build:
        build = new_build(client, title)
        self.dashboards.append(build.dashboard_id)
        return build

    def chart(self, answer: dict) -> dict:
        found = answer.get("chart_id") if isinstance(answer, dict) else None
        if type(found) is int and found not in self.charts:
            self.charts.append(found)
        return answer

    def dataset(self, answer: dict, dashboard_id: int) -> dict:
        found = answer.get("dataset_id") if isinstance(answer, dict) else None
        if type(found) is int and (found, dashboard_id) not in self.datasets:
            self.datasets.append((found, dashboard_id))
        return answer

    def remove(self) -> list:
        """Remove everything recorded; what could not be removed, by kind and id."""
        import psycopg
        from psycopg import sql

        problems: list = []
        client = as_user(self.live, BUILDER)
        for kind, ids, remove in (
                ("chart", self.charts, lambda i: client.call("delete_chart", {"identifier": i})),
                ("dataset", self.datasets, lambda pair: call(client, "delete_dataset", dataset_id=pair[0],
                                                             dashboard_id=pair[1])),
                ("dashboard", self.dashboards, lambda i: client.call("delete_dashboard", {"identifier": i}))):
            for item in ids:
                try:
                    answer = remove(item)
                except Exception as exc:  # noqa: BLE001 (reported, so the rest is still removed)
                    problems.append((kind, item, type(exc).__name__))
                    continue
                if answer.get("success") is False or (kind == "dataset" and answer != {"ok": True,
                                                                                         "soft_deleted": True}):
                    problems.append((kind, item, answer.get("error_type") or answer))
        for name in self.tables:
            with psycopg.connect(self.live.pg_dsn, autocommit=True, connect_timeout=10) as con:
                con.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier("public", name)))
        return problems


@pytest.fixture
def made(live):
    """What the test makes, removed when it ends (`Made`); a removal that fails fails the test's teardown."""
    found = Made(live)
    yield found
    problems = found.remove()
    assert not problems, problems


def virtual_dataset(live, client: SupersetMcp, description: str | None, **more) -> int:
    """A virtual dataset over `policy` made with MCP's `create_virtual_dataset`, with `description` as it is given and
    any saved metrics of its own (`metrics`): a dataset a person or another tool made, its creator its one editor. A
    build makes its datasets with the extension's `create_dataset` (`marked_dataset`)."""
    request = {"database_id": live.database_id, "sql": POLICY_COLUMNS, "schema_name": "public",
               "dataset_name": f"mnemiq_live_{uuid.uuid4().hex[:10]}", **more}
    if description is not None:
        request["description"] = description
    answer = client.call("create_virtual_dataset", request)
    assert answer.get("id"), answer.get("error")
    return answer["id"]


def marked_dataset(live, build: Build, step: str) -> int:
    """A virtual dataset over `policy` made for the build's dashboard by the extension's `create_dataset`, as a build
    makes one: its prose and the step's marker as its description, and as editors the caller and the dashboard's
    users whose grants already reach it."""
    answer = call(build.client, "create_dataset", database_id=live.database_id, schema="public",
                  table_name=f"mnemiq_live_{uuid.uuid4().hex[:10]}", sql=POLICY_COLUMNS,
                  description=f"Made by mnemiq for dashboard {build.dashboard_id}.", dashboard_id=build.dashboard_id,
                  marker=build.marker(step))
    assert set(answer) == {"dataset_id"}, answer
    return answer["dataset_id"]


def changed_on(client: SupersetMcp, kind: str, object_id: int) -> str:
    """The `changed_on` MCP's own info tool answers for a chart, a dataset or a dashboard: what mnemiq reads and sends
    back as `expect_changed_on`. `get_dataset_info` answers it only when asked (`select_columns`)."""
    request: dict = {"identifier": object_id}
    if kind == "dataset":
        request["select_columns"] = ["id", "changed_on"]
    answer = client.call(f"get_{kind}_info", request)
    assert isinstance(answer.get("changed_on"), str), answer.get("changed_on")
    return answer["changed_on"]


def chart_data(client: SupersetMcp, chart_id: int) -> dict:
    return client.call("get_chart_data", {"identifier": chart_id})


def stored_chart(chart_id: int) -> dict:
    (row,) = metadata("SELECT params, query_context, description, viz_type, changed_on FROM slices WHERE id = %s",
                      (chart_id,))
    return {"params": json.loads(row[0] or "{}"), "query_context": json.loads(row[1]) if row[1] else None,
            "description": row[2], "viz_type": row[3], "changed_on": row[4]}


def editor_ids(table: str, column: str, object_id: int) -> list[int]:
    return [row[0] for row in metadata(f"SELECT subject_id FROM {table} WHERE {column} = %s ORDER BY 1",
                                       (object_id,))]


def label_of(username: str) -> str:
    (row,) = metadata("SELECT s.label FROM subjects s JOIN ab_user u ON u.id = s.user_id WHERE u.username = %s",
                      (username,))
    return row[0]


def members(dashboard_id: int) -> list[int]:
    return [row[0] for row in metadata("SELECT slice_id FROM dashboard_slices WHERE dashboard_id = %s ORDER BY 1",
                                       (dashboard_id,))]


def stored_metrics(dataset_id: int) -> dict[str, tuple]:
    rows = metadata("SELECT metric_name, id, verbose_name, metric_type, expression, description, d3format, "
                    "warning_text, extra, uuid::text, currency::text FROM sql_metrics WHERE table_id = %s",
                    (dataset_id,))
    return {row[0]: row[1:] for row in rows}


def positions(chart_ids: list[int]) -> dict:
    """A layout placing each chart on a row of its own, in the shape Superset's dashboard saves."""
    nodes: dict = {"DASHBOARD_VERSION_KEY": "v2",
                   "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
                   "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]}}
    for chart_id in chart_ids:
        row, chart = f"ROW-{chart_id}", f"CHART-{chart_id}"
        nodes["GRID_ID"]["children"].append(row)
        nodes[row] = {"type": "ROW", "id": row, "children": [chart], "parents": ["ROOT_ID", "GRID_ID"],
                      "meta": {"background": "BACKGROUND_TRANSPARENT"}}
        nodes[chart] = {"type": "CHART", "id": chart, "children": [], "parents": ["ROOT_ID", "GRID_ID", row],
                        "meta": {"chartId": chart_id, "width": 4, "height": 50}}
    return nodes


def ranked(rows: list[tuple]) -> list[tuple]:
    """Rows whose first value is a time, with that time replaced by its rank among the rows' times: the chart data
    and the source spell a month differently, and only their order is compared."""
    times = sorted({row[0] for row in rows})
    return sorted((times.index(row[0]), *row[1:]) for row in rows)


# --- racing two writers on one row -----------------------------------------------------------------------------------


def _hold(table: str, row_id: int):
    """A connection of the test's own to Superset's metadata database, holding `table`'s row `row_id` locked."""
    import psycopg

    con = psycopg.connect(required("MNEMIQ_TEST_SUPERSET_METADATA_URI")["MNEMIQ_TEST_SUPERSET_METADATA_URI"],
                          connect_timeout=10)
    con.execute(f"SELECT id FROM {table} WHERE id = %s FOR UPDATE", (row_id,))
    return con


def _waiting_on(table: str) -> int:
    """How many sessions wait for a row lock while reading `table`: a tool's locking read of the whole entity, which
    starts `SELECT <table>.`. A plain read never waits on a row lock in Postgres, and `pg_stat_activity` keeps only
    the first 1024 bytes of a query, which cuts the `FOR UPDATE` off a read of every column."""
    return metadata("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock' AND query ILIKE %s AND pid <> pg_backend_pid()",
                    (f"SELECT {table}.%",))[0][0]


def _hold_marker(marker: str):
    """A connection of the test's own to Superset's metadata database, inside a transaction that holds an advisory lock
    on the key a create carrying `marker` takes its lock on (`hashtextextended(marker, 0)`) until it is rolled back.
    Postgres's advisory locks of a transaction and of a session share their keys, so the create's own lock on that
    key (`pg_advisory_lock`) waits for this one."""
    import psycopg

    con = psycopg.connect(required("MNEMIQ_TEST_SUPERSET_METADATA_URI")["MNEMIQ_TEST_SUPERSET_METADATA_URI"],
                          connect_timeout=10)
    con.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (marker,))
    return con


def _waiting_on_advisory_locks() -> int:
    """How many other sessions of the metadata database wait for an advisory lock: a create waiting for its marker's."""
    return metadata("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock' AND wait_event = 'advisory' AND pid <> pg_backend_pid()")[0][0]


def _marker_lock_holders(marker: str) -> list[int]:
    """The pid of each metadata database session granted an advisory lock on the key a create carrying `marker` takes
    its lock on (`hashtextextended(marker, 0)`), from `pg_locks`, which lists a bigint key as its high 32 bits
    (`classid`) and its low 32 bits (`objid`), with `objsubid` 1."""
    return [row[0] for row in metadata(
        "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted AND objsubid = 1 "
        "AND classid = ((hashtextextended(%s, 0) >> 32) & 4294967295)::oid "
        "AND objid = (hashtextextended(%s, 0) & 4294967295)::oid ORDER BY pid", (marker, marker))]


def race(table: str, row_id: int, writers: list[Callable[[], dict]]) -> list[dict]:
    """Each writer's answer, the writers started together while the test holds the row's lock, and the lock let go
    only once every writer waits for it. So the writers run one after another, each after the lock it waited for,
    whatever the machine's timing: a writer that compared `changed_on` before taking the lock would compare the same
    stale value as the others and pass."""
    held = _hold(table, row_id)
    answers: list = [None] * len(writers)
    errors: list[BaseException] = []

    def run(index: int, writer: Callable[[], dict]) -> None:
        try:
            answers[index] = writer()
        except BaseException as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i, w)) for i, w in enumerate(writers)]
    try:
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 60
        while _waiting_on(table) < len(writers):
            assert time.monotonic() < deadline, f"the writers never all waited on the {table} row's lock"
            assert not errors, errors
            time.sleep(0.1)
    finally:
        held.rollback()
        held.close()
    for thread in threads:
        thread.join(90)
    assert not errors, errors
    return answers


# --- the token route ------------------------------------------------------------------------------------------------


def _claims(live, token: str) -> dict:
    return jwt.decode(token, (live.keys / "public.pem").read_bytes(), algorithms=["RS256"], audience=live.audience,
                      issuer=live.issuer)


def test_token_route_mints_for_the_signed_in_user_only(live, extension):
    """Signed in as west, with a CSRF token, a `build` request whose body also names east answers 200 with
    `{token, expires_in: 300}`; the token is west's alone (`sub` and `preferred_username` west), carries exactly the
    build scopes and the tenant `live`, lives 300 seconds, and verifies with the test identity provider's public key,
    issuer, audience and key id. An `ask` request gets exactly the ask scopes. A body with no valid purpose answers
    400 with no token. Gamma may mint too (`superset init` grants every role the route), and gets its
    own name."""
    with live.signed_in("west") as (client, headers):
        csrf = {**headers, "X-CSRFToken": live.csrf_token(client, headers)}
        answer = client.post(TOKEN_ROUTE, json={"purpose": "build", "username": "east", "sub": "east",
                                                "preferred_username": "east"}, headers=csrf)
        assert answer.status_code == 200, answer.status_code
        body = answer.json()
        assert set(body) == {"token", "expires_in"} and body["expires_in"] == 300
        claims = _claims(live, body["token"])
        assert claims["sub"] == claims["preferred_username"] == "west"
        assert set(claims["scope"].split()) == set(BUILD_SCOPES)
        assert claims["tenant"] == TENANT
        assert claims["exp"] - claims["iat"] == 300
        assert jwt.get_unverified_header(body["token"])["kid"] == live.kid
        ask = client.post(TOKEN_ROUTE, json={"purpose": "ask"}, headers=csrf)
        assert ask.status_code == 200
        assert set(_claims(live, ask.json()["token"])["scope"].split()) == set(ASK_SCOPES)
        for bad in ({}, {"purpose": "admin"}, {"purpose": ["build"]}):
            refused = client.post(TOKEN_ROUTE, json=bad, headers=csrf)
            assert refused.status_code == 400 and "token" not in refused.json(), refused.status_code
    gamma = _claims(live, live.minted("gamma_only", "build"))
    assert gamma["sub"] == "gamma_only"
    observe("token_route", west_scopes=sorted(claims["scope"].split()), lifetime=claims["exp"] - claims["iat"],
            gamma_minted=True)


def test_token_route_needs_csrf(live, extension):
    """A signed-in POST to the token route without `X-CSRFToken`, or with one that is not this session's, answers 400
    and no token (Flask-WTF's CSRF protection is global in Superset). Not signed in at all, it answers 400 too: the
    CSRF check runs before the route. The control: the same request with the session's CSRF token answers 200."""
    with live.signed_in("west") as (client, headers):
        csrf = live.csrf_token(client, headers)
        for sent in (headers, {**headers, "X-CSRFToken": "not-this-sessions-token"}):
            refused = client.post(TOKEN_ROUTE, json={"purpose": "build"}, headers=sent)
            assert refused.status_code == 400, refused.status_code
            assert "token" not in refused.json()
        assert client.post(TOKEN_ROUTE, json={"purpose": "build"},
                           headers={**headers, "X-CSRFToken": csrf}).status_code == 200
    import httpx

    with httpx.Client(base_url=live.url, timeout=30) as anonymous:
        refused = anonymous.post(TOKEN_ROUTE, json={"purpose": "build"})
    assert refused.status_code == 400 and "token" not in refused.json(), refused.status_code


REPO = Path(__file__).resolve().parents[2]
# A configuration of a throwaway Superset process's own: the suite's (`superset_config.py`), then `settings`.
_OVERLAY = """import os
import runpy

for _name, _value in runpy.run_path(os.environ["MNEMIQ_TEST_SUPERSET_BASE_CONFIG"]).items():
    if _name.isupper():
        globals()[_name] = _value
"""


@pytest.fixture(scope="module")
def built_extension(tmp_path_factory) -> Path:
    """The extension built from this checkout by its own packer (`scripts/build_superset_extension.py`), for the
    throwaway Superset processes a test starts: they load the code under test, whatever the running services were
    built from."""
    import importlib.util

    out = tmp_path_factory.mktemp("extension")
    spec = importlib.util.spec_from_file_location("build_superset_extension",
                                                  REPO / "scripts" / "build_superset_extension.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
    assert packer.main(["--out", str(out)]) == 0
    return out


@contextlib.contextmanager
def throwaway_superset(args: list[str], settings: str, extension_dir: Path, ready: Callable[[bytes], bool],
                       **env: str):
    """A Superset process of the test's own, `superset <args>`, on the suite's configuration followed by `settings`
    (Python, run after it), loading the extension in `extension_dir`, with `env` added to its environment; yielded
    once `ready(its output so far)`, and stopped when the block ends. It shares the metadata database and keys with
    the running services and changes nothing in them: Superset starts with `AppBuilder(update_perms=False)`
    (`superset/extensions/__init__.py`), so a start writes no permission."""
    python = required("MNEMIQ_TEST_SUPERSET_PYTHON", "MNEMIQ_TEST_SUPERSET_KEYS_DIR",
                      "MNEMIQ_TEST_SUPERSET_METADATA_URI")["MNEMIQ_TEST_SUPERSET_PYTHON"]
    with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryFile() as log:
        config = Path(folder) / "superset_config.py"
        config.write_text(_OVERLAY + settings)
        child_env = {**os.environ, "SUPERSET_CONFIG_PATH": str(config), "FLASK_APP": "superset",
                     "MNEMIQ_TEST_SUPERSET_BASE_CONFIG": str(Path(__file__).with_name("superset_config.py")),
                     "MNEMIQ_TEST_SUPERSET_EXTENSION_DIR": str(extension_dir), **env}
        process = subprocess.Popen([str(Path(python).with_name("superset")), *args], env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                log.seek(0)
                if ready(log.read()):
                    break
                assert process.poll() is None, "the throwaway Superset process exited"
                assert time.monotonic() < deadline, "the throwaway Superset process did not start"
                time.sleep(1)
            yield
        finally:
            process.terminate()
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()


GUEST_AUDIENCE = "mnemiq-live-guest"
# Embedded Superset on, as a deployment that embeds dashboards runs it: a guest token (`superset/security/
# guest_token.py`, `GuestUser`) makes the request's user a guest in `GUEST_ROLE_NAME`, here Gamma, which `superset init`
# grants the token route's permission (`can_read on mnemiq`).
_EMBEDDED = """
FEATURE_FLAGS = {**globals().get("FEATURE_FLAGS", {}), "EMBEDDED_SUPERSET": True}
GUEST_ROLE_NAME = "Gamma"
GUEST_TOKEN_JWT_SECRET = os.environ["MNEMIQ_TEST_GUEST_SECRET"]
GUEST_TOKEN_JWT_AUDIENCE = "%s"
""" % GUEST_AUDIENCE


def guest_token(secret: str, username: str) -> str:
    """A guest token as Superset's `/api/v1/security/guest_token/` signs one (`GuestToken`: `user`, `resources`,
    `rls_rules`, `type: guest`), for a guest the embedding host names `username`."""
    now = int(time.time())
    return jwt.encode({"user": {"username": username, "first_name": "Guest", "last_name": "User"}, "resources": [],
                       "rls_rules": [], "type": "guest", "iat": now, "exp": now + 300, "aud": GUEST_AUDIENCE},
                      secret, algorithm="HS256")


def test_token_route_refuses_an_embedded_guest(live, extension, built_extension):
    """In a web server of the test's own with embedded Superset on (the suite's configuration plus guests in the Gamma
    role; the running services are not touched), a POST to the token route carrying a guest token whose user is named
    analyst, with a CSRF token of its own session, answers 403 with the route's own message and no token: a guest's
    username is whatever the embedding host wrote, not a Superset sign-in. That the route's own message answers shows
    Superset took the guest as a Gamma user who holds the route's permission. The
    controls, on the same server: a guest token signed with another secret is no guest at all (401), and west, signed
    in, still gets a token for west."""
    import secrets as secrets_module

    import httpx

    secret = secrets_module.token_urlsafe(32)
    # A port no other process holds: on a fixed one, a Superset already listening there (another stack on the same
    # machine) answers the health check below, and the test would ask the wrong server.
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    def up(_output: bytes) -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    with throwaway_superset(["run", "-h", "127.0.0.1", "-p", str(port), "--with-threads"], _EMBEDDED,
                            built_extension, up, MNEMIQ_TEST_GUEST_SECRET=secret):
        with httpx.Client(base_url=url, timeout=30) as browser:
            login_page = browser.get("/login/").text
            csrf = login_page.split('name="csrf_token"', 1)[1].split('value="', 1)[1].split('"', 1)[0]
            asked = browser.post(TOKEN_ROUTE, json={"purpose": "build"},
                                 headers={"X-CSRFToken": csrf, "X-GuestToken": guest_token(secret, BUILDER)})
            forged = browser.post(TOKEN_ROUTE, json={"purpose": "build"},
                                  headers={"X-CSRFToken": csrf,
                                           "X-GuestToken": guest_token(secrets_module.token_urlsafe(32), BUILDER)})
        minted = asked.json().get("token") if asked.status_code == 200 else None
        observe("guest", status=asked.status_code, forged_status=forged.status_code,
                minted_for=_claims(live, minted)["sub"] if minted else None)
        assert asked.status_code == 403, (asked.status_code, "token" in asked.text)
        assert asked.json() == {"message": "the token route mints only for a signed-in Superset user"}
        assert forged.status_code == 401, forged.status_code
        assert _claims(live, replace(live, url=url).minted("west", "build"))["sub"] == "west"


def _free_port() -> int:
    """A port on 127.0.0.1 that was free a moment ago, as the operating system picks one."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _serve(settings, monkeypatch):
    """The app `mnemiq serve --http` serves for `settings`, uvicorn alone replaced (as `test_contract.serve`)."""
    from fastapi.testclient import TestClient

    import mnemiq.server.app as app_module

    built: dict = {}
    monkeypatch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
    app_module.serve_http(settings, host="127.0.0.1", port=8080)
    return TestClient(built["app"])


def test_minted_token_passes_mcp_and_mnemiqs_door(live, extension, model, enriched, recorder, monkeypatch):
    """A token the route mints passes both doors as it is. West's `ask` token, sent to mnemiq's `/v1/ask`, is verified
    by mnemiq's door and handed on to Superset unchanged: west is answered `[["west", 100]]` (west's row rule, the
    source's count), and every call that ask makes to Superset carries that very token. Analyst's `build` token, sent
    to Superset's MCP service, creates a dashboard through the extension, which Superset records as made by analyst."""
    settings, _ = enriched
    app = _serve(settings, monkeypatch)
    model.sql = "SELECT region, COUNT(*) AS n FROM policy GROUP BY region"
    truth = dict(ground_truth(live, "SELECT region, count(*) FROM policy GROUP BY region"))
    token = live.minted("west", "ask")
    mark = recorder.mark()
    response = app.post("/v1/ask", json={"question": "How many policies are there in each region?"},
                        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preview"]["rows"] == [["west", truth["west"]]], body
    calls = recorder.since(mark)
    assert calls and {c.credential for c in calls} == {f"DelegatedToken:{fingerprint_of(token)}"}
    build_token = live.minted(BUILDER, "build")
    build_id = str(uuid.uuid4())
    made = SupersetMcp(live.mcp_url, DelegatedToken(build_token)).call(
        TOOL_PREFIX + "create_dashboard", {"title": f"minted {build_id[:8]}",
                                           "marker": format_marker(build_id, "dashboard", None)})
    (row,) = metadata("SELECT u.username FROM dashboards d JOIN ab_user u ON u.id = d.created_by_fk WHERE d.id = %s",
                      (made["dashboard_id"],))
    assert row == (BUILDER,)
    observe("minted", ask_calls=len(calls), dashboard_id=made["dashboard_id"])


# --- the saved query context ----------------------------------------------------------------------------------------


def test_saved_query_context_applies_the_grain(live, extension, policy_id):
    """A monthly line chart (COUNT(*) by `issued`, grain P1M) made by the extension's `create_chart` as a build makes
    it is saved with a query context whose time axis carries the grain, the dashboard's editors and the build's marker
    as its description, and answers `get_chart_data` with one row per month: 12 rows, each month's count the source's.
    The control: the same chart saved by MCP's `generate_chart` has no saved query context, and rc2 rebuilds its query
    from `form_data` without the grain, so it does not answer the twelve months."""
    client = as_user(live, BUILDER)
    build = new_build(client, "grain")
    line = spec(Family.LINE, policy_id, time_column="issued", time_grain="P1M")
    truth = [count for _, count in ground_truth(live, MONTHLY)]
    assert len(truth) == 12
    control = mcp_chart(client, line)
    before = chart_data(client, control)
    before_counts = [row.get("policies") for row in sorted(before.get("data") or [], key=lambda r: r["issued"])]
    assert stored_chart(control)["query_context"] is None
    assert before_counts != truth
    chart_id = made_chart(build, line, "by_month")
    stored = stored_chart(chart_id)
    (axis,) = [c for c in stored["query_context"]["queries"][0]["columns"] if isinstance(c, dict)]
    assert axis["timeGrain"] == "P1M" and axis["columnType"] == "BASE_AXIS"
    assert stored["description"] == build.marker("by_month")
    assert editor_ids("chart_editors", "chart_id", chart_id) == editor_ids("dashboard_editors", "dashboard_id",
                                                                           build.dashboard_id)
    after = chart_data(client, chart_id)
    rows = sorted(after["data"], key=lambda r: r["issued"])
    assert [row["policies"] for row in rows] == truth
    observe("grain", before_rows=len(before.get("data") or []), before_error=before.get("error_type"),
            after_rows=len(rows), first_month=rows[0]["issued"])


def test_saved_query_context_keeps_pivot_and_waterfall_grouping(live, extension, policy_id):
    """A pivot (rows region, columns status, COUNT(*)) made by the extension's `create_chart` as a build makes it
    answers `get_chart_data` with one row per region and status, nine in all, each count the source's, and no totals
    row. A waterfall (x-axis `issued` by month, broken down by status) answers one row per month and status, each
    count the source's: its x-axis kept and its grain applied. The control: the same two saved by MCP's
    `generate_chart`, with no query context, answer fewer rows than the source's grouping has (rc2's rebuild from
    `form_data` reads neither the pivot's rows and columns nor the waterfall's x-axis); what each answered is
    observed."""
    client = as_user(live, BUILDER)
    build = new_build(client, "grouping")
    charts = {"pivot": spec(Family.PIVOT, policy_id, dimensions=("region", "status")),
              "waterfall": spec(Family.WATERFALL, policy_id, time_column="issued", time_grain="P1M",
                                dimensions=("status",))}
    pivot_truth = sorted(ground_truth(live, BY_REGION_AND_STATUS))
    waterfall_truth = ground_truth(live, MONTHLY_BY_STATUS)
    assert len(pivot_truth) == 9
    before = {name: chart_data(client, mcp_chart(client, chart)) for name, chart in charts.items()}
    assert len(before["pivot"].get("data") or []) != len(pivot_truth)
    assert len(before["waterfall"].get("data") or []) != len(waterfall_truth)
    pivot, waterfall = (made_chart(build, chart, name) for name, chart in charts.items())
    pivot_rows = sorted((r["region"], r["status"], r["policies"]) for r in chart_data(client, pivot)["data"])
    assert pivot_rows == pivot_truth
    waterfall_rows = [(r["issued"], r["status"], r["policies"]) for r in chart_data(client, waterfall)["data"]]
    assert ranked(waterfall_rows) == ranked(waterfall_truth)
    observe("grouping", pivot_before_rows=len(before["pivot"].get("data") or []),
            pivot_before_error=before["pivot"].get("error_type"),
            waterfall_before_rows=len(before["waterfall"].get("data") or []),
            waterfall_before_columns=sorted((before["waterfall"].get("data") or [{}])[0]),
            waterfall_after_rows=len(waterfall_rows))


# The seeded `policy` table (`seed.sql`) as the snapshot mnemiq's own statement is decided against, and the builder's
# grant of it: analyst reads `policy` with no row rule.
POLICY = Snapshot(version="live", source_id="live", created_at="seed", columns=[
    Column(id=f"policy.{name}", object_id="policy", name=name, data_type=data_type)
    for name, data_type in (("id", "INTEGER"), ("region", "VARCHAR"), ("status", "VARCHAR"), ("premium", "DECIMAL"),
                            ("issued", "DATE"))])
BY_REGION_AND_STATUS_PREMIUM = "SELECT region, status, sum(premium) FROM policy GROUP BY 1, 2"


def written(dataset_id: int, label: str, expression: str) -> WrittenRef:
    return WrittenRef(Measure(name=label.replace(" ", "_"), label=label, expression=expression,
                              kind=MeasureKind.CURRENCY, dataset_id=dataset_id, description=label, asked_by=BUILDER))


def test_charts_made_by_the_extension_read_checked_by_the_value_check(live, extension, policy_id, recorder):
    """A monthly line chart, a pivot (rows region, columns status) and a waterfall (`issued` by month, by status), each
    of the measure `SUM(premium)`, made by the extension's `create_chart` from mnemiq's own `config_chart_request`,
    read CHECKED through `check_chart`, with mnemiq's real client and adapter as the builder. The pivot
    keeps MCP's totals and the waterfall MCP's total bar, as mnemiq's request leaves them; both are sums of a SUM. The
    check compares 12, 9 and 35 groups, and judges each chart's format. The pivot's numbers are the source's, read
    directly. A pivot of the ratio `SUM(premium) / COUNT(*)` with its totals reads NOT_CHECKED, since its totals are
    not the sums of its cells, and nothing is asked for its data."""
    client = as_user(live, BUILDER)
    build = new_build(client, "checked")
    total = written(policy_id, "total premium", "SUM(premium)")
    charts = {
        "line": spec(Family.LINE, policy_id, measures=(total,), time_column="issued", time_grain="P1M"),
        "pivot": spec(Family.PIVOT, policy_id, measures=(total,), dimensions=("region", "status")),
        "waterfall": spec(Family.WATERFALL, policy_id, measures=(total,), time_column="issued", time_grain="P1M",
                          dimensions=("status",)),
        "ratio_pivot": spec(Family.PIVOT, policy_id, measures=(written(policy_id, "premium per policy",
                                                                       "SUM(premium) / COUNT(*)"),),
                            dimensions=("region", "status")),
    }
    ids = {name: made_chart(build, chart, name) for name, chart in charts.items()}
    stored = {name: stored_chart(chart_id)["params"] for name, chart_id in ids.items()}
    assert (stored["pivot"]["rowTotals"], stored["pivot"]["colTotals"], stored["waterfall"]["show_total"]) == (
        True, True, True)
    adapter = SupersetSqlAdapter(client, database_id=live.database_id, schema="public")
    kinds = {"total premium": MeasureKind.CURRENCY, "premium per policy": MeasureKind.CURRENCY}
    recorder.calls.clear()
    checks = {name: check_chart(adapter, chart_id=chart_id, dataset_id=policy_id, kinds=kinds, saved={},
                                snapshot=POLICY, grants=GrantSet(frozenset({"policy"})), guard_fanout=False)
              for name, chart_id in ids.items()}
    observe("value check", **{name: (ids[name], check.status.value, check.reason,
                                     len(check.chart_values or ()), (check.chart_values or (None,))[0])
                              for name, check in checks.items()})
    for name, groups in (("line", 12), ("pivot", 9), ("waterfall", 35)):
        check = checks[name]
        assert check.status is ChartStatus.CHECKED, (name, check.reason)
        assert len(check.chart_values) == groups and check.chart_values == check.mnemiq_values, name
    truth = sorted((region, status, float(total)) for region, status, total in ground_truth(
        live, BY_REGION_AND_STATUS_PREMIUM))
    assert sorted(checks["pivot"].chart_values) == truth
    ratio = checks["ratio_pivot"]
    assert ratio.status is ChartStatus.NOT_CHECKED and "totals" in ratio.reason, ratio.reason
    assert ratio.chart_values is None and ratio.mnemiq_values is None
    assert not [c for c in recorder.calls if c.tool == "get_chart_data"
                and c.request.get("identifier") == ids["ratio_pivot"]]


def test_a_time_axis_its_row_limit_cuts_reads_not_checked(live, extension, policy_id, recorder):
    """A monthly line of premium with a row limit of 3, made by the extension as a chart made by hand would be: rc2's
    query ranks by the measure before its LIMIT and the axis sorts by time afterwards, so it shows the 3 largest months,
    not the latest. The check reads it as not checked and says why. The control: the same line with no row limit is
    checked, all 12 months."""
    client = as_user(live, BUILDER)
    build = new_build(client, "time-axis-cut")
    total = written(policy_id, "total premium", "SUM(premium)")
    line = dict(measures=(total,), time_column="issued", time_grain="P1M")
    ids = {"cut": made_chart(build, spec(Family.LINE, policy_id, row_limit=3, **line), "cut"),
           "whole": made_chart(build, spec(Family.LINE, policy_id, **line), "whole")}
    adapter = SupersetSqlAdapter(client, database_id=live.database_id, schema="public")
    checks = {name: check_chart(adapter, chart_id=chart_id, dataset_id=policy_id,
                                kinds={"total premium": MeasureKind.CURRENCY}, saved={}, snapshot=POLICY,
                                grants=GrantSet(frozenset({"policy"})), guard_fanout=False)
              for name, chart_id in ids.items()}
    shown = client.call("get_chart_data", {"identifier": ids["cut"]}).get("data") or []
    largest = ground_truth(live, "SELECT SUM(premium) FROM policy GROUP BY date_trunc('month', issued) "
                                 "ORDER BY 1 DESC LIMIT 3")
    observe("time-axis cut", **{name: (ids[name], check.status.value, check.reason) for name, check in checks.items()},
            shown=len(shown))
    assert checks["whole"].status is ChartStatus.CHECKED and len(checks["whole"].chart_values) == 12
    assert len(shown) == 3
    assert sorted(float(next(v for k, v in row.items() if k != "issued")) for row in shown) == sorted(
        float(total) for (total,) in largest)  # rc2 kept the 3 largest months
    cut = checks["cut"]
    assert cut.status is ChartStatus.NOT_CHECKED, cut.reason
    assert cut.reason.endswith("its x-axis is a time axis, where a row limit keeps the points its query ranks first, "
                               "not the latest"), cut.reason


# --- patch_chart ----------------------------------------------------------------------------------------------------


def _made_line(live, client: SupersetMcp, policy_id: int, title: str) -> tuple[Build, int]:
    """A build's dashboard and a monthly line chart the extension made for it, as a build makes one."""
    build = new_build(client, title)
    return build, made_chart(build, spec(Family.LINE, policy_id, time_column="issued", time_grain="P1M"), "line")


def test_patch_with_stale_changed_on_is_refused(live, extension, policy_id):
    """A patch that sends a `changed_on` `get_chart_info` answered before another patch replaced it answers exactly
    `{stale: true}` and changes nothing: the chart's params, query context and `changed_on` are as the other patch left
    them. The same patch with the `changed_on` read after that answers `{ok: true}` and sets only `row_limit`, every
    other params key as it was, and the query context follows (`row_limit` 77)."""
    client = as_user(live, BUILDER)
    _, chart_id = _made_line(live, client, policy_id, "stale")
    old = changed_on(client, "chart", chart_id)
    assert call(client, "patch_chart", chart_id=chart_id, set={"color_scheme": "bnbColors"},
                expect_changed_on=old) == {"ok": True}
    other = stored_chart(chart_id)
    assert call(client, "patch_chart", chart_id=chart_id, set={"row_limit": 77}, expect_changed_on=old) == {
        "stale": True}
    assert stored_chart(chart_id) == other
    fresh = changed_on(client, "chart", chart_id)
    assert fresh != old
    assert call(client, "patch_chart", chart_id=chart_id, set={"row_limit": 77}, expect_changed_on=fresh) == {
        "ok": True}
    patched = stored_chart(chart_id)
    assert patched["params"] == {**other["params"], "row_limit": 77}
    assert patched["query_context"]["queries"][0]["row_limit"] == 77
    observe("stale", first_read=old, after_other_patch=fresh)


def test_simultaneous_patches_lose_nothing(live, extension, policy_id):
    """Two patches of one chart, each setting a different key and both sending the same `changed_on`, start together
    while the test holds the chart's row lock, and the lock is let go once both wait for it. One answers
    `{ok: true}` and the other exactly `{stale: true}`: the second waited for the first one's commit, then saw its
    `changed_on`. The winner's key is set and the loser's is not, and every other params key is as it was."""
    client = as_user(live, BUILDER)
    _, chart_id = _made_line(live, client, policy_id, "race")
    before = stored_chart(chart_id)["params"]
    seen = changed_on(client, "chart", chart_id)
    keys = {"row_limit": 321, "color_scheme": "bnbColors"}
    answers = race("slices", chart_id, [
        lambda key=key, value=value: call(as_user(live, BUILDER), "patch_chart", chart_id=chart_id,
                                          set={key: value}, expect_changed_on=seen)
        for key, value in keys.items()])
    assert sorted(answers, key=lambda a: "stale" in a) == [{"ok": True}, {"stale": True}], answers
    winner, loser = [key for key, answer in zip(keys, answers, strict=True) if answer == {"ok": True}] + [
        key for key, answer in zip(keys, answers, strict=True) if answer == {"stale": True}]
    after = stored_chart(chart_id)["params"]
    assert after[winner] == keys[winner]
    assert after.get(loser) == before.get(loser)
    assert {k: v for k, v in after.items() if k != winner} == {k: v for k, v in before.items() if k != winner}
    observe("patch_race", answers=answers, winner=winner)


# --- add_saved_metric -----------------------------------------------------------------------------------------------


def test_a_write_held_up_past_the_lock_timeout_answers_busy(live, extension, policy_id, built_extension):
    """With the lock timeout at 3 seconds (`MNEMIQ_LOCK_TIMEOUT_S`, in an MCP process of the test's own on the suite's
    configuration; the running services keep the default), a `patch_chart` on a chart whose row another transaction
    holds answers exactly `{busy: true}` once it has waited about 3 seconds, while the row is still held (the test
    holds it 20 seconds at most), and changes nothing. The control: once the row is let go, the same request answers
    `{ok}` and sets the key."""
    client = as_user(live, BUILDER)
    _, chart_id = _made_line(live, client, policy_id, "busy")
    request = {"chart_id": chart_id, "set": {"row_limit": 77}, "unset": [],
               "expect_changed_on": changed_on(client, "chart", chart_id)}
    port = 15011
    with throwaway_superset(["mcp", "run", "--host", "127.0.0.1", "--port", str(port)], "MNEMIQ_LOCK_TIMEOUT_S = 3\n",
                            built_extension, lambda output: b"Application startup complete" in output):
        bounded = SupersetMcp(f"http://127.0.0.1:{port}/mcp", DelegatedToken(live.token(BUILDER, scope=BUILD)),
                              timeout_s=90)
        answers: list = []
        errors: list[BaseException] = []

        def patch() -> None:
            try:
                answers.append(call(bounded, "patch_chart", **request))
            except BaseException as exc:  # noqa: BLE001 (reported below)
                errors.append(exc)

        held = _hold("slices", chart_id)
        try:
            started = time.monotonic()
            worker = threading.Thread(target=patch)
            worker.start()
            worker.join(20)
            waited = time.monotonic() - started
            answered_while_held = bool(answers or errors)
        finally:
            held.rollback()
            held.close()
        worker.join(90)
        observe("busy", answers=answers, waited_s=round(waited, 1), answered_while_held=answered_while_held)
        assert not errors, errors
        assert answered_while_held, "no answer while the chart's row was held"
        assert answers == [{"busy": True}]
        assert 3 <= waited < 20, waited
        assert "row_limit" not in stored_chart(chart_id)["params"] or stored_chart(chart_id)["params"]["row_limit"] != 77
        assert call(bounded, "patch_chart", **request) == {"ok": True}
    assert stored_chart(chart_id)["params"]["row_limit"] == 77


def _seeded_dataset(live, build: Build, step: str, metrics: list[dict]) -> int:
    """A dataset with saved metrics of its own, the step's marker its description: made by MCP's
    `create_virtual_dataset`, since the extension's `create_dataset` takes no metrics (a build adds its own through
    `add_saved_metric`, which sets no warning text). Its creator, analyst, is its one editor."""
    return virtual_dataset(live, build.client, build.marker(step), metrics=metrics)


def _metric(name: str, expression: str = "AVG(premium)", **more) -> dict:
    return {"metric_name": name, "expression": expression, **more}


def test_simultaneous_metric_saves_keep_both_or_refuse_one(live, extension):
    """Two saved metrics added to one dataset, both sending the same `changed_on`, start together while the test holds
    the dataset's row lock, let go once both wait. One answers `{ok: true}` and the other exactly `{stale: true}`. No
    metric is lost: every metric the dataset had is there unchanged, the winner's is there, and the loser's is not."""
    client = as_user(live, BUILDER)
    build = new_build(client, "metric race")
    dataset_id = _seeded_dataset(live, build, "metrics",
                                 [{"metric_name": "total_premium", "expression": "SUM(premium)"}])
    before = stored_metrics(dataset_id)
    seen = changed_on(client, "dataset", dataset_id)
    names = ("avg_premium", "max_premium")
    answers = race("tables", dataset_id, [
        lambda name=name, expression=expression: call(as_user(live, BUILDER), "add_saved_metric",
                                                      dataset_id=dataset_id, metric=_metric(name, expression),
                                                      expect_changed_on=seen)
        for name, expression in zip(names, ("AVG(premium)", "MAX(premium)"), strict=True)])
    assert sorted(answers, key=lambda a: "stale" in a) == [{"ok": True}, {"stale": True}], answers
    winner = names[answers.index({"ok": True})]
    after = stored_metrics(dataset_id)
    assert {name: row for name, row in after.items() if name in before} == before
    assert set(after) == set(before) | {winner}
    observe("metric_race", answers=answers, winner=winner, existing=sorted(before))


def test_add_saved_metric_keeps_existing_metrics(live, extension):
    """On a dataset made with a metric of its own (`total_premium`, with a verbose name, a format and a warning), adding
    `avg_premium` (with a verbose name, a format, a USD currency and a description) answers `{ok: true}`, then adding
    `max_premium` answers `{ok: true}`. Each metric the dataset had, and `avg_premium` after the second add, is
    exactly as it was in every stored field; the new ones carry what was sent. Adding `avg_premium` again answers
    `{exists: true}` and changes nothing."""
    client = as_user(live, BUILDER)
    build = new_build(client, "metrics")
    dataset_id = _seeded_dataset(live, build, "metrics", [{
        "metric_name": "total_premium", "expression": "SUM(premium)", "verbose_name": "Total premium",
        "d3format": ",.2f", "warning_text": "kept as it was"}])
    before = stored_metrics(dataset_id)
    assert "total_premium" in before
    average = _metric("avg_premium", verbose_name="Average premium", d3format=",.2f",
                      currency={"symbol": "USD", "symbolPosition": "prefix"}, description="per policy")
    assert call(client, "add_saved_metric", dataset_id=dataset_id, metric=average,
                expect_changed_on=changed_on(client, "dataset", dataset_id)) == {"ok": True}
    middle = stored_metrics(dataset_id)
    assert call(client, "add_saved_metric", dataset_id=dataset_id, metric=_metric("max_premium", "MAX(premium)"),
                expect_changed_on=changed_on(client, "dataset", dataset_id)) == {"ok": True}
    after = stored_metrics(dataset_id)
    assert {name: after[name] for name in before} == before
    assert after["avg_premium"] == middle["avg_premium"]
    _, verbose, _, expression, description, d3format, _, _, _, currency = after["avg_premium"]
    assert (verbose, expression, description, d3format) == ("Average premium", "AVG(premium)", "per policy", ",.2f")
    assert json.loads(currency) == {"symbol": "USD", "symbolPosition": "prefix"}
    assert set(after) == set(before) | {"avg_premium", "max_premium"}
    assert call(client, "add_saved_metric", dataset_id=dataset_id, metric=average,
                expect_changed_on=changed_on(client, "dataset", dataset_id)) == {"exists": True}
    assert stored_metrics(dataset_id) == after
    observe("metrics_kept", before=sorted(before), after=sorted(after))


def test_add_saved_metric_refuses_a_row_level_expression(live, extension):
    """A metric whose expression does not aggregate (`premium`, or the windowed `SUM(premium) OVER ()`) answers
    `{refused: "row_level"}`, Superset's own `has_aggregate` reading it, and the dataset's metrics and `changed_on`
    are as they were. The control: `SUM(premium)` on the same dataset answers `{ok: true}`."""
    client = as_user(live, BUILDER)
    build = new_build(client, "row level")
    dataset_id = marked_dataset(live, build, "metrics")
    before, seen = stored_metrics(dataset_id), changed_on(client, "dataset", dataset_id)
    for expression in ("premium", "SUM(premium) OVER ()"):
        assert call(client, "add_saved_metric", dataset_id=dataset_id, metric=_metric("row_level", expression),
                    expect_changed_on=seen) == {"refused": "row_level"}
    assert stored_metrics(dataset_id) == before
    assert changed_on(client, "dataset", dataset_id) == seen
    assert call(client, "add_saved_metric", dataset_id=dataset_id, metric=_metric("summed", "SUM(premium)"),
                expect_changed_on=seen) == {"ok": True}


# --- set_layout -----------------------------------------------------------------------------------------------------


def _layout(build: Build, chart_ids: list[int], *, add=(), remove=()) -> dict:
    return call(build.client, "set_layout", dashboard_id=build.dashboard_id, positions=positions(chart_ids),
                add_chart_ids=list(add), remove_chart_ids=list(remove),
                expect_changed_on=changed_on(build.client, "dashboard", build.dashboard_id))


def test_set_layout_moves_membership_with_positions(live, extension, policy_id):
    """A layout placing charts 1 and 2 (both added) answers `{ok: true}`, and the dashboard's charts become exactly
    1 and 2, each placed in the stored layout. Then a layout placing 1 and 3, adding 3 and removing 2, makes them 1
    and 3. Refused, changing nothing: a layout that leaves out a current chart it does not remove
    (`{refused: "chart_set", missing: [...]}`), one adding a chart that does not exist
    (`{refused: "chart_access", chart_ids: [...]}`), and one sent with a `changed_on` since replaced (`{stale}`)."""
    client = as_user(live, BUILDER)
    build = new_build(client, "layout")
    one, two, three = (made_chart(build, spec(Family.BIG_NUMBER, policy_id), f"kpi_{i}") for i in range(3))
    first = changed_on(client, "dashboard", build.dashboard_id)
    assert _layout(build, [one, two], add=[one, two]) == {"ok": True}
    assert members(build.dashboard_id) == sorted([one, two])
    assert _layout(build, [one, three], add=[three], remove=[two]) == {"ok": True}
    assert members(build.dashboard_id) == sorted([one, three])
    (stored,) = metadata("SELECT position_json FROM dashboards WHERE id = %s", (build.dashboard_id,))
    placed = {node["meta"]["chartId"] for node in json.loads(stored[0]).values()
              if isinstance(node, dict) and node.get("type") == "CHART"}
    assert placed == {one, three}
    assert _layout(build, [one]) == {"refused": "chart_set", "missing": [three], "unexpected": []}
    missing_chart = 2_000_000_000
    assert _layout(build, [one, three, missing_chart], add=[missing_chart]) == {
        "refused": "chart_access", "chart_ids": [missing_chart]}
    assert call(client, "set_layout", dashboard_id=build.dashboard_id, positions=positions([one]), remove_chart_ids=
                [three], expect_changed_on=first) == {"stale": True}
    assert members(build.dashboard_id) == sorted([one, three])


def test_simultaneous_layout_saves(live, extension, policy_id):
    """Two layouts of one dashboard, one adding chart A and the other chart B, both sending the same `changed_on`,
    start together while the test holds the dashboard's row lock, let go once both wait. One answers `{ok: true}` and
    the other exactly `{stale: true}`, and the dashboard's charts are the winner's alone."""
    client = as_user(live, BUILDER)
    build = new_build(client, "layout race")
    charts = [made_chart(build, spec(Family.BIG_NUMBER, policy_id), f"kpi_{i}") for i in range(2)]
    seen = changed_on(client, "dashboard", build.dashboard_id)
    answers = race("dashboards", build.dashboard_id, [
        lambda chart=chart: call(as_user(live, BUILDER), "set_layout", dashboard_id=build.dashboard_id,
                                 positions=positions([chart]), add_chart_ids=[chart], expect_changed_on=seen)
        for chart in charts])
    assert sorted(answers, key=lambda a: "stale" in a) == [{"ok": True}, {"stale": True}], answers
    assert members(build.dashboard_id) == [charts[answers.index({"ok": True})]]
    observe("layout_race", answers=answers)


def test_layout_save_keeps_filters_and_reports(live, extension, policy_id):
    """A dashboard with a chart laid out, then a native filter and an active report that names the filter (made by
    `setup_superset.py filter-and-report`). A layout adding a second chart answers `{ok: true}`; afterwards the
    dashboard still has the filter, and the report is still active."""
    client = as_user(live, BUILDER)
    build = new_build(client, "filters")
    one, two = (made_chart(build, spec(Family.BIG_NUMBER, policy_id), f"kpi_{i}") for i in range(2))
    assert _layout(build, [one], add=[one]) == {"ok": True}
    made = json.loads(run_setup("filter-and-report", str(build.dashboard_id), BUILDER))

    def state() -> tuple[list[str], bool]:
        (stored,) = metadata("SELECT json_metadata FROM dashboards WHERE id = %s", (build.dashboard_id,))
        ((active,),) = metadata("SELECT active FROM report_schedule WHERE id = %s", (made["report_id"],))
        filters = json.loads(stored[0] or "{}").get("native_filter_configuration", [])
        return [f["id"] for f in filters], active

    assert state() == ([made["filter_id"]], True)
    assert _layout(build, [one, two], add=[two]) == {"ok": True}
    assert members(build.dashboard_id) == sorted([one, two])
    assert state() == ([made["filter_id"]], True)
    observe("filters_kept", **made)


# --- create_chart ---------------------------------------------------------------------------------------------------


def test_create_patch_read_sankey(live, extension, policy_id):
    """A sankey (source region, target status, COUNT(*)) made by the extension's `create_chart` answers
    `{chart_id}`, stores mnemiq's params, the build's marker and the dashboard's editors, and a query context grouped
    by region and status. Patched (`row_limit` 50) it answers `{ok: true}`, and its query context follows: still
    grouped by region and status, row limit 50. `get_chart_data` then answers one row per region and status, nine,
    each count the source's."""
    client = as_user(live, BUILDER)
    build = new_build(client, "sankey")
    made = extension_chart(build, spec(Family.SANKEY, policy_id, source="region", target="status"), "flow")
    assert set(made) == {"chart_id"}, made
    chart_id = made["chart_id"]
    stored = stored_chart(chart_id)
    assert (stored["params"]["source"], stored["params"]["target"]) == ("region", "status")
    assert stored["description"] == build.marker("flow") and stored["viz_type"] == "sankey_v2"
    assert editor_ids("chart_editors", "chart_id", chart_id) == editor_ids("dashboard_editors", "dashboard_id",
                                                                           build.dashboard_id)
    assert stored["query_context"]["queries"][0]["columns"] == ["region", "status"]
    assert call(client, "patch_chart", chart_id=chart_id, set={"row_limit": 50},
                expect_changed_on=changed_on(client, "chart", chart_id)) == {"ok": True}
    query = stored_chart(chart_id)["query_context"]["queries"][0]
    assert query["columns"] == ["region", "status"] and query["row_limit"] == 50
    rows = sorted((r["region"], r["status"], r["policies"]) for r in chart_data(client, chart_id)["data"])
    assert rows == sorted(ground_truth(live, BY_REGION_AND_STATUS))


def test_funnel_params_and_query_context(live, extension, policy_id):
    """A funnel (group by status, COUNT(*)) made by `create_chart` answers `{chart_id}`, stores mnemiq's params with
    its viz type and dataset, and a query context grouped by status; `get_chart_data` answers one row per status,
    each count the source's. A funnel grouped by a column the dataset lacks answers `{refused: "query_failed"}` and no
    chart of that name is saved: its query is run before anything is."""
    client = as_user(live, BUILDER)
    build = new_build(client, "funnel")
    funnel = spec(Family.FUNNEL, policy_id, dimensions=("status",))
    made = extension_chart(build, funnel, "stages")
    assert set(made) == {"chart_id"}, made
    stored = stored_chart(made["chart_id"])
    sent = extension_chart_request(funnel, {}, {}, currency="USD")["params"]
    assert stored["params"] == sent and stored["viz_type"] == "funnel"
    assert stored["query_context"]["queries"][0]["columns"] == ["status"]
    rows = sorted((r["status"], r["policies"]) for r in chart_data(client, made["chart_id"])["data"])
    assert rows == sorted(ground_truth(live, BY_STATUS))
    broken = spec(Family.FUNNEL, policy_id, dimensions=("no_such_column",))
    assert extension_chart(build, broken, "broken") == {"refused": "query_failed"}
    assert metadata("SELECT count(*) FROM slices WHERE slice_name = %s", (broken.title,)) == [(0,)]


# --- a create carrying a build's marker -----------------------------------------------------------------------------


def test_create_chart_twice_with_one_marker_makes_one_chart(live, extension, policy_id, made):
    """A monthly line chart made by the extension's `create_chart` from mnemiq's own request (`config_chart_request`)
    answers `{chart_id}`, and its description is the build's marker alone. The same request sent again, as a build
    sends a create whose answer was lost, answers the same `{chart_id}` and changes nothing: the chart's row is as it
    was, and Superset holds one chart carrying the marker and one of that name. The control: the same request with
    another build's marker for the same step and dashboard makes a chart of its own, and the first chart still carries
    its own build's marker: a create finds a chart by its marker, never by its name, dataset or dashboard."""
    client = as_user(live, BUILDER)
    build = made.build(client, "once")
    chart = spec(Family.LINE, policy_id, time_column="issued", time_grain="P1M")
    marker = build.marker("by_month")
    first = made.chart(call(client, "create_chart", **config_request(build, chart, marker)))
    assert set(first) == {"chart_id"}, first
    chart_id = first["chart_id"]
    stored = stored_chart(chart_id)
    assert stored["description"] == marker
    assert parse_marker(stored["description"]) == Marker(build.build_id, "by_month", build.dashboard_id)
    again = made.chart(call(client, "create_chart", **config_request(build, chart, marker)))
    assert again == first
    assert stored_chart(chart_id) == stored
    assert carrying("slices", marker) == [(chart_id, None)]
    named = "SELECT id FROM slices WHERE slice_name = %s ORDER BY id"
    assert metadata(named, (chart.title,)) == [(chart_id,)]
    other = format_marker(str(uuid.uuid4()), "by_month", build.dashboard_id)
    theirs = made.chart(call(client, "create_chart", **config_request(build, chart, other)))
    assert set(theirs) == {"chart_id"} and theirs["chart_id"] != chart_id, theirs
    assert stored_chart(chart_id) == stored
    assert carrying("slices", other) == [(theirs["chart_id"], None)]
    assert metadata(named, (chart.title,)) == [(chart_id,), (theirs["chart_id"],)]
    observe("create_twice", chart_id=chart_id, other_build=theirs["chart_id"])


def test_concurrent_creates_with_one_marker_make_one_chart(live, extension, policy_id, made):
    """Two clients send one create (a bar chart by region, one marker) at once. The test holds the marker's lock from
    a connection of its own until both creates wait for it, so both have passed the caller's checks, which run before
    the lock, and neither has looked for the chart or checked its query yet; then it lets the lock go well inside the
    lock timeout. The first to take it looks, checks the chart's query and makes the chart, holding the lock through
    the commit; the other waits for it, then finds that chart. Both answer the same `{chart_id}`, and Superset holds
    one chart carrying the marker and one of that name."""
    client = as_user(live, BUILDER)
    build = made.build(client, "concurrent")
    chart = spec(Family.BAR, policy_id, dimensions=("region",))
    marker = build.marker("by_region")
    request = config_request(build, chart, marker)
    answers: list = [None, None]
    errors: list[BaseException] = []

    def send(index: int) -> None:
        try:
            answers[index] = call(as_user(live, BUILDER), "create_chart", **request)
        except BaseException as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)

    threads = [threading.Thread(target=send, args=(index,)) for index in range(2)]
    held = _hold_marker(marker)
    started = time.monotonic()
    try:
        for thread in threads:
            thread.start()
        while _waiting_on_advisory_locks() < 2:
            assert time.monotonic() - started < LOCK_TIMEOUT_S - 5, "the creates never both waited for the lock"
            assert not errors, errors
            time.sleep(0.1)
        both_waiting = time.monotonic() - started
    finally:
        held.rollback()
        held.close()
        for thread in threads:
            thread.join(90)
        for answer in answers:
            made.chart(answer or {})
    observe("concurrent_creates", answers=answers, both_waiting_after_s=round(both_waiting, 1))
    assert not errors, errors
    assert set(answers[0]) == {"chart_id"} and answers[1] == answers[0], answers
    assert carrying("slices", marker) == [(answers[0]["chart_id"], None)]
    assert metadata("SELECT count(*) FROM slices WHERE slice_name = %s", (chart.title,)) == [(1,)]


def test_a_create_behind_a_held_marker_lock_answers_busy(live, extension, policy_id, made):
    """While a connection of the test's own holds the marker's lock inside a transaction (as a create carrying that
    marker holds it from its look until after its commit), a create carrying the marker (a pie by status) waits for it
    the lock timeout (30 seconds, the extension's default, which the suite's Superset keeps) and answers exactly
    `{busy: true}`, within 40 seconds of the call, having made nothing: no chart carries the marker or the chart's
    name. Let go, the same request makes the chart: `{chart_id}`, the one chart carrying the marker."""
    client = as_user(live, BUILDER)
    build = made.build(client, "held lock")
    chart = spec(Family.PIE, policy_id, dimensions=("status",))
    marker = build.marker("by_status")
    request = config_request(build, chart, marker)
    named = "SELECT count(*) FROM slices WHERE slice_name = %s"
    held = _hold_marker(marker)
    try:
        started = time.monotonic()
        answer = made.chart(call(client, "create_chart", **request))
        waited = time.monotonic() - started
        while_held = (carrying("slices", marker), metadata(named, (chart.title,)))
    finally:
        held.rollback()
        held.close()
    observe("held_marker_lock", answer=answer, waited_s=round(waited, 1))
    assert answer == {"busy": True}
    assert LOCK_TIMEOUT_S <= waited < 40, waited
    assert while_held == ([], [(0,)])
    after = made.chart(call(client, "create_chart", **request))
    assert set(after) == {"chart_id"}, after
    assert carrying("slices", marker) == [(after["chart_id"], None)]
    assert metadata(named, (chart.title,)) == [(1,)]


def _source_reads_waiting_on(live, table: str) -> int:
    """How many other sessions of the source database wait for a lock while running a query that names `table`."""
    import psycopg

    with psycopg.connect(live.pg_dsn, connect_timeout=10) as con:
        return con.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                           "AND wait_event_type = 'Lock' AND query LIKE %s AND pid <> pg_backend_pid()",
                           (f"%{table}%",)).fetchone()[0]


def test_a_create_whose_lock_connection_ends_mid_check_makes_nothing(live, extension, made):
    """A pie by status on a virtual dataset of the build's over a table of the test's own. The test holds that table
    locked (`ACCESS EXCLUSIVE`, from a connection of its own) while the create runs, so the chart's query check waits
    for it, and the create holds its marker's lock, idle on a connection of its own, through that wait. With the check
    waiting, the test finds the marker lock's database session in `pg_locks` by the marker's key and ends it
    (`pg_terminate_backend`, which waits for the session to exit), as an idle timeout shorter than a chart's query
    check would, so the lock goes with it; only once the session has ended does it let the table go. Reaching its
    command, the create finds no chart and finds its lock no longer held: it answers exactly `{busy: true}`, and no
    chart carries the marker or the chart's name. The same request sent again makes the chart: `{chart_id}`, the one
    chart carrying the marker. Every other create in this suite passes the same hold check, so a hold check that
    misread `pg_locks` would answer busy there.

    (A source that sleeps cannot hold the check: Superset 7.0.0rc2 refuses `pg_sleep` in a chart's query, as one of
    its `DISALLOWED_SQL_FUNCTIONS`.)"""
    import psycopg
    from psycopg import sql

    client = as_user(live, BUILDER)
    build = made.build(client, "lost lock")
    table = f"live_held_{uuid.uuid4().hex[:8]}"
    made.tables.append(table)
    with psycopg.connect(live.pg_dsn, autocommit=True, connect_timeout=10) as con:
        con.execute(sql.SQL("CREATE TABLE {} AS {}").format(sql.Identifier("public", table), sql.SQL(POLICY_COLUMNS)))
    data = made.dataset(call(client, "create_dataset", database_id=live.database_id, schema="public",
                             table_name=f"mnemiq_live_{uuid.uuid4().hex[:10]}",
                             sql=f"SELECT id, region, status, premium, issued FROM public.{table}",
                             description=f"Made by mnemiq for dashboard {build.dashboard_id}.",
                             dashboard_id=build.dashboard_id, marker=build.marker("held_data")), build.dashboard_id)
    assert set(data) == {"dataset_id"}, data
    chart = spec(Family.PIE, data["dataset_id"], dimensions=("status",))
    marker = build.marker("by_status")
    request = config_request(build, chart, marker)
    named = "SELECT count(*) FROM slices WHERE slice_name = %s"
    answers: list = []
    errors: list[BaseException] = []

    def send() -> None:
        try:
            answers.append(made.chart(call(as_user(live, BUILDER), "create_chart", **request)))
        except BaseException as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)

    first = threading.Thread(target=send)
    held = psycopg.connect(live.pg_dsn, connect_timeout=10)
    try:
        held.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(sql.Identifier("public", table)))
        started = time.monotonic()
        first.start()
        while not ((holders := _marker_lock_holders(marker)) and _source_reads_waiting_on(live, table)):
            assert time.monotonic() - started < LOCK_TIMEOUT_S, ("the create never held its lock through its check",
                                                                 holders)
            assert first.is_alive(), ("the create answered before its check waited", answers, errors)
            time.sleep(0.05)
        waiting_after = time.monotonic() - started
        (holder,) = holders
        # With a timeout, Postgres waits up to 5 seconds for the session to exit and answers whether it did: the lock
        # has gone with it before the check can run on.
        terminated = metadata("SELECT pg_terminate_backend(%s, 5000)", (holder,))
        assert terminated == [(True,)], terminated
    finally:
        held.rollback()
        held.close()
        if first.ident is not None:
            first.join(90)
    answered_after = time.monotonic() - started
    after_first = (carrying("slices", marker), metadata(named, (chart.title,)))
    observe("lock_lost_mid_check", answers=answers, check_waiting_after_s=round(waiting_after, 2),
            terminated=terminated, answered_after_s=round(answered_after, 2), chart_rows=after_first)
    assert not errors, errors
    assert answers == [{"busy": True}], answers
    assert after_first == ([], [(0,)])
    again = made.chart(call(client, "create_chart", **request))
    observe("lock_lost_mid_check_sent_again", answer=again)
    assert set(again) == {"chart_id"}, again
    assert carrying("slices", marker) == [(again["chart_id"], None)]
    assert metadata(named, (chart.title,)) == [(1,)]


def test_create_dataset_physical_and_virtual_carry_the_marker(live, extension, made):
    """For a build's dashboard, as a build makes them, each by the extension's `create_dataset`: a physical dataset
    over a table of the test's own (made in the source database, so no dataset exists on it), and a virtual one over a
    statement of `policy`. Each answers `{dataset_id}`; Superset holds it on that table with no SQL, or with that SQL;
    its description's last line is the build's marker for its step (the virtual one's prose before it); and analyst,
    the dashboard's one editor, is its one editor. Each request sent again answers the same id and leaves its table,
    SQL, description and editors as they were, the physical one found by its marker before rc2's check that a table
    holds one dataset; Superset holds one dataset carrying each marker. The control: the same table asked for with
    another build's marker answers `{exists: true}` and makes nothing."""
    import psycopg
    from psycopg import sql

    client = as_user(live, BUILDER)
    build = made.build(client, "datasets")
    table = f"live_marked_{uuid.uuid4().hex[:8]}"
    made.tables.append(table)
    with psycopg.connect(live.pg_dsn, autocommit=True, connect_timeout=10) as con:
        con.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY, region text NOT NULL)").format(
            sql.Identifier("public", table)))
    base = {"database_id": live.database_id, "schema": "public", "dashboard_id": build.dashboard_id}
    requests = {
        "physical": {**base, "table_name": table, "marker": build.marker("physical")},
        "virtual": {**base, "table_name": f"mnemiq_live_{uuid.uuid4().hex[:10]}", "sql": POLICY_COLUMNS,
                    "description": f"Made by mnemiq for dashboard {build.dashboard_id}.",
                    "marker": build.marker("virtual")}}
    ids = {}
    for step, request in requests.items():
        answer = made.dataset(call(client, "create_dataset", **request), build.dashboard_id)
        assert set(answer) == {"dataset_id"}, (step, answer)
        ids[step] = answer["dataset_id"]
    columns = "SELECT table_name, schema, sql, description, created_by_fk, deleted_at FROM tables WHERE id = %s"
    rows = {step: metadata(columns, (dataset_id,))[0] for step, dataset_id in ids.items()}
    assert rows["physical"][:4] == (table, "public", None, requests["physical"]["marker"]), rows["physical"]
    assert rows["virtual"][:4] == (requests["virtual"]["table_name"], "public", POLICY_COLUMNS,
                                   f"{requests['virtual']['description']}\n{requests['virtual']['marker']}")
    for step, dataset_id in ids.items():
        assert parse_marker(rows[step][3]) == Marker(build.build_id, step, build.dashboard_id)
        assert rows[step][5] is None
        assert editor_ids("sqlatable_editors", "table_id", dataset_id) == [subject_of(username=BUILDER)]
    for step, request in requests.items():
        again = made.dataset(call(client, "create_dataset", **request), build.dashboard_id)
        assert again == {"dataset_id": ids[step]}, (step, again)
        assert metadata(columns, (ids[step],))[0] == rows[step]
        assert editor_ids("sqlatable_editors", "table_id", ids[step]) == [subject_of(username=BUILDER)]
        assert carrying("tables", request["marker"]) == [(ids[step], None)]
    other = {**requests["physical"], "marker": format_marker(str(uuid.uuid4()), "physical", build.dashboard_id)}
    assert made.dataset(call(client, "create_dataset", **other), build.dashboard_id) == {"exists": True}
    assert carrying("tables", other["marker"]) == []
    assert metadata("SELECT count(*) FROM tables WHERE table_name = %s", (table,)) == [(1,)]
    observe("create_dataset", **ids)


# The family neither MCP's `generate_chart` nor the extension's `create_chart` saves in rc2, each run through MCP's own
# checks first: the interactive pivot, not among this rc2's enabled chart types (its plugin is not in rc2), which the
# config's validation refuses, as mnemiq's plan check refuses it before a build where it is not offered.
# A histogram with no grouping column is saved by the extension alone: rc2's `generate_chart` cannot save one, because
# its compile check builds the query from a chart's metrics and groupby, which a histogram's form data (its column
# under `column` alone) does not carry, so the database answers "Empty query?"; the extension checks it with its own
# query instead.
BOTH_REFUSE = {"interactive_pivot"}


def test_create_chart_leaves_out_no_value_from_the_metric_as_superset_saved_it(live, extension, policy_id, made):
    """A bar of `policies` by region with a row limit, as a build sends it (`config_request`, which asks to leave out
    groups with no value): saved with a SQL HAVING filter last among its filters whose condition is built from the
    metric as Superset saved it, and in its query context's `extras.having`, wrapped as Superset's frontend wraps a
    clause. A big number asked the same, which ranks by no one metric, is refused and nothing carrying its marker is
    saved."""
    from mnemiq.superset.charts import not_null_condition

    client = as_user(live, BUILDER)
    build = new_build(client, "leave_out")
    chart = spec(Family.BAR, policy_id, dimensions=("region",), row_limit=10)
    request = config_request(build, chart, build.marker("leave_out"))
    assert request.get("leave_out_no_value") is True, request
    answer = made.chart(call(client, "create_chart", **request))
    assert set(answer) == {"chart_id"}, answer
    stored = stored_chart(answer["chart_id"])
    condition = not_null_condition(stored["params"]["metrics"][0]["sqlExpression"])
    last = stored["params"]["adhoc_filters"][-1]
    assert (last["clause"], last["expressionType"], last["sqlExpression"]) == ("HAVING", "SQL", condition)
    assert stored["query_context"]["queries"][0]["extras"]["having"] == f"({condition})"
    big_marker = build.marker("big_number")
    big = {**config_request(build, spec(Family.BIG_NUMBER, policy_id), big_marker), "leave_out_no_value": True}
    assert made.chart(call(client, "create_chart", **big)) == {"refused": "leave_out_no_value"}
    assert carrying("slices", big_marker) == []
    observe("leave out", chart_id=answer["chart_id"], having=stored["query_context"]["queries"][0]["extras"]["having"])


def test_a_config_chart_saves_what_generate_chart_saves(live, extension, policy_id, made):
    """For every one of MCP's chart families, one chart spec sent both ways, each request built by mnemiq: to MCP's
    `generate_chart` (`generate_chart_request`) and to the extension's `create_chart` (`config_chart_request`). The
    two saved charts' params are equal key by key, once the format keys the extension's request carries under `set`
    (`owed_format_params`: a table's column currencies, a treemap's number format), which MCP's chart config cannot
    carry, are merged into MCP's; their name, viz type and dataset are equal. What differs is what the extension adds:
    its description is the build's marker, where MCP's chart has none, and MCP's saves no query context.

    Where MCP saves nothing, the extension saves nothing either: it answers a refusal and no chart of the name
    exists. Those are the histogram and the interactive pivot alone (`BOTH_REFUSE`), so a family that starts failing
    both ways fails here."""
    client = as_user(live, BUILDER)
    build = made.build(client, "same params")
    count = WrittenRef(policies(policy_id))
    total = written(policy_id, "total premium", "SUM(premium)")
    monthly = {"time_column": "issued", "time_grain": "P1M"}
    charts = {
        "line": spec(Family.LINE, policy_id, measures=(total,), **monthly),
        "bar": spec(Family.BAR, policy_id, dimensions=("region",)),
        "area": spec(Family.AREA, policy_id, measures=(total,), dimensions=("status",), **monthly),
        "scatter": spec(Family.SCATTER, policy_id, **monthly),
        "table": spec(Family.TABLE, policy_id, measures=(total, count), dimensions=("region", "status")),
        "pie": spec(Family.PIE, policy_id, measures=(total,), dimensions=("status",)),
        "big_number": spec(Family.BIG_NUMBER, policy_id, measures=(total,)),
        "gauge": spec(Family.GAUGE, policy_id),
        "histogram": spec(Family.HISTOGRAM, policy_id, measures=(), dimensions=("premium",)),
        "histogram_grouped": spec(Family.HISTOGRAM, policy_id, measures=(), dimensions=("premium", "region"),
                                  filters=(Filter("status", "=", "active"),), row_limit=500),
        "box_plot": spec(Family.BOX_PLOT, policy_id, dimensions=("region",)),
        "pivot": spec(Family.PIVOT, policy_id, measures=(total,), dimensions=("region", "status")),
        "interactive_pivot": spec(Family.INTERACTIVE_PIVOT, policy_id, measures=(total,),
                                  dimensions=("region", "status")),
        "treemap": spec(Family.TREEMAP, policy_id, measures=(total,), dimensions=("region", "status")),
        "waterfall": spec(Family.WATERFALL, policy_id, measures=(total,), dimensions=("status",), **monthly),
        "mixed": spec(Family.MIXED, policy_id, measures=(total, count), **monthly),
        "handlebars": spec(Family.HANDLEBARS, policy_id, dimensions=("region",)),
    }
    assert {chart.family for chart in charts.values()} == set(Family) - {Family.FUNNEL, Family.SANKEY}
    identity = "SELECT slice_name, viz_type, datasource_id, datasource_type FROM slices WHERE id = %s"
    differing, contexts, refused = {}, {}, {}
    for name, chart in charts.items():
        answer = client.call("generate_chart", generate_chart_request(chart, {}, {}, currency="USD"))
        mcp_id = (answer.get("chart") or {}).get("id")
        made.chart({"chart_id": mcp_id})
        own = made.chart(config_chart(build, chart, name))
        if name == "histogram":
            assert not answer.get("success") and mcp_id is None, answer
            assert set(own) == {"chart_id"}, own
            saved = stored_chart(own["chart_id"])
            params = saved["params"]
            observe("histogram_saved", params=params, query_context=saved["query_context"])
            assert params["viz_type"] == saved["viz_type"] == "histogram_v2", params
            assert params["column"] == "premium" and params["groupby"] == [], params
            assert {"bins", "normalize", "cumulative", "row_limit"} <= set(params), params
            assert saved["query_context"] is None
            continue
        if not answer.get("success"):
            refused[name] = {"generate_chart": (answer.get("error") or {}).get("details"), "create_chart": own}
            assert mcp_id is None and "chart_id" not in own and "refused" in own, (name, own)
            assert metadata("SELECT count(*) FROM slices WHERE slice_name = %s", (chart.title,)) == [(0,)], name
            continue
        assert type(mcp_id) is int, name
        assert set(own) == {"chart_id"}, (name, own)
        theirs, ours = stored_chart(mcp_id), stored_chart(own["chart_id"])
        expected = {**theirs["params"], **owed_format_params(chart, {}, {}, currency="USD")}
        differing[name] = sorted(key for key in set(expected) | set(ours["params"])
                                 if key not in expected or key not in ours["params"]
                                 or expected[key] != ours["params"][key])
        contexts[name] = ours["query_context"] is not None
        if name == "histogram_grouped":
            grouped = ours["params"]
            assert grouped["viz_type"] == "histogram_v2" and grouped["column"] == "premium", grouped
            assert grouped["groupby"] == ["region"] and grouped["row_limit"] == 500, grouped
            assert {"bins", "normalize", "cumulative"} <= set(grouped), grouped
            # rc2's mapping adds its own no-op time-range filter on the dataset's time column beside the chart's.
            assert ("status", "==", "active") in [(f["subject"], f["operator"], f["comparator"])
                                                   for f in grouped["adhoc_filters"]], grouped
            assert ours["query_context"] is None
        assert metadata(identity, (mcp_id,)) == metadata(identity, (own["chart_id"],)), name
        assert (theirs["description"], ours["description"]) == (None, build.marker(name)), name
        assert theirs["query_context"] is None, name
    observe("same_params", differing=differing, query_context=contexts, both_refused=refused)
    assert not any(differing.values()), differing
    assert set(refused) == BOTH_REFUSE, refused


def test_a_histogram_over_a_text_column_is_refused_and_nothing_is_made(live, extension, policy_id, made):
    """A histogram of the text column `region`, sent to the extension's `create_chart` as mnemiq's own
    `config_chart_request` builds it, is refused and no chart of its name exists afterwards. The histogram over a
    numeric column (`premium`) is still made: `test_a_config_chart_saves_what_generate_chart_saves` saves it
    (its "histogram" family), so it is not repeated here."""
    client = as_user(live, BUILDER)
    build = made.build(client, "text histogram")
    chart = spec(Family.HISTOGRAM, policy_id, measures=(), dimensions=("region",))
    answer = config_chart(build, chart, "text_histogram")
    observe("text_histogram", answer=answer)
    # rc2 refuses it in `map_config_to_form_data`: the histogram plugin's `post_map_validate` raises
    # NON_NUMERIC_HISTOGRAM_COLUMN for a column it knows is text, which the extension answers as a refused config.
    assert answer == {"refused": "config"}, answer
    assert metadata("SELECT count(*) FROM slices WHERE slice_name = %s", (chart.title,)) == [(0,)]


# --- editors and deletes ----------------------------------------------------------------------------------------------


def test_non_editor_gets_editor_labels_not_a_change(live, extension, policy_id):
    """West (Alpha, able to read analyst's chart and dataset, editor of neither) is answered
    `{not_editor: true, editors: [analyst's label]}` when patching analyst's chart, adding a metric to analyst's
    dataset, sending again the very create that made analyst's chart (its marker, analyst's dashboard, published so
    west may read it), as a retry by a caller who does not edit the dashboard, and deleting analyst's dataset for
    west's dashboard (whose id the dataset's marker does not name: not being an editor is answered first). The chart
    and the dataset are as they were, and the chart is still the one chart carrying its marker."""
    analyst = as_user(live, BUILDER)
    build = new_build(analyst, "editors")
    kpi = spec(Family.BIG_NUMBER, policy_id)
    chart_id = made_chart(build, kpi, "kpi")
    dataset_id = marked_dataset(live, build, "data")
    published = analyst.call("update_dashboard", {"identifier": build.dashboard_id, "published": True})
    assert published.get("success") is not False, published.get("error")
    chart_before, metrics_before = stored_chart(chart_id), stored_metrics(dataset_id)
    west = as_user(live, "west")
    refused = {"not_editor": True, "editors": [label_of(BUILDER)]}
    assert call(west, "patch_chart", chart_id=chart_id, set={"row_limit": 5},
                expect_changed_on=changed_on(west, "chart", chart_id)) == refused
    assert call(west, "add_saved_metric", dataset_id=dataset_id, metric=_metric("west_metric"),
                expect_changed_on=changed_on(west, "dataset", dataset_id)) == refused
    assert call(west, "create_chart", **config_request(build, kpi, build.marker("kpi"))) == refused
    west_build = new_build(west, "west's")
    assert call(west, "delete_dataset", dataset_id=dataset_id, dashboard_id=west_build.dashboard_id) == refused
    assert stored_chart(chart_id) == chart_before and stored_metrics(dataset_id) == metrics_before
    assert carrying("slices", build.marker("kpi")) == [(chart_id, None)]
    assert metadata("SELECT deleted_at FROM tables WHERE id = %s", (dataset_id,)) == [(None,)]
    observe("not_editor", editors=refused["editors"])


def subject_of(*, username: str | None = None, role: str | None = None) -> int:
    """The editor subject standing for the user `username`, or for the role `role`."""
    if username is not None:
        (row,) = metadata("SELECT s.id FROM subjects s JOIN ab_user u ON u.id = s.user_id "
                          "WHERE u.username = %s AND s.type = 1", (username,))
    else:
        (row,) = metadata("SELECT s.id FROM subjects s JOIN ab_role r ON r.id = s.role_id "
                          "WHERE r.name = %s AND s.type = 2", (role,))
    return row[0]


def labels_of(subject_ids: list[int]) -> list[str]:
    return [row[0] for row in metadata("SELECT label FROM subjects WHERE id = ANY(%s) ORDER BY 1", (subject_ids,))]


def set_dashboard_editors(live, username: str, dashboard_id: int, subject_ids: list[int]) -> None:
    """As `username`, signed in, the dashboard's editors set to `subject_ids` through Superset's own REST API
    (`PUT /api/v1/dashboard/<id>`), as a person sets them in Superset: users and roles alike."""
    with live.signed_in(username) as (client, headers):
        csrf = live.csrf_token(client, headers)
        answer = client.put(f"/api/v1/dashboard/{dashboard_id}", json={"editors": subject_ids},
                            headers={**headers, "X-CSRFToken": csrf})
    assert answer.status_code == 200, f"HTTP {answer.status_code}"


def dataset_rows_as(live, username: str, dataset_id: int):
    """`username`'s request, signed in, for the dataset's policies by region through Superset's own chart data API
    (`POST /api/v1/chart/data`), whose access check (`raise_for_access` on the query context's datasource) counts
    being the dataset's editor as access."""
    body = {"datasource": {"id": dataset_id, "type": "table"}, "force": True, "result_format": "json",
            "result_type": "full",
            "queries": [{"columns": ["region"], "orderby": [], "row_limit": 100,
                         "metrics": [{"expressionType": "SQL", "sqlExpression": "COUNT(*)", "label": "policies"}]}]}
    with live.signed_in(username) as (client, headers):
        csrf = live.csrf_token(client, headers)
        return client.post("/api/v1/chart/data", json=body, headers={**headers, "X-CSRFToken": csrf})


def test_a_created_dataset_gets_only_editors_who_could_already_read_it(live, extension):
    """In rc2 a dataset's editors may read its data (`raise_for_access` admits them), and a role made an editor makes
    every member one, so `create_dataset` makes as editors the caller and only those of the dashboard's users whose
    grants already reach the dataset and its table, never a role.

    analyst's dashboard is edited, through Superset's own API, by analyst, `west` (Alpha: may read every dataset),
    `gamma_only` (Gamma: may not read the dataset), `policy_only` (granted the `policy` dataset alone, not this one),
    and the roles Gamma and Alpha. analyst makes a virtual dataset over `policy` for the dashboard, as a build makes
    one (`create_dataset`):
    - the dataset's editors are then exactly analyst and west;
    - `gamma_only` still may not read the dataset: Superset's chart data API answers its request 403, where
      analyst's same request answers each region's count, the source's;
    - `east` (Alpha, who may read the dataset, and edits the dashboard through the Alpha role alone) adding a metric
      to it is answered `{not_editor: true, editors: [analyst's and west's labels]}`, and the dataset's metrics are as
      they were."""
    analyst = as_user(live, BUILDER)
    build = new_build(analyst, "dataset editors")
    editors = ([subject_of(username=name) for name in (BUILDER, "west", "gamma_only", "policy_only")]
               + [subject_of(role=name) for name in ("Gamma", "Alpha")])
    set_dashboard_editors(live, BUILDER, build.dashboard_id, editors)
    assert editor_ids("dashboard_editors", "dashboard_id", build.dashboard_id) == sorted(editors)
    dataset_id = marked_dataset(live, build, "data")
    after = editor_ids("sqlatable_editors", "table_id", dataset_id)
    refused_data = dataset_rows_as(live, "gamma_only", dataset_id)
    observe("dataset_editors", dashboard=labels_of(editors), dataset=labels_of(after),
            gamma_only_data=refused_data.status_code)
    assert after == sorted([subject_of(username=BUILDER), subject_of(username="west")]), labels_of(after)
    assert refused_data.status_code == 403 and "result" not in refused_data.json(), refused_data.status_code
    control = dataset_rows_as(live, BUILDER, dataset_id)
    assert control.status_code == 200, control.status_code
    rows = sorted((row["region"], row["policies"]) for row in control.json()["result"][0]["data"])
    assert rows == sorted(ground_truth(live, "SELECT region, count(*) FROM policy GROUP BY 1"))
    metrics_before = stored_metrics(dataset_id)
    east = as_user(live, "east")
    refused = {"not_editor": True, "editors": sorted([label_of(BUILDER), label_of("west")])}
    assert call(east, "add_saved_metric", dataset_id=dataset_id, metric=_metric("east_metric"),
                expect_changed_on=changed_on(east, "dataset", dataset_id)) == refused
    assert stored_metrics(dataset_id) == metrics_before
    observe("dataset_editors_not_editor", editors=refused["editors"], analyst_rows=len(rows))


def test_delete_dataset_refuses_while_a_chart_uses_it(live, extension):
    """A dataset the build made for its dashboard (`create_dataset`), with a funnel on it: `delete_dataset` answers
    `{in_use: true, chart_ids: [the funnel]}` and the dataset stays. Once the funnel is deleted (MCP's
    `delete_chart`), `delete_dataset` answers `{ok: true, soft_deleted: true}` and the dataset is in Superset's
    trash (`deleted_at` set); asked again, it answers `{not_found: "dataset"}`."""
    client = as_user(live, BUILDER)
    build = new_build(client, "delete")
    dataset_id = marked_dataset(live, build, "data")
    made = extension_chart(build, spec(Family.FUNNEL, dataset_id, dimensions=("status",)), "stages")
    chart_id = made["chart_id"]
    deleted_at = "SELECT deleted_at FROM tables WHERE id = %s"
    assert call(client, "delete_dataset", dataset_id=dataset_id, dashboard_id=build.dashboard_id) == {
        "in_use": True, "chart_ids": [chart_id]}
    assert metadata(deleted_at, (dataset_id,)) == [(None,)]
    removed = client.call("delete_chart", {"identifier": chart_id})
    assert removed.get("success") is not False, removed.get("error")
    assert call(client, "delete_dataset", dataset_id=dataset_id, dashboard_id=build.dashboard_id) == {
        "ok": True, "soft_deleted": True}
    assert metadata(deleted_at, (dataset_id,))[0][0] is not None
    assert call(client, "delete_dataset", dataset_id=dataset_id, dashboard_id=build.dashboard_id) == {
        "not_found": "dataset"}


def test_delete_dataset_refuses_without_the_marker(live, extension):
    """A dataset with no description, and one whose marker is pasted on a middle line of its description, each answer
    `delete_dataset` with `{refused: "no_marker"}`, and each stays."""
    client = as_user(live, BUILDER)
    build = new_build(client, "no marker")
    bare = virtual_dataset(live, client, None)
    pasted = virtual_dataset(live, client, f"Made by hand.\n{build.marker('data')}\nStill by hand.")
    for dataset_id in (bare, pasted):
        assert call(client, "delete_dataset", dataset_id=dataset_id, dashboard_id=build.dashboard_id) == {
            "refused": "no_marker"}
        assert metadata("SELECT deleted_at FROM tables WHERE id = %s", (dataset_id,)) == [(None,)]


def test_delete_dataset_refuses_another_dashboards_marker(live, extension):
    """Dashboard 12 asks to delete a dataset marked for dashboard 123, and dashboard 123 one marked for 12: each answers
    `{refused: "another_dashboard"}` (the ids compared as integers, never as text), as does one carrying a dashboard
    shell's `new` marker. Each dataset stays."""
    client = as_user(live, BUILDER)
    build_id = str(uuid.uuid4())
    cases = [(format_marker(build_id, "data", 123), 12), (format_marker(build_id, "data", 12), 123),
             (format_marker(build_id, "data", None), 12)]
    for marker, asking in cases:
        dataset_id = virtual_dataset(live, client, marker)
        assert call(client, "delete_dataset", dataset_id=dataset_id, dashboard_id=asking) == {
            "refused": "another_dashboard"}
        assert metadata("SELECT deleted_at FROM tables WHERE id = %s", (dataset_id,)) == [(None,)]


# --- who may call them ----------------------------------------------------------------------------------------------


def test_tools_need_the_write_scope(live, extension):
    """Analyst (Alpha, who holds `can_write` on Chart, Dataset and Dashboard) with an ask token, which carries no
    Superset write scope: `tools/list` shows none of the seven write tools, and calling `create_dashboard` is refused
    by `mcp_auth_hook` (`SupersetPermissionDenied`), creating nothing. The control: with a build token `tools/list`
    shows all seven."""
    ask_token = live.token(BUILDER, scope=ASK)
    assert not {TOOL_PREFIX + base for base in WRITE_TOOLS} & listed_tools(live, ask_token)
    assert {TOOL_PREFIX + base for base in WRITE_TOOLS} <= listed_tools(live, live.token(BUILDER, scope=BUILD))
    title = f"ask scope {uuid.uuid4().hex[:8]}"
    with pytest.raises(SupersetPermissionDenied):
        SupersetMcp(live.mcp_url, DelegatedToken(ask_token)).call(
            TOOL_PREFIX + "create_dashboard", {"title": title, "marker": format_marker(str(uuid.uuid4()), "d", None)})
    assert metadata("SELECT count(*) FROM dashboards WHERE dashboard_title = %s", (title,)) == [(0,)]


def test_gamma_sees_only_what_its_roles_allow(live, extension, policy_id):
    """The tools are gated by stock class permissions, so Gamma gets from them what its roles already give
    it through MCP's own tools: rc2's Gamma holds `can_write` on Chart and Dashboard, as for `generate_chart` and
    `generate_dashboard`, and only reads Dataset (`GAMMA_READ_ONLY_MODEL_VIEWS`, `superset/security/manager.py`).

    With a build token, `gamma_only` (Gamma) and `policy_only` (Gamma, sql_lab and the `policy` dataset) each list
    exactly the four chart and dashboard tools and none of the three dataset tools. Each tool Gamma sees still refuses
    it what its roles do not reach:
    - `create_chart` on `policy`, which `gamma_only` may not read, for a dashboard `gamma_only` made, answers
      `{not_found: "dataset"}` and saves no chart;
    - analyst's dashboard, published and without charts, is one Gamma may read and does not edit: `set_layout` as
      `gamma_only`, and `create_chart` of a chart on `policy` for it as `policy_only` (who may read `policy`), each
      answer `{not_editor: true, editors: [analyst's label]}`; no such chart is saved, and the dashboard is as it was;
    - a dataset tool, which Gamma does not list, is refused when called anyway (`SupersetPermissionDenied`).

    These are Superset's guards. mnemiq's own door refuses a non-builder (`NOT_A_BUILDER`: building needs Alpha and
    sql_lab) before it calls any tool."""
    chart_and_dashboard = {"create_dashboard", "create_chart", "patch_chart", "set_layout"}
    gamma = as_user(live, "gamma_only")
    reader = as_user(live, "policy_only")
    for username in ("gamma_only", "policy_only"):
        listed = {name.removeprefix(TOOL_PREFIX) for name in listed_tools(live, live.token(username, scope=BUILD))}
        assert listed & set(WRITE_TOOLS) == chart_and_dashboard, (username, sorted(listed & set(WRITE_TOOLS)))
    analyst = as_user(live, BUILDER)
    build = new_build(analyst, "published")
    published = analyst.call("update_dashboard", {"identifier": build.dashboard_id, "published": True})
    assert published.get("success") is not False, published.get("error")
    (before,) = metadata("SELECT published, changed_on, position_json, json_metadata FROM dashboards WHERE id = %s",
                         (build.dashboard_id,))
    assert before[0] is True
    refused = {"not_editor": True, "editors": [label_of(BUILDER)]}
    gamma_build = new_build(gamma, "gamma's")
    funnel = spec(Family.FUNNEL, policy_id, dimensions=("status",))
    assert extension_chart(gamma_build, funnel, "stages") == {"not_found": "dataset"}
    assert metadata("SELECT count(*) FROM slices WHERE slice_name = %s", (funnel.title,)) == [(0,)]
    assert call(gamma, "set_layout", dashboard_id=build.dashboard_id, positions=positions([]),
                expect_changed_on=changed_on(gamma, "dashboard", build.dashboard_id)) == refused
    kpi = spec(Family.BIG_NUMBER, policy_id)
    assert call(reader, "create_chart", **config_request(build, kpi, build.marker("kpi"))) == refused
    assert metadata("SELECT count(*) FROM slices WHERE slice_name = %s", (kpi.title,)) == [(0,)]
    assert metadata("SELECT published, changed_on, position_json, json_metadata FROM dashboards WHERE id = %s",
                    (build.dashboard_id,)) == [before]
    assert members(build.dashboard_id) == []
    with pytest.raises(SupersetPermissionDenied):
        call(gamma, "add_saved_metric", dataset_id=policy_id, metric=_metric("gamma_metric"),
             expect_changed_on="2026-01-01T00:00:00")
    observe("gamma", listed=sorted(chart_and_dashboard), editors=refused["editors"])


def test_set_layout_refuses_a_chart_the_caller_cannot_read(live, extension, policy_id):
    """`set_dash_metadata` makes every chart a layout names a member with no access check of its own, so `set_layout`
    asks first (`can_access_chart`). `gamma_only`, on a dashboard it made and edits, adding analyst's chart on `policy`
    (a dataset `gamma_only` may not read) answers exactly `{refused: "chart_access", chart_ids: [that chart]}`, and the
    dashboard has no member and the layout it had. The control: analyst adding the same chart to a dashboard of its
    own answers `{ok}`, so the refusal is the caller's access, not the chart."""
    analyst = as_user(live, BUILDER)
    own = new_build(analyst, "analyst's layout")
    chart_id = made_chart(own, spec(Family.BIG_NUMBER, policy_id), "kpi")
    gamma = as_user(live, "gamma_only")
    gamma_build = new_build(gamma, "gamma's layout")
    (before,) = metadata("SELECT position_json, json_metadata FROM dashboards WHERE id = %s",
                         (gamma_build.dashboard_id,))
    answer = call(gamma, "set_layout", dashboard_id=gamma_build.dashboard_id, positions=positions([chart_id]),
                  add_chart_ids=[chart_id], expect_changed_on=changed_on(gamma, "dashboard", gamma_build.dashboard_id))
    assert answer == {"refused": "chart_access", "chart_ids": [chart_id]}
    assert members(gamma_build.dashboard_id) == []
    assert metadata("SELECT position_json, json_metadata FROM dashboards WHERE id = %s",
                    (gamma_build.dashboard_id,)) == [before]
    assert call(analyst, "set_layout", dashboard_id=own.dashboard_id, positions=positions([chart_id]),
                add_chart_ids=[chart_id], expect_changed_on=changed_on(analyst, "dashboard", own.dashboard_id)) == {
        "ok": True}
    assert members(own.dashboard_id) == [chart_id]
    observe("chart_access", chart_id=chart_id)


# --- the read tools -------------------------------------------------------------------------------------------------

# Superset's MCP response size guard (`superset/mcp_service/middleware.py`) blocks an answer over this many bytes.
SIZE_GUARD = 50_000


def write_metadata(sql: str, params: tuple = ()) -> None:
    """A write the suite makes to Superset's metadata database directly, only to set up what Superset's own API does not
    set (a metric's certification, an oversized layout): never by mnemiq or the extension."""
    import psycopg

    uri = required("MNEMIQ_TEST_SUPERSET_METADATA_URI")["MNEMIQ_TEST_SUPERSET_METADATA_URI"]
    with psycopg.connect(uri, connect_timeout=10) as con:
        con.execute(sql, params)


def test_dataset_metrics_answers_an_empty_request(live, extension):
    """The probe contract mnemiq's client relies on: `dataset_ids: []` answers exactly `{datasets: []}` and does not
    fail, as every role that may call the tool (analyst, Alpha; `gamma_only`, Gamma)."""
    for username in (BUILDER, "gamma_only"):
        assert call(as_user(live, username), "dataset_metrics", dataset_ids=[]) == {"datasets": []}


def test_dataset_metrics_reads_currency_and_certification(live, extension):
    """A dataset made with a metric of its own (`total_premium`, with a format and a warning: `_seeded_dataset`), then
    `avg_premium` added with a verbose name, a `,.2f` format, a USD currency and a description, and a certification
    written on `avg_premium` (`extra`: certified by `Finance`, `audited`) the way Superset's own dataset editor stores
    it. `dataset_metrics` for that dataset answers one entry `{dataset_id, metrics}` whose metrics, in the dataset's
    order, carry exactly the seven fields `metric_name, expression, d3format, currency, certified_by,
    certification_details, warning_text`: `avg_premium` with currency the dict `{symbol: USD, symbolPosition: prefix}`
    (rc2's `Currency` object, as stored), `,.2f`, `Finance` and `audited`; `total_premium` with its format and
    warning, no currency (`null`) and no certification (`null`, `null`). Each value equals what the metadata database
    stores."""
    client = as_user(live, BUILDER)
    build = new_build(client, "metric details")
    dataset_id = _seeded_dataset(live, build, "metrics", [{
        "metric_name": "total_premium", "expression": "SUM(premium)", "d3format": ",.0f",
        "warning_text": "counts every row"}])
    assert call(client, "add_saved_metric", dataset_id=dataset_id, metric=_metric(
        "avg_premium", verbose_name="Average premium", d3format=",.2f",
        currency={"symbol": "USD", "symbolPosition": "prefix"}, description="per policy"),
        expect_changed_on=changed_on(client, "dataset", dataset_id)) == {"ok": True}
    write_metadata("UPDATE sql_metrics SET extra = %s WHERE table_id = %s AND metric_name = 'avg_premium'",
                   (json.dumps({"certification": {"certified_by": "Finance", "details": "audited"}}), dataset_id))
    answer = call(client, "dataset_metrics", dataset_ids=[dataset_id])
    assert set(answer) == {"datasets"} and [d["dataset_id"] for d in answer["datasets"]] == [dataset_id], answer
    (entry,) = answer["datasets"]
    assert set(entry) == {"dataset_id", "metrics"}
    fields = {"metric_name", "expression", "d3format", "currency", "certified_by", "certification_details",
              "warning_text"}
    assert all(set(metric) == fields for metric in entry["metrics"]), entry["metrics"]
    by_name = {metric["metric_name"]: metric for metric in entry["metrics"]}
    # `create_virtual_dataset` adds Superset's own `count` metric to every dataset.
    assert set(by_name) == {"total_premium", "avg_premium", "count"}
    assert by_name["avg_premium"] == {
        "metric_name": "avg_premium", "expression": "AVG(premium)", "d3format": ",.2f",
        "currency": {"symbol": "USD", "symbolPosition": "prefix"}, "certified_by": "Finance",
        "certification_details": "audited", "warning_text": None}
    assert by_name["total_premium"] == {
        "metric_name": "total_premium", "expression": "SUM(premium)", "d3format": ",.0f", "currency": None,
        "certified_by": None, "certification_details": None, "warning_text": "counts every row"}
    stored = stored_metrics(dataset_id)
    assert json.loads(stored["avg_premium"][9]) == by_name["avg_premium"]["currency"]
    assert json.loads(stored["total_premium"][9] or "null") is None  # a stored JSON null
    observe("dataset_metrics", metrics=sorted(by_name), currency=by_name["avg_premium"]["currency"])


def test_dataset_metrics_leaves_out_an_unreachable_dataset_without_naming_it(live, extension, policy_id):
    """`policy_only` (Gamma and sql_lab, granted the `policy` dataset alone) asks for analyst's virtual dataset, a
    dataset id that does not exist, and `policy`. It is answered `policy`'s entry only, and that answer is identical to
    the one for `policy` alone: neither the unreachable dataset's id nor its name appears, and a dataset that exists but
    is out of reach reads as one that does not. `gamma_only`, which reaches no dataset, is answered `{datasets: []}`
    for all three."""
    analyst = as_user(live, BUILDER)
    build = new_build(analyst, "unreachable")
    hidden = marked_dataset(live, build, "hidden")
    (name,) = metadata("SELECT table_name FROM tables WHERE id = %s", (hidden,))[0]
    missing = 2_000_000_000
    reader = as_user(live, "policy_only")
    alone = call(reader, "dataset_metrics", dataset_ids=[policy_id])
    asked = call(reader, "dataset_metrics", dataset_ids=[hidden, missing, policy_id])
    assert [d["dataset_id"] for d in alone["datasets"]] == [policy_id]
    assert asked == alone
    assert name not in json.dumps(asked) and str(hidden) not in json.dumps(asked)
    assert call(analyst, "dataset_metrics", dataset_ids=[hidden])["datasets"][0]["dataset_id"] == hidden
    assert call(as_user(live, "gamma_only"), "dataset_metrics", dataset_ids=[hidden, missing, policy_id]) == {
        "datasets": []}
    observe("unreachable", answered=[d["dataset_id"] for d in asked["datasets"]])


def test_row_rules_says_only_whether_a_rule_applies_to_the_caller(live, extension, policy_id):
    """`policy` carries west's and east's Regular rules (`setup_superset.py`). Asked by `west`, `row_rules` answers that
    a rule applies on `policy`; asked by `analyst`, whom no rule narrows, that none does; the answer is exactly
    `{datasets: [{dataset_id, applies}]}`, never a rule's clause or name. A dataset out of the caller's reach and one
    that does not exist are both left out: `gamma_only` is answered `{datasets: []}`."""
    missing = 2_000_000_000
    west = call(as_user(live, "west"), "row_rules", dataset_ids=[policy_id, missing])
    analyst = call(as_user(live, BUILDER), "row_rules", dataset_ids=[policy_id])
    assert west == {"datasets": [{"dataset_id": policy_id, "applies": True}]}, west
    assert analyst == {"datasets": [{"dataset_id": policy_id, "applies": False}]}, analyst
    assert "west" not in json.dumps(west) and "region" not in json.dumps(west)
    assert call(as_user(live, "gamma_only"), "row_rules", dataset_ids=[policy_id, missing]) == {"datasets": []}
    observe("row_rules", west=west, analyst=analyst)


def test_table_rules_answers_as_supersets_own_lookup_by_table(live, extension, made):
    """`table_rules` answers, as the caller and by table, whether Superset's own lookup of the rules a statement reading
    that table takes (rc2's `get_predicates_for_table`) finds one. Four tables of the test's own in the source, each
    holding one row: `nulled`, whose one dataset has no schema, which Superset reads as the database's default schema
    (`public`); `cased`, whose one dataset's schema is `PUBLIC`; `hidden`, whose one dataset `policy_only` may not read
    or list; and `free`, with no dataset. Each of the three datasets carries a Base rule that keeps no row and applies
    to everyone (`setup_superset.py rule-on`). The datasets are made by the extension's `create_dataset` as analyst;
    the null schema and the case-variant one are then set in the metadata database, since Superset's create holds a
    schema to the database's own.

    Asked about `policy` and the four in `public`, analyst is answered, in the order asked and each exactly
    `{table, applies}`: `policy` false (west's and east's rules are not analyst's), `nulled`, `cased` and `hidden`
    true, `free` false. west is answered `policy` true; `policy_only`, whose `dataset_metrics` of `hidden`'s dataset
    answers nothing, `hidden` true and `free` false. Superset agrees when it runs a statement: as analyst, `COUNT(*)`
    through `execute_sql` reads 0 rows of each of the three ruled tables and 1 of `free`. `gamma_only`, who may see no
    table of the database, and a database that does not exist, are answered `{not_found: "database"}`."""
    import psycopg
    from psycopg import sql

    analyst = as_user(live, BUILDER)
    build = made.build(analyst, "table rules")
    suffix = uuid.uuid4().hex[:8]
    tables = {kind: f"live_rules_{kind}_{suffix}" for kind in ("nulled", "cased", "hidden", "free")}
    made.tables.extend(tables.values())
    with psycopg.connect(live.pg_dsn, autocommit=True, connect_timeout=10) as con:
        for name in tables.values():
            con.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY)").format(sql.Identifier("public", name)))
            con.execute(sql.SQL("INSERT INTO {} VALUES (1)").format(sql.Identifier("public", name)))
    datasets = {}
    for kind in ("nulled", "cased", "hidden"):
        answer = made.dataset(call(analyst, "create_dataset", database_id=live.database_id, schema="public",
                                   table_name=tables[kind], dashboard_id=build.dashboard_id,
                                   marker=build.marker(kind)), build.dashboard_id)
        assert set(answer) == {"dataset_id"}, (kind, answer)
        datasets[kind] = answer["dataset_id"]
    rules: list[str] = []
    try:
        write_metadata("UPDATE tables SET schema = NULL WHERE id = %s", (datasets["nulled"],))
        write_metadata("UPDATE tables SET schema = 'PUBLIC' WHERE id = %s", (datasets["cased"],))
        for kind in ("nulled", "cased", "hidden"):
            said = run_setup("rule-on", str(datasets[kind]))
            assert said.startswith("rule "), said
            rules.append(said.split()[-1])
        asked = ["policy", *tables.values()]
        answer = call(analyst, "table_rules", database_id=live.database_id, schema="public", tables=asked)
        assert answer == {"tables": [{"table": name, "applies": applies} for name, applies in zip(
            asked, (False, True, True, True, False))]}, answer
        west = call(as_user(live, "west"), "table_rules", database_id=live.database_id, schema="public",
                    tables=["policy", tables["free"]])
        assert west == {"tables": [{"table": "policy", "applies": True},
                                   {"table": tables["free"], "applies": False}]}, west
        reader = as_user(live, "policy_only")
        assert call(reader, "dataset_metrics", dataset_ids=[datasets["hidden"]]) == {"datasets": []}
        hidden = call(reader, "table_rules", database_id=live.database_id, schema="public",
                      tables=[tables["hidden"], tables["free"]])
        assert hidden == {"tables": [{"table": tables["hidden"], "applies": True},
                                     {"table": tables["free"], "applies": False}]}, hidden
        source = SupersetSqlAdapter(as_user(live, BUILDER, ASK), database_id=live.database_id, schema="public")
        counted = {kind: source.execute_arrow(f'SELECT COUNT(*) AS n FROM "{name}"').column(0)[0].as_py()
                   for kind, name in tables.items()}
        assert counted == {"nulled": 0, "cased": 0, "hidden": 0, "free": 1}, counted
        for client, database_id in ((as_user(live, "gamma_only"), live.database_id), (analyst, 2_000_000_000)):
            assert call(client, "table_rules", database_id=database_id, schema="public",
                        tables=[tables["free"]]) == {"not_found": "database"}
        observe("table_rules", analyst=answer, policy_only=hidden, counted=counted)
    finally:
        for rule in rules:
            run_setup("rule-off", rule)
        write_metadata("UPDATE tables SET schema = 'public' WHERE id = ANY(%s)",
                       ([datasets["nulled"], datasets["cased"]],))


def test_session_binding_refuses_a_database_that_is_not_oracle_and_finds_none_the_caller_cannot_see(live, extension):
    """`session_binding` finds the database through Superset's own filter of the databases the caller may see, and runs
    its statement only on an Oracle one. The live suite's database is PostgreSQL: analyst, who may see it, is answered
    `{refused: "backend"}`, and so is `policy_only`, granted one dataset and not the database, whom no table access
    check stops short of the tool's own; `gamma_only`, who may see no table of it, and a database that does not exist
    are answered `{not_found: "database"}`; a request naming a statement is refused as a request. mnemiq's reading of
    each is None, which it reads as not bound. (On an Oracle database the tool's answer is the live Oracle run's: this
    Superset has none.)"""
    from mnemiq.superset.extension import session_binding

    asked = {"database_id": live.database_id, "service": "freepdb1", "schema": "PUBLIC"}
    analyst = as_user(live, BUILDER, ASK)
    assert call(analyst, "session_binding", **asked) == {"refused": "backend"}
    assert call(as_user(live, "policy_only", ASK), "session_binding", **asked) == {"refused": "backend"}
    assert call(as_user(live, "gamma_only", ASK), "session_binding", **asked) == {"not_found": "database"}
    assert call(analyst, "session_binding", **{**asked, "database_id": 2_000_000_000}) == {"not_found": "database"}
    assert call(analyst, "session_binding", **asked, sql="SELECT 1") == {"refused": "request", "unknown": ["sql"]}
    assert session_binding(analyst, **asked) is None
    observe("session_binding", analyst=call(analyst, "session_binding", **asked))


def test_editors_of_names_editors_of_readable_objects_only(live, extension, policy_id):
    """analyst's chart and dataset, each made by the extension for a build's dashboard (editors: the dashboard's,
    analyst), are answered to analyst as `{charts: [{chart_id, is_editor: true, editors: [analyst's label]}],
    datasets: [{dataset_id, is_editor: true, editors: [analyst's label]}]}`, and to west (Alpha, who reads both) the
    same with `is_editor: false`; a chart and a dataset id that do not exist are left out. `gamma_only`, which reads neither, is answered
    `{charts: [], datasets: []}`, with no id and no label."""
    analyst = as_user(live, BUILDER)
    build = new_build(analyst, "editors of")
    chart_id = made_chart(build, spec(Family.BIG_NUMBER, policy_id), "kpi")
    dataset_id = marked_dataset(live, build, "data")
    missing = 2_000_000_000
    answer = call(analyst, "editors_of", chart_ids=[chart_id, missing], dataset_ids=[dataset_id, missing])
    assert answer == {"charts": [{"chart_id": chart_id, "is_editor": True, "editors": [label_of(BUILDER)]}],
                      "datasets": [{"dataset_id": dataset_id, "is_editor": True, "editors": [label_of(BUILDER)]}]}, \
        answer
    # west (Alpha) reads both and edits neither, and is told so beside the same editors.
    seen = call(as_user(live, "west"), "editors_of", chart_ids=[chart_id], dataset_ids=[dataset_id])
    assert seen == {"charts": [{"chart_id": chart_id, "is_editor": False, "editors": [label_of(BUILDER)]}],
                    "datasets": [{"dataset_id": dataset_id, "is_editor": False, "editors": [label_of(BUILDER)]}]}, seen
    hidden = call(as_user(live, "gamma_only"), "editors_of", chart_ids=[chart_id], dataset_ids=[dataset_id])
    assert hidden == {"charts": [], "datasets": []}, hidden
    observe("editors_of", editors=[label_of(BUILDER)])


def test_get_layout_round_trips_through_set_layout(live, extension, policy_id):
    """After a layout of two charts is saved, `get_layout` answers `{dashboard_id, position_json, changed_on}`:
    `position_json` the stored text exactly (equal to the metadata database's), `changed_on` the very text MCP's
    `get_dashboard_info` answers. Sending the layout it read back through `set_layout`, with that `changed_on`,
    answers `{ok: true}`, the dashboard's charts are as they were and `get_layout` answers the same layout. A layout
    changed (chart `one` made wider) and sent back with the `changed_on` just read answers `{ok: true}`; `get_layout`
    then answers the wider chart and a `changed_on` that is not the one sent. A dashboard that does not exist is
    answered `{not_found: "dashboard"}`."""
    client = as_user(live, BUILDER)
    build = new_build(client, "layout read")
    one, two = (made_chart(build, spec(Family.BIG_NUMBER, policy_id), f"kpi_{i}") for i in range(2))
    assert _layout(build, [one, two], add=[one, two]) == {"ok": True}
    answer = call(client, "get_layout", dashboard_id=build.dashboard_id)
    assert set(answer) == {"dashboard_id", "position_json", "changed_on"} and answer["dashboard_id"] == build.dashboard_id
    ((stored,),) = metadata("SELECT position_json FROM dashboards WHERE id = %s", (build.dashboard_id,))
    assert answer["position_json"] == stored
    assert answer["changed_on"] == changed_on(client, "dashboard", build.dashboard_id)
    layout = json.loads(answer["position_json"])
    assert {node["meta"]["chartId"] for node in layout.values() if isinstance(node, dict) and node.get("type") == "CHART"
            } == {one, two}
    assert call(client, "set_layout", dashboard_id=build.dashboard_id, positions=layout,
                expect_changed_on=answer["changed_on"]) == {"ok": True}
    assert members(build.dashboard_id) == sorted([one, two])
    again = call(client, "get_layout", dashboard_id=build.dashboard_id)
    assert json.loads(again["position_json"]) == layout
    layout[f"CHART-{one}"]["meta"]["width"] = 8
    assert call(client, "set_layout", dashboard_id=build.dashboard_id, positions=layout,
                expect_changed_on=again["changed_on"]) == {"ok": True}
    wider = call(client, "get_layout", dashboard_id=build.dashboard_id)
    assert json.loads(wider["position_json"])[f"CHART-{one}"]["meta"]["width"] == 8
    assert wider["changed_on"] != again["changed_on"]
    assert call(client, "get_layout", dashboard_id=2_000_000_000) == {"not_found": "dashboard"}
    observe("layout_round_trip", bytes=len(answer["position_json"]))


def test_answers_stay_under_the_size_guard(live, extension, policy_id):
    """A 20-dataset request (analyst's twenty virtual datasets, three saved metrics each and Superset's own `count`) is answered all twenty,
    well under Superset's 50,000-byte response guard. `get_layout` of a dashboard whose stored layout is made over 40 KB
    (set directly) is answered exactly `{too_large: true}` and not an error; the same dashboard with its layout cut
    back answers its layout again, also under the guard."""
    client = as_user(live, BUILDER)
    build = new_build(client, "size")
    metrics = [{"metric_name": f"m{i}", "expression": "SUM(premium)", "d3format": ",.2f"} for i in range(3)]
    ids = [virtual_dataset(live, client, build.marker(f"d{i}"), metrics=metrics) for i in range(20)]
    answer = call(client, "dataset_metrics", dataset_ids=ids)
    size = len(json.dumps(answer).encode())
    assert [d["dataset_id"] for d in answer["datasets"]] == ids and all(len(d["metrics"]) == 4 for d in answer["datasets"])
    assert size < SIZE_GUARD, size
    assert call(client, "get_layout", dashboard_id=build.dashboard_id)["dashboard_id"] == build.dashboard_id
    small = json.dumps({"ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": []}})
    big = json.dumps({"ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": []}, "PAD": "x" * 41_000})
    write_metadata("UPDATE dashboards SET position_json = %s WHERE id = %s", (big, build.dashboard_id))
    assert call(client, "get_layout", dashboard_id=build.dashboard_id) == {"too_large": True}
    write_metadata("UPDATE dashboards SET position_json = %s WHERE id = %s", (small, build.dashboard_id))
    layout = call(client, "get_layout", dashboard_id=build.dashboard_id)
    assert layout["position_json"] == small and len(json.dumps(layout).encode()) < SIZE_GUARD
    observe("sizes", dataset_metrics_bytes=size)


def test_mnemiqs_client_finds_the_extension_and_a_superset_without_it_is_missing(live, extension, policy_id):
    """mnemiq's own `call_tool`, against the running MCP service, answers `dataset_metrics` and does not raise
    `ExtensionMissing`, with an ask token (the read tools need `superset:dataset:read`, which an ask token carries);
    its probe of the same tool is that very answer. Against a second MCP process started here
    on Superset's configuration without the extension (the same Superset, the same keys, no extension directory),
    the same call raises `ExtensionMissing`, naming mnemiq's own sentence and not Superset's text."""
    from mnemiq.superset.extension import ExtensionMissing, call_tool

    answer = call_tool(as_user(live, BUILDER, ASK), "dataset_metrics", {"dataset_ids": [policy_id]})
    assert [d["dataset_id"] for d in answer["datasets"]] == [policy_id]
    with _mcp_without_the_extension(live) as url:
        bare = SupersetMcp(url, DelegatedToken(live.token(BUILDER, scope=BUILD)), timeout_s=90)
        with pytest.raises(ExtensionMissing) as caught:
            call_tool(bare, "dataset_metrics", {"dataset_ids": [policy_id]})
        assert "Superset does not offer mnemiq's extension tool" in str(caught.value)
        observe("missing_extension", message=str(caught.value))


@contextlib.contextmanager
def _mcp_without_the_extension(live, port: int = 15010):
    """A throwaway MCP process on `port`: the suite's Superset configuration with no extension directory, so it
    serves Superset's own tools alone. It shares the metadata database and keys with the running services and changes
    nothing in them."""
    env = required("MNEMIQ_TEST_SUPERSET_PYTHON", "MNEMIQ_TEST_SUPERSET_KEYS_DIR", "MNEMIQ_TEST_SUPERSET_METADATA_URI",
                   "MNEMIQ_TEST_SUPERSET_PG_DSN")
    child_env = {**os.environ, "SUPERSET_CONFIG_PATH": str(Path(__file__).with_name("superset_config.py")),
                 "FLASK_APP": "superset"}
    child_env.pop("MNEMIQ_TEST_SUPERSET_EXTENSION_DIR", None)
    superset = Path(env["MNEMIQ_TEST_SUPERSET_PYTHON"]).with_name("superset")
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([str(superset), "mcp", "run", "--host", "127.0.0.1", "--port", str(port)],
                                   env=child_env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                log.seek(0)
                if b"Application startup complete" in log.read():
                    break
                assert process.poll() is None, "the extension-less MCP process exited"
                assert time.monotonic() < deadline, "the extension-less MCP process did not start"
                time.sleep(1)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            process.terminate()
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()

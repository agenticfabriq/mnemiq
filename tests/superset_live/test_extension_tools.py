"""mnemiq's Superset extension, live: its token route on Superset's web server, and its write tools on Superset's MCP
service, in the suite's Superset 7.0.0rc2 with the extension loaded (`setup_superset.py extension`).

Every tool is called over MCP as a person whose token Superset verifies, with the scopes a build token carries, and
every chart is made as mnemiq makes it: MCP's `generate_chart` with mnemiq's own `generate_chart_request`, or the
extension's `create_chart` with `extension_chart_request`. Each test makes the dashboards, datasets and charts it
needs on the seeded `policy` table, so none depends on another. What a tool wrote is read back from Superset's
metadata database directly (`metadata`), and every number a chart answers with is held to the source read directly
(`ground_truth`).

Each test states its expectation in its docstring. Observations a run should keep are printed as `OBSERVED` lines,
which `pytest -rA` shows for passing tests too.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import jwt
import pytest

from mnemiq.superset.charts import ChartSpec, Family, WrittenRef, extension_chart_request, generate_chart_request
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetPermissionDenied
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.marker import format_marker
from mnemiq.superset.measures import Measure, MeasureKind
from tests.superset_extension_path import use_extension_source
from tests.superset_live.live import (
    TENANT,
    TOKEN_ROUTE,
    fingerprint_of,
    ground_truth,
    metadata,
    required,
    run_setup,
)

use_extension_source()

from agenticfabriq.mnemiq.token import ASK_SCOPES, BUILD_SCOPES  # noqa: E402

pytestmark = pytest.mark.integration

BUILD = " ".join(BUILD_SCOPES)
ASK = " ".join(ASK_SCOPES)
WRITE_TOOLS = ("create_dashboard", "claim_chart", "claim_dataset", "create_chart", "patch_chart", "add_saved_metric",
               "delete_dataset", "set_layout")
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
    missing = [base for base in WRITE_TOOLS if TOOL_PREFIX + base not in listed]
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
    """A chart MCP's `generate_chart` saves from mnemiq's own request, as a build makes one."""
    answer = client.call("generate_chart", generate_chart_request(chart, {}, {}, currency="USD"))
    assert answer.get("success") and (answer.get("chart") or {}).get("id"), {k: answer.get(k) for k in
                                                                           ("success", "error")}
    return answer["chart"]["id"]


def extension_chart(build: Build, chart: ChartSpec, step: str = "chart") -> dict:
    """The extension's `create_chart` answer for mnemiq's own request for a funnel or a sankey."""
    request = extension_chart_request(chart, {}, {}, currency="USD")
    return call(build.client, "create_chart", **request, dashboard_id=build.dashboard_id, marker=build.marker(step))


def virtual_dataset(live, client: SupersetMcp, description: str | None, **more) -> int:
    """A virtual dataset over `policy` made with MCP's `create_virtual_dataset`, as a build makes one."""
    request = {"database_id": live.database_id, "sql": POLICY_COLUMNS, "schema_name": "public",
               "dataset_name": f"mnemiq_live_{uuid.uuid4().hex[:10]}", **more}
    if description is not None:
        request["description"] = description
    answer = client.call("create_virtual_dataset", request)
    assert answer.get("id"), answer.get("error")
    return answer["id"]


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
    """A monthly line chart made by `generate_chart` (COUNT(*) by `issued`, grain P1M) and claimed by the extension
    answers `get_chart_data` with one row per month: 12 rows, each month's count the source's. The claim also gives
    the chart the dashboard's editors and the build's marker as its description. The control: before the claim the
    chart has no saved query context, and rc2 rebuilds its query from `form_data` without the grain, so it does not
    answer the twelve months."""
    client = as_user(live, BUILDER)
    build = new_build(client, "grain")
    chart_id = mcp_chart(client, spec(Family.LINE, policy_id, time_column="issued", time_grain="P1M"))
    truth = [count for _, count in ground_truth(live, MONTHLY)]
    assert len(truth) == 12
    before = chart_data(client, chart_id)
    before_counts = [row.get("policies") for row in sorted(before.get("data") or [], key=lambda r: r["issued"])]
    assert stored_chart(chart_id)["query_context"] is None
    assert before_counts != truth
    assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                marker=build.marker("by_month")) == {"ok": True}
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
    """A pivot (rows region, columns status, COUNT(*)) made by `generate_chart` and claimed answers `get_chart_data`
    with one row per region and status, nine in all, each count the source's, and no totals row. A waterfall
    (x-axis `issued` by month, broken down by status) answers one row per month and status, each count the source's:
    its x-axis kept and its grain applied. Before the claim, each answers otherwise (rc2's rebuild from `form_data`
    reads neither the pivot's rows and columns nor the waterfall's x-axis); what each answered is observed."""
    client = as_user(live, BUILDER)
    build = new_build(client, "grouping")
    pivot = mcp_chart(client, spec(Family.PIVOT, policy_id, dimensions=("region", "status")))
    waterfall = mcp_chart(client, spec(Family.WATERFALL, policy_id, time_column="issued", time_grain="P1M",
                                       dimensions=("status",)))
    before = {name: chart_data(client, chart_id) for name, chart_id in (("pivot", pivot), ("waterfall", waterfall))}
    for name, chart_id in (("pivot", pivot), ("waterfall", waterfall)):
        assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                    marker=build.marker(name)) == {"ok": True}
    pivot_rows = sorted((r["region"], r["status"], r["policies"]) for r in chart_data(client, pivot)["data"])
    assert pivot_rows == sorted(ground_truth(live, BY_REGION_AND_STATUS))
    assert len(pivot_rows) == 9
    waterfall_rows = [(r["issued"], r["status"], r["policies"]) for r in chart_data(client, waterfall)["data"]]
    assert ranked(waterfall_rows) == ranked(ground_truth(live, MONTHLY_BY_STATUS))
    observe("grouping", pivot_before_rows=len(before["pivot"].get("data") or []),
            pivot_before_error=before["pivot"].get("error_type"),
            waterfall_before_rows=len(before["waterfall"].get("data") or []),
            waterfall_before_columns=sorted((before["waterfall"].get("data") or [{}])[0]),
            waterfall_after_rows=len(waterfall_rows))


# --- patch_chart ----------------------------------------------------------------------------------------------------


def _claimed_line(live, client: SupersetMcp, policy_id: int, title: str) -> tuple[Build, int]:
    build = new_build(client, title)
    chart_id = mcp_chart(client, spec(Family.LINE, policy_id, time_column="issued", time_grain="P1M"))
    assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                marker=build.marker("line")) == {"ok": True}
    return build, chart_id


def test_patch_with_stale_changed_on_is_refused(live, extension, policy_id):
    """A patch that sends the `changed_on` `get_chart_info` answered before the claim (since replaced) answers
    exactly `{stale: true}` and changes nothing: the chart's params, query context and `changed_on` are as they were.
    The same patch with the `changed_on` read after the claim answers `{ok: true}` and sets only `row_limit`, every
    other params key as it was, and the query context follows (`row_limit` 77)."""
    client = as_user(live, BUILDER)
    build = new_build(client, "stale")
    chart_id = mcp_chart(client, spec(Family.LINE, policy_id, time_column="issued", time_grain="P1M"))
    old = changed_on(client, "chart", chart_id)
    assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                marker=build.marker("line")) == {"ok": True}
    claimed = stored_chart(chart_id)
    assert call(client, "patch_chart", chart_id=chart_id, set={"row_limit": 77}, expect_changed_on=old) == {
        "stale": True}
    assert stored_chart(chart_id) == claimed
    fresh = changed_on(client, "chart", chart_id)
    assert fresh != old
    assert call(client, "patch_chart", chart_id=chart_id, set={"row_limit": 77}, expect_changed_on=fresh) == {
        "ok": True}
    patched = stored_chart(chart_id)
    assert patched["params"] == {**claimed["params"], "row_limit": 77}
    assert patched["query_context"]["queries"][0]["row_limit"] == 77
    observe("stale", before_claim=old, after_claim=fresh)


def test_simultaneous_patches_lose_nothing(live, extension, policy_id):
    """Two patches of one chart, each setting a different key and both sending the same `changed_on`, start together
    while the test holds the chart's row lock, and the lock is let go once both wait for it. One answers
    `{ok: true}` and the other exactly `{stale: true}`: the second waited for the first one's commit, then saw its
    `changed_on`. The winner's key is set and the loser's is not, and every other params key is as it was."""
    client = as_user(live, BUILDER)
    _, chart_id = _claimed_line(live, client, policy_id, "race")
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


def _claimed_dataset(live, build: Build, step: str, **more) -> int:
    dataset_id = virtual_dataset(live, build.client, build.marker(step), **more)
    assert call(build.client, "claim_dataset", dataset_id=dataset_id, dashboard_id=build.dashboard_id,
                marker=build.marker(step)) == {"ok": True}
    return dataset_id


def _metric(name: str, expression: str = "AVG(premium)", **more) -> dict:
    return {"metric_name": name, "expression": expression, **more}


def test_simultaneous_metric_saves_keep_both_or_refuse_one(live, extension):
    """Two saved metrics added to one dataset, both sending the same `changed_on`, start together while the test holds
    the dataset's row lock, let go once both wait. One answers `{ok: true}` and the other exactly `{stale: true}`. No
    metric is lost: every metric the dataset had is there unchanged, the winner's is there, and the loser's is not."""
    client = as_user(live, BUILDER)
    build = new_build(client, "metric race")
    dataset_id = _claimed_dataset(live, build, "metrics",
                                  metrics=[{"metric_name": "total_premium", "expression": "SUM(premium)"}])
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
    dataset_id = _claimed_dataset(live, build, "metrics", metrics=[{
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
    dataset_id = _claimed_dataset(live, build, "metrics")
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
    one, two, three = (mcp_chart(client, spec(Family.BIG_NUMBER, policy_id)) for _ in range(3))
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
    charts = [mcp_chart(client, spec(Family.BIG_NUMBER, policy_id)) for _ in range(2)]
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
    one, two = (mcp_chart(client, spec(Family.BIG_NUMBER, policy_id)) for _ in range(2))
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


# --- claims, editors and deletes ------------------------------------------------------------------------------------


def test_claim_refuses_another_builds_chart(live, extension, policy_id):
    """A chart build A claimed for its dashboard, claimed again with build B's marker for the same dashboard, answers
    `{claimed_by_another: true}` and keeps build A's marker. Claimed again with build A's own marker it answers
    `{ok: true}`, and its description is still one marker line, A's."""
    client = as_user(live, BUILDER)
    build = new_build(client, "claims")
    chart_id = mcp_chart(client, spec(Family.BIG_NUMBER, policy_id))
    assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                marker=build.marker("kpi")) == {"ok": True}
    other = format_marker(str(uuid.uuid4()), "kpi", build.dashboard_id)
    assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id, marker=other) == {
        "claimed_by_another": True}
    assert stored_chart(chart_id)["description"] == build.marker("kpi")
    assert call(client, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                marker=build.marker("kpi")) == {"ok": True}
    assert stored_chart(chart_id)["description"] == build.marker("kpi")


def test_simultaneous_claims_by_two_builds_leave_one(live, extension, policy_id):
    """Two builds claim one chart for the same dashboard at once, each with its own build's marker, started together
    while the test holds the chart's row lock and let go once both wait. One answers `{ok: true}` and the other exactly
    `{claimed_by_another: true}`: the second waited for the first one's commit, then read its marker under the lock.
    The chart's description is the winner's marker alone."""
    client = as_user(live, BUILDER)
    build = new_build(client, "claim race")
    chart_id = mcp_chart(client, spec(Family.BIG_NUMBER, policy_id))
    markers = [build.marker("kpi"), format_marker(str(uuid.uuid4()), "kpi", build.dashboard_id)]
    answers = race("slices", chart_id, [
        lambda marker=marker: call(as_user(live, BUILDER), "claim_chart", chart_id=chart_id,
                                   dashboard_id=build.dashboard_id, marker=marker)
        for marker in markers])
    assert sorted(answers, key=lambda a: "claimed_by_another" in a) == [{"ok": True}, {"claimed_by_another": True}], (
        answers)
    assert stored_chart(chart_id)["description"] == markers[answers.index({"ok": True})]
    observe("claim_race", answers=answers)


def test_non_editor_gets_editor_labels_not_a_change(live, extension, policy_id):
    """West (Alpha, able to read analyst's chart and dataset, editor of neither) is answered
    `{not_editor: true, editors: [analyst's label]}` when patching analyst's claimed chart, adding a metric to
    analyst's claimed dataset, claiming analyst's chart for west's own dashboard, and deleting analyst's dataset for
    west's dashboard (whose id the dataset's marker does not name: not being an editor is answered first); the chart
    and the dataset are as they were."""
    analyst = as_user(live, BUILDER)
    build = new_build(analyst, "editors")
    chart_id = mcp_chart(analyst, spec(Family.BIG_NUMBER, policy_id))
    assert call(analyst, "claim_chart", chart_id=chart_id, dashboard_id=build.dashboard_id,
                marker=build.marker("kpi")) == {"ok": True}
    dataset_id = _claimed_dataset(live, build, "data")
    chart_before, metrics_before = stored_chart(chart_id), stored_metrics(dataset_id)
    west = as_user(live, "west")
    refused = {"not_editor": True, "editors": [label_of(BUILDER)]}
    assert call(west, "patch_chart", chart_id=chart_id, set={"row_limit": 5},
                expect_changed_on=changed_on(west, "chart", chart_id)) == refused
    assert call(west, "add_saved_metric", dataset_id=dataset_id, metric=_metric("west_metric"),
                expect_changed_on=changed_on(west, "dataset", dataset_id)) == refused
    west_build = new_build(west, "west's")
    assert call(west, "claim_chart", chart_id=chart_id, dashboard_id=west_build.dashboard_id,
                marker=west_build.marker("kpi")) == refused
    assert call(west, "delete_dataset", dataset_id=dataset_id, dashboard_id=west_build.dashboard_id) == refused
    assert stored_chart(chart_id) == chart_before and stored_metrics(dataset_id) == metrics_before
    assert metadata("SELECT deleted_at FROM tables WHERE id = %s", (dataset_id,)) == [(None,)]
    observe("not_editor", editors=refused["editors"])


def test_delete_dataset_refuses_while_a_chart_uses_it(live, extension):
    """A dataset the build made and claimed for its dashboard, with a funnel on it: `delete_dataset` answers
    `{in_use: true, chart_ids: [the funnel]}` and the dataset stays. Once the funnel is deleted (MCP's
    `delete_chart`), `delete_dataset` answers `{ok: true, soft_deleted: true}` and the dataset is in Superset's
    trash (`deleted_at` set); asked again, it answers `{not_found: "dataset"}`."""
    client = as_user(live, BUILDER)
    build = new_build(client, "delete")
    dataset_id = _claimed_dataset(live, build, "data")
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
    Superset write scope: `tools/list` shows none of the eight write tools, and calling `create_dashboard` is refused
    by `mcp_auth_hook` (`SupersetPermissionDenied`), creating nothing. The control: with a build token `tools/list`
    shows all eight."""
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
    exactly the five chart and dashboard tools and none of the three dataset tools. Each tool Gamma sees still refuses
    it what its roles do not reach:
    - `create_chart` on `policy`, which `gamma_only` may not read, for a dashboard `gamma_only` made, answers
      `{not_found: "dataset"}` and saves no chart;
    - analyst's dashboard, published and without charts, is one Gamma may read and does not edit: `set_layout` as
      `gamma_only`, and `claim_chart` of a chart `policy_only` made for it, each answer
      `{not_editor: true, editors: [analyst's label]}`, and the dashboard is as it was;
    - a dataset tool, which Gamma does not list, is refused when called anyway (`SupersetPermissionDenied`).

    These are Superset's guards. mnemiq's own door refuses a non-builder (`NOT_A_BUILDER`: building needs Alpha and
    sql_lab) before it calls any tool."""
    chart_and_dashboard = {"create_dashboard", "claim_chart", "create_chart", "patch_chart", "set_layout"}
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
    own_chart = mcp_chart(reader, spec(Family.BIG_NUMBER, policy_id))
    assert call(reader, "claim_chart", chart_id=own_chart, dashboard_id=build.dashboard_id,
                marker=format_marker(str(uuid.uuid4()), "kpi", build.dashboard_id)) == refused
    assert metadata("SELECT published, changed_on, position_json, json_metadata FROM dashboards WHERE id = %s",
                    (build.dashboard_id,)) == [before]
    assert members(build.dashboard_id) == []
    with pytest.raises(SupersetPermissionDenied):
        call(gamma, "add_saved_metric", dataset_id=policy_id, metric=_metric("gamma_metric"),
             expect_changed_on="2026-01-01T00:00:00")
    observe("gamma", listed=sorted(chart_and_dashboard), editors=refused["editors"])

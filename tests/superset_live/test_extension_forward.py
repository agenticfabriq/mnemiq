"""The extension's forwarder, live: a person signed in to Superset 7.0.0rc2 reaches mnemiq through
`/extensions/agenticfabriq/mnemiq/v1/...` with their own session, and mnemiq answers them as that person.

mnemiq is served for the module's life by uvicorn on the address `MNEMIQ_TEST_SUPERSET_MNEMIQ_URL` names, which the live
Superset's config gives the forwarder as `MNEMIQ_URL`. Its builds are planned by the stub planner (no model is asked
to plan), and a chat answer comes from the suite's stub model. Each request is made as a browser makes it: Superset's
session (here the sign-in's token), and a CSRF token on a POST; no mnemiq token is ever held by the test's client.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from urllib.parse import urlsplit

import pytest

from mnemiq.superset.charts import ChartSpec, Family, WrittenRef
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan, FakePlanner
from tests.superset_live.live import metadata, required

pytestmark = pytest.mark.integration

FORWARD = "/extensions/agenticfabriq/mnemiq/v1"
BUILDER = "analyst"
STORE_DATABASE = "mnemiq_build_store_forward"


@pytest.fixture(scope="module")
def store_dsn(live) -> Iterator[str]:
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
def mnemiq(live, model, enriched, store_dsn):
    """`mnemiq serve --http` in Superset mode, served by uvicorn on `MNEMIQ_TEST_SUPERSET_MNEMIQ_URL` until the module ends;
    the runtime is kept so a test can give the default profile its stub planner."""
    import uvicorn

    import mnemiq.runtime as runtime_module
    import mnemiq.server.app as app_module

    address = urlsplit(required("MNEMIQ_TEST_SUPERSET_MNEMIQ_URL")["MNEMIQ_TEST_SUPERSET_MNEMIQ_URL"])
    settings, _ = enriched
    settings = settings.model_copy(update={"control_dsn": store_dsn})
    built: dict = {}
    with pytest.MonkeyPatch.context() as patch:
        real_runtime = runtime_module.build_runtime
        patch.setattr(runtime_module, "build_runtime", lambda s: built.setdefault("runtime", real_runtime(s)))
        patch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
        app_module.serve_http(settings, host=address.hostname, port=address.port)
    server = uvicorn.Server(uvicorn.Config(built["app"], host=address.hostname, port=address.port,
                                           log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started, "mnemiq did not start serving"
    try:
        yield built["runtime"]
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _one_kpi(policy_id: int) -> BuildPlan:
    policies = Measure(name="forwarded_policies", label="Forwarded policies", expression="COUNT(id)",
                       kind=MeasureKind.COUNT, dataset_id=policy_id, description="Policies", asked_by="")
    kpi = ChartSpec(key="forwarded_kpi", family=Family.BIG_NUMBER, title="Forwarded policies", dataset_id=policy_id,
                    measures=(WrittenRef(policies),), dimensions=(), time_column=None, time_grain=None, filters=(),
                    row_limit=None, source=None, target=None)
    return BuildPlan(charts=(kpi,), measures=(policies,))


@pytest.fixture(scope="module")
def policy_id(live) -> int:
    (row,) = metadata("SELECT id FROM tables WHERE table_name = 'policy'")
    return row[0]


def test_forwarder_needs_csrf_on_post(live, mnemiq, policy_id):
    """A POST through the forwarder without the session's CSRF token is refused by Superset before mnemiq sees it; a
    GET, which changes nothing, needs none, and mnemiq answers it, uncached. (A POST with the token is the next
    test's build.)"""
    mnemiq.profiles[mnemiq.default_profile].planner = FakePlanner(_one_kpi(policy_id))
    with live.signed_in(BUILDER) as (client, headers):
        refused = client.post(f"{FORWARD}/superset/builds", json={"goal": "policies counted"}, headers=headers)
        assert refused.status_code in (400, 403), refused.status_code
        allowed = client.get(f"{FORWARD}/superset/builds", headers=headers)
        assert allowed.status_code == 200, (allowed.status_code, allowed.text[:300])
        assert allowed.headers.get("cache-control") == "no-store"


def test_forwarder_runs_a_build_as_the_signed_in_builder(live, mnemiq, policy_id):
    """A build started and run through the forwarder, one `/next` per step, to done; the dashboard it made is the
    signed-in builder's own."""
    mnemiq.profiles[mnemiq.default_profile].planner = FakePlanner(_one_kpi(policy_id))
    with live.signed_in(BUILDER) as (client, headers):
        csrf = {**headers, "X-CSRFToken": live.csrf_token(client, headers)}
        started = client.post(f"{FORWARD}/superset/builds", json={"goal": "policies counted, forwarded"},
                              headers=csrf)
        assert started.status_code == 201, (started.status_code, started.text[:300])
        build = started.json()["build"]
        assert build["plan"]["charts"][0]["title"] == "Forwarded policies"
        for _ in range(8):
            step = client.post(f"{FORWARD}/superset/builds/{build['build_id']}/next", headers=csrf)
            assert step.status_code == 200, (step.status_code, step.text[:300])
            build = step.json()["build"]
            if build["status"] != "building":
                break
        assert build["status"] == "done", build
        dashboard_id = build["dashboard_id"]
    [(created_by,)] = metadata("SELECT created_by_fk FROM dashboards WHERE id = %s", (dashboard_id,))
    [(analyst,)] = metadata("SELECT id FROM ab_user WHERE username = %s", (BUILDER,))
    assert created_by == analyst


def test_forwarder_streams_a_chat_answer(live, mnemiq, model):
    """A chat answer comes back through the forwarder as mnemiq streams it: the run starts before the stream ends."""
    model.sql = "SELECT count(*) AS policies FROM policy"
    with live.signed_in(BUILDER) as (client, headers):
        csrf = {**headers, "X-CSRFToken": live.csrf_token(client, headers)}
        with client.stream("POST", f"{FORWARD}/chat", json={"question": "how many policies are there?",
                                                            "mode": "thinking", "history": []},
                           headers=csrf, timeout=120) as answer:
            assert answer.status_code == 200, answer.status_code
            assert answer.headers["content-type"].startswith("text/event-stream")
            seen = []
            for line in answer.iter_lines():
                if line.startswith("data:"):
                    seen.append(line)
                    if '"RUN_STARTED"' in line:
                        break
    assert seen and '"RUN_STARTED"' in seen[-1]


def test_gamma_hears_the_roles_through_the_forwarder(live, mnemiq):
    """gamma_only, signed in to Superset, reaches mnemiq through the forwarder and is told the roles building needs,
    and nothing else."""
    with live.signed_in("gamma_only") as (client, headers):
        answer = client.get(f"{FORWARD}/superset/dashboards/1/measures", headers=headers)
    assert answer.status_code == 403, (answer.status_code, answer.text[:300])
    assert "Alpha" in answer.json()["detail"] and "sql_lab" in answer.json()["detail"]


def test_an_unlisted_route_is_not_forwarded(live, mnemiq):
    with live.signed_in(BUILDER) as (client, headers):
        answer = client.get(f"{FORWARD}/whoami", headers=headers)
    assert answer.status_code == 404
    assert answer.json()["detail"] == "This extension does not forward that request to mnemiq."

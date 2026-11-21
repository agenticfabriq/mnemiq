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
from mnemiq.superset.grants import NOT_A_BUILDER
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan, FakePlanner
from tests.superset_live.live import metadata, required

pytestmark = pytest.mark.integration

FORWARD = "/extensions/agenticfabriq/mnemiq/v1"
BUILDER = "analyst"
STORE_DATABASE = "mnemiq_build_store_forward"
# The headers of each request mnemiq received, as it received them, in order: what the forwarder sent.
RECEIVED: list[dict[str, str]] = []


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
    app = built["app"]

    async def recording(scope, receive, send):
        if scope["type"] == "http":
            RECEIVED.append({"path": scope["path"],
                             **{k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}})
        await app(scope, receive, send)

    server = uvicorn.Server(uvicorn.Config(recording, host=address.hostname, port=address.port,
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


def _held_open_transactions() -> int:
    """Connections to Superset's metadata database left idle inside a transaction for more than two seconds."""
    [(count,)] = metadata("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND "
                          "pid <> pg_backend_pid() AND state = 'idle in transaction' AND "
                          "now() - state_change > interval '2 seconds'")
    return count


def test_the_held_transaction_count_sees_one(live):
    """The control for the stream test's count of held connections: a transaction this test opens on Superset's
    metadata database and leaves idle is counted."""
    import psycopg

    before = _held_open_transactions()
    uri = required("MNEMIQ_TEST_SUPERSET_METADATA_URI")["MNEMIQ_TEST_SUPERSET_METADATA_URI"]
    with psycopg.connect(uri, connect_timeout=10) as con:
        con.execute("SELECT 1").fetchall()
        time.sleep(3)
        assert _held_open_transactions() - before >= 1
        con.rollback()


def test_forwarder_streams_a_chat_answer_and_holds_no_metadata_connection(live, mnemiq, model):
    """A chat answer comes back through the forwarder as mnemiq streams it: the run's start arrives while the model is
    still held, before it answers, so a forwarder that buffered the answer fails here. Meanwhile the request holds no
    connection to Superset's metadata database idle in a transaction; the stream carries the forwarder's headers."""
    released, answered = threading.Event(), threading.Event()

    def hold(_count: int) -> None:
        released.wait(20)
        answered.set()

    model.sql = "SELECT count(*) AS policies FROM policy"
    model.on_call = hold
    try:
        before = _held_open_transactions()
        with live.signed_in(BUILDER) as (client, headers):
            csrf = {**headers, "X-CSRFToken": live.csrf_token(client, headers)}
            with client.stream("POST", f"{FORWARD}/chat", json={"question": "how many policies are there?",
                                                                "mode": "thinking", "history": []},
                               headers=csrf, timeout=120) as answer:
                assert answer.status_code == 200, answer.status_code
                assert answer.headers["content-type"].startswith("text/event-stream")
                assert answer.headers.get("x-accel-buffering") == "no"
                assert answer.headers.get("cache-control") == "no-store"
                lines = answer.iter_lines()
                for line in lines:
                    if line.startswith("data:") and '"RUN_STARTED"' in line:
                        break
                assert not answered.is_set(), "the run's start arrived only after the model answered"
                time.sleep(3)
                held = _held_open_transactions() - before
                released.set()
                ended = [line for line in lines if line.startswith("data:")]
        assert held <= 0, f"{held} metadata connection(s) held idle in a transaction while the answer streamed"
        assert any('"RUN_FINISHED"' in line or '"RUN_ERROR"' in line for line in ended)
    finally:
        released.set()
        model.on_call = None


def test_mnemiq_receives_only_the_forwarders_headers(live, mnemiq):
    """What mnemiq receives for a forwarded POST: a token minted for the signed-in user, not Superset's own sign-in
    token the browser sent; no cookie, no CSRF header, and no other header of the browser's or of the forwarder's
    HTTP library."""
    import jwt

    RECEIVED.clear()
    with live.signed_in(BUILDER) as (client, headers):
        csrf = {**headers, "X-CSRFToken": live.csrf_token(client, headers), "X-Extra": "from-the-browser",
                "User-Agent": "the-browser"}
        assert client.cookies, "the client holds Superset's session cookie, so the test can see it is not sent"
        answer = client.post(f"{FORWARD}/superset/builds", json={"goal": ""}, headers=csrf)
        assert answer.status_code < 500, (answer.status_code, answer.text[:300])
    [received] = [r for r in RECEIVED if r["path"] == "/v1/superset/builds"]
    assert set(received) - {"path"} <= {"host", "authorization", "accept", "content-type", "content-length",
                                        "accept-encoding"}, sorted(received)
    assert received.get("accept-encoding", "identity") == "identity"
    sent = headers.get("Authorization") or headers.get("authorization")
    assert received["authorization"] != sent
    claims = jwt.decode(received["authorization"].removeprefix("Bearer "), options={"verify_signature": False})
    assert (claims["sub"], claims["preferred_username"]) == (BUILDER, BUILDER)


def test_gamma_hears_the_roles_through_the_forwarder(live, mnemiq):
    """gamma_only, signed in to Superset, reaches mnemiq through the forwarder and is told the roles building needs,
    and nothing else."""
    with live.signed_in("gamma_only") as (client, headers):
        answer = client.get(f"{FORWARD}/superset/dashboards/1/measures", headers=headers)
    assert answer.status_code == 403, (answer.status_code, answer.text[:300])
    assert "Alpha" in answer.json()["detail"] and "sql_lab" in answer.json()["detail"]


def test_the_panels_readiness_probe_answers_through_the_forwarder(live, mnemiq):
    """The panel's readiness probe, `GET v1/schema` through the forwarder: a builder is answered the opening
    questions (a list), and gamma_only mnemiq's roles sentence, as a 403, which the panel shows as the whole of it."""
    with live.signed_in(BUILDER) as (client, headers):
        ready = client.get(f"{FORWARD}/schema", headers=headers)
    assert ready.status_code == 200, (ready.status_code, ready.text[:300])
    assert isinstance(ready.json()["starters"], list)
    with live.signed_in("gamma_only") as (client, headers):
        refused = client.get(f"{FORWARD}/schema", headers=headers)
    assert refused.status_code == 403, (refused.status_code, refused.text[:300])
    assert refused.json()["detail"] == NOT_A_BUILDER


def test_an_unlisted_route_is_not_forwarded(live, mnemiq):
    with live.signed_in(BUILDER) as (client, headers):
        answer = client.get(f"{FORWARD}/whoami", headers=headers)
    assert answer.status_code == 404
    assert answer.json()["detail"] == "This extension does not forward that request to mnemiq."

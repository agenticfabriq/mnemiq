"""The live contract: mnemiq in Superset mode against a real Superset 7.0.0rc2, holding no database credential, each
question answered through Superset's MCP service as the person asking, and enrichment read as a service user.

Each test states its expectation in its docstring. Each answer is held to the ground truth read from the source database
directly (`seed.sql`: 300 policies, 100 in each region, 60 of each region's 100 active). Observations a run should
keep (Superset's own words, counts and timings) are printed as `OBSERVED` lines, which `pytest -rA` shows for passing
tests too.

`conftest.py` says how the Superset is described to the suite, and `setup_superset.py` how it is set up.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from pathlib import Path

import pyarrow as pa
import pytest

from mnemiq.contract import HistoryTurn, IdentityContext
from mnemiq.contract.seams import DeferralReason
from mnemiq.runtime import build_runtime
from mnemiq.server.superset_refusals import SIGN_IN_AGAIN
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot
from mnemiq.superset.client import (
    _ERROR_PREFIX,
    _ROLE_DENIAL,
    DelegatedToken,
    SupersetMcp,
    SupersetPermissionDenied,
    SupersetRefusedCaller,
    SupersetResponseTooLarge,
)
from mnemiq.superset.executor import superset_executor
from mnemiq.superset.grants import NOT_A_BUILDER, _probe, _tables, resolve_grants
from mnemiq.superset.payload import _CUT as CUT
from mnemiq.superset.payload import table_from_payload
from mnemiq.superset.service import ServiceKey, enrichment_adapter
from mnemiq.superset.settings import SUPERSET_SOURCE_JOB
from tests.sweep.stub_model import _WRITER, _messages, _proposal
from tests.superset_live.live import (
    SCHEMA,
    SOURCE_ID,
    TENANT,
    direct_settings,
    enrich_and_build,
    fingerprint_of,
    ground_truth,
    required,
    run_setup,
    superset_settings,
)

pytestmark = pytest.mark.integration

QUESTION = "How many policies are there in each region?"
REGION_COUNTS = "SELECT region, COUNT(*) AS n FROM policy GROUP BY region"
# Every row of `policy`, each column named: the decider refuses `SELECT *`.
WIDE = "SELECT id, region, status, premium, issued, doc FROM policy"
AGGREGATE = "SELECT status, COUNT(*) AS n FROM policy GROUP BY status"
# A statement that runs for several seconds on the seeded Postgres.
SLOW = "SELECT count(*) AS n FROM generate_series(1, 100000000) AS g"
TRUTH_BY_REGION = "SELECT region, count(*) FROM policy GROUP BY region"
TRUTH_ACTIVE_BY_REGION = "SELECT region, count(*) FROM policy WHERE status = 'active' GROUP BY region"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "superset_execute_sql_types.json"
# One statement whose result holds a numeric, a date, a timestamp with and without a zone, a json value and NULLs.
TYPES_SQL = ("SELECT id, premium, issued, issued + TIME '09:30' AS issued_at, "
             "(issued + TIME '09:30') AT TIME ZONE 'UTC' AS issued_utc, doc, CAST(NULL AS text) AS nothing "
             "FROM policy WHERE id IN (24, 25) ORDER BY id")


def observe(test: str, **facts) -> None:
    """Print what a run should keep, one line per fact set, for the run's record."""
    print(f"OBSERVED {test} {json.dumps(facts, default=str, sort_keys=True)}")


def identity(username: str) -> IdentityContext:
    return IdentityContext(tenant_id=TENANT, principal_id=username, roles=[])


def ask(runtime, live, username: str, *, history=None, question: str = QUESTION):
    return runtime.ask(question, identity(username), history=history,
                       delegation=DelegatedToken(live.token(username)))


def rows_of(answer) -> list[list]:
    assert answer.preview is not None, (answer.answer, answer.failed, answer.deferred, answer.reason_code)
    return sorted(answer.preview.rows)


def snapshot_of(settings):
    con = init_store(settings.store_path)
    try:
        return load_snapshot(con, current_version(con, SOURCE_ID))
    finally:
        con.close()


def serve(settings, monkeypatch):
    """The app `mnemiq serve --http` serves for `settings`: `serve_http` checks the settings, builds the runtime and
    the doors, and hands the app to uvicorn, which alone is replaced."""
    from fastapi.testclient import TestClient

    import mnemiq.server.app as app_module

    built: dict = {}
    monkeypatch.setattr(app_module, "_run_uvicorn", lambda app, host, port: built.update(app=app))
    app_module.serve_http(settings, host="127.0.0.1", port=8080)
    return TestClient(built["app"])


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- L1 ---------------------------------------------------------------------------------------------------------------


def test_enrich_through_superset_matches_the_direct_path(live, model, enriched, tmp_path):
    """L1. Enriched through Superset as `svc`, and enriched directly from the same Postgres with
    MNEMIQ_SUPERSET_MCP_URL unset, the two snapshots name the same tables, the same columns and the same views.

    The Superset snapshot carries the `superset:source` job (database id, schema `public`, backend `postgresql`, values
    `not harvested`); it has no coded values and so no top-k samples, its store's value index holds nothing for the
    source, and it has no stored examples. Two-valued partition labels may be in `unique_within` (`party.role` holds
    two values), and none of their values reaches anything the model reads: the cards `mnemiq build` indexed, or a
    prompt the model is sent on an ask. The direct snapshot does hold coded values, so the checks can see some."""
    settings, printed = enriched
    superset = snapshot_of(settings)
    direct = direct_settings(live, model, tmp_path / "direct.duckdb")
    assert not direct.superset_mode
    enrich_and_build(direct, None)
    plain = snapshot_of(direct)

    def tables(snapshot):
        return {binding.source_object for binding in snapshot.source_bindings if binding.binding_type == "table"}

    assert tables(superset) == tables(plain)
    assert {"policy", "party", "policy_by_region", "Policy Lines", "coverage"} <= tables(superset)
    assert {column.id for column in superset.columns} == {column.id for column in plain.columns}
    assert {view.object_id for view in superset.views} == {view.object_id for view in plain.views} == {
        "policy_by_region"}

    jobs = [job for job in superset.jobs if job.id == SUPERSET_SOURCE_JOB]
    assert [(job.kind, job.status, json.loads(job.detail)) for job in jobs] == [
        ("superset", "done", {"database_id": live.database_id, "schema": SCHEMA, "backend": "postgresql",
                              "values": "not harvested"})]
    assert "0 indexed values" in printed
    assert not [column.id for column in superset.columns if column.coded_values]
    assert [column.id for column in plain.columns if column.coded_values]  # the direct path does harvest
    assert superset.examples == []
    con = init_store(settings.store_path)
    try:
        assert con.execute("SELECT count(*) FROM value_index WHERE source_id = ?", [SOURCE_ID]).fetchone() == (0,)
        # `mnemiq build` makes the example table only when there is an example to index.
        if con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'example'").fetchone() != (0,):
            assert con.execute("SELECT count(*) FROM example WHERE source_id = ?", [SOURCE_ID]).fetchone() == (0,)
        cards = " ".join(card for (card,) in con.execute("SELECT card FROM semantic_object").fetchall())
    finally:
        con.close()
    assert cards

    labels = {label for column in superset.columns for label in (column.unique_within or {})}
    values = {value for label in labels for value in re.findall(r"'([^']*)'", label)}
    assert {"insured", "nominee"} <= values, labels  # the passing control: the labels are there to be found
    model.sql = REGION_COUNTS
    mark = len(model.requests)
    answer = ask(build_runtime(settings), live, "analyst")
    assert not answer.failed and not answer.deferred, answer.answer
    prompts = json.dumps(model.requests[mark:])
    assert prompts and not [value for value in values if value in cards or value in prompts]
    observe("L1", tables=sorted(tables(superset)), columns=len(superset.columns), labels=sorted(labels),
            views=sorted(view.object_id for view in superset.views))


# Every foreign key Postgres holds in `public`, each column pair once, and whether the row is Postgres's own copy of a
# key for one partition (`conparentid` names the key it was copied from) rather than a key as declared. The seed's keys
# are single-column, so the first column of each is the whole key.
DECLARED_KEYS = ("SELECT src.relname, a.attname, dst.relname, b.attname, con.conparentid <> 0 "
                 "FROM pg_constraint AS con JOIN pg_class AS src ON src.oid = con.conrelid "
                 "JOIN pg_namespace AS n ON n.oid = src.relnamespace JOIN pg_class AS dst ON dst.oid = con.confrelid "
                 "JOIN pg_attribute AS a ON a.attrelid = con.conrelid AND a.attnum = con.conkey[1] "
                 "JOIN pg_attribute AS b ON b.attrelid = con.confrelid AND b.attnum = con.confkey[1] "
                 "WHERE con.contype = 'f' AND n.nspname = 'public'")


def test_foreign_keys_through_superset_are_the_declared_keys(live, enriched):
    """L1, the foreign keys. The keys the service user reads through Superset's catalogue, each as (from table, from
    column, to table, to column), are the keys the source declares, read by the suite from `pg_constraint` directly:
    `party`, `Policy Lines` and `coverage` to `policy`, and `coverage_claim` to `coverage`.

    `coverage` is partitioned, and a foreign key both references it and is declared on it, so Postgres also keeps a
    constraint row for each partition, naming the partition in place of the table. Superset mode reads the declared
    keys only (`conparentid = 0`), so those copies are not among its keys: a partition's own copy of a key declared on
    its parent is left out too. The snapshot carries the declared keys as its relationships."""
    settings, _ = enriched
    through = enrichment_adapter(settings, ServiceKey(Path(live.service_key_file).read_text().strip()))
    superset_keys = {tuple(key[:4]) for key in through.foreign_keys()}
    held = ground_truth(live, DECLARED_KEYS)
    declared = {row[:4] for row in held if not row[4]}
    copies = {row[:4] for row in held if row[4]}
    assert declared == {("party", "policy_id", "policy", "id"), ("Policy Lines", "policy_id", "policy", "id"),
                        ("coverage", "policy_id", "policy", "id"), ("coverage_claim", "coverage_id", "coverage", "id")}
    assert copies  # the passing control: the partition copies are there to be left out
    assert superset_keys == declared
    assert {relationship.id for relationship in snapshot_of(settings).relationships} == {
        f"{source}.{column}->{target}" for source, column, target, _ in declared}
    observe("L1-fk", superset=sorted(superset_keys), partition_copies_left_out=sorted(copies))


# --- L2 and L11's first half ------------------------------------------------------------------------------------------


def _two_builders(live, model, settings, recorder, monkeypatch, test: str) -> None:
    """West and east ask the same question through `/v1/ask`, each with a Bearer the identity provider signed; each is
    answered with their own region's count alone, every call to Superset carrying their own token."""
    client = serve(settings, monkeypatch)
    model.sql = REGION_COUNTS
    truth = dict(ground_truth(live, TRUTH_BY_REGION))
    for user in ("west", "east"):
        token = live.token(user)
        mark = recorder.mark()
        started = time.perf_counter()
        response = client.post("/v1/ask", json={"question": QUESTION}, headers=bearer(token))
        seconds = time.perf_counter() - started
        assert response.status_code == 200, response.text
        body = response.json()
        assert not body["failed"] and not body["deferred"], body
        assert body["preview"]["rows"] == [[user, truth[user]]]
        calls = recorder.since(mark)
        assert calls and {call.credential for call in calls} == {f"DelegatedToken:{fingerprint_of(token)}"}
        answering = [call for call in calls if call.answers]
        executed = [statement.get("executed_sql") for call in answering for statement in
                    (call.payload or {}).get("statements") or []]
        methods: dict[str, int] = {}
        for call in calls:
            for method, _ in call.rpc:
                methods[method] = methods.get(method, 0) + 1
        observe(test, user=user, seconds=round(seconds, 2), sessions=len(calls),
                calls_by_tool={tool: sum(1 for c in calls if c.tool == tool) for tool in {c.tool for c in calls}},
                rpc=methods, rpc_seconds={m: round(sum(s for c in calls for n, s in c.rpc if n == m), 2)
                                          for m in methods},
                call_seconds=round(sum(call.seconds for call in calls), 2),
                decided=[call.sql for call in answering], executed=executed)


def test_two_builders_get_their_own_rows(live, model, enriched, recorder, monkeypatch):
    """L2. The same question as `west` and as `east`, through `/v1/ask` with a Bearer the test identity provider
    signed, so the door verifies it and hands the same token on. The stub model writes `SELECT region, COUNT(*) AS n
    FROM policy GROUP BY region`; west is answered `[["west", 100]]` and east `[["east", 100]]`, each region's count
    in the source, and every call each ask makes to Superset carries that person's own token and nothing else."""
    settings, _ = enriched
    _two_builders(live, model, settings, recorder, monkeypatch, "L2")


# --- L3 and L11's second half -----------------------------------------------------------------------------------------


def _narrowed_follow_up(live, model, settings, recorder, test: str) -> None:
    """West asks; west's row rule is narrowed to west's active policies; west asks again in the same conversation and
    is answered with west's active policies alone, and the earlier turn run again reads them alone too."""
    runtime = build_runtime(settings)
    model.sql = REGION_COUNTS
    everything = dict(ground_truth(live, TRUTH_BY_REGION))["west"]
    active = dict(ground_truth(live, TRUTH_ACTIVE_BY_REGION))["west"]
    assert active < everything
    started = time.perf_counter()
    first = ask(runtime, live, "west")
    first_seconds = time.perf_counter() - started
    assert rows_of(first) == [["west", everything]]
    turn = HistoryTurn(question=QUESTION, sql=first.trace.asked_sql, columns=first.preview.columns,
                       rows=first.preview.rows, grant_fingerprint=first.grant_fingerprint)
    try:
        observe(test, rule=run_setup("narrow", "west"))
        mark = recorder.mark()
        started = time.perf_counter()
        second = ask(runtime, live, "west", history=[turn])
        follow_up_seconds = time.perf_counter() - started
        assert rows_of(second) == [["west", active]]
        statements = [call for call in recorder.since(mark) if call.answers]
        # The earlier turn run again, then the answer: each reads west's active policies alone.
        assert [call.payload["rows"] for call in statements] == [[{"region": "west", "n": active}]] * 2
        observe(test, first_seconds=round(first_seconds, 2), follow_up_seconds=round(follow_up_seconds, 2),
                follow_up_sessions=len(recorder.since(mark)),
                statements=[round(call.seconds, 2) for call in statements])
    finally:
        observe(test, rule=run_setup("widen", "west"))


def test_narrowed_access_shows_on_the_next_ask(live, model, enriched, recorder):
    """L3. West asks and is answered `[["west", 100]]`. Then `setup_superset.py narrow west` narrows west's row rule
    to west's active policies, and west asks again in the same conversation, the first turn sent back with its rows:
    the answer is `[["west", 60]]`, and the earlier turn, run again as west, returns 60 too, never the 100 the client
    sent back. The rule is widened again afterwards, whatever happened."""
    settings, _ = enriched
    _narrowed_follow_up(live, model, settings, recorder, "L3")


# --- L4 and L5 --------------------------------------------------------------------------------------------------------


def test_gamma_only_is_told_the_roles(live, model, enriched, recorder, monkeypatch):
    """L4. `gamma_only` (Gamma, no sql_lab) asks through `/v1/ask`: 403 with `NOT_A_BUILDER`, which names Alpha and
    sql_lab, and the model is never asked. What Superset sent is its role denial, which the client reads as
    `SupersetPermissionDenied` by its pattern (`_ROLE_DENIAL`): `Permission denied: <permission> on <view>` after the
    `Error: ` Superset's outermost middleware puts first."""
    settings, _ = enriched
    client = serve(settings, monkeypatch)
    model.sql = REGION_COUNTS
    asked = model.calls
    response = client.post("/v1/ask", json={"question": QUESTION}, headers=bearer(live.token("gamma_only")))
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail == NOT_A_BUILDER and "Alpha" in detail and "sql_lab" in detail
    assert model.calls == asked
    refused = [call.raised for call in recorder.calls if call.raised is not None]
    assert refused and isinstance(refused[0], SupersetPermissionDenied)
    said = refused[0].detail
    assert said.startswith(_ERROR_PREFIX) and _ROLE_DENIAL.match(said.removeprefix(_ERROR_PREFIX)), said
    observe("L4", status=response.status_code, detail=detail, superset_said=said)


def test_unknown_user_is_refused(live, model, enriched, recorder, monkeypatch):
    """L5. A token the identity provider validly signed for `nobody`, a user Superset has no account for: `/v1/ask`
    answers 401 with the sign-in sentence (`SIGN_IN_AGAIN`), and the model is never asked. The client reads what
    Superset sent as `SupersetRefusedCaller`."""
    settings, _ = enriched
    client = serve(settings, monkeypatch)
    model.sql = REGION_COUNTS
    asked = model.calls
    response = client.post("/v1/ask", json={"question": QUESTION}, headers=bearer(live.token("nobody")))
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == SIGN_IN_AGAIN
    assert model.calls == asked
    refused = [call.raised for call in recorder.calls if call.raised is not None]
    assert refused and isinstance(refused[0], SupersetRefusedCaller)
    observe("L5", status=response.status_code, detail=response.json()["detail"], superset_said=refused[0].detail)


def test_a_token_without_supersets_scopes_is_refused(live, model, enriched, recorder, monkeypatch):
    """A token carrying mnemiq's scope and none of Superset's, for `west`, who holds Alpha and sql_lab: Superset
    holds a token that carries any scope to its own scope map, so `execute_sql` is refused, and the request ends as a
    403 refusal, never an answer."""
    settings, _ = enriched
    client = serve(settings, monkeypatch)
    model.sql = REGION_COUNTS
    response = client.post("/v1/ask", json={"question": QUESTION},
                           headers=bearer(live.token("west", scope="openid db.read")))
    assert response.status_code == 403, response.text
    refused = [call.raised for call in recorder.calls if call.raised is not None]
    assert refused
    observe("scopes", status=response.status_code, detail=response.json()["detail"], superset_said=refused[0].detail,
            read_as=type(refused[0]).__name__)


# --- L6 ---------------------------------------------------------------------------------------------------------------


def test_a_large_result_is_cut_never_complete(live, model, enriched, recorder):
    """L6. `analyst`, whom no row rule narrows, is answered from every row of `policy`, its columns named (the decider
    refuses `SELECT *`): 300 rows, past Superset's default response size guard once `execute_sql` carries each row
    twice. The answer is repaired or stated as cut, never answered as all 300 rows, nor as the part Superset sent:

    - A model that writes the same statement on every attempt is told each time that Superset cut the result, and the
      answer ends failed (`execution_failed`, not deferred) with no preview: every statement that ran came back
      marked `_response_truncated`, or refused as too large to send (`SupersetResponseTooLarge`).
    - A model that answers the cut with an aggregate is answered with the aggregate, each status's count in the
      source."""
    settings, _ = enriched
    assert ground_truth(live, "SELECT count(*) FROM policy") == [(300,)]
    runtime = build_runtime(settings)
    model.sql = WIDE
    mark, asked = recorder.mark(), len(model.requests)
    answer = ask(runtime, live, "analyst")
    assert answer.failed and not answer.deferred, answer.answer
    assert answer.reason_code == DeferralReason.EXECUTION_FAILED and answer.preview is None
    statements = [call for call in recorder.since(mark) if call.answers]
    assert statements and all((call.payload or {}).get("_response_truncated")
                              or isinstance(call.raised, SupersetResponseTooLarge) for call in statements)
    assert CUT in json.dumps(model.requests[asked:])  # the model was told why
    observe("L6", answer=answer.answer, failed=answer.failed, deferred=answer.deferred, reason=str(answer.reason_code),
            statements=len(statements), rows_sent=[len((call.payload or {}).get("rows") or []) for call in statements],
            how=[type(call.raised).__name__ if call.raised else "_response_truncated" for call in statements])

    def aggregate_once_cut(request):
        system, user = _messages(request)
        return _proposal(AGGREGATE) if _WRITER in system and CUT in user else None

    model.rule = aggregate_once_cut
    try:
        repaired = ask(runtime, live, "analyst")
    finally:
        model.rule = None
    assert rows_of(repaired) == sorted([status, n] for status, n in ground_truth(
        live, "SELECT status, count(*) FROM policy GROUP BY status"))
    observe("L6", repaired=repaired.answer, rows=repaired.preview.rows, attempts=repaired.attempts)


# --- L7 ---------------------------------------------------------------------------------------------------------------


def test_reachable_functions_follow_the_schema(live, model, enriched, recorder):
    """L7. `other.median(int)` is in `user_functions()`, which spans every schema a user can define a function in, and
    not in `reachable_user_functions()`, since Superset sets the search path to `public`; Postgres lists no builtin
    `median`. An unqualified `SELECT median(1)` uses the builtin and never `other.median`: Superset runs every
    statement as its SQL parser writes it back for Postgres, which spells `median(x)` as the builtin
    `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`, answering the double 1.0, where `SELECT other.median(1)` answers
    `other.median`'s integer 1. Asked through mnemiq, the statement is answered with the builtin's 1.0."""
    settings, _ = enriched
    adapter = superset_executor(settings, DelegatedToken(live.token("analyst")))
    assert "median" in adapter.user_functions()
    assert "median" not in adapter.reachable_user_functions()
    assert "median" not in adapter.builtin_functions()

    def run(sql: str) -> dict:
        return adapter.client.call("execute_sql", {"database_id": live.database_id, "sql": sql, "schema": SCHEMA,
                                                   "timeout": 30})

    unqualified, qualified = run("SELECT median(1) AS m"), run("SELECT other.median(1) AS m")
    executed = unqualified["statements"][0]["executed_sql"]
    assert "PERCENTILE_CONT" in executed.upper() and "MEDIAN" not in executed.upper(), executed
    assert (unqualified["rows"], unqualified["columns"][0]["type"]) == ([{"m": 1.0}], "float64")
    assert (qualified["rows"], qualified["columns"][0]["type"]) == ([{"m": 1}], "int64")
    model.sql = "SELECT median(1) AS m"
    mark = recorder.mark()
    answer = ask(build_runtime(settings), live, "analyst", question="What is the median of one?")
    ran = [statement["executed_sql"] for call in recorder.since(mark) if call.answers and call.payload
           for statement in call.payload["statements"]]
    assert rows_of(answer) == [[1.0]] and isinstance(answer.preview.rows[0][0], float)
    assert ran and all("PERCENTILE_CONT" in sql.upper() for sql in ran)
    observe("L7", decided=answer.trace.asked_sql, executed=ran, answer=answer.answer)


# --- L8 ---------------------------------------------------------------------------------------------------------------


def test_dataset_only_builder_reaches_only_datasets(live, model, enriched, recorder):
    """L8. `policy_only` (Gamma, sql_lab and `datasource_access` on the `policy` dataset alone) is a builder granted
    datasets, not the database.

    - The grants' first probe, a dry run naming every snapshot table, comes back as a payload with `success: false`
      and an `error_type` ending `_SECURITY_ACCESS_ERROR`, never as a tool error: `execute_sql` catches Superset's
      security exception itself. The fallback then grants the tables of `policy` alone, while `party` stays in the
      snapshot.
    - The catalogue is refused to them (`force_dataset_match`), so their function inventory cannot be read, and a
      statement that calls a function (`lower(region)`) is refused and never answered. So is every other statement:
      with the inventory unreadable the decider cannot prove a statement free of calls (an operator can bind to a
      function as a call does), so it answers none, `COUNT(*)` and a plain column read alike, each deferred as
      ungovernable with the same sentence, and nothing is run.
    - The run prints, for each statement, the sentence the answer carries, how many catalogue reads it made and how
      many were refused."""
    settings, _ = enriched
    snapshot = snapshot_of(settings)
    tables = _tables(snapshot)
    assert {"policy", "party"} <= set(tables)
    token = live.token("policy_only")
    probe = SupersetMcp(live.mcp_url, DelegatedToken(token)).call(
        "execute_sql", {"database_id": live.database_id, "sql": _probe(list(tables)), "schema": SCHEMA, "timeout": 30,
                        "dry_run": True})
    assert probe["success"] is False and probe["error_type"].endswith("_SECURITY_ACCESS_ERROR"), probe
    grants = resolve_grants(superset_executor(settings, DelegatedToken(token)), snapshot)
    assert grants.objects == frozenset(tables["policy"])

    runtime = build_runtime(settings)
    seen = {}
    for name, sql in (("count", REGION_COUNTS), ("function", "SELECT lower(region) AS r FROM policy"),
                      ("plain", "SELECT id, region FROM policy WHERE id <= 3 ORDER BY id")):
        model.sql = sql
        mark = recorder.mark()
        answer = ask(runtime, live, "policy_only")
        calls = recorder.since(mark)
        seen[name] = (answer, [call for call in calls if call.answers])
        observe("L8", statement=name, sql=sql, answer=answer.answer, failed=answer.failed, deferred=answer.deferred,
                reason=str(answer.reason_code), attempts=answer.attempts, answered=answer.preview is not None,
                catalogue_reads=len([call for call in calls if call.request.get("force_refresh")]),
                catalogue_refused=len([call for call in calls if call.request.get("force_refresh")
                                       and (call.payload or {}).get("success") is False]))
    sentences = {answer.answer for answer, _ in seen.values()}
    assert len(sentences) == 1, sentences
    for name, (answer, statements) in seen.items():
        assert answer.deferred and answer.reason_code == DeferralReason.UNGOVERNABLE, (name, answer.answer)
        assert answer.preview is None and statements == [], name
    observe("L8", probe_error_type=probe["error_type"], probe_error=probe.get("error"),
            granted=sorted(grants.objects), sentence=sentences.pop())


# --- L9 ---------------------------------------------------------------------------------------------------------------


def test_value_encodings(live):
    """L9. One `execute_sql` payload holding a `numeric`, a `date`, a `timestamp`, a `timestamptz`, a `json` value and
    NULLs is the payload recorded in `tests/fixtures/superset_execute_sql_types.json` (its columns, rows and row
    count; the values are the seed's, synthetic), and `table_from_payload` types each column by the dtype Superset
    reports for it: a timestamp as `timestamp[us]`, a zoned one as `timestamp[us]` in UTC, an integer as `int64`, and
    every `object` column as pyarrow infers it (the numeric as a double, the date as text, the json value as a struct,
    the all-NULL column as null). `test_a_recorded_live_payload_reads_with_its_types` in `tests/test_superset_payload.py` reads the recorded
    payload in CI."""
    client = SupersetMcp(live.mcp_url, DelegatedToken(live.token("analyst")))
    payload = client.call("execute_sql", {"database_id": live.database_id, "sql": TYPES_SQL, "schema": SCHEMA,
                                          "timeout": 30})
    recorded = json.loads(FIXTURE.read_text())
    stable = ("success", "columns", "rows", "row_count", "error", "error_type")
    assert {key: payload.get(key) for key in stable} == {key: recorded.get(key) for key in stable}
    table = table_from_payload(payload["columns"], payload["rows"])
    observe("L9", columns=payload["columns"], arrow=[str(t) for t in table.schema.types])
    assert dict(zip(table.column_names, table.schema.types, strict=True)) == {
        "id": pa.int64(), "premium": pa.float64(), "issued": pa.string(), "issued_at": pa.timestamp("us"),
        "issued_utc": pa.timestamp("us", tz="UTC"),
        "doc": pa.struct([("channel", pa.string()), ("riders", pa.int64())]), "nothing": pa.null()}


# --- L10 --------------------------------------------------------------------------------------------------------------


class _Listener:
    """A TCP listener on 127.0.0.1 that counts the connections it accepts and closes each at once."""

    def __init__(self) -> None:
        self.accepted = 0
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(16)
        self._socket.settimeout(0.2)
        self.port = self._socket.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.accepted += 1
            connection.close()

    def __enter__(self) -> _Listener:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(5)
        self._socket.close()


def test_no_connection_to_the_source_dsn(live, model, tmp_path, monkeypatch):
    """L10. `MNEMIQ_PG_DSN` (and the settings' `pg_dsn`) points at a TCP listener that counts the connections it
    accepts. Over an enrichment through Superset, its build and three asks (west, east, analyst), it counts zero. The
    passing control first: a direct deployment's adapter on the same DSN does connect to it."""
    from mnemiq.adapters.resolve import adapter_for, source_spec

    with _Listener() as listener:
        dsn = f"postgresql://mnemiq@127.0.0.1:{listener.port}/live"
        plain = direct_settings(live, model, tmp_path / "direct.duckdb", pg_dsn=dsn)
        with pytest.raises(Exception):  # noqa: B017 -- whatever the driver raises for a server that hangs up
            adapter_for(source_spec(plain), plain).introspect()
        assert listener.accepted >= 1
        listener.accepted = 0
        monkeypatch.setenv("MNEMIQ_PG_DSN", dsn)
        settings = superset_settings(live, model, tmp_path / "store.duckdb", pg_dsn=dsn)
        assert settings.superset_mode and settings.pg_dsn == dsn
        enrich_and_build(settings, live.service_key_file)
        runtime = build_runtime(settings)
        model.sql = REGION_COUNTS
        truth = dict(ground_truth(live, TRUTH_BY_REGION))
        for user, expected in (("west", [["west", truth["west"]]]), ("east", [["east", truth["east"]]]),
                               ("analyst", sorted([region, n] for region, n in truth.items()))):
            assert rows_of(ask(runtime, live, user)) == expected
        assert listener.accepted == 0


# --- L11 --------------------------------------------------------------------------------------------------------------


def test_with_a_real_superset_cache(live, model, enriched, recorder, monkeypatch, tmp_path):
    """L11. L2 and L3 again, through a second MCP service on the same Superset whose result cache is a
    `FileSystemCache`. Both hold as they do uncached: each
    builder their own region, and west's narrowed rule seen on the next ask, since Superset keys a cached result on
    the statement after its row rules are applied.

    The cache is shown to be live: a catalogue statement sent without `force_refresh` and asked again after a function
    is created is answered from the cache, without the new function. And a function created after a first ask is in
    the next request's function inventory, which mnemiq reads with `force_refresh`."""
    cached = required("MNEMIQ_TEST_SUPERSET_CACHED_MCP_URL")["MNEMIQ_TEST_SUPERSET_CACHED_MCP_URL"]
    base, _ = enriched
    settings = base.model_copy(update={"superset_mcp_url": cached})
    _two_builders(live, model, settings, recorder, monkeypatch, "L11-L2")
    _narrowed_follow_up(live, model, settings, recorder, "L11-L3")

    import psycopg

    listing = ("SELECT p.proname FROM pg_catalog.pg_proc AS p JOIN pg_catalog.pg_namespace AS n "
               "ON n.oid = p.pronamespace WHERE n.nspname = 'public' ORDER BY 1")
    client = SupersetMcp(cached, DelegatedToken(live.token("analyst")))

    def listed(force_refresh: bool) -> list:
        request = {"database_id": live.database_id, "sql": listing, "schema": SCHEMA, "timeout": 30}
        if force_refresh:
            request["force_refresh"] = True
        return [row["proname"] for row in client.call("execute_sql", request)["rows"]]

    name = f"mnemiq_live_{int(time.time())}"
    before = superset_executor(settings, DelegatedToken(live.token("analyst"))).user_functions()
    assert name not in before
    assert name not in listed(False)  # now in Superset's cache
    try:
        with psycopg.connect(live.pg_dsn, autocommit=True) as con:
            con.execute(f"CREATE FUNCTION public.{name}() RETURNS integer LANGUAGE sql AS $$ SELECT 1 $$")
        assert name not in listed(False)  # the cache answered, as it was
        assert name in listed(True)
        after = superset_executor(settings, DelegatedToken(live.token("analyst"))).user_functions()
        assert name in after
    finally:
        with psycopg.connect(live.pg_dsn, autocommit=True) as con:
            con.execute(f"DROP FUNCTION IF EXISTS public.{name}()")
    observe("L11", created=name)


# --- live checks the earlier tasks left owed --------------------------------------------------------------------------


def test_a_statement_past_its_time_limit_reads_as_timed_out(live, enriched, monkeypatch):
    """A statement past the time limit mnemiq gives it reads as that limit (`QueryTimedOut`), the statement's cost for
    the model to make cheaper, never as Superset being unavailable.

    Superset's own limit takes effect only once the database call returns (`SigalrmTimeout`): given 1 second, a count
    over 100 million generated rows (several seconds on the seeded Postgres; Superset refuses `pg_sleep` outright) is
    answered `timed_out` only after it has run to its end. A statement still running when the transport gives up
    reads as the limit too: here the transport's margin is cut to 1 second, so it gives up at 2 seconds, mid-statement.
    """
    import mnemiq.superset.adapter as adapter_module
    from mnemiq.adapters.base import QueryTimedOut
    from mnemiq.superset.client import SupersetCallTimedOut

    settings, _ = enriched
    adapter = superset_executor(settings, DelegatedToken(live.token("analyst")))
    started = time.perf_counter()
    with pytest.raises(QueryTimedOut) as answered:
        adapter.execute_arrow(SLOW, timeout_s=1)
    ran = time.perf_counter() - started
    assert answered.value.__cause__ is None and ran > 2  # Superset's own `timed_out`, once the statement had ended
    monkeypatch.setattr(adapter_module, "TRANSPORT_MARGIN_S", 1.0)
    started = time.perf_counter()
    with pytest.raises(QueryTimedOut) as gave_up:
        adapter.execute_arrow(SLOW, timeout_s=1)
    waited = time.perf_counter() - started
    assert isinstance(gave_up.value.__cause__, SupersetCallTimedOut)
    observe("timeout", superset_answered_after=round(ran, 2), transport_gave_up_after=round(waited, 2))

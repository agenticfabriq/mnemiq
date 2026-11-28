"""A snapshot enriched directly from Oracle, for Superset mode: `mnemiq enrich --for-superset` reads Oracle over a direct
connection and records, in the `superset:oracle` job, the service the connection reached and the schema its unqualified
names resolve in, as the session reports them, with whether values were harvested and how the tables were profiled.
A server in Superset mode serves such a snapshot when the schema it is configured for, as Superset shows it, is the
schema the snapshot was read from, as Oracle stores it, matched through the one name mapping; it then sets its SQL
dialect to Oracle's, once, and never swaps to a snapshot read from another database.

The source of the enrichment tests is the shop in SQLite, with the dictionary reads statistics mode makes
(`tests/test_statistics_enrich._Recording`) and the session read an Oracle connection answers; the live test at the end
reads the Oracle container itself.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3

import pytest

import mnemiq.adapters.resolve as resolve_module
import mnemiq.enrichment.pipeline as pipeline
import mnemiq.runtime as runtime_module
from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.cli import _cmd_enrich, _cmd_refresh
from mnemiq.config import Settings
from mnemiq.contract.semantic import Job
from mnemiq.enrichment.accumulate import merge_run
from mnemiq.enrichment.pipeline import content_version, enrich_structural
from mnemiq.runtime import SnapshotMissing, build_runtime, superset_source_refusal
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot
import mnemiq.superset.settings as superset_settings
from mnemiq.server.superset_refusals import Refused, superset_refusal
from mnemiq.superset.settings import (
    SNAPSHOT_REFUSED,
    SUPERSET_ORACLE_JOB,
    SUPERSET_SOURCE_JOB,
    SupersetSnapshotRefused,
    superset_dialect,
    use_superset_dialect,
)
from tests.superset_fake import FakeSuperset
from tests.superset_shared import SNAPSHOT_REFUSED as SHARED_SNAPSHOT_REFUSED
from tests.superset_shared import (
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_SETTING,
    ORACLE_SERVICE,
    SOURCE_ID,
    _oracle_job,
    _snapshot,
    _source_job,
    superset_deployment,
)
from tests.sweep.stub_model import StubModel
from tests.test_statistics_enrich import SHOP, _Recording

# The statement that reads what the job records of the connection, exactly as it is sent.
BINDING_SQL = "SELECT SYS_CONTEXT('USERENV', 'SERVICE_NAME'), SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL"


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


class _OracleSession(_Recording):
    """The shop in SQLite, answering the statement that reads an Oracle session's service and current schema as that
    session would (`session`; an exception is raised as the read's failure). Every statement is recorded."""

    def __init__(self, inner, session=(ORACLE_SERVICE, ORACLE_SCHEMA)):
        super().__init__(inner)
        self.session = session

    def execute(self, sql, *args, **kwargs):
        if "SYS_CONTEXT" in sql:
            self.sql.append(sql)
            if isinstance(self.session, Exception):
                raise self.session
            return [self.session] if isinstance(self.session, tuple) else self.session
        return super().execute(sql, *args, **kwargs)


def _job(snapshot, job_id: str = SUPERSET_ORACLE_JOB) -> Job | None:
    return next((job for job in snapshot.jobs if job.id == job_id), None)


def _detail(snapshot) -> dict:
    return json.loads(_job(snapshot).detail)


@pytest.fixture
def shop_file(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SHOP)
    con.commit()
    con.close()
    return path


# --- the job enrichment writes ---------------------------------------------------------------------------------------


def test_the_job_records_the_service_and_schema_the_connection_reads(shop_file):
    adapter = _OracleSession(SQLiteAdapter(str(shop_file)))
    snapshot = enrich_structural(adapter, "shop", superset_oracle=(ORACLE_SERVICE, ORACLE_SCHEMA))
    job = _job(snapshot)
    assert (job.kind, job.status, job.source_id) == ("superset", "done", "shop")
    # Exactly these fields, in this order: the detail is part of the snapshot's version.
    assert job.detail == ('{"backend": "oracle", "service": "freepdb1", "schema": "SHOP", "values": "harvested", '
                          '"profile": "exact"}')
    assert _job(snapshot, SUPERSET_SOURCE_JOB) is None


def test_the_job_says_how_the_run_profiled_and_whether_it_harvested(shop_file):
    adapter = _OracleSession(SQLiteAdapter(str(shop_file)))
    binding = (ORACLE_SERVICE, ORACLE_SCHEMA)
    statistics = enrich_structural(adapter, "shop", superset_oracle=binding, statistics=())
    assert (_detail(statistics)["profile"], _detail(statistics)["values"]) == ("statistics", "harvested")
    exact_tables = enrich_structural(adapter, "shop", superset_oracle=binding, statistics=("orders",))
    assert _detail(exact_tables)["profile"] == "statistics"
    unharvested = enrich_structural(adapter, "shop", superset_oracle=binding, harvest_values=False)
    assert (_detail(unharvested)["profile"], _detail(unharvested)["values"]) == ("exact", "not harvested")


def test_a_snapshot_is_enriched_through_superset_or_from_oracle_for_it_not_both(shop_file):
    adapter = _OracleSession(SQLiteAdapter(str(shop_file)))
    with pytest.raises(ValueError, match="not both"):
        enrich_structural(adapter, "shop", harvest_values=False, superset_source=(7, "public"),
                          superset_oracle=(ORACLE_SERVICE, ORACLE_SCHEMA))


def test_postgresqls_job_is_written_as_it_was(shop_file):
    """The `superset:source` job enrichment through Superset writes, byte for byte, and the version of a snapshot
    carrying it, as computed before the Oracle job existed."""
    snapshot = enrich_structural(SQLiteAdapter(str(shop_file)), "shop", harvest_values=False,
                                 superset_source=(7, "public"))
    assert _job(snapshot, SUPERSET_SOURCE_JOB).detail == (
        '{"database_id": 7, "schema": "public", "backend": "postgresql", "values": "not harvested"}')
    assert _job(snapshot) is None
    assert content_version(_snapshot([_source_job()])) == "4f2edab9f727"
    assert content_version(_snapshot([])) == "22f26166da5b"


def test_the_job_is_part_of_the_snapshots_version():
    """Two snapshots alike but for the job are not the same snapshot: one is served in Superset mode and the other is
    not, and one read from another service or schema describes another database. The store keys a snapshot by its
    version, so without this a run for Superset mode would be stored as the run without it that it repeats."""
    bare = content_version(_snapshot([]))
    versions = {content_version(_snapshot([_oracle_job(**fields)])) for fields in
                ({}, {"service": "otherpdb"}, {"schema": "SHOP2"}, {"values": "not harvested"},
                 {"profile": "statistics"})}
    assert len(versions) == 5 and bare not in versions


def test_the_session_is_read_with_one_statement():
    class Session:
        sent: list[str] = []

        def execute(self, sql, timeout_s=None):
            self.sent.append(sql)
            return [("freepdb1", "SHOP")]

    session = Session()
    assert pipeline.oracle_binding(session) == ("freepdb1", "SHOP")
    assert session.sent == [BINDING_SQL]


@pytest.mark.parametrize("rows", [[], [("freepdb1",)], [("freepdb1", "SHOP"), ("freepdb1", "SHOP")],
                                  [(None, "SHOP")], [("freepdb1", None)], [("", "SHOP")], [("freepdb1", "")],
                                  [(1, "SHOP")]],
                         ids=["no-row", "one-column", "two-rows", "no-service", "no-schema", "blank-service",
                              "blank-schema", "service-not-text"])
def test_a_session_read_that_says_nothing_usable_is_refused(rows):
    class Session:
        def execute(self, sql, timeout_s=None):
            return rows

    with pytest.raises(ValueError, match="service and current schema"):
        pipeline.oracle_binding(Session())


# --- `mnemiq enrich --for-superset` -----------------------------------------------------------------------------------


@pytest.fixture
def oracle_source(tmp_path, shop_file, monkeypatch):
    """A deployment whose one source is Oracle, reached through `_OracleSession` over the shop, with semantic
    enrichment stubbed: `source(kind=..., profile=...)` gives its settings and the adapter every run reads."""
    adapter = _OracleSession(SQLiteAdapter(str(shop_file)))
    monkeypatch.setattr(resolve_module, "_adapter_for", lambda *a, **k: adapter)
    runs: list[str] = []

    def described(snap, enricher, **kwargs):
        runs.append("called")
        return snap

    monkeypatch.setattr("mnemiq.enrichment.semantic.enrich_semantic", described)

    def source(kind: str = "oracle", **entry) -> Settings:
        manifest = tmp_path / "sources.json"
        manifest.write_text(json.dumps([{"id": SOURCE_ID, "kind": kind, "target": "127.0.0.1:1521/FREEPDB1",
                                         "catalog": "shop", "schema": ORACLE_SCHEMA, **entry}]))
        return Settings(sources_path=str(manifest), store_path=str(tmp_path / "store.duckdb"),
                        oracle_user="reader", oracle_password="pw", llm_base_url="http://127.0.0.1:9/v1",
                        llm_api_key="x", llm_model="m", enrich_examples=False, enrich_facts=False)

    source.adapter = adapter
    source.runs = runs
    return source


def _stored(settings: Settings):
    con = init_store(settings.store_path)
    try:
        version = current_version(con, SOURCE_ID)
        return load_snapshot(con, version) if version else None
    finally:
        con.close()


def test_enrich_for_superset_writes_the_job_from_the_connections_own_session(oracle_source):
    settings = oracle_source()
    assert _cmd_enrich(settings, for_superset=True) == 0
    snapshot = _stored(settings)
    assert _detail(snapshot) == {"backend": "oracle", "service": ORACLE_SERVICE, "schema": ORACLE_SCHEMA,
                                 "values": "harvested", "profile": "exact"}
    # Read once, from the connection enrichment reads the tables through, before any table is read.
    assert oracle_source.adapter.sql.count(BINDING_SQL) == 1
    assert oracle_source.adapter.sql.index(BINDING_SQL) == 0


def test_enrich_for_superset_records_what_the_session_says_not_the_manifest(oracle_source):
    """The schema is the session's CURRENT_SCHEMA, the one Oracle resolves the snapshot's unqualified names in, whatever
    the manifest spells; the service is the one the connection reached."""
    oracle_source.adapter.session = ("otherpdb", "SALES")
    settings = oracle_source()
    assert _cmd_enrich(settings, for_superset=True) == 0
    assert (_detail(_stored(settings))["service"], _detail(_stored(settings))["schema"]) == ("otherpdb", "SALES")


def test_enrich_for_superset_in_statistics_mode_says_so(oracle_source):
    settings = oracle_source(profile="statistics")
    assert _cmd_enrich(settings, for_superset=True) == 0
    assert _detail(_stored(settings))["profile"] == "statistics"


def test_enrich_without_the_flag_writes_no_job(oracle_source):
    settings = oracle_source()
    assert _cmd_enrich(settings) == 0
    assert _job(_stored(settings)) is None
    assert BINDING_SQL not in oracle_source.adapter.sql


@pytest.mark.parametrize("kind", ["postgres", "sqlite", "duckdb"])
def test_enrich_for_superset_refuses_a_source_that_is_not_oracle(oracle_source, monkeypatch, capsys, kind):
    """Refused before any connection is opened: a snapshot for Superset mode on PostgreSQL is enriched through Superset."""
    def reached(*args, **kwargs):
        raise AssertionError("a connection was opened")

    monkeypatch.setattr(resolve_module, "_adapter_for", reached)
    settings = oracle_source(kind=kind)
    assert _cmd_enrich(settings, for_superset=True) == 1
    err = capsys.readouterr().err
    assert "--for-superset" in err and "Oracle" in err and kind in err
    assert _stored(settings) is None and oracle_source.runs == []


def test_enrich_for_superset_refuses_in_superset_mode(oracle_source, capsys):
    settings = oracle_source().model_copy(update={"superset_mcp_url": "http://127.0.0.1:9/mcp",
                                                  "superset_database_id": 7})
    assert _cmd_enrich(settings, for_superset=True) == 1
    assert "MNEMIQ_SUPERSET_MCP_URL" in capsys.readouterr().err
    assert oracle_source.adapter.sql == [] and oracle_source.runs == []


@pytest.mark.parametrize("session", [RuntimeError("ORA-00942: no such view"), []],
                         ids=["the-read-fails", "the-read-answers-nothing"])
def test_enrich_for_superset_refuses_when_the_session_cannot_be_read(oracle_source, capsys, session):
    oracle_source.adapter.session = session
    settings = oracle_source()
    assert _cmd_enrich(settings, for_superset=True) == 1
    assert "service and current schema" in capsys.readouterr().err
    # Before any table is read or any model call is made, and nothing is saved.
    assert oracle_source.adapter.sql == [BINDING_SQL]
    assert oracle_source.runs == [] and _stored(settings) is None


def test_the_command_line_flag_reaches_enrich(monkeypatch):
    from mnemiq import cli

    seen = []
    monkeypatch.setattr(cli, "_cmd_enrich", lambda settings, add=False, for_superset=False:
                        seen.append((add, for_superset)) and 0)
    cli.main(["enrich", "--for-superset"])
    cli.main(["enrich", "--add", "--for-superset"])
    cli.main(["enrich"])
    assert seen == [(False, True), (True, True), (False, False)]


# --- --add and refresh keep a snapshot for Superset mode one -----------------------------------------------------------


def test_add_for_superset_merges_into_a_snapshot_enriched_for_superset(oracle_source):
    """The merged snapshot carries one job for all its tables: statistics where either run read statistics."""
    assert _cmd_enrich(oracle_source(tables=["orders"]), for_superset=True) == 0
    settings = oracle_source(tables=["customers"], profile="statistics")
    assert _cmd_enrich(settings, add=True, for_superset=True) == 0
    snapshot = _stored(settings)
    assert {b.object_id for b in snapshot.source_bindings} == {"orders", "customers"}
    assert [job.id for job in snapshot.jobs].count(SUPERSET_ORACLE_JOB) == 1
    assert _detail(snapshot) == {"backend": "oracle", "service": ORACLE_SERVICE, "schema": ORACLE_SCHEMA,
                                 "values": "harvested", "profile": "statistics"}
    assert snapshot.version == content_version(snapshot)


@pytest.mark.parametrize("first, then, says", [
    (False, True, "run --for-superset without --add first"),
    (True, False, "pass --for-superset"),
], ids=["to-a-snapshot-not-for-superset", "without-the-flag-to-one-for-superset"])
def test_add_refuses_to_mix_a_run_for_superset_with_one_not(oracle_source, capsys, first, then, says):
    assert _cmd_enrich(oracle_source(tables=["orders"]), for_superset=first) == 0
    runs = list(oracle_source.runs)
    settings = oracle_source(tables=["customers"])
    assert _cmd_enrich(settings, add=True, for_superset=then) == 1
    assert says in capsys.readouterr().err
    assert oracle_source.runs == runs, "refused before any model call"
    assert {b.object_id for b in _stored(settings).source_bindings} == {"orders"}


def test_add_for_superset_refuses_a_snapshot_read_from_another_schema(oracle_source, capsys):
    assert _cmd_enrich(oracle_source(tables=["orders"]), for_superset=True) == 0
    oracle_source.adapter.session = (ORACLE_SERVICE, "SHOP2")
    settings = oracle_source(tables=["customers"])
    assert _cmd_enrich(settings, add=True, for_superset=True) == 1
    err = capsys.readouterr().err
    assert "'SHOP2'" in err and "without --add" in err


def test_a_merge_keeps_the_job_only_where_both_runs_carry_it_for_one_schema():
    """`merge_run` itself, past the command's own refusals: the job vouches for every table, so a merge whose runs do
    not both carry it, for one service and schema and finished, carries none, and Superset mode will not serve it."""
    def merged(kept_jobs, run_jobs):
        kept = _snapshot(kept_jobs).model_copy(update={"source_id": "only"})
        run = _snapshot(run_jobs).model_copy(update={"source_id": "only"})
        snapshot, _ = merge_run(kept, run)
        return [job for job in snapshot.jobs if job.id == SUPERSET_ORACLE_JOB]

    assert merged([], [_oracle_job()]) == []
    assert merged([_oracle_job()], []) == []
    assert merged([_oracle_job(schema="SHOP2")], [_oracle_job()]) == []
    assert merged([_oracle_job(service="otherpdb")], [_oracle_job()]) == []
    assert merged([_oracle_job(status="failed")], [_oracle_job()]) == []
    assert merged([_oracle_job(detail="not json")], [_oracle_job()]) == []
    [job] = merged([_oracle_job(values="not harvested", profile="statistics")],
                   [_oracle_job(values="not harvested")])
    assert json.loads(job.detail) == json.loads(_oracle_job(values="not harvested", profile="statistics").detail)
    [job] = merged([_oracle_job(values="not harvested")], [_oracle_job()])
    assert json.loads(job.detail)["values"] == "harvested"
    [job] = merged([_oracle_job()], [_oracle_job(values="not harvested")])
    assert json.loads(job.detail)["values"] == "harvested"


def test_a_merge_reads_the_job_as_the_sources_own_beside_a_table_named_for_it():
    """A kept table named `oracle` does not take the job as its own: the job is about the whole source."""
    from mnemiq.contract import Column
    from mnemiq.contract.semantic import SourceBinding

    kept = _snapshot([_oracle_job()]).model_copy(update={
        "columns": [Column(id="oracle.col_0", object_id="oracle", name="col_0", data_type="VARCHAR")],
        "source_bindings": [SourceBinding(id="sb:oracle", source_id=SOURCE_ID, object_id="oracle",
                                          source_object="oracle", binding_type="table")]})
    snapshot, _ = merge_run(kept, _snapshot([_oracle_job(profile="statistics")]))
    assert {b.object_id for b in snapshot.source_bindings} == {"oracle", "wide"}
    [job] = [job for job in snapshot.jobs if job.id == SUPERSET_ORACLE_JOB]
    assert json.loads(job.detail)["profile"] == "statistics"


def test_refresh_enriches_a_snapshot_for_superset_again_for_superset(oracle_source, monkeypatch):
    settings = oracle_source()
    assert _cmd_enrich(settings, for_superset=True) == 0
    monkeypatch.setattr("mnemiq.cli._cmd_build", lambda settings: 0)
    # A changed catalogue, so refresh enriches again.
    monkeypatch.setattr("mnemiq.enrichment.refresh.catalog_diff", lambda adapter, snapshot: type(
        "Diff", (), {"added": ["x"], "changed": [], "dropped": [], "has_changes": True})())
    before = oracle_source.adapter.sql.count(BINDING_SQL)
    assert _cmd_refresh(settings) == 0
    assert oracle_source.adapter.sql.count(BINDING_SQL) == before + 1
    assert _job(_stored(settings)) is not None


# --- the gate a server in Superset mode holds the snapshot to ----------------------------------------------------------


def _settings(schema: str = ORACLE_SCHEMA_SETTING) -> Settings:
    return Settings(superset_mcp_url="http://127.0.0.1:9/mcp", superset_database_id=7, superset_schema=schema)


def test_the_gate_serves_a_snapshot_read_from_the_schema_superset_shows_as_the_setting():
    """MNEMIQ_SUPERSET_SCHEMA in Superset's spelling (`shop`), the snapshot's schema in Oracle's (`SHOP`)."""
    assert superset_source_refusal(_snapshot([_oracle_job()]), _settings()) is None
    for values, profile in (("not harvested", "exact"), ("harvested", "statistics")):
        assert superset_source_refusal(_snapshot([_oracle_job(values=values, profile=profile)]), _settings()) is None


@pytest.mark.parametrize("stored, setting, served", [
    ("SHOP", "shop", True),
    ("SHOP2", "shop", False),
    ("SHOP", "shop2", False),
    # A schema created quoted in lower case is stored `shop`; Superset shows it `shop`, as it shows SHOP, and reads a
    # `shop` it is given as SHOP, so the setting names SHOP and not the snapshot's schema. Compared the other way, as
    # Superset shows the snapshot's schema, the two would be taken for one.
    ("shop", "shop", False),
    # The setting in Oracle's own spelling names the schema, but Superset's datasets name it `shop`: refused, with the
    # spelling to set.
    ("SHOP", "SHOP", False),
    # A mixed-case quoted schema is shown and read as stored, case kept.
    ("Shop", "Shop", True),
    ("Shop", "shop", False),
    ("SHOP", "Shop", False),
], ids=["superset-spelling", "another-schema", "another-setting", "quoted-lower-case", "oracle-spelling",
        "mixed-case", "mixed-case-against-lower", "upper-against-mixed"])
def test_the_gate_matches_the_schema_through_the_name_mapping(stored, setting, served):
    refusal = superset_source_refusal(_snapshot([_oracle_job(schema=stored)]), _settings(setting))
    assert (refusal is None) is served, refusal
    if not served:
        assert f"{stored!r}" in refusal and "MNEMIQ_SUPERSET_SCHEMA" in refusal


@pytest.mark.parametrize("stored, setting, spelling", [("SHOP", "SHOP", "shop"), ("ORDERS_2026", "ORDERS_2026",
                                                                            "orders_2026")])
def test_a_setting_in_oracles_spelling_is_refused_naming_the_one_superset_shows(stored, setting, spelling):
    """MNEMIQ_SUPERSET_SCHEMA must be the schema's name as Superset shows it, the name its datasets record: the
    operator is told the spelling to set. The person is told the deployment's own sentence, as for any snapshot it
    will not serve."""
    refusal = superset_source_refusal(_snapshot([_oracle_job(schema=stored)]), _settings(setting))
    assert refusal is not None and "MNEMIQ_SUPERSET_SCHEMA" in refusal and f"{spelling!r}" in refusal
    assert superset_source_refusal(_snapshot([_oracle_job(schema=stored)]), _settings(spelling)) is None
    assert superset_dialect() == "postgres"  # judged before boot sets the dialect, as Oracle's mapping reads it


def test_the_gate_compares_the_name_superset_reads_for_the_setting():
    """The direction pinned: the stored name Superset reads for the setting against the snapshot's, the two cases where
    it differs from comparing the snapshot's name as Superset shows it against the setting."""
    from mnemiq.superset.names import stored_name, superset_name

    use_superset_dialect("oracle")
    assert superset_name("shop") == "shop" and stored_name("shop") == "SHOP"
    assert superset_name("SHOP") == "shop" and stored_name("SHOP") == "SHOP"


def test_the_gate_reads_the_mapping_as_oracles_before_boot_sets_the_dialect():
    """Boot judges the snapshot before it sets the dialect, which is PostgreSQL's until then: the gate maps the setting
    as Oracle's mapping does whatever the process's dialect reads, and leaves the dialect unset."""
    assert superset_dialect() == "postgres"
    assert superset_source_refusal(_snapshot([_oracle_job()]), _settings()) is None
    assert superset_source_refusal(_snapshot([_oracle_job(schema="shop")]), _settings()) is not None
    use_superset_dialect("oracle")  # refused, had the gate set one


@pytest.mark.parametrize("job, says", [
    (_oracle_job(without=("service",)), "cannot be read"),
    (_oracle_job(without=("schema",)), "cannot be read"),
    (_oracle_job(without=("backend",)), "cannot be read"),
    (_oracle_job(without=("values",)), "cannot be read"),
    (_oracle_job(without=("profile",)), "cannot be read"),
    (_oracle_job(database_id=7), "cannot be read"),
    (_oracle_job(backend="postgresql"), "cannot be read"),
    (_oracle_job(backend="Oracle"), "cannot be read"),
    (_oracle_job(backend="oracle+oracledb"), "cannot be read"),
    (_oracle_job(values="some"), "cannot be read"),
    (_oracle_job(values=None), "cannot be read"),
    (_oracle_job(profile="sampled"), "cannot be read"),
    (_oracle_job(service=""), "cannot be read"),
    (_oracle_job(service=1), "cannot be read"),
    (_oracle_job(service=None), "cannot be read"),
    (_oracle_job(schema=""), "cannot be read"),
    (_oracle_job(schema=["SHOP"]), "cannot be read"),
    (_oracle_job(detail="not json"), "cannot be read"),
    (_oracle_job(detail=json.dumps(["backend", "oracle"])), "cannot be read"),
    (_oracle_job(detail=None).model_copy(update={"detail": None}), "cannot be read"),
    (_oracle_job(status="failed"), "did not finish"),
    (_oracle_job(status="running"), "did not finish"),
], ids=["no-service", "no-schema", "no-backend", "no-values", "no-profile", "an-extra-field", "postgresql",
        "backend-case", "backend-driver", "values-other", "values-null", "profile-other", "service-blank",
        "service-number", "service-null", "schema-blank", "schema-list", "not-json", "not-an-object", "no-detail",
        "failed", "running"])
def test_the_gate_refuses_an_oracle_job_it_cannot_read_exactly(job, says):
    refusal = superset_source_refusal(_snapshot([job]), _settings())
    assert refusal is not None and says in refusal and SUPERSET_ORACLE_JOB in refusal


def test_the_gate_refuses_a_snapshot_recording_both_jobs():
    refusal = superset_source_refusal(_snapshot([_source_job(), _oracle_job()]),
                                      _settings().model_copy(update={"superset_schema": "public"}))
    assert refusal is not None and SUPERSET_SOURCE_JOB in refusal and SUPERSET_ORACLE_JOB in refusal


def test_the_gate_refuses_two_oracle_jobs_when_either_names_another_schema():
    assert superset_source_refusal(_snapshot([_oracle_job(), _oracle_job()]), _settings()) is None
    assert "MNEMIQ_SUPERSET_SCHEMA" in superset_source_refusal(_snapshot([_oracle_job(), _oracle_job(schema="X")]),
                                                               _settings())


def test_the_gate_still_holds_postgresqls_job_to_its_own_rules():
    """The `superset:source` job is read as before: no Oracle field is taken there, nor its backend."""
    settings = _settings().model_copy(update={"superset_schema": "public"})
    assert superset_source_refusal(_snapshot([_source_job()]), settings) is None
    assert "cannot be read" in superset_source_refusal(_snapshot([_source_job(backend="oracle")]), settings)
    assert "cannot be read" in superset_source_refusal(_snapshot([_source_job(values="harvested")]), settings)
    assert "cannot be read" in superset_source_refusal(_snapshot([_source_job(profile="exact")]), settings)


def test_the_gate_says_how_to_enrich_a_snapshot_for_oracle_when_none_was():
    refusal = superset_source_refusal(_snapshot([]), _settings())
    assert "was not enriched through Superset" in refusal and "--for-superset" in refusal


# --- boot and a swap --------------------------------------------------------------------------------------------------


@pytest.fixture
def served():
    """Superset and the model, each a real server on loopback that records every call."""
    with FakeSuperset() as fake, StubModel(sql="SELECT 1") as model:
        yield fake, model


def _deployment(tmp_path, served, jobs) -> Settings:
    fake, model = served
    schema = ORACLE_SCHEMA_SETTING if any(job.id == SUPERSET_ORACLE_JOB for job in jobs) else "public"
    return superset_deployment(tmp_path, model.url, fake.url, jobs=jobs, superset_schema=schema)


def test_boot_sets_oracles_dialect_for_a_snapshot_read_from_oracle(tmp_path, served):
    fake, model = served
    runtime = build_runtime(_deployment(tmp_path, served, [_oracle_job()]))
    assert superset_dialect() == "oracle" and runtime.adapter.dialect == "oracle"
    assert fake.calls == [] and model.calls == 0  # booting asks Superset nothing


def test_boot_sets_postgresqls_dialect_for_a_snapshot_enriched_through_superset(tmp_path, served):
    runtime = build_runtime(_deployment(tmp_path, served, [_source_job()]))
    assert superset_dialect() == "postgres" and runtime.adapter.dialect == "postgres"


def test_boot_refuses_an_oracle_snapshot_for_another_schema_and_sets_no_dialect(tmp_path, served):
    fake, model = served
    settings = superset_deployment(tmp_path, model.url, fake.url, jobs=[_oracle_job(schema="SHOP2")],
                                   superset_schema=ORACLE_SCHEMA_SETTING)
    with pytest.raises(SnapshotMissing, match="MNEMIQ_SUPERSET_SCHEMA"):
        build_runtime(settings)
    use_superset_dialect("postgres")  # refused, had the refused boot set Oracle's


@pytest.mark.parametrize("first, second", [("oracle", "postgres"), ("postgres", "oracle")])
def test_a_second_runtime_in_another_dialect_is_refused(tmp_path, served, first, second):
    jobs = {"oracle": [_oracle_job()], "postgres": [_source_job()]}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    build_runtime(_deployment(tmp_path / "a", served, jobs[first]))
    build_runtime(_deployment(tmp_path / "b", served, jobs[first]))  # the same dialect again is no change
    with pytest.raises(RuntimeError, match=f"already {first} in this process"):
        build_runtime(_deployment(tmp_path / "c", served, jobs[second]))
    assert superset_dialect() == first


def _swap(runtime, monkeypatch, snapshot) -> None:
    runtime.settings = runtime.settings.model_copy(update={"control_dsn": "postgresql://control"})
    monkeypatch.setattr(runtime_module, "load_current_snapshot",
                        lambda settings, con: (snapshot, {SOURCE_ID: snapshot.version}))


@pytest.mark.parametrize("boots, published", [("postgres", "oracle"), ("oracle", "postgres")])
def test_a_swap_to_a_snapshot_read_from_another_database_is_refused(tmp_path, served, monkeypatch, caplog, boots,
                                                                    published):
    """A newer snapshot in another dialect is refused: the dialect is set once, at boot. The snapshot serving is kept
    in memory and answers nothing, as for any snapshot Superset mode will not serve, and the log says to restart."""
    fake, model = served
    # Each one the gate accepts for the same settings, so only the dialect refuses the swap.
    jobs = {"oracle": [_oracle_job()], "postgres": [_source_job(schema=ORACLE_SCHEMA_SETTING)]}
    runtime = build_runtime(superset_deployment(tmp_path, model.url, fake.url, jobs=jobs[boots],
                                                superset_schema=ORACLE_SCHEMA_SETTING))
    serving = runtime.snapshot
    newer = _snapshot(jobs[published], version="v2")
    assert superset_source_refusal(newer, runtime.settings) is None
    _swap(runtime, monkeypatch, newer)
    with caplog.at_level("WARNING", logger="mnemiq.runtime"):
        with pytest.raises(SupersetSnapshotRefused) as refused:
            runtime.reload_if_stale()
    # The person's sentence is the one for the database this server answers from.
    assert str(refused.value) == {"postgres": SNAPSHOT_REFUSED, "oracle": ORACLE_SNAPSHOT_REFUSED}[boots]
    assert f"read from {published}" in refused.value.detail and "restart" in refused.value.detail
    assert any("restart" in record.getMessage() for record in caplog.records)
    assert runtime.snapshot is serving and superset_dialect() == boots
    assert fake.calls == [] and model.calls == 0


# What a person asking a server that answers from Oracle is told when the snapshot published for it is not one it will
# serve, word for word: what to do differs from PostgreSQL's, whose sentence is unchanged (`superset_shared`).
ORACLE_SNAPSHOT_REFUSED = ("This deployment's knowledge is not the one enriched for the Oracle database it answers "
                           "from, so nothing is answered until an operator enriches it directly from Oracle for "
                           "Superset mode (`mnemiq enrich --for-superset`); mnemiq's log records why.")


@pytest.mark.parametrize("published", [[], [_oracle_job(schema="SHOP2")], [_oracle_job(status="failed")],
                                       [_source_job(schema=ORACLE_SCHEMA_SETTING)]],
                         ids=["no-job", "another-schema", "not-finished", "read-from-postgresql"])
def test_a_server_for_oracle_tells_the_person_to_enrich_from_oracle(tmp_path, served, monkeypatch, published):
    """The sentence a refused swap gives the person, on every door (`superset_refusal`), names what an operator does
    for this deployment, which answers from Oracle: enrich directly from Oracle for Superset mode."""
    runtime = build_runtime(_deployment(tmp_path, served, [_oracle_job()]))
    _swap(runtime, monkeypatch, _snapshot(published, version="v2"))
    with pytest.raises(SupersetSnapshotRefused) as refused:
        runtime.reload_if_stale()
    assert str(refused.value) == refused.value.message == ORACLE_SNAPSHOT_REFUSED
    assert superset_refusal(refused.value) == Refused(503, ORACLE_SNAPSHOT_REFUSED)
    assert superset_settings.SNAPSHOT_REFUSED_ORACLE == ORACLE_SNAPSHOT_REFUSED


def test_a_server_for_postgresql_tells_the_person_what_it_always_did(tmp_path, served, monkeypatch):
    runtime = build_runtime(_deployment(tmp_path, served, [_source_job()]))
    _swap(runtime, monkeypatch, _snapshot([_oracle_job(schema="SHOP2")], version="v2"))
    with pytest.raises(SupersetSnapshotRefused) as refused:
        runtime.reload_if_stale()
    # Word for word as before (`superset_shared.SNAPSHOT_REFUSED` is the literal the HTTP tests read).
    assert str(refused.value) == refused.value.message == SHARED_SNAPSHOT_REFUSED == SNAPSHOT_REFUSED
    assert superset_refusal(refused.value) == Refused(503, SHARED_SNAPSHOT_REFUSED)


def test_the_schema_setting_is_described_as_superset_shows_the_schema():
    """On both databases MNEMIQ_SUPERSET_SCHEMA is the schema's name as Superset shows it: on Oracle that is not the
    name Oracle stores, so the setting is not described by PostgreSQL's storage."""
    from mnemiq.config import Settings as Configured

    description = Configured.model_fields["superset_schema"].description
    assert "as Superset shows it" in description and "Oracle" in description
    assert "spelled as Postgres stores it" not in description


def test_a_swap_to_a_newer_oracle_snapshot_is_served(tmp_path, served, monkeypatch):
    runtime = build_runtime(_deployment(tmp_path, served, [_oracle_job()]))
    newer = _snapshot([_oracle_job(profile="statistics")], version="v2")
    _swap(runtime, monkeypatch, newer)
    runtime.reload_if_stale()
    assert runtime.snapshot is newer and superset_dialect() == "oracle"


# --- the whole path, without a database --------------------------------------------------------------------------------


def test_a_snapshot_enriched_for_superset_is_served_in_superset_mode(oracle_source, served, tmp_path):
    """Enriched with --for-superset into a store, then booted from that store in Superset mode: the setting in
    Superset's spelling matches the session's schema, and the dialect is Oracle's."""
    fake, model = served
    settings = oracle_source()
    assert _cmd_enrich(settings, for_superset=True) == 0
    serving = superset_deployment(tmp_path, model.url, fake.url, snapshot=False,
                                  superset_schema=ORACLE_SCHEMA_SETTING).model_copy(
        update={"store_path": settings.store_path})
    runtime = build_runtime(serving)
    assert runtime.snapshot.version == _stored(settings).version
    assert superset_dialect() == "oracle" and fake.calls == []


# --- live: the Oracle container ----------------------------------------------------------------------------------------

LIVE_DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
LIVE_USER = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
LIVE_PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")


@pytest.fixture
def live_table():
    """A table of this test's own in the test user's schema, dropped after it."""
    if not LIVE_DSN:
        pytest.skip("set MNEMIQ_ORACLE_TEST_DSN to run the live Oracle enrichment test")
    oracledb = pytest.importorskip("oracledb", reason="the Oracle adapter needs mnemiq[oracle]")
    con = oracledb.connect(user=LIVE_USER, password=LIVE_PASSWORD, dsn=LIVE_DSN)
    name = f"SNAPSHOT_PROBE_{secrets.token_hex(4).upper()}"
    cur = con.cursor()
    cur.execute(f"CREATE TABLE {name} (ID NUMBER PRIMARY KEY, STATUS VARCHAR2(10))")
    cur.execute(f"INSERT INTO {name} VALUES (1, 'A')")
    cur.execute(f"INSERT INTO {name} VALUES (2, 'B')")
    con.commit()
    cur.execute("SELECT SYS_CONTEXT('USERENV', 'SERVICE_NAME'), SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL")
    session = cur.fetchone()
    try:
        yield name, session
    finally:
        cur.execute(f"DROP TABLE {name} PURGE")
        con.close()


@pytest.mark.integration
def test_live_enrich_for_superset_records_the_containers_session_and_boots_in_oracles_dialect(
        live_table, tmp_path, served, monkeypatch):
    fake, model = served
    name, (service, schema) = live_table
    monkeypatch.setattr("mnemiq.enrichment.semantic.enrich_semantic", lambda snap, enricher, **kwargs: snap)
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": SOURCE_ID, "kind": "oracle", "target": LIVE_DSN, "catalog": "live",
                                     "schema": LIVE_USER, "tables": [name]}]))
    store = tmp_path / "store.duckdb"
    settings = Settings(sources_path=str(manifest), store_path=str(store), oracle_user=LIVE_USER,
                        oracle_password=LIVE_PASSWORD, llm_base_url="http://127.0.0.1:9/v1", llm_api_key="x",
                        llm_model="m", enrich_examples=False, enrich_facts=False)
    assert _cmd_enrich(settings, for_superset=True) == 0
    snapshot = _stored(settings)
    assert [b.object_id for b in snapshot.source_bindings] == [name]
    assert _detail(snapshot) == {"backend": "oracle", "service": service, "schema": schema, "values": "harvested",
                                 "profile": "exact"}
    assert schema == LIVE_USER.upper()
    # Served in Superset mode with the schema as Superset shows the test user's: lower case.
    serving = superset_deployment(tmp_path, model.url, fake.url, snapshot=False,
                                  superset_schema=LIVE_USER.lower()).model_copy(update={"store_path": str(store)})
    runtime = build_runtime(serving)
    assert runtime.snapshot.version == snapshot.version and superset_dialect() == "oracle"
    assert fake.calls == []

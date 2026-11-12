"""Enrichment in Superset mode: `mnemiq enrich` reads the source through Superset's MCP service as a service user whose
API key sits in a file only the CLI's enrichment reads. It opens no database connection of its own, harvests no values,
sends `force_refresh` with every statement, and records the Superset database and schema it read in the job the server
checks before it serves the snapshot.

Superset is `FakeSuperset`, a real MCP server on loopback, so every call crosses the wire carrying the bearer the fake
records. Behind it, `SourceBehindSuperset` answers `execute_sql` as Superset would over a small Postgres schema: each
catalogue statement from the schema's catalogue rows, and any other statement run by DuckDB over the schema's tables.
The model is the scripted one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import threading

import duckdb
import psycopg
import pytest

import mnemiq.adapters.resolve as resolve_module
import mnemiq.runtime as runtime_module
from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.cli import _cmd_enrich, _cmd_refresh
from mnemiq.config import Settings
from mnemiq.contract.semantic import CodeScheme, Job
from mnemiq.enrichment.grounding import ground_codes
from mnemiq.enrichment.pipeline import content_version, enrich_structural
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords
from mnemiq.runtime import SnapshotMissing, build_runtime
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode
from mnemiq.store.bootstrap import init_store
from mnemiq.semantic.values import ValueIndex
from mnemiq.store.snapshot_store import current_version, load_snapshot, save_snapshot
from mnemiq.superset.adapter import TRANSPORT_MARGIN_S
from mnemiq.superset.client import DEFAULT_TIMEOUT_S, Bearer, DelegatedToken, SupersetMcp, SupersetUnavailable
from mnemiq.superset.executor import NoSourceConnection, superset_executor
from mnemiq.superset.pg_catalog import CATALOGUE_SQL, PAGE_ROWS, PostgresCatalogue
from mnemiq.superset.service import (
    SERVICE_KEY_ENV,
    ServiceKey,
    enrichment_adapter,
    service_credential_from_env,
)
from mnemiq.superset.settings import SUPERSET_SOURCE_JOB, SupersetConfigError
from tests.superset_fake import FakeSuperset
from tests.sweep.stub_model import StubModel
from tests.test_superset_runtime import PROFILES, DATABASE_ID, SCHEMA, SOURCE_ID, _answer_rows, superset_deployment

# The service key, made for this run. Each test that reads it checks it never shows: in a message, a log line, a
# repr, or a bearer on the ask path.
KEY = f"sst_x{secrets.token_hex(12)}"
DSN = "postgresql://reader@127.0.0.1:5432/warehouse"
# The most a service key file may hold, in bytes.
KEY_FILE_LIMIT = 64 * 1024
# Words only Superset says in these tests, so where they end up can be read.
SAID = "superset-said-this"

# The schema behind Superset: each table's columns as `format_type` names their types, and its rows. `claim.region` is
# a small vocabulary profiling would harvest, and `claim.status_id` a key whose labels lookup grounding would read from
# `claim_status`; neither may reach a snapshot enriched through Superset.
TABLES: dict[str, list[tuple[str, str]]] = {
    "claim_status": [("status_id", "integer"), ("status_name", "text")],
    "claim": [("claim_id", "integer"), ("status_id", "integer"), ("region", "text"), ("amount", "integer")],
}
ROWS: dict[str, list[tuple]] = {
    "claim_status": [(1, "Lodged"), (2, "Settled"), (3, "Reopened")],
    "claim": [(1, 1, "Northgate", 10), (2, 2, "Southmere", 20), (3, 1, "Northgate", 30), (4, 3, "Eastwick", 40),
              (5, 2, "Southmere", 50), (6, 1, "Northgate", 60)],
}
# Every text value the tables hold: none may be in a snapshot or a value index built through Superset.
CELL_TEXT = frozenset(value for rows in ROWS.values() for row in rows for value in row if isinstance(value, str))
FOREIGN_KEYS = [("claim", "status_id", "claim_status", "status_id", "claim_status_id_fkey", 1)]
VIEWS = [("active_claims", " SELECT claim.claim_id\n   FROM claim\n  WHERE claim.status_id = 1;")]
# The column names each catalogue statement selects, as `pg_catalog`'s statements name them.
CATALOGUE_COLUMNS = {
    "tables": ("table_name",),
    "columns": ("table_name", "column_name", "data_type", "column_number"),
    "foreign_keys": ("from_table", "from_column", "to_table", "to_column", "constraint_name", "key_position"),
    "views": ("view_name", "definition"),
    "user_functions": ("function_name",),
    "builtin_functions": ("function_name",),
    "reachable_user_functions": ("function_name",),
    "virtual_columns": ("table_name", "column_name", "expression", "column_number"),
}
_DUCKDB_TYPES = {"integer": "INTEGER", "text": "VARCHAR"}
# The statement profiling's `_top_k` sends to read a column's values, as it is written for Postgres.
_TOP_K = re.compile(r"COUNT\(\*\) AS c\b.*GROUP BY", re.IGNORECASE | re.DOTALL)
# What `get_database_info` answers for a Postgres database, as `DatabaseInfo` in Superset's
# `superset/mcp_service/database/schemas.py` serializes it.
POSTGRES_DATABASE = {"id": DATABASE_ID, "uuid": "1b0c6c43-7f4e-4f0a-9c39-6a2b51f2f6c1", "database_name": "warehouse",
                     "backend": "postgresql", "expose_in_sqllab": True, "allow_ctas": False, "allow_cvas": False,
                     "allow_dml": False, "allow_file_upload": False, "allow_run_async": False, "cache_timeout": None,
                     "configuration_method": "sqlalchemy_form", "force_ctas_schema": None, "impersonate_user": False,
                     "is_managed_externally": False, "external_url": None, "extra": {}}


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


class SourceBehindSuperset:
    """`execute_sql` over the schema `TABLES` and `ROWS` hold, as Superset answers it: the first page of each
    `PostgresCatalogue` statement from the schema's catalogue rows, and any other statement run by DuckDB over its
    tables. A catalogue method named in `refused` is answered `success: false`, as Superset answers a statement the
    database refused."""

    def __init__(self) -> None:
        self.tables = {name: list(columns) for name, columns in TABLES.items()}
        self.refused: set[str] = set()
        self._lock = threading.Lock()
        self._data = duckdb.connect()
        for name, columns in TABLES.items():
            self._data.execute(f'CREATE TABLE "{name}" ('
                               + ", ".join(f'"{column}" {_DUCKDB_TYPES[kind]}' for column, kind in columns) + ")")
            self._data.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(columns))})', ROWS[name])
        catalogue = PostgresCatalogue(lambda sql: [], SCHEMA)
        self._first_pages = {f"{catalogue.statement(method)} LIMIT {PAGE_ROWS}": method for method in CATALOGUE_SQL}

    def add_column(self, table: str, column: str, kind: str) -> None:
        self.tables[table].append((column, kind))
        with self._lock:
            self._data.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {_DUCKDB_TYPES[kind]}')

    def __call__(self, request: dict, authorization: str | None) -> dict:
        sql = request["sql"]
        method = self._first_pages.get(sql)
        if method in self.refused:
            return {"success": False, "error": f"permission denied for table pg_class {SAID}",
                    "error_type": "GENERIC_DB_ENGINE_ERROR"}
        if method is not None:
            return _payload(CATALOGUE_COLUMNS[method], self._catalogue(method))
        with self._lock:
            try:
                cursor = self._data.execute(sql)
                names, rows = [d[0] for d in cursor.description], cursor.fetchall()
            except duckdb.Error as exc:
                return {"success": False, "error": str(exc), "error_type": "GENERIC_DB_ENGINE_ERROR"}
        return _payload(tuple(names), rows)

    def _catalogue(self, method: str) -> list[tuple]:
        if method == "tables":
            return [(name,) for name in sorted(self.tables)]
        if method == "columns":
            return [(name, column, kind, number) for name in sorted(self.tables)
                    for number, (column, kind) in enumerate(self.tables[name], 1)]
        if method == "foreign_keys":
            return list(FOREIGN_KEYS)
        if method == "views":
            return list(VIEWS)
        if method == "builtin_functions":
            return [("abs",), ("count",)]
        return []


def _payload(names: tuple[str, ...], rows: list[tuple]) -> dict:
    """A successful `execute_sql` answer: each row an object keyed by column name, each column typed as Superset types
    it, an integer column `int64` and any other `object`."""
    kinds = ["int64" if all(type(row[i]) is int for row in rows if row[i] is not None) else "object"
             for i in range(len(names))]
    return {"success": True, "rows": [dict(zip(names, row, strict=True)) for row in rows],
            "columns": [{"name": name, "type": kind} for name, kind in zip(names, kinds, strict=True)],
            "row_count": len(rows), "affected_rows": None, "error": None, "error_type": None}


@pytest.fixture
def source():
    return SourceBehindSuperset()


@pytest.fixture
def fake(source):
    with FakeSuperset() as served:
        served.reply("execute_sql", source)
        served.reply("get_database_info", lambda request, authorization: dict(POSTGRES_DATABASE))
        yield served


@pytest.fixture
def model():
    with StubModel() as stub:
        yield stub
    assert not stub.all_errors()


@pytest.fixture
def key_file(tmp_path, monkeypatch):
    """The service key in a file only its owner may read, named by the environment as an operator names it."""
    path = tmp_path / "service-key"
    path.write_text(f"{KEY}\n")
    path.chmod(0o600)
    monkeypatch.setenv(SERVICE_KEY_ENV, str(path))
    return path


@pytest.fixture
def no_direct_source(monkeypatch):
    """Every way to a connection of mnemiq's own fails the test if reached, and says which was."""
    reached: list[str] = []

    def refuse(name):
        def reach(*args, **kwargs):
            reached.append(name)
            raise AssertionError(f"Superset mode reached {name}")

        return reach

    monkeypatch.setattr(resolve_module, "adapter_for", refuse("adapter_for"))
    monkeypatch.setattr(runtime_module, "source_adapter", refuse("source_adapter"))
    monkeypatch.setattr(psycopg, "connect", refuse("psycopg.connect"))
    return reached


def _settings(tmp_path, model, fake, **over) -> Settings:
    """The Superset-mode deployment `tests/test_superset_runtime.py` builds, over an empty store."""
    return superset_deployment(tmp_path, model.url, fake.url, snapshot=False, embed_base_url=model.url,
                               embed_api_key="stub-key", embed_model="stub-embed", **over)


def _saved(settings: Settings):
    con = init_store(settings.store_path)
    version = current_version(con, SOURCE_ID)
    assert version is not None, "no snapshot was saved"
    return load_snapshot(con, version)


def _statements(fake: FakeSuperset) -> list[dict]:
    return [call.request for call in fake.calls if call.tool == "execute_sql"]


def _reads_values(sql: str) -> bool:
    """Whether `sql` returns a column's values: it selects DISTINCT values or groups rows by a column. The catalogue's
    own statements read `pg_catalog`, never a table's rows."""
    upper = sql.upper()
    return "PG_CATALOG." not in upper and ("SELECT DISTINCT" in upper or "GROUP BY" in upper)


def _source_job(database_id: int = DATABASE_ID, schema: str = SCHEMA) -> Job:
    return Job(id=SUPERSET_SOURCE_JOB, source_id=SOURCE_ID, kind="superset", status="done",
               detail=json.dumps({"database_id": database_id, "schema": schema, "backend": "postgresql",
                                  "values": "not harvested"}))


# --- no connection of its own ---------------------------------------------------------------------------------------


def test_enrich_opens_no_database_connection(tmp_path, model, fake, key_file, monkeypatch, no_direct_source):
    """Enrichment extends the existing rule that Superset mode opens no connection of its own: with a direct source
    configured, it never reaches `adapter_for` or a database driver, and reads everything through Superset as the
    service user."""
    monkeypatch.setenv("MNEMIQ_PG_DSN", DSN)
    settings = _settings(tmp_path, model, fake)
    assert settings.pg_dsn == DSN  # the direct source is configured, and still never connected to
    assert _cmd_enrich(settings) == 0
    assert no_direct_source == []
    snapshot = _saved(settings)
    assert {binding.object_id for binding in snapshot.source_bindings} == set(TABLES)
    assert fake.calls and {call.authorization for call in fake.calls} == {f"Bearer {KEY}"}


def test_refresh_reads_through_superset_and_reenriches_a_change(tmp_path, model, fake, source, key_file,
                                                              no_direct_source, capsys):
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 0
    assert _cmd_refresh(settings) == 0
    assert "catalog unchanged" in capsys.readouterr().out
    source.add_column("claim", "handler", "text")
    assert _cmd_refresh(settings) == 0
    assert "changed=['claim']" in capsys.readouterr().out
    snapshot = _saved(settings)
    assert "claim.handler" in {column.id for column in snapshot.columns}
    assert _source_job() in snapshot.jobs
    assert no_direct_source == []
    assert {call.authorization for call in fake.calls} == {f"Bearer {KEY}"}


@pytest.mark.parametrize("stored", ["direct", "another-database", "another-schema"])
def test_refresh_reenriches_a_snapshot_not_read_from_this_source(tmp_path, model, fake, key_file, no_direct_source,
                                                                 capsys, stored):
    """With the catalogue unchanged, refresh still re-enriches where the stored snapshot is not one this deployment's
    server would serve: read over a direct connection, or through Superset from another database or schema."""
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 0
    enriched = _saved(settings)
    jobs = [job for job in enriched.jobs if job.id != SUPERSET_SOURCE_JOB]
    if stored == "another-database":
        jobs.append(_source_job(database_id=DATABASE_ID + 1))
    elif stored == "another-schema":
        jobs.append(_source_job(schema="sales"))
    other = enriched.model_copy(update={"jobs": jobs})
    other.version = content_version(other)
    assert other.version != enriched.version
    save_snapshot(init_store(settings.store_path), other)
    assert _saved(settings).version == other.version
    capsys.readouterr()
    assert _cmd_refresh(settings) == 0
    out = capsys.readouterr().out
    assert "added=[] changed=[] dropped=[]" in out and "nothing to refresh" not in out
    assert [job for job in _saved(settings).jobs if job.id == SUPERSET_SOURCE_JOB] == [_source_job()]
    assert no_direct_source == []


# --- no values ------------------------------------------------------------------------------------------------------


def test_enrich_harvests_no_values(tmp_path, model, fake, key_file, capsys):
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 0
    assert "0 indexed values" in capsys.readouterr().out
    snapshot = _saved(settings)
    assert snapshot.columns and not [column.id for column in snapshot.columns if column.coded_values]
    sent = [request["sql"] for request in _statements(fake)]
    assert not [sql for sql in sent if _TOP_K.search(sql)]
    # Nor grounding, the value index or anything else: each would ask for a column's values.
    assert not [sql for sql in sent if _reads_values(sql)]
    assert not [value for value in CELL_TEXT if value in snapshot.model_dump_json()]
    con = init_store(settings.store_path)
    assert con.execute("SELECT count(*) FROM value_index").fetchone() == (0,)


def test_the_harvest_checks_see_a_harvest(tmp_path, model, fake, key_file):
    """The positive control for `test_enrich_harvests_no_values`: the same source read with harvesting on, through the
    same adapter, does send `_top_k`'s statement and the lookup's DISTINCT, and its values do reach the snapshot."""
    adapter = enrichment_adapter(_settings(tmp_path, model, fake), service_credential_from_env())
    snapshot = ground_codes(adapter, enrich_structural(adapter, SOURCE_ID))
    coded = {column.id: {value.code for value in column.coded_values} for column in snapshot.columns
             if column.coded_values}
    assert coded["claim.region"] == {"Northgate", "Southmere", "Eastwick"}
    assert coded["claim.status_id"] == {"1", "2", "3"}
    sent = [request["sql"] for request in _statements(fake)]
    assert [sql for sql in sent if _TOP_K.search(sql)]
    assert [sql for sql in sent if "SELECT DISTINCT" in sql and _reads_values(sql)]
    assert [value for value in CELL_TEXT if value in snapshot.model_dump_json()]


def test_concept_schemes_bind_nothing_from_values(tmp_path, model, fake, key_file):
    """With concept schemes configured, enrichment binds a column to one by probing the column's values, and suggests
    bindings by the same probe, writing the values that did not match into its suggestions file. Through Superset it
    does neither."""
    records = tmp_path / "records.json"
    records.write_text(OntologyRecords(schemes=[ConceptScheme(id="region", label="Region", concepts=[
        Concept(id=f"region:{code}", notation=code, pref_label=f"Region {code}") for code in ("N1", "S1")])])
        .model_dump_json())
    suggestions = tmp_path / "binding-suggestions.json"
    settings = _settings(tmp_path, model, fake, ontology_records_path=str(records),
                         binding_suggestions_path=str(suggestions))
    assert _cmd_enrich(settings) == 0
    assert not [request["sql"] for request in _statements(fake) if _reads_values(request["sql"])]
    assert not suggestions.exists()
    assert not [value for value in CELL_TEXT if value in _saved(settings).model_dump_json()]


def test_an_explicit_concept_binding_is_applied_without_reading_values(tmp_path, model, fake, key_file):
    """An operator's explicit binding of a column to a concept scheme is taken on their word and reads nothing, so
    enrichment through Superset applies it; only the binder's probe for bindings of its own reads values."""
    records = tmp_path / "records.json"
    records.write_text(OntologyRecords(
        schemes=[ConceptScheme(id="region", label="Region", concepts=[
            Concept(id=f"region:{code}", notation=code, pref_label=f"Region {code}") for code in ("N1", "S1")])],
        bindings={"claim.region": "region"}).model_dump_json())
    settings = _settings(tmp_path, model, fake, ontology_records_path=str(records))
    assert _cmd_enrich(settings) == 0
    schemes = {column.id: column.code_scheme for column in _saved(settings).columns if column.code_scheme}
    assert schemes == {"claim.region": CodeScheme(id="region", label="Region")}
    assert not [request["sql"] for request in _statements(fake) if _reads_values(request["sql"])]


def test_the_binder_without_its_probe_reads_nothing(tmp_path):
    """`bind_schemes(..., probe=False)` applies the explicit bindings and sends the source nothing, even given an
    adapter; with its probe, the same call reads the values of each column it may bind."""
    from mnemiq.ontology.binder import bind_schemes

    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE sale (id INTEGER PRIMARY KEY, city TEXT, region TEXT);"
                      "INSERT INTO sale VALUES (1, 'NYC', 'N1'), (2, 'LA', 'S1'), (3, 'NYC', 'N1');")
    con.commit()
    con.close()
    snapshot = enrich_structural(SQLiteAdapter(str(path)), "shop", harvest_values=False)
    records = OntologyRecords(schemes=[ConceptScheme(id="region", label="Region", concepts=[
        Concept(id=f"region:{code}", notation=code, pref_label=code) for code in ("N1", "S1")])],
        bindings={"sale.city": "region"})

    class Recording(SQLiteAdapter):
        sent: list[str] = []

        def execute(self, sql):
            self.sent.append(sql)
            return super().execute(sql)

    adapter = Recording(str(path))
    bound = bind_schemes(adapter, snapshot, records, probe=False)
    assert {c.id: c.code_scheme.id for c in bound.columns if c.code_scheme} == {"sale.city": "region"}
    assert adapter.sent == []
    bind_schemes(adapter, snapshot, records)
    assert [sql for sql in adapter.sent if "DISTINCT" in sql.upper()]  # the control: the probe reads values


def test_relationship_inference_is_told_to_harvest_nothing(monkeypatch):
    """Key detection reads counts only. With harvesting off, every profile it asks for is told so as well; with it on,
    each is asked exactly as before."""
    from mnemiq.catalog import ColumnInfo, TableInfo
    from mnemiq.enrichment import joins
    from mnemiq.enrichment.profiling import ColumnStats

    asked: list[dict] = []

    def profile(adapter, table, column, k=10, data_type=None, **keywords):
        asked.append(keywords)
        return ColumnStats(table=table, column=column, row_count=1, distinct_count=1, null_count=0)

    class Source:
        dialect = "duckdb"

        def foreign_keys(self):
            return []

        def execute(self, sql):
            return [(0,)]

    monkeypatch.setattr(joins, "profile_column", profile)
    catalog = [TableInfo("claim", [ColumnInfo("claim_id", "integer")]),
               TableInfo("payment", [ColumnInfo("claim_id", "integer")])]
    assert [r.id for r in joins.build_relationships(Source(), catalog, harvest_values=False)] == [
        "payment.claim_id->claim"]
    assert asked and asked == [{"harvest_values": False}] * len(asked)
    asked.clear()
    joins.build_relationships(Source(), catalog)
    assert asked and asked == [{}] * len(asked)


def test_a_value_index_left_by_a_direct_enrichment_is_cleared(tmp_path, model, fake, key_file):
    """A store enriched over a direct connection before keeps that source's values in its value index, which the
    server reads; enriching the same source through Superset leaves none of them."""
    settings = _settings(tmp_path, model, fake)
    con = init_store(settings.store_path)
    con.execute("CREATE TABLE IF NOT EXISTS value_index (source_id TEXT, object_id TEXT, column_name TEXT, "
                "value_text TEXT)")
    con.execute("INSERT INTO value_index VALUES (?, 'claim', 'region', 'Northgate')", [SOURCE_ID])
    assert _cmd_enrich(settings) == 0
    assert con.execute("SELECT count(*) FROM value_index WHERE source_id = ?", [SOURCE_ID]).fetchone() == (0,)


def test_examples_left_by_a_direct_build_are_cleared(tmp_path, model, fake, key_file):
    """A store built over a direct connection before keeps that source's few-shot examples, whose SQL can carry
    literals read with mnemiq's own credential; enriching the same source through Superset leaves none of them. Another
    source's examples are its own build's to clear, and the ask path serves none of them in Superset mode either
    (`test_superset_mode_serves_no_examples`)."""
    settings = _settings(tmp_path, model, fake)
    con = init_store(settings.store_path)
    con.execute("CREATE TABLE IF NOT EXISTS example (question TEXT, sql TEXT, tables TEXT, object_id TEXT, "
                "source_id TEXT, embedding FLOAT[3])")
    for source_id in (SOURCE_ID, "acme"):
        con.execute("INSERT INTO example VALUES ('claims in the south', ?, '[\"claim\"]', 'claim', ?, [1.0, 0.0, 0.0])",
                    ["SELECT count(*) FROM claim WHERE region = 'Southmere'", source_id])
    assert _cmd_enrich(settings) == 0
    assert con.execute("SELECT source_id, count(*) FROM example GROUP BY source_id").fetchall() == [("acme", 1)]


def test_superset_mode_serves_no_value_index(tmp_path, model, fake, no_direct_source):
    """Superset mode harvests no values, so it serves none, whatever the store holds: here, values a direct
    enrichment left under another source id, which the index reads with no regard to source. Every agent of every
    profile, the ask path's every read of the index, finds no column indexed and no value."""
    settings = superset_deployment(tmp_path, model.url, fake.url, profiles=PROFILES)
    con = init_store(settings.store_path)
    con.execute("CREATE TABLE IF NOT EXISTS value_index (source_id TEXT, object_id TEXT, column_name TEXT, "
                "value_text TEXT)")
    con.execute("INSERT INTO value_index VALUES ('acme', 'wide', 'col_0', 'Northgate')")
    con.close()
    runtime = build_runtime(settings)
    assert ValueIndex(runtime.con).nearest("wide", "col_0", "North") == ["Northgate"]  # the rows are there
    agents = [agent for loaded in runtime.profiles.values() for agent in loaded.agents.values()]
    assert len(runtime.profiles) == 2 and agents
    for agent in agents:
        assert not agent.values.has("wide", "col_0")
        assert not agent.values.contains("wide", "col_0", "Northgate")
        assert agent.values.nearest("wide", "col_0", "North") == []
    assert no_direct_source == []


def test_direct_enrichment_harvests_as_before(tmp_path):
    """Direct mode's default is to harvest: the snapshot is the same whether `harvest_values` is left out or set, and
    turning it off keeps every count while reading no value."""
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE sale (id INTEGER PRIMARY KEY, city TEXT, amount REAL);"
                      "INSERT INTO sale VALUES (1, 'NYC', 10.0), (2, 'LA', 5.0), (3, 'NYC', 7.0);")
    con.commit()
    con.close()

    class Recording(SQLiteAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sent: list[str] = []

        def execute(self, sql):
            self.sent.append(sql)
            return super().execute(sql)

    def dumped(snapshot) -> str:
        return snapshot.model_copy(update={"created_at": ""}).model_dump_json()

    default, harvesting, unharvested = Recording(str(path)), Recording(str(path)), Recording(str(path))
    by_default = enrich_structural(default, "shop")
    assert dumped(by_default) == dumped(enrich_structural(harvesting, "shop", harvest_values=True))
    assert default.sent == harvesting.sent
    city = next(column for column in by_default.columns if column.id == "sale.city")
    assert {value.code for value in city.coded_values} == {"NYC", "LA"}
    without = enrich_structural(unharvested, "shop", harvest_values=False)
    assert not [column.id for column in without.columns if column.coded_values]
    assert [(c.id, c.row_count, c.distinct_count, c.null_count) for c in without.columns] == \
        [(c.id, c.row_count, c.distinct_count, c.null_count) for c in by_default.columns]
    assert [sql for sql in default.sent if _TOP_K.search(sql)]
    assert not [sql for sql in unharvested.sent if _TOP_K.search(sql)]


def test_a_superset_source_is_recorded_only_where_no_value_was_harvested(tmp_path):
    path = tmp_path / "shop.sqlite"
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match="harvest"):
        enrich_structural(SQLiteAdapter(str(path)), "shop", superset_source=(DATABASE_ID, SCHEMA))


# --- the source job, joined to the server that reads it --------------------------------------------------------------


def test_snapshot_carries_the_superset_source_job(tmp_path, model, fake, key_file):
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 0
    jobs = [job for job in _saved(settings).jobs if job.id == SUPERSET_SOURCE_JOB]
    assert jobs == [_source_job()]


def test_the_enriched_snapshot_starts_the_deployment_it_was_read_for(tmp_path, model, fake, key_file,
                                                                     no_direct_source):
    """The job enrichment writes is the one the server checks: enriched through Superset and saved, the snapshot boots
    a runtime with the same settings that opens no source and asks Superset nothing, and is refused under another
    database or schema."""
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 0
    fake.calls.clear()
    runtime = build_runtime(settings)
    assert isinstance(runtime.adapter, NoSourceConnection)
    assert runtime.snapshot == _saved(settings)
    assert fake.calls == [] and no_direct_source == []
    for other, named in (({"superset_database_id": DATABASE_ID + 1}, "Superset database"),
                         ({"superset_schema": "sales"}, "schema")):
        with pytest.raises(SnapshotMissing, match=f"enriched from {named}"):
            build_runtime(settings.model_copy(update=other))


def test_the_source_a_snapshot_was_read_from_is_part_of_its_version():
    """Two snapshots alike but for the Superset database they were read from are two versions, so a deployment that
    publishes one never takes it for the other; a snapshot with no such job keeps the version it always had."""
    from mnemiq.contract import Snapshot

    def snapshot(*jobs: Job) -> Snapshot:
        return Snapshot(version="", source_id=SOURCE_ID, created_at="t", jobs=list(jobs))

    direct = snapshot(Job(id="infer:relationships", source_id=SOURCE_ID, kind="join", status="done"))
    assert content_version(direct) == content_version(snapshot())
    seven, eight = snapshot(_source_job(DATABASE_ID)), snapshot(_source_job(DATABASE_ID + 1))
    assert len({content_version(direct), content_version(seven), content_version(eight)}) == 3
    assert content_version(seven) == content_version(snapshot(_source_job(DATABASE_ID)))


# --- force_refresh, and the deadlines --------------------------------------------------------------------------------


def test_every_enrichment_statement_forces_a_refresh(tmp_path, model, fake, key_file):
    """A re-enrich reads the source as it is now, so every statement enrichment sends, profiling's included, asks
    Superset to run it rather than answer from its result cache. Each carries the database, the schema, the statement
    and the time limit, and nothing else."""
    assert _cmd_enrich(_settings(tmp_path, model, fake)) == 0
    statements = _statements(fake)
    assert any(request["sql"].startswith("SELECT count(*), count(DISTINCT") for request in statements)
    assert statements and [request for request in statements
                           if set(request) != {"database_id", "sql", "schema", "timeout", "force_refresh"}] == []
    assert {(r["database_id"], r["schema"], r["timeout"], r["force_refresh"]) for r in statements} == {
        (DATABASE_ID, SCHEMA, 300, True)}
    assert [call.request for call in fake.calls if call.tool == "get_database_info"] == [{"identifier": DATABASE_ID}]


def test_the_ask_path_never_forces_a_refresh_on_a_data_statement(tmp_path):
    """The other direction: an adapter `superset_executor` builds sends no `force_refresh` with a data statement, a
    proof or a dry run, so Superset's result cache, which checks the person's access first, still serves them. Its
    catalogue reads do send it."""
    with FakeSuperset() as fake:
        fake.reply("execute_sql", _answer_rows)
        settings = Settings(superset_mcp_url=fake.url, superset_database_id=DATABASE_ID, superset_schema=SCHEMA)
        adapter = superset_executor(settings, DelegatedToken("tok-1"))
        adapter.execute_arrow("SELECT COUNT(*) AS n FROM wide")
        adapter.execute("SELECT COUNT(*) AS n FROM wide")
        adapter.validate("SELECT COUNT(*) AS n FROM wide")
        adapter.dry_run("SELECT COUNT(*) AS n FROM wide")
        data = list(fake.calls)
        adapter.introspect()
    assert len(data) == 4 and [call.request for call in data if "force_refresh" in call.request] == []
    assert [call.request.get("force_refresh") for call in fake.calls[len(data):]] == [True]


def test_the_enrichment_deadlines(tmp_path, model, fake, key_file, monkeypatch):
    """Enrichment gives Superset 300 seconds a statement, the most `execute_sql` accepts, and waits that plus the
    adapter's margin for each answer; the client's own deadline is no shorter, so nothing undercuts a statement's. The
    backend check is a metadata call, and waits only the client's default."""
    deadlines: list[tuple[str, float | None]] = []
    real = SupersetMcp.call

    def recording(self, tool, request, *, timeout_s=None):
        deadlines.append((tool, timeout_s))
        return real(self, tool, request, timeout_s=timeout_s)

    monkeypatch.setattr(SupersetMcp, "call", recording)
    adapter = enrichment_adapter(_settings(tmp_path, model, fake), service_credential_from_env())
    assert deadlines == [("get_database_info", DEFAULT_TIMEOUT_S)] and DEFAULT_TIMEOUT_S == 45
    assert adapter.client._timeout_s >= 300 + TRANSPORT_MARGIN_S
    assert adapter.execute('SELECT count(*) FROM "claim"') == [(6,)]
    assert deadlines[1:] == [("execute_sql", 300 + TRANSPORT_MARGIN_S)]
    assert _statements(fake)[-1]["timeout"] == 300


# --- what Superset could not answer -----------------------------------------------------------------------------------


def test_refused_catalogue_query_is_could_not_look(tmp_path, model, fake, source, key_file):
    """A catalogue statement Superset refuses is a source that could not be asked, never one with nothing in it:
    enrichment records the views it could not read as a failed discovery, and on the ask path a function inventory
    Superset refuses leaves a statement that calls a function undecidable."""
    source.refused.add("views")
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 0
    snapshot = _saved(settings)
    assert [job.status for job in snapshot.jobs if job.id == "discover:views"] == ["failed"]
    assert snapshot.views == []

    asked = "SELECT abs(amount) AS size FROM claim"
    visible = {"claim": {"claim_id", "status_id", "region", "amount"}}
    answered = decide(asked, visible, adapter=superset_executor(settings, DelegatedToken("tok-1")),
                      dialect="postgres", target="postgres")
    assert isinstance(answered, Approved), answered  # the control: the inventory answered, the statement clears
    source.refused.add("user_functions")
    refused = decide(asked, visible, adapter=superset_executor(settings, DelegatedToken("tok-1")),
                     dialect="postgres", target="postgres")
    assert isinstance(refused, Refusal) and refused.code == RefusalCode.UNRESOLVABLE_CALLS, refused


# --- the service key ------------------------------------------------------------------------------------------------


def _chain(exc: BaseException):
    """`exc` and every exception it was raised from or during."""
    seen: list[BaseException] = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        yield exc
        exc = exc.__cause__ or exc.__context__


def _never_shows_the_key(exc: BaseException, caplog) -> None:
    for raised in _chain(exc):
        for text in (str(raised), repr(raised), repr(raised.args)):
            assert "sst_x" not in text and KEY not in text, text
    assert "sst_x" not in caplog.text


def _key_environment(tmp_path, case: str) -> dict[str, str]:
    """An environment naming the service key file as `case` describes it."""
    path = tmp_path / "service-key"
    if case == "unset":
        return {}
    if case == "blank":
        return {SERVICE_KEY_ENV: "  "}
    if case == "the-key-itself":  # the key put where its file's path belongs
        return {SERVICE_KEY_ENV: KEY}
    if case == "missing":
        return {SERVICE_KEY_ENV: str(path)}
    if case == "directory":
        path.mkdir(mode=0o700)
        (path / "key").write_text(KEY)
        return {SERVICE_KEY_ENV: str(path)}
    contents = {"empty": "", "whitespace": " \n\t\n", "space-inside": f"{KEY} {KEY}\n",
                "not-ascii": f"{KEY}é\n", "larger-than-a-key": KEY * (KEY_FILE_LIMIT // len(KEY) + 1)}
    contents, mode = (contents[case], 0o600) if case in contents else (f"{KEY}\n", int(case, 8))
    path.write_text(contents, encoding="utf-8")
    path.chmod(mode)
    return {SERVICE_KEY_ENV: str(path)}


@pytest.mark.parametrize("case", ["unset", "blank", "the-key-itself", "missing", "directory", "0644", "0640", "0604",
                                  "0620", "0601", "empty", "whitespace", "space-inside", "not-ascii",
                                  "larger-than-a-key"])
def test_service_key_file_rules(tmp_path, caplog, case):
    """The key file must be named, a regular file, readable by its owner alone, no larger than a key could be, and
    hold a key a header can carry. Each refusal names the variable, and none of them, nor anything raised or logged on
    the way, holds the key: not even where the variable holds the key itself, in place of its file's path."""
    environ = _key_environment(tmp_path, case)
    with caplog.at_level(logging.DEBUG), pytest.raises(SystemExit) as refused:
        service_credential_from_env(environ)
    assert isinstance(refused.value, SupersetConfigError)
    assert SERVICE_KEY_ENV in str(refused.value)
    _never_shows_the_key(refused.value, caplog)


def test_a_key_file_as_large_as_the_limit_is_read(tmp_path):
    path = tmp_path / "service-key"
    path.write_bytes(b"k" * KEY_FILE_LIMIT)
    path.chmod(0o600)
    assert service_credential_from_env({SERVICE_KEY_ENV: str(path)}).value == "k" * KEY_FILE_LIMIT


def test_a_key_file_is_read_no_further_than_the_limit(tmp_path, monkeypatch):
    """A file far larger than any key is refused having read one byte past the limit, never the whole file."""
    import mnemiq.superset.service as service_module

    path = tmp_path / "service-key"
    path.write_bytes(b"k" * (16 * KEY_FILE_LIMIT))
    path.chmod(0o600)
    read: list[int] = []

    class Counting:
        def __getattr__(self, name):
            return getattr(os, name)

        def read(self, descriptor, size):
            chunk = os.read(descriptor, size)
            read.append(len(chunk))
            return chunk

    monkeypatch.setattr(service_module, "os", Counting())
    with pytest.raises(SupersetConfigError, match="larger"):
        service_credential_from_env({SERVICE_KEY_ENV: str(path)})
    assert 0 < sum(read) <= KEY_FILE_LIMIT + 1


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_a_key_file_its_owner_alone_may_read_gives_the_key(tmp_path, monkeypatch, caplog, mode):
    path = tmp_path / "service-key"
    path.write_text(f"  {KEY}\n")
    path.chmod(mode)
    monkeypatch.setenv(SERVICE_KEY_ENV, str(path))
    with caplog.at_level(logging.DEBUG):
        key = service_credential_from_env()
    assert isinstance(key, ServiceKey) and isinstance(key, Bearer) and not isinstance(key, DelegatedToken)
    assert key.value == KEY
    assert KEY not in repr(key) and KEY not in str(key) and KEY not in f"{key}" and "sst_x" not in caplog.text


def test_the_service_key_is_no_setting(tmp_path, monkeypatch, key_file):
    """The key file is read by enrichment's own module and nothing else: it is not a setting, so no configuration
    dump, template or profile view carries its path."""
    settings = Settings()
    assert not [name for name in Settings.model_fields if "service" in name and "key" in name]
    assert str(key_file) not in settings.model_dump_json()
    assert SERVICE_KEY_ENV not in Settings.env_example()


def test_the_ask_path_refuses_the_service_key(tmp_path):
    settings = Settings(superset_mcp_url="http://127.0.0.1:1/mcp", superset_database_id=DATABASE_ID)
    with pytest.raises(TypeError) as refused:
        superset_executor(settings, ServiceKey(KEY))
    assert KEY not in str(refused.value)


def test_enrichment_takes_the_service_key_only(tmp_path, model, fake):
    with pytest.raises(TypeError):
        enrichment_adapter(_settings(tmp_path, model, fake), DelegatedToken("tok-1"))
    assert fake.calls == []


# --- the backend, and settings Superset mode cannot honour -----------------------------------------------------------


NOT_POSTGRES = {
    "mysql": {**POSTGRES_DATABASE, "backend": "mysql"},
    "postgres-spelled-otherwise": {**POSTGRES_DATABASE, "backend": "postgres"},
    "not-found": {"error": f"DatabaseInfo with identifier '7' not found {SAID}", "error_type": "not_found",
                  "timestamp": "2026-11-12T00:00:00Z"},
    "no-backend": {key: value for key, value in POSTGRES_DATABASE.items() if key != "backend"},
    "null-backend": {**POSTGRES_DATABASE, "backend": None},
    "another-database": {**POSTGRES_DATABASE, "id": DATABASE_ID + 1},
    "id-as-text": {**POSTGRES_DATABASE, "id": str(DATABASE_ID)},
    "tool-error": RuntimeError(SAID),
}


@pytest.mark.parametrize("answer", NOT_POSTGRES.values(), ids=NOT_POSTGRES.keys())
def test_non_postgres_backend_is_refused(tmp_path, model, fake, key_file, caplog, answer):
    """Superset mode reads PostgreSQL only, so enrichment asks Superset which backend the database is before it reads
    anything, and refuses any other answer, or one it cannot read, with its own sentence. What Superset said is logged
    and goes nowhere else."""
    fake.reply("get_database_info", lambda request, authorization: answer)
    with caplog.at_level(logging.WARNING), pytest.raises(SupersetConfigError) as refused:
        enrichment_adapter(_settings(tmp_path, model, fake), service_credential_from_env())
    message = str(refused.value)
    assert str(DATABASE_ID) in message
    assert SAID not in message and "mysql" not in message
    if isinstance(answer, Exception) or "error" in answer or answer.get("backend") == "mysql":
        assert SAID in caplog.text or "mysql" in caplog.text
    assert [call.tool for call in fake.calls] == ["get_database_info"]  # nothing was read
    _never_shows_the_key(refused.value, caplog)


def test_enrich_refuses_a_non_postgres_backend_and_saves_nothing(tmp_path, model, fake, key_file, capsys):
    fake.reply("get_database_info", lambda request, authorization: {**POSTGRES_DATABASE, "backend": "mysql"})
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 1
    assert "PostgreSQL" in capsys.readouterr().err
    assert current_version(init_store(settings.store_path), SOURCE_ID) is None


def test_a_service_key_superset_refuses_is_said_so(tmp_path, model, fake, key_file, capsys):
    fake.refuse_token(KEY)
    settings = _settings(tmp_path, model, fake)
    assert _cmd_enrich(settings) == 1
    err = capsys.readouterr().err
    assert "service key" in err and KEY not in err


REFUSED_SETTINGS = {
    "examples": ({"enrich_examples": True}, "MNEMIQ_ENRICH_EXAMPLES"),
    "facts": ({"enrich_facts": True}, "MNEMIQ_ENRICH_FACTS"),
    "manifest": ({"sources_path": "sources.json"}, "MNEMIQ_SOURCES_PATH"),
    "no-database": ({"superset_database_id": None}, "MNEMIQ_SUPERSET_DATABASE_ID"),
    "schema": ({"superset_schema": "two words"}, "MNEMIQ_SUPERSET_SCHEMA"),
    "credentials-in-url": ({"superset_mcp_url": "https://enricher:hunter2@superset.example.com/mcp"},
                           "MNEMIQ_SUPERSET_MCP_URL"),
}


@pytest.mark.parametrize("over, named", REFUSED_SETTINGS.values(), ids=REFUSED_SETTINGS.keys())
def test_enrichment_refuses_settings_superset_mode_cannot_honour(tmp_path, model, fake, key_file, no_direct_source,
                                                                 capsys, over, named):
    """The rules the server starts under that bind what enrichment reads: one database and a plain schema, at a URL
    that carries no credential of its own, and no experimental phase that keeps what enrichment read. Refused before
    Superset is asked anything."""
    settings = _settings(tmp_path, model, fake).model_copy(update=over)
    assert _cmd_enrich(settings) == 1
    err = capsys.readouterr().err
    assert named in err and "hunter2" not in err
    assert fake.calls == [] and no_direct_source == []


def test_an_http_url_to_another_host_is_warned_of(tmp_path, model, fake, key_file, caplog, monkeypatch):
    """Superset reached over http on a private network, by a Compose service's name, is accepted with a warning, as
    the server accepts it. The call itself is stood in for, so no name is looked up."""
    def unreachable(self, tool, request, *, timeout_s=None):
        raise SupersetUnavailable("Superset's MCP service could not be reached")

    monkeypatch.setattr(SupersetMcp, "call", unreachable)
    settings = _settings(tmp_path, model, fake).model_copy(update={"superset_mcp_url": "http://superset:8088/mcp"})
    with caplog.at_level(logging.WARNING), pytest.raises(SupersetConfigError, match="could not be reached"):
        enrichment_adapter(settings, service_credential_from_env())
    assert any("service key" in record.getMessage() and "unencrypted" in record.getMessage()
               for record in caplog.records)

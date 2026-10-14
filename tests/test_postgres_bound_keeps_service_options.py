"""Codex's review of #134 (M152): the statement-timeout bound keeps a connection service's options,
and a federated source's server cancellation reads as a timeout too.

A DSN naming a service (`service=prod`, PGSERVICE, `servicefile=`) takes its settings from a
service file, which psycopg's conninfo parser does not expand -- so the bound arrived as the
connection's WHOLE `options`, and libpq let it replace the service's. Measured on a throwaway
Postgres 16: a service setting `-c search_path=from_service` connected with `from_service`
unbounded and with `"$user", public` bounded. Unqualified names then resolve elsewhere. The settings
the connection's startup applied are now read back from the server (`pg_settings`, source
'client') and carried behind the bound -- exact wherever libpq found them, a service in libpq's
compiled-in directory included, which a file parser here could not see.

The federated adapter had no translation of the server's cancellation, so a source's timeout
reached the model as "The source rejected this query" rather than the runner's timed-out guidance.
"""

from __future__ import annotations

import logging
import os

import pytest

from mnemiq.adapters.duckdb import bounded_dsn


def _options(dsn: str) -> str:
    from psycopg.conninfo import conninfo_to_dict

    return conninfo_to_dict(dsn).get("options") or ""


@pytest.fixture(autouse=True)
def _no_libpq_environment(monkeypatch):
    for name in ("PGOPTIONS", "PGSERVICE"):
        monkeypatch.delenv(name, raising=False)


def test_the_servers_account_of_the_startup_is_carried_behind_the_bound():
    """What libpq applied -- from a service file wherever it lives, the DSN, or PGOPTIONS -- is
    what `pg_settings` reports with source 'client', and it is carried whole."""
    dsn = bounded_dsn("service=prod", 30,
                      client_settings=[("search_path", "from_service"), ("work_mem", "7168")])
    assert _options(dsn) == ("-c statement_timeout=30000 -c search_path=from_service "
                             "-c work_mem=7168")


def test_a_carried_value_with_a_space_is_escaped_as_libpq_reads_it():
    dsn = bounded_dsn("host=h", 30, client_settings=[("search_path", "a, b")])
    assert _options(dsn) == "-c statement_timeout=30000 -c search_path=a,\\ b"


@pytest.mark.parametrize("dsn,env", [("service=prod", None), ("host=h", "prod"),
                                     ("host=h servicefile=/etc/x.conf", None)])
def test_a_service_whose_settings_the_server_did_not_report_is_left_unbounded(
        dsn, env, monkeypatch, caplog):
    """Its options cannot be known from here, and adding ours would replace them."""
    if env:
        monkeypatch.setenv("PGSERVICE", env)
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        assert bounded_dsn(dsn, 30) == dsn
    assert any("statement timeout" in r.getMessage() for r in caplog.records)


def test_a_plain_dsn_without_the_servers_account_still_carries_pgoptions(monkeypatch):
    monkeypatch.setenv("PGOPTIONS", "-c work_mem=64MB")
    assert _options(bounded_dsn("host=h", 30)) == "-c statement_timeout=30000 -c work_mem=64MB"


def test_attach_carries_the_servers_account_into_the_bounded_connection(monkeypatch):
    """The join: `attach_postgres` reads the client settings in its probe and re-attaches with
    them, not with the DSN alone."""
    import mnemiq.adapters.duckdb as module
    from mnemiq.adapters.duckdb import DuckDBAdapter

    sql = []

    class _Con:
        def execute(self, statement, *args):
            sql.append(statement)
            return self

        def fetchall(self):
            return [("statement_timeout", "0", "default"),
                    ("search_path", "from_service", "client"),
                    ("transaction_read_only", "on", "session")]  # DuckDB's own: never carried

    monkeypatch.setattr(module, "connect_duckdb", lambda **kw: _Con())
    monkeypatch.setattr(module, "load_extension", lambda *a, **kw: None)
    monkeypatch.setattr(module, "lock_configuration", lambda *a, **kw: None)
    DuckDBAdapter.postgres("service=prod", statement_timeout_s=30)
    attaches = [s for s in sql if s.startswith("ATTACH")]
    assert len(attaches) == 2, sql
    assert "search_path=from_service" in attaches[-1] and "statement_timeout=30000" in attaches[-1]
    assert "transaction_read_only" not in attaches[-1]


def test_a_federated_source_s_cancellation_reaches_the_model_as_a_timeout():
    from mnemiq.adapters.federated import FederatedAdapter
    from mnemiq.execute.runner import ExecutionError, run

    class _Cancelling:
        def execute(self, sql):
            return self

        def to_arrow_table(self):
            raise RuntimeError("IO Error: Failed to fetch header for COPY: ERROR:  canceling "
                               "statement due to statement timeout")

        def interrupt(self):
            pass

    adapter = FederatedAdapter.__new__(FederatedAdapter)
    adapter._con = _Cancelling()
    with pytest.raises(ExecutionError) as caught:
        run(adapter, "SELECT 1", timeout_s=30)
    assert "timed out" in caught.value.repair_text


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_a_live_service_keeps_its_search_path_and_gains_the_bound(tmp_path, monkeypatch):
    """Live: the measurement that found it, as a test. A service built from MNEMIQ_PG_DSN with a
    search_path of its own keeps it once bounded."""
    from psycopg.conninfo import conninfo_to_dict

    from mnemiq.adapters.duckdb import DuckDBAdapter

    params = conninfo_to_dict(os.environ["MNEMIQ_PG_DSN"])
    params.pop("options", None)
    lines = ["[live]", *(f"{k}={v}" for k, v in params.items()), "options=-c search_path=from_service"]
    (tmp_path / "pg_service.conf").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("PGSERVICEFILE", str(tmp_path / "pg_service.conf"))
    adapter = DuckDBAdapter.postgres("service=live", statement_timeout_s=30)
    rows = adapter.execute("SELECT * FROM postgres_query('src', 'SELECT "
                           "current_setting(''search_path'') AS sp, current_setting(''statement_timeout'') AS st')")
    assert rows == [("from_service", "30s")]

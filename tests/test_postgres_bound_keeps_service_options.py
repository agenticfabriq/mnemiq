"""Codex's review of #134 (M152): the statement-timeout bound keeps a connection service's options,
and a federated source's server cancellation reads as a timeout too.

A DSN naming a service (`service=prod`, or PGSERVICE) takes its settings from pg_service.conf, and
psycopg's conninfo parser does not expand the service -- so the bound arrived as the connection's
WHOLE `options`, and libpq let it replace the service's. Measured on a throwaway Postgres 16: a
service setting `-c search_path=from_service` connected with `from_service` unbounded and with
`"$user", public` bounded. Unqualified names then resolve elsewhere. The service's options are now
read as libpq reads them and merged behind the bound, and a named service that cannot be found
leaves the DSN unbounded, with a warning, rather than dropping its settings.

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


@pytest.fixture
def service_file(tmp_path, monkeypatch):
    path = tmp_path / "pg_service.conf"
    path.write_text("# a comment\n[prod]\nhost=h\ndbname=d\noptions=-c search_path=from_service\n\n"
                    "[bare]\nhost=h\n")
    monkeypatch.setenv("PGSERVICEFILE", str(path))
    monkeypatch.delenv("PGOPTIONS", raising=False)
    monkeypatch.delenv("PGSERVICE", raising=False)
    monkeypatch.setenv("PGSYSCONFDIR", str(tmp_path / "no-such-dir"))
    return path


def test_a_service_s_options_survive_behind_the_bound(service_file):
    assert _options(bounded_dsn("service=prod", 30)) == \
        "-c statement_timeout=30000 -c search_path=from_service"


def test_a_service_named_by_pgservice_is_read_too(service_file, monkeypatch):
    monkeypatch.setenv("PGSERVICE", "prod")
    assert _options(bounded_dsn("host=h dbname=d", 30)).endswith("-c search_path=from_service")


def test_a_service_s_options_outrank_pgoptions_as_libpq_ranks_them(service_file, monkeypatch):
    monkeypatch.setenv("PGOPTIONS", "-c work_mem=64MB")
    assert "work_mem" not in _options(bounded_dsn("service=prod", 30))


def test_options_in_the_dsn_outrank_the_service(service_file):
    dsn = bounded_dsn("service=prod options='-c search_path=from_dsn'", 30)
    assert _options(dsn) == "-c statement_timeout=30000 -c search_path=from_dsn"


def test_a_service_without_options_falls_back_to_pgoptions(service_file, monkeypatch):
    monkeypatch.setenv("PGOPTIONS", "-c work_mem=64MB")
    assert _options(bounded_dsn("service=bare", 30)) == "-c statement_timeout=30000 -c work_mem=64MB"


def test_a_service_that_cannot_be_found_is_left_unbounded_and_says_so(service_file, caplog):
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        assert bounded_dsn("service=missing", 30) == "service=missing"
    assert any("missing" in r.getMessage() and "statement timeout" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


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

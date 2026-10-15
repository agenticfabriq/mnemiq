"""Codex's reviews of #134 and #140 (M152): the statement-timeout bound never drops a setting the
connection's startup applied.

The bound becomes the connection's `options`, and libpq lets that REPLACE the options a connection
service supplies -- measured on a throwaway Postgres 16, a service setting `-c
search_path=from_service` connected with `"$user", public` once bounded (#134's review). #140 then
rebuilt a service's options from the server's account of them (`pg_settings`, source 'client', plus
`current_setting('role')`), and that account is incomplete: `pg_settings` hides every custom
setting -- `-c app.tenant_id=42` was in effect and not listed, measured -- so a bounded service
lost the value a row policy may read (#140's review). There is no complete account to be had from
the server, and the service file may sit in libpq's compiled-in directory, so a service without
options in its DSN is now left unbounded, with a warning. A DSN's own options and PGOPTIONS are
known exactly and carried verbatim.

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


def test_a_dsn_s_own_options_are_carried_verbatim_a_role_included():
    """A role and a custom setting are exactly what the server's account could not see."""
    dsn = bounded_dsn("host=h options='-c role=reporting -c app.tenant_id=42'", 30)
    assert _options(dsn) == "-c statement_timeout=30000 -c role=reporting -c app.tenant_id=42"


def test_a_service_dsn_s_own_options_outrank_the_services():
    """libpq lets a DSN's `options` replace its service's, so they are what the startup applied
    and are carried verbatim."""
    dsn = bounded_dsn("service=prod options='-c role=reporting'", 30)
    assert _options(dsn) == "-c statement_timeout=30000 -c role=reporting"


@pytest.mark.parametrize("dsn,env", [("service=prod", None), ("host=h", "prod"),
                                     ("host=h servicefile=/etc/x.conf", None),
                                     ("postgresql://h/d?servicefile=/etc/x.conf", None)])
def test_a_service_without_options_in_its_dsn_is_left_unbounded(dsn, env, monkeypatch, caplog):
    """Its options cannot be known from here, and adding ours would replace them."""
    if env:
        monkeypatch.setenv("PGSERVICE", env)
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        assert bounded_dsn(dsn, 30) == dsn
    assert any("statement timeout" in r.getMessage() for r in caplog.records)


def test_a_plain_dsn_still_carries_pgoptions(monkeypatch):
    monkeypatch.setenv("PGOPTIONS", "-c work_mem=64MB")
    assert _options(bounded_dsn("host=h", 30)) == "-c statement_timeout=30000 -c work_mem=64MB"


class _Con:
    """Records what the adapter executes; `limit` is the source's own statement_timeout in ms."""

    def __init__(self, limit):
        self.sql, self.limit = [], limit

    def execute(self, statement, *args):
        self.sql.append(statement)
        return self

    def fetchone(self):
        return (self.limit,)


def _attach(monkeypatch, dsn, limit):
    import mnemiq.adapters.duckdb as module
    from mnemiq.adapters.duckdb import DuckDBAdapter

    con = _Con(limit)
    monkeypatch.setattr(module, "connect_duckdb", lambda **kw: con)
    monkeypatch.setattr(module, "load_extension", lambda *a, **kw: None)
    monkeypatch.setattr(module, "lock_configuration", lambda *a, **kw: None)
    DuckDBAdapter.postgres(dsn, statement_timeout_s=30)
    return [s for s in con.sql if s.startswith(("ATTACH", "DETACH"))]


def test_attach_never_rebuilds_a_service_from_what_the_server_reports(monkeypatch):
    """The join: a service with no limit of its own is attached once, as configured."""
    attaches = _attach(monkeypatch, "service=prod", "0")
    assert len(attaches) == 1 and "statement_timeout" not in attaches[0], attaches


def test_a_service_bounded_by_its_own_options_raises_no_warning(monkeypatch, caplog):
    """The limit is read before the DSN is judged, so an operator who followed the warning --
    `-c statement_timeout=...` in the service -- stops seeing it."""
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        attaches = _attach(monkeypatch, "service=prod", "30000")
    assert len(attaches) == 1, attaches
    assert not any("statement timeout" in r.getMessage() for r in caplog.records)


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


_SETTINGS = ("SELECT * FROM postgres_query('src', 'SELECT current_setting(''search_path'') AS sp, "
             "current_setting(''app.tenant_id'', true) AS t, "
             "current_setting(''statement_timeout'') AS st')")


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_a_live_service_keeps_every_setting_it_applies(tmp_path, monkeypatch):
    """Live: the measurement that found it, as a test. A service with a custom setting -- one
    `pg_settings` never lists -- keeps it, and its search_path, and is not bounded by us."""
    from psycopg.conninfo import conninfo_to_dict

    from mnemiq.adapters.duckdb import DuckDBAdapter

    params = conninfo_to_dict(os.environ["MNEMIQ_PG_DSN"])
    params.pop("options", None)
    lines = ["[live]", *(f"{k}={v}" for k, v in params.items()),
             "options=-c search_path=from_service -c app.tenant_id=42"]
    (tmp_path / "pg_service.conf").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("PGSERVICEFILE", str(tmp_path / "pg_service.conf"))
    adapter = DuckDBAdapter.postgres("service=live", statement_timeout_s=30)
    assert adapter.execute(_SETTINGS) == [("from_service", "42", "0")]


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_a_live_dsn_keeps_its_custom_setting_and_gains_the_bound():
    """Live: the verbatim carry, through DuckDB's own libpq."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from mnemiq.adapters.duckdb import DuckDBAdapter

    params = conninfo_to_dict(os.environ["MNEMIQ_PG_DSN"])
    params["options"] = "-c search_path=from_dsn -c app.tenant_id=42"
    adapter = DuckDBAdapter.postgres(make_conninfo(**params), statement_timeout_s=30)
    assert adapter.execute(_SETTINGS) == [("from_dsn", "42", "30s")]

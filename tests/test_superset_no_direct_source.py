"""Superset mode opens no database connection of its own, at every place that would open one.

Every source connection mnemiq opens goes through `source_adapter` (the runtime, and the sweeps grading against the
same sources), `adapter_for` (`source_adapter`'s single source, enrich and refresh) or `run_acme` (`mnemiq eval`,
which reads MNEMIQ_PG_DSN itself). With MNEMIQ_SUPERSET_MCP_URL set, each refuses before it reads a manifest or
builds an adapter, so a deployment configured for Superset never
falls back to reading the source with mnemiq's own credential. With the URL unset, each behaves as it always has:
the direct-mode paths are also covered by tests/test_source_resolve.py, among them
`test_the_default_postgres_source_attaches_exactly_as_before` and `test_build_runtime_dispatches_on_the_spec_not_on_pg_dsn`.
"""

import json
import os

import pytest

from mnemiq.adapters import resolve
from mnemiq.config import Settings
from mnemiq.superset.settings import SupersetConfigError

URL = "https://superset.example.com/mcp"
DSN = "postgresql://reader@127.0.0.1:5432/warehouse"


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment: a variable set in the shell running the tests must not decide them."""
    for name in [n for n in os.environ if n.startswith("MNEMIQ_") and n != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


class _Built:
    """Stands in for an adapter class, and records that one was built."""

    made: list = []

    def __init__(self, *args, **kwargs):
        _Built.made.append((args, kwargs))


@pytest.fixture
def built(monkeypatch):
    """Every adapter class a source connection is built from, replaced by a recorder: a test sees whether one was
    built, and none ever connects."""
    _Built.made = []
    monkeypatch.setattr(resolve, "DuckDBAdapter", _Built)
    monkeypatch.setattr(resolve, "OracleAdapter", _Built)
    monkeypatch.setattr("mnemiq.adapters.federated.FederatedAdapter", _Built)
    return _Built.made


def _manifest(tmp_path) -> str:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([
        {"id": "a", "kind": "sqlite", "target": str(tmp_path / "a.db"), "catalog": "a", "schema": "main"},
        {"id": "b", "kind": "postgres", "target": DSN, "catalog": "b", "schema": "public"}]))
    return str(path)


def _refused(call) -> str:
    with pytest.raises(SupersetConfigError) as exc:
        call()
    message = str(exc.value)
    assert "MNEMIQ_SUPERSET_MCP_URL" in message, message
    assert "no database connection of its own" in message, message
    return message


def test_source_adapter_refuses_in_superset_mode(built):
    from mnemiq.runtime import source_adapter

    settings = Settings(superset_mcp_url=URL, superset_database_id=1, pg_dsn=DSN)
    _refused(lambda: source_adapter(settings, read_only=True))
    assert built == []


def test_adapter_for_refuses_in_superset_mode(built):
    settings = Settings(superset_mcp_url=URL, superset_database_id=1, pg_dsn=DSN)
    _refused(lambda: resolve.adapter_for(settings.source_specs()[0], settings))
    assert built == []


def test_a_sources_manifest_is_refused_before_federation(built, tmp_path):
    from mnemiq.runtime import source_adapter

    settings = Settings(superset_mcp_url=URL, superset_database_id=1, sources_path=_manifest(tmp_path))
    assert len(settings.source_specs()) == 2
    _refused(lambda: source_adapter(settings, read_only=True))
    assert built == []


def test_the_refusal_comes_before_the_manifest_is_read(built, tmp_path):
    """A manifest that cannot even be read is not reached: the refusal is about the mode, not the sources."""
    from mnemiq.runtime import source_adapter

    settings = Settings(superset_mcp_url=URL, superset_database_id=1, sources_path=str(tmp_path / "missing.json"))
    _refused(lambda: source_adapter(settings, read_only=True))


@pytest.mark.parametrize("unset", [None, ""])
def test_direct_mode_builds_its_source_as_before(built, tmp_path, unset):
    """The known-good case: with the URL unset (or blank, as a sourced `.env.example` leaves it), both doors build
    the adapter they always built."""
    from mnemiq.runtime import source_adapter

    single = Settings(superset_mcp_url=unset, pg_dsn=DSN)
    assert isinstance(resolve.adapter_for(single.source_specs()[0], single), _Built)
    assert isinstance(source_adapter(single, read_only=True), _Built)
    federated = Settings(superset_mcp_url=unset, sources_path=_manifest(tmp_path))
    assert isinstance(source_adapter(federated, read_only=True), _Built)
    assert len(built) == 3


class _Reached(Exception):
    """Raised by a stand-in adapter class: the command got as far as building its source connection."""


def _reached(*args, **kwargs):
    raise _Reached


def test_eval_refuses_in_superset_mode(monkeypatch):
    """`mnemiq eval` builds its own Postgres adapter on MNEMIQ_PG_DSN rather than through `adapter_for`."""
    from mnemiq.eval import run

    monkeypatch.setattr(run, "DuckDBPostgresAdapter", _reached)
    settings = Settings(superset_mcp_url=URL, superset_database_id=1, pg_dsn=DSN)
    _refused(lambda: run.run_acme(settings))


def test_eval_in_direct_mode_reaches_its_source_as_before(monkeypatch):
    from mnemiq.eval import run

    monkeypatch.setattr(run, "DuckDBPostgresAdapter", _reached)
    with pytest.raises(_Reached):
        run.run_acme(Settings(pg_dsn=DSN))

from __future__ import annotations

import os

import pytest

from tests.starter_fixtures import shipped_starters_untouched


@pytest.fixture(autouse=True)
def _shipped_starters_untouched():
    """No test writes to the starters folder the engine ships. A test that needs starters makes its own folder
    (`starter_fixtures.starter_folder`), and one that writes to the shipped folder fails, naming the file."""
    with shipped_starters_untouched():
        yield


@pytest.fixture(scope="session", autouse=True)
def _seed_acme(request):
    """Seed ACME into Postgres once per session when integration tests are selected.

    Removes any ordering dependency between the seed and adapter tests, and makes a
    CI integration run self-sufficient. No-op unless integration tests are collected
    and both MNEMIQ_PG_DSN and MNEMIQ_ACME_DATA_DIR are set.
    """
    if not any(item.get_closest_marker("integration") for item in request.session.items):
        return
    dsn = os.getenv("MNEMIQ_PG_DSN")
    data_dir = os.getenv("MNEMIQ_ACME_DATA_DIR")
    if dsn and data_dir:
        from scripts.seed_acme import seed

        seed(dsn, data_dir)


@pytest.fixture(autouse=True)
def _superset_dialect_kept(monkeypatch):
    """Superset mode's SQL dialect put back after each test as it was before: boot sets it once in a process
    (`use_superset_dialect`) and refuses another, so a test that boots a runtime would otherwise decide the dialect of
    every test after it."""
    from mnemiq.superset import settings as superset_settings

    monkeypatch.setattr(superset_settings, "_dialect", superset_settings._dialect)


@pytest.fixture
def superset_dialect_as(monkeypatch):
    """Put this test's process in a dialect, as a process that booted for a snapshot read from that database is, for a
    test that compares what mnemiq writes in two dialects, which no one process does. Put back after the test."""
    from mnemiq.superset import settings as superset_settings

    def put(dialect: str) -> None:
        assert dialect in ("postgres", "oracle")
        monkeypatch.setattr(superset_settings, "_dialect", dialect)

    return put


@pytest.fixture
def superset_oracle(superset_dialect_as):
    """Superset mode's SQL dialect set to Oracle's for one test, as boot sets it for a snapshot read from Oracle, and
    put back as it was after the test."""
    superset_dialect_as("oracle")


@pytest.fixture(autouse=True)
def _isolate_verity_cache(tmp_path_factory, monkeypatch, request):
    """Keep the Verity record cache out of the working tree.

    `Settings.store_path` defaults to `mnemiq.duckdb` in the CWD, and the record sidecar is
    resolved beside the store -- so any test constructing `Settings(verity_records_url=...)`
    without an explicit path wrote `verity-watermark.json` into the repo. That was invisible
    while the sidecar held one string; now that it holds the cached records it is worse than
    untidy, because the next test READS them and its result depends on collection order.

    `test_config` asserts the default itself, so it opts out.
    """
    if request.node.get_closest_marker("uses_real_store_default"):
        return
    if request.node.nodeid.startswith("tests/test_config.py"):
        return
    monkeypatch.setenv("MNEMIQ_STORE_PATH", str(tmp_path_factory.mktemp("store") / "mnemiq.duckdb"))

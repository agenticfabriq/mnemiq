from __future__ import annotations

import os

import pytest


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

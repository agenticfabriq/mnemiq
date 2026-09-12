import os

import pytest

from acme_dsn import acme_dsn, requires_acme

from scripts.seed_acme import seed

pytestmark = [pytest.mark.integration, requires_acme]


def _dsn():
    return acme_dsn()


def test_seed_loads_party_rows():
    # SKIP, not KeyError. `conftest._seed_acme` already treats this var as optional -- it
    # no-ops unless both it and MNEMIQ_PG_DSN are set -- so a checkout with Postgres running
    # and no CSV directory is a supported state, and this was the one place that turned it
    # into a red test. Six failures in the local suite is how a seventh goes unnoticed.
    data_dir = os.getenv("MNEMIQ_ACME_DATA_DIR")  # .../ACME_Insurance
    if not data_dir:
        pytest.skip("set MNEMIQ_ACME_DATA_DIR to the ACME CSV directory to run the seed test")
    counts = seed(_dsn(), data_dir)
    assert len(counts) == 29  # all 29 CSVs load (13 typed from DDL + 16 inferred)
    assert counts["party"] == 30  # Party.csv has 30 data rows (inferred table)
    assert counts["claim"] >= 1  # DDL-typed table
    assert "agreement" in counts  # duplicate-header CSV still loads

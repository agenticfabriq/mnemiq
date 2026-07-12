import os

import pytest

from scripts.seed_acme import seed

pytestmark = pytest.mark.integration


def _dsn():
    return os.getenv("MNEMIQ_PG_DSN", "postgresql://mnemiq:mnemiq@localhost:5432/acme")


def test_seed_loads_party_rows():
    data_dir = os.environ["MNEMIQ_ACME_DATA_DIR"]  # .../ACME_Insurance
    counts = seed(_dsn(), data_dir)
    assert len(counts) == 29  # all 29 CSVs load (13 typed from DDL + 16 inferred)
    assert counts["party"] == 30  # Party.csv has 30 data rows (inferred table)
    assert counts["claim"] >= 1  # DDL-typed table
    assert "agreement" in counts  # duplicate-header CSV still loads

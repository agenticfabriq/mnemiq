import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.eval.golden import load_cases

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"
_PATH = "evals/acme.json"


def test_the_golden_set_is_the_promised_size():
    cases = load_cases(_PATH)
    assert len(cases) == 30
    assert sum(1 for c in cases if not c.answerable) == 5


def test_every_case_id_is_unique():
    cases = load_cases(_PATH)
    assert len({c.id for c in cases}) == len(cases)


def test_answerable_cases_have_gold_sql_and_unanswerable_ones_do_not():
    for case in load_cases(_PATH):
        if case.answerable:
            assert case.gold_sql, f"{case.id} is answerable but has no gold SQL"
        else:
            assert case.gold_sql is None, f"{case.id} is unanswerable but has gold SQL"


@pytest.mark.integration
def test_every_gold_query_runs_and_returns_rows():
    """Ground truth must actually be true. A golden set nobody ran is a wish list."""
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))

    for case in load_cases(_PATH):
        if not case.answerable:
            continue
        table = adapter.execute_arrow(case.gold_sql, timeout_s=30)
        assert table.num_rows > 0, f"{case.id}: gold query returned nothing"

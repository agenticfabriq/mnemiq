import os

import pyarrow as pa
import pytest

from acme_dsn import acme_dsn, requires_acme

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.execute.runner import ExecutionError, run

pytestmark = [pytest.mark.integration, requires_acme]

_DSN = acme_dsn()


def _adapter():
    return DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))


def test_a_query_returns_an_arrow_table():
    result = run(_adapter(), "SELECT claim_identifier FROM claim LIMIT 5")

    assert isinstance(result.table, pa.Table)
    assert result.table.schema.names == ["claim_identifier"]
    assert result.row_count == result.table.num_rows <= 5
    assert result.elapsed_ms >= 0


def test_an_aggregate_comes_back_typed():
    result = run(_adapter(), "SELECT count(*) AS n FROM claim")
    assert result.row_count == 1
    assert result.table.column("n")[0].as_py() > 0


def test_a_bad_query_raises_an_error_the_model_can_repair():
    with pytest.raises(ExecutionError) as excinfo:
        run(_adapter(), "SELECT no_such_column FROM claim")
    # `repair_text`, not `str(exc)`: the source's words are the repairable part and they are
    # deliberately not in the default string, because that is the one a caller-facing message
    # gets written from by accident.
    assert "no_such_column" in excinfo.value.repair_text
    assert "no_such_column" not in str(excinfo.value)


def test_a_runaway_query_is_interrupted():
    # LIMIT bounds what comes back; it does not bound a query that never reaches the limit.
    with pytest.raises(ExecutionError) as excinfo:
        run(_adapter(), "SELECT count(*) FROM range(100000000000)", timeout_s=0.5)
    assert "timed out" in str(excinfo.value).lower()

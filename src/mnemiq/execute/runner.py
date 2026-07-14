from __future__ import annotations

import time
from dataclasses import dataclass

import pyarrow as pa

DEFAULT_TIMEOUT_S = 30.0


class ExecutionError(Exception):
    """The source refused to run the query. The message is written for the model to repair."""


@dataclass
class ExecutionResult:
    table: pa.Table
    row_count: int
    elapsed_ms: float


def run(adapter, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> ExecutionResult:
    started = time.perf_counter()
    try:
        table = adapter.execute_arrow(sql, timeout_s=timeout_s)
    except Exception as exc:
        if "interrupt" in type(exc).__name__.lower() or "interrupt" in str(exc).lower():
            raise ExecutionError(
                f"The query timed out after {timeout_s}s. Make it cheaper: fewer joins, "
                "narrower filters, or an aggregate instead of raw rows."
            ) from exc
        raise ExecutionError(f"The source rejected this query: {exc}") from exc

    elapsed = (time.perf_counter() - started) * 1000
    return ExecutionResult(table=table, row_count=table.num_rows, elapsed_ms=elapsed)

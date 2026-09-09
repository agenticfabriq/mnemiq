from __future__ import annotations

import time
from dataclasses import dataclass

import pyarrow as pa

DEFAULT_TIMEOUT_S = 30.0


class ExecutionError(Exception):
    """The source refused to run the query.

    `message` is written for a reader: the model repairs against it and the caller may be
    shown it, so it holds only words this engine authored. When the source's own exception
    is what explains the failure, it goes in `source_detail` instead -- that text names the
    relation and column it refused and can carry a DSN, and the agent forwards the
    caller-facing half into the answer. Same split as `Refusal`, for the same reason.
    """

    def __init__(self, message: str, source_detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.source_detail = source_detail

    @property
    def repair_text(self) -> str:
        """What the model repairs against: our sentence, plus the source's own words."""
        return f"{self.message} {self.source_detail}" if self.source_detail else self.message


@dataclass
class ExecutionResult:
    table: pa.Table
    row_count: int
    elapsed_ms: float


def run(adapter, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> ExecutionResult:
    started = time.perf_counter()
    try:
        table = adapter.execute_arrow(sql, timeout_s=timeout_s)
    except ExecutionError:
        # Already in this form, with its two halves assigned. Re-wrapping would demote a
        # sentence we wrote into `source_detail` and hide it from the caller -- which the old
        # `f"...: {exc}"` concatenation obscured, because a nested message still came out
        # somewhere in the string.
        raise
    except Exception as exc:
        if "interrupt" in type(exc).__name__.lower() or "interrupt" in str(exc).lower():
            raise ExecutionError(
                f"The query timed out after {timeout_s}s. Make it cheaper: fewer joins, "
                "narrower filters, or an aggregate instead of raw rows."
            ) from exc
        raise ExecutionError("The source rejected this query.", source_detail=str(exc)) from exc

    elapsed = (time.perf_counter() - started) * 1000
    return ExecutionResult(table=table, row_count=table.num_rows, elapsed_ms=elapsed)

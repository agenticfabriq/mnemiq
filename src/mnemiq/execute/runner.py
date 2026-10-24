from __future__ import annotations

import time
from dataclasses import dataclass

import pyarrow as pa

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.sql.prove import source_did_not_answer

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
        """What the model repairs against: our sentence, plus the source's own words -- with any
        credential in them taken out first (M96), as `Refusal.repair_text` does."""
        if not self.source_detail:
            return self.message
        from mnemiq.redact import scrub_credentials

        return f"{self.message} {scrub_credentials(self.source_detail)}"

    @property
    def full_text(self) -> str:
        """The whole of it, credentials included -- for the operator's log only, never a prompt.
        The operator holds the credential already, and a log scrubbed like the prompt would hide
        which connection failed from the one person who can fix it."""
        return f"{self.message} {self.source_detail}" if self.source_detail else self.message


class SourceUnavailableError(ExecutionError):
    """The source did not answer while the statement ran: a connection that dropped after the
    proof, or mid-fetch (M188). The proof seam reads the same fault before the run as
    SOURCE_UNAVAILABLE; here it reached the model as "The source rejected this query." and was
    repaired, a correct query rewritten because the network failed. Not the statement's failure,
    so never the model's to repair. An `ExecutionError` still, so every caller catching that
    keeps its reading, a statement that produced no table."""


class QueryCancelledError(ExecutionError):
    """The source ended the statement on a cancel the bound did not send: an operator's or another
    session's. Neither the statement's failure nor an outage, and ended on, not repaired: handed to
    the repair loop, a model that wrote the same SQL again ran the stopped statement a second time
    (Codex's second review of #179). An `ExecutionError` still, as `SourceUnavailableError` is."""


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
        # The timeout first: DuckDB's classifier calls its interrupt unavailable, the proof seam's
        # reading of a proof that never came back, and here it is the statement's cost.
        if (isinstance(exc, QueryTimedOut) or "interrupt" in type(exc).__name__.lower()
                or "interrupt" in str(exc).lower()):
            raise ExecutionError(
                f"The query timed out after {timeout_s}s. Make it cheaper: fewer joins, "
                "narrower filters, or an aggregate instead of raw rows."
            ) from exc
        if isinstance(exc, QueryCancelled):
            # Before the outage question, which the driver's classifier answers yes for its own
            # cancel: read as an outage, the statement someone stopped would be run again.
            raise QueryCancelledError(
                "The source cancelled this query before it finished: an operator or another "
                "session stopped it, so it was not run again. Ask again once whoever stopped it "
                "is done; the server log has the source's words.",
                source_detail=str(exc),
            ) from exc
        if source_did_not_answer(adapter, exc):
            raise SourceUnavailableError(
                "The source did not answer while this query ran. That is not about the query. "
                "Try again, and tell the operator if it persists; the server log has the "
                "source's error.",
                source_detail=str(exc),
            ) from exc
        raise ExecutionError("The source rejected this query.", source_detail=str(exc)) from exc

    elapsed = (time.perf_counter() - started) * 1000
    return ExecutionResult(table=table, row_count=table.num_rows, elapsed_ms=elapsed)

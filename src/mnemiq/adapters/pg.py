from __future__ import annotations

import pyarrow as pa


def unavailable_fault(exc: BaseException) -> bool:
    """Whether a psycopg error says the source could not be asked, rather than that it answered
    no: the proof seam's `source_unavailable` (Codex's review of #159). psycopg types a lost
    connection and a cancelled statement (its statement timeout, `QueryCanceled`)
    `OperationalError`, and a statement's own faults (`UndefinedColumn`, `SyntaxError`,
    `InsufficientPrivilege`) `ProgrammingError`, which keep their reading as a refusal."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import psycopg
    except ImportError:  # pragma: no cover -- a psycopg error exists only with the driver
        return False
    return isinstance(exc, psycopg.OperationalError)


class PostgresAdapter:
    """Native Postgres via psycopg -- executes BIRD gold SQL exactly as written (PG dialect),
    for grading. The engine's own SQL runs via DuckDBAdapter (DuckDB ATTACH); this is the gold
    side only, so a gold query is never transpiled."""

    dialect = "postgres"

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._con = psycopg.connect(dsn, autocommit=True)

    def execute(self, sql: str) -> list[tuple]:
        with self._con.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def source_unavailable(self, exc: BaseException) -> bool:
        """The proof seam's question: did the source not answer, rather than answer no."""
        return unavailable_fault(exc)

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        with self._con.cursor() as cur:
            if timeout_s is not None:
                cur.execute(f"SET statement_timeout = {int(timeout_s * 1000)}")
            cur.execute(sql)
            names = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        # Column-major, by position -- so duplicate output names survive (a dict would not).
        columns = list(zip(*rows, strict=False)) if rows else [() for _ in names]
        arrays = [pa.array(list(col)) for col in columns]
        return pa.Table.from_arrays(arrays, names=names)

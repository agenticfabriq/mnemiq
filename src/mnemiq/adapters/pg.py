from __future__ import annotations

import pyarrow as pa


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

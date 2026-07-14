from __future__ import annotations

import threading

import duckdb
import pyarrow as pa


class DuckDBPostgresAdapter:
    """Query a Postgres source through DuckDB's postgres extension (ATTACH)."""

    def __init__(self, dsn: str, schema: str = "src") -> None:
        self._schema = schema
        self._con = duckdb.connect()
        self._con.execute("INSTALL postgres; LOAD postgres;")
        self._con.execute(f"ATTACH '{dsn}' AS {schema} (TYPE POSTGRES, READ_ONLY)")
        self._con.execute(f"USE {schema}.public")

    def introspect(self) -> list[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_catalog = '{self._schema}' AND table_schema = 'public'"
        ).fetchall()
        return [r[0] for r in rows]

    def list_columns(self) -> list[tuple[str, str, str]]:
        rows = self._con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            f"WHERE table_catalog = '{self._schema}' AND table_schema = 'public' "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        """Declared FKs: (from_table, from_col, to_table, to_col, constraint_id).

        Run through postgres_query so the information_schema joins execute with real
        Postgres semantics (not DuckDB's proxy). The constraint_name is the id that keeps
        independent FKs to the same parent distinct. Any failure -> [] (fall back to inference).
        """
        query = (
            "SELECT kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name, "
            "  tc.constraint_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            "  AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "  ON tc.constraint_name = ccu.constraint_name "
            "  AND tc.table_schema = ccu.table_schema "
            "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public' "
            "ORDER BY kcu.table_name, kcu.ordinal_position"
        )
        try:
            rows = self._con.execute(
                f"SELECT * FROM postgres_query('{self._schema}', $q${query}$q$)"
            ).fetchall()
        except Exception:
            return []
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """Run a query and return Arrow, cancelling it if it overruns.

        A LIMIT bounds the rows that come back. It does not bound a query that never
        produces a first row -- only an interrupt does that.
        """
        timer = None
        if timeout_s is not None:
            timer = threading.Timer(timeout_s, self._con.interrupt)
            timer.start()
        try:
            return self._con.execute(sql).to_arrow_table()
        finally:
            if timer is not None:
                timer.cancel()

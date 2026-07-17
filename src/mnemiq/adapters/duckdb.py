from __future__ import annotations

import threading

import duckdb
import pyarrow as pa

# The declared-FK query for Postgres sources: run through postgres_query so the
# information_schema joins execute with real Postgres semantics, not DuckDB's proxy.
_PG_FK_QUERY = (
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


class DuckDBAdapter:
    """DuckDB as the universal executor: ATTACH a source and read it via DuckDB's scanner.

    The engine's plan is written in duckdb and executed here, so DuckDB-native functions
    (YEAR, EXTRACT, LISTAGG) always work -- regardless of what the underlying source is.
    Build one via a factory: DuckDBAdapter.postgres(dsn) or DuckDBAdapter.sqlite(path).
    """

    dialect = "duckdb"

    def __init__(
        self,
        *,
        attach_target: str,
        attach_type: str,
        extension: str,
        catalog: str,
        table_schema: str,
        fk_via_postgres: bool,
        read_only: bool = True,
    ) -> None:
        self._catalog = catalog
        self._table_schema = table_schema
        self._fk_via_postgres = fk_via_postgres
        self._con = duckdb.connect()
        self._con.execute(f"INSTALL {extension}; LOAD {extension};")
        # READ_ONLY unless a write is explicitly enabled -- the backstop under the write path.
        clause = f"(TYPE {attach_type}, READ_ONLY)" if read_only else f"(TYPE {attach_type})"
        self._con.execute(f"ATTACH '{attach_target}' AS {catalog} {clause}")
        self._con.execute(f"USE {catalog}.{table_schema}")

    @classmethod
    def postgres(cls, dsn: str, schema: str = "src", read_only: bool = True) -> "DuckDBAdapter":
        return cls(attach_target=dsn, attach_type="POSTGRES", extension="postgres",
                   catalog=schema, table_schema="public", fk_via_postgres=True, read_only=read_only)

    @classmethod
    def sqlite(cls, path: str, schema: str = "s", read_only: bool = True) -> "DuckDBAdapter":
        return cls(attach_target=path, attach_type="SQLITE", extension="sqlite",
                   catalog=schema, table_schema="main", fk_via_postgres=False, read_only=read_only)

    def introspect(self) -> list[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_catalog = '{self._catalog}' AND table_schema = '{self._table_schema}'"
        ).fetchall()
        return [r[0] for r in rows]

    def list_columns(self) -> list[tuple[str, str, str]]:
        rows = self._con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            f"WHERE table_catalog = '{self._catalog}' AND table_schema = '{self._table_schema}' "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        """Declared FKs: (from_table, from_col, to_table, to_col, constraint_id).

        Postgres exposes them via information_schema; DuckDB's SQLite scanner does not, so a
        SQLite source returns [] (enrichment falls back to data-driven inference). Any
        failure -> [] (fall back to inference), never an exception.
        """
        if not self._fk_via_postgres:
            return []
        try:
            rows = self._con.execute(
                f"SELECT * FROM postgres_query('{self._catalog}', $q${_PG_FK_QUERY}$q$)"
            ).fetchall()
        except Exception:
            return []
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """Run a query and return Arrow, cancelling it if it overruns. A LIMIT bounds rows;
        only an interrupt bounds a query that never produces a first row."""
        timer = None
        if timeout_s is not None:
            timer = threading.Timer(timeout_s, self._con.interrupt)
            timer.start()
        try:
            return self._con.execute(sql).to_arrow_table()
        finally:
            if timer is not None:
                timer.cancel()


class DuckDBPostgresAdapter(DuckDBAdapter):
    """Compat: Query a Postgres source through DuckDB's postgres extension (ATTACH).

    Preserved as a thin subclass so existing imports and call sites are unchanged.
    """

    def __init__(self, dsn: str, schema: str = "src", read_only: bool = True) -> None:
        super().__init__(attach_target=dsn, attach_type="POSTGRES", extension="postgres",
                         catalog=schema, table_schema="public", fk_via_postgres=True,
                         read_only=read_only)

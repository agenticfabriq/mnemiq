from __future__ import annotations

import duckdb


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

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

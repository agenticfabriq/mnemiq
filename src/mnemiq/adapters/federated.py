from __future__ import annotations

import threading

import duckdb
import pyarrow as pa

from mnemiq.config import SourceSpec

_EXT = {"postgres": ("postgres", "POSTGRES"), "sqlite": ("sqlite", "SQLITE")}


class FederatedAdapter:
    """DuckDB as the federated executor: ATTACH every source as its own catalog into ONE
    connection; DuckDB's optimizer pushes predicates/projections into each source scanner and
    performs the cross-source join. Plans are written and executed in duckdb (no transpile).
    Used only when >=2 sources are configured; a single source keeps the DuckDBAdapter path."""

    dialect = "duckdb"

    def __init__(self, specs: list[SourceSpec], read_only: bool = True) -> None:
        self.registry: dict[str, str] = {s.catalog: s.schema for s in specs}
        self.catalogs = frozenset(self.registry)
        self._con = duckdb.connect()
        loaded: set[str] = set()
        for spec in specs:
            ext, attach_type = _EXT[spec.kind]
            if ext not in loaded:
                self._con.execute(f"INSTALL {ext}; LOAD {ext};")
                loaded.add(ext)
            clause = f"(TYPE {attach_type}, READ_ONLY)" if read_only else f"(TYPE {attach_type})"
            self._con.execute(f"ATTACH '{spec.target}' AS {spec.catalog} {clause}")
        # No single USE: every query is catalog-qualified (catalog.schema.table).

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        timer = None
        if timeout_s is not None:
            timer = threading.Timer(timeout_s, self._con.interrupt)
            timer.start()
        try:
            return self._con.execute(sql).to_arrow_table()
        finally:
            if timer is not None:
                timer.cancel()

from __future__ import annotations

import threading

import duckdb
import pyarrow as pa

from mnemiq.config import SourceSpec

_EXT = {"postgres": ("postgres", "POSTGRES"), "sqlite": ("sqlite", "SQLITE")}


class UnfederatableSource(ValueError):
    """A source names a kind DuckDB has no ATTACH scanner for, so it cannot be federated."""


class FederatedAdapter:
    """DuckDB as the federated executor: ATTACH every source as its own catalog into ONE
    connection; DuckDB's optimizer pushes predicates/projections into each source scanner and
    performs the cross-source join. Plans are written and executed in duckdb (no transpile).
    Used only when >=2 sources are configured; a single source keeps the DuckDBAdapter path."""

    dialect = "duckdb"

    def __init__(self, specs: list[SourceSpec], read_only: bool = True) -> None:
        self.registry: dict[str, str] = {s.catalog: s.schema for s in specs}
        self.catalogs = frozenset(self.registry)
        # Validate the WHOLE manifest before connecting to any of it. Checked inside the attach
        # loop, an unfederatable source in position three would be reported only after two live
        # connections had already been made -- a half-built adapter that then raises. A bare
        # KeyError told an operator nothing either: DuckDB has no ATTACH scanner for Oracle, so
        # say that, and say which source, rather than failing with the name of a lookup table.
        for spec in specs:
            if spec.kind not in _EXT:
                raise UnfederatableSource(
                    f"source {spec.id!r} has kind={spec.kind!r}, which DuckDB cannot ATTACH; "
                    f"federation supports {', '.join(sorted(_EXT))}. Configure it as the only "
                    "source, or reach it through a source DuckDB can attach."
                )
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

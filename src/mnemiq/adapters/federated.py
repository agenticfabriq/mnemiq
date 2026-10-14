from __future__ import annotations

import pyarrow as pa

from mnemiq.adapters.duckdb import (
    UnparseableDSN,
    attach_literal,
    attach_postgres,
    duckdb_builtin_functions,
    duckdb_reachable_user_functions,
    duckdb_user_functions,
    execute_bounded,
)
from mnemiq.config import SourceSpec
from mnemiq.duckdb_extensions import (
    SOURCE_EXTENSIONS,
    connect_duckdb,
    load_extension,
    lock_configuration,
)

_EXT = {"postgres": (SOURCE_EXTENSIONS["postgres"], "POSTGRES"),
        "sqlite": (SOURCE_EXTENSIONS["sqlite"], "SQLITE")}


class UnfederatableSource(ValueError):
    """A source names a kind DuckDB has no ATTACH scanner for, so it cannot be federated."""


class FederatedAdapter:
    """DuckDB as the federated executor: ATTACH every source as its own catalog into ONE
    connection; DuckDB's optimizer pushes predicates/projections into each source scanner and
    performs the cross-source join. Plans are written and executed in duckdb (no transpile).
    Used only when >=2 sources are configured; a single source keeps the DuckDBAdapter path."""

    dialect = "duckdb"

    def __init__(
        self, specs: list[SourceSpec], read_only: bool = True, *, local_only: bool = False,
        statement_timeout_s: float | None = None,
    ) -> None:
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
        self._con = connect_duckdb(local_only=local_only)
        loaded: set[str] = set()
        for spec in specs:
            ext, attach_type = _EXT[spec.kind]
            if ext not in loaded:
                load_extension(self._con, ext, local_only=local_only)
                loaded.add(ext)
            clause = f"(TYPE {attach_type}, READ_ONLY)" if read_only else f"(TYPE {attach_type})"
            # Each Postgres catalog bounded server-side, as the single-source path is (M152).
            if attach_type == "POSTGRES":
                try:
                    attach_postgres(self._con, spec.target, spec.catalog, clause,
                                    statement_timeout_s)
                except UnparseableDSN as exc:  # name the source, as the single-source door does
                    raise UnparseableDSN(f"source {spec.id!r}: {exc}") from None
            else:
                self._con.execute(f"ATTACH {attach_literal(spec.target)} AS {spec.catalog} {clause}")
        # No single USE: every query is catalog-qualified (catalog.schema.table).
        lock_configuration(self._con, local_only=local_only)

    # The same three catalogue reads the single-source DuckDB path uses, on the same kind of
    # connection. Without them a federated deployment got `never_asked` and M43's residual fix
    # reached it not at all (M99) -- and the gap was invisible, because federation is DuckDB and
    # every argument written for the DuckDB adapter reads as if it applied here.
    #
    # `reachable` is usually empty in this shape and that is correct rather than a shortcut:
    # every query here is catalog-qualified and nothing issues a `USE`, so an attached source's
    # schema is not on the search path and its macros answer only to a spelled qualification --
    # which `called_names` reads off the rendered statement.
    def user_functions(self) -> list[str]:
        return duckdb_user_functions(self._con)

    def builtin_functions(self) -> list[str]:
        return duckdb_builtin_functions(self._con)

    def reachable_user_functions(self) -> list[str]:
        return duckdb_reachable_user_functions(self._con)

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        return execute_bounded(self._con, sql, timeout_s)

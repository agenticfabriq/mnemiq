from __future__ import annotations

import sqlite3
import threading

import pyarrow as pa

from mnemiq.adapters.base import QueryTimedOut


class SQLiteAdapter:
    """Read a SQLite database through the stdlib driver.

    Ground truth for BIRD is defined as "gold SQL run on SQLite", so gold executes here
    exactly as written -- never transpiled. The database is opened read-only: the engine
    is a read plane, and the decider is not the only thing standing between it and a write.
    """

    dialect = "sqlite"  # only relevant if used as an execution adapter; BIRD executes the
    # engine's SQL via DuckDBAdapter and uses this adapter for enrichment + native gold.

    def __init__(self, path: str) -> None:
        # mode=ro rejects every write at the driver; interrupt() is documented safe to call
        # from another thread even though the connection itself is single-threaded.
        self._con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def introspect(self) -> list[str]:
        rows = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            r"AND name NOT LIKE 'sqlite\_%' ESCAPE '\' ORDER BY name"  # `_` is a LIKE wildcard: escape it, or a user table like `sqlitex` is dropped (M225, Codex review)
        ).fetchall()
        return [r[0] for r in rows]

    def view_definitions(self) -> list[tuple[str, str, str]]:
        """SQLite stores the whole `CREATE VIEW ... AS ...` statement, not the bare body; the
        parser normalises that, because string-stripping the prefix is a second SQL dialect
        implemented in `str.partition`."""
        rows = self._con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'view' "
            r"AND name NOT LIKE 'sqlite\_%' ESCAPE '\' ORDER BY name"  # `_` is a LIKE wildcard: escape it, as introspect does
        ).fetchall()
        return [(r[0], r[1], "sqlite") for r in rows if r[1]]

    def list_columns(self) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for table in self.introspect():
            # PRAGMA table_xinfo, not table_info: xinfo lists GENERATED and hidden columns too, and the
            # engine reads those by name, so a governance inventory must hold them (a generated column
            # named like a SQL word was read while table_info omitted it, M225, Codex's review).
            # (cid, name, type, notnull, dflt_value, pk, hidden). The table-valued form, with the name as a
            # parameter: interpolated into PRAGMA table_xinfo("..."), a table named with a double quote
            # broke the statement, and since the live check inventories every table, one such table
            # refused every guarded-word query on the source (Codex's fourth review of that change).
            for row in self._con.execute("SELECT * FROM pragma_table_xinfo(?)", [table]).fetchall():
                out.append((table, row[1], (row[2] or "unknown")))
        return out

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        """Declared FKs: (from_table, from_col, to_table, to_col, constraint_id).

        The constraint_id groups a composite FK's columns and, crucially, keeps independent
        FKs to the same parent (eye/hair/skin -> colour) as distinct constraints.
        """
        out: list[tuple[str, str, str, str, str]] = []
        for table in self.introspect():
            # PRAGMA foreign_key_list: (id, seq, table(parent), from(child col), to(parent col), ...)
            # `id` is per-constraint (shared by a composite's columns).
            for row in self._con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
                fk_id, parent, from_col, to_col = row[0], row[2], row[3], row[4]
                out.append((table, from_col, parent, to_col, f"{table}:{fk_id}"))
        return out

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        timer = None
        if timeout_s is not None:
            timer = threading.Timer(timeout_s, self._con.interrupt)
            timer.start()
        try:
            cursor = self._con.execute(sql)
            names = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            if "interrupt" in str(exc).lower():
                raise QueryTimedOut(f"query timed out after {timeout_s}s") from exc
            raise
        finally:
            if timer is not None:
                timer.cancel()

        # Column-major, by position -- so duplicate output names survive (a dict would not).
        columns = list(zip(*rows)) if rows else [() for _ in names]
        arrays = [pa.array(list(col)) for col in columns]
        return pa.Table.from_arrays(arrays, names=names)

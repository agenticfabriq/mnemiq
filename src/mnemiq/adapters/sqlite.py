from __future__ import annotations

import sqlite3
import threading

import pyarrow as pa


class SQLiteAdapter:
    """Read a SQLite database through the stdlib driver.

    Ground truth for BIRD is defined as "gold SQL run on SQLite", so gold executes here
    exactly as written -- never transpiled. The database is opened read-only: the engine
    is a read plane, and the decider is not the only thing standing between it and a write.
    """

    def __init__(self, path: str) -> None:
        # mode=ro rejects every write at the driver; interrupt() is documented safe to call
        # from another thread even though the connection itself is single-threaded.
        self._con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def introspect(self) -> list[str]:
        rows = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def list_columns(self) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for table in self.introspect():
            # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
            for row in self._con.execute(f'PRAGMA table_info("{table}")').fetchall():
                out.append((table, row[1], (row[2] or "unknown")))
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
                raise RuntimeError(f"query timed out after {timeout_s}s") from exc
            raise
        finally:
            if timer is not None:
                timer.cancel()

        # Column-major, by position -- so duplicate output names survive (a dict would not).
        columns = list(zip(*rows)) if rows else [() for _ in names]
        arrays = [pa.array(list(col)) for col in columns]
        return pa.Table.from_arrays(arrays, names=names)

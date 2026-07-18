from __future__ import annotations

import threading

_CREATE = (
    "CREATE TABLE IF NOT EXISTS {t} "
    "(cache_key TEXT PRIMARY KEY, payload BYTEA, created_at TIMESTAMPTZ DEFAULT now())"
)
_GET = "SELECT payload FROM {t} WHERE cache_key = %s"
_PUT = (
    "INSERT INTO {t} (cache_key, payload) VALUES (%s, %s) "
    "ON CONFLICT (cache_key) DO UPDATE SET payload = EXCLUDED.payload, created_at = now()"
)


class PostgresCache:
    """Cross-replica L2 over a Postgres table. Fail-soft: a DB error is a miss / no-op, never an
    exception on the request path (a stale/absent cache is acceptable; a crash is not)."""

    def __init__(self, dsn: str, table: str = "mnemiq_cache", connect=None) -> None:
        self._table = table
        if connect is None:
            import psycopg

            def connect():
                return psycopg.connect(dsn, autocommit=True)

        self._connect = connect
        self._con = None
        self._lock = threading.Lock()
        with self._lock:
            self._ensure()

    def _reconnect(self) -> None:
        try:
            self._con = self._connect()
        except Exception:
            self._con = None

    def _ensure(self) -> None:
        if self._con is None:
            self._reconnect()
        if self._con is None:
            return
        try:
            self._con.execute(_CREATE.format(t=self._table))
        except Exception:
            self._con = None  # degrade; a later call retries

    def get(self, key: str) -> bytes | None:
        with self._lock:
            self._ensure()
            if self._con is None:
                return None
            try:
                row = self._con.execute(_GET.format(t=self._table), (key,)).fetchone()
                return bytes(row[0]) if row and row[0] is not None else None
            except Exception:
                self._con = None
                return None

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._ensure()
            if self._con is None:
                return
            try:
                self._con.execute(_PUT.format(t=self._table), (key, value))
            except Exception:
                self._con = None

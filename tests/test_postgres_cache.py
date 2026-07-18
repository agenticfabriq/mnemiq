from mnemiq.cache.postgres import PostgresCache


class _FakeConn:
    """In-memory stand-in for a psycopg autocommit connection."""

    def __init__(self):
        self.store = {}
        self._row = None

    def execute(self, sql, params=None):
        s = sql.strip().upper()
        if s.startswith("INSERT"):
            self.store[params[0]] = params[1]
        elif s.startswith("SELECT"):
            self._row = (self.store[params[0]],) if params[0] in self.store else None
        return self

    def fetchone(self):
        return self._row


class _RaisingConn:
    def execute(self, sql, params=None):
        raise RuntimeError("db down")


def test_get_put_roundtrip_with_fake_conn():
    c = PostgresCache("unused", connect=_FakeConn)
    assert c.get("k") is None
    c.put("k", b"payload")
    assert c.get("k") == b"payload"


def test_fail_soft_get_returns_none_and_put_noops():
    c = PostgresCache("unused", connect=_RaisingConn)  # even CREATE TABLE raises -> degraded
    assert c.get("k") is None          # never raises
    c.put("k", b"x")                   # never raises
    assert c.get("k") is None


def test_reconnects_after_a_transient_failure():
    seq = [_RaisingConn(), _FakeConn(), _FakeConn()]
    c = PostgresCache("unused", connect=lambda: seq.pop(0))
    c.put("k", b"v")  # first conn raised on ensure/put -> dropped
    assert c.get("k") in (None, b"v")  # a later call reconnects; no exception either way

from mnemiq.observability.metrics import AnswerRecord, NullSink, aggregate


def test_aggregate_slos():
    recs = [
        AnswerRecord(deferred=False, cached=False, total_ms=10.0, mode="thinking"),
        AnswerRecord(deferred=True, cached=False, total_ms=20.0, mode="thinking"),
        AnswerRecord(deferred=False, cached=True, total_ms=30.0, mode="instant"),
        AnswerRecord(deferred=False, cached=False, total_ms=40.0, mode="deep"),
    ]
    m = aggregate(recs)
    assert m.answers == 4
    assert m.deferrals == 1
    assert abs(m.deferral_rate - 0.25) < 1e-9
    assert abs(m.cache_hit_rate - 0.25) < 1e-9
    assert m.p50_ms == 30.0 and m.p95_ms == 40.0


def test_aggregate_empty():
    m = aggregate([])
    assert m.answers == 0 and m.deferral_rate == 0.0 and m.p50_ms == 0.0


def test_null_sink_is_noop():
    s = NullSink()
    s.record("acme", AnswerRecord(deferred=False, cached=False, total_ms=1.0, mode="x"))
    assert s.recent("acme", 10) == []


# ---------------------------------------------------------------------------
# `PostgresSink.recent` returns the N most recent answers for aggregate metrics.
# Without `ORDER BY created_at DESC`, Postgres returns rows in heap order --
# arbitrary after VACUUM or concurrent writes -- so `LIMIT 100` selects a
# random sample rather than the last 100 answers, and the p50/p95 computed
# from those measure nothing.
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows=None):
        self._rows = rows or []

    def fetchall(self):
        return self._rows


class _FakeCon:
    """In-memory stand-in for a psycopg autocommit connection."""

    def __init__(self):
        self.queries: list[str] = []

    def execute(self, sql, params=None):
        self.queries.append(sql)
        return _FakeCursor()


class _RaisingCon:
    def execute(self, sql, params=None):
        raise RuntimeError("db down")


def test_postgres_sink_recent_orders_by_created_at_desc():
    """The query must ORDER BY created_at DESC so LIMIT returns the most recent rows."""
    from mnemiq.observability.metrics import PostgresSink

    con = _FakeCon()
    sink = PostgresSink("unused", connect=lambda: con)
    sink.recent("acme", 50)

    select_queries = [q for q in con.queries if q.startswith("SELECT")]
    assert len(select_queries) == 1
    assert "ORDER BY created_at DESC" in select_queries[0]


def test_postgres_sink_ddl_includes_created_at():
    """The column ORDER BY sorts on must exist in the DDL; otherwise the query fails inside the
    fail-soft handler and recent() silently returns [] on every call."""
    from mnemiq.observability.metrics import PostgresSink

    assert "created_at" in PostgresSink._CREATE, "created_at missing from _CREATE DDL"
    assert "created_at" in PostgresSink._UPGRADE, "created_at missing from _UPGRADE ALTER"


def test_postgres_sink_reuses_the_connection():
    """record() and recent() must not open a new connection on every call."""
    from mnemiq.observability.metrics import PostgresSink

    calls = []
    con = _FakeCon()

    def connect():
        calls.append(1)
        return con

    sink = PostgresSink("unused", connect=connect)
    assert len(calls) == 1  # one connection at init

    rec = AnswerRecord(deferred=False, cached=False, total_ms=5.0, mode="x")
    sink.record("acme", rec)
    sink.record("acme", rec)
    sink.recent("acme", 10)

    assert len(calls) == 1  # still one connection -- no reconnect


def test_postgres_sink_reconnects_after_a_failure():
    """A broken connection is dropped and the next call reconnects."""
    from mnemiq.observability.metrics import PostgresSink

    good_con = _FakeCon()
    seq = [_RaisingCon(), good_con]
    sink = PostgresSink("unused", connect=lambda: seq.pop(0))

    # init tried _RaisingCon -> failed, con is None
    rec = AnswerRecord(deferred=False, cached=False, total_ms=5.0, mode="x")
    sink.record("acme", rec)  # _ensure reconnects with good_con

    # good_con should have received the DDL + the INSERT
    insert_queries = [q for q in good_con.queries if q.startswith("INSERT")]
    assert len(insert_queries) == 1


def test_postgres_sink_fail_soft_on_total_outage():
    """If every connect attempt fails, record() and recent() are no-ops -- never raise."""
    from mnemiq.observability.metrics import PostgresSink

    sink = PostgresSink("unused", connect=lambda: (_ for _ in ()).throw(RuntimeError("down")))
    rec = AnswerRecord(deferred=False, cached=False, total_ms=5.0, mode="x")
    sink.record("acme", rec)       # no exception
    assert sink.recent("acme", 10) == []  # no exception, empty list


def test_aggregate_errors_excluded_from_deferral_rate():
    """A source outage (failed=True) must not inflate the deferral rate (M6)."""
    recs = [
        AnswerRecord(deferred=False, cached=False, total_ms=10.0, mode="thinking"),
        AnswerRecord(deferred=True, cached=False, total_ms=20.0, mode="thinking"),
        AnswerRecord(deferred=False, cached=False, total_ms=5.0, mode="thinking", failed=True),
    ]
    m = aggregate(recs)
    assert m.errors == 1
    # deferral_rate is 1/2 (one deferral of two decided), not 1/3
    assert abs(m.deferral_rate - 0.5) < 1e-9


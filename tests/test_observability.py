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


def test_postgres_sink_recent_orders_by_created_at_desc():
    """The query must ORDER BY created_at DESC so LIMIT returns the most recent rows."""
    from mnemiq.observability.metrics import PostgresSink

    queries: list[str] = []

    class _FakeCursor:
        def fetchall(self):
            return []

    class _FakeCon:
        def execute(self, sql, params=None):
            queries.append(sql)
            return _FakeCursor()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class _Connect:
        """Stand-in for psycopg.connect that returns a fake connection."""
        def __call__(self, *a, **kw):
            return _FakeCon()

    # Monkey-patch psycopg.connect via the import inside `recent`
    import types
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = _Connect()

    import sys
    original = sys.modules.get("psycopg")
    sys.modules["psycopg"] = fake_psycopg
    try:
        sink = PostgresSink("unused-dsn")
        sink.recent("acme", 50)
    finally:
        if original is not None:
            sys.modules["psycopg"] = original
        else:
            sys.modules.pop("psycopg", None)

    select_queries = [q for q in queries if q.startswith("SELECT")]
    assert len(select_queries) == 1
    assert "ORDER BY created_at DESC" in select_queries[0]


def test_postgres_sink_ddl_includes_created_at():
    """The column ORDER BY sorts on must exist in the DDL; otherwise the query fails inside the
    fail-soft handler and recent() silently returns [] on every call."""
    from mnemiq.observability.metrics import PostgresSink

    assert "created_at" in PostgresSink._CREATE, "created_at missing from _CREATE DDL"
    assert "created_at" in PostgresSink._UPGRADE, "created_at missing from _UPGRADE ALTER"


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

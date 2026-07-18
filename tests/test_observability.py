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

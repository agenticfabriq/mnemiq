from mnemiq.eval.report import Report
from mnemiq.eval.trend import RunRecord, check_regression, last_run, record_run


def _report(correct, total):
    return Report(total=total, correct=correct, wrong=total - correct)


def test_check_regression_flags_a_drop():
    prev = RunRecord(source_id="acme", run_at="t0", accuracy=0.80, strict_accuracy=0.70,
                     total=10, correct=7, correct_facts=1, wrong=2, deferred_wrongly=0, error=0)
    msg = check_regression(_report(6, 10), prev, tolerance=0.02)  # 0.60 vs 0.80
    assert msg is not None and "regress" in msg.lower()


def test_check_regression_within_tolerance_is_none():
    prev = RunRecord(source_id="acme", run_at="t0", accuracy=0.80, strict_accuracy=0.70,
                     total=10, correct=8, correct_facts=0, wrong=2, deferred_wrongly=0, error=0)
    assert check_regression(_report(8, 10), prev, tolerance=0.02) is None  # 0.80 vs 0.80


def test_check_regression_no_baseline_is_none():
    assert check_regression(_report(5, 10), None) is None


def test_record_and_last_run_roundtrip_local_json(tmp_path):
    p = str(tmp_path / "trend.json")
    record_run(None, "acme", _report(8, 10), path=p)
    record_run(None, "acme", _report(9, 10), path=p)
    rec = last_run(None, "acme", path=p)
    assert rec is not None and rec.correct == 9 and rec.total == 10

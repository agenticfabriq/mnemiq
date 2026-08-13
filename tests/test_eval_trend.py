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


# --- M34: a gate that cannot fail is not a gate ---------------------------------


def test_the_gate_fails_when_there_is_no_baseline_to_compare_against():
    """The whole finding. `check_regression` is right to answer "no regression" when there is
    nothing to compare -- but the GATE must not read that as a pass, or it passes forever.
    Measured consequence: a published 100% drifted 16 points with CI green throughout."""
    from mnemiq.eval.trend import gate_outcome

    msg = gate_outcome(_report(15, 30), previous=None)
    assert msg is not None and "baseline" in msg


def test_the_gate_fails_differently_when_the_trend_store_is_unreachable():
    """An outage and a first run are different states, and collapsing them is what let the
    gate pass silently -- `last_run` swallowed every exception into the same `None` that means
    "no baseline yet". Same collapse M2 and M6 removed elsewhere."""
    from mnemiq.eval.trend import gate_outcome

    msg = gate_outcome(_report(15, 30), previous=None, store_error="connection refused")
    assert msg is not None and "connection refused" in msg
    assert "baseline" not in msg.split("(")[0], "an outage must not read as a missing baseline"


def test_the_gate_passes_when_it_actually_compared_and_nothing_regressed():
    from mnemiq.eval.trend import RunRecord, gate_outcome

    previous = RunRecord(source_id="acme", run_at="", accuracy=0.80, strict_accuracy=0.7,
                         total=30, correct=24, correct_facts=0, wrong=0, deferred_wrongly=0,
                         error=0)
    assert gate_outcome(_report(25, 30), previous=previous) is None


def test_the_gate_still_fails_on_a_real_regression():
    from mnemiq.eval.trend import RunRecord, gate_outcome

    previous = RunRecord(source_id="acme", run_at="", accuracy=0.96, strict_accuracy=0.84,
                         total=30, correct=21, correct_facts=3, wrong=1, deferred_wrongly=0,
                         error=0)
    msg = gate_outcome(_report(24, 30), previous=previous)
    assert msg is not None and "regressed" in msg


def test_an_unreadable_trend_store_is_reported_rather_than_swallowed(tmp_path):
    """`last_run` returned None on any exception, so a misconfigured control DSN disarmed the
    gate exactly like a missing file. The Postgres path is the one that looks configured."""
    from mnemiq.eval.trend import TrendUnavailable, last_run

    bad = tmp_path / "trend.json"
    bad.write_text("{ not json")
    try:
        last_run(None, "acme", path=str(bad))
    except TrendUnavailable:
        return
    raise AssertionError("a corrupt trend file was reported as 'no baseline'")


def test_a_genuinely_absent_baseline_is_not_an_outage(tmp_path):
    from mnemiq.eval.trend import last_run

    assert last_run(None, "acme", path=str(tmp_path / "absent.json")) is None


def test_a_regressed_run_must_not_become_the_new_baseline():
    """The defect that would have survived arming the gate. `--gate` recorded the run before
    comparing it, so a regression failed once and then WAS the baseline -- one red build, then
    green forever, while the number kept sliding. A ratchet, not a gate."""
    from mnemiq.eval.trend import RunRecord, should_record

    previous = RunRecord(source_id="acme", run_at="", accuracy=0.96, strict_accuracy=0.84,
                         total=30, correct=29, correct_facts=0, wrong=1, deferred_wrongly=0,
                         error=0)
    regressed = _report(24, 30)  # 0.80 against a 0.96 baseline
    assert should_record(record=True, gate=True, gate_failed=True) is False
    assert should_record(record=True, gate=True, gate_failed=False) is True
    from mnemiq.eval.trend import gate_outcome
    assert gate_outcome(regressed, previous) is not None  # it did fail, as it must


def test_recording_without_gating_always_writes():
    """`--record` alone is how a baseline gets established in the first place."""
    from mnemiq.eval.trend import should_record

    assert should_record(record=True, gate=False, gate_failed=False) is True


def test_gating_alone_never_writes():
    """A gate is a reader. Writing was what let it move its own goalposts."""
    from mnemiq.eval.trend import should_record

    assert should_record(record=False, gate=True, gate_failed=False) is False
    assert should_record(record=False, gate=True, gate_failed=True) is False

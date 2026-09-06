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


# --- M87: the exact-match rate is recorded on every row and compared by nothing ------------------


def _shaped(*, correct, correct_facts, total):
    """A report whose two rates differ, which is the whole point of having both."""
    return Report(total=total, correct=correct, correct_facts=correct_facts,
                  wrong=total - correct - correct_facts)


def test_exact_matches_decaying_into_right_shape_is_a_regression():
    """The degradation `accuracy` cannot see, and the reason `strict_accuracy` exists.

    `CORRECT_FACTS` is "right data, different shape" and is deliberately not a failure for the
    product metric -- which is exactly why something has to watch the exact-match rate separately.
    Three exact matches turning into differently-shaped ones leaves accuracy untouched and takes
    strict from 0.84 to 0.72, and the gate was green through it.
    """
    previous = RunRecord(source_id="acme", run_at="t0", accuracy=0.96, strict_accuracy=0.84,
                         total=25, correct=21, correct_facts=3, wrong=1, deferred_wrongly=0,
                         error=0)
    now = _shaped(correct=18, correct_facts=6, total=25)
    assert now.accuracy == previous.accuracy, "the product metric must be unmoved by this shape"

    message = check_regression(now, previous, tolerance=0.02)
    assert message is not None, "an exact-match collapse must not pass as a green run"
    assert "exact" in message.lower(), message


def test_a_strict_drop_names_which_number_moved():
    """Two numbers can fail, so the message has to say which, or an operator chases the wrong one."""
    previous = RunRecord(source_id="acme", run_at="t0", accuracy=0.96, strict_accuracy=0.84,
                         total=25, correct=21, correct_facts=3, wrong=1, deferred_wrongly=0,
                         error=0)
    strict_only = check_regression(_shaped(correct=18, correct_facts=6, total=25), previous)
    assert "exact" in strict_only.lower() and "got-the-facts" not in strict_only.lower()

    # When BOTH fall, both must be named. Reporting only the first left the exact-match drop
    # invisible here AND in the run output, since the failure list excludes `CORRECT_FACTS` -- so
    # an operator would fix the case they were told about and ship the other.
    both = check_regression(_shaped(correct=15, correct_facts=3, total=25), previous)
    assert both is not None
    assert "got-the-facts" in both.lower() and "exact" in both.lower(), both


def test_one_exact_match_decaying_is_already_a_regression():
    """The exact-match tolerance sits in the same "no case may regress" band as the other.

    Without this, raising the strict tolerance to 0.05 leaves every test in this file green while
    a single case silently stops being caught -- and a single case is 4 points on 25, which is the
    whole reason the tolerance is a proxy for zero rather than a tuned band.
    """
    previous = RunRecord(source_id="acme", run_at="t0", accuracy=0.96, strict_accuracy=0.84,
                         total=25, correct=21, correct_facts=3, wrong=1, deferred_wrongly=0,
                         error=0)
    one_case = _shaped(correct=20, correct_facts=4, total=25)
    assert one_case.accuracy == previous.accuracy
    assert one_case.strict_accuracy == 0.80
    # No explicit tolerance: the DEFAULT is what the gate runs with, and passing 0.02 here would
    # leave the default free to drift to a band that lets one case through.
    message = check_regression(one_case, previous)
    assert message is not None and "exact" in message.lower(), message


def test_each_rate_is_compared_against_its_own_predecessor():
    """Substituting `previous.accuracy` for `previous.strict_accuracy` must not pass.

    Not pinned with an inverted baseline, which was the first attempt: `strict_accuracy` counts
    exact matches and `accuracy` counts those PLUS right-shape ones, so strict can never exceed
    accuracy and a fixture where it does proves nothing about a real run. With realistic numbers
    the mix-up makes the check stricter rather than looser, so what catches it is an UNCHANGED
    run -- the mutation reads 84.0% against the previous 96.0% and cries regression at a run that
    reproduced its baseline exactly, which is the nightly reddening every night.
    """
    previous = RunRecord(source_id="acme", run_at="t0", accuracy=0.96, strict_accuracy=0.84,
                         total=25, correct=21, correct_facts=3, wrong=1, deferred_wrongly=0,
                         error=0)
    unchanged = _shaped(correct=21, correct_facts=3, total=25)
    assert unchanged.accuracy == 0.96 and unchanged.strict_accuracy == 0.84
    assert check_regression(unchanged, previous) is None

    # The MIRROR substitution, which is the loosening one and therefore the one that matters:
    # got-the-facts compared against the previous STRICT rate. With realistic baselines strict sits
    # below accuracy, so the product metric would be judged against the lower number and this run
    # -- accuracy 96.0% -> 88.0%, two answerable cases lost -- would pass green.
    dropped = _shaped(correct=22, correct_facts=0, total=25)
    assert dropped.accuracy == 0.88 and dropped.strict_accuracy == 0.88
    message = check_regression(dropped, previous)
    assert message is not None and "got-the-facts" in message.lower(), message


def test_the_exact_rate_improving_is_not_a_regression():
    previous = RunRecord(source_id="acme", run_at="t0", accuracy=0.96, strict_accuracy=0.84,
                         total=25, correct=21, correct_facts=3, wrong=1, deferred_wrongly=0,
                         error=0)
    assert check_regression(_shaped(correct=24, correct_facts=0, total=25), previous) is None

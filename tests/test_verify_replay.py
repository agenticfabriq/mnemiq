import pytest

from mnemiq.eval.verify_replay import judge_scores, replay, sweep, tally
from mnemiq.verify.judge import FakeJudge
from mnemiq.verify.verifier import Verifier


def _rec(outcome, sql, rows, q="q"):
    return {"question": q, "sql": sql, "engine_rows": rows, "outcome": outcome}


def test_replay_counts_wrong_caught_and_correct_lost():
    records = [
        _rec("wrong", "SELECT n FROM t", []),                                   # empty -> sanity
        _rec("wrong", "SELECT SUM(x) FROM t", [{"s": 5}], q="total for 'SME'"),  # ungrounded -> grounding
        _rec("correct", "SELECT COUNT(*) FROM t", [{"n": 3}]),                   # clean -> kept
        _rec("correct", "SELECT n FROM t", []),                                  # empty correct -> lost
    ]
    out = replay(records, Verifier(grounding=True))  # exercise both deterministic layers
    assert out["answerable"] == 4
    assert out["wrong_before"] == 2 and out["correct_before"] == 2
    assert out["wrong_caught"] == 2
    assert out["correct_lost"] == 1
    assert out["wrong_after_rate"] == 0.0
    assert out["ex_after"] == 0.25


def test_judge_sweep_deferral_grows_as_threshold_rises():
    records = [
        {"question": "q", "sql": "SELECT 1", "engine_rows": [{"n": 1}], "outcome": "wrong", "db_id": "d"},
        {"question": "q", "sql": "SELECT 2", "engine_rows": [{"n": 2}], "outcome": "correct", "db_id": "d"},
    ]
    scores = judge_scores(records, FakeJudge(0.4), cards_for=lambda db: "TABLE t")
    assert scores == [0.4, 0.4]
    rows = sweep(records, scores, thresholds=[0.3, 0.5])
    # threshold 0.3: 0.4 >= 0.3 -> keep both; threshold 0.5: 0.4 < 0.5 -> defer both
    assert rows[0]["threshold"] == 0.3 and rows[0]["wrong_caught"] == 0
    assert rows[1]["threshold"] == 0.5 and rows[1]["wrong_caught"] == 1 and rows[1]["correct_lost"] == 1


def test_combined_sweep_is_the_union_not_the_sum():
    """Two layers running beside each other catch the union of what each catches, and the union of
    two overlapping sets is smaller than their sum. The number card published a `sanity + judge`
    row that could not be reproduced for exactly this reason, so the tool now computes the union
    rather than leaving a reader to add two rows."""
    records = [{"outcome": "wrong", "question": "q1", "sql": "select 1", "engine_rows": []},
               {"outcome": "wrong", "question": "q2", "sql": "select 2", "engine_rows": []},
               {"outcome": "correct", "question": "q3", "sql": "select 3", "engine_rows": []}]
    # The judge catches record 1 (score below threshold); the deterministic layer catches 1 and 2.
    rows = sweep(records, [0.0, 1.0, 1.0], [0.5], also_defer=[True, True, False])
    assert rows[0]["wrong_caught"] == 2, "the union of {1} and {1,2} is {1,2}, not three catches"
    assert rows[0]["correct_lost"] == 0
    judge_only = sweep(records, [0.0, 1.0, 1.0], [0.5])
    assert judge_only[0]["wrong_caught"] == 1


def test_sweep_refuses_a_score_list_that_does_not_line_up():
    """The judge scores and the deterministic deferrals are matched BY POSITION. A silent
    mismatch would attribute one case's judgement to another -- the same collapse the retry
    wrapper exists to prevent, one layer up."""
    records = [{"outcome": "wrong", "question": "q", "sql": "s", "engine_rows": []}]
    with pytest.raises(ValueError, match="2 judge scores for 1 answerable"):
        sweep(records, [0.1, 0.2], [0.5])
    with pytest.raises(ValueError, match="2 deferrals for 1 answerable"):
        tally(records, [True, False])


def test_sanity_verdicts_all_defer():
    """The replay combines layers with OR; the product short-circuits on the first verdict. Those
    agree only while every sanity verdict is a deferral. If sanity ever gains an APPROVING verdict,
    the combined row stops describing the product and this test is where that is caught."""
    import pyarrow as pa

    from mnemiq.verify.sanity import sanity_check

    tables = [pa.table({}),                        # no rows at all
              pa.table({"n": [None]}),             # one empty value
              pa.table({"n": [None, None]}),       # only empty values
              pa.table({"n": [1, 2]})]             # a real answer
    for t in tables:
        v = sanity_check("q", t)
        assert v is None or v.defer, "a non-deferring sanity verdict breaks the replay's OR"

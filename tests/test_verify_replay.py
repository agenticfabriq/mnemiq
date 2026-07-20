from mnemiq.eval.verify_replay import judge_scores, replay, sweep
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

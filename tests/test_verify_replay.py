from mnemiq.eval.verify_replay import replay
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
    out = replay(records, Verifier())
    assert out["answerable"] == 4
    assert out["wrong_before"] == 2 and out["correct_before"] == 2
    assert out["wrong_caught"] == 2
    assert out["correct_lost"] == 1
    assert out["wrong_after_rate"] == 0.0
    assert out["ex_after"] == 0.25

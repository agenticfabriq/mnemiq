import json

from mnemiq.feedback.capture import capture_fix


def test_capture_appends_golden_case_and_example(tmp_path):
    g, e = str(tmp_path / "golden.json"), str(tmp_path / "ex.json")
    cid = capture_fix("How many claims?", "SELECT count(*) FROM claim", ["claim"], "acme", g, e)
    golden = json.load(open(g))
    examples = json.load(open(e))
    assert golden[0]["id"] == cid and golden[0]["question"] == "How many claims?"
    assert golden[0]["gold_sql"] == "SELECT count(*) FROM claim"
    assert golden[0]["answerable"] is True and "feedback" in golden[0]["tags"]
    assert examples[0]["sql"] == "SELECT count(*) FROM claim" and examples[0]["tables"] == ["claim"]


def test_capture_is_idempotent(tmp_path):
    g, e = str(tmp_path / "golden.json"), str(tmp_path / "ex.json")
    capture_fix("q", "SELECT 1", ["t"], "acme", g, e)
    capture_fix("q", "SELECT 1", ["t"], "acme", g, e)  # same fix twice
    assert len(json.load(open(g))) == 1 and len(json.load(open(e))) == 1

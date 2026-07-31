import pyarrow as pa

from mnemiq.agent.loop import AgentAnswer, result_preview


def test_preview_caps_rows_and_flags_truncation():
    t = pa.table({"a": list(range(150)), "b": [str(i) for i in range(150)]})
    p = result_preview(t, 100)
    assert p.columns == ["a", "b"]
    assert len(p.rows) == 100
    assert p.rows[0] == [0, "0"]
    assert p.row_count == 150
    assert p.truncated is True


def test_preview_under_cap_is_complete_and_not_truncated():
    t = pa.table({"a": [1, 2]})
    p = result_preview(t, 100)
    assert p.rows == [[1], [2]]
    assert p.row_count == 2
    assert p.truncated is False


def test_agent_answer_defaults_to_no_preview():
    assert AgentAnswer(answer="x").preview is None

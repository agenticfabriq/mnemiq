import pyarrow as pa

from mnemiq.execute.render import render_result


def _table(rows: int, value: str = "x") -> pa.Table:
    return pa.table({"n": list(range(rows)), "label": [value] * rows})


def test_a_small_result_renders_every_row():
    text = render_result(_table(3))
    assert "n" in text and "label" in text
    assert text.count("\n") >= 3
    assert "truncated" not in text.lower()


def test_a_large_result_says_it_is_truncated():
    text = render_result(_table(500), max_rows=10)
    assert "500" in text  # the true row count is stated
    assert "10" in text  # and how many are shown
    assert "truncat" in text.lower()  # never let the model summarize a partial view silently


def test_a_wide_cell_is_clipped():
    text = render_result(_table(1, value="y" * 500), max_cell=20)
    assert "y" * 21 not in text


def test_an_empty_result_is_stated_plainly():
    text = render_result(pa.table({"n": pa.array([], type=pa.int64())}))
    assert "0 rows" in text  # "no rows" is a real answer, and must not read as an error


def test_nulls_render_as_null_not_none():
    table = pa.table({"n": pa.array([None, 1], type=pa.int64())})
    assert "NULL" in render_result(table)
